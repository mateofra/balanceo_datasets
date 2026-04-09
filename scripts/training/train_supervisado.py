from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stgcn.hand_graph import build_adjacency_matrix
from src.stgcn.stgcn_model import STGCN


class HaGRIDDataset(Dataset):
    def __init__(self, manifest_path: str | Path, seq_dir: str | Path = "data/processed/secuencias_stgcn"):
        df = pd.read_csv(manifest_path)

        if "landmark_quality" in df.columns:
            df = df[
                (df["dataset"] == "hagrid")
                & (df["landmark_quality"] == "annotation_2d_projected")
            ].reset_index(drop=True)
        else:
            df = df[df["dataset"] == "hagrid"].reset_index(drop=True)

        self.seq_dir = Path(seq_dir)
        self.df = df.copy()
        self.df["seq_path"] = self.df["sample_id"].apply(lambda sid: self.seq_dir / f"{sid}.npy")
        self.df = self.df[self.df["seq_path"].apply(lambda p: p.exists())].reset_index(drop=True)

        clases = sorted(self.df["label"].unique())
        self.class_to_idx = {clase: idx for idx, clase in enumerate(clases)}
        self.idx_to_class = {idx: clase for clase, idx in self.class_to_idx.items()}

        self.paths = self.df["seq_path"].tolist()
        self.labels = [self.class_to_idx[label] for label in self.df["label"]]
        self.mst = self.df["mst"].tolist()

        print(f"Dataset cargado: {len(self.paths)} muestras, {len(clases)} clases")
        print(f"Clases: {clases}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        x = np.load(self.paths[idx]).astype(np.float32)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long), self.mst[idx]


def train(
    manifest_path: str = "output/training/train_manifest_stgcn_fixed.csv",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_ratio: float = 0.15,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    training_dir = Path("output/training")
    training_dir.mkdir(parents=True, exist_ok=True)

    dataset = HaGRIDDataset(manifest_path)
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = len(dataset.class_to_idx)
    adjacency = build_adjacency_matrix().to(device)
    model = STGCN(adjacency, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history_path = training_dir / "training_history_supervisado.json"
    checkpoint_path = training_dir / "best_stgcn_supervisado.pth"
    history: list[dict[str, float]] = []
    best_val = 0.0
    start_epoch = 1

    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if history:
                start_epoch = int(history[-1]["epoch"]) + 1
                best_val = max(float(item.get("val_acc", 0.0)) for item in history)
        except Exception:
            history = []
            start_epoch = 1

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            best_val = float(checkpoint.get("best_val", best_val))

    print(f"Reanudando desde época: {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits, attn = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits, _ = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_acc = correct / total if total else 0.0
        avg_loss = train_loss / len(train_loader)
        history.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc})
        print(f"Epoch {epoch:03d}/{epochs} | loss: {avg_loss:.4f} | val_acc: {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "class_to_idx": dataset.class_to_idx,
                    "idx_to_class": dataset.idx_to_class,
                    "best_val": best_val,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nMejor val_acc: {best_val:.3f}")
    print(f"Modelo guardado: {checkpoint_path}")


if __name__ == "__main__":
    train()