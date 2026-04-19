from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stgcn.hand_graph import build_adjacency_matrix
from src.stgcn.stgcn_model import RealSTGCN

OUTPUT_DIR = REPO_ROOT / "output/final_manifests"
COMBINED_MANIFEST = OUTPUT_DIR / "manifest_verificacion_temporal_4sets.csv"
RESULTS_JSON = OUTPUT_DIR / "training_verificacion_temporal_real_stgcn.json"
BEST_MODEL = OUTPUT_DIR / "model_real_stgcn_best.pth"

SOURCE_MANIFESTS = {
    "hagrid": REPO_ROOT / "output/final_manifests/manifest_hagrid_secuencias.csv",
    "freihand": REPO_ROOT / "output/final_manifests/manifest_freihand_secuencias.csv",
    "synthetic": REPO_ROOT / "output/final_manifests/manifest_synthetic_secuencias.csv",
    "mano": REPO_ROOT / "output/final_manifests/manifest_mano_secuencias.csv",
}

MANO_LABEL_CANDIDATES = ("label", "gesture", "gesture_label", "class")

SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mst_to_condition(mst: int) -> str:
    if mst <= 4:
        return "claro"
    if mst <= 7:
        return "medio"
    return "oscuro"


def parse_mst_from_sample(sample_id: str) -> int:
    match = re.search(r"MST[_-]?(\d+)", sample_id)
    if match:
        value = int(match.group(1))
        if 1 <= value <= 10:
            return value
    return 5


def infer_mano_gesture_label(df: pd.DataFrame) -> pd.Series | None:
    """Try to recover a real MANO gesture label from source columns or sample ids."""

    for column in MANO_LABEL_CANDIDATES:
        if column in df.columns:
            values = df[column].astype(str).str.strip()
            non_empty = values[(values != "") & (values.str.lower() != "unknown")]
            if not non_empty.empty:
                return values.where((values != "") & (values.str.lower() != "unknown"), other=None)

    extracted: list[str | None] = []
    for sample_id in df["sample_id"].astype(str):
        match = re.search(r"(?:gesture|label|class)[_-]?([A-Za-z][A-Za-z0-9]+)", sample_id, flags=re.IGNORECASE)
        extracted.append(match.group(1).lower() if match else None)

    series = pd.Series(extracted, index=df.index, dtype="object")
    if series.notna().any():
        return series
    return None


def resolve_sequence_path(raw_path: str) -> Path:
    path = Path(str(raw_path).replace("\\", "/"))
    if path.is_absolute() and path.exists():
        return path

    candidates = [REPO_ROOT / path, REPO_ROOT / "data/processed" / path, REPO_ROOT / "data" / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return REPO_ROOT / path


def build_combined_manifest() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    mano_label_source = "missing"

    for dataset, manifest_path in SOURCE_MANIFESTS.items():
        if not manifest_path.exists():
            raise FileNotFoundError(f"No existe manifiesto para {dataset}: {manifest_path}")

        df = pd.read_csv(manifest_path)
        df["dataset"] = dataset

        if "label" not in df.columns:
            if dataset == "mano":
                mano_labels = infer_mano_gesture_label(df)
                if mano_labels is None:
                    df["label"] = "unknown"
                else:
                    df["label"] = mano_labels.fillna("unknown")
                    mano_label_source = "recovered"
            else:
                raise RuntimeError(f"El manifiesto {dataset} no contiene la columna label")
        elif dataset == "mano":
            cleaned = df["label"].astype(str).str.strip().str.lower()
            if cleaned.isin({"", "unknown"}).all():
                inferred = infer_mano_gesture_label(df)
                if inferred is not None:
                    df["label"] = inferred.fillna(df["label"])
                    mano_label_source = "recovered"
                else:
                    mano_label_source = "unavailable"
            else:
                mano_label_source = "provided"

        if "mst" not in df.columns:
            if dataset == "mano":
                df["mst"] = df["sample_id"].astype(str).map(parse_mst_from_sample)
            else:
                df["mst"] = 5

        if "condition" not in df.columns:
            df["condition"] = df["mst"].astype(int).map(mst_to_condition)

        if "mst_origin" not in df.columns:
            df["mst_origin"] = "synthetic" if dataset == "mano" else "imputed"

        required = ["sample_id", "path_secuencia", "label", "condition", "dataset", "mst", "mst_origin"]
        for column in required:
            if column not in df.columns:
                raise RuntimeError(f"Falta columna requerida {column} en manifiesto {dataset}")

        frames.append(df[required].copy())

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["sample_id", "dataset"], keep="first").reset_index(drop=True)

    mano_rows = combined[combined["dataset"] == "mano"]
    mano_labeled = int((mano_rows["label"].astype(str).str.strip().str.lower() != "unknown").sum())
    mano_total = len(mano_rows)
    print(f"MANO labels: {mano_labeled}/{mano_total} recovered ({mano_label_source})")

    rng = np.random.default_rng(SEED)
    splits: list[pd.DataFrame] = []
    for _, group in combined.groupby(["label", "condition"], sort=False):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)

        n_rows = len(indices)
        n_train = int(round(n_rows * TRAIN_FRACTION))
        n_val = int(round(n_rows * VAL_FRACTION))
        if n_train + n_val > n_rows:
            n_val = max(0, n_rows - n_train)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        splits.append(combined.loc[train_idx].assign(split="train"))
        splits.append(combined.loc[val_idx].assign(split="val"))
        splits.append(combined.loc[test_idx].assign(split="test"))

    manifest = pd.concat(splits, ignore_index=True)
    manifest = manifest.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return manifest


class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]

        seq_path = resolve_sequence_path(row["path_secuencia"])
        if not seq_path.exists():
            raise FileNotFoundError(f"No existe secuencia: {seq_path}")

        seq = np.load(seq_path).astype(np.float32)  # (T, 21, 3)
        x = torch.from_numpy(np.transpose(seq, (2, 0, 1))).float()  # (3, T, 21)
        y = torch.tensor(self.label_to_idx[row["label"]], dtype=torch.long)

        return {
            "landmarks": x,
            "label": y,
            "condition": row["condition"],
            "mst": torch.tensor(float(row["mst"]), dtype=torch.float),
        }


def build_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    label_counts = df["label"].value_counts().to_dict()
    mst_counts = df["mst"].value_counts().to_dict()
    label_total = sum(label_counts.values())
    mst_total = sum(mst_counts.values())

    weights = []
    for _, row in df.iterrows():
        label_weight = label_total / label_counts[row["label"]]
        mst_value = int(row["mst"])
        mst_weight = mst_total / mst_counts[mst_value] if mst_value in mst_counts else 1.0
        weights.append(label_weight * mst_weight)

    weights_tensor = torch.tensor(weights, dtype=torch.double)
    weights_tensor = weights_tensor / weights_tensor.sum() * len(weights_tensor)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)


def to_loader(dataset: Dataset, batch_size: int, sampler=None, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=0,
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["landmarks"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.shape[0]

    return total_loss / max(1, len(loader)), 100 * correct / max(1, total)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["landmarks"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.shape[0]

    return total_loss / max(1, len(loader)), 100 * correct / max(1, total)


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if COMBINED_MANIFEST.exists():
        manifest = pd.read_csv(COMBINED_MANIFEST)
    else:
        manifest = build_combined_manifest()
        manifest.to_csv(COMBINED_MANIFEST, index=False)

    label_order = sorted(manifest["label"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(label_order)}

    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

    train_ds = SequenceDataset(train_df, label_to_idx)
    val_ds = SequenceDataset(val_df, label_to_idx)
    test_ds = SequenceDataset(test_df, label_to_idx)

    train_loader = to_loader(train_ds, BATCH_SIZE, sampler=build_sampler(train_df))
    val_loader = to_loader(val_ds, BATCH_SIZE, shuffle=False)
    test_loader = to_loader(test_ds, BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adjacency = build_adjacency_matrix().to(device)
    model = RealSTGCN(
        num_classes=len(label_order),
        adjacency=adjacency,
        in_channels=3,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.1)

    print(f"Device: {device}")
    print(f"Combined manifest: {COMBINED_MANIFEST}")
    print(f"Splits: {manifest['split'].value_counts().to_dict()}")
    print(f"Datasets in split: {manifest.groupby(['split', 'dataset']).size().to_dict()}")
    print("Scheduler: CosineAnnealingLR(T_max=20, eta_min=0.0001)")

    history = []
    best_val_accuracy = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        if val_acc >= best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), BEST_MODEL)

        scheduler.step()

    if BEST_MODEL.exists():
        model.load_state_dict(torch.load(BEST_MODEL, map_location=device))

    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "architecture": "RealSTGCN(3 blocks: graph spatial conv + temporal conv)",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "scheduler": "CosineAnnealingLR",
            "scheduler_eta_min": LEARNING_RATE * 0.1,
            "seed": SEED,
            "source_manifests": {k: str(v) for k, v in SOURCE_MANIFESTS.items()},
            "combined_manifest": str(COMBINED_MANIFEST),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "num_classes": len(label_order),
        },
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "final_val_accuracy": history[-1]["val_accuracy"],
        "final_val_loss": history[-1]["val_loss"],
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }

    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)

    print(f"Saved model: {BEST_MODEL}")
    print(f"Saved results: {RESULTS_JSON}")
    print(f"Val losses: {[round(h['val_loss'], 4) for h in history]}")
    print(f"Final val acc: {round(history[-1]['val_accuracy'], 3)}")


if __name__ == "__main__":
    main()
