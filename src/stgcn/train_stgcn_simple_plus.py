"""
Variante ST-GCN mas capaz para comparar contra el baseline simple.

Mantiene el mismo split canonico y el mismo sampler label + MST,
pero aumenta la capacidad del modelo y limita el entrenamiento a 20 epochs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = REPO_ROOT / "output/train_manifest_stgcn.csv"
CANONICAL_MANIFEST = REPO_ROOT / "output/train_manifest_stgcn_canonical.csv"
OUTPUT_DIR = REPO_ROOT / "output/training_logs_simple_plus"
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 42


class HandLandmarkDataset(Dataset):
    """Dataset simple para landmarks de mano."""

    def __init__(self, df: pd.DataFrame, label_to_idx: dict[str, int], num_frames: int = 1):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.num_frames = num_frames

        print(f"🎯 Clases: {len(self.label_to_idx)}")
        print(f"   {self.label_to_idx}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        landmark_path = Path(row["path_landmarks"])
        if not landmark_path.exists():
            fallback_path = REPO_ROOT / "data/processed/landmarks" / f"{row['sample_id']}.npy"
            landmark_path = fallback_path if fallback_path.exists() else landmark_path

        if landmark_path.exists():
            landmarks = np.load(landmark_path).astype(np.float32)
        else:
            landmarks = np.random.uniform(0, 1, (21, 3)).astype(np.float32)

        if self.num_frames == 1:
            x = landmarks.T[:, np.newaxis, :]
        else:
            x = np.tile(landmarks.T[:, np.newaxis, :], (1, self.num_frames, 1))

        x = torch.from_numpy(x).float()
        label = torch.tensor(self.label_to_idx[row["label"]], dtype=torch.long)

        return {
            "landmarks": x,
            "label": label,
            "mst": torch.tensor(float(row.get("mst", 5)), dtype=torch.float),
            "condition": row.get("condition", "medio"),
            "sample_id": row["sample_id"],
        }


class SimplePlusSTGCN(nn.Module):
    """Version mas ancha y profunda del modelo basico."""

    def __init__(self, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(0.15),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)


def build_canonical_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    split_frames: list[pd.DataFrame] = []

    group_columns = ["label", "condition"] if "condition" in df.columns else ["label"]
    for _, group in df.groupby(group_columns, sort=False):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n_rows = len(indices)
        n_train = int(round(n_rows * TRAIN_FRACTION))
        n_val = int(round(n_rows * VAL_FRACTION))
        if n_train + n_val > n_rows:
            n_val = max(0, n_val - (n_train + n_val - n_rows))
        n_test = n_rows - n_train - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:n_train + n_val + n_test]

        split_frames.append(df.loc[train_idx].assign(split="train"))
        split_frames.append(df.loc[val_idx].assign(split="val"))
        split_frames.append(df.loc[test_idx].assign(split="test"))

    canonical = pd.concat(split_frames, ignore_index=True)
    canonical = canonical.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return canonical


def _load_or_build_split(source_manifest: Path) -> pd.DataFrame:
    df = pd.read_csv(source_manifest)
    split_values = set(df["split"].dropna().unique()) if "split" in df.columns else set()
    if {"train", "val", "test"}.issubset(split_values):
        return df

    canonical = build_canonical_split(df)
    canonical.to_csv(CANONICAL_MANIFEST, index=False)
    print(f"✅ Manifiesto canónico guardado: {CANONICAL_MANIFEST}")
    print(f"   Train/Val/Test: {canonical['split'].value_counts().to_dict()}")
    return canonical


def _build_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    label_counts = train_df["label"].value_counts().to_dict()
    mst_counts = train_df["mst"].value_counts().to_dict()
    label_total = sum(label_counts.values())
    mst_total = sum(mst_counts.values())

    weights = []
    for _, row in train_df.iterrows():
        label_weight = label_total / label_counts[row["label"]]
        mst_value = int(row["mst"])
        mst_weight = mst_total / mst_counts[mst_value] if mst_value in mst_counts else 1.0
        weights.append(label_weight * mst_weight)

    weights_tensor = torch.tensor(weights, dtype=torch.double)
    weights_tensor = weights_tensor / weights_tensor.sum() * len(weights_tensor)

    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights_tensor),
        replacement=True,
    )


def _to_loader(dataset: Dataset, batch_size: int, shuffle: bool = False, sampler=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
    )


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
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

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.1f}%"})

    return {
        "loss": total_loss / max(1, len(train_loader)),
        "accuracy": 100 * correct / max(1, total),
    }


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    conditions = []
    msts = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch["landmarks"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.shape[0]

            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            conditions.extend(batch["condition"])
            msts.extend(batch["mst"].tolist())

    return {
        "loss": total_loss / max(1, len(data_loader)),
        "accuracy": 100 * correct / max(1, total),
        "y_true": y_true,
        "y_pred": y_pred,
        "conditions": conditions,
        "msts": msts,
    }


def summarize_predictions(result: dict, label_order: list[str]) -> dict[str, object]:
    cm = np.zeros((len(label_order), len(label_order)), dtype=np.int64)
    for true_idx, pred_idx in zip(result["y_true"], result["y_pred"]):
        cm[true_idx, pred_idx] += 1

    per_class_accuracy: dict[str, float] = {}
    for idx, label in enumerate(label_order):
        denom = cm[idx].sum()
        per_class_accuracy[label] = float(cm[idx, idx] / denom) if denom else float("nan")

    by_condition = defaultdict(lambda: {"correct": 0, "total": 0})
    for true_idx, pred_idx, condition in zip(result["y_true"], result["y_pred"], result["conditions"]):
        by_condition[condition]["total"] += 1
        by_condition[condition]["correct"] += int(true_idx == pred_idx)

    by_mst = defaultdict(lambda: {"correct": 0, "total": 0})
    for true_idx, pred_idx, mst in zip(result["y_true"], result["y_pred"], result["msts"]):
        by_mst[int(mst)]["total"] += 1
        by_mst[int(mst)]["correct"] += int(true_idx == pred_idx)

    return {
        "per_class_accuracy": per_class_accuracy,
        "accuracy_by_condition": {
            condition: stats["correct"] / stats["total"] if stats["total"] else 0.0
            for condition, stats in by_condition.items()
        },
        "accuracy_by_mst": {
            str(mst): stats["correct"] / stats["total"] if stats["total"] else 0.0
            for mst, stats in by_mst.items()
        },
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 ST-GCN Simple Plus Training")
    print(f"   Device: {device}")
    print(f"   Source manifest: {SOURCE_MANIFEST}")

    print("\n📊 Cargando datos...")
    manifest_df = _load_or_build_split(SOURCE_MANIFEST)
    label_order = sorted(manifest_df["label"].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(label_order)}

    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    test_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            "El split canónico quedó incompleto. Revisa el manifiesto de entrada y la estratificación."
        )

    train_dataset = HandLandmarkDataset(train_df, label_to_idx, num_frames=1)
    val_dataset = HandLandmarkDataset(val_df, label_to_idx, num_frames=1)
    test_dataset = HandLandmarkDataset(test_df, label_to_idx, num_frames=1)

    train_sampler = _build_sampler(train_df)
    train_loader = _to_loader(train_dataset, BATCH_SIZE, sampler=train_sampler)
    val_loader = _to_loader(val_dataset, BATCH_SIZE, shuffle=False)
    test_loader = _to_loader(test_dataset, BATCH_SIZE, shuffle=False)

    print(f"   ✓ Train: {len(train_dataset)} samples | {len(train_loader)} batches")
    print(f"   ✓ Val:   {len(val_dataset)} samples | {len(val_loader)} batches")
    print(f"   ✓ Test:  {len(test_dataset)} samples | {len(test_loader)} batches")
    print(f"   ✓ Clases: {len(label_order)}")

    print("\n🧠 Creando modelo...")
    model = SimplePlusSTGCN(num_classes=len(label_order), hidden_dim=128).to(device)
    print(f"   ✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = OUTPUT_DIR / "model_best.pth"
    final_path = OUTPUT_DIR / "model_final.pth"
    results_path = OUTPUT_DIR / "training_results.json"

    results: dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_classes": len(label_order),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "label_balance_mode": "label_plus_mst_sampler",
            "split_seed": SEED,
            "model_variant": "simple_plus",
            "hidden_dim": 128,
        },
        "history": [],
        "best_val_accuracy": 0.0,
    }

    print(f"\n🔥 Entrenando ({EPOCHS} epochs)...")
    best_val_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        results["history"].append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.1f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.1f}%"
        )

        if val_metrics["accuracy"] >= best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_path)
            print(f"   ✓ Best checkpoint: {best_path}")

        if epoch % 5 == 0:
            ckpt_path = OUTPUT_DIR / f"model_epoch_{epoch:02d}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"   ✓ Checkpoint: {ckpt_path}")

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics = evaluate(model, test_loader, criterion, device)
    test_summary = summarize_predictions(test_metrics, label_order)

    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Modelo final guardado: {final_path}")
    print(f"✅ Mejor checkpoint: {best_path}")

    results["best_val_accuracy"] = best_val_accuracy
    results["test"] = {
        "loss": test_metrics["loss"],
        "accuracy": test_metrics["accuracy"],
        **test_summary,
    }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)

    print(f"✅ Resultados: {results_path}")
    print("\n📈 Resumen Final:")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Best val accuracy: {best_val_accuracy:.2f}%")
    print(f"   Test accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"   Test loss: {test_metrics['loss']:.4f}")
    print(f"   Split canónico: {CANONICAL_MANIFEST}")


if __name__ == "__main__":
    main()
