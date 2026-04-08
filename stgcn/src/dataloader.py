"""
DataLoader para ST-GCN con dataset balanceado por MST.

Características:
- Carga 21 landmarks (hand joints) desde .npy
- Normalización opcional por hand (centra en muñeca)
- Balanceo por tono de piel MST si lo deseas
- Compatible con PyTorch DataLoader
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class STGCNHandDataset(Dataset):
    """Dataset de landmarks 3D de manos para ST-GCN."""

    def __init__(
        self,
        manifest_csv: Path | str,
        landmarks_root: Path | str = "data",
        normalize: bool = True,
        balance_by_mst: bool = False,
        gesture_label_map: dict[str, int] | None = None,
    ):
        """Inicializa dataset desde manifiesto ST-GCN.
        
        Args:
            manifest_csv: Ruta a train_manifest_stgcn_fixed.csv
            landmarks_root: Directorio base de landmarks
            normalize: Si True, centra landmarks en muñeca (wrist, idx=0)
            balance_by_mst: Si True, usa WeightedRandomSampler por tono
            gesture_label_map: Mapeo manual gesture→label
        """
        self.manifest_path = Path(manifest_csv)
        self.landmarks_root = Path(landmarks_root)
        self.normalize = normalize
        self.balance_by_mst = balance_by_mst

        # Cargar manifiesto
        self.rows = self._load_manifest()

        # Mapeo de gestos → labels enteros
        unique_gestures = sorted(set(r["label"] for r in self.rows))
        self.gesture_to_label = gesture_label_map or {
            g: i for i, g in enumerate(unique_gestures)
        }
        self.label_to_gesture = {v: k for k, v in self.gesture_to_label.items()}
        self.num_classes = len(self.gesture_to_label)

        print(f"✅ Dataset: {len(self.rows)} muestras, {self.num_classes} gestos")

    def _load_manifest(self) -> list[dict]:
        """Carga CSV del manifiesto."""
        rows = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        """Retorna {"landmarks": (21, 3), "label": int, ...}"""
        record = self.rows[idx]

        # Construir ruta a landmarks
        sample_id = record["sample_id"]
        npy_path = self.landmarks_root / f"{sample_id}.npy"
        
        if not npy_path.exists():
            raise FileNotFoundError(f"Landmarks no encontrado: {npy_path}")

        landmarks = np.load(npy_path).astype(np.float32)  # (21, 3)

        # Normalizar por muñeca (índice 0)
        if self.normalize and landmarks.shape[0] >= 1:
            wrist = landmarks[0]
            landmarks = landmarks - wrist[None, :]

        # Gesture label
        gesture = record["label"]
        label = self.gesture_to_label[gesture]

        # MST info
        mst = int(record["mst"])

        return {
            "landmarks": torch.from_numpy(landmarks),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": record["sample_id"],
            "mst": torch.tensor(mst, dtype=torch.long),
            "condition": record["condition"],
            "dataset": record["dataset"],
        }

    def get_weighted_sampler(self):
        """Retorna WeightedRandomSampler para balanceo MST."""
        mst_counts = Counter(int(r["mst"]) for r in self.rows)
        total = len(self.rows)
        mst_weights = {mst: total / count for mst, count in mst_counts.items()}
        weights = torch.tensor(
            [mst_weights[int(self.rows[i]["mst"])] for i in range(len(self.rows))]
        )
        weights = weights / weights.sum() * len(weights)
        
        return WeightedRandomSampler(weights, len(self.rows), replacement=True)


def create_dataloaders(
    manifest_csv: Path | str,
    landmarks_root: Path | str = "data",
    batch_size: int = 32,
    num_workers: int = 0,
    normalize: bool = True,
    balance_by_mst: bool = False,
) -> dict:
    """Crea DataLoaders para training.
    
    Args:
        manifest_csv: Ruta a train_manifest_stgcn_fixed.csv
        landmarks_root: Directorio base de landmarks
        batch_size: Batch size
        num_workers: Workers para data loading
        normalize: Normalizar landmarks por muñeca
        balance_by_mst: Si True, usa WeightedRandomSampler
    
    Returns:
        Dict con train_loader, dataset, num_classes, gesture_to_label
    """
    dataset = STGCNHandDataset(
        manifest_csv=manifest_csv,
        landmarks_root=landmarks_root,
        normalize=normalize,
        balance_by_mst=balance_by_mst,
    )

    if balance_by_mst:
        sampler = dataset.get_weighted_sampler()
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    return {
        "train": train_loader,
        "dataset": dataset,
        "num_classes": dataset.num_classes,
        "gesture_to_label": dataset.gesture_to_label,
    }
