"""
DataLoader para ST-GCN con dataset balanceado (FreiHAND + HaGRID).

Features:
- Carga 21 landmarks (hand joints) desde .npy
- Normalización por hand (centra en muñeca)
- Balanceo por tono de piel MST si lo deseas
- Compatible con PyTorch DataLoader
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class STGCNHandDataset(Dataset):
    """Dataset de landmarks 3D de manos para ST-GCN.
    
    Puede usar:
    - 21 hand joints (landmarks MediaPipe/Freihand)
    - Normalización por hand (centra en muñeca)
    - Balanceo por tono de piel (MST)
    """

    def __init__(
        self,
        manifest_csv: Path | str,
        normalize: bool = True,
        balance_by_mst: bool = False,
        gesture_label_map: dict[str, int] | None = None,
    ):
        """Inicializa dataset desde manifiesto ST-GCN.
        
        Args:
            manifest_csv: Ruta a output/train_manifest_stgcn.csv
            normalize: Si True, centra landmarks en muñeca (wrist, idx=0)
            balance_by_mst: Si True, usa MST para sampling balanceado
            gesture_label_map: Mapeo manual gesture→label (default: auto ordenado)
        """
        self.manifest_path = Path(manifest_csv)
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

        print(f"Dataset cargado: {len(self.rows)} muestras, {self.num_classes} gestos")
        print(f"Gestos: {list(self.gesture_to_label.keys())[:5]}...")
        print(f"Balance MST: {balance_by_mst}")

    def _load_manifest(self) -> list[dict]:
        """Carga CSV del manifiesto."""
        rows = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if rows and "landmark_quality" in rows[0]:
            rows = [
                row for row in rows
                if row.get("landmark_quality") == "real_3d_freihand"
            ]
            print(f"Muestras de entrenamiento: {len(rows)}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        """Retorna: {"landmarks": (21, 3), "label": int, "sample_id": str, "mst": int}"""
        record = self.rows[idx]

        # Cargar landmarks
        path = Path(record["path_landmarks"])
        if not path.exists():
            # Fallback: intentar construcción alternativa de ruta
            # (fix para cambios de path Windows→Linux)
            alt_path = Path("data/processed/landmarks") / Path(record["sample_id"]).name
            alt_path = alt_path.with_suffix(".npy")
            if alt_path.exists():
                path = alt_path
            else:
                raise FileNotFoundError(f"No encontrado: {path} (se buscó también {alt_path})")

        landmarks = np.load(path).astype(np.float32)  # (21, 3)

        # Normalizar por muñeca (índice 0)
        if self.normalize and landmarks.shape[0] >= 1:
            wrist = landmarks[0]  # (3,)
            landmarks = landmarks - wrist[None, :]

        # Label
        gesture = record["label"]
        label = self.gesture_to_label[gesture]

        # MST info
        mst = int(record["mst"])

        return {
            "landmarks": torch.from_numpy(landmarks),  # (21, 3)
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": record["sample_id"],
            "mst": torch.tensor(mst, dtype=torch.long),
            "condition": record["condition"],  # claro/medio/oscuro
            "dataset": record["dataset"],  # freihand/hagrid
        }

    def get_weighted_sampler(self, mode: Literal["mst", "balanced"] = "mst"):
        """Retorna WeightedRandomSampler para balanceo durante training.
        
        Args:
            mode: "mst" = balancea por tono de piel
                  "balanced" = todas muestras igual peso
        """
        if mode == "balanced":
            weights = torch.ones(len(self.rows))
        elif mode == "mst":
            # Inverso del conteo por MúST
            from collections import Counter

            mst_counts = Counter(int(r["mst"]) for r in self.rows)
            total = len(self.rows)
            mst_weights = {mst: total / count for mst, count in mst_counts.items()}
            weights = torch.tensor(
                [mst_weights[int(self.rows[i]["mst"])] for i in range(len(self.rows))]
            )
        else:
            raise ValueError(f"Modo desconocido: {mode}")

        # Normalizar
        weights = weights / weights.sum() * len(weights)

        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.rows),
            replacement=True,
        )


def create_dataloaders(
    manifest_csv: Path | str,
    batch_size: int = 32,
    num_workers: int = 0,
    normalize: bool = True,
    balance_by_mst: bool = False,
) -> dict[str, DataLoader]:
    """Crea DataLoaders para training/validation.
    
    Args:
        manifest_csv: Ruta a output/train_manifest_stgcn.csv
        batch_size: Batch size para training
        num_workers: Workers para data loading
        normalize: Normalizar landmarks por muñeca
        balance_by_mst: Si True, usa WeightedRandomSampler por MST
    
    Returns:
        {"train": DataLoader, "info": dict con estadísticas}
    """
    dataset = STGCNHandDataset(
        manifest_csv=manifest_csv,
        normalize=normalize,
        balance_by_mst=balance_by_mst,
    )

    if balance_by_mst:
        sampler = dataset.get_weighted_sampler(mode="mst")
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    return {
        "train": train_loader,
        "dataset": dataset,
        "num_classes": dataset.num_classes,
        "gesture_to_label": dataset.gesture_to_label,
    }


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # 1. Crear dataset y dataloaders
    # Nota: usar manifiesto reparado (_fixed) que contiene solo muestras con landmarks disponibles
    loaders = create_dataloaders(
        manifest_csv="output/training/train_manifest_stgcn_fixed.csv",
        batch_size=32,
        num_workers=0,  # num_workers=0 por compatibilidad Windows
        normalize=True,
        balance_by_mst=True,  # Balancea por tono de piel
    )

    train_loader = loaders["train"]
    dataset = loaders["dataset"]
    num_classes = loaders["num_classes"]

    print(f"\n✓ Dataset listo para training:")
    print(f"  - DataLoader: {len(train_loader)} batches")
    print(f"  - Clases: {num_classes}")
    print(f"  - Landmarks por muestra: 21 joints × 3 coords")

    # 2. Iterar una muestra de prueba
    print(f"\n✓ Muestra de batch:")
    batch = next(iter(train_loader))

    print(f"  - Landmarks: {batch['landmarks'].shape}")  # (BS, 21, 3)
    print(f"    Ejemplo muestra 0: min={batch['landmarks'][0].min():.3f}, max={batch['landmarks'][0].max():.3f}")
    print(f"  - Labels: {batch['label'].shape}, valores: {batch['label'][:5]}")
    print(f"  - MST: {batch['mst'][:5]}")
    print(f"  - Condición (tono): {batch['condition'][:3]}")
    print(f"  - Dataset: {batch['dataset'][:3]}")

    # 3. Preparar entrada para modelo ST-GCN
    print(f"\n✓ Formato compatible ST-GCN:")
    landmarks = batch["landmarks"]  # (BS, 21, 3)
    print(f"  Shape Input: {landmarks.shape}")
    print(f"  Típicamente ST-GCN consume: (BS, C, T, V)")
    print(f"    Si usas 1 frame: reshape a (BS, 3, 1, 21)")

    # Expandir tiempo T=1
    landmarks_expanded = landmarks.permute(0, 2, 1).unsqueeze(2)  # (BS, 3, 1, 21)
    print(f"  Shape expandido: {landmarks_expanded.shape}")

    print("\n✓ Listo para entrenar modelo ST-GCN")
