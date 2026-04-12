"""
Script simplificado para generar landmarks sintéticos SIN dependencias de torch.
Solo usa numpy y pandas (ya disponibles).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LANDMARK_STATS = {
    "mean": np.array([0.5, 0.5, 0.5], dtype=np.float32),
    "std": np.array([0.15, 0.15, 0.1], dtype=np.float32),
}

JOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def generate_realistic_hand_landmarks(
    seed: int | None = None,
    scale: float = 1.0,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Genera landmarks 3D realistas para una mano."""
    if seed is not None:
        np.random.seed(seed)
    
    landmarks = np.random.normal(
        loc=LANDMARK_STATS["mean"],
        scale=LANDMARK_STATS["std"] * scale,
        size=(21, 3),
    ).astype(np.float32)
    
    wrist = landmarks[0]
    
    for digit_start in [1, 5, 9, 13, 17]:
        digit_offset = np.array([
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.3, 0.1),
            np.random.uniform(-0.1, 0.1),
        ], dtype=np.float32)
        
        for j in range(digit_start, min(digit_start + 4, 21)):
            landmarks[j] = wrist + digit_offset * (1 + (j - digit_start) * 0.15)
    
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=(21, 3)).astype(np.float32)
        landmarks += noise
    
    landmarks = np.clip(landmarks, 0, 1).astype(np.float32)
    return landmarks


def load_balanced_manifest(csv_path: Path) -> pd.DataFrame:
    """Carga el manifiesto balanceado."""
    return pd.read_csv(csv_path)


def generate_landmarks_for_manifest(
    manifest_csv: Path,
    output_dir: Path,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, str]:
    """Genera landmarks sintéticos para cada sample del manifiesto."""
    manifest_df = load_balanced_manifest(manifest_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mapping = {}
    np.random.seed(seed)
    
    for idx, row in manifest_df.iterrows():
        sample_id = row["sample_id"]
        sample_seed = seed + idx
        
        landmarks = generate_realistic_hand_landmarks(
            seed=sample_seed,
            scale=np.random.uniform(0.8, 1.2),
            noise_std=0.02,
        )
        
        output_file = output_dir / f"{sample_id}.npy"
        np.save(output_file, landmarks)
        
        try:
            rel_path = output_file.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            rel_path = output_file
        mapping[sample_id] = str(rel_path).replace("\\", "/")
        
        if verbose and (idx + 1) % 2000 == 0:
            print(f"✓ Procesados {idx + 1}/{len(manifest_df)} landmarks sintéticos")
    
    if verbose:
        print(f"✓ {len(manifest_df)} archivos .npy generados en {output_dir}")
    
    return mapping


def generate_stgcn_manifest(
    balanced_manifest_csv: Path,
    landmarks_mapping: dict[str, str],
    output_csv: Path,
    verbose: bool = True,
) -> None:
    """Genera manifiesto ST-GCN integrando landmarks."""
    manifest_df = pd.read_csv(balanced_manifest_csv)
    
    manifest_df["path_landmarks"] = manifest_df["sample_id"].map(
        lambda sid: landmarks_mapping.get(sid, "")
    )
    
    missing_landmarks = manifest_df[manifest_df["path_landmarks"].isna()].shape[0]
    if missing_landmarks > 0:
        print(f"⚠️  {missing_landmarks} samples sin landmarks")
    
    output_columns = [
        "sample_id",
        "path_landmarks",
        "gesture",
        "mst",
        "source",
        "mst_origin",
        "sampling_weight",
    ]
    
    final_columns = [col for col in output_columns if col in manifest_df.columns]
    stgcn_df = manifest_df[final_columns]
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stgcn_df.to_csv(output_csv, index=False)
    
    if verbose:
        print(f"✓ Manifiesto ST-GCN guardado: {output_csv}")
        print(f"  - Total samples: {len(stgcn_df)}")
        print(f"  - Columnas: {', '.join(final_columns)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generar landmarks sintéticos para ST-GCN (sin torch)"
    )
    parser.add_argument(
        "--balanced-manifest",
        type=Path,
        default=Path("output/train_manifest_balanceado_freihand_hagrid.csv"),
        help="Manifiesto balanceado de entrada",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/landmarks"),
        help="Directorio de salida para landmarks .npy",
    )
    parser.add_argument(
        "--output-stgcn-csv",
        type=Path,
        default=Path("output/train_manifest_stgcn_synthetic.csv"),
        help="CSV de salida para ST-GCN",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed para reproducibilidad",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Mostrar progreso",
    )
    
    args = parser.parse_args()
    
    print(f"🔄 Generando landmarks sintéticos...")
    mapping = generate_landmarks_for_manifest(
        args.balanced_manifest,
        args.output_dir,
        seed=args.seed,
        verbose=args.verbose,
    )
    
    print(f"📊 Generando manifiesto ST-GCN...")
    generate_stgcn_manifest(
        args.balanced_manifest,
        mapping,
        args.output_stgcn_csv,
        verbose=args.verbose,
    )
    
    print(f"\n✅ Listo para entrenar ST-GCN!")


if __name__ == "__main__":
    main()
