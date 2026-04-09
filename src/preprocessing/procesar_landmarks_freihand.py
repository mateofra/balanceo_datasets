"""
Procesa landmarks 3D de FreiHAND (training_xyz.json) y genera archivos .npy individuales.

Pipeline:
1. Cargar training_xyz.json (array NxMx3 de coordinates)
2. Para cada índice n, guardar como data/processed/landmarks/freihand_XXXXXXXX.npy (21x3)
3. Generar mapeo sample_id → path_landmarks para integración en manifiesto ST-GCN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_freihand_xyz(xyz_path: Path) -> np.ndarray:
    """Carga training_xyz.json de FreiHAND.
    
    Returns:
        np.ndarray: Shape (N, 21, 3) - N samples, 21 landmarks, 3 coords (x, y, z)
    """
    with xyz_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    arr = np.array(data, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (21, 3):
        raise ValueError(
            f"training_xyz.json debe tener shape (N, 21, 3), obtenido {arr.shape}"
        )
    return arr


def process_freihand_landmarks(
    xyz_path: Path,
    output_dir: Path,
    verbose: bool = True,
) -> dict[str, str]:
    """Procesa landmarks de FreiHAND y genera mapeo sample_id → ruta."""
    xyz_data = load_freihand_xyz(xyz_path)
    n_samples = xyz_data.shape[0]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_mapping = {}
    
    for idx in range(n_samples):
        landmarks = xyz_data[idx]  # Shape: (21, 3)
        sample_id = f"freihand_{idx:08d}"
        
        # Guardar como .npy
        output_file = output_dir / f"{sample_id}.npy"
        np.save(output_file, landmarks)
        
        # Mapeo para manifiesto (ruta relativa desde cwd)
        try:
            rel_path = output_file.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            rel_path = output_file.relative_to(output_dir.parent)
        sample_mapping[sample_id] = str(rel_path)
        
        if verbose and (idx + 1) % 1000 == 0:
            print(f"Procesados {idx + 1}/{n_samples} landmarks FreiHAND")
    
    if verbose:
        print(f"✓ {n_samples} archivos .npy generados en {output_dir}")
    
    return sample_mapping


def save_mapping_json(mapping: dict[str, str], output_path: Path) -> None:
    """Guarda mapeo sample_id → path_landmarks como JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=True, indent=2)
    print(f"✓ Mapeo guardado: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Procesa landmarks FreiHAND (training_xyz.json) → .npy individual"
    )
    parser.add_argument(
        "--freihand-xyz",
        type=Path,
        default=Path("datasets/training_xyz.json"),
        help="Ruta a datasets/training_xyz.json de FreiHAND.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/landmarks"),
        help="Directorio de salida para archivos .npy",
    )
    parser.add_argument(
        "--output-mapping",
        type=Path,
        default=Path("csv/freihand_landmarks_mapping.json"),
        help="JSON de mapeo sample_id → path_landmarks (opcional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Mostrar progreso",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not args.freihand_xyz.exists():
        raise FileNotFoundError(f"No existe {args.freihand_xyz}")
    
    mapping = process_freihand_landmarks(
        args.freihand_xyz,
        args.output_dir,
        verbose=args.verbose,
    )
    
    save_mapping_json(mapping, args.output_mapping)


if __name__ == "__main__":
    main()
