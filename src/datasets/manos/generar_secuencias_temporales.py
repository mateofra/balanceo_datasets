from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.preprocessing.temporal_sequence_utils import generate_temporal_sequence, sample_seed, validate_temporal_sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "data/synthetic_samples"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/secuencias_stgcn"
DEFAULT_MANIFEST = REPO_ROOT / "output/manifest_mano_secuencias.csv"


def _load_landmarks(path: Path) -> np.ndarray | None:
    landmarks = np.load(path)
    if landmarks.shape == (21, 3):
        return landmarks.astype(np.float32)
    if landmarks.shape == (21, 2):
        xyz = np.zeros((21, 3), dtype=np.float32)
        xyz[:, :2] = landmarks.astype(np.float32)
        return xyz
    return None


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def generar_secuencia_mano(landmarks: np.ndarray, T: int, seed: int | None = None) -> np.ndarray:
    """Genera una secuencia temporal MANO con la misma logica suave compartida."""
    sequence = generate_temporal_sequence(landmarks, T=T, seed=seed)
    validate_temporal_sequence(sequence)
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera secuencias temporales para muestras MANO")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directorio con *_landmarks3d.npy")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directorio de secuencias de salida")
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_MANIFEST, help="CSV de manifiesto de secuencias")
    parser.add_argument("--T", type=int, default=16, help="Cantidad de frames por secuencia")
    parser.add_argument("--seed", type=int, default=42, help="Semilla base reproducible")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de entrada: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    candidates = sorted(args.input_dir.rglob("*_landmarks3d.npy"))
    if not candidates:
        candidates = sorted(args.input_dir.rglob("*_landmarks.npy"))

    if not candidates:
        raise FileNotFoundError(f"No se encontraron landmarks MANO en {args.input_dir}")

    rows: list[dict[str, object]] = []
    for index, landmark_path in enumerate(candidates):
        landmarks = _load_landmarks(landmark_path)
        if landmarks is None:
            continue

        sample_id = landmark_path.stem.replace("_landmarks3d", "").replace("_landmarks", "")
        sequence = generar_secuencia_mano(
            landmarks,
            T=args.T,
            seed=sample_seed(args.seed, sample_id),
        )

        output_path = args.output_dir / f"mano_{sample_id}.npy"
        np.save(output_path, sequence)

        rows.append(
            {
                "sample_id": sample_id,
                "path_secuencia": _relative_path(output_path),
                "path_landmarks": _relative_path(landmark_path),
                "dataset": "mano",
                "source_kind": "synthetic",
                "T": args.T,
            }
        )

        if (index + 1) % 500 == 0:
            print(f"  Procesadas {index + 1}/{len(candidates)} muestras MANO")

    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "path_secuencia", "path_landmarks", "dataset", "source_kind", "T"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[mano] Secuencias generadas: {len(rows)}")
    print(f"[mano] Manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()