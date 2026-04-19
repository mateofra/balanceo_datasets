from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.temporal_sequence_utils import generate_temporal_sequence, sample_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "output/train_manifest_stgcn.csv"
DEFAULT_OUTPUT_MANIFEST = REPO_ROOT / "output/manifest_canonico.csv"
DEFAULT_SEQUENCE_DIR = REPO_ROOT / "data/processed/secuencias_stgcn"
DEFAULT_DATASET = "hagrid"
DEFAULT_QUALITY = "annotation_2d_projected"
DEFAULT_SEED = 42
DEFAULT_FRAMES = 16


def generar_secuencia(landmarks: np.ndarray, T: int = DEFAULT_FRAMES, sigma: float = 0.015, seed: int | None = None) -> np.ndarray:
    """Convierte landmarks (21, 3) en una secuencia temporal suave."""
    return generate_temporal_sequence(landmarks, T=T, sigma=sigma, seed=seed)


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path.replace("\\", "/"))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _stratified_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in df.groupby("label", sort=True):
        indices = group.index.to_list()
        rng.shuffle(indices)

        total = len(indices)
        train_count = int(round(total * 0.70))
        val_count = int(round(total * 0.15))
        test_count = total - train_count - val_count

        if test_count < 0:
            deficit = -test_count
            reduce_val = min(deficit, max(0, val_count - 1))
            val_count -= reduce_val
            deficit -= reduce_val
            train_count = max(1, train_count - deficit)
            test_count = total - train_count - val_count

        if total >= 3:
            if train_count == 0:
                train_count = 1
                if test_count > val_count:
                    test_count -= 1
                else:
                    val_count -= 1
            if val_count == 0:
                val_count = 1
                if test_count > train_count:
                    test_count -= 1
                else:
                    train_count -= 1
            if test_count == 0:
                test_count = 1
                if train_count > val_count:
                    train_count -= 1
                else:
                    val_count -= 1

        train_end = train_count
        val_end = train_count + val_count

        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])

    return (
        df.loc[train_indices].reset_index(drop=True),
        df.loc[val_indices].reset_index(drop=True),
        df.loc[test_indices].reset_index(drop=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara secuencias y manifiesto canónico para ST-GCN")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Manifiesto de entrada con path_landmarks")
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST, help="CSV de salida con splits canónicos")
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR, help="Directorio de secuencias temporales")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset a conservar en el flujo")
    parser.add_argument("--landmark-quality", default=DEFAULT_QUALITY, help="Calidad de landmarks a conservar")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla base reproducible")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="Cantidad de frames por secuencia")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    df = df.drop_duplicates(subset="sample_id", keep="first")
    df = df[
        (df["dataset"] == args.dataset)
        & (df["landmark_quality"] == args.landmark_quality)
        & (df["label"] != "unknown")
    ].reset_index(drop=True)
    print(f"Muestras base: {len(df)}")

    args.sequence_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    rutas_seq: list[str | None] = []
    generadas = 0
    fallidas = 0

    for _, row in df.iterrows():
        src = _resolve_repo_path(str(row["path_landmarks"]))
        if not src.exists():
            rutas_seq.append(None)
            fallidas += 1
            continue

        dst = args.sequence_dir / f"{row['sample_id']}.npy"
        if not dst.exists():
            landmarks = np.load(str(src))
            sequence = generar_secuencia(landmarks, T=args.frames, seed=sample_seed(args.seed, str(row["sample_id"])))
            np.save(str(dst), sequence)

        rutas_seq.append(_relative_to_repo(dst))
        generadas += 1

    print(f"Generadas: {generadas} | Fallidas: {fallidas}")

    df["path_secuencia"] = rutas_seq
    df = df[df["path_secuencia"].notna()].reset_index(drop=True)

    train, val, test = _stratified_split(df, seed=args.seed)

    train = train.copy(); train["split"] = "train"
    val = val.copy(); val["split"] = "val"
    test = test.copy(); test["split"] = "test"

    canonico = pd.concat([train, val, test]).reset_index(drop=True)
    canonico.to_csv(args.output_manifest, index=False)

    print(f"\nManifiesto canónico guardado: {args.output_manifest}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Clases: {df['label'].nunique()}")


if __name__ == "__main__":
    main()
