from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def _mst_block(mst_level: int) -> str:
    if 1 <= mst_level <= 4:
        return "claro"
    if 5 <= mst_level <= 7:
        return "medio"
    if 8 <= mst_level <= 10:
        return "oscuro"
    raise ValueError(f"Nivel MST invalido: {mst_level}")


def _parse_mst_level(sample_id: str) -> int:
    parts = sample_id.split("_")
    for index, part in enumerate(parts):
        if part == "MST" and index + 1 < len(parts):
            try:
                level = int(parts[index + 1])
            except ValueError as exc:
                raise ValueError(f"No se pudo leer MST en '{sample_id}'") from exc
            if 1 <= level <= 10:
                return level
    raise ValueError(f"No se encontro nivel MST en '{sample_id}'")


def build_manifest(
    sample_dir: Path,
    output_manifest: Path,
    dataset_name: str,
    source_kind: str,
) -> Path:
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de muestras: {sample_dir}")

    rows: list[dict[str, object]] = []
    for png_path in sorted(sample_dir.glob("*.png")):
        sample_id = png_path.stem
        mst_level = _parse_mst_level(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "path": str(png_path.resolve()),
                "dataset": dataset_name,
                "mst_level": mst_level,
                "mst_block": _mst_block(mst_level),
                "source_kind": source_kind,
                "input_csv": str(sample_dir.resolve()),
            }
        )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "path", "dataset", "mst_level", "mst_block", "source_kind", "input_csv"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_manifest


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Construir un manifiesto CSV a partir de muestras MANO en disco")
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path("datasets/synthetic_mst/mano_samples_balanced"),
        help="Directorio con PNGs MANO a registrar.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("datasets/synthetic_mst/metadata/manifest_mano_samples_balanced.csv"),
        help="Ruta del CSV de salida.",
    )
    parser.add_argument(
        "--dataset-name",
        default="mano",
        help="Nombre de dataset a registrar en la columna dataset.",
    )
    parser.add_argument(
        "--source-kind",
        default="synthetic",
        help="Valor para la columna source_kind.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output = build_manifest(args.sample_dir, args.output_manifest, args.dataset_name, args.source_kind)
    print(f"[mano-manifest] Manifest guardado en {output}")


if __name__ == "__main__":
    main()