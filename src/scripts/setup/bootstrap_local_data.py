from __future__ import annotations

import argparse
from pathlib import Path


# Estructura minima y recomendada para un clon nuevo.
IGNORED_DIRS = [
    "datasets",
    "datasets/FreiHAND_pub_v2",
    "datasets/FreiHAND_pub_v2/training",
    "datasets/FreiHAND_pub_v2/training/rgb",
    "datasets/ann_subsample",
    "datasets/ann_train_val",
    "datasets/ann_test",
    "data",
    "data/raw",
    "data/raw/images",
    "data/processed",
    "data/processed/landmarks",
    "data/processed/secuencias_stgcn",
    "stgcn/data",
]


# Archivos clave esperados por el pipeline principal.
KEY_DATA_FILES = [
    "datasets/training_xyz.json",
    "datasets/training_K.json",
    "datasets/training_scale.json",
]


def create_dirs(repo_root: Path) -> list[Path]:
    created: list[Path] = []
    for rel in IGNORED_DIRS:
        path = repo_root / rel
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created


def verify_layout(repo_root: Path) -> tuple[list[Path], list[Path]]:
    present: list[Path] = []
    missing: list[Path] = []
    for rel in KEY_DATA_FILES:
        path = repo_root / rel
        if path.exists():
            present.append(path)
        else:
            missing.append(path)
    return present, missing


def write_local_readme(repo_root: Path) -> Path:
    out = repo_root / "datasets" / "README_LOCAL.md"
    content = """# Datasets Locales (No Versionados)

Este directorio esta ignorado por git y se crea localmente.

## Archivos minimos esperados

- `training_xyz.json`
- `training_K.json`
- `training_scale.json`
- `ann_subsample/` (JSON de HaGRID por gesto)

## Notas

- FreiHAND: descarga manual desde la web oficial.
- HaGRID: descarga manual (o con Kaggle CLI) y copiar anotaciones a `ann_subsample/`.
- Este archivo se regenera con `src/scripts/setup/bootstrap_local_data.py`.
"""
    out.write_text(content, encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea la estructura local de directorios ignorados por git para datasets "
            "y verifica archivos minimos esperados."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Ruta raiz del repositorio (default: cwd)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="No crea directorios; solo verifica si los archivos clave existen.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not args.verify_only:
        created = create_dirs(root)
        readme_path = write_local_readme(root)
        print(f"Directorios creados: {len(created)}")
        for path in created:
            print(f"  + {path}")
        print(f"README local generado: {readme_path}")

    present, missing = verify_layout(root)
    print("\nVerificacion de archivos clave:")
    for path in present:
        print(f"  OK  {path}")
    for path in missing:
        print(f"  MISS {path}")

    if missing:
        print("\nEstado: estructura creada, pero faltan datasets reales por descargar/copiar.")
        return 2

    print("\nEstado: estructura y archivos clave listos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
