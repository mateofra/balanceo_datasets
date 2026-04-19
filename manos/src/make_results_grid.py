"""Construye una cuadricula comparativa con una imagen por nivel MST.

Uso:
    uv run python src/make_results_grid.py
    uv run python src/make_results_grid.py --sample-dir data/synthetic_samples --out reports/mst_grid.png
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import cv2
import numpy as np

MST_GROUPS = [f"MST_{i}" for i in range(1, 11)]
_LEGACY_TYPE_TO_MST = {
    "Type_I": "MST_1",
    "Type_II": "MST_2",
    "Type_III": "MST_4",
    "Type_IV": "MST_6",
    "Type_V": "MST_8",
    "Type_VI": "MST_10",
}
_TILE_SIZE = 128
_HEADER_HEIGHT = 28
_GAP = 12
_BORDER = 16
_TITLE_BLOCK = 56
_BG_COLOR = (18, 18, 24)
_TEXT_COLOR = (240, 240, 240)
_SUBTEXT_COLOR = (170, 170, 170)
_DEFAULT_SAMPLE_DIR = "datasets/synthetic_mst/mano_samples_balanced"
_SAMPLE_DIR_CANDIDATES = [
    _DEFAULT_SAMPLE_DIR,
    "data/synthetic_samples",
    "data/synthetic_samples_balanced",
    "data/synthetic_samples_conservative",
    "data/synthetic_samples_extreme",
]


def parse_group(stem: str) -> str | None:
    parts = stem.split("_")
    try:
        idx = parts.index("MST")
        candidate = f"MST_{parts[idx + 1]}"
        if candidate in MST_GROUPS:
            return candidate
    except (ValueError, IndexError):
        pass

    try:
        idx = parts.index("Type")
        legacy = f"Type_{parts[idx + 1]}"
        return _LEGACY_TYPE_TO_MST.get(legacy)
    except (ValueError, IndexError):
        return None


def find_example_images(sample_dir: str) -> dict[str, str]:
    examples: dict[str, str] = {}
    for filename in sorted(os.listdir(sample_dir)):
        if not filename.endswith(".png"):
            continue
        group = parse_group(filename[:-4])
        if group is None or group in examples:
            continue
        examples[group] = os.path.join(sample_dir, filename)
        if len(examples) == len(MST_GROUPS):
            break
    return examples


def resolve_sample_dir(sample_dir: str) -> str:
    """Resuelve un directorio de muestras valido con fallback automatico.

    Si la ruta solicitada no existe y corresponde al valor por defecto historico,
    se intenta con los nuevos datasets de produccion.
    """
    if os.path.isdir(sample_dir):
        return sample_dir

    if sample_dir != _DEFAULT_SAMPLE_DIR:
        raise FileNotFoundError(f"No existe el directorio de muestras: {sample_dir}")

    for candidate in _SAMPLE_DIR_CANDIDATES[1:]:
        if os.path.isdir(candidate):
            return candidate

    candidates = ", ".join(_SAMPLE_DIR_CANDIDATES)
    raise FileNotFoundError(
        "No se encontro un directorio de muestras valido. "
        f"Rutas probadas: {candidates}"
    )


def load_tile(image_path: str, tile_size: int = _TILE_SIZE) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")
    if image.shape[:2] != (tile_size, tile_size):
        image = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    return image


def draw_label(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def build_grid(sample_dir: str, out_path: str, title: str = "Muestras sinteticas de mano por nivel MST") -> str:
    sample_dir = resolve_sample_dir(sample_dir)
    examples = find_example_images(sample_dir)
    missing = [group for group in MST_GROUPS if group not in examples]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Falta al menos una imagen para: {missing_str}. Genera muestras primero en {sample_dir}."
        )

    cols = 5
    rows = 2
    grid_top = _BORDER + _TITLE_BLOCK
    canvas_w = _BORDER * 2 + cols * _TILE_SIZE + (cols - 1) * _GAP
    canvas_h = grid_top + rows * (_HEADER_HEIGHT + _TILE_SIZE) + (rows - 1) * _GAP + _BORDER
    canvas = np.full((canvas_h, canvas_w, 3), _BG_COLOR, dtype=np.uint8)

    draw_label(canvas, title, _BORDER, 24)
    cv2.putText(
        canvas,
        f"Un ejemplo por subgrupo desde {sample_dir}",
        (_BORDER, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        _SUBTEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    for index, group in enumerate(MST_GROUPS):
        row = index // cols
        col = index % cols
        x = _BORDER + col * (_TILE_SIZE + _GAP)
        y = grid_top + row * (_HEADER_HEIGHT + _TILE_SIZE + _GAP)

        draw_label(canvas, group, x, y + 18)
        tile = load_tile(examples[group])
        canvas[y + _HEADER_HEIGHT:y + _HEADER_HEIGHT + _TILE_SIZE, x:x + _TILE_SIZE] = tile
        cv2.rectangle(
            canvas,
            (x - 1, y + _HEADER_HEIGHT - 1),
            (x + _TILE_SIZE, y + _HEADER_HEIGHT + _TILE_SIZE),
            (70, 70, 80),
            1,
        )

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    if not cv2.imwrite(out_path, canvas):
        raise RuntimeError(f"No se pudo guardar la cuadricula en {out_path}")
    return out_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Crear una cuadricula comparativa de manos sinteticas")
    parser.add_argument(
        "--sample-dir",
        default=_DEFAULT_SAMPLE_DIR,
        help=(
            "Directorio con muestras .png generadas "
            "(por defecto: data/synthetic_samples; si no existe, usa fallback automatico)"
        ),
    )
    parser.add_argument(
        "--out",
        default="reports/mst_grid.png",
        help="Ruta de imagen de salida (por defecto: reports/mst_grid.png)",
    )
    parser.add_argument(
        "--title",
        default="Muestras sinteticas de mano por nivel MST",
        help="Titulo a mostrar en la parte superior de la cuadricula",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_path = build_grid(args.sample_dir, args.out, args.title)
    print(f"[grid] Cuadricula guardada en {output_path}")


if __name__ == "__main__":
    main()
