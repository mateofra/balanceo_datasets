"""Construye un grid comparativo de modos de pose: conservative vs balanced vs extreme.

Uso:
    uv run python src/make_pose_mode_grid.py
    uv run python src/make_pose_mode_grid.py --rows 5 --mst-level 4 --out reports/pose_mode_comparison.png
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import cv2
import numpy as np

_MODES = ["conservative", "balanced", "extreme"]
_TILE = 128
_COL_GAP = 18
_ROW_GAP = 12
_BORDER = 20
_HEADER_H = 62
_MODE_LABEL_H = 28
_SUBTEXT_COLOR = (170, 170, 170)
_TEXT_COLOR = (240, 240, 240)
_BG = (16, 18, 24)


def _draw_text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.55, color: tuple[int, int, int] = _TEXT_COLOR) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _pick_images(sample_dir: str, group_tag: str, rows: int) -> list[str]:
    selected: list[str] = []
    for name in sorted(os.listdir(sample_dir)):
        if not name.endswith(".png"):
            continue
        if group_tag not in name:
            continue
        selected.append(os.path.join(sample_dir, name))
        if len(selected) == rows:
            break

    if len(selected) < rows:
        raise RuntimeError(
            f"No hay suficientes imagenes en {sample_dir} para {group_tag}. "
            f"Necesarias: {rows}, encontradas: {len(selected)}"
        )
    return selected


def _load(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"No se pudo leer la imagen: {path}")
    if img.shape[:2] != (_TILE, _TILE):
        img = cv2.resize(img, (_TILE, _TILE), interpolation=cv2.INTER_AREA)
    return img


def build_pose_mode_grid(
    rows: int,
    group_tag: str,
    out_path: str,
    conservative_dir: str,
    balanced_dir: str,
    extreme_dir: str,
) -> str:
    dirs = {
        "conservative": conservative_dir,
        "balanced": balanced_dir,
        "extreme": extreme_dir,
    }

    picks: dict[str, list[str]] = {}
    for mode, path in dirs.items():
        if not os.path.isdir(path):
            raise RuntimeError(f"No existe el directorio para {mode}: {path}")
        picks[mode] = _pick_images(path, group_tag, rows)

    width = _BORDER * 2 + len(_MODES) * _TILE + (len(_MODES) - 1) * _COL_GAP
    grid_top = _BORDER + _HEADER_H
    height = grid_top + rows * (_MODE_LABEL_H + _TILE) + (rows - 1) * _ROW_GAP + _BORDER
    canvas = np.full((height, width, 3), _BG, dtype=np.uint8)

    _draw_text(canvas, "Comparativa de diversidad de pose (MANO mano derecha)", _BORDER, _BORDER + 22, 0.62)
    _draw_text(canvas, f"Columnas: conservative | balanced | extreme   |   Grupo: {group_tag}", _BORDER, _BORDER + 48, 0.45, _SUBTEXT_COLOR)

    for col, mode in enumerate(_MODES):
        x = _BORDER + col * (_TILE + _COL_GAP)
        _draw_text(canvas, mode, x, grid_top + 18, 0.52)

        for row in range(rows):
            y = grid_top + row * (_MODE_LABEL_H + _TILE + _ROW_GAP) + _MODE_LABEL_H
            tile = _load(picks[mode][row])
            canvas[y:y + _TILE, x:x + _TILE] = tile
            cv2.rectangle(canvas, (x - 1, y - 1), (x + _TILE, y + _TILE), (80, 84, 96), 1)

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    if not cv2.imwrite(out_path, canvas):
        raise RuntimeError(f"No se pudo guardar el grid en {out_path}")
    return out_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Crear grid comparativo de modos de pose")
    parser.add_argument("--rows", type=int, default=5, help="Filas por modo (por defecto: 5)")
    parser.add_argument("--mst-level", type=int, default=4, help="Nivel MST a visualizar (1-10; por defecto: 4)")
    parser.add_argument(
        "--fitzpatrick",
        default=None,
        help="Compatibilidad legacy: usa un tag Type_* para filtrar (sobrescribe --mst-level)",
    )
    parser.add_argument("--conservative-dir", default="data/samples_pose_conservative", help="Directorio para conservative")
    parser.add_argument("--balanced-dir", default="data/samples_pose_balanced", help="Directorio para balanced")
    parser.add_argument("--extreme-dir", default="data/samples_pose_extreme", help="Directorio para extreme")
    parser.add_argument("--out", default="reports/pose_mode_comparison.png", help="Ruta de salida del grid")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.fitzpatrick:
        group_tag = args.fitzpatrick
    else:
        if args.mst_level < 1 or args.mst_level > 10:
            raise ValueError("--mst-level debe estar en el rango 1..10")
        group_tag = f"MST_{args.mst_level}"

    path = build_pose_mode_grid(
        rows=args.rows,
        group_tag=group_tag,
        out_path=args.out,
        conservative_dir=args.conservative_dir,
        balanced_dir=args.balanced_dir,
        extreme_dir=args.extreme_dir,
    )
    print(f"[pose-grid] Grid comparativo guardado en {path}")


if __name__ == "__main__":
    main()
