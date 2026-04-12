from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

MST_HEX = [
    "#f6ede4",
    "#f3e7db",
    "#f7ead0",
    "#eadaba",
    "#d7bd96",
    "#a07e56",
    "#825c43",
    "#604134",
    "#3a312a",
    "#292420",
]


def hex_to_rgb(color: str) -> np.ndarray:
    value = color.lstrip("#")
    return np.array(
        [int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)],
        dtype=np.float32,
    )


MST_RGB = np.stack([hex_to_rgb(c) for c in MST_HEX], axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clasifica imágenes reales por MST y genera una gráfica comparativa "
            "de categorías claro/medio/oscuro y niveles 1..10."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("datasets/hagrid_images/no_gesture"),
        help="Directorio de imágenes reales a clasificar.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("csv/mst_imagenes_reales_hagrid_no_gesture.csv"),
        help="CSV con clasificación MST por imagen.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("graficos/stgcn_atencion_mst/comparacion_mst_imagenes_reales.png"),
        help="Imagen final de comparación MST.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("graficos/stgcn_atencion_mst/comparacion_mst_imagenes_reales.json"),
        help="Resumen JSON de la clasificación.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para selección reproducible de ejemplos.",
    )
    return parser.parse_args()


def mst_block(mst_level: int) -> str:
    if 1 <= mst_level <= 3:
        return "claro"
    if 4 <= mst_level <= 7:
        return "medio"
    return "oscuro"


def load_image_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32)


def estimate_skin_rgb(image_rgb: np.ndarray) -> np.ndarray:
    # Regla simple de máscara de piel en RGB (robusta y sin dependencias nativas).
    r = image_rgb[..., 0]
    g = image_rgb[..., 1]
    b = image_rgb[..., 2]

    mask = (
        (r > 95)
        & (g > 40)
        & (b > 20)
        & ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15)
        & (np.abs(r - g) > 15)
        & (r > g)
        & (r > b)
    )

    if np.count_nonzero(mask) < 300:
        # Fallback: región central de la imagen si no se detecta piel suficiente.
        h, w = image_rgb.shape[:2]
        y0 = int(h * 0.25)
        y1 = int(h * 0.75)
        x0 = int(w * 0.25)
        x1 = int(w * 0.75)
        patch = image_rgb[y0:y1, x0:x1]
        return np.median(patch.reshape(-1, 3), axis=0).astype(np.float32)

    pixels = image_rgb[mask]
    return np.median(pixels.reshape(-1, 3), axis=0).astype(np.float32)


def classify_mst_level(rgb: np.ndarray) -> int:
    distances = np.linalg.norm(MST_RGB - rgb[None, :], axis=1)
    return int(np.argmin(distances)) + 1


def classify_images(image_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in image_paths:
        image_rgb = load_image_rgb(path)
        skin_rgb = estimate_skin_rgb(image_rgb)
        mst_level_value = classify_mst_level(skin_rgb)
        rows.append(
            {
                "image_id": path.stem,
                "image_path": str(path.resolve()),
                "mst_level": mst_level_value,
                "mst_block": mst_block(mst_level_value),
                "source": "real_image_classifier",
            }
        )
    return pd.DataFrame(rows)


def render_comparison_plot(df: pd.DataFrame, output_plot: Path, seed: int) -> None:
    random.seed(seed)

    block_order = ["claro", "medio", "oscuro"]
    level_order = list(range(1, 11))

    block_counts = (
        df.groupby("mst_block").size().reindex(block_order, fill_value=0).astype(int)
    )
    level_counts = (
        df.groupby("mst_level").size().reindex(level_order, fill_value=0).astype(int)
    )

    fig = plt.figure(figsize=(16, 10), dpi=180)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.25])

    ax_block = fig.add_subplot(gs[0, :2])
    ax_level = fig.add_subplot(gs[0, 2:])

    block_colors = {"claro": "#f2d6bd", "medio": "#b78f66", "oscuro": "#4a3a2e"}
    ax_block.bar(block_order, [block_counts[b] for b in block_order], color=[block_colors[b] for b in block_order])
    ax_block.set_title("Comparación por bloque MST")
    ax_block.set_ylabel("Número de imágenes")

    level_colors = [MST_HEX[i - 1] for i in level_order]
    ax_level.bar([str(i) for i in level_order], [level_counts[i] for i in level_order], color=level_colors)
    ax_level.set_title("Comparación por nivel MST (1-10)")
    ax_level.set_ylabel("Número de imágenes")

    # Muestras reales por bloque para comparación visual.
    sample_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
        fig.add_subplot(gs[2, 3]),
    ]

    slots_per_block = {"claro": 2, "medio": 3, "oscuro": 3}
    slot_idx = 0
    for block_name in block_order:
        block_rows = df[df["mst_block"] == block_name]
        paths = [Path(p) for p in block_rows["image_path"].tolist()]
        random.shuffle(paths)

        needed = slots_per_block[block_name]
        selected = paths[:needed]

        for image_path in selected:
            ax = sample_axes[slot_idx]
            img = plt.imread(image_path)
            ax.imshow(img)
            ax.set_title(f"{block_name}\n{image_path.name}", fontsize=8)
            ax.axis("off")
            slot_idx += 1

    while slot_idx < len(sample_axes):
        sample_axes[slot_idx].axis("off")
        slot_idx += 1

    fig.suptitle("Comparación de categorías MST con imágenes reales (HaGRID no_gesture)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    images_dir = (ROOT / args.images_dir).resolve()
    output_csv = (ROOT / args.output_csv).resolve()
    output_plot = (ROOT / args.output_plot).resolve()
    output_summary = (ROOT / args.output_summary).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"No existe directorio de imágenes: {images_dir}")

    image_paths = sorted(
        [
            p
            for p in images_dir.glob("**/*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )
    if not image_paths:
        raise RuntimeError(f"No se encontraron imágenes en {images_dir}")

    df = classify_images(image_paths)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    render_comparison_plot(df, output_plot, seed=args.seed)

    summary = {
        "images_dir": str(images_dir),
        "total_images": int(len(df)),
        "by_mst_block": {
            k: int(v)
            for k, v in df.groupby("mst_block").size().sort_index().to_dict().items()
        },
        "by_mst_level": {
            str(k): int(v)
            for k, v in df.groupby("mst_level").size().sort_index().to_dict().items()
        },
        "output_csv": str(output_csv),
        "output_plot": str(output_plot),
    }

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OK clasificación MST: {output_csv}")
    print(f"OK gráfico comparación: {output_plot}")
    print(f"OK resumen: {output_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
