from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]

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
            "Genera gráfica comparativa de distribución MST para HaGRID y FreiHAND "
            "usando imágenes reales."
        )
    )
    parser.add_argument(
        "--hagrid-images-dir",
        type=Path,
        default=Path("datasets/hagrid_images/no_gesture"),
        help="Directorio de imágenes reales HaGRID.",
    )
    parser.add_argument(
        "--freihand-images-dir",
        type=Path,
        default=Path("datasets/FreiHAND_pub_v2/training/rgb"),
        help="Directorio de imágenes reales FreiHAND.",
    )
    parser.add_argument(
        "--freihand-xyz-json",
        type=Path,
        default=Path("datasets/FreiHAND_pub_v2/training_xyz.json"),
        help="JSON con landmarks 3D FreiHAND.",
    )
    parser.add_argument(
        "--freihand-k-json",
        type=Path,
        default=Path("datasets/FreiHAND_pub_v2/training_K.json"),
        help="JSON con intrínsecas de cámara FreiHAND.",
    )
    parser.add_argument(
        "--freihand-canonical-rgb-manifest",
        type=Path,
        default=Path("output/auditoria/freihand_rgb_canonical_manifest.csv"),
        help=(
            "CSV canónico de FreiHAND (sample_id) para filtrar imágenes válidas "
            "al mismo subconjunto usado en balanceo/entrenamiento."
        ),
    )
    parser.add_argument(
        "--hagrid-csv",
        type=Path,
        default=Path("csv/mst_hagrid_real_images.csv"),
        help="CSV cache de clasificación MST HaGRID.",
    )
    parser.add_argument(
        "--freihand-csv",
        type=Path,
        default=Path("csv/mst_freihand_real_images.csv"),
        help="CSV cache de clasificación MST FreiHAND.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("graficos/stgcn_atencion_mst/distribucion_mst_hagrid_vs_freihand.png"),
        help="Ruta de la figura final.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("graficos/stgcn_atencion_mst/distribucion_mst_hagrid_vs_freihand.json"),
        help="Ruta de resumen JSON.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recalcula MST aunque existan CSV cache.",
    )
    parser.add_argument(
        "--freihand-limit",
        type=int,
        default=0,
        help="Limita cantidad de imágenes FreiHAND a procesar (0=sin límite).",
    )
    return parser.parse_args()


def load_image_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32)


def classify_mst_level(rgb: np.ndarray) -> int:
    distances = np.linalg.norm(MST_RGB - rgb[None, :], axis=1)
    return int(np.argmin(distances)) + 1


def project_xyz_with_k(frame_xyz: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    z = np.clip(frame_xyz[:, 2], 1e-6, None)
    u = k_matrix[0, 0] * (frame_xyz[:, 0] / z) + k_matrix[0, 2]
    v = k_matrix[1, 1] * (frame_xyz[:, 1] / z) + k_matrix[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32)


def estimate_skin_rgb_from_landmarks(image_rgb: np.ndarray, coords_px: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    patches = []
    for x, y in coords_px:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            continue
        x0 = max(0, xi - 2)
        x1 = min(w, xi + 3)
        y0 = max(0, yi - 2)
        y1 = min(h, yi + 3)
        patch = image_rgb[y0:y1, x0:x1, :3]
        if patch.size == 0:
            continue
        patches.append(np.median(patch.reshape(-1, 3), axis=0))

    if patches:
        return np.median(np.stack(patches, axis=0), axis=0).astype(np.float32)

    # fallback central
    y0 = int(h * 0.25)
    y1 = int(h * 0.75)
    x0 = int(w * 0.25)
    x1 = int(w * 0.75)
    center = image_rgb[y0:y1, x0:x1, :3]
    return np.median(center.reshape(-1, 3), axis=0).astype(np.float32)


def estimate_skin_rgb_hagrid(image_rgb: np.ndarray) -> np.ndarray:
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
        h, w = image_rgb.shape[:2]
        y0 = int(h * 0.25)
        y1 = int(h * 0.75)
        x0 = int(w * 0.25)
        x1 = int(w * 0.75)
        patch = image_rgb[y0:y1, x0:x1]
        return np.median(patch.reshape(-1, 3), axis=0).astype(np.float32)

    pixels = image_rgb[mask]
    return np.median(pixels.reshape(-1, 3), axis=0).astype(np.float32)


def classify_hagrid_images(images_dir: Path) -> pd.DataFrame:
    image_paths = sorted(
        [
            p
            for p in images_dir.glob("**/*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )
    rows: list[dict[str, object]] = []
    for idx, image_path in enumerate(image_paths, start=1):
        image_rgb = load_image_rgb(image_path)
        skin_rgb = estimate_skin_rgb_hagrid(image_rgb)
        mst = classify_mst_level(skin_rgb)
        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path.resolve()),
                "mst_level": mst,
                "dataset": "hagrid",
            }
        )
        if idx % 500 == 0:
            print(f"HaGRID procesadas: {idx}/{len(image_paths)}")
    return pd.DataFrame(rows, columns=["image_id", "image_path", "mst_level", "dataset"])


def classify_freihand_images(images_dir: Path, xyz_json: Path, k_json: Path, limit: int) -> pd.DataFrame:
    xyz_payload = json.loads(xyz_json.read_text(encoding="utf-8"))
    k_payload = json.loads(k_json.read_text(encoding="utf-8"))

    xyz_all = [np.asarray(x, dtype=np.float32) for x in xyz_payload]
    k_all = [np.asarray(k, dtype=np.float32) for k in k_payload]

    image_paths = sorted(
        [
            p
            for p in images_dir.glob("*.jpg")
            if p.is_file() and p.stem.isdigit()
        ]
    )
    if limit > 0:
        image_paths = image_paths[:limit]

    rows: list[dict[str, object]] = []
    for idx, image_path in enumerate(image_paths, start=1):
        image_idx = int(image_path.stem)
        if image_idx >= len(xyz_all) or image_idx >= len(k_all):
            continue

        image_rgb = load_image_rgb(image_path)
        coords_px = project_xyz_with_k(xyz_all[image_idx], k_all[image_idx])
        skin_rgb = estimate_skin_rgb_from_landmarks(image_rgb, coords_px)
        mst = classify_mst_level(skin_rgb)

        rows.append(
            {
                "image_id": f"freihand_{image_idx:08d}",
                "image_path": str(image_path.resolve()),
                "mst_level": mst,
                "dataset": "freihand",
            }
        )

        if idx % 2000 == 0:
            print(f"FreiHAND procesadas: {idx}/{len(image_paths)}")

    return pd.DataFrame(rows, columns=["image_id", "image_path", "mst_level", "dataset"])


def classify_freihand_images_with_canonical_manifest(
    images_dir: Path,
    xyz_json: Path,
    k_json: Path,
    canonical_manifest_csv: Path,
    limit: int,
) -> pd.DataFrame:
    xyz_payload = json.loads(xyz_json.read_text(encoding="utf-8"))
    k_payload = json.loads(k_json.read_text(encoding="utf-8"))

    xyz_all = [np.asarray(x, dtype=np.float32) for x in xyz_payload]
    k_all = [np.asarray(k, dtype=np.float32) for k in k_payload]

    if not canonical_manifest_csv.exists():
        print(
            "Aviso: no existe manifiesto canónico FreiHAND; "
            "se usa procesamiento estándar de training/rgb."
        )
        return classify_freihand_images(images_dir, xyz_json, k_json, limit)

    canonical_df = pd.read_csv(canonical_manifest_csv)
    if "sample_id" not in canonical_df.columns:
        print(
            "Aviso: manifiesto canónico sin columna sample_id; "
            "se usa procesamiento estándar de training/rgb."
        )
        return classify_freihand_images(images_dir, xyz_json, k_json, limit)

    selected_ids: list[int] = []
    for raw in canonical_df["sample_id"].astype(str).tolist():
        raw = raw.strip().lower()
        if not raw.startswith("freihand_"):
            continue
        suffix = raw.split("_", 1)[1]
        if suffix.isdigit():
            selected_ids.append(int(suffix))

    selected_ids = sorted(set(selected_ids))
    if limit > 0:
        selected_ids = selected_ids[:limit]

    rows: list[dict[str, object]] = []
    total = len(selected_ids)
    for idx, image_idx in enumerate(selected_ids, start=1):
        image_path = images_dir / f"{image_idx:08d}.jpg"
        if not image_path.exists():
            continue
        if image_idx >= len(xyz_all) or image_idx >= len(k_all):
            continue

        image_rgb = load_image_rgb(image_path)
        coords_px = project_xyz_with_k(xyz_all[image_idx], k_all[image_idx])
        skin_rgb = estimate_skin_rgb_from_landmarks(image_rgb, coords_px)
        mst = classify_mst_level(skin_rgb)

        rows.append(
            {
                "image_id": f"freihand_{image_idx:08d}",
                "image_path": str(image_path.resolve()),
                "mst_level": mst,
                "dataset": "freihand",
            }
        )

        if idx % 2000 == 0:
            print(f"FreiHAND canónicas procesadas: {idx}/{total}")

    return pd.DataFrame(rows, columns=["image_id", "image_path", "mst_level", "dataset"])


def ensure_or_compute_csv(
    csv_path: Path,
    recompute: bool,
    compute_fn,
) -> pd.DataFrame:
    if csv_path.exists() and not recompute:
        return pd.read_csv(csv_path)

    df = compute_fn()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def render_plot(hagrid_df: pd.DataFrame, freihand_df: pd.DataFrame, output_plot: Path) -> None:
    levels = list(range(1, 11))

    h_counts = hagrid_df.groupby("mst_level").size().reindex(levels, fill_value=0).astype(int)
    f_counts = freihand_df.groupby("mst_level").size().reindex(levels, fill_value=0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=180)
    fig.patch.set_facecolor("#e9e9e9")

    for ax in axes:
        ax.set_facecolor("#efefef")

    colors = [MST_HEX[l - 1] for l in levels]

    ax_h = axes[0]
    ax_h.bar(levels, [h_counts[l] for l in levels], color=colors, edgecolor="#3a3a3a", linewidth=0.5)
    ax_h.set_title("Distribución MST por categoría - hagrid", fontsize=10)
    ax_h.set_xlabel("Nivel MST", fontsize=9)
    ax_h.set_ylabel("Número de imágenes", fontsize=9)
    ax_h.set_xticks(levels)
    ax_h.grid(axis="y", linestyle="--", alpha=0.3)

    for lvl in levels:
        value = int(h_counts[lvl])
        if value > 0:
            ax_h.text(lvl, value + max(1, int(0.01 * max(h_counts.max(), 1))), str(value), ha="center", va="bottom", fontsize=6)

    ax_f = axes[1]
    ax_f.bar(levels, [f_counts[l] for l in levels], color=colors, edgecolor="#3a3a3a", linewidth=0.5)
    ax_f.set_title("Distribución MST por categoría - freihand", fontsize=10)
    ax_f.set_xlabel("Nivel MST", fontsize=9)
    ax_f.set_ylabel("Número de imágenes", fontsize=9)
    ax_f.set_xticks(levels)
    ax_f.grid(axis="y", linestyle="--", alpha=0.3)

    for lvl in levels:
        value = int(f_counts[lvl])
        if value > 0:
            ax_f.text(lvl, value + max(1, int(0.01 * max(f_counts.max(), 1))), str(value), ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    hagrid_dir = (ROOT / args.hagrid_images_dir).resolve()
    freihand_dir = (ROOT / args.freihand_images_dir).resolve()
    xyz_json = (ROOT / args.freihand_xyz_json).resolve()
    k_json = (ROOT / args.freihand_k_json).resolve()

    hagrid_csv = (ROOT / args.hagrid_csv).resolve()
    freihand_csv = (ROOT / args.freihand_csv).resolve()
    output_plot = (ROOT / args.output_plot).resolve()
    output_summary = (ROOT / args.output_summary).resolve()

    if not hagrid_dir.exists():
        raise FileNotFoundError(f"No existe directorio HaGRID: {hagrid_dir}")
    if not freihand_dir.exists():
        raise FileNotFoundError(f"No existe directorio FreiHAND: {freihand_dir}")
    if not xyz_json.exists():
        raise FileNotFoundError(f"No existe training_xyz.json: {xyz_json}")
    if not k_json.exists():
        raise FileNotFoundError(f"No existe training_K.json: {k_json}")

    hagrid_df = ensure_or_compute_csv(
        csv_path=hagrid_csv,
        recompute=args.recompute,
        compute_fn=lambda: classify_hagrid_images(hagrid_dir),
    )
    freihand_df = ensure_or_compute_csv(
        csv_path=freihand_csv,
        recompute=args.recompute,
        compute_fn=lambda: classify_freihand_images_with_canonical_manifest(
            images_dir=freihand_dir,
            xyz_json=xyz_json,
            k_json=k_json,
            canonical_manifest_csv=(ROOT / args.freihand_canonical_rgb_manifest).resolve(),
            limit=args.freihand_limit,
        ),
    )

    render_plot(hagrid_df, freihand_df, output_plot)

    summary = {
        "hagrid_total": int(len(hagrid_df)),
        "freihand_total": int(len(freihand_df)),
        "hagrid_counts": {
            str(k): int(v)
            for k, v in hagrid_df.groupby("mst_level").size().sort_index().to_dict().items()
        },
        "freihand_counts": {
            str(k): int(v)
            for k, v in freihand_df.groupby("mst_level").size().sort_index().to_dict().items()
        },
        "hagrid_csv": str(hagrid_csv),
        "freihand_csv": str(freihand_csv),
        "output_plot": str(output_plot),
    }

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OK gráfico: {output_plot}")
    print(f"OK resumen: {output_summary}")
    print(f"OK CSV hagrid: {hagrid_csv}")
    print(f"OK CSV freihand: {freihand_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
