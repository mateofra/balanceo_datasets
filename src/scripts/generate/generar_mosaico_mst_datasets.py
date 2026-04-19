from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stgcn.hand_graph import HAND_EDGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un mosaico por categorias MST con 4 paneles por categoria: "
            "2 muestras FreiHAND y 2 muestras HaGRID."
        )
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("output/train_manifest_stgcn.csv"),
        help="Manifiesto ST-GCN con columnas dataset,mst,sample_id,path_landmarks.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("graficos/stgcn_atencion_mst/mosaico_mst_4_por_categoria.png"),
        help="Ruta de salida del mosaico.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("graficos/stgcn_atencion_mst/mosaico_mst_4_por_categoria.json"),
        help="Ruta de salida de metadata del mosaico.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo reproducible.",
    )
    parser.add_argument(
        "--category-mode",
        choices=["mst_block", "mst_level"],
        default="mst_block",
        help=(
            "Agrupacion de categorias: mst_block=claro/medio/oscuro, "
            "mst_level=1..10."
        ),
    )
    return parser.parse_args()


def freihand_image_path(sample_id: str) -> Path:
    raw_idx = sample_id.split("_", 1)[1]
    return ROOT / "datasets" / "FreiHAND_pub_v2" / "training" / "rgb" / f"{raw_idx}.jpg"


def find_hagrid_image(sample_id: str) -> Path | None:
    roots = [
        ROOT / "datasets" / "hagrid_dataset",
        ROOT / "data" / "raw" / "images",
        ROOT / "datasets" / "hagrid_kaggle_raw",
    ]
    exts = (".jpg", ".jpeg", ".png", ".webp")

    for root in roots:
        if not root.exists():
            continue
        for ext in exts:
            direct = root / f"{sample_id}{ext}"
            if direct.exists():
                return direct
            matches = list(root.rglob(f"{sample_id}{ext}"))
            if matches:
                return matches[0]
    return None


def _sample_idx_from_freihand(sample_id: str) -> int | None:
    if not sample_id.startswith("freihand_"):
        return None
    raw = sample_id.split("_", 1)[1]
    if not raw.isdigit():
        return None
    return int(raw)


def _load_freihand_k() -> list[np.ndarray] | None:
    k_path = ROOT / "datasets" / "FreiHAND_pub_v2" / "training_K.json"
    if not k_path.exists():
        return None
    payload = json.loads(k_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return None
    mats: list[np.ndarray] = []
    for item in payload:
        arr = np.asarray(item, dtype=np.float32)
        if arr.shape == (3, 3):
            mats.append(arr)
    return mats if mats else None


def _project_xyz(frame_xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = np.clip(frame_xyz[:, 2], 1e-6, None)
    u = k[0, 0] * (frame_xyz[:, 0] / z) + k[0, 2]
    v = k[1, 1] * (frame_xyz[:, 1] / z) + k[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32)


def infer_freihand_block(sample_id: str, landmark_path: Path, image_path: Path, k_mats: list[np.ndarray] | None) -> str:
    if (k_mats is None) or (not image_path.exists()) or (not landmark_path.exists()):
        return "medio"

    idx = _sample_idx_from_freihand(sample_id)
    if idx is None or idx >= len(k_mats):
        return "medio"

    xyz = np.load(landmark_path).astype(np.float32)
    if xyz.shape != (21, 3):
        return "medio"

    img = plt.imread(image_path)
    h, w = img.shape[:2]
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    coords = _project_xyz(xyz, k_mats[idx])
    vals: list[float] = []
    for x, y in coords:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            continue
        x0 = max(0, xi - 2)
        x1 = min(w, xi + 3)
        y0 = max(0, yi - 2)
        y1 = min(h, yi + 3)
        patch = img[y0:y1, x0:x1, :3]
        if patch.size == 0:
            continue
        luma = 0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2]
        vals.append(float(np.median(luma)))

    if not vals:
        return "medio"

    score = float(np.median(np.asarray(vals, dtype=np.float32)))
    if score >= 0.58:
        return "claro"
    if score >= 0.44:
        return "medio"
    return "oscuro"


def draw_landmark_panel(ax, landmark_path: Path, title: str) -> None:
    ax.set_facecolor("#111111")
    if not landmark_path.exists():
        ax.text(0.5, 0.5, "landmark\nno disponible", color="white", ha="center", va="center")
        ax.set_title(title, fontsize=8, color="white")
        ax.axis("off")
        return

    pts = np.load(landmark_path).astype(np.float32)
    if pts.shape != (21, 3):
        ax.text(0.5, 0.5, f"shape invalido\n{pts.shape}", color="white", ha="center", va="center")
        ax.set_title(title, fontsize=8, color="white")
        ax.axis("off")
        return

    xy = pts[:, :2].copy()

    # Si parece normalizado, usar rango [0,1].
    if float(xy.max()) <= 2.0 and float(xy.min()) >= -1.0:
        pass
    else:
        x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
        y_min, y_max = float(xy[:, 1].min()), float(xy[:, 1].max())
        if x_max > x_min:
            xy[:, 0] = (xy[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            xy[:, 1] = (xy[:, 1] - y_min) / (y_max - y_min)

    for s, d in HAND_EDGES:
        ax.plot([xy[s, 0], xy[d, 0]], [xy[s, 1], xy[d, 1]], color="#5ad1ff", lw=1.6, alpha=0.9)

    ax.scatter(xy[:, 0], xy[:, 1], c="#ffd166", s=22, edgecolors="#111111", linewidths=0.4)
    ax.set_title(title, fontsize=8, color="white")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_image_panel(ax, image_path: Path, title: str) -> None:
    ax.set_facecolor("black")
    if image_path.exists():
        img = plt.imread(image_path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "imagen no\ndisponible", color="white", ha="center", va="center")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    manifest_path = (ROOT / args.manifest_csv).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe manifiesto: {manifest_path}")

    df = pd.read_csv(manifest_path)
    req = {"sample_id", "dataset", "mst", "path_landmarks"}
    missing = req.difference(df.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas en manifiesto: {sorted(missing)}")

    df = df.dropna(subset=["mst"]).copy()
    df["mst"] = df["mst"].astype(int)

    if args.category_mode == "mst_block":
        if "condition" not in df.columns:
            raise RuntimeError("No existe columna condition en el manifiesto para mst_block.")
        categories = ["claro", "medio", "oscuro"]
        category_label = "MST"
    else:
        categories = sorted([m for m in df["mst"].unique().tolist() if 1 <= int(m) <= 10])
        if not categories:
            raise RuntimeError("No hay niveles MST validos (1..10) en el manifiesto.")
        category_label = "MST"

    k_mats = _load_freihand_k()

    n_rows = len(categories)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, max(2.8 * n_rows, 6)), dpi=180)
    if n_rows == 1:
        axes = np.array([axes])

    records: list[dict[str, object]] = []

    for row_i, category in enumerate(categories):
        if args.category_mode == "mst_block":
            subset = df[df["condition"].astype(str).str.lower() == str(category)]
        else:
            subset = df[df["mst"] == int(category)]

        frei = subset[subset["dataset"].str.lower() == "freihand"]
        hagrid = subset[subset["dataset"].str.lower() == "hagrid"]

        if args.category_mode == "mst_block" and len(frei):
            # Prioriza consistencia visual para FreiHAND usando inferencia en imagen.
            inferred_rows = []
            for _, rec in frei.iterrows():
                sid = str(rec["sample_id"])
                lmk = (ROOT / str(rec["path_landmarks"])).resolve()
                img = freihand_image_path(sid)
                inferred = infer_freihand_block(sid, lmk, img, k_mats)
                if inferred == category:
                    inferred_rows.append(rec)
                if len(inferred_rows) >= 40:
                    break
            if inferred_rows:
                frei_sel = pd.DataFrame(inferred_rows)
            else:
                frei_sel = frei
        else:
            frei_sel = frei

        frei_pick = frei_sel.sample(n=min(2, len(frei_sel)), random_state=int(rng.integers(0, 1_000_000))) if len(frei_sel) else frei_sel
        hagrid_pick = hagrid.sample(n=min(2, len(hagrid)), random_state=int(rng.integers(0, 1_000_000))) if len(hagrid) else hagrid

        picks = []
        picks.extend([("freihand", rec) for _, rec in frei_pick.iterrows()])
        picks.extend([("hagrid", rec) for _, rec in hagrid_pick.iterrows()])

        while len(picks) < 4:
            picks.append(("none", None))

        for col_i, (source, rec) in enumerate(picks[:4]):
            ax = axes[row_i, col_i]
            if rec is None:
                ax.text(0.5, 0.5, "sin muestra", ha="center", va="center")
                ax.axis("off")
                continue

            sample_id = str(rec["sample_id"])
            landmark_path = (ROOT / str(rec["path_landmarks"])).resolve()

            if source == "freihand":
                img_path = freihand_image_path(sample_id)
                title = f"{category_label} {category} | FreiHAND | {sample_id}"
                draw_image_panel(ax, img_path, title)
                records.append(
                    {
                        "category": category,
                        "dataset": "freihand",
                        "sample_id": sample_id,
                        "image_path": str(img_path),
                        "image_exists": img_path.exists(),
                    }
                )
            else:
                img_path = find_hagrid_image(sample_id)
                if img_path is not None and img_path.exists():
                    title = f"{category_label} {category} | HaGRID | {sample_id}"
                    draw_image_panel(ax, img_path, title)
                    records.append(
                        {
                            "category": category,
                            "dataset": "hagrid",
                            "sample_id": sample_id,
                            "image_path": str(img_path),
                            "image_exists": True,
                            "render_mode": "photo",
                        }
                    )
                else:
                    title = f"{category_label} {category} | HaGRID landmarks | {sample_id}"
                    draw_landmark_panel(ax, landmark_path, title)
                    records.append(
                        {
                            "category": category,
                            "dataset": "hagrid",
                            "sample_id": sample_id,
                            "image_path": None,
                            "image_exists": False,
                            "render_mode": "landmark_fallback",
                            "landmark_path": str(landmark_path),
                        }
                    )

        axes[row_i, 0].set_ylabel(f"{category_label} {category}", fontsize=10)

    fig.suptitle(
        "Mosaico MST por dataset (4 por categoria: 2 FreiHAND + 2 HaGRID)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_img = (ROOT / args.output_image).resolve()
    out_json = (ROOT / args.output_metadata).resolve()
    out_img.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_img)
    plt.close(fig)

    out_json.write_text(
        json.dumps(
            {
                "manifest_csv": str(manifest_path),
                "category_mode": args.category_mode,
                "categories": categories,
                "note": (
                    "Si no hay fotos HaGRID locales, se usa fallback de landmarks para mantener 2+2 por categoria."
                ),
                "items": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"OK imagen: {out_img}")
    print(f"OK metadata: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
