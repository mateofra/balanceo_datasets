from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def hex_to_rgb(h: str) -> np.ndarray:
    v = h.lstrip("#")
    return np.array([int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)], dtype=np.float32)


MST_RGB = np.stack([hex_to_rgb(h) for h in MST_HEX], axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera MST real a partir de imagen + landmarks proyectados."
    )
    p.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("output/train_manifest_stgcn.csv"),
        help="Manifiesto base con sample_id,dataset,path_landmarks",
    )
    p.add_argument(
        "--freihand-k-json",
        type=Path,
        default=Path("datasets/FreiHAND_pub_v2/training_K.json"),
        help="Matriz intrinseca de camara FreiHAND",
    )
    p.add_argument(
        "--freihand-image-dir",
        type=Path,
        default=Path("datasets/FreiHAND_pub_v2/training/rgb"),
        help="Directorio de imagenes FreiHAND",
    )
    p.add_argument(
        "--hagrid-image-roots",
        nargs="+",
        default=["datasets/hagrid_dataset", "data/raw/images"],
        help="Raices de imagenes HaGRID",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("csv/mst_real_dataset.csv"),
        help="CSV salida con sample_id,mst_level",
    )
    p.add_argument(
        "--output-debug-csv",
        type=Path,
        default=Path("csv/mst_real_dataset_debug.csv"),
        help="CSV debug con estado por muestra",
    )
    p.add_argument(
        "--output-summary",
        type=Path,
        default=Path("output/mst_real_summary.json"),
        help="Resumen de cobertura",
    )
    return p.parse_args()


def load_image_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    return arr


def sample_idx_from_freihand(sample_id: str) -> int | None:
    if not sample_id.startswith("freihand_"):
        return None
    raw = sample_id.split("_", 1)[1]
    if not raw.isdigit():
        return None
    return int(raw)


def project_freihand_xyz_to_pixels(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = np.clip(xyz[:, 2], 1e-6, None)
    u = k[0, 0] * (xyz[:, 0] / z) + k[0, 2]
    v = k[1, 1] * (xyz[:, 1] / z) + k[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32)


def map_xy_to_pixels(xy: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = xy.astype(np.float32).copy()
    if float(pts.max()) <= 2.0 and float(pts.min()) >= -1.0:
        pts[:, 0] *= w
        pts[:, 1] *= h
    return pts


def estimate_skin_rgb(image_rgb: np.ndarray, coords_px: np.ndarray) -> np.ndarray | None:
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

    if not patches:
        return None
    return np.median(np.stack(patches, axis=0), axis=0).astype(np.float32)


def classify_mst_from_rgb(rgb: np.ndarray) -> int:
    d = np.linalg.norm(MST_RGB - rgb[None, :], axis=1)
    idx = int(np.argmin(d))
    return idx + 1


def find_hagrid_image(sample_id: str, roots: list[Path]) -> Path | None:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    for root in roots:
        if not root.exists():
            continue
        for ext in exts:
            p = root / f"{sample_id}{ext}"
            if p.exists():
                return p
            matches = list(root.rglob(f"{sample_id}{ext}"))
            if matches:
                return matches[0]
    return None


def main() -> int:
    args = parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    needed = {"sample_id", "dataset", "path_landmarks"}
    miss = needed.difference(manifest.columns)
    if miss:
        raise RuntimeError(f"Faltan columnas en manifiesto: {sorted(miss)}")

    k_payload = json.loads(args.freihand_k_json.read_text(encoding="utf-8"))
    k_mats = [np.asarray(k, dtype=np.float32) for k in k_payload]

    hagrid_roots = [Path(p) for p in args.hagrid_image_roots]

    debug_rows: list[dict[str, object]] = []
    mst_rows: list[dict[str, object]] = []

    summary = {
        "freihand": {"total": 0, "with_image": 0, "classified": 0, "missing_image": 0, "errors": 0},
        "hagrid": {"total": 0, "with_image": 0, "classified": 0, "missing_image": 0, "errors": 0},
    }

    for i, row in manifest.iterrows():
        sample_id = str(row["sample_id"]).strip()
        dataset = str(row["dataset"]).strip().lower()
        lmk_path = (ROOT / str(row["path_landmarks"])).resolve()

        if dataset not in ("freihand", "hagrid"):
            continue

        summary[dataset]["total"] += 1

        image_path: Path | None = None
        if dataset == "freihand":
            idx = sample_idx_from_freihand(sample_id)
            if idx is not None:
                image_path = args.freihand_image_dir / f"{idx:08d}.jpg"
        else:
            image_path = find_hagrid_image(sample_id, hagrid_roots)

        if image_path is None or not image_path.exists():
            summary[dataset]["missing_image"] += 1
            debug_rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "mst_level": "",
                    "status": "missing_image",
                    "image_path": "" if image_path is None else str(image_path),
                    "landmark_path": str(lmk_path),
                }
            )
            continue

        summary[dataset]["with_image"] += 1

        try:
            img = load_image_rgb(image_path)
            if img is None:
                raise RuntimeError("image_load_failed")
            if not lmk_path.exists():
                raise RuntimeError("missing_landmark")

            lmk = np.load(lmk_path).astype(np.float32)
            if lmk.shape != (21, 3):
                raise RuntimeError(f"invalid_landmark_shape:{lmk.shape}")

            h, w = img.shape[:2]
            if dataset == "freihand":
                idx = sample_idx_from_freihand(sample_id)
                if idx is None or idx >= len(k_mats):
                    raise RuntimeError("missing_camera_matrix")
                coords_px = project_freihand_xyz_to_pixels(lmk, k_mats[idx])
            else:
                coords_px = map_xy_to_pixels(lmk[:, :2], w=w, h=h)

            skin_rgb = estimate_skin_rgb(img, coords_px)
            if skin_rgb is None:
                raise RuntimeError("empty_skin_sample")

            mst = classify_mst_from_rgb(skin_rgb)
            mst_rows.append({"sample_id": sample_id, "mst_level": mst})
            debug_rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "mst_level": mst,
                    "status": "ok",
                    "image_path": str(image_path),
                    "landmark_path": str(lmk_path),
                }
            )
            summary[dataset]["classified"] += 1

        except Exception as exc:
            summary[dataset]["errors"] += 1
            debug_rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "mst_level": "",
                    "status": f"error:{exc}",
                    "image_path": str(image_path),
                    "landmark_path": str(lmk_path),
                }
            )

        if (i + 1) % 1000 == 0:
            print(
                f"{i + 1}/{len(manifest)} | "
                f"F ok={summary['freihand']['classified']} miss={summary['freihand']['missing_image']} err={summary['freihand']['errors']} | "
                f"H ok={summary['hagrid']['classified']} miss={summary['hagrid']['missing_image']} err={summary['hagrid']['errors']}"
            )

    # Escribir CSV para balanceador.
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["sample_id", "mst_level"])
        wr.writeheader()
        wr.writerows(mst_rows)

    # Debug y resumen.
    args.output_debug_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_debug_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=["sample_id", "dataset", "mst_level", "status", "image_path", "landmark_path"],
        )
        wr.writeheader()
        wr.writerows(debug_rows)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"CSV MST: {args.output_csv}")
    print(f"Debug: {args.output_debug_csv}")
    print(f"Resumen: {args.output_summary}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
