from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


MST_RGB = {
    1: (246, 237, 228),
    2: (243, 231, 219),
    3: (247, 234, 208),
    4: (234, 218, 186),
    5: (215, 189, 150),
    6: (160, 126, 86),
    7: (130, 92, 67),
    8: (96, 65, 52),
    9: (58, 49, 42),
    10: (41, 36, 32),
}


def _parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _mst_block(mst_level: int) -> str:
    if 1 <= mst_level <= 4:
        return "claro"
    if 5 <= mst_level <= 7:
        return "medio"
    if 8 <= mst_level <= 10:
        return "oscuro"
    raise ValueError(f"Nivel MST invalido: {mst_level}")


def _recolor_image(image_bgr: np.ndarray, target_rgb: tuple[int, int, int], strength: float) -> np.ndarray:
    strength = max(0.0, min(1.0, strength))
    overlay_bgr = np.zeros_like(image_bgr, dtype=np.float32)
    overlay_bgr[:, :, 0] = target_rgb[2]
    overlay_bgr[:, :, 1] = target_rgb[1]
    overlay_bgr[:, :, 2] = target_rgb[0]

    src = image_bgr.astype(np.float32)
    out = (1.0 - strength) * src + strength * overlay_bgr
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera imagenes sinteticas desde un manifest de solicitudes")
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--output-images-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.request_manifest.is_file():
        raise FileNotFoundError(f"No existe request-manifest: {args.request_manifest}")

    args.output_images_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)

    generated_rows: list[dict[str, object]] = []
    summary = {
        "total_requests": 0,
        "generated": 0,
        "missing_source": 0,
        "read_errors": 0,
        "write_errors": 0,
        "invalid_level": 0,
    }

    with args.request_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            summary["total_requests"] += 1

            request_id = str(row.get("request_id", f"req_{index:07d}")).strip() or f"req_{index:07d}"
            source_path = Path(str(row.get("source_image_path", "")).strip())
            target_level = _parse_int(str(row.get("target_mst_level", "")))

            output_name = f"synth_{request_id}.jpg"
            output_path = args.output_images_dir / output_name

            record: dict[str, object] = {
                "sample_id": f"synth_{request_id}",
                "path": str(output_path),
                "dataset": "synthetic_mst",
                "mst_level": target_level if target_level is not None else "",
                "mst_block": _mst_block(target_level) if target_level is not None else "",
                "source_kind": "synthetic",
                "input_csv": str(args.request_manifest),
                "source_sample_id": str(row.get("source_sample_id", "")).strip(),
                "status": "",
            }

            if target_level is None or target_level not in MST_RGB:
                summary["invalid_level"] += 1
                record["status"] = "error:invalid_mst_level"
                generated_rows.append(record)
                continue

            if not source_path.is_file():
                summary["missing_source"] += 1
                record["status"] = "error:missing_source"
                generated_rows.append(record)
                continue

            source_img = cv2.imread(str(source_path))
            if source_img is None:
                summary["read_errors"] += 1
                record["status"] = "error:read_failed"
                generated_rows.append(record)
                continue

            synthetic_img = _recolor_image(source_img, MST_RGB[target_level], args.strength)
            ok = cv2.imwrite(str(output_path), synthetic_img)
            if not ok:
                summary["write_errors"] += 1
                record["status"] = "error:write_failed"
                generated_rows.append(record)
                continue

            summary["generated"] += 1
            record["status"] = "ok"
            generated_rows.append(record)

    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "path",
                "dataset",
                "mst_level",
                "mst_block",
                "source_kind",
                "input_csv",
                "source_sample_id",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(generated_rows)

    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[synthetic-generate] Manifest guardado en {args.output_manifest}")
    print(f"[synthetic-generate] Resumen guardado en {args.output_summary}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())