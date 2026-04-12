from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classification.clasificar_mst_mediapipe import MSTClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera CSV de MST real por imagen para FreiHAND y HaGRID (si hay imagenes)."
        )
    )
    parser.add_argument(
        "--freihand-rgb-dir",
        type=Path,
        default=Path("datasets/FreiHAND_pub_v2/training/rgb"),
        help="Directorio de imagenes FreiHAND RGB.",
    )
    parser.add_argument(
        "--freihand-limit",
        type=int,
        default=32560,
        help=(
            "Cantidad de sample_id FreiHAND a procesar (0..limit-1). "
            "Default 32560 para alinear con training_xyz."
        ),
    )
    parser.add_argument(
        "--hagrid-image-roots",
        nargs="+",
        default=["datasets/hagrid_dataset", "data/raw/images"],
        help="Directorios donde buscar imagenes HaGRID por UUID.",
    )
    parser.add_argument(
        "--hagrid-annotations-dir",
        type=Path,
        default=Path("datasets/ann_subsample"),
        help="Anotaciones HaGRID para obtener lista de sample_id.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/hand_landmarker.task",
        help="Ruta al modelo MediaPipe hand_landmarker.task",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("csv/mst_real_dataset.csv"),
        help="CSV de salida con sample_id,mst_level,dataset,image_path,status",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("output/mst_real_summary.json"),
        help="Resumen JSON de cobertura y errores",
    )
    parser.add_argument(
        "--max-hagrid",
        type=int,
        default=0,
        help=(
            "Limite opcional de muestras HaGRID a procesar (0=sin limite)."
        ),
    )
    return parser.parse_args()


def _collect_hagrid_ids(annotations_dir: Path) -> list[str]:
    ids: list[str] = []
    if not annotations_dir.exists():
        return ids

    for ann_path in sorted(annotations_dir.glob("*.json")):
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for image_id in payload.keys():
            sid = str(image_id).strip().lower()
            if sid:
                ids.append(sid)

    # Unicos preservando orden.
    seen = set()
    uniq: list[str] = []
    for sid in ids:
        if sid in seen:
            continue
        seen.add(sid)
        uniq.append(sid)
    return uniq


def _find_hagrid_image(sample_id: str, roots: list[Path]) -> Path | None:
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

    classifier = MSTClassifier(model_path=args.model_path)

    output_csv = args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "freihand": {"total": 0, "with_image": 0, "classified": 0, "errors": 0},
        "hagrid": {"total": 0, "with_image": 0, "classified": 0, "errors": 0},
    }

    rows: list[dict[str, object]] = []

    # FreiHAND: alinear sample_id con indices de training_xyz (0..freihand_limit-1)
    print("Procesando FreiHAND...")
    for idx in range(args.freihand_limit):
        sid = f"freihand_{idx:08d}"
        img_path = args.freihand_rgb_dir / f"{idx:08d}.jpg"
        summary["freihand"]["total"] += 1

        if not img_path.exists():
            rows.append(
                {
                    "sample_id": sid,
                    "dataset": "freihand",
                    "mst_level": "",
                    "image_path": str(img_path),
                    "status": "missing_image",
                }
            )
            continue

        summary["freihand"]["with_image"] += 1
        result = classifier.classify(str(img_path))
        if "mst_level" in result:
            rows.append(
                {
                    "sample_id": sid,
                    "dataset": "freihand",
                    "mst_level": int(result["mst_level"]),
                    "image_path": str(img_path),
                    "status": "ok",
                }
            )
            summary["freihand"]["classified"] += 1
        else:
            rows.append(
                {
                    "sample_id": sid,
                    "dataset": "freihand",
                    "mst_level": "",
                    "image_path": str(img_path),
                    "status": f"error:{result.get('error', 'unknown')}",
                }
            )
            summary["freihand"]["errors"] += 1

        if (idx + 1) % 1000 == 0:
            print(
                f"  FreiHAND {idx + 1}/{args.freihand_limit} | "
                f"ok={summary['freihand']['classified']} err={summary['freihand']['errors']}"
            )

    print("Procesando HaGRID...")
    hagrid_roots = [Path(p) for p in args.hagrid_image_roots]
    hagrid_ids = _collect_hagrid_ids(args.hagrid_annotations_dir)
    if args.max_hagrid > 0:
        hagrid_ids = hagrid_ids[: args.max_hagrid]

    summary["hagrid"]["total"] = len(hagrid_ids)

    for i, sid in enumerate(hagrid_ids):
        img_path = _find_hagrid_image(sid, hagrid_roots)
        if img_path is None:
            rows.append(
                {
                    "sample_id": sid,
                    "dataset": "hagrid",
                    "mst_level": "",
                    "image_path": "",
                    "status": "missing_image",
                }
            )
            continue

        summary["hagrid"]["with_image"] += 1
        result = classifier.classify(str(img_path))
        if "mst_level" in result:
            rows.append(
                {
                    "sample_id": sid,
                    "dataset": "hagrid",
                    "mst_level": int(result["mst_level"]),
                    "image_path": str(img_path),
                    "status": "ok",
                }
            )
            summary["hagrid"]["classified"] += 1
        else:
            rows.append(
                {
                    "sample_id": sid,
                    "dataset": "hagrid",
                    "mst_level": "",
                    "image_path": str(img_path),
                    "status": f"error:{result.get('error', 'unknown')}",
                }
            )
            summary["hagrid"]["errors"] += 1

        if (i + 1) % 1000 == 0:
            print(
                f"  HaGRID {i + 1}/{len(hagrid_ids)} | "
                f"ok={summary['hagrid']['classified']} err={summary['hagrid']['errors']}"
            )

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "dataset", "mst_level", "image_path", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    output_summary = args.output_summary
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"CSV MST real: {output_csv}")
    print(f"Resumen: {output_summary}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
