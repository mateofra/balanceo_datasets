from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classification.clasificar_mst_mediapipe import MSTClassifier


def _parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC MST de imagenes sinteticas y filtrado de aceptadas")
    parser.add_argument("--generated-manifest", type=Path, required=True)
    parser.add_argument("--accepted-manifest", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--model-path", type=str, default="models/hand_landmarker.task")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.generated_manifest.is_file():
        raise FileNotFoundError(f"No existe generated-manifest: {args.generated_manifest}")

    classifier = MSTClassifier(model_path=args.model_path)

    accepted_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    summary = {
        "total_rows": 0,
        "evaluated": 0,
        "accepted": 0,
        "rejected": 0,
        "missing_image": 0,
        "classifier_errors": 0,
        "invalid_target": 0,
        "skipped_non_ok_status": 0,
    }

    with args.generated_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            summary["total_rows"] += 1
            status = str(row.get("status", "ok")).strip().lower()
            if status and status != "ok":
                summary["skipped_non_ok_status"] += 1
                continue

            target_level = _parse_int(str(row.get("mst_level", "")))
            if target_level is None:
                summary["invalid_target"] += 1
                continue

            image_path = Path(str(row.get("path", "")).strip())
            if not image_path.is_file():
                summary["missing_image"] += 1
                continue

            summary["evaluated"] += 1
            result = classifier.classify(str(image_path))
            if "mst_level" not in result:
                summary["classifier_errors"] += 1
                report_rows.append(
                    {
                        "sample_id": str(row.get("sample_id", "")).strip(),
                        "target_mst": target_level,
                        "pred_mst": "",
                        "delta": "",
                        "accepted": False,
                        "reason": str(result.get("error", "unknown")),
                    }
                )
                continue

            pred_level = int(result["mst_level"])
            delta = abs(pred_level - target_level)
            accepted = delta <= max(0, int(args.tolerance))

            report_rows.append(
                {
                    "sample_id": str(row.get("sample_id", "")).strip(),
                    "target_mst": target_level,
                    "pred_mst": pred_level,
                    "delta": delta,
                    "accepted": accepted,
                    "reason": "ok" if accepted else "outside_tolerance",
                }
            )

            if accepted:
                summary["accepted"] += 1
                accepted_rows.append(
                    {
                        "sample_id": str(row.get("sample_id", "")).strip(),
                        "path": str(row.get("path", "")).strip(),
                        "dataset": str(row.get("dataset", "synthetic_mst")).strip() or "synthetic_mst",
                        "mst_level": target_level,
                        "mst_block": str(row.get("mst_block", "")).strip(),
                        "source_kind": str(row.get("source_kind", "synthetic")).strip() or "synthetic",
                        "input_csv": str(row.get("input_csv", args.generated_manifest)).strip(),
                        "qc_pred_mst": pred_level,
                        "qc_delta": delta,
                    }
                )
            else:
                summary["rejected"] += 1

    args.accepted_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.accepted_manifest.open("w", encoding="utf-8", newline="") as handle:
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
                "qc_pred_mst",
                "qc_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(accepted_rows)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": summary,
        "tolerance": int(args.tolerance),
        "rows": report_rows,
    }
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[synthetic-qc] Aceptadas: {summary['accepted']} de {summary['evaluated']}")
    print(f"[synthetic-qc] Manifest aceptado: {args.accepted_manifest}")
    print(f"[synthetic-qc] Reporte: {args.report_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())