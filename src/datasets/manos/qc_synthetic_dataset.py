from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classification.clasificar_mst_mediapipe import MSTClassifier


def _parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def worker_init(model_path: str):
    """Inicializa el clasificador una sola vez por proceso trabajador."""
    global classifier
    classifier = MSTClassifier(model_path=model_path)


def process_row_task(args_tuple):
    """Tarea individual para clasificar una imagen."""
    row, tolerance = args_tuple
    
    target_level = _parse_int(str(row.get("mst_level", "")))
    if target_level is None:
        return "invalid_target", row, None

    image_path = Path(str(row.get("path", "")).strip())
    if not image_path.is_file():
        return "missing_image", row, None

    try:
        result = classifier.classify(str(image_path))
    except Exception as e:
        return "error", row, {"error": str(e)}

    if "mst_level" not in result:
        return "classifier_error", row, result

    pred_level = int(result["mst_level"])
    delta = abs(pred_level - target_level)
    accepted = delta <= max(0, int(tolerance))
    
    return "ok", row, {
        "pred_mst": pred_level,
        "delta": delta,
        "accepted": accepted,
        "reason": "ok" if accepted else "outside_tolerance"
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC MST de imagenes sinteticas (PARALELIZADO)")
    parser.add_argument("--generated-manifest", type=Path, required=True)
    parser.add_argument("--accepted-manifest", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--model-path", type=str, default="models/hand_landmarker.task")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.generated_manifest.is_file():
        raise FileNotFoundError(f"No existe generated-manifest: {args.generated_manifest}")

    print(f"Iniciando QC paralelo con {args.workers} workers...")
    
    rows_to_process = []
    with args.generated_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = str(row.get("status", "ok")).strip().lower()
            if not status or status == "ok":
                rows_to_process.append(row)

    total_to_eval = len(rows_to_process)
    print(f"Cargadas {total_to_eval} filas para evaluar.")

    accepted_rows = []
    report_rows = []
    summary = {
        "total_rows": total_to_eval,
        "evaluated": 0,
        "accepted": 0,
        "rejected": 0,
        "missing_image": 0,
        "classifier_errors": 0,
        "invalid_target": 0,
        "skipped_non_ok_status": 0,
    }

    # Ejecucion en paralelo
    with ProcessPoolExecutor(
        max_workers=args.workers, 
        initializer=worker_init, 
        initargs=(args.model_path,)
    ) as executor:
        # Preparamos las tareas
        tasks = [(row, args.tolerance) for row in rows_to_process]
        
        # Procesamos con seguimiento de progreso
        for i, (status, row, result) in enumerate(executor.map(process_row_task, tasks)):
            summary["evaluated"] += 1
            
            sample_id = str(row.get("sample_id", "")).strip()
            target_mst = _parse_int(str(row.get("mst_level", "")))

            if status == "ok":
                res_data = result
                report_rows.append({
                    "sample_id": sample_id,
                    "target_mst": target_mst,
                    "pred_mst": res_data["pred_mst"],
                    "delta": res_data["delta"],
                    "accepted": res_data["accepted"],
                    "reason": res_data["reason"],
                })
                
                if res_data["accepted"]:
                    summary["accepted"] += 1
                    accepted_rows.append({
                        "sample_id": sample_id,
                        "path": str(row.get("path", "")).strip(),
                        "dataset": str(row.get("dataset", "synthetic_mst")).strip() or "synthetic_mst",
                        "mst_level": target_mst,
                        "mst_block": str(row.get("mst_block", "")).strip(),
                        "source_kind": str(row.get("source_kind", "synthetic")).strip() or "synthetic",
                        "input_csv": str(row.get("input_csv", args.generated_manifest)).strip(),
                        "qc_pred_mst": res_data["pred_mst"],
                        "qc_delta": res_data["delta"],
                    })
                else:
                    summary["rejected"] += 1
            
            elif status == "invalid_target":
                summary["invalid_target"] += 1
            elif status == "missing_image":
                summary["missing_image"] += 1
            else:
                summary["classifier_errors"] += 1
                report_rows.append({
                    "sample_id": sample_id,
                    "target_mst": target_mst,
                    "pred_mst": "",
                    "delta": "",
                    "accepted": False,
                    "reason": str(result.get("error", "unknown") if result else "unknown"),
                })

            if (i + 1) % 1000 == 0:
                print(f"  Progreso QC: {i+1}/{total_to_eval} | Aceptadas={summary['accepted']} Rechazadas={summary['rejected']}")

    # Guardar resultados
    args.accepted_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.accepted_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample_id", "path", "dataset", "mst_level", "mst_block", 
            "source_kind", "input_csv", "qc_pred_mst", "qc_delta"
        ])
        writer.writeheader()
        writer.writerows(accepted_rows)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "tolerance": int(args.tolerance), "rows": report_rows}
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[synthetic-qc] FINALIZADO. Aceptadas: {summary['accepted']} de {summary['evaluated']}")
    return 0


if __name__ == "__main__":
    # Necesario para multiprocessing en Windows/Linux
    multiprocessing.freeze_support()
    raise SystemExit(main())