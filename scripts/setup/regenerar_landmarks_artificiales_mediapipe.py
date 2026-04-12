from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.procesar_landmarks_hagrid_mediapipe import HaGRIDLandmarkProcessor


def _to_repo_relative_posix(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenera landmarks HaGRID artificiales (synthetic_gesture_mean) "
            "usando MediaPipe cuando exista imagen local."
        )
    )
    parser.add_argument(
        "--quality-json",
        type=Path,
        default=Path("csv/hagrid_landmarks_quality.json"),
        help="JSON sample_id -> landmark_quality.",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=Path("csv/hagrid_landmarks_mapping.json"),
        help="JSON sample_id -> path_landmarks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/landmarks"),
        help="Directorio base para guardar .npy regenerados.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/hand_landmarker.task"),
        help="Ruta al modelo de MediaPipe hand_landmarker.task.",
    )
    parser.add_argument(
        "--images-roots",
        nargs="+",
        type=Path,
        default=[
            Path("datasets/hagrid_images"),
            Path("data/raw/images"),
            Path("datasets/hagrid_dataset"),
        ],
        help="Raices donde buscar imagenes HaGRID.",
    )
    parser.add_argument(
        "--write-summary",
        type=Path,
        default=Path("output/regeneracion_landmarks_artificiales_mediapipe.json"),
        help="JSON de resumen de ejecucion.",
    )
    return parser.parse_args()


def _infer_gesture_from_mapping(mapping_path: str | None) -> str:
    if not mapping_path:
        return "unknown"
    parts = Path(mapping_path).parts
    if "hagrid" in parts:
        idx = parts.index("hagrid")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def _build_image_index(roots: list[Path]) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    index: dict[str, Path] = {}

    for root in roots:
        root_abs = (ROOT / root).resolve() if not root.is_absolute() else root.resolve()
        if not root_abs.exists():
            continue

        for img_path in root_abs.rglob("*"):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in exts:
                continue
            sid = img_path.stem.strip().lower()
            if sid and sid not in index:
                index[sid] = img_path

    return index


def main() -> int:
    args = _parse_args()

    quality_path = (ROOT / args.quality_json).resolve()
    mapping_path = (ROOT / args.mapping_json).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    model_path = (ROOT / args.model_path).resolve()
    summary_path = (ROOT / args.write_summary).resolve()

    if not quality_path.exists():
        raise FileNotFoundError(f"No existe quality JSON: {quality_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"No existe mapping JSON: {mapping_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo MediaPipe: {model_path}")

    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    synthetic_ids = [sid for sid, q in quality.items() if q == "synthetic_gesture_mean"]

    image_index = _build_image_index(args.images_roots)
    detector = HaGRIDLandmarkProcessor(model_path=str(model_path))

    regenerated = 0
    no_image = 0
    no_detection = 0
    errors = 0

    for sid in synthetic_ids:
        sid_norm = str(sid).strip().lower()
        image_path = image_index.get(sid_norm)
        if image_path is None:
            no_image += 1
            continue

        try:
            landmarks = detector.extract_landmarks(image_path)
            if landmarks is None or landmarks.shape != (21, 3):
                no_detection += 1
                continue

            gesture = _infer_gesture_from_mapping(mapping.get(sid_norm))
            out_path = output_dir / "hagrid" / gesture / f"{sid_norm}.npy"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, landmarks.astype(np.float32))

            mapping[sid_norm] = _to_repo_relative_posix(out_path)
            quality[sid_norm] = "mediapipe_detected"
            regenerated += 1
        except Exception:
            errors += 1

    quality_path.write_text(json.dumps(quality, ensure_ascii=True, indent=2), encoding="utf-8")
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=True, indent=2), encoding="utf-8")

    summary = {
        "synthetic_candidates": len(synthetic_ids),
        "regenerated_with_mediapipe": regenerated,
        "without_local_image": no_image,
        "without_hand_detection": no_detection,
        "errors": errors,
        "remaining_synthetic": sum(1 for v in quality.values() if v == "synthetic_gesture_mean"),
        "quality_json": _to_repo_relative_posix(quality_path),
        "mapping_json": _to_repo_relative_posix(mapping_path),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print("Regeneracion de landmarks artificiales completada:")
    for k, v in summary.items():
        print(f"- {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
