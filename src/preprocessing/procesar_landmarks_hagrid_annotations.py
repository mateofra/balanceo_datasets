"""Genera landmarks de HaGRID desde anotaciones JSON.

Este script existe para desbloquear el pipeline cuando no hay imagenes crudas
pero si hay anotaciones con landmarks 2D en datasets/ann_subsample.

Salida esperada por el balanceador ST-GCN:
- data/processed/landmarks/hagrid/<gesture>/<image_id>.npy  (shape 21x3)
- csv/hagrid_landmarks_mapping.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _to_repo_relative_posix(path: Path) -> str:
    """Convierte a ruta relativa al repo con separador POSIX."""
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def _landmark_xy_from_payload(payload: dict) -> np.ndarray | None:
    """Extrae landmarks XY (21x2) desde payload HaGRID, si existen."""
    landmarks = payload.get("landmarks", [])
    if not landmarks or not isinstance(landmarks, list):
        return None

    first = landmarks[0] if landmarks else None
    if not first or not isinstance(first, list):
        return None

    if len(first) < 21:
        return None

    pts = np.asarray(first[:21], dtype=np.float32)
    if pts.shape != (21, 2):
        return None
    return pts


def _xy_to_xyz(xy: np.ndarray) -> np.ndarray:
    """Convierte landmarks XY a XYZ con z=0 para compatibilidad (21x3)."""
    xyz = np.zeros((21, 3), dtype=np.float32)
    xyz[:, :2] = xy
    return xyz


def _load_annotations(annotations_dir: Path, gestures: list[str]) -> dict[str, dict]:
    """Carga anotaciones por gesto."""
    data: dict[str, dict] = {}
    for gesture in gestures:
        ann_path = annotations_dir / f"{gesture}.json"
        if not ann_path.exists():
            continue
        with ann_path.open("r", encoding="utf-8") as f:
            data[gesture] = json.load(f)
    return data


def generate_hagrid_landmarks_from_annotations(
    annotations_dir: Path,
    output_dir: Path,
    mapping_json: Path,
    quality_json: Path,
    gestures: list[str],
) -> dict[str, int]:
    """Genera .npy para HaGRID usando landmarks de anotacion y fallbacks robustos."""
    ann_by_gesture = _load_annotations(annotations_dir, gestures)
    if not ann_by_gesture:
        raise FileNotFoundError(
            f"No se encontraron anotaciones en {annotations_dir} para gestos: {gestures}"
        )

    # Paso 1: construir promedios por gesto y global con muestras validas.
    by_gesture_valid: dict[str, list[np.ndarray]] = {g: [] for g in ann_by_gesture}
    global_valid: list[np.ndarray] = []

    for gesture, samples in ann_by_gesture.items():
        for payload in samples.values():
            xy = _landmark_xy_from_payload(payload)
            if xy is None:
                continue
            xyz = _xy_to_xyz(xy)
            by_gesture_valid[gesture].append(xyz)
            global_valid.append(xyz)

    if not global_valid:
        raise RuntimeError("No hay landmarks validos en anotaciones para construir fallbacks.")

    global_mean = np.mean(np.stack(global_valid, axis=0), axis=0).astype(np.float32)
    gesture_mean: dict[str, np.ndarray] = {}
    for gesture, arrs in by_gesture_valid.items():
        if arrs:
            gesture_mean[gesture] = np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)

    # Paso 2: exportar archivos y mapping.
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    quality_mapping: dict[str, str] = {}

    created_valid = 0
    created_fallback = 0
    total = 0

    for gesture, samples in ann_by_gesture.items():
        for image_id, payload in samples.items():
            sample_id = str(image_id).strip().lower()
            if not sample_id:
                continue

            xy = _landmark_xy_from_payload(payload)
            if xy is not None:
                xyz = _xy_to_xyz(xy)
                created_valid += 1
                quality = "annotation_2d_projected"
            else:
                xyz = gesture_mean.get(gesture, global_mean).copy()
                created_fallback += 1
                quality = "synthetic_gesture_mean"

            out_path = output_dir / "hagrid" / gesture / f"{sample_id}.npy"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, xyz)

            mapping[sample_id] = _to_repo_relative_posix(out_path)
            quality_mapping[sample_id] = quality
            total += 1

    mapping_json.parent.mkdir(parents=True, exist_ok=True)
    with mapping_json.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=True, indent=2)

    quality_json.parent.mkdir(parents=True, exist_ok=True)
    with quality_json.open("w", encoding="utf-8") as f:
        json.dump(quality_mapping, f, ensure_ascii=True, indent=2)

    return {
        "total": total,
        "valid_from_annotations": created_valid,
        "fallback_from_means": created_fallback,
        "mapping_entries": len(mapping),
        "quality_entries": len(quality_mapping),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera landmarks HaGRID (21x3) a partir de anotaciones JSON"
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("datasets/ann_subsample"),
        help="Directorio con archivos de anotaciones HaGRID por gesto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/landmarks"),
        help="Directorio base de salida para .npy",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=Path("csv/hagrid_landmarks_mapping.json"),
        help="Ruta del mapeo sample_id -> path_landmarks",
    )
    parser.add_argument(
        "--quality-json",
        type=Path,
        default=Path("csv/hagrid_landmarks_quality.json"),
        help="Ruta del mapeo sample_id -> landmark_quality",
    )
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=[
            "call",
            "dislike",
            "fist",
            "four",
            "like",
            "mute",
            "ok",
            "one",
            "palm",
            "peace",
            "peace_inverted",
            "rock",
            "stop",
            "stop_inverted",
            "three",
            "three2",
            "two_up",
            "two_up_inverted",
        ],
        help="Gestos HaGRID a procesar",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = generate_hagrid_landmarks_from_annotations(
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        mapping_json=args.mapping_json,
        quality_json=args.quality_json,
        gestures=args.gestures,
    )

    print("Landmarks HaGRID generados desde anotaciones:")
    for key, value in stats.items():
        print(f"- {key}: {value}")
    print(f"- mapping_json: {args.mapping_json}")
    print(f"- quality_json: {args.quality_json}")


if __name__ == "__main__":
    main()
