"""
Procesa landmarks de HaGRID usando MediaPipe Hand Landmarker.

Pipeline:
1. Cargar imágenes de HaGRID desde data/raw/images/
2. Para cada imagen, detectar landmarks con MediaPipe
3. Guardar como data/processed/landmarks/hagrid_gesture_imageid.npy (21x3)
4. Generar mapeo sample_id → path_landmarks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HaGRIDLandmarkProcessor:
    """Extrae landmarks de HaGRID usando MediaPipe Hand Landmarker."""

    def __init__(self, model_path: str = "models/hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def extract_landmarks(self, image_path: Path) -> np.ndarray | None:
        """Extrae landmarks (21x3) de una imagen."""
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            return None

        hand_landmarks = detection_result.hand_landmarks[0]
        landmarks_array = np.zeros((21, 3), dtype=np.float32)

        for idx, landmark in enumerate(hand_landmarks):
            landmarks_array[idx, 0] = landmark.x  # x normalizado [0, 1]
            landmarks_array[idx, 1] = landmark.y  # y normalizado [0, 1]
            landmarks_array[idx, 2] = landmark.z  # z normalizado [0, 1]

        return landmarks_array


def process_hagrid_images(
    images_dir: Path,
    output_dir: Path,
    model_path: str = "models/hand_landmarker.task",
    extensions: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, str]:
    """Procesa imágenes de HaGRID y extrae landmarks."""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]

    if not images_dir.exists():
        raise FileNotFoundError(f"No existe directorio de imágenes: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processor = HaGRIDLandmarkProcessor(model_path=model_path)

    sample_mapping = {}
    processed = 0
    failed = 0

    # Procesar imágenes recursivas
    image_files = []
    for ext in extensions:
        image_files.extend(images_dir.glob(f"**/*{ext}"))
        image_files.extend(images_dir.glob(f"**/*{ext.upper()}"))

    if not image_files:
        if verbose:
            print(f"⚠ No se encontraron imágenes en {images_dir}")
        return sample_mapping

    for idx, image_path in enumerate(image_files):
        try:
            # Generar sample_id desde estructura del archivo
            # Ej: gesture/image.jpg → hagrid_gesture_image
            relative = image_path.relative_to(images_dir)
            parts = relative.parts
            gesture = parts[0] if len(parts) > 1 else "unknown"
            filename = relative.stem

            sample_id = f"hagrid_{gesture}_{filename}"

            landmarks = processor.extract_landmarks(image_path)
            if landmarks is None:
                failed += 1
                if verbose and (failed % 100 == 0):
                    print(f"  ⚠ {failed} imágenes sin detección de mano")
                continue

            # Guardar .npy
            output_file = output_dir / f"{sample_id}.npy"
            np.save(output_file, landmarks)

            # Mapeo
            sample_mapping[sample_id] = str(output_file.relative_to(Path.cwd()))
            processed += 1

            if verbose and (processed + 1) % 100 == 0:
                print(f"  Procesadas {processed} imágenes HaGRID")

        except Exception as exc:
            if verbose:
                print(f"  Error procesando {image_path}: {exc}")
            failed += 1

    if verbose:
        print(f"✓ {processed} landmarks generados, {failed} fallos")
        print(f"  Guardar en {output_dir}")

    return sample_mapping


def save_mapping_json(mapping: dict[str, str], output_path: Path) -> None:
    """Guarda mapeo sample_id → path_landmarks como JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=True, indent=2)
    print(f"✓ Mapeo guardado: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Procesa imágenes HaGRID → landmarks .npy con MediaPipe"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/raw/images"),
        help="Directorio con imágenes de HaGRID (estructura: gesture/image.jpg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/landmarks"),
        help="Directorio de salida para archivos .npy",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/hand_landmarker.task",
        help="Ruta al modelo hand_landmarker.task",
    )
    parser.add_argument(
        "--output-mapping",
        type=Path,
        default=Path("csv/hagrid_landmarks_mapping.json"),
        help="JSON de mapeo sample_id → path_landmarks (opcional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Mostrar progreso",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mapping = process_hagrid_images(
        args.images_dir,
        args.output_dir,
        model_path=args.model_path,
        verbose=args.verbose,
    )

    if mapping:
        save_mapping_json(mapping, args.output_mapping)
    else:
        print("⚠ No se generó ningún landmark. ¿Verifica que las imágenes existan?")


if __name__ == "__main__":
    main()
