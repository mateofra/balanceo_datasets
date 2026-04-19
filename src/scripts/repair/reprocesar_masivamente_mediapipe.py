"""
Reprocesamiento Masivo de FreiHAND utilizando MediaPipe.

Debido a que FreiHAND entrega sus landmarks en coordenadas métricas globales (X,Y,Z en metros),
la proyección plana 2D colapsa. Este script lee las imágenes RGB crudas de FreiHAND
y utiliza MediaPipe Hand Landmarker para forzar la estandarización [0, 1] idéntica a HaGRID.
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


class MediaPipeProcessor:
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
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            return None

        hand_landmarks = detection_result.hand_landmarks[0]
        landmarks_array = np.zeros((21, 3), dtype=np.float32)

        for idx, landmark in enumerate(hand_landmarks):
            landmarks_array[idx, 0] = landmark.x 
            landmarks_array[idx, 1] = landmark.y 
            landmarks_array[idx, 2] = landmark.z 

        return landmarks_array


def process_freihand_mediapipe(
    rgb_dir: Path,
    output_dir: Path,
    mappings_out: Path,
    model_path: str = "models/hand_landmarker.task",
    verbose: bool = True
) -> None:
    if not rgb_dir.exists():
        raise FileNotFoundError(f"Directorio FreiHAND no existe: {rgb_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processor = MediaPipeProcessor(model_path=model_path)

    sample_mapping = {}
    processed = 0
    failed = 0

    image_files = sorted(list(rgb_dir.glob("*.jpg")))
    total = len(image_files)

    if total == 0:
        print("No se encontraron imágenes en el directorio.")
        return

    print(f"Iniciando inferencia MediaPipe sobre {total} imágenes de FreiHAND...")

    for image_path in image_files:
        numeric_id = image_path.stem # "00001234"
        sample_id = f"freihand_{numeric_id}"
        
        landmarks = processor.extract_landmarks(image_path)
        if landmarks is None:
            failed += 1
            continue

        output_file = output_dir / f"{numeric_id}.npy"
        np.save(output_file, landmarks)

        try:
            rel_path = output_file.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            rel_path = output_file

        sample_mapping[sample_id] = str(rel_path).replace("\\", "/")
        processed += 1

        if verbose and (processed + failed) % 1000 == 0:
            print(f"[{processed+failed}/{total}] Exitos: {processed} | Fallos: {failed}")

    print(f"\nFinalizado. Exitosos: {processed}, Fallidos: {failed}")
    
    mappings_out.parent.mkdir(parents=True, exist_ok=True)
    with mappings_out.open("w", encoding="utf-8") as f:
        json.dump(sample_mapping, f, ensure_ascii=True, indent=2)
    print(f"Mapeo guardado en {mappings_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Reprocesa FreiHAND con MediaPipe")
    parser.add_argument(
        "--rgb-dir", type=Path, default=Path("datasets/FreiHAND_pub_v2/training/rgb")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/landmarks/freihand")
    )
    parser.add_argument(
        "--mappings-out", type=Path, default=Path("csv/freihand_landmarks_mapping_mediapipe.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_freihand_mediapipe(
        args.rgb_dir,
        args.output_dir,
        args.mappings_out
    )
