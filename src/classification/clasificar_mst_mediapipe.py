import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError as exc:  # pragma: no cover - dependencia opcional en runtime
    raise RuntimeError(
        "MediaPipe no esta instalado. Instala 'mediapipe' para usar este script."
    ) from exc


class MSTClassifier:
    """Clasificador de tono de piel (MST 1-10) usando MediaPipe y LAB."""

    def __init__(self, model_path: str = "models/hand_landmarker.task") -> None:
        self.mst_hex = [
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
        self.mst_labels = list(range(1, 11))
        self.mst_rgb = np.array([self._hex_to_rgb(h) for h in self.mst_hex], dtype=np.uint8)
        self.mst_lab = self._rgb_array_to_lab(self.mst_rgb)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _hex_to_rgb(hex_str: str) -> list[int]:
        value = hex_str.lstrip("#")
        return [int(value[i : i + 2], 16) for i in (0, 2, 4)]

    @staticmethod
    def _rgb_array_to_lab(rgb_array: np.ndarray) -> np.ndarray:
        rgb_img = rgb_array.reshape((-1, 1, 3))
        lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        return lab_img.reshape((-1, 3)).astype(np.float32)

    def _get_hand_mask(self, image_rgb: np.ndarray) -> np.ndarray | None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            return None

        h, w, _ = image_rgb.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        hand_landmarks = detection_result.hand_landmarks[0]
        palm_indices = [0, 1, 5, 9, 13, 17]

        points = []
        for idx in palm_indices:
            lm = hand_landmarks[idx]
            x = int(np.clip(lm.x * w, 0, w - 1))
            y = int(np.clip(lm.y * h, 0, h - 1))
            points.append([x, y])

        points_array = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(mask, points_array, 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @staticmethod
    def _euclidean_nn(query_lab: np.ndarray, refs_lab: np.ndarray) -> tuple[int, float]:
        distances = np.linalg.norm(refs_lab - query_lab[None, :], axis=1)
        idx = int(np.argmin(distances))
        return idx, float(distances[idx])

    def classify(self, image_path: str) -> dict:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return {"error": "Imagen no encontrada"}

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = self._get_hand_mask(image_rgb)
        if mask is None:
            return {"error": "No se detecto mano"}

        skin_pixels = image_rgb[mask == 255]
        if skin_pixels.size == 0:
            return {"error": "Fallo en segmentacion"}

        median_rgb = np.median(skin_pixels, axis=0).astype(np.uint8)
        median_lab = self._rgb_array_to_lab(median_rgb.reshape(1, 3))[0]

        closest_idx, dist = self._euclidean_nn(median_lab, self.mst_lab)

        return {
            "mst_level": self.mst_labels[closest_idx],
            "hex_reference": self.mst_hex[closest_idx],
            "rgb_detected_median": median_rgb.tolist(),
            "lab_detected_median": [round(float(v), 2) for v in median_lab],
            "lab_distance": round(dist, 3),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clasifica tono de mano en escala MST")
    parser.add_argument("image", help="Ruta de imagen de mano")
    parser.add_argument(
        "--model-path",
        default="models/hand_landmarker.task",
        help="Ruta al modelo hand_landmarker.task",
    )
    args = parser.parse_args()

    classifier = MSTClassifier(model_path=args.model_path)
    result = classifier.classify(args.image)
    print(result)


if __name__ == "__main__":
    main()
