# extraer_landmarks_hagrid_nuevo.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_DIR  = REPO_ROOT / 'datasets/hagrid_raw/hagrid-sample-30k-384p'
OUT_DIR  = REPO_ROOT / 'data/processed/landmarks/hagrid_nuevo'
LOG_PATH = REPO_ROOT / 'output/extraccion_hagrid_nuevo.json'
MODEL_PATH = REPO_ROOT / 'models/hand_landmarker.task'

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f'No existe modelo MediaPipe: {MODEL_PATH}')
if not IMG_DIR.exists():
    raise FileNotFoundError(f'No existe carpeta de imagenes: {IMG_DIR}')

procesadas = 0
fallidas   = 0
log        = []

imagenes = list(IMG_DIR.rglob('*.jpg')) + list(IMG_DIR.rglob('*.png'))
print(f"Imágenes a procesar: {len(imagenes)}")

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
)

with vision.HandLandmarker.create_from_options(options) as hands:
    for i, img_path in enumerate(imagenes):
        # Clase = nombre del directorio padre
        clase = img_path.parent.name
        out_clase = OUT_DIR / clase
        out_clase.mkdir(exist_ok=True)

        dst = out_clase / f"{img_path.stem}.npy"
        if dst.exists():
            procesadas += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            fallidas += 1
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = hands.detect(mp_image)

        if results.hand_landmarks:
            lm = results.hand_landmarks[0]
            coords = np.array([[p.x, p.y, p.z] for p in lm],
                              dtype=np.float32)  # (21, 3)
            np.save(str(dst), coords)
            procesadas += 1
        else:
            fallidas += 1

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(imagenes)} | OK: {procesadas} | Sin mano: {fallidas}")

print(f"\nFinalizado: {procesadas} landmarks | {fallidas} sin detección")

with open(LOG_PATH, 'w') as f:
    json.dump({'procesadas': procesadas, 'fallidas': fallidas,
               'total': len(imagenes)}, f, indent=2)
