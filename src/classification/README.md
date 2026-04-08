# 🎨 Módulo de Clasificación

Clasifica atributos de manos como tono de piel (MST).

## Archivos

- `clasificar_mst_mediapipe.py` - Clasificación de tonos MST usando MediaPipe

## Funcionalidad

- Detecta tono de piel en escala MST (1-10)
- Utiliza detección de mano de MediaPipe
- Categorización: claro (1-3) / medio (4-7) / oscuro (8-10)
- Retorna scores de confianza

## Uso

```python
from src.classification import clasificar_mst_mediapipe

# Clasificar tono MST en imagen
mst_level, confidence = clasificar_mst_mediapipe.classify(
    image_path='data/hand_image.jpg'
)

# Clasificar múltiples imágenes
results = clasificar_mst_mediapipe.batch_classify(
    image_paths=['img1.jpg', 'img2.jpg', 'img3.jpg']
)
```

## Output

- MST level (1-10)
- Confidence score (0-1)
- Categoria (claro/medio/oscuro)
