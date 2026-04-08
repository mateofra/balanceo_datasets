# 🔄 Módulo de Preprocesamiento

Procesa landmarks de diferentes fuentes (FreiHAND, HaGRID) y los normaliza.

## Archivos

| Archivo | Propósito |
|---------|-----------|
| `landmarks_normalizer.py` | Normaliza landmarks a escala estándar |
| `procesar_landmarks_freihand.py` | Extrae y procesa landmarks de FreiHAND |
| `procesar_landmarks_hagrid_mediapipe.py` | Procesa HaGRID usando MediaPipe |

## Funcionalidad

### landmarks_normalizer.py
- Normalización a escala [-1, 1] o [0, 1]
- Rotación invariante
- Escalado por tamaño de mano
- Almacenamiento eficiente en .npy

### procesar_landmarks_freihand.py
- Extrae 21 joints de formato FreiHAND
- Valida integridad de datos
- Genera .npy balanceados

### procesar_landmarks_hagrid_mediapipe.py
- Detecta manos con MediaPipe
- Extrae 21 landmarks
- Maneja múltiples manos en frame
- Genera secuencias temporales

## Uso

```python
from src.preprocessing import landmarks_normalizer, procesar_landmarks_freihand

# Normalizar landmarks
normalized = landmarks_normalizer.normalize(
    landmarks,
    method='zscore',
    scale=(-1, 1)
)

# Procesar FreiHAND
freihand_data = procesar_landmarks_freihand.process(
    json_path='data/freihand.json',
    output_path='data/processed/landmarks/'
)
```

## Output

- Archivos `.npy` en `data/processed/landmarks/`
- Manifiestos CSV con referencias
