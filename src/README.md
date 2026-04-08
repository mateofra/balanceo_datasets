# src

Contiene scripts ejecutables para procesar landmarks, clasificar tonos de piel y balancear datasets.

## Uso

Ejecuta los scripts con `uv run python ...` desde la raiz del repositorio.

### Clasificación de Tonos MST

- `clasificar_mst_mediapipe.py`: Clasifica tonos de piel (MST 1-10) en imágenes de manos usando MediaPipe Hand Landmarker y análisis de color en espacio LAB.
  ```bash
  uv run python src/clasificar_mst_mediapipe.py imagen.jpg --model-path models/hand_landmarker.task
  ```
  Salida: `{"mst_level": 1-10, "hex_reference": "#...", "rgb_detected_median": [...], ...}`

### Procesamiento de Landmarks

- `procesar_landmarks_freihand.py`: Convierte `datasets/training_xyz.json` → archivos `.npy` individuales en `data/processed/landmarks/`.
  ```bash
  uv run python src/procesar_landmarks_freihand.py
  ```
  Output: 32,560 archivos `.npy` (21×3 coordinates por muestra)

- `procesar_landmarks_hagrid_mediapipe.py`: Extrae landmarks de imágenes HaGRID usando MediaPipe Hand Landmarker.
  ```bash
  uv run python src/procesar_landmarks_hagrid_mediapipe.py --images-dir data/raw/images
  ```
  Output: Archivos `.npy` para gestos HaGRID

### Balanceo de Dataset

- `balancear_freihand_hagrid.py`: Construye manifiesto de entrenamiento balanceado entre FreiHAND y HaGRID.
  - Soporta integración de MST desde CSV externo (via `--mst-csv`).
  - Oversampling de extremos MST (1,2,3,10), augmentación virtual para MST 8-9.
  - Exportación de manifiesto ST-GCN con rutas a landmarks.
  ```bash
  uv run python src/balancear_freihand_hagrid.py \
    --output-csv output/train_manifest_balanceado_freihand_hagrid.csv \
    --output-stgcn-manifest-csv output/train_manifest_stgcn.csv \
    --landmarks-root-dir data/processed/landmarks
  ```

- `generar_graficos_balanceo.py`: Genera gráficos PNG y reporte Markdown para validar la distribución del balanceo.
  ```bash
  uv run python src/generar_graficos_balanceo.py \
    --manifest-csv output/train_manifest_balanceado_freihand_hagrid.csv \
    --summary-json output/resumen_balanceo_freihand_hagrid.json \
    --output-dir output/
  ```

## Pipeline Completo

Ver [PIPELINE_MST_STGCN.md](../PIPELINE_MST_STGCN.md) para arquitectura end-to-end.
