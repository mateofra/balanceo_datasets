# src

Codigo fuente organizado por dominio.

## Estructura

- `balancer/`: balanceo FreiHAND + HaGRID y graficos asociados.
- `classification/`: clasificacion de tonos MST con MediaPipe.
- `preprocessing/`: conversion y procesamiento de landmarks.
- `stgcn/`: grafo anatomico, modelo ST-GCN, dataloader y ejemplo de entrenamiento.

## Uso

Ejecuta scripts con `uv run python ...` desde la raiz del repositorio.

### Balanceo

```bash
uv run python src/balancer/balancear_freihand_hagrid.py \
  --output-csv output/train_manifest_balanceado_freihand_hagrid.csv \
  --output-stgcn-manifest-csv output/train_manifest_stgcn.csv \
  --landmarks-root-dir data/processed/landmarks
```

### Clasificacion MST

```bash
uv run python src/classification/clasificar_mst_mediapipe.py imagen.jpg --model-path models/hand_landmarker.task
```

### Preprocesamiento

```bash
uv run python src/preprocessing/procesar_landmarks_freihand.py
uv run python src/preprocessing/procesar_landmarks_hagrid_mediapipe.py --images-dir data/raw/images
uv run python src/preprocessing/procesar_landmarks_hagrid_annotations.py
```

### ST-GCN

```bash
uv run python src/stgcn/train_stgcn_example.py
```

## Pipeline completo

Ver [PIPELINE_MST_STGCN.md](../PIPELINE_MST_STGCN.md) para arquitectura end-to-end.
