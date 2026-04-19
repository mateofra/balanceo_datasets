# src/pipeline

Scripts de orquestacion y preparacion de datos para entrenamiento.

## Contenido

- `unificar_datasets_y_split.py`: unifica datasets y crea splits.
- `preparar_entrenamiento.py`: prepara secuencias/manifiestos para training a partir de `output/train_manifest_stgcn.csv` por defecto.
- `preparar_y_entrenar.py`: flujo completo de preparacion y entrenamiento.
- `generar_secuencias_hagrid_nuevo.py`: genera secuencias ST-GCN para HaGRID nuevo con movimiento temporal suave por frame.
- `generate_landmarks_simple.py`: generador simplificado de landmarks sinteticos.
- `extraer_landmarks_hagrid_nuevo.py`: extraccion de landmarks con MediaPipe.
- `temporal_sequence_utils.py`: utilidad compartida para secuencias temporales suaves y validacion.

## Uso

```bash
uv run python src/pipeline/unificar_datasets_y_split.py
```
