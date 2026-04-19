# Training

Scripts y artefactos de entrenamiento del flujo ST-GCN.

## Scripts

| Script | Uso |
|--------|-----|
| `generar_secuencias_stgcn.py` | Genera secuencias temporales desde landmarks estaticos. |
| `train_autosupervisado.py` | Pretexto de prediccion del ultimo frame; se ajusto para evitar leakage temporal. |
| `train_supervisado.py` | Clasificacion supervisada sobre HaGRID real con secuencias existentes. |
| `train_stgcn.py` | Version demo/legacy de clasificacion con el loader temporal. |
| `visualizar_pipeline.py` | Inspeccion visual de la tuberia de datos. |

## Como se elaboraron

- `generar_secuencias_stgcn.py` partio de landmarks estaticos y se le añadió ruido temporal suave.
- `train_autosupervisado.py` paso de enmascarar nodos a predecir el ultimo frame cuando se detecto que la secuencia era casi estatica.
- `train_supervisado.py` se construyo para usar solo HaGRID con `annotation_2d_projected` y secuencias existentes en disco.

## Errores y soluciones

- Se detecto leakage temporal en secuencias con frames casi identicos; se corrigio cambiando la tarea auto-supervisada.
- La version CPU-only de PyTorch impedia usar la RTX 4050; se resolvio fijando el indice CUDA en `pyproject.toml`.
- El manifiesto fijo tenia rutas inconsistentes en algunos casos; el trainer supervisado filtra los archivos inexistentes al arrancar.

## Salidas

- Historial de entrenamiento: `output/training/training_history_supervisado.json`
- Checkpoints: `output/training/best_stgcn_supervisado.pth`
- Logs historicos: `output/training_logs/`
