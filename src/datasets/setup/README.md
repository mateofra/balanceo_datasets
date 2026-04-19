# Setup Datasets

Scripts canonicos para descargar y preparar datos de HaGRID y FreiHAND.

## Contenido

- `download_hagrid_kaggle.py`: descarga HaGRID con Kaggle y prepara `datasets/ann_subsample/`.
- `generar_mst_real_datasets.py`: clasifica FreiHAND y HaGRID reales en MST y genera el CSV base.

## Uso

Ejecuta los scripts desde la raiz del repositorio con `uv run python src/datasets/setup/<script>.py`.

Ver [README-Datasets.md](../../../README-Datasets.md) para el flujo completo.