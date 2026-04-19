# Manos Datasets

Scripts canonicos para el flujo de sintesis, QC y balanceo MST por bloques.

## Contenido

- `build_synthetic_manifest.py`: construye solicitudes sinteticas a partir de CSV reales.
- `generate_synthetic_skin_tones.py`: genera imagenes sinteticas por recoloracion controlada.
- `qc_synthetic_dataset.py`: filtra sinteticos aceptados por QC MST.
- `build_balanced_block_manifest.py`: balancea el conjunto final por bloques `claro`, `medio` y `oscuro`.
- `build_mano_samples_manifest.py`: registra en CSV la carpeta `datasets/synthetic_mst/mano_samples_balanced/` con niveles MST 1-10.

## Uso

Ejecuta los scripts desde la raiz del repositorio con `uv run python src/datasets/manos/<script>.py`.

Ver [README-Datasets.md](../../../README-Datasets.md) para el orden exacto del flujo.