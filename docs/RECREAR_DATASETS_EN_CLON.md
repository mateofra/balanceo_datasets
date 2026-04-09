# Recrear Datasets En Un Clon Nuevo

Este documento describe como dejar operativo un clon nuevo cuando `datasets/`, `data/` y `stgcn/data/` no vienen en git (por `.gitignore`).

## Resultado De La Revision

- No hay un script unico que descargue automaticamente FreiHAND y HaGRID de extremo a extremo.
- Si hay documentacion de descarga/manual en `docs/README_evaluacion_datasets.md`.
- Se agrego el script `scripts/setup/bootstrap_local_data.py` para recrear la estructura local ignorada y verificar prerequisitos.

## 1) Crear Estructura Local Ignorada

Desde la raiz del repo:

```bash
uv run python scripts/setup/bootstrap_local_data.py
```

Esto crea (si faltan):

- `datasets/` y subdirectorios esperados
- `data/raw/`, `data/processed/`
- `stgcn/data/`
- `datasets/README_LOCAL.md` (archivo local no versionado)

## 2) Descargar/Copiar Datasets Reales

Seguir los pasos de:

- `docs/README_evaluacion_datasets.md`

Para HaGRID tambien puedes usar el asistente por CLI:

```bash
uv run python scripts/setup/download_hagrid_kaggle.py
```

Ejecucion real:

```bash
uv run python scripts/setup/download_hagrid_kaggle.py --execute --prepare-ann-subsample
```

Resumen:

- FreiHAND: descargar desde la web oficial y colocar archivos en `datasets/FreiHAND_pub_v2/`.
- HaGRID: descargar desde Kaggle (manual o CLI) y copiar anotaciones en `datasets/ann_subsample/` o `datasets/hagrid_annotations/train/` segun flujo.

## 3) Verificar Prerequisitos Minimos

```bash
uv run python scripts/setup/bootstrap_local_data.py --verify-only
```

El script verifica presencia de:

- `datasets/training_xyz.json`
- `datasets/training_K.json`
- `datasets/training_scale.json`

Si faltan, reporta `MISS` y codigo de salida distinto de cero.

## 4) Validar Pipeline Basico

```bash
uv run python main.py --help
uv run python main.py test-forward
```

Con esto confirmas que el repo y la parte de modelo estan operativos; luego puedes ejecutar balanceo/training con datos reales.
