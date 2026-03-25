# balanceo_datasets

Repositorio para construir conjuntos de entrenamiento balanceados entre FreiHAND y HaGRID, con foco en reducir sesgo por distribucion de tonos MST durante entrenamiento.

## Objetivo

- Balancear la mezcla FreiHAND + HaGRID en entrenamiento por cuotas reproducibles.
- Mantener validacion y test sin rebalance artificial.
- Permitir activar balance por bloques MST (claro/medio/oscuro) cuando exista CSV de auditoria previa.

## Requisitos

- Python 3.13+
- uv

Instalacion:

```powershell
uv sync
```

## Balanceo FreiHAND + HaGRID

Script principal: [src/balancear_freihand_hagrid.py](src/balancear_freihand_hagrid.py)

Entrenamiento balanceado por fuente (50/50):

```powershell
uv run python src/balancear_freihand_hagrid.py \
	--freihand-training-xyz datasets/training_xyz.json \
	--hagrid-annotations-dir datasets/ann_subsample \
	--target-size 20000 \
	--hagrid-ratio 0.5 \
	--extreme-mst-levels 1 2 3 10 \
	--extreme-factor 2.0 \
	--seed 42 \
	--output-csv csv/train_manifest_balanceado_freihand_hagrid.csv \
	--output-summary csv/resumen_balanceo_freihand_hagrid.json
```

Con MST de auditoria previa (si ya tienes un CSV con `sample_id`/`image_id` + `mst`):

```powershell
uv run python src/balancear_freihand_hagrid.py \
	--freihand-training-xyz datasets/training_xyz.json \
	--hagrid-annotations-dir datasets/ann_subsample \
	--mst-csv csv/auditoria_mst.csv \
	--target-size 20000 \
	--hagrid-ratio 0.5 \
	--extreme-mst-levels 1 2 3 10 \
	--extreme-factor 2.0 \
	--dark-jitter-factor 0.5 \
	--output-tone-sets-dir csv/sets_tonos_train \
	--seed 42
```

Cuando se provee `--mst-csv`, el script intenta distribuir entrenamiento por bloques:

- claro: MST 1-4
- medio: MST 5-7
- oscuro: MST 8-10

Para asegurar clasificacion completa, por defecto el script imputa MST faltante y elimina el bloque `sin_mst`:

- `--impute-missing-mst` (por defecto activado)
- `--no-impute-missing-mst` (si quieres desactivarlo)

Ademas, si hay MST disponible, aplica oversampling por peso para niveles extremos configurables con:

- `--extreme-mst-levels` (por defecto 1 2 3 10)
- `--extreme-factor` (por defecto 2.0)

Y opcion de augmentacion cromatica controlada (representada como replicacion virtual de candidatos MST 8-9):

- `--dark-jitter-factor` (por defecto 0.0, desactivado)

Si indicas `--output-tone-sets-dir` y existe MST, exporta 3 sets de entrenamiento:

- `train_set_claro.csv` (MST 1-4)
- `train_set_medio.csv` (MST 5-7)
- `train_set_oscuro.csv` (MST 8-10)

## Salidas

- `csv/train_manifest_balanceado_freihand_hagrid.csv`: manifiesto de entrenamiento balanceado.
- `csv/resumen_balanceo_freihand_hagrid.json`: resumen de composicion por fuente, gesto y bloque MST.
- `csv/sets_tonos_train/*.csv`: sets por tono (si activas `--output-tone-sets-dir` y hay MST).

El manifiesto incluye columnas extra de trazabilidad:

- `sampling_weight`: peso usado por regla de oversampling MST extremo.
- `augmentation_hint`: marca `color_jitter_dark_candidate` para muestras MST 8-9.
- `mst_origin`: `original` o `imputed`, para distinguir etiquetas de auditoria vs imputacion.

## Pruebas

```powershell
uv run python -m unittest discover -s tests
```

## Graficos de validacion del balanceo

Generar graficos a partir del manifiesto y resumen:

```powershell
uv run python src/generar_graficos_balanceo.py \
	--manifest-csv csv/train_manifest_balanceado_freihand_hagrid_smoke2.csv \
	--summary-json csv/resumen_balanceo_freihand_hagrid_smoke2.json \
	--output-dir graficos/balanceo_smoke2
```

Se generan PNG para:

- composicion por fuente
- composicion por bloques MST
- niveles MST (si hay dato)
- top de gestos

Tambien se genera un reporte en `reporte_graficos_balanceo.md` con conteos de respaldo.

## Nota sobre MST

En los JSON crudos actuales de FreiHAND/HaGRID no aparece un campo MST explicito. Por eso el balance por bloques MST depende de un CSV externo de auditoria previa.

Si faltan etiquetas para algunos objetos, el pipeline las imputa en entrenamiento para que todos queden clasificados en claro/medio/oscuro.
