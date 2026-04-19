# README Datasets

Este documento deja definidos los pasos exactos para trabajar con los datasets del proyecto y el flujo de balanceo MST.

Los scripts canonicos para este flujo viven ahora en `src/datasets/`.

Entrada rápida del flujo completo:

```bash
./ejecutar_flujo_datasets.sh
```

## Variables MANO en el flujo

El script [ejecutar_flujo_datasets.sh](ejecutar_flujo_datasets.sh) admite estas variables de entorno para la parte de generacion MANO:

- `MANO_MODEL` (default: `manos/models/MANO_RIGHT.pkl`)
- `MANO_SIZE` (default: `1000`)
- `MANO_POSE_MODE` (default: `balanced`)
- `MANO_GENERATE` (default: `1`)

Comportamiento:

1. Antes de construir el manifiesto MANO, ejecuta la generacion con:

```bash
uv run python manos/src/generator.py --model "$MANO_MODEL" --size "$MANO_SIZE" --pose-mode "$MANO_POSE_MODE" --out "$MANO_SAMPLES_DIR"
```

2. Si `MANO_GENERATE != 1`, omite la generacion y usa el contenido ya existente en `MANO_SAMPLES_DIR` para construir el manifiesto.

3. Al final del script, el resumen imprime los parametros MANO usados (`model`, `size`, `pose_mode`, `output`).

Ejecutar sin regenerar MANO (usar dataset ya existente):

```bash
MANO_GENERATE=0 ./ejecutar_flujo_datasets.sh
```

Validacion aplicada al script:

```bash
bash -n ejecutar_flujo_datasets.sh
```

Resultado esperado: `bash syntax ok`.

## Estructura canonica

- `src/datasets/setup/`: descarga de HaGRID y generacion de MST reales.
- `src/datasets/manos/`: flujo de sintesis, QC y balanceo por bloques MST.

## Orden exacto del flujo

### 1. Descargar HaGRID y preparar anotaciones

Usa el wrapper canonico:

```bash
uv run python src/datasets/setup/download_hagrid_kaggle.py --execute --prepare-ann-subsample
```

Que hace:

1. Descarga el dataset `kapitanov/hagrid` con Kaggle CLI.
2. Descomprime el contenido en `datasets/hagrid_kaggle_raw/`.
3. Copia los JSON de anotaciones utiles a `datasets/ann_subsample/`.

Salida esperada:

- `datasets/hagrid_kaggle_raw/`
- `datasets/ann_subsample/`

### 2. Extraer FreiHAND y generar MST reales

Usa el script canonico:

```bash
uv run python src/datasets/setup/generar_mst_real_datasets.py \
  --freihand-rgb-dir datasets/FreiHAND_pub_v2/training/rgb \
  --freihand-limit 32560 \
  --hagrid-image-roots datasets/hagrid_dataset data/raw/images \
  --hagrid-annotations-dir datasets/ann_subsample \
  --output-csv csv/mst_real_dataset.csv \
  --output-summary output/mst_real_summary.json
```

Que hace:

1. Recorre FreiHAND desde `datasets/FreiHAND_pub_v2/training/rgb/`.
2. Busca imagenes de HaGRID en las rutas indicadas.
3. Clasifica cada imagen en un nivel MST del 1 al 10.
4. Guarda el CSV base que luego alimenta el resto del balanceo.

Salida esperada:

- `csv/mst_real_dataset.csv`
- `output/mst_real_summary.json`

### 3. Construir solicitudes sinteticas de balanceo

Usa el wrapper canonico:

```bash
uv run python src/datasets/manos/build_synthetic_manifest.py \
  --real-csv csv/mst_hagrid_sample_30k_384p.csv \
  --real-csv csv/mst_freihand_real_images.csv \
  --target-block-count-claro 4000 \
  --target-block-count-medio 60214 \
  --target-block-count-oscuro 5000 \
  --expected-qc-acceptance-rate 0.46 \
  --output-manifest datasets/synthetic_mst/metadata/manifest_synthetic_requests_blocks_qc_adjusted.csv \
  --output-summary output/auditoria/synthetic_request_summary_blocks_qc_adjusted.json
```

Que hace:

1. Calcula el deficit por bloque MST.
2. Genera solicitudes sinteticas solo para los bloques que lo necesitan.
3. Ajusta la cantidad bruta esperada segun la tasa de aceptacion del QC.

Salida esperada:

- `datasets/synthetic_mst/metadata/manifest_synthetic_requests_blocks_qc_adjusted.csv`
- `output/auditoria/synthetic_request_summary_blocks_qc_adjusted.json`

### 4. Generar imagenes sinteticas

Usa el wrapper canonico:

```bash
uv run python src/datasets/manos/generate_synthetic_skin_tones.py \
  --request-manifest datasets/synthetic_mst/metadata/manifest_synthetic_requests_blocks_qc_adjusted.csv \
  --output-images-dir datasets/synthetic_mst/images_blocks_qc_adjusted \
  --output-manifest datasets/synthetic_mst/metadata/manifest_synthetic_generated_blocks_qc_adjusted.csv \
  --output-summary output/auditoria/synthetic_generation_summary_blocks_qc_adjusted.json \
  --strength 0.85
```

Salida esperada:

- `datasets/synthetic_mst/images_blocks_qc_adjusted/`
- `datasets/synthetic_mst/metadata/manifest_synthetic_generated_blocks_qc_adjusted.csv`
- `output/auditoria/synthetic_generation_summary_blocks_qc_adjusted.json`

### 5. Ejecutar QC de sinteticos

Usa el wrapper canonico:

```bash
uv run python src/datasets/manos/qc_synthetic_dataset.py \
  --generated-manifest datasets/synthetic_mst/metadata/manifest_synthetic_generated_blocks_qc_adjusted.csv \
  --accepted-manifest datasets/synthetic_mst/metadata/manifest_synthetic_accepted_blocks_qc_adjusted.csv \
  --report-json output/auditoria/synthetic_qc_report_blocks_qc_adjusted.json \
  --tolerance 1
```

Salida esperada:

- `datasets/synthetic_mst/metadata/manifest_synthetic_accepted_blocks_qc_adjusted.csv`
- `output/auditoria/synthetic_qc_report_blocks_qc_adjusted.json`

### 6. Generar secuencias temporales para MANO

Usa el wrapper canonico:

```bash
uv run python src/datasets/manos/generar_secuencias_temporales.py \
  --input-dir datasets/synthetic_mst/mano_samples_balanced \
  --output-dir data/processed/secuencias_stgcn/mano \
  --output-manifest output/manifest_mano_secuencias.csv \
  --T 16 \
  --seed 42
```

Salida esperada:

- `data/processed/secuencias_stgcn/mano/`
- `output/manifest_mano_secuencias.csv`

### 7. Balancear por bloque MST de forma estricta

Usa el wrapper canonico:

```bash
uv run python src/datasets/manos/build_balanced_block_manifest.py \
  --input-csv csv/mst_hagrid_sample_30k_384p.csv \
  --input-csv csv/mst_freihand_real_images.csv \
  --input-csv datasets/synthetic_mst/metadata/manifest_mano_samples_balanced.csv \
  --output-manifest output/manifest_balanced_blocks.csv \
  --output-summary output/auditoria/manifest_balanced_blocks_summary.json
```

Que hace:

1. Une las fuentes reales con el lote MANO ya registrado.
2. Infere el bloque MST de cada fila.
3. Hace muestreo descendente del bloque dominante hasta el minimo comun.

La carpeta fuente para esa tercera entrada es `datasets/synthetic_mst/mano_samples_balanced/`, registrada con `src/datasets/manos/build_mano_samples_manifest.py`.

Resultado verificado:

- `claro`: 1691
- `medio`: 1691
- `oscuro`: 1691

## Archivos de referencia

- [src/datasets/README.md](src/datasets/README.md)
- [src/datasets/setup/README.md](src/datasets/setup/README.md)
- [src/datasets/manos/README.md](src/datasets/manos/README.md)