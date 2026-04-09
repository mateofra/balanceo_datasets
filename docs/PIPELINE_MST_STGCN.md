# Pipeline Completo: Balanceo MST â†’ ST-GCN

## Objetivo
Generar un dataset balanceado por tono de piel (escala MST 1-10) a partir de FreiHAND + HaGRID, con landmarks preprocesados listos para entrenar ST-GCN.

---

## Arquitectura General

```
ENTRADA
  â”œâ”€ ImÃ¡genes de manos (FreiHAND + HaGRID)
  â”œâ”€ Anotaciones (HaGRID .json)
  â””â”€ Datos 3D (FreiHAND training_xyz.json)
       â†“
[FASE 1] ClassificaciÃ³n de Tonos MST
  â”œâ”€ Input: ImÃ¡genes .jpg
  â”œâ”€ Script: src/classification/clasificar_mst_mediapipe.py
  â”œâ”€ Proceso: MediaPipe Hand â†’ Palma â†’ LAB color â†’ Clasificar en MST 1-10
  â””â”€ Output: CSV con columnas [sample_id, image_id, mst_level]
       â†“
[FASE 2] ExtracciÃ³n de Landmarks
  â”œâ”€ Input: ImÃ¡genes + anotaciones HaGRID
  â”œâ”€ Herramienta: MediaPipe Hand Landmarker o FreiHAND xyz
  â”œâ”€ Proceso: Extraer 21 puntos de mano â†’ normalizar â†’ guardar como .npy
  â”œâ”€ Output: data/processed/landmarks/
  â”‚   â”œâ”€ freihand_00000001.npy (21Ã—3 coords)
  â”‚   â””â”€ hagrid_gesture_imageid.npy (21Ã—3 coords)
  â””â”€ Estructura esperada: Nx21x3 (N muestras, 21 landmarks, 3 coords xyz)
       â†“
[FASE 3] Balanceo con InformaciÃ³n MST
  â”œâ”€ Input: 
  â”‚   â”œâ”€ CSV de MST (Phase 1 output)
  â”‚   â”œâ”€ datasets/training_xyz.json (FreiHAND)
  â”‚   â”œâ”€ datasets/ann_subsample/ (HaGRID anotaciones)
  â”‚   â””â”€ CSV anterior si existe (para reproducibilidad)
  â”œâ”€ Script: src/balancer/balancear_freihand_hagrid.py
  â”œâ”€ ConfiguraciÃ³n:
  â”‚   â”œâ”€ --target-size 20000 (muestras objetivo)
  â”‚   â”œâ”€ --hagrid-ratio 0.5 (50% HaGRID, 50% FreiHAND)
  â”‚   â”œâ”€ --mst-csv <path_a_csv_fase1>
  â”‚   â”œâ”€ --extreme-factor 2.0 (peso extra para MST 1,2,3,10)
  â”‚   â”œâ”€ --dark-jitter-factor 0.5 (replicaciÃ³n virtual MST 8-9)
  â”‚   â””â”€ --impute-missing-mst (llenar MST faltantes con imputation)
  â”œâ”€ Output CSV columnas:
  â”‚   â”œâ”€ sample_id
  â”‚   â”œâ”€ source (freihand/hagrid)
  â”‚   â”œâ”€ gesture
  â”‚   â”œâ”€ mst (1-10)
  â”‚   â””â”€ mst_origin (csv/imputed)
  â””â”€ Output: output/train_manifest_balanceado_freihand_hagrid.csv
       â†“
[FASE 4] GeneraciÃ³n Manifiesto ST-GCN
  â”œâ”€ Input:
  â”‚   â”œâ”€ train_manifest_balanceado_freihand_hagrid.csv (Phase 3)
  â”‚   â”œâ”€ data/processed/landmarks/ (Phase 2)
  â”œâ”€ Script: (ya integrado en balancear_freihand_hagrid.py)
  â”œâ”€ Flag: --output-stgcn-manifest-csv output/train_manifest_stgcn.csv
  â”œâ”€ Output columnas:
  â”‚   â”œâ”€ sample_id
  â”‚   â”œâ”€ path_landmarks (ruta relativa a .npy)
  â”‚   â”œâ”€ label (gesto)
  â”‚   â”œâ”€ condition (claro/medio/oscuro)
  â”‚   â”œâ”€ dataset (freihand/hagrid)
  â”‚   â”œâ”€ mst (1-10)
  â”‚   â”œâ”€ mst_origin (csv/imputed)
  â”‚   â””â”€ split (train/val/test)
  â””â”€ Output: output/train_manifest_stgcn.csv
       â†“
[SALIDA FINAL]
  â””â”€ Listo para ST-GCN training:
     â”œâ”€ Datos: data/processed/landmarks/*.npy
     â”œâ”€ Manifiesto: output/train_manifest_stgcn.csv
     â”œâ”€ Metadata: output/resumen_balanceo_*.json
     â””â”€ Splits: train/val balanceados por tono MST
```

---

## Fases Detalladas

### [FASE 1] ClasificaciÃ³n de Tonos MST

**Objetivo**: Para cada imagen de mano, detectar el tono de piel en escala MST.

**Script**:
```bash
uv run python src/classification/clasificar_mst_mediapipe.py <imagen> --model-path models/hand_landmarker.task
```

**Input requerido**:
- Carpeta de imÃ¡genes: `data/raw/images/` (structure similar a HaGRID)
- Modelo: `models/hand_landmarker.task` âœ“ (ya existe)

**Output CSV esperado**:
```csv
sample_id,image_id,mst_level,mst_label
freihand_00000001,freihand_1,7,medio-oscuro
hagrid_ok_image001,ok_001,3,claro
...
```

**Costos**:
- ~50-100ms por imagen (con GPU: ~10-20ms)
- 20K imÃ¡genes â‰ˆ 30-50 min (sin GPU) o 5-10 min (con GPU)

**Notas**:
- Requiere `mediapipe`, `opencv-python`, `numpy` (ya instalados)
- Si falta imagen â†’ skip automÃ¡tico
- Guarda en `csv/mst_classifications.csv`

---

### [FASE 2] ExtracciÃ³n de Landmarks (Opcional pero Recomendado)

**Objetivo**: Convertir 21 landmarks de MediaPipe a archivos `.npy` normalizados.

**Dos opciones**:

#### OpciÃ³n A: Usar landmarks de FreiHAND (ya disponibles)
```bash
# Los datos ya estÃ¡n en datasets/training_xyz.json
# Solo necesita conversiÃ³n a estructura Nx21x3 â†’ .npy individual
```

**Script necesario** (crear):
```python
# src/preprocessing/procesar_landmarks_freihand.py
# Lee datasets/training_xyz.json
# Para cada Ã­ndice, guarda data/processed/landmarks/freihand_XXXXXXX.npy
```

#### OpciÃ³n B: Generar de HaGRID con MediaPipe
```bash
# Requiere: imÃ¡genes de HaGRID + mediapipe detector
```

**Script necesario** (crear):
```python
# src/preprocessing/procesar_landmarks_hagrid_mediapipe.py
# Lee imÃ¡genes dataset/images/hagrid/*
# Extrae 21 landmarks per imagen â†’ data/processed/landmarks/hagrid_gesture_imageid.npy
```

**Output esperado**:
```
data/processed/landmarks/
â”œâ”€ freihand_00000001.npy  (shape: 21Ã—3)
â”œâ”€ freihand_00000002.npy
â”œâ”€ hagrid_ok_image001.npy
â””â”€ ...
```

**Nota**: Para este MVP, si no tienen imÃ¡genes completas de HaGRID, pueden usar solo FreiHAND landmarks.

---

### [FASE 3] Balanceo con InformaciÃ³n MST

**Objetivo**: Generar manifiesto balanceado con cuotas MST.

**Comandos**:

```bash
# BÃ¡sico (sin MST):
uv run python src/balancer/balancear_freihand_hagrid.py \
  --output-csv output/train_manifest_balanceado_freihand_hagrid.csv \
  --output-summary output/resumen_balanceo_freihand_hagrid.json

# Con informaciÃ³n MST (Phase 1):
uv run python src/balancer/balancear_freihand_hagrid.py \
  --mst-csv csv/mst_classifications.csv \
  --output-csv output/train_manifest_balanceado_freihand_hagrid.csv \
  --output-summary output/resumen_balanceo_freihand_hagrid.json \
  --target-size 20000 \
  --hagrid-ratio 0.5 \
  --extreme-factor 2.0 \
  --dark-jitter-factor 0.5

# Con manifiesto ST-GCN (Phase 4):
uv run python src/balancer/balancear_freihand_hagrid.py \
  --mst-csv csv/mst_classifications.csv \
  --output-csv output/train_manifest_balanceado_freihand_hagrid.csv \
  --output-summary output/resumen_balanceo_freihand_hagrid.json \
  --output-stgcn-manifest-csv output/train_manifest_stgcn.csv \
  --landmarks-root-dir data/processed/landmarks
```

**CSV Output**:
```csv
sample_id,source,gesture,mst,mst_origin,split,condition
freihand_00000001,freihand,unknown,7,imputed,train,medio-oscuro
hagrid_ok_image001,hagrid,ok,3,csv,train,claro
...
```

---

### [FASE 4] GeneraciÃ³n Manifiesto ST-GCN

**Objetivo**: Integrar landmarks + balanceo â†’ CSV lista para ST-GCN training.

**Output CSV** (ST-GCN format):
```csv
sample_id,path_landmarks,label,condition,dataset,mst,mst_origin,split
freihand_00000001,data/processed/landmarks/freihand_00000001.npy,unknown,medio-oscuro,freihand,7,imputed,train
hagrid_ok_image001,data/processed/landmarks/hagrid_ok_image001.npy,ok,claro,hagrid,3,csv,train
...
```

**Reproducibilidad**:
- Seed: `--seed 42` (fijo, reproducible)
- Summary JSON incluye: seed, configuraciÃ³n de sampling, estadÃ­sticas balanceo
- Ejemplo: `output/resumen_balanceo_freihand_hagrid.json` (guarda configuraciÃ³n exacta)

---

## Plan de ImplementaciÃ³n Paso a Paso

### Semana 1: MVP BÃ¡sico (Sin ImÃ¡genes de MST)

1. âœ… **Crear script de clasificaciÃ³n MST** (`src/classification/clasificar_mst_mediapipe.py`)
2. **Ejecutar balanceo bÃ¡sico** (sin MST, solo 50-50 FreiHAND/HaGRID)
   ```bash
   uv run python src/balancer/balancear_freihand_hagrid.py
   ```
3. **Generar landmarks FreiHAND** (convertir `training_xyz.json` â†’ `.npy` individual)
4. **Generar manifiesto ST-GCN** bÃ¡sico (sin MST por tono, pero con estructura)

### Semana 2: IntegraciÃ³n MST

1. **Procesar imÃ¡genes de entrenamiento** (si estÃ¡n disponibles)
   ```bash
   for img in data/raw/images/*.jpg; do
     uv run python src/classification/clasificar_mst_mediapipe.py "$img"
   done > csv/mst_classifications.csv
   ```
2. **Ejecutar balanceo con MST**
   ```bash
   uv run python src/balancer/balancear_freihand_hagrid.py --mst-csv csv/mst_classifications.csv ...
   ```
3. **Validar distribuciÃ³n MST** con grÃ¡ficos
   ```bash
   uv run python src/balancer/generar_graficos_balanceo.py
   ```

### Semana 3: Entrenamiento ST-GCN

1. **Usar manifiesto ST-GCN para training**
2. **Validar convergencia por tono de piel**
3. **Evaluar sesgos en mÃ©tricas por MST**

---

## Archivos Necesarios (Estado Actual)

| Archivo | Status | Notas |
|---------|--------|-------|
| `src/classification/clasificar_mst_mediapipe.py` | âœ… Creado | Listo para ejecutar |
| `src/balancer/balancear_freihand_hagrid.py` | âœ… Existente | Probado |
| `src/preprocessing/procesar_landmarks_freihand.py` | âŒ Falta | Prioridad media |
| `src/preprocessing/procesar_landmarks_hagrid_mediapipe.py` | âŒ Falta | Prioridad media |
| `models/hand_landmarker.task` | âœ… Existe | Modelo MediaPipe |
| `datasets/training_xyz.json` | âœ… Existe | FreiHAND data |
| `datasets/ann_subsample/` | âœ… Existe | HaGRID anotaciones |
| `data/raw/images/` | âŒ Falta | ImÃ¡genes a procesar (opcional) |
| `data/processed/landmarks/` | âŒ Falta | Output landmarks .npy |

---

## RecomendaciÃ³n Inmediata

**OpciÃ³n A: MVP RÃ¡pido (hoy)**
```bash
# Sin MST, sin landmarks procesados
# Solo genera manifiesto CSV balanceado bÃ¡sico
uv run python src/balancer/balancear_freihand_hagrid.py \
  --output-csv output/train_manifest_balanceado_freihand_hagrid.csv \
  --output-summary output/resumen_balanceo_freihand_hagrid.json
```

**OpciÃ³n B: Completo (con landmarks FreiHAND)**
```bash
# Requiere: crear src/preprocessing/procesar_landmarks_freihand.py
# Luego generar manifiesto ST-GCN
```

Â¿CuÃ¡l prefieres primero?

