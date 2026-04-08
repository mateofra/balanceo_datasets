# Pipeline Completo: Balanceo MST → ST-GCN

## Objetivo
Generar un dataset balanceado por tono de piel (escala MST 1-10) a partir de FreiHAND + HaGRID, con landmarks preprocesados listos para entrenar ST-GCN.

---

## Arquitectura General

```
ENTRADA
  ├─ Imágenes de manos (FreiHAND + HaGRID)
  ├─ Anotaciones (HaGRID .json)
  └─ Datos 3D (FreiHAND training_xyz.json)
       ↓
[FASE 1] Classificación de Tonos MST
  ├─ Input: Imágenes .jpg
  ├─ Script: src/clasificar_mst_mediapipe.py
  ├─ Proceso: MediaPipe Hand → Palma → LAB color → Clasificar en MST 1-10
  └─ Output: CSV con columnas [sample_id, image_id, mst_level]
       ↓
[FASE 2] Extracción de Landmarks
  ├─ Input: Imágenes + anotaciones HaGRID
  ├─ Herramienta: MediaPipe Hand Landmarker o FreiHAND xyz
  ├─ Proceso: Extraer 21 puntos de mano → normalizar → guardar como .npy
  ├─ Output: data/processed/landmarks/
  │   ├─ freihand_00000001.npy (21×3 coords)
  │   └─ hagrid_gesture_imageid.npy (21×3 coords)
  └─ Estructura esperada: Nx21x3 (N muestras, 21 landmarks, 3 coords xyz)
       ↓
[FASE 3] Balanceo con Información MST
  ├─ Input: 
  │   ├─ CSV de MST (Phase 1 output)
  │   ├─ datasets/training_xyz.json (FreiHAND)
  │   ├─ datasets/ann_subsample/ (HaGRID anotaciones)
  │   └─ CSV anterior si existe (para reproducibilidad)
  ├─ Script: src/balancear_freihand_hagrid.py
  ├─ Configuración:
  │   ├─ --target-size 20000 (muestras objetivo)
  │   ├─ --hagrid-ratio 0.5 (50% HaGRID, 50% FreiHAND)
  │   ├─ --mst-csv <path_a_csv_fase1>
  │   ├─ --extreme-factor 2.0 (peso extra para MST 1,2,3,10)
  │   ├─ --dark-jitter-factor 0.5 (replicación virtual MST 8-9)
  │   └─ --impute-missing-mst (llenar MST faltantes con imputation)
  ├─ Output CSV columnas:
  │   ├─ sample_id
  │   ├─ source (freihand/hagrid)
  │   ├─ gesture
  │   ├─ mst (1-10)
  │   └─ mst_origin (csv/imputed)
  └─ Output: output/train_manifest_balanceado_freihand_hagrid.csv
       ↓
[FASE 4] Generación Manifiesto ST-GCN
  ├─ Input:
  │   ├─ train_manifest_balanceado_freihand_hagrid.csv (Phase 3)
  │   ├─ data/processed/landmarks/ (Phase 2)
  ├─ Script: (ya integrado en balancear_freihand_hagrid.py)
  ├─ Flag: --output-stgcn-manifest-csv output/train_manifest_stgcn.csv
  ├─ Output columnas:
  │   ├─ sample_id
  │   ├─ path_landmarks (ruta relativa a .npy)
  │   ├─ label (gesto)
  │   ├─ condition (claro/medio/oscuro)
  │   ├─ dataset (freihand/hagrid)
  │   ├─ mst (1-10)
  │   ├─ mst_origin (csv/imputed)
  │   └─ split (train/val/test)
  └─ Output: output/train_manifest_stgcn.csv
       ↓
[SALIDA FINAL]
  └─ Listo para ST-GCN training:
     ├─ Datos: data/processed/landmarks/*.npy
     ├─ Manifiesto: output/train_manifest_stgcn.csv
     ├─ Metadata: output/resumen_balanceo_*.json
     └─ Splits: train/val balanceados por tono MST
```

---

## Fases Detalladas

### [FASE 1] Clasificación de Tonos MST

**Objetivo**: Para cada imagen de mano, detectar el tono de piel en escala MST.

**Script**:
```bash
uv run python src/clasificar_mst_mediapipe.py <imagen> --model-path models/hand_landmarker.task
```

**Input requerido**:
- Carpeta de imágenes: `data/raw/images/` (structure similar a HaGRID)
- Modelo: `models/hand_landmarker.task` ✓ (ya existe)

**Output CSV esperado**:
```csv
sample_id,image_id,mst_level,mst_label
freihand_00000001,freihand_1,7,medio-oscuro
hagrid_ok_image001,ok_001,3,claro
...
```

**Costos**:
- ~50-100ms por imagen (con GPU: ~10-20ms)
- 20K imágenes ≈ 30-50 min (sin GPU) o 5-10 min (con GPU)

**Notas**:
- Requiere `mediapipe`, `opencv-python`, `numpy` (ya instalados)
- Si falta imagen → skip automático
- Guarda en `csv/mst_classifications.csv`

---

### [FASE 2] Extracción de Landmarks (Opcional pero Recomendado)

**Objetivo**: Convertir 21 landmarks de MediaPipe a archivos `.npy` normalizados.

**Dos opciones**:

#### Opción A: Usar landmarks de FreiHAND (ya disponibles)
```bash
# Los datos ya están en datasets/training_xyz.json
# Solo necesita conversión a estructura Nx21x3 → .npy individual
```

**Script necesario** (crear):
```python
# src/procesar_landmarks_freihand.py
# Lee datasets/training_xyz.json
# Para cada índice, guarda data/processed/landmarks/freihand_XXXXXXX.npy
```

#### Opción B: Generar de HaGRID con MediaPipe
```bash
# Requiere: imágenes de HaGRID + mediapipe detector
```

**Script necesario** (crear):
```python
# src/procesar_landmarks_hagrid_mediapipe.py
# Lee imágenes dataset/images/hagrid/*
# Extrae 21 landmarks per imagen → data/processed/landmarks/hagrid_gesture_imageid.npy
```

**Output esperado**:
```
data/processed/landmarks/
├─ freihand_00000001.npy  (shape: 21×3)
├─ freihand_00000002.npy
├─ hagrid_ok_image001.npy
└─ ...
```

**Nota**: Para este MVP, si no tienen imágenes completas de HaGRID, pueden usar solo FreiHAND landmarks.

---

### [FASE 3] Balanceo con Información MST

**Objetivo**: Generar manifiesto balanceado con cuotas MST.

**Comandos**:

```bash
# Básico (sin MST):
uv run python src/balancear_freihand_hagrid.py \
  --output-csv output/train_manifest_balanceado_freihand_hagrid.csv \
  --output-summary output/resumen_balanceo_freihand_hagrid.json

# Con información MST (Phase 1):
uv run python src/balancear_freihand_hagrid.py \
  --mst-csv csv/mst_classifications.csv \
  --output-csv output/train_manifest_balanceado_freihand_hagrid.csv \
  --output-summary output/resumen_balanceo_freihand_hagrid.json \
  --target-size 20000 \
  --hagrid-ratio 0.5 \
  --extreme-factor 2.0 \
  --dark-jitter-factor 0.5

# Con manifiesto ST-GCN (Phase 4):
uv run python src/balancear_freihand_hagrid.py \
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

### [FASE 4] Generación Manifiesto ST-GCN

**Objetivo**: Integrar landmarks + balanceo → CSV lista para ST-GCN training.

**Output CSV** (ST-GCN format):
```csv
sample_id,path_landmarks,label,condition,dataset,mst,mst_origin,split
freihand_00000001,data/processed/landmarks/freihand_00000001.npy,unknown,medio-oscuro,freihand,7,imputed,train
hagrid_ok_image001,data/processed/landmarks/hagrid_ok_image001.npy,ok,claro,hagrid,3,csv,train
...
```

**Reproducibilidad**:
- Seed: `--seed 42` (fijo, reproducible)
- Summary JSON incluye: seed, configuración de sampling, estadísticas balanceo
- Ejemplo: `output/resumen_balanceo_freihand_hagrid.json` (guarda configuración exacta)

---

## Plan de Implementación Paso a Paso

### Semana 1: MVP Básico (Sin Imágenes de MST)

1. ✅ **Crear script de clasificación MST** (`src/clasificar_mst_mediapipe.py`)
2. **Ejecutar balanceo básico** (sin MST, solo 50-50 FreiHAND/HaGRID)
   ```bash
   uv run python src/balancear_freihand_hagrid.py
   ```
3. **Generar landmarks FreiHAND** (convertir `training_xyz.json` → `.npy` individual)
4. **Generar manifiesto ST-GCN** básico (sin MST por tono, pero con estructura)

### Semana 2: Integración MST

1. **Procesar imágenes de entrenamiento** (si están disponibles)
   ```bash
   for img in data/raw/images/*.jpg; do
     uv run python src/clasificar_mst_mediapipe.py "$img"
   done > csv/mst_classifications.csv
   ```
2. **Ejecutar balanceo con MST**
   ```bash
   uv run python src/balancear_freihand_hagrid.py --mst-csv csv/mst_classifications.csv ...
   ```
3. **Validar distribución MST** con gráficos
   ```bash
   uv run python src/generar_graficos_balanceo.py
   ```

### Semana 3: Entrenamiento ST-GCN

1. **Usar manifiesto ST-GCN para training**
2. **Validar convergencia por tono de piel**
3. **Evaluar sesgos en métricas por MST**

---

## Archivos Necesarios (Estado Actual)

| Archivo | Status | Notas |
|---------|--------|-------|
| `src/clasificar_mst_mediapipe.py` | ✅ Creado | Listo para ejecutar |
| `src/balancear_freihand_hagrid.py` | ✅ Existente | Probado |
| `src/procesar_landmarks_freihand.py` | ❌ Falta | Prioridad media |
| `src/procesar_landmarks_hagrid_mediapipe.py` | ❌ Falta | Prioridad media |
| `models/hand_landmarker.task` | ✅ Existe | Modelo MediaPipe |
| `datasets/training_xyz.json` | ✅ Existe | FreiHAND data |
| `datasets/ann_subsample/` | ✅ Existe | HaGRID anotaciones |
| `data/raw/images/` | ❌ Falta | Imágenes a procesar (opcional) |
| `data/processed/landmarks/` | ❌ Falta | Output landmarks .npy |

---

## Recomendación Inmediata

**Opción A: MVP Rápido (hoy)**
```bash
# Sin MST, sin landmarks procesados
# Solo genera manifiesto CSV balanceado básico
uv run python src/balancear_freihand_hagrid.py \
  --output-csv output/train_manifest_balanceado_freihand_hagrid.csv \
  --output-summary output/resumen_balanceo_freihand_hagrid.json
```

**Opción B: Completo (con landmarks FreiHAND)**
```bash
# Requiere: crear src/procesar_landmarks_freihand.py
# Luego generar manifiesto ST-GCN
```

¿Cuál prefieres primero?
