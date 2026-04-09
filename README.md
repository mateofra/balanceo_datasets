# 🎯 balanceo_datasets

Repositorio para construir conjuntos de entrenamiento balanceados entre **FreiHAND** y **HaGRID**, reduciendo sesgos por distribución de tonos **MST** en modelos de reconocimiento de gestos (**ST-GCN**).

## 📋 Estructura del Proyecto

```
balanceo_datasets/
├── docs/                    📚 Documentación técnica
├── src/                     📦 Código fuente modular
│   ├── balancer/            ⚖️  Balanceo de datasets
│   ├── preprocessing/       🔄 Preprocesamiento de landmarks
│   ├── classification/      🎨 Clasificación MST
│   └── stgcn/               🕸️  Modelo y utilidades ST-GCN
├── scripts/                 🔧 Scripts ejecutables
│   ├── diagnosis/           🔍 Diagnóstico y validación
│   ├── training/            🎓 Entrenamiento y visualización
│   ├── repair/              🔧 Reparación de datos
│   └── generate/            📊 Generación de reportes
├── data/                    💾 Datos procesados
├── output/                  📤 Manifiestos y reportes
├── datasets/                📦 Datos de referencia
├── models/                  🤖 Modelos preentrenados
├── csv/                     📋 CSVs de datos
├── graficos/                📈 Visualizaciones
├── stgcn/                   🕸️  Módulo ST-GCN
├── tests/                   ✅ Pruebas unitarias
└── pyproject.toml
```

## 🎯 Objetivos del Proyecto

- ✅ Balancear FreiHAND + HaGRID por cuotas reproducibles
- ✅ Equilibrar por tono de piel (escala MST 1-10)
- ✅ Categorización: claro/medio/oscuro
- ✅ Generar manifiesto listo para ST-GCN
- ✅ Mantener validación/test sin artificio

## 🚀 Quick Start: Entrenar ST-GCN

### ⚡ En 3 Pasos

**1) Datos disponibles:**
```
output/manifests/
└── train_manifest_stgcn_fixed.csv    # 10K muestras FreiHAND

data/processed/landmarks/
└── freihand_*.npy                    # 32,560 landmarks (21×3 coords)
```

**2) Importar DataLoader:**
```python
from src.stgcn.st_gcn_dataloader import create_dataloaders

loaders = create_dataloaders(
    manifest_csv="output/manifests/train_manifest_stgcn_fixed.csv",
    batch_size=32,
    normalize=True,
    balance_by_mst=True  # ⭐ Balancea por tono de piel
)

train_loader = loaders["train"]

# Iterar batches
for batch in train_loader:
    landmarks = batch["landmarks"]  # (BS, 21, 3)
    labels = batch["label"]         # Gesture labels
    mst = batch["mst"]              # Tono MST (1-10)
    condition = batch["condition"]  # claro/medio/oscuro
```

**3) Entrenar:**
```bash
uv run python src/stgcn/train_stgcn_example.py
```

📍 Checkpoints → `output/training_logs/`
📍 TensorBoard → `tensorboard --logdir=output/training_logs/tensorboard`

---

## 📚 Documentación

| Documento | Propósito |
|-----------|-----------|
| [docs/PIPELINE_MST_STGCN.md](docs/PIPELINE_MST_STGCN.md) | Arquitectura end-to-end |
| [docs/GUIA_TRAINING_STGCN.md](docs/GUIA_TRAINING_STGCN.md) | Guía de entrenamiento |
| [docs/RECREAR_DATASETS_EN_CLON.md](docs/RECREAR_DATASETS_EN_CLON.md) | Como recrear directorios ignorados y preparar datasets en un clon nuevo |
| [src/README.md](src/README.md) | Referencia de código |
| [scripts/README.md](scripts/README.md) | Scripts disponibles |

## 📊 Dataset Balanceado - Estado Actual

| Métrica | Valor |
|---------|-------|
| **Total samples** | 20,000 |
| **FreiHAND** | 10,000 (50%) ✅ |
| **HaGRID** | 10,000 (50%) ⏳ |
| **Balanceo MST** | Claro/Medio/Oscuro ≈ 33% c/u |
| **Oversampling** | MST extremos (1,2,3,10) |

## Estado actual del dataset

| Fuente | Calidad | Usado en entrenamiento |
|--------|---------|----------------------|
| FreiHAND (10,000) | real_3d_freihand | ✓ Sí |
| HaGRID annotation_2d (9,419) | annotation_2d_projected | ✗ Pendiente reprocesado con MediaPipe |
| HaGRID synthetic_mean (581) | synthetic_gesture_mean | ✗ Excluido |

HaGRID pendiente: requiere descarga de imagenes crudas y reprocesado con MediaPipe
para obtener landmarks 3D reales equivalentes a FreiHAND.

## Organización actual por finalidad

### Balanceo

- Artefactos de balanceo y curado en [output/balanceo/](output/balanceo/README.md)
- Scripts de balanceo en [src/balancer/balancear_freihand_hagrid.py](src/balancer/balancear_freihand_hagrid.py)

### Training

- Artefactos de entrenamiento en [output/training/](output/training/README.md)
- Entrenamiento supervisado en [scripts/training/train_supervisado.py](scripts/training/train_supervisado.py)
- Entrenamiento auto-supervisado en [scripts/training/train_autosupervisado.py](scripts/training/train_autosupervisado.py)

### Auditoria

- Resultados y graficas en [output/auditoria/](output/auditoria/README.md)
- Grafica resumen en [graficos/auditoria_dpr/](graficos/auditoria_dpr/README.md)
- Auditoria DPR en [scripts/auditoria/auditoria_dpr.py](scripts/auditoria/auditoria_dpr.py)

### Scripts

- Indice ordenado de scripts en [scripts/README.md](scripts/README.md)

## Errores y soluciones resueltos

- Rutas HaGRID inconsistentes: se corrigieron con resolucion por `sample_id`, rutas canonicas y filtrado de archivos inexistentes.
- Secuencias sintéticas demasiado estaticas: se diagnosticó leakage temporal y se cambio la tarea auto-supervisada para evitar trampa trivial.
- Entorno CPU-only: se fijo PyTorch con indice CUDA para usar la RTX 4050 desde `uv`.
- TVD mal definido: se reemplazo por TVD canonico entre distribuciones de error por bloque MST.
- Datos mezclados por finalidad: se separaron artefactos de balanceo, training y auditoria en carpetas especificas.

## Estructura de directorios excluidos con .gitignore

Para evitar subir datos pesados o sensibles, las carpetas ignoradas deben seguir esta convención:

```text
data/
  raw/                # datos crudos locales, nunca versionados
  processed/          # intermedios regenerables
datasets/             # dumps externos o datasets descargados
stgcn/data/           # datos del submodulo ST-GCN
.venv/                # entorno local de Python
```

Regla operativa:

- Si un artefacto puede regenerarse con scripts del repo, debe ir a un directorio ignorado.
- Si un artefacto es parte del entregable (manifiestos finales, historiales, auditoria), debe ir en `output/` o `graficos/` en carpetas tematicas.
- No mezclar datos crudos con artefactos finales.

## 🔧 Requisitos

- Python 3.13+
- `uv`

**Instalación:**
```bash
uv sync
```

## Launcher unificado

Se puede usar un único punto de entrada con `main.py`:

```bash
# Ver comandos disponibles
uv run python main.py --help

# Entrenamiento supervisado (modo por defecto)
uv run python main.py train

# Entrenamiento auto-supervisado por 2 epocas
uv run python main.py train --modo autosupervisado --epochs 2

# Auditoria y grafica
uv run python main.py auditoria

# Pipeline completo: entrenamiento + auditoria
uv run python main.py pipeline --modo supervisado --epochs 30

# Solo auditoria usando artefactos existentes (sin reentrenar)
uv run python main.py pipeline --skip-train

# Nota: pipeline imprime un resumen final con estado y rutas de artefactos

# Setup de datos locales (crear estructura ignorada + verificar)
uv run python main.py setup-data

# Setup de datos + asistente HaGRID en dry-run
uv run python main.py setup-data --download-hagrid
# Nota: en dry-run no falla si falta Kaggle CLI; solo informa y sigue con verify.

# Setup de datos + descarga real de HaGRID + preparar ann_subsample
uv run python main.py setup-data --download-hagrid --execute-download --prepare-ann-subsample

# Forzar que setup-data falle si la descarga real falla
uv run python main.py setup-data --download-hagrid --execute-download --strict-download

# Test rapido de forward
uv run python main.py test-forward
```

---

## ⚖️ Balanceador: FreiHAND + HaGRID

**Ubicación:** [`src/balancer/balancear_freihand_hagrid.py`](src/balancer/README.md)

### Uso Básico

Balanceo simple 50/50:
```bash
uv run python src/balancer/balancear_freihand_hagrid.py \
  --freihand-training-xyz datasets/training_xyz.json \
  --hagrid-annotations-dir datasets/ann_subsample \
  --target-size 20000 \
  --hagrid-ratio 0.5 \
  --extreme-mst-levels 1 2 3 10 \
  --extreme-factor 2.0 \
  --seed 42 \
  --output-csv output/manifests/train_manifest_balanceado.csv \
  --output-summary output/reports/resumen_balanceo.json
```

### Con Auditoria MST

Si tienes CSV con auditoría de tonos:
```bash
uv run python src/balancer/balancear_freihand_hagrid.py \
  --freihand-training-xyz datasets/training_xyz.json \
  --hagrid-annotations-dir datasets/ann_subsample \
  --mst-csv csv/auditoria_mst.csv \
  --target-size 20000 \
  --hagrid-ratio 0.5 \
  --seed 42 \
  --output-tone-sets-dir csv/sets_tonos_train
```

Genera:
- `csv/sets_tonos_train/train_set_claro.csv`
- `csv/sets_tonos_train/train_set_medio.csv`
- `csv/sets_tonos_train/train_set_oscuro.csv`

### Para ST-GCN

Manifiesto único con rutas a `.npy`:
```bash
uv run python src/balancer/balancear_freihand_hagrid.py \
  --freihand-training-xyz datasets/training_xyz.json \
  --hagrid-annotations-dir datasets/ann_subsample \
  --mst-csv csv/auditoria_mst.csv \
  --target-size 20000 \
  --hagrid-ratio 0.5 \
  --landmarks-root-dir data/processed/landmarks \
  --output-stgcn-manifest-csv output/manifests/train_manifest_stgcn.csv \
  --seed 42
```

### Columnas del Manifiesto ST-GCN

| Columna | Descripción |
|---------|-------------|
| `sample_id` | ID único |
| `path_landmarks` | Ruta a `.npy` |
| `label` | Gesto (o `unknown` para FreiHAND) |
| `condition` | claro/medio/oscuro |
| `dataset` | freihand o hagrid |
| `mst` | Nivel MST (1-10) |
| `mst_origin` | original o imputed |
| `split` | train |
| `sampling_weight` | Peso de oversampling |
| `augmentation_hint` | Marcas de augmentación |

---

## 📊 Generar Gráficos

**Ubicación:** [`src/balancer/generar_graficos_balanceo.py`](src/balancer/README.md)

```bash
uv run python src/balancer/generar_graficos_balanceo.py \
  --manifest-csv csv/train_manifest_balanceado_freihand_hagrid.csv \
  --summary-json output/reports/resumen_balanceo.json \
  --output-dir output/graphics
```

**Salida:**
- Composición por fuente (FreiHAND vs HaGRID)
- Distribución MST (claro/medio/oscuro)
- Histogramas de gestos
- Reporte: `output/graphics/reporte_graficos_balanceo.md`
