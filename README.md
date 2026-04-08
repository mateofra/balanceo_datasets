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
│   └── st_gcn_dataloader.py 🔗 DataLoader para ST-GCN
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
from src.st_gcn_dataloader import create_dataloaders

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
uv run python src/train_stgcn_example.py
```

📍 Checkpoints → `output/training_logs/`
📍 TensorBoard → `tensorboard --logdir=output/training_logs/tensorboard`

---

## 📚 Documentación

| Documento | Propósito |
|-----------|-----------|
| [docs/PIPELINE_MST_STGCN.md](docs/PIPELINE_MST_STGCN.md) | Arquitectura end-to-end |
| [docs/GUIA_TRAINING_STGCN.md](docs/GUIA_TRAINING_STGCN.md) | Guía de entrenamiento |
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

## 🔧 Requisitos

- Python 3.13+
- `uv`

**Instalación:**
```bash
uv sync
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

**Ubicación:** [`scripts/generate/generar_graficos_balanceo.py`](scripts/generate/README.md)

```bash
uv run python scripts/generate/generar_graficos_balanceo.py \
  --manifest-csv csv/train_manifest_balanceado_freihand_hagrid.csv \
  --summary-json output/reports/resumen_balanceo.json \
  --output-dir output/graphics
```

**Salida:**
- Composición por fuente (FreiHAND vs HaGRID)
- Distribución MST (claro/medio/oscuro)
- Histogramas de gestos
- Reporte: `output/graphics/reporte_graficos_balanceo.md`
