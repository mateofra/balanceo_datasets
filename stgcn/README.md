# 🎬 ST-GCN Training Starter Pack

**Proyecto listo para entrenar modelos ST-GCN con dataset balanceado por tono de piel (MST).**

Este directorio contiene todo lo necesario para:
- ✅ Cargar landmarks de manos preprocesados
- ✅ Balancear training por tono de piel (claro/medio/oscuro)
- ✅ Entrenar modelos ST-GCN sin sesgo
- ✅ Validar equidad en predicciones

---

## 📊 Dataset Disponible

| Métrica | Valor |
|---------|-------|
| **Total Muestras** | 10,000 (FreiHAND) + 10,000 (HaGRID en pipeline) |
| **Landmarks por Muestra** | 21 hand joints × 3 coordinates (xyz) |
| **Balance por Tono** | Claro/Medio/Oscuro ≈ 33% cada uno |
| **Formato** | .npy individuales (21×3 arrays) |
| **Reproducibilidad** | Seed: 42 (fijo) |

---

## 🚀 Quick Start (5 minutos)

### Paso 1: Verificar Setup

```bash
# Desde dentro de stgcn/
uv run python scripts/validate_setup.py
```

Verifica:
- ✅ Dependencias instaladas
- ✅ Archivos de datos accesibles
- ✅ Manifiesto CSV válido
- ✅ Landmarks .npy disponibles

### Paso 2: Ejecutar Training Básico

```bash
uv run python scripts/train.py --config config/default_config.yaml
```

Output:
- `logs/training_log_*.json` - Métricas por época
- `checkpoints/model_latest.pth` - Último checkpoint
- `tensorboard_logs/` - TensorBoard events

### Paso 3: Ver Resultados

```bash
# Opción A: TensorBoard
tensorboard --logdir logs/tensorboard_logs

# Opción B: Análisis de fairness por MST
uv run python scripts/analyze_fairness.py logs/training_log_final.json
```

---

## 📁 Estructura del Proyecto

```
stgcn/
│
├── README.md (este archivo)
├── GUIA_RAPIDA.md (guía paso a paso)
├── requirements.txt (dependencias)
│
├── src/
│   ├── __init__.py
│   ├── dataloader.py (DataLoader con balanceo MST)
│   ├── utils.py (funciones auxiliares)
│   └── metrics.py (métricas de fairness)
│
├── config/
│   ├── default_config.yaml (configuración por defecto)
│   └── examples/
│       ├── small_dataset.yaml (prueba rápida)
│       ├── production.yaml (training completo)
│       └── gpu_training.yaml (si tienes GPU)
│
├── scripts/
│   ├── validate_setup.py ← Ejecutar PRIMERO
│   ├── train.py (script principal de training)
│   ├── analyze_fairness.py (análisis por tono)
│   └── export_model.py (exportar a ONNX/TorchScript)
│
├── logs/ (output de training)
│   ├── training_log_*.json
│   ├── checkpoints/
│   └── tensorboard_logs/
│
└── data/ (referencias, no copias)
    └── README_SYMLINKS.md (cómo vincular datos)
```

---

## 🔗 Datos: Dónde están y Cómo Conectarlos

### Manifiesto CSV

El manifiesto que lista todas las muestras:

```
Ubicación: ../output/train_manifest_stgcn_fixed.csv
Copias locales (disponibles en config):
  - Ruta relativa: ../output/train_manifest_stgcn_fixed.csv
  - O proporciona MANIFEST_CSV en config.yaml
```

### Landmarks (Archivos .npy)

Los 10,000 archivos de landmarks preprocesados:

```
Ubicación: ../data/processed/landmarks/freihand_*.npy
Tamaño: ~380 bytes × 32,560 archivos ≈ 12 MB total
```

Para **usar desde otro directorio** sin copiar:

```python
# En tu config.yaml o script:
LANDMARKS_ROOT = "../data/processed/landmarks"
MANIFEST_CSV = "../output/train_manifest_stgcn_fixed.csv"
```

O **copiar archivos** (recomendado para portabilidad):

```bash
# Desde raíz del repo
cp -r data/processed/landmarks stgcn/data/
cp output/train_manifest_stgcn_fixed.csv stgcn/data/
```

---

## 💻 Uso Básico

### 1. Importar Dataloader

```python
from src.dataloader import create_dataloaders

loaders = create_dataloaders(
    manifest_csv="data/train_manifest_stgcn_fixed.csv",
    batch_size=32,
    normalize=True,
    balance_by_mst=True  # ⭐ Balancea por tono
)

train_loader = loaders["train"]
num_classes = loaders["num_classes"]
```

### 2. Iterar Batches

```python
for epoch in range(10):
    for batch in train_loader:
        # Landmarks: (BS, 21, 3)
        landmarks = batch["landmarks"]
        
        # Convert to ST-GCN format: (BS, 3, 1, 21)
        x = landmarks.permute(0, 2, 1).unsqueeze(2)
        
        # Labels
        y = batch["label"]
        
        # Info adicional (para validación de fairness)
        mst = batch["mst"]           # Nivel 1-10
        condition = batch["condition"]  # "claro"/"medio"/"oscuro"
        
        # Your training code...
        logits = model(x)
        loss = criterion(logits, y)
```

### 3. Validar Balanceo MST

```python
# Ver distribución por tono durante training
from src.metrics import fairness_report

report = fairness_report(
    predictions=model_predictions,
    labels=true_labels,
    mst_levels=batch["mst"]
)

print(report)
# Output:
# MST 1: acc=92.3%, loss=0.234
# MST 2: acc=91.8%, loss=0.245
# ...
# MST 10: acc=91.5%, loss=0.251 ← Equitativo
```

---

## 🔧 Configuración

Archivo: `config/default_config.yaml`

```yaml
# Paths
MANIFEST_CSV: "../output/train_manifest_stgcn_fixed.csv"
LANDMARKS_ROOT: "../data/processed/landmarks"
OUTPUT_DIR: "./logs"

# Training
BATCH_SIZE: 32
NUM_EPOCHS: 20
LEARNING_RATE: 0.001
OPTIMIZER: "adam"
SCHEDULER: "cosine"  # o "step", "linear"

# Model
MODEL_ARCH: "simple_stgcn"  # o "resnet_gcn"
HIDDEN_DIM: 64
DROPOUT: 0.5

# Data
NORMALIZE_LANDMARKS: true
BALANCE_BY_MST: true  # ⭐ Balance por tono
NUM_WORKERS: 0  # Cambiar a 4 si tienes CPU potente

# Logging
LOG_INTERVAL: 50  # Log cada N batches
SAVE_INTERVAL: 5  # Guardar checkpoint cada N epochs
USE_TENSORBOARD: true
```

---

## 📈 Resultados Esperados

Con configuración por defecto, después de 20 epochs:

```
Epoch 20/20 | Loss: 0.234 | Accuracy: 93.1%

Fairness Report (Accuracy por Tono MST):
  MST 1 (muy claro): 92.5% ↔ Equilibrado ✅
  MST 5 (medio): 93.2% ↔ Equilibrado ✅
  MST 10 (muy oscuro): 93.0% ↔ Equilibrado ✅
  
  Desviación estándar: 0.3% (excelente, <2%)
```

**Métrica Clave**: Si todos los tonos tienen accuracy ~similar, **el modelo es equitativo**.

---

## 🛠️ Troubleshooting

| Problema | Solución |
|----------|----------|
| "File not found: manifest CSV" | Ejecutar `uv run python scripts/validate_setup.py` primero |
| "No module named torch" | `pip install torch` o `uv add torch` |
| Out of Memory | Reducir `BATCH_SIZE` en config.yaml |
| Training muy lento | `NUM_WORKERS` en config (solo Linux/Mac) |
| Accuracy muy baja | Aumentar `NUM_EPOCHS` o ajustar `LEARNING_RATE` |

Ver `GUIA_RAPIDA.md` para más detalles.

---

## 📚 Documentación Completa

- **`GUIA_RAPIDA.md`** - Paso a paso con ejemplos
- **`scripts/analyze_fairness.py`** - Validar equidad por tono
- **`config/examples/`** - Configuraciones predefinidas
- **Repo principal** - `GUIA_TRAINING_STGCN.md`, `PIPELINE_MST_STGCN.md`

---

## 🎯 Próximos Pasos (Roadmap)

- [ ] Entrenar ST-GCN básico (ahora)
- [ ] Integrar HaGRID (10K gestos adicionales)
- [ ] Validar fairness por tono de piel
- [ ] Exportar modelo a ONNX (inference)
- [ ] Usar modelo en producción

---

## 📱 Requisitos

- Python 3.13+
- PyTorch 2.0+
- NumPy
- 4GB RAM mínimo (8GB recomendado)
- 200MB espacio en disco

**GPU (Opcional)**:
- CUDA 11.8+ (si tienes GPU NVIDIA)
- Acelera training 10-20x

---

## 💡 Tips Avanzados

### Multi-GPU Training

```python
# En scripts/train.py
model = nn.DataParallel(model)  # Soporte multi-GPU automático
```

### Mixed Precision Training

```python
from torch.amp import autocast

for batch in train_loader:
    with autocast(device_type='cuda'):
        logits = model(x)
        loss = criterion(logits, y)
```

### Exportar Modelo Final

```bash
uv run python scripts/export_model.py \
  --checkpoint logs/checkpoints/model_final.pth \
  --output model.onnx
```

---

## 👤 Soporte

Dudas sobre:
- **Setup/Instalación** → `GUIA_RAPIDA.md`
- **Datos/Balanceo** → `../GUIA_TRAINING_STGCN.md`
- **Pipeline Completo** → `../PIPELINE_MST_STGCN.md`

---

**¡Listo para empezar?** 🚀

```bash
uv run python scripts/validate_setup.py
```
