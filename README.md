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

---

## 📅 Roadmap y Estado Actual del Sprint (KPNBODY26)

Este segmento documenta el progreso de las fases planificadas para el sprint en curso, rindiendo cuentas sobre su éxito e iterando de manera continua sobre los cuellos de botella tecnológicos de la arquitectura actual para priorizar futuras correcciones. 

### ✅ Fases Estables e Implementadas

- **Fase 0/0.1 (Obtención y Evaluación de Datasets)**: 
  Completada exitosamente. Se integraron `FreiHAND` y `HaGRID`. Se descubrió la deficiencia masiva (problema fenotípico "WEIRD") respecto a las sub-menciones en los estratos de piel de grados extremos (**MST 1 al 3 y predominantemente MST 10**) en las anotaciones reales analizadas.
- **Fase 0.2 (Balanceo de Datasets)**: 
  Completada exitosamente. Se encapsuló la abstracción de muestreo en un pipeline estricto (OOP), induciendo equidad del `50%` contra el `50%` cruzado de datasets y estableciendo robustamente una cuota racial de tres tercios para bloques de Tono de Piel (Claro 33%, Medio 33%, Oscuro 33%) mediante un _oversampling_ y parametrización asimétrica. 
- **Fase 2 y 4 (Parcial - Modelado ST-GCN y Auditoría DPR)**: 
  La fase de extracción referencial probabilística (_Demographic Parity Ratio_) computada en `auditoria_dpr.py` calcula fiablemente las métricas iniciales sesgadas. Adicionalmente, el tensor raíz de consumo de datos (`st_gcn_dataloader.py`) está funcional junto al modelo generador base.

### 🔄 Fases Pendientes y Pipeline Iterativo (Mejora Continua)

Las siguientes fases exhiben manifiestas diferencias entre su ideal técnico en el documento rector y la cruda implementación material existente en el código. Se enlistan a continuación trazando la hoja de ruta correctiva para ser atendida y perfeccionada:

- **Iteración para Fase 1 (Transición de Síntesis Estadística a MANO / SMPL-X)**
  > *Estado Factual:* Actualmente, la base temporal de tu script sintético de relleno (`generate_synthetic_landmarks.py`) **NO** emplea un andamiaje jerarquizado oficial de `MANO` (Kinema). En lugar de ello aproxima tensores agregando ruido pseudo-Gausiano sobre medias posicionales arrojadas por estequiometría superficial desde `MediaPipe`.
  > *Mejora a Pipeline:* Modificar e integrar el core framework `smplx` en PyTorch. Renderizar de base mallas topológicas manipulando asertivamente los parámetros esqueléticos de _Pose_ $(\theta)$ y de _Forma_ anatómica $(\beta)$ dotando de hiperrealismo orgánico a los landmarks sintéticos en piel oscura.

- **Iteración para Fase 1.1 (Reestructuración Hiper-Secuencial $T=1 \to T=10$)**
  > *Estado Factual:* El `Dataloader` traga todo el modelado como un vector estático para instantes dislocados y ahueca el _frame time_ a uno (expansión manual $T=1$).
  > *Mejora a Pipeline:* Modificar en tiempo de pre-procesado las secuencias inyectando micro-temblores dinámicos empleando _Random Walk_ inter-frame interpolatorio sobre un mínimo de `10 frames`.

- **Iteración para Fase 3 (Adversarial Branching - *FairGenderGen*)**
  > *Estado Factual:* El código ST-GCN original ignora capas de castigo o inversión de gradiente algorítmico.
  > *Mejora a Pipeline:* Inyectar sobre el autómata extractor de `stgcn_model.py` una capa adicional conectada por medio de una capa inversora de retorno (*GRL - Gradient Reversal Layer*). Entrenaremos un módulo "adversario" intentando inferir fenotipos demográficos. Al fallar el GRL en reversa el modelo forzará al autómata central a ser cinemáticamente daltónico para extraer solo *landmarks funcionales*, neutralizando envenenamientos fenotípicos de forma profunda.

- **Iteración para Fase 4 (Concreción de Divergencia TVD - Total Variation Distance)**
  > *Estado Factual:* Careciste de implementación directa empírica de TVD.
  > *Mejora a Pipeline:* A través de derivadas espaciales de los puntos cinemático del frame, modelar perfiles probabilísticos de densidad poblacional aislando clases `Claro` y clases `Oscuro`. Cotejarlas analíticamente entre sí midiendo traslape en áreas para obtener coeficientes reales TVD.
