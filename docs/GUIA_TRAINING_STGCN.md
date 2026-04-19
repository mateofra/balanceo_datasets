# Guía: Entrenar ST-GCN con Dataset Balanceado por MST

Para preparar la entrada de datos, descarga, extracción y balanceo, usa [README-Datasets.md](../README-Datasets.md) como referencia canónica.

## Resumen RÃ¡pido

```bash
# 1. Usar manifiesto generado + landmarks preprocesados
output/train_manifest_stgcn_fixed.csv
data/processed/landmarks/freihand_*.npy

# 2. Importar dataloader
from src.st_gcn_dataloader import create_dataloaders

# 3. Crear dataloaders
loaders = create_dataloaders(
    manifest_csv="output/train_manifest_stgcn_fixed.csv",
    batch_size=32,
    normalize=True,
    balance_by_mst=True  # Importante: balancea por tono de piel
)

# 4. Entrenar modelo ST-GCN
# - Input shape: (Batch, Channels=3, Frames=T, Vertices=21)
# - Ver src/stgcn/train_stgcn_example.py para ejemplo completo
```

---

## Estructura Completa

### Inputs Disponibles

1. **Manifiesto ST-GCN reparado** (`output/train_manifest_stgcn_fixed.csv`)
   - Contiene 10,000 muestras (FreiHAND) con landmarks disponibles
   - Columnas clave:
     - `path_landmarks`: ruta a archivo .npy (21Ã—3 coordinates)
     - `label`: gesto (actualmente solo 'unknown' para FreiHAND)
     - `condition`: tono de piel (**claro**, **medio**, **oscuro**)
     - `mst`: nivel de tono 1-10
     - `dataset`: fuente (freihand)

   ```csv
   sample_id,path_landmarks,label,condition,dataset,mst,mst_origin,split
   freihand_00000000,data/processed/landmarks/freihand_00000000.npy,unknown,medio,freihand,5,imputed,train
   freihand_00000001,data/processed/landmarks/freihand_00000001.npy,unknown,claro,freihand,4,imputed,train
   ...
   ```

2. **Landmarks preprocesados** (`data/processed/landmarks/freihand_*.npy`)
   - 32,560 archivos .npy (uno por sample)
   - Shape: (21, 3) - 21 hand joints, 3 coordinates (x, y, z)
   - Valores normalizados [0, 1] (de MediaPipe)

### Dataloader Personalizado

**Archivo**: `src/stgcn/st_gcn_dataloader.py`

**Features**:
- âœ… Carga landmarks desde .npy
- âœ… NormalizaciÃ³n por muÃ±eca (opcional)
- âœ… Balanceo por tono MST (weighted sampling)
- âœ… Compatible con PyTorch DataLoader

**Uso**:

```python
from src.st_gcn_dataloader import create_dataloaders

loaders = create_dataloaders(
    manifest_csv="output/train_manifest_stgcn_fixed.csv",
    batch_size=32,
    num_workers=0,
    normalize=True,        # Centra landmarks en muÃ±eca
    balance_by_mst=True,   # Usa WeightedRandomSampler por tono
)

train_loader = loaders["train"]
dataset = loaders["dataset"]
num_classes = loaders["num_classes"]
gesture_to_label = loaders["gesture_to_label"]

# Iterar batches
for batch in train_loader:
    landmarks = batch["landmarks"]  # Shape: (BS, 21, 3)
    labels = batch["label"]         # Shape: (BS,)
    mst = batch["mst"]              # Shape: (BS,) - nivel 1-10
    condition = batch["condition"]  # Lista: ["claro", "medio", "oscuro"]
    dataset = batch["dataset"]      # Lista: ["freihand"]
```

### Formato ST-GCN

ST-GCN espera: **(Batch, Channels=3, Frames=T, Vertices=21)**

**ConversiÃ³n**:

```python
# Batch del dataloader: (BS, 21, 3)
landmarks = batch["landmarks"]

# Convertir a formato ST-GCN: (BS, 3, T, 21)
# Si T=1 frame:
x = landmarks.permute(0, 2, 1).unsqueeze(2)  # (BS, 3, 1, 21)

# Si T>1 frames (secuencia temporal):
# landmarks_seq shape: (BS, T, 21, 3)
x = landmarks_seq.permute(0, 3, 1, 2)  # (BS, 3, T, 21)

# Forward pass
logits = st_gcn_model(x)  # (BS, num_classes)
```

---

## Ejemplo Completo de Training

**Script**: `src/stgcn/train_stgcn_example.py`

```bash
uv run python src/stgcn/train_stgcn_example.py
```

**QuÃ© hace**:
1. Carga dataset (10,000 muestras FreiHAND)
2. Crea modelo ST-GCN simplificado
3. Entrena 10 epochs
4. Guarda checkpoints cada 5 epochs
5. Registra mÃ©tricas en TensorBoard

**Output**:

```
output/training_logs/
â”œâ”€â”€ tensorboard/
â”‚   â””â”€â”€ events.out.tfevents  # Logs para TensorBoard
â”œâ”€â”€ model_epoch_5.pth
â”œâ”€â”€ model_epoch_10.pth
â””â”€â”€ model_final.pth
```

**Ver TensorBoard**:

```bash
tensorboard --logdir=output/training_logs/tensorboard
```

---

## Balanceo por Tono de Piel (MST)

### CÃ³mo Funciona

Por defecto, `balance_by_mst=True` en el dataloader usa **WeightedRandomSampler**:

```python
# Calcula peso inverso de cada tono
mst_counts = {1: 100, 2: 150, 3: 120, ..., 10: 200}
mst_weights = {mst: total / count for mst, count in mst_counts.items()}

# Asigna pesos a cada sample segÃºn su MST
weights = [mst_weights[sample.mst] for sample in dataset]

# Sampler: equilibra tonosextrem (1,2,3,10) vs centrales (5,6,7)
```

### Resultados

Dataset balanceado:

```
CondiciÃ³n (tono agrupado):
  - Claro (MST 1-3): 33%
  - Medio (MST 4-7): 33%
  - Oscuro (MST 8-10): 34%

Niveles MST (distribuidos):
  - MST 1: 1,873 muestras (sobrerrepresentado)
  - MST 2: 1,945
  - MST 3: 1,888
  - MST 10: 3,407 (extremo oscuro, extra)
  - ...
```

### Validar Balanceo

```python
from collections import Counter

# Contar distribuciones durante training
mst_counts = Counter()
condition_counts = Counter()

for batch in train_loader:
    mst_counts.update(batch["mst"].tolist())
    condition_counts.update(batch["condition"])

print("MST distribution:", dict(mst_counts))
print("Condition distribution:", dict(condition_counts))

# DeberÃ­a estar casi uniforme (balanceado)
```

---

## IntegraciÃ³n con Modelo ST-GCN Real

Si usas un modelo ST-GCN existente (ej: [yysijie/st-gcn](https://github.com/yysijie/st-gcn)):

### 1. Adaptar entradas

```python
# Tu modelo ST-GCN
model = st_gcn.Model(...)

# DataLoader personalizado
train_loader = create_dataloaders(
    manifest_csv="output/train_manifest_stgcn_fixed.csv",
    batch_size=64,
    balance_by_mst=True
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        x = batch["landmarks"].permute(0, 2, 1).unsqueeze(2)  # (BS, 3, 1, 21)
        y = batch["label"]
        
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
```

### 2. Registrar balanceo por MST

```python
# Logger personalizado para validar fairness por tono
class MST_Logger:
    def log_metrics_by_mst(self, batch, preds, losses):
        for mst in range(1, 11):
            mask = batch["mst"] == mst
            acc_by_mst = (preds[mask] == batch["label"][mask]).mean()
            loss_by_mst = losses[mask].mean()
            
            # Registrar en tensorboard / logger
            print(f"MST {mst}: acc={acc_by_mst:.1%}, loss={loss_by_mst:.4f}")
```

### 3. Asegurar equidad

```python
# Validar que el modelo no sea sesgado por tono
# Loss + accuracy por MST deberÃ­a ser similar

results_by_mst = defaultdict(lambda: {"acc": [], "loss": []})

for batch in val_loader:
    x = batch["landmarks"].permute(0, 2, 1).unsqueeze(2)
    y = batch["label"]
    mst = batch["mst"]
    
    logits = model(x)
    preds = logits.argmax(dim=1)
    
    for m in range(1, 11):
        mask = mst == m
        if mask.any():
            results_by_mst[m]["acc"].append((preds[mask] == y[mask]).mean().item())
            
print("Accuracy por MST (deberÃ­a ser uniforme):")
for mst, metrics in sorted(results_by_mst.items()):
    avg_acc = np.mean(metrics["acc"]) if metrics["acc"] else 0
    print(f"MST {mst:2d}: {avg_acc:.1%}")
```

---

## PrÃ³ximos Pasos (Roadmap)

### Corto plazo (Ahora disponible)
- âœ… Dataloader con landmarks de FreiHAND (10K)
- âœ… Balanceo por MST
- âœ… Script de training ST-GCN simplificado

### Mediano plazo (Requiere imÃ¡genes HaGRID)
- â³ Procesar landmarks HaGRID con MediaPipe
- â³ Generar CSV de MST para todas las imÃ¡genes
- â³ Manifiesto ST-GCN con 20K muestras FreiHAND + HaGRID

### Largo plazo (EvaluaciÃ³n en producciÃ³n)
- â³ Testing fairness: evaluar modelo por tono MST
- â³ Comparar predicciones: MST 1 vs MST 10
- â³ Metricas de equidad: Â¿accuracy similar para todos tonos?

---

## Troubleshooting

### Problema: "Path not found" al cargar landmarks

**Causa**: Rutas en manifiesto no corresponden a archivos reales.

**SoluciÃ³n**:

```bash
# Ejecutar reparaciÃ³n de rutas
uv run python repair_manifest.py
# Genera: output/train_manifest_stgcn_fixed.csv
# Usa este en tu dataloader
```

### Problema: Out of Memory durante training

**SoluciÃ³n**:
```python
loaders = create_dataloaders(
    manifest_csv="output/train_manifest_stgcn_fixed.csv",
    batch_size=16,  # Reducir de 32 a 16
    num_workers=0,  # Mantener en 0 para Windows
)
```

### Problema: Accuracy 100% (overfitting)

**Causa**: Dataset FreiHAND solo tiene un gesto ("unknown").

**SoluciÃ³n**: Esperar a procesar HaGRID con mÃºltiples gestos.

---

## Referencias

- **ST-GCN**: https://github.com/yysijie/st-gcn
- **MediaPipe Hand Landmarker**: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- **MST Scale**: https://www.monkskintonescale.org/
- **FreiHAND**: https://freihanda.is.tue.mpg.de/
- **HaGRID**: https://hagrid.hasty.ai/

---

## Contacto

Para preguntas sobre el pipeline de balanceo:
- Ver: `PIPELINE_MST_STGCN.md`
- Scripts: `src/`
- Datos: `output/`

