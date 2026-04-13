# 🚀 Guía Rápida: 5 Pasos para Training ST-GCN

## ⏱️ Tiempo Total: ~10 minutos

---

## Paso 1: Verificar Instalación (1 min)

```bash
# Desde directorio stgcn/
uv run python scripts/validate_setup.py
```

**Resultado esperado:**
```
✓ Python 3.13+
✓ PyTorch instalado
✓ Manifiesto CSV accesible
✓ 10,000 landmarks .npy disponibles
✓ Setup validado correctamente
```

Si falla algo, ver sección "Troubleshooting" al final.

---

## Paso 2: Entender los Datos (2 min)

**Dataloader automáticamente:**
- ✅ Carga 21 landmarks de 3 coordinadas (xyz)
- ✅ Normaliza por muñeca (opcional)
- ✅ Balancea por tono MST (claro/medio/oscuro)
- ✅ Retorna batches listos para ST-GCN

**Formato de salida:**

```python
batch = {
    "landmarks":  torch.Size([32, 21, 3]),      # Hand joints
    "label":      torch.Size([32]),              # Gesture (int)
    "mst":        torch.Size([32]),              # Skin tone 1-10
    "condition":  ["claro", "medio", "oscuro"],  # Tone category
    "dataset":    ["freihand", ...],             # Source
}
```

**Convertir para ST-GCN:**

```python
# Input del dataloader: (BS, 21, 3)
x = batch["landmarks"].permute(0, 2, 1).unsqueeze(2)
# Output ST-GCN compatible: (BS, 3, 1, 21) ✅
```

---

## Paso 3: Ejecutar Training Básico (5 min)

```bash
# Opción A: Con configuración por defecto
uv run python scripts/train.py

# Opción B: Con configuración personalizada
uv run python scripts/train.py --config config/examples/small_dataset.yaml

# Opción C: Con argumentos CLI
uv run python scripts/train.py --batch-size 16 --num-epochs 5 --learning-rate 0.0005
```

**Durante training verás:**

```
Epoch   1/20 [================================] 313/313 | Loss: 0.456 | Acc: 87.3% | Time: 24s
Epoch   2/20 [================================] 313/313 | Loss: 0.321 | Acc: 90.1% | Time: 23s
Epoch   3/20 [================================] 313/313 | Loss: 0.245 | Acc: 92.3% | Time: 22s
...
```

---

## Paso 4: Validar Fairness por Tono (2 min)

Después del training, verifica que **no haya sesgo** por tono:

```bash
# Generar reporte de equidad
uv run python scripts/analyze_fairness.py logs/training_log_final.json
```

**Resultado esperado:**

```
FAIRNESS REPORT
===============

Accuracy por Tono MST:
  MST 1 (muy claro): 92.5%
  MST 2: 92.0%
  MST 3: 91.8%
  MST 4: 93.0%
  MST 5 (medio): 92.9%
  ...
  MST 10 (muy oscuro): 92.8%

Desviación estándar: 0.4% ✅ EXCELENTE (< 2%)
Conclusión: Modelo es EQUITATIVO entre tonos
```

**Si la desviación es > 5%** → El modelo tiene sesgo. Soluciones:
- Aumentar `BALANCE_BY_MST: true` en config
- Aumentar `NUM_EPOCHS` (más training)
- Revisar `LEARNING_RATE`

---

## Paso 5: Guardar Modelo y Resultados (1 min)

```bash
# Automático: El script guarda en logs/checkpoints/
ls -lh logs/checkpoints/
```

Output:
```
model_latest.pth     (modelo más reciente)
model_best.pth       (mejor accuracy en validación)
model_epoch_5.pth    (checkpoint cada 5 epochs)
model_final.pth      (modelo final)
```

**Para usar el modelo:**

```python
import torch

model = YourSTGCNModel()
checkpoint = torch.load("logs/checkpoints/model_final.pth")
model.load_state_dict(checkpoint)
model.eval()

# Inference
with torch.no_grad():
    landmarks = torch.randn(1, 21, 3)  # Input: 1 sample
    x = landmarks.permute(0, 2, 1).unsqueeze(2)  # (1, 3, 1, 21)
    output = model(x)  # Predicción
    print(output.argmax(dim=1))  # Clase predicha
```

---

## 🎯 Flujo Completo en Código

```python
# 1. Load data
from src.dataloader import create_dataloaders
loaders = create_dataloaders(
    manifest_csv="data/train_manifest_stgcn_fixed.csv",
    batch_size=32,
    balance_by_mst=True
)

# 2. Setup model
import torch
import torch.nn as nn

model = YourSTGCNModel(num_classes=loaders["num_classes"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 3. Training loop
for epoch in range(20):
    for batch in loaders["train"]:
        x = batch["landmarks"].permute(0, 2, 1).unsqueeze(2)  # (BS, 3, 1, 21)
        y = batch["label"]
        
        logits = model(x)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")

# 4. Save
torch.save(model.state_dict(), "model.pth")
```

---

## ⚙️ Configuraciones Rápidas

### Setup Mínimo (Prueba Rápida - 30 seg)

```bash
uv run python scripts/train.py \
  --batch-size 32 \
  --num-epochs 1 \
  --log-interval 10
```
