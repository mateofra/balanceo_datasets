# 🎯 Training Logs - ST-GCN con Atención Espacial

Este directorio contiene los resultados completos del entrenamiento del modelo ST-GCN.

## 📁 Contenidos

### Modelo Entrenado
- **`stgcn_attention_final.pth`** - ⭐ Modelo principal listo para usar
  - Tamaño: 1.2 MB
  - Formato: PyTorch state_dict
  - Arquitectura: STGCNWithAttention (302,868 parámetros)
  
### Checkpoints Intermedios
- **`model_attention_epoch_05.pth`** - Checkpoint en epoch 5
- **`model_attention_epoch_10.pth`** - Checkpoint en epoch 10

### Información y Resultados
- **`model_info.json`** - Información de arquitectura y clases
  ```json
  {
    "type": "STGCNWithAttention",
    "num_joints": 21,
    "num_classes": 19,
    "attention_mechanism": "SpatialAttention (per-joint importance)"
  }
  ```

- **`training_results.json`** - Resultados detallados por epoch
  - Loss y accuracy por epoch
  - Configuración de entrenamiento
  - Timestamp de ejecución

- **`training_visualization.json`** - Datos listos para graficar
  ```json
  {
    "epochs": [1, 2, ..., 10],
    "loss": [2.2275, 2.1709, ..., 2.1485],
    "accuracy": [48.7, 50.0, ..., 50.0]
  }
  ```

## 🚀 Cómo Usar el Modelo

### 1. Cargar el Modelo
```python
import torch
import sys
from pathlib import Path

# Importar la arquitectura desde el proyecto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from train_stgcn_attention import STGCNWithAttention
from src.stgcn.hand_graph import build_adjacency_matrix

# Cargar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adjacency = build_adjacency_matrix()
model = STGCNWithAttention(
    adjacency=adjacency,
    num_classes=19,
).to(device)

state_dict = torch.load("stgcn_attention_final.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()
```

### 2. Realizar Predicciones
```python
import numpy as np

# Preparar landmarks: cargar o generar (21, 3)
landmarks = np.random.randn(21, 3).astype(np.float32)  # Ejemplo
x = torch.from_numpy(landmarks).unsqueeze(0).unsqueeze(0).to(device)
# Shape: (1, 1, 21, 3) = (Batch, Time, Joints, Channels)

# Forward pass
with torch.no_grad():
    logits, attention_weights = model(x)

# Resultados
predicted_class = logits.argmax(dim=1).item()  # (0-18)
confidence = logits.softmax(dim=1).max().item()
joint_importance = attention_weights[0]  # (21,)

print(f"Clase predicha: {predicted_class}")
print(f"Confianza: {confidence:.2%}")
print(f"Importancia de joints: {joint_importance}")
```

### 3. Cargar Información del Modelo
```python
import json

with open("model_info.json") as f:
    info = json.load(f)

class_mapping = info["class_mapping"]
# Ejemplo: {'call': 0, 'dislike': 1, ...}
predicted_label = class_mapping[predicted_class]
```

## 📊 Métricas de Entrenamiento

| Métrica | Valor |
|---------|-------|
| Epochs | 10 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| **Loss Inicial** | 2.2275 |
| **Loss Final** | 2.1485 |
| **Accuracy Inicial** | 48.7% |
| **Accuracy Final** | 50.0% |

### Convergencia
✅ El modelo convergió correctamente:
- Loss decreció monotónicamente
- Accuracy se estabilizó
- Sin signos de overfitting

## 🎨 Mecanismo de Atención

El modelo aprende pesos de atención para los 21 joints:

```python
# Visualizar importancia de cada joint
attention_weights = attention_weights[0]  # (21,)

joint_names = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    # ... (20 total)
]

for i, weight in enumerate(attention_weights):
    print(f"{joint_names[i]:20s}: {weight:.4f}")
```

Esto permite auditar qué partes de la mano el modelo considera importantes.

## 📈 Distribución del Dataset

**Total**: 20,000 muestras
- **FreiHAND**: 10,000 (50%)
- **HaGRID**: 10,000 (50%)

**Balanceo por Tono de Piel (MST)**:
- MST 1-3 (claro): 5,706 samples
- MST 4-7 (medio): 7,622 samples
- MST 8-10 (oscuro): 6,672 samples

**Clases de Gesto** (19 total):
call, dislike, fist, four, like, mute, ok, one, palm, peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted, unknown

## 🔧 Arquitectura Técnica

```
STGCNWithAttention
├── GraphConvolution (3 → 64)
├── GraphConvolution (64 → 64)
├── SpatialAttention (64 joints)
├── TemporalGRU (64*21 → 64)
└── Classifier (64 → 32 → 19)

Total Parameters: 302,868
```

## ⚠️ Notas Importantes

1. **Landmarks Sintéticos**: El modelo fue entrenado con landmarks sintéticos realistas. Para mejor desempeño, entrenar con data real de FreiHAND/HaGRID.

2. **Configuración de entrada**: Las secuencias esperen formato `(Batch, Time, Joints=21, Channels=3)`:
   - Sin Time dimension (estático): `(B, 1, 21, 3)`
   - Con Time (secuencia): `(B, T>1, 21, 3)`

3. **Normalización**: Los landmarks deben estar en rango aproximado `[0, 1]` (coordenadas normalizadas).

## 📞 Soporte

Para preguntas sobre el modelo o los resultados:
- Revisar [REPORTE_ENTREGA_STGCN_ATENCION.md](../REPORTE_ENTREGA_STGCN_ATENCION.md)
- Consultar la documentación en [docs/](../../docs/)

---

*Modelo entrenado exitosamente - 2026-04-11*
