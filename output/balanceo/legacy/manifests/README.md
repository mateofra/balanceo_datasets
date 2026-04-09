# 📋 Manifiestos

Archivos CSV con manifiestos de training y evaluación.

## Archivos Principales

| Archivo | Descripción |
|---------|-------------|
| `train_manifest_stgcn.csv` | Manifiesto para ST-GCN con 10K muestras |
| `train_manifest_stgcn_fixed.csv` | Versión reparada del anterior |
| `train_manifest_balanceado_freihand_hagrid.csv` | Balanceado FreiHAND + HaGRID |
| `train_manifest_balanceado_tonos_demo.csv` | Demo con balanceo MST |

## Estructura CSV

Columnas típicas:
- `path` - Ruta a archivo .npy de landmarks
- `label` - Etiqueta de gesto
- `source` - Fuente (FreiHAND/HaGRID)
- `mst` - Tono MST (1-10)
- `condition` - Categoría (claro/medio/oscuro)
- `split` - (train/val/test)

## Uso

```python
import pandas as pd

manifest = pd.read_csv('manifests/train_manifest_stgcn.csv')
print(f"Total samples: {len(manifest)}")
print(manifest['condition'].value_counts())
```

## Generación

Los manifiestos se generan con:
```bash
python scripts/training/generar_secuencias_stgcn.py
```
