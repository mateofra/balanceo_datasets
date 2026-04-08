# 🔧 Scripts de Reparación

Scripts para reparar y mantener integridad de datos.

## Scripts

| Script | Propósito |
|--------|-----------|
| `repair_manifest.py` | Repara manifiestos CSV dañados o inconsistentes |
| `check_npy_shape.py` | Verifica y reporta shapes de archivos .npy |

## Uso

```bash
# Reparar manifiesto
python repair_manifest.py input.csv --output output.csv

# Verificar shapes de archivos .npy
python check_npy_shape.py data/processed/landmarks/
```

## Validaciones

- Verifica que todos los paths existan
- Valida formato CSV
- Confirma integridad de shapes en .npy
