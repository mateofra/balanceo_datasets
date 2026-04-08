# 🔍 Scripts de Diagnóstico

Scripts para validar y diagnosticar el estado de los datos y la pipeline.

## Scripts

| Script | Propósito |
|--------|-----------|
| `diagnosticar_normalizacion.py` | Verifica la normalización de landmarks y genera reporte JSON |
| `diagnose_paths.py` | Valida rutas de datos y manifiestos |
| `verificar_landmarks.py` | Inspecciona integridad de landmarks |
| `verificar_secuencias.py` | Valida secuencias generadas para ST-GCN |

## Uso

```bash
# Diagnóstico completo de normalización
python diagnosticar_normalizacion.py

# Verificar rutas
python diagnose_paths.py

# Inspeccionar landmarks
python verificar_landmarks.py

# Validar secuencias
python verificar_secuencias.py
```

## Salida

Los reportes se guardan típicamente en `output/reports/` formato JSON.
