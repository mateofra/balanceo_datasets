# 📤 Salidas del Proyecto

Este directorio contiene todos los outputs generados por los scripts.

## 📂 Subdirectorios

### manifests/
Manifiestos CSV para training:
- `train_manifest_stgcn*.csv` - Manifiestos para ST-GCN
- `train_manifest_balanceado*.csv` - Manifiestos con balanceo por MST

### reports/
Reportes y análisis:
- `reporte_graficos_balanceo.md` - Reporte con visualizaciones
- `diagnostico_normalizacion.json` - Diagnóstico de normalización
- `resumen_balanceo*.json` - Resúmenes de balanceo por dataset

### graphics/
Gráficos y visualizaciones (PNG, PDF):
- Distribuciones por tono MST
- Proporciones FreiHAND vs HaGRID
- Histogramas de clases

### training_logs/
Logs de entrenamiento y checkpoints:
- Logs de tensorboard
- Checkpoints de modelos
- Métricas de training

## 🚀 Uso

Los scripts escriben automáticamente aquí:
```bash
python scripts/generate/generar_graficos_balanceo.py
# Salida: output/graphics/*.png

python scripts/diagnosis/diagnosticar_normalizacion.py
# Salida: output/reports/diagnostico_normalizacion.json
```

## 📌 Nota

Algunos archivos (.npy grandes) pueden estar en `.gitignore` pero los manifiestos CSV deben commitarse.
