# 🔧 Scripts Utilitarios

Este directorio contiene scripts ejecutables organizados por funcionalidad.

## 📂 Subdirectorios

### diagnosis/
Scripts para diagnóstico y verificación de datos:
- `diagnosticar_normalizacion.py` - Analiza la normalización de landmarks
- `diagnose_paths.py` - Verifica rutas de datos
- `verificar_landmarks.py` - Valida integridad de landmarks
- `verificar_secuencias.py` - Verifica secuencias de datos

### training/
Scripts para entrenamiento y visualización:
- `train_stgcn.py` - Entrena el modelo ST-GCN
- `generar_secuencias_stgcn.py` - Genera secuencias para ST-GCN
- `visualizar_pipeline.py` - Visualiza el pipeline de datos

### repair/
Scripts para reparación y mantenimiento:
- `repair_manifest.py` - Repara manifiestos dañados
- `check_npy_shape.py` - Verifica shapes de archivos .npy

### generate/
Scripts para generar reportes y visualizaciones:
- `generar_graficos_balanceo.py` - Genera gráficos de balanceo

## 🚀 Uso General

```bash
python scripts/diagnosis/diagnosticar_normalizacion.py
python scripts/training/train_stgcn.py
```
