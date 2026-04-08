# 📊 Scripts de Generación

Scripts para generar reportes, métricas y visualizaciones.

## Scripts

| Script | Propósito |
|--------|-----------|
| `generar_graficos_balanceo.py` | Genera gráficos de distribución del balanceo por tono MST |

## Uso

```bash
# Generar gráficos de balanceo
python generar_graficos_balanceo.py \
  --manifest output/manifests/train_manifest_stgcn.csv \
  --output output/graphics/
```

## Salida

Los gráficos se guardan en `output/graphics/` con formatos PNG/PDF.

## Visualizaciones

- Distribución por tono MST (claro/medio/oscuro)
- Proporción FreiHAND vs HaGRID
- Histogramas de clases
