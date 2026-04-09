# 📈 Gráficos

Visualizaciones generadas automáticamente.

## Contenido

Se generan gráficos PNG/PDF para:
- Distribución de tonos MST (claro/medio/oscuro)
- Proporción FreiHAND vs HaGRID
- Histogramas de clases
- Curvas de balanceo
- Heatmaps de sesgo

## Uso

```bash
# Generar todos los gráficos
python scripts/generate/generar_graficos_balanceo.py \
  --manifest output/manifests/train_manifest_stgcn.csv
```

## Ubicación

Los gráficos se guardan con fecha/hora:
- `output/graphics/` - Root
- `output/graphics/balanceo_tonos_demo_fullclass/` - Gráficos subdivididos

## Integración

Se pueden embeber en reportes:
```markdown
![Distribución MST](../graphics/mst_distribution.png)
```
