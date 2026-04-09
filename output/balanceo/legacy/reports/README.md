# 📊 Reportes

Reportes JSON y markdown con análisis de datos.

## Archivos

| Archivo | Contenido |
|---------|-----------|
| `diagnostico_normalizacion.json` | Análisis de normalización de landmarks |
| `resumen_balanceo_freihand_hagrid.json` | Reporte de balanceo combinado |
| `resumen_balanceo_tonos_demo.json` | Reporte demo |
| `reporte_graficos_balanceo.md` | Documento con gráficos embebidos |

## Contenido Típico

Los JSON incluyen:
- Estadísticas por fuente (FreiHAND, HaGRID)
- Distribución por tono MST
- Factores de balanceo aplicados
- Conteos de muestras
- Ratios claro:medio:oscuro

## Usar Reportes

```python
import json

with open('reports/resumen_balanceo_freihand_hagrid.json') as f:
    summary = json.load(f)
    
print(f"Muestras FreiHAND: {summary['freihand_count']}")
print(f"Muestras HaGRID: {summary['hagrid_count']}")
print(f"Distribucion MST: {summary['mst_distribution']}")
```

## Generación

```bash
python scripts/diagnosis/diagnosticar_normalizacion.py
python scripts/generate/generar_graficos_balanceo.py
```
