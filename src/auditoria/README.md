# src/auditoria

Scripts para auditoria de equidad y generacion de graficas de informe.

## Contenido

- `auditoria_final.py`: calcula metricas de auditoria sobre el split test.
- `auditar_mapeo_freihand_rgb.py`: verifica mapeo 1:1 entre `training_xyz.json` y `training/rgb`, y exporta subset canonico + extras.
- `generar_graficas_informe.py`: genera graficas estaticas para el informe final.
- `generar_graficas_avanzadas.py`: genera visualizaciones avanzadas de analisis.

## Uso

Ejecutar desde la raiz del repositorio, por ejemplo:

```bash
uv run python src/auditoria/auditoria_final.py
uv run python src/auditoria/auditar_mapeo_freihand_rgb.py
```
