# Auditoria DPR

Graficas generadas a partir de los resultados de auditoria sobre HaGRID real.

## Contenido

- accuracy por bloque MST
- TVD canonico por par de bloques
- figura resumen en PNG

## Uso

Generar la figura con:

```bash
uv run python scripts/generate/generar_grafica_auditoria_dpr.py
```

La grafica ayuda a interpretar si la diferencia entre bloques MST esta en accuracy global o en la distribucion de errores por clase.