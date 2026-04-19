# Auditoria

Scripts para evaluar sesgo y rendimiento por bloques MST.

## Scripts

- `auditoria_dpr.py`: calcula accuracy por bloque, DPR y TVD canonico por pares.

## Uso

```bash
uv run python src/scripts/auditoria/auditoria_dpr.py
```

## Salidas

- `output/auditoria/auditoria_dpr_resultados.csv`
- metricas en consola (DPR, TVD por par y TVD maximo)
