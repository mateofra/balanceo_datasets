# graficos

Carpeta para salidas visuales de la composicion del dataset balanceado.

## Uso

Generar graficos desde manifiesto + resumen:

```powershell
uv run python src/generar_graficos_balanceo.py \
  --manifest-csv csv/train_manifest_balanceado_freihand_hagrid_smoke2.csv \
  --summary-json csv/resumen_balanceo_freihand_hagrid_smoke2.json \
  --output-dir graficos/balanceo_smoke2
```

Los PNG generados ayudan a validar:

- balance por fuente
- balance por bloque MST
- distribucion por nivel MST
- distribucion por gesto
