# graficos

Carpeta para salidas visuales del proyecto, separadas por objetivo.

## Subcarpetas

- `balanceo_tonos_demo/` y variantes: composicion de los datasets balanceados.
- `auditoria_dpr/`: resumen visual de accuracy por bloque MST y TVD canónico.

## Uso

Graficos de balanceo:

```powershell
uv run python src/balancer/generar_graficos_balanceo.py \
  --manifest-csv csv/train_manifest_balanceado_freihand_hagrid_smoke2.csv \
  --summary-json csv/resumen_balanceo_freihand_hagrid_smoke2.json \
  --output-dir graficos/balanceo_smoke2
```

Grafica de auditoria:

```powershell
uv run python scripts/generate/generar_grafica_auditoria_dpr.py
```

## Lo que validan

- balance por fuente
- balance por bloque MST
- distribucion por nivel MST
- distribucion por gesto
- coherencia entre accuracy global y divergencia por clases de error
