# Generate

Scripts para generar graficas y reportes reproducibles.

## Scripts

| Script | Uso |
|--------|-----|
| `src/balancer/generar_graficos_balanceo.py` | Grafica la composicion del dataset balanceado. |
| `generar_grafica_auditoria_dpr.py` | Genera la figura resumen de la auditoria DPR. |
| `generar_grafica_medidas_balanceo.py` | Compara distribucion original vs final y resume medidas activas de balanceo con subset canónico FreiHAND. |
| `generar_grafica_mst_hagrid_freihand.py` | Compara distribución MST HaGRID vs FreiHAND y permite filtrar FreiHAND por manifiesto canónico. |
| `visualizar_atencion_stgcn_mst.py` | Genera una imagen por tono MST con esqueleto y brillo por atencion espacial del ST-GCN. |
| `generar_grafica_stack_final.py` | Construye una grafica final con resultados de balanceo, entrenamiento, auditoria y representacion. |

## Como se elaboraron

- Las graficas de balanceo se basan en manifiestos balanceados y resument la distribucion por fuente, MST y gesto.
- La grafica de auditoria se construye sobre `output/auditoria/auditoria_dpr_resultados.csv` y resume accuracy por bloque MST junto con TVD por par.

## Errores y soluciones

- Se detecto que el TVD original no era canónico: se corrigio la formulacion y se separo en una grafica especifica de auditoria.
- Se evitó mezclar salidas de balanceo con salidas de auditoria creando carpetas separadas por finalidad.

## Uso

```bash
uv run python src/balancer/generar_graficos_balanceo.py --help
uv run python generar_grafica_auditoria_dpr.py
uv run python src/scripts/generate/generar_grafica_medidas_balanceo.py \
	--freihand-canonical-rgb-manifest output/auditoria/freihand_rgb_canonical_manifest.csv
uv run python src/scripts/generate/generar_grafica_mst_hagrid_freihand.py \
	--freihand-canonical-rgb-manifest output/auditoria/freihand_rgb_canonical_manifest.csv
uv run python src/scripts/generate/visualizar_atencion_stgcn_mst.py
uv run python src/scripts/generate/generar_grafica_stack_final.py
```
