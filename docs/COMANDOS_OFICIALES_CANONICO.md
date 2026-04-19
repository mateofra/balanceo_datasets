# Comandos oficiales (flujo canónico FreiHAND)

Este flujo asegura que FreiHAND use el subconjunto canónico 1:1 con `training_xyz.json` y que reportes/entrenamiento sean consistentes.

Para el flujo de descarga, extracción y balanceo de datasets, consulta [README-Datasets.md](../README-Datasets.md).

## 1) Auditoría de mapeo canónico FreiHAND

```bash
uv run python src/auditoria/auditar_mapeo_freihand_rgb.py
```

Artefactos clave:
- `output/auditoria/freihand_rgb_mapping_summary.json`
- `output/auditoria/freihand_rgb_canonical_manifest.csv`
- `output/auditoria/freihand_rgb_extra_manifest.csv`

## 2) Balanceo usando subset canónico

```bash
uv run python src/balancer/balancear_freihand_hagrid.py \
  --freihand-canonical-rgb-manifest output/auditoria/freihand_rgb_canonical_manifest.csv \
  --output-csv output/train_manifest_balanceado_freihand_hagrid.csv \
  --output-summary output/resumen_balanceo_freihand_hagrid.json
```

## 3) Métricas/figura de medidas de balanceo (consistentes)

```bash
uv run python src/scripts/generate/generar_grafica_medidas_balanceo.py \
  --summary-after output/resumen_balanceo_freihand_hagrid.json \
  --manifest-active output/train_manifest_balanceado_freihand_hagrid.csv \
  --freihand-canonical-rgb-manifest output/auditoria/freihand_rgb_canonical_manifest.csv \
  --output graficos/stgcn_atencion_mst/medidas_balanceo_con_ejemplos_canonico.png
```

## 4) Comparativa MST HaGRID vs FreiHAND (consistente)

```bash
uv run python src/scripts/generate/generar_grafica_mst_hagrid_freihand.py \
  --freihand-canonical-rgb-manifest output/auditoria/freihand_rgb_canonical_manifest.csv \
  --output-plot graficos/stgcn_atencion_mst/distribucion_mst_hagrid_vs_freihand_canonico.png \
  --output-summary graficos/stgcn_atencion_mst/distribucion_mst_hagrid_vs_freihand_canonico.json
```

## Nota

Si no existe `output/auditoria/freihand_rgb_canonical_manifest.csv`, los scripts usan fallback a comportamiento previo. Para consistencia de resultados, ejecutar siempre primero el paso 1.
