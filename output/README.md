# Salidas del proyecto

Estructura oficial de artefactos por finalidad.

## Subdirectorios oficiales

- `output/balanceo/`: manifiestos balanceados y resumentes de cuotas MST.
- `output/training/`: manifiestos ST-GCN, historiales y checkpoints.
- `output/auditoria/`: resultados de auditoria DPR/TVD y figuras derivadas.

## Legacy

- `output/balanceo/legacy/` y `output/training/legacy/` contienen artefactos historicos reubicados.
- Se conservan solo para trazabilidad y comparacion, no como fuente principal.

## Regla de oro

- Nuevos artefactos deben escribirse en una de las tres carpetas oficiales.
- Evitar dejar archivos sueltos en la raiz de `output/`.
