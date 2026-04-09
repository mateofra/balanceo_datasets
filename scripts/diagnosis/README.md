# Diagnosis

Scripts de verificacion usados para detectar problemas reales de la tuberia.

## Scripts

| Script | Uso |
|--------|-----|
| `diagnosticar_normalizacion.py` | Verifica normalizacion de landmarks y produce JSON. |
| `diagnose_paths.py` | Comprueba rutas de datos y manifiestos. |
| `verificar_landmarks.py` | Valida integridad y shape de landmarks. |
| `verificar_secuencias.py` | Comprueba secuencias generadas para ST-GCN. |

## Hallazgos resueltos

- Se detecto que HaGRID no tenia raw images en el workspace, solo anotaciones.
- Se detectaron desajustes entre sample_id, rutas Windows y rutas relativas.
- Se confirmo que algunas secuencias eran casi estaticas y generaban una tarea trivial.

## Salidas

- Reportes JSON y logs de diagnostico: `output/reports/`
- Resumenes puntuales: `scripts/diagnosis/diagnostico_normalizacion.json`
