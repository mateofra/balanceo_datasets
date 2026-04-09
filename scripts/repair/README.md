# Repair

Scripts para reparar manifiestos y consistencia de datos.

## Scripts

| Script | Uso |
|--------|-----|
| `repair_manifest.py` | Repara manifiestos CSV, resuelve rutas y preserva trazabilidad. |
| `check_npy_shape.py` | Verifica shapes de archivos `.npy`. |

## Como se elaboraron

- `repair_manifest.py` nacio para arreglar discrepancias entre nombres de archivo, sample_id y rutas en Windows.
- Se extendio para conservar `landmark_quality` y no perder el contexto de origen de cada muestra.

## Errores y soluciones

- Rutas HaGRID no encontradas: se añadio resolucion multi-convencion por `sample_id` y escaneo recursivo.
- Manifestos sin calidad: se incorporo `landmark_quality` para distinguir `real_3d_freihand`, `annotation_2d_projected` y `synthetic_gesture_mean`.

## Uso

```bash
python repair_manifest.py input.csv --output output.csv
python check_npy_shape.py data/processed/landmarks/
```
