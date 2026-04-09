# stgcn_atencion_mst

Visualizaciones de atencion espacial por nodo del modelo ST-GCN supervisado entrenado.

## Que contiene

- `claro_stgcn_atencion.png`: muestra representativa del tono MST claro.
- `medio_stgcn_atencion.png`: muestra representativa del tono MST medio.
- `oscuro_stgcn_atencion.png`: muestra representativa del tono MST oscuro.
- `claro_stgcn_composicion.png`: composicion imagen original + esqueleto con atencion (tono claro).
- `medio_stgcn_composicion.png`: composicion imagen original + esqueleto con atencion (tono medio).
- `oscuro_stgcn_composicion.png`: composicion imagen original + esqueleto con atencion (tono oscuro).
- `metadata_atencion_stgcn.json`: metadatos de muestras, predicciones y pesos de atencion por nodo.

## Como se genero

Desde la raiz del repo:

```bash
uv run python scripts/generate/visualizar_atencion_stgcn_mst.py
```

Por defecto el script exige imagen original real para cada tono y falla si no la encuentra.
Si tus imagenes estan en otra ruta, define `--image-roots`.

```bash
uv run python scripts/generate/visualizar_atencion_stgcn_mst.py \
  --image-roots data/raw/images datasets/hagrid_dataset
```

Solo para depuracion (permite fallback sin imagen real):

```bash
uv run python scripts/generate/visualizar_atencion_stgcn_mst.py --allow-missing-originals
```

Parametros comunes:

```bash
uv run python scripts/generate/visualizar_atencion_stgcn_mst.py \
  --checkpoint output/training/best_stgcn_supervisado.pth \
  --manifest-csv output/training/train_manifest_stgcn_fixed.csv \
  --seq-dir data/processed/secuencias_stgcn \
  --output-dir graficos/stgcn_atencion_mst
```

## Interpretacion visual

- Cada nodo (0..20) representa un landmark de mano.
- El brillo del nodo corresponde a la atencion espacial normalizada del modelo para esa muestra.
- Las aristas del esqueleto tambien aumentan brillo/grosor segun la media de atencion de sus nodos extremos.
- Si usas `--allow-missing-originals` y falta imagen local, la composicion muestra un panel informativo en su lugar.
