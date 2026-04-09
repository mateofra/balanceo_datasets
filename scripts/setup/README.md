# Setup

Scripts para preparar el entorno local en un clon nuevo.

## Script disponible

- `bootstrap_local_data.py`: crea directorios ignorados por git para datasets y datos locales, y verifica archivos clave esperados.
- `download_hagrid_kaggle.py`: descarga asistida de HaGRID por Kaggle CLI (con `--dry-run` por defecto) y preparación opcional de `datasets/ann_subsample`.

Comportamiento robusto:

- En `dry-run`, si no existe Kaggle CLI o credenciales, el script no falla; informa y termina en modo simulación.
- En `--execute`, la falta de Kaggle CLI/credenciales sí devuelve error para evitar estados ambiguos.
- Desde `main.py setup-data`, puedes decidir si ese error corta el flujo con `--strict-download`.
- El script intenta usar credenciales locales del proyecto en `secrets/kaggle/kaggle.json`.
- Si no existe ese archivo, crea automáticamente el directorio y un `kaggle.json` plantilla.
- Si el `kaggle.json` local está mal formado o incompleto, crea backup `kaggle.json.bak` y regenera plantilla.
- Puedes cargar credenciales desde terminal con argumentos `--kaggle-username` y `--kaggle-key`.
- También puedes usar `--ask-credentials` para que el script pida username/key por prompt interactivo.

## Uso

```bash
uv run python scripts/setup/bootstrap_local_data.py
```

Solo verificacion:

```bash
uv run python scripts/setup/bootstrap_local_data.py --verify-only
```

Descarga asistida de HaGRID (simulación):

```bash
uv run python scripts/setup/download_hagrid_kaggle.py
```

Descarga real y preparación de anotaciones:

```bash
uv run python scripts/setup/download_hagrid_kaggle.py --execute --prepare-ann-subsample
```

Cargar credenciales desde terminal (sin editar archivo manualmente):

```bash
uv run python scripts/setup/download_hagrid_kaggle.py \
	--kaggle-username TU_USUARIO \
	--kaggle-key TU_KEY \
	--execute
```

Solicitar credenciales por prompt interactivo:

```bash
uv run python scripts/setup/download_hagrid_kaggle.py --ask-credentials --execute
```
