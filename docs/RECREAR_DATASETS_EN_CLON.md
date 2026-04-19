# Recrear y Descargar Datasets en un Clon Nuevo

Este documento describe **la forma correcta e irrompible** de dejar operativo un entorno nuevo, descargando y armando la estructura de carpetas cuando `datasets/`, `data/` y `stgcn/data/` no vienen incluidos en git (por `.gitignore`).

> [!WARNING]  
> **REGLA DE ORO:** **Siempre** ejecuta los comandos desde la carpeta **RAÍZ** del repositorio (donde se encuentra `pyproject.toml` y `src/cli/main.py`). Ejecutar los scripts directamente desde adentro de `src/datasets/` o `docs/` romperá las rutas y las importaciones.

---

## 1. Crear Estructura Local de Carpetas (Bootstrap)

Abre una terminal en la raíz del repositorio y ejecuta el orquestador principal:

```bash
uv run python src/cli/main.py setup-data
```

**¿Qué hace esto?**
- Crea automáticamente toda la estructura esperada: `datasets/`, `data/raw/`, `data/processed/` y `stgcn/data/`.
- Prepara los placeholders internos ignorados por Git.
- Realiza una validación pasiva (en modo *dry-run*) confirmando qué componentes faltan.

---

## 2. Descarga Automatizada de HaGRID (Vía Kaggle)

El pipeline soporta descarga automática del dataset HaGRID directo desde Kaggle, extrayendo las anotaciones JSON y re-colocándolas en la carpeta correcta (`datasets/ann_subsample/`).

### Requisito Previo (Credenciales de Kaggle)
1. Consigue tu archivo `kaggle.json` desde the sitio web de Kaggle (Settings > Create New Token).
2. Debes tener ese nombre de usuario y tu API Key listas.
3. El propio pipeline te los pedirá de manera interactiva o podrás inyectarlos creando el archivo local `secrets/kaggle/kaggle.json`.

### Comando de Execución Real (Siempre Funciona)

Para forzar la descarga de los gigabytes reales y la reorganización de carpetas de anotación:

```bash
uv run python src/cli/main.py setup-data --download-hagrid --execute-download --prepare-ann-subsample --strict-download
```

> **📌 Tip de Seguridad:** Si te da un error de credenciales persistente, simplemente crea de forma manual el archivo en `secrets/kaggle/kaggle.json` con el contenido del tuyo y re-ejecuta.

---

## 3. Descarga Manual Obligatoria de FreiHAND

Lamentablemente, FreiHAND exige un llenado estricto de formulario académico manual para soltar su enlace temporal de descarga. Por diseño **este script NO puede bypassearlo**.

1. Ingresa a la [página oficial de FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/).
2. Completa el formulario con tus credenciales.
3. Extrae todo el contenido del `.zip` directamente dentro de la carpeta: `datasets/FreiHAND_pub_v2/`

La ruta final esperada por el pipeline debe verse obligatoriamente así:
- `datasets/FreiHAND_pub_v2/training_xyz.json`
- `datasets/FreiHAND_pub_v2/training_K.json`
- `datasets/FreiHAND_pub_v2/training/rgb/00000001.jpg`

---

## 4. Auditoría de Verificación Final

Una vez que descargaste HaGRID (automático) y FreiHAND (manual), asegúrate de que el pipeline de ML vea exitosamente los archivos ejecutando este test:

```bash
uv run python src/cli/main.py setup-data
```

Si prefieres ejecutar los pasos de forma directa sin el orquestador, consulta [README-Datasets.md](../README-Datasets.md).

Al final del reporte deberías ver todos los *status* críticos en **`[FOUND]`**. Si alguno muestra `[MISS]`, el sistema fallará durante el balanceo o entrenamiento. Revisa tus rutas en ese caso.

> ¡Hecho esto, tu clon estará cimentado y listo para invocar `uv run python src/cli/main.py balanceo`!
