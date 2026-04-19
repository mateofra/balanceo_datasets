# evaluacion_datsets

Proyecto para auditar diversidad de tono de piel en FreiHAND y unificarlo con HaGRID en un CSV común.

## Requisitos

- Python 3.13+
- uv (gestor de entornos y paquetes)
- Dependencias del proyecto (definidas en [pyproject.toml](pyproject.toml))

Instalación rápida con uv:

```powershell
uv sync
```

## Crear tu propio directorio `datasets/`

Los scripts esperan que los datos estén dentro de [datasets/](datasets/) en la raíz del proyecto.

En Windows (PowerShell):

```powershell
mkdir datasets\FreiHAND_pub_v2\training\rgb
mkdir datasets\hagrid_annotations\train
mkdir datasets\hagrid_dataset
```

`mkdir` en Windows crea directorios intermedios, por eso no hace falta un equivalente de `-p`.

En Linux/macOS:

```bash
mkdir -p datasets/FreiHAND_pub_v2/training/rgb
mkdir -p datasets/hagrid_annotations/train
mkdir -p datasets/hagrid_dataset
```

Estructura final esperada (mínima):

```text
datasets/
	FreiHAND_pub_v2/
		training_xyz.json
		training_K.json
		training_scale.json
		training/
			rgb/
	hagrid_annotations/
		train/
			palm.json
			fist.json
			like.json
			ok.json
	hagrid_dataset/
		palm/
		fist/
		like/
		ok/
```

## Flujo canónico actual

El recorrido vigente para descarga, extracción, generación de MST reales, síntesis y balanceo está documentado en [README-Datasets.md](../README-Datasets.md).

Usa ese archivo como referencia principal para los comandos actuales bajo `src/datasets/`.

## Nota histórica

Las secciones anteriores de este documento se conservan solo como contexto del flujo antiguo. Si vas a reproducir el estado actual del proyecto, sigue el README nuevo de datasets.

## Problemas comunes

- Error `No existe la ruta requerida`: revisa que FreiHAND esté exactamente en [datasets/FreiHAND_pub_v2/](datasets/FreiHAND_pub_v2/).
- Aviso `no existe anotacion para gesto`: falta el JSON de ese gesto en [datasets/hagrid_annotations/train/](datasets/hagrid_annotations/train/).
- Aviso `no hay imagenes locales para gesto`: falta la carpeta o imágenes del gesto en [datasets/hagrid_dataset/](datasets/hagrid_dataset/).
