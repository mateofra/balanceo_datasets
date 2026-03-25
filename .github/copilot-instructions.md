# Instrucciones de Copilot para balanceo_datasets

## Objetivo del proyecto
Este repositorio se enfoca en estrategias de balanceo de datasets para reducir sesgos por distribucion de tonos MST en entrenamiento de modelos.

## Reglas de trabajo
- Priorizar cambios pequenos y verificables.
- Preservar compatibilidad con Python 3.13 o superior.
- Evitar dependencias nuevas salvo que aporten valor claro.
- Mantener codigo legible, con funciones cortas y nombres descriptivos.
- No mezclar logica de entrenamiento y evaluacion en la misma funcion si afecta claridad.
- Al crear cualquier directorio nuevo, crear siempre un README.md dentro del directorio.
- Cada README.md de directorio debe explicar que contiene esa carpeta y como usar sus contenidos.
- Al revisar un directorio ya existente, leer primero su README.md como primera fuente de contexto antes de abrir otros archivos.

## Datos y balanceo
- Aplicar balanceo solo en entrenamiento.
- No aplicar oversampling ni augmentations en validacion o test.
- Documentar en comentarios o logs cualquier regla de muestreo por clase MST.
- Al proponer augmentations, priorizar transformaciones realistas y controladas.

## Reproducibilidad
- Fijar semillas cuando se agregue aleatoriedad.
- Registrar configuraciones clave (pesos por clase, factores de oversampling, parametros de augmentations).
- Evitar valores magicos: usar constantes configurables.

## Calidad minima esperada
- Agregar o actualizar pruebas cuando se modifique logica critica.
- Incluir docstrings breves en funciones no triviales.
- Si se cambia comportamiento, actualizar README.md o notas tecnicas correspondientes.

## Estilo de respuestas del agente
- Explicar primero la solucion y luego los cambios por archivo.
- Citar riesgos y supuestos cuando falte contexto.
- Proponer siguientes pasos concretos cuando aplique.
