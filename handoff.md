# Handoff de sesion

Fecha: 2026-04-20

## Objetivo de esta sesion
Definir si es viable recuperar o crear etiquetas de gesto para las 10,000 muestras MANO, y proponer una estrategia antes de modificar manifiestos o datos.

## Trabajo realizado
- Se inspecciono el arbol de MANO en `datasets/synthetic_mst/`:
  - `mano_samples_balanced/` contiene imagenes y landmarks.
  - `metadata/` contiene manifiestos de balanceo por MST.
- Se confirmo que no existen sidecars con etiqueta de gesto (JSON/YAML/CSV de clase) para MANO.
- Se audito una muestra representativa de landmarks MANO:
  - 30 muestras (3 por cada MST 1..10).
  - Descripcion de configuraciones de dedos (abierto/medio/cerrado) por muestra.
  - Medidas de diversidad geometrica entre muestras.
- Se ejecuto una auditoria ampliada sobre 1,200 muestras aleatorias para estimar diversidad de patrones de pose.
- Se reviso `manos/src/generator.py` y `manos/src/build_balanced_block_manifest.py` para verificar si existe una variable de gesto persistida en generacion/manifiestos.
- Se extrajeron las clases de referencia de los manifiestos finales:
  - HaGRID: 18 clases.
  - FreiHAND: solo `unknown` en este repo/flujo.

## Hallazgos clave
- Las poses MANO si tienen variabilidad suficiente para que el etiquetado aporte valor:
  - En 30 muestras: 18 patrones distintos de estado de dedos.
  - En 1,200 muestras: 73 patrones distintos.
- El generador MANO usa parametros de pose aleatorios (open/relaxed/flexed + jitter), pero no guarda etiqueta de gesto por muestra.
- El `sample_id` de MANO codifica MST (`sample_XXXXX_MST_N`), no clase de gesto.
- En el estado actual, no se puede reconstruir automaticamente una etiqueta de gesto "real" desde metadatos existentes de esas 10,000 muestras.

## Decision de trabajo acordada
- Proceder con estrategia de etiquetado desde cero para recuperar utilidad de las 10,000 muestras MANO.
- No tocar aun manifiestos ni archivos de datos hasta definir el flujo final de etiquetado.

## Estrategia propuesta (MVP)
- Evitar etiquetado completamente frame-a-frame.
- Usar enfoque semiautomatico:
  1. Sugerencia automatica por geometria de landmarks (features por dedos y palma).
  2. Clustering de poses para etiquetar por grupos similares.
  3. Revision humana por lotes (grid + atajos) solo en casos ambiguos.
  4. Control de calidad y marca de ambiguos.

## Evidencia principal en codigo/datos
- `manos/src/generator.py`: generacion de pose sin campo de clase de gesto.
- `manos/src/build_balanced_block_manifest.py`: manifiesto centrado en MST/bloque, sin gesto.
- `datasets/synthetic_mst/metadata/manifest_mano_samples_balanced.csv`: columnas de MST sin etiqueta de gesto.
- `output/final_manifests/manifest_hagrid_secuencias.csv`: 18 clases de referencia.
- `output/final_manifests/manifest_freihand_secuencias.csv`: clase `unknown`.

## Pendientes para la proxima sesion
- Diseñar e implementar el script MVP de pre-etiquetado MANO (sin sobreescribir manifiestos finales aun).
- Definir taxonomia objetivo de clases para MANO (subconjunto confiable vs 18 clases completas).
- Implementar interfaz de validacion por lotes y formato de export de etiquetas revisadas.
- Una vez validado, integrar etiquetas MANO al pipeline de entrenamiento.
