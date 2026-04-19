Adelante con el MVP completo de etiquetado asistido para las 10.000 muestras MANO. Implementa los 4 pasos en orden:

**Paso A — Auto-sugerencia por geometría**
- Calcular features por muestra: extensión por dedo, ángulos MCP/PIP aproximados, apertura de palma
- Mapear a etiqueta candidata usando reglas determinístas sobre el núcleo de clases separables: fist, palm, one, two_up, peace, three, four, rock, ok
- Para clases con variantes (peace vs peace_inverted, two_up vs two_up_inverted, stop vs stop_inverted) incluir heurística de orientación si es viable con landmarks 2D
- Guardar sugerencia + confianza por muestra en un CSV intermedio (no tocar el manifiesto final todavía)

**Paso B — Clustering**
- Calcular embeddings de landmarks normalizados para las 10.000 muestras
- Aplicar clustering (K-Means o similar) con K ~ número de clases objetivo
- Asignar etiqueta dominante por cluster y medir coherencia intra-cluster
- Marcar como `ambiguous` los clusters con baja coherencia o mezcla de sugerencias

**Paso C — Interfaz de validación por lotes**
- Script HTML o Jupyter interactivo con grid de 64-100 muestras visualizadas como esqueleto de mano
- Mostrar etiqueta candidata + cluster asignado por muestra
- Permitir aceptar, corregir o marcar como ambiguous
- Exportar resultado de validación a CSV

**Paso D — Control de calidad y propagación**
- Medir distribución final por clase y detectar desbalanceo severo
- Solo cuando el etiquetado esté validado, propagar al manifest_mano_secuencias.csv y al manifiesto combinado
- No tocar manifiestos de HaGRID, FreiHAND ni sintéticos

Empieza por A y B, repórtame resultados antes de construir la interfaz del Paso C.