# src

Contiene scripts ejecutables para preparar y balancear datasets.

## Uso

Ejecuta los scripts con `uv run python ...` desde la raiz del repositorio.

Script principal de balanceo:

- `balancear_freihand_hagrid.py`: construye un manifiesto de entrenamiento balanceado entre FreiHAND y HaGRID, con opcion de integrar bloques MST desde un CSV externo de auditoria.
	Tambien soporta oversampling de extremos MST (1,2,3,10), augmentacion cromatica virtual para MST 8-9 y exportacion de tres sets por tono (claro/medio/oscuro), siempre para entrenamiento.
- `generar_graficos_balanceo.py`: genera graficos PNG y un reporte Markdown para validar que la composicion del manifiesto quede alineada con el objetivo de balanceo.
