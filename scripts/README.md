# Scripts

Indice ordenado por uso real dentro del flujo del proyecto.

## 1. Balanceo y curado

- [src/balancer/balancear_freihand_hagrid.py](../src/balancer/balancear_freihand_hagrid.py) construye manifiestos balanceados y trazables.
- [scripts/repair/repair_manifest.py](repair/README.md) repara rutas y consistencia de CSV.
- [scripts/diagnosis/](diagnosis/README.md) valida que las fuentes y landmarks sean coherentes antes de entrenar.

## 2. Preparacion ST-GCN

- [scripts/training/generar_secuencias_stgcn.py](training/README.md) convierte landmarks estaticos en secuencias temporales.
- [scripts/training/train_autosupervisado.py](training/train_autosupervisado.py) entrena el pretexto auto-supervisado de prediccion temporal.
- [scripts/training/train_supervisado.py](training/train_supervisado.py) entrena el clasificador final sobre HaGRID real.
- [src/stgcn/hand_graph.py](../src/stgcn/hand_graph.py) define la topologia anatomica.
- [src/stgcn/stgcn_model.py](../src/stgcn/stgcn_model.py) define el modelo con cabeza intercambiable.

## 3. Auditoria y graficos

- [scripts/auditoria/auditoria_dpr.py](auditoria/auditoria_dpr.py) mide DPR y TVD sobre HaGRID.
- [src/balancer/generar_graficos_balanceo.py](../src/balancer/generar_graficos_balanceo.py) genera graficos de balanceo.
- [scripts/generate/generar_grafica_auditoria_dpr.py](generate/README.md) genera la grafica resumen de la auditoria.

## 4. Notas de uso

- Los scripts de entrenamiento asumen `uv run` y el entorno del proyecto.
- Las salidas oficiales se guardan en `output/` y `graficos/` segun categoria.
- Los diagnósticos se usaron para corregir errores de rutas, calidad de landmarks y leakage temporal.
