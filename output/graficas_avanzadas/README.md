# Graficas Avanzadas

Esta carpeta contiene visualizaciones de caja blanca para explicar el comportamiento cinemático y ético del modelo ST-GCN/STAEGCN.

## Archivos esperados

- 07_jerarquia_birdwhistell.png
- 08_mapa_calor_atencion_nodos.png
- 09_matriz_confusion_top_errores.png
- 10_tsne_clusters_gestos.png
- 11_analisis_ruido_robustez.png
- 12_latencia_inferencia_hardware.png

## Como generar

Ejecutar desde la raiz del repo:

```powershell
& 'C:\Users\usuario\Mateo\balanceo_2\balanceo_datasets\.venv\Scripts\python.exe' generar_graficas_avanzadas.py
```

## Datos usados

- output/training_history_canonico.json
- output/manifest_unificado_final.csv
- output/best_stgcn_canonico.pth
