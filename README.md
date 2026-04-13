# balanceo_datasets (repo activo)

Repositorio consolidado para pipeline ST-GCN con manifiesto unificado y auditoria de equidad.

## Estado de consolidacion

- Codigo ST-GCN local en `src/stgcn/`.
- Scripts principales con rutas relativas al repo (sin dependencias a `balanceo_2` para codigo/datos).
- Modelo MediaPipe local en `models/hand_landmarker.task`.
- Manifiestos y artefactos en `output/`.

## Requisito de interprete

El proyecto se ejecuta con el interprete corporativo:

`C:\Users\usuario\Mateo\balanceo_2\balanceo_datasets\.venv\Scripts\python.exe`

## Flujo de ejecucion

1. Unificar datasets:

```powershell
Set-Location 'C:\Users\usuario\Mateo\balanceo_datasets'
& 'C:\Users\usuario\Mateo\balanceo_2\balanceo_datasets\.venv\Scripts\python.exe' unificar_datasets_y_split.py
```

2. Entrenar ST-GCN:

```powershell
Set-Location 'C:\Users\usuario\Mateo\balanceo_datasets'
& 'C:\Users\usuario\Mateo\balanceo_2\balanceo_datasets\.venv\Scripts\python.exe' -u preparar_y_entrenar.py
```

3. Auditoria DPR/TVD:

```powershell
Set-Location 'C:\Users\usuario\Mateo\balanceo_datasets'
& 'C:\Users\usuario\Mateo\balanceo_2\balanceo_datasets\.venv\Scripts\python.exe' auditoria_final.py
```

4. Generar graficas:

```powershell
Set-Location 'C:\Users\usuario\Mateo\balanceo_datasets'
& 'C:\Users\usuario\Mateo\balanceo_2\balanceo_datasets\.venv\Scripts\python.exe' generar_graficas_informe.py
```

## Archivos de salida esperados

- `output/manifest_unificado_final.csv`
- `output/best_stgcn_canonico.pth`
- `output/training_history_canonico.json`
- `output/auditoria_final_test.csv`
- `output/graficas_informe/*.png`
