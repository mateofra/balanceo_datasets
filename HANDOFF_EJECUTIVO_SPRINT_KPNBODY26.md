# Handoff Ejecutivo: Proyecto ST-GCN (Sprint KPNBODY26)

**Becario:** Mateo Fraguas Abal  
**Fecha:** 13/04/2026  
**Estado:** Finalizado - Reentrenamiento a Gran Escala Completado

## Metricas de Rendimiento (Comparativa)

| Metrica | Baseline (1.8k muestras) | Final (30.4k muestras) | Mejora |
| :--- | :---: | :---: | :---: |
| Dataset Total | 1,800 | 30,446 | +1,591% |
| Best Val Acc | 0.633 | 0.825 | +30.3% |
| Test Accuracy | 0.549 | 0.822 | +49.7% |

## Auditoria de Equidad (MST)

- Resultados por bloque:
  - Claro: 0.560 (n=100)
  - Medio: 0.832 (n=4381) - Bloque dominante por imputacion
  - Oscuro: 0.663 (n=86)
- DPR (Disparate Impact): 0.673
  - Nota: El sesgo detectado se debe a la distribucion masiva del bloque medio (datos nuevos imputados), lo que genera disparidad estadistica frente a los bloques originales mas pequenos.

## Artefactos Generados (Repo Activo)

1. Modelo: output/best_stgcn_canonico.pth
2. Historial: output/training_history_canonico.json
3. Auditoria: output/auditoria_final_test.csv
4. Graficas para informe: output/graficas_informe/ (01 a 06)

## Codigo y Autonomia

El repositorio activo (C:/Users/usuario/Mateo/balanceo_datasets) es autonomo:

- Contiene logica local en src/stgcn/
- Scripts src/auditoria/auditoria_final.py y src/auditoria/generar_graficas_informe.py actualizados para trabajar con manifiesto unificado de 30k muestras
- Interprete validado: .venv del repo secundario

## Notas para Informe

1. Escalabilidad: El modelo demostro capacidad de aprendizaje profundo, estabilizando la perdida y mejorando la generalizacion de forma significativa.
2. Limitacion de fairness: Recomendado para el proximo sprint recolectar etiquetas MST reales para los 30k nuevos datos; la imputacion actual sesga la metrica DPR hacia el bloque mayoritario.
3. Hito alcanzado: Se supero el objetivo de precision del 80% propuesto para este sprint.
