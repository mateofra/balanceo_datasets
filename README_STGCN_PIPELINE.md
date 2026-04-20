# Pipeline de Entrenamiento ST-GCN y Etiquetado Asistido

Este documento resume el flujo de trabajo implementado para el entrenamiento del modelo **Spatio-Temporal Graph Convolutional Network (ST-GCN)** especializado en reconocimiento de gestos de la mano.

## 📌 Resumen del Proyecto
El objetivo es clasificar gestos utilizando un dataset híbrido de ~50,000 muestras (HaGRID, FreiHAND y 10,000 muestras sintéticas MANO). El principal desafío resuelto fue el etiquetado de las muestras sintéticas que inicialmente carecían de clase de gesto.

---

## 🛠️ Desarrollo e Implementación

### 1. Modelo ST-GCN con Explicabilidad
*   **Arquitectura**: Implementación de `RealSTGCN` utilizando el grafo anatómico de la mano (21 nodos).
*   **Atención Espacial**: Se integró una capa de `SpatialAttention` para identificar qué articulaciones son críticas en cada predicción.
*   **Robustez**: Corrección de errores de contigüidad de memoria mediante la migración de `.view()` a `.reshape()` en el pass forward.

### 2. Pipeline de Etiquetado Asistido (MVP)
Se desarrolló un sistema de 4 pasos para resolver las 10,000 etiquetas `unknown` del dataset MANO:
*   **Normalización de Orientación**: Alineación automática de la mano al eje vertical antes de procesar heurísticas para manejar rotaciones de hasta 180°.
*   **Heurísticas Anatómicas**: Cálculo de ángulos MCP-PIP y PIP-DIP para detectar flexión de dedos y distancias inter-digitales para gestos como `ok`.
*   **Clustering de "Unknowns"**: Uso de K-Means sobre el vector de características para derivar umbrales empíricos de clases no detectadas originalmente.
*   **Resultados**:
    *   **1,184 muestras reclasificadas** exitosamente (`rock`, `two_up`, `call`).
    *   **105 outliers eliminados** (`bad_pose`).
    *   Reducción del pool de ruido de clase en un **43%**.

### 3. Herramientas de Auditoría y Calidad
*   **Validador Visual HTML**: Una herramienta autocontenida (`output/validador_etiquetas.html`) que permite la revisión humana rápida de esqueletos de mano en un grid interactivo.
*   **Reporte de Confianza**: Script automatizado que valida la estabilidad de las nuevas etiquetas antes de su propagación al manifiesto.
*   **Auditoría de Landmarks**: Verificación del 100% del dataset buscando NaNs, Infs o coordenadas fuera de rango.

### 4. Estrategia de Entrenamiento
*   **Augmentation on-the-fly**: Implementación de **Node Dropping (p=0.1)** para aumentar la robustez del modelo ante oclusiones o fallos en la detección de landmarks.
*   **Dataset Balanceado**: Uso de un manifiesto unificado con trazabilidad total mediante la columna `label_source`.

---

## 🚀 Estado Actual y Resultados
*   **Baseline de Accuracy**: ~24% (limitado por etiquetas `unknown`).
*   **Hito Actual**: Dataset limpio y enriquecido con 1,184 nuevas etiquetas validadas.
*   **Próximo Paso**: Ejecución del entrenamiento final en `entrenar_stgcn.ipynb` utilizando el nuevo manifiesto propagado.

---

## 📁 Archivos Clave
*   `entrenar_stgcn.ipynb`: Notebook principal de entrenamiento e inferencia.
*   `src/stgcn/stgcn_model.py`: Definición de la arquitectura con atención.
*   `scratch/assisted_labeling_step_a.py`: Motor de heurísticas refinado.
*   `output/validador_etiquetas.html`: Herramienta de auditoría visual.
*   `proyecto_stgcn_resumen.md`: Resumen técnico para agentes de IA.
