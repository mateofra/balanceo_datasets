# 📝 Resumen Técnico: Pipeline ST-GCN + Etiquetado Asistido

**Contexto:**
El proyecto tiene como objetivo el entrenamiento de un modelo **ST-GCN (Spatio-Temporal Graph Convolutional Network)** para clasificar gestos de la mano, utilizando un dataset híbrido de 10.000 muestras sintéticas (MANO) y datos reales (HaGRID/FreiHAND).

## 🛠️ Arquitectura y Flujo Actual

### 1. Integridad de Datos y Propagación
*   Se corrigió un error crítico donde las etiquetas (`label`) se perdían durante el balanceo de datasets. Ahora, el script `build_balanced_block_manifest.py` es agnóstico al esquema y preserva todas las columnas de metadatos.
*   Se implementó un enriquecimiento retroactivo para recuperar etiquetas de datasets reales que habían sido filtradas accidentalmente.
*   Se modificaron los generadores de landmarks y secuencias para preservar la columna `path`, permitiendo la trazabilidad visual desde la secuencia `.npy` hasta la imagen original `.png/.jpg`.

### 2. Arquitectura del Modelo (`RealSTGCN`)
*   **Base**: GCN espacial sobre el grafo anatómico de la mano (21 articulaciones) con convoluciones temporales y bloques residuales.
*   **Atención Espacial**: Se añadió una capa de `SpatialAttention` que calcula un score de importancia por nodo. Esto permite la **explicabilidad del modelo** (saber qué articulaciones influyen más en la predicción).
*   **Robustez**: Se migró de `.view()` a `.reshape()` en el `forward` pass para manejar tensores no contiguos.

---

## ⚠️ Puntos Críticos (Actualización)

### 1. Estado del problema de etiquetas MANO y FreiHAND
*   **Muestras MANO**: Se han reclasificado exitosamente **1.184 muestras** mediante heurísticas de segunda pasada (`rock`, `two_up`, `call`).
*   **Outliers**: Se han excluido **105 muestras** marcadas como `bad_pose` tras el análisis de clustering.
*   **Pendiente**: Quedan ~1.800 muestras `unknown` en MANO y el dataset FreiHAND completo (etiquetado como `hand`).
*   **Impacto**: El pool de ruido de clase en MANO se ha reducido un **43%**, lo que debería permitir superar el techo de accuracy actual.

### 2. Mejoras SOA en curso (No completadas)
*   **Mejora 1: Auditoría de calidad**: Detectar y excluir secuencias con >30% landmarks fuera de rango [0,1] o presencia de NaN/Inf.
*   **Mejora 2: Augmentation on-the-fly**: Implementar *Node Dropping* (p=0.1) y *Edge Perturbation* (p=0.05) durante el entrenamiento.
*   **Mejora 3: Migración a STDA-GCN**: Modelo de ~6M parámetros con atención dinámica y warm-up de 5 épocas.
*   *Nota: Las mejoras 1 y 2 son prioritarias antes de migrar al modelo STDA.*

### 3. Tabla de Evolución de Accuracy (Referencia)
| Configuración Modelo | Épocas | Val Accuracy | Test Accuracy | Conclusión |
| :--- | :---: | :---: | :---: | :--- |
| Baseline (SimpleST_GCN, estático) | 20 | 3.16% | 2.34% | - |
| SimpleST_GCN (Secuencias temporales) | 8 | 23.68% | - | Mejora drástica con T=16 |
| RealSTGCN + Coseno Scheduler | 15 | 23.87% (pico) | 23.54% (final) | Estabilidad mejorada |
| RealSTGCN + Coseno Scheduler | 20 | 23.95% (pico) | 23.70% (final) | Techo actual alcanzado |
| **RealSTGCN + Heuristic_v2** | - | *Pendiente* | *Pendiente* | **Próximo Entrenamiento** |

**Conclusión:** Se ha desbloqueado la diversidad del dataset MANO. El próximo entrenamiento utilizará las nuevas etiquetas validadas y el augmentation de Node Dropping para intentar romper el techo del 24%.

### 4. Dependencias entre tareas
Para cualquier agente de IA o humano operando el sistema:
*   **BLOQUEADO**: El entrenamiento final con el dataset completo está bloqueado hasta resolver las etiquetas de MANO/FreiHAND.
*   **PARALELO (No bloqueado)**:
    *   Ejecución de auditoría de calidad y limpieza de landmarks.
    *   Implementación de técnicas de data augmentation.
    *   Desarrollo de la arquitectura STDA-GCN.
    *   Ejecución del MVP de etiquetado asistido (Pasos A-D).

---
*Este documento sirve como "Source of Truth" para el estado del entrenamiento de gestos.*
