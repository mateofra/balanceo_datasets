#!/usr/bin/env python3
"""
PIPELINE ST-GCN: Visualización del flujo de datos normalizados.

Este script genera un resumen gráfico de cómo fluyen los datos desde
landmarks estáticos hasta training del modelo ST-GCN.
"""

import json
from pathlib import Path


def visualizar_pipeline():
    """Crear resumen visual del pipeline."""
    
    print("\n" + "=" * 80)
    print("ST-GCN PIPELINE: ARQUITECTURA DE DATOS NORMALIZADOS")
    print("=" * 80)
    
    print("""
    
    FASE 1: GENERAR SECUENCIAS TEMPORALES
    ────────────────────────────────────────────────────────────────
    
        Landmarks estáticos (FreiHAND)         HaGRID (MediaPipe)
               ↓                                      ↓
           (21, 3)                              (21, 3)
           float32                              float32
           
           Coordenadas en metros                Coordenadas normalizadas
           X,Y: ±0.09m                         X,Y: [0, 1]
           Z: [0.46, 0.91]m                    Z: [0, 1]
           
               ↓ (generar_secuencias_stgcn.py)
        
        Ruido motor Gaussiano (σ=2mm)
        Suavizado temporal
        16 frames sintéticos por landmark
        
               ↓
           (16, 21, 3)
           Secuencias temporales
    
    
    FASE 2: CALCULAR ESTADÍSTICAS DE NORMALIZACIÓN
    ────────────────────────────────────────────────────────────────
    
        Muestreo: 500 imágenes × 8 frames = 8,000 landmarks/ freihand
        
        Por cada coordenada:
        - Calcular media (μ)
        - Calcular desviación estándar (σ)
        
        Resultado: landmarks_normalizer.json
        
        FreiHAND: Media global = 0.226, Std = 0.051
        
        ✓ Agnóstico respecto a escala original
        ✓ Aplicable a datasets diferentes
    
    
    FASE 3: DATALOADER CON NORMALIZACIÓN
    ────────────────────────────────────────────────────────────────
    
        st_gcn_dataloader.py
        
        Para cada muestra:
        
            1. Cargar secuencia (16, 21, 3)
            2. Z-score normalization: (x - μ) / σ
            3. Temporal augmentation (opcional): Dropout
            4. Convertir a Tensor
            5. Retornar tupla (secuencia, label)
        
        Output del DataLoader:
        ┌─────────────────────────────────────┐
        │ Batch de secuencias normalizadas     │
        ├─────────────────────────────────────┤
        │ Shape: (32, 16, 21, 3)              │
        │ - Batch: 32                         │
        │ - T (frames): 16                    │
        │ - Joints: 21                        │
        │ - Coords: 3 (x, y, z)              │
        │                                     │
        │ Valores: Media ≈ 0, Std ≈ 1       │
        │ Dtype: torch.float32                │
        └─────────────────────────────────────┘
        
        Labels: (32,) - índices de clase
    
    
    FASE 4: ENTRENAMIENTO ST-GCN
    ────────────────────────────────────────────────────────────────
    
        train_stgcn.py
        
        Modelo:
        ┌──────────────────────────┐
        │  Input (B, T, 21, 3)    │
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │ Permute → (B, 3, T, 21) │  Formato ST-GCN
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │ Spatial Conv (vertices)  │  Conv2d(3, 128)
        │ Output: (B, 128, T, 21) │
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │ Temporal LSTM (frames)   │  2 layers
        │ Output: (B, 128)        │
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │ Classification Head      │  Dense layers
        │ Output: (B, num_classes)│
        └──────────────────────────┘
        
        Loss: CrossEntropyLoss
        Optimizer: Adam (lr=0.001)
        Early stopping: patience=10
    
    
    RESUMEN DE DATOS
    ────────────────────────────────────────────────────────────────
    """)
    
    # Leer estadísticas si existen
    try:
        with open("landmarks_normalizer.json", "r") as f:
            stats = json.load(f)
        
        if stats['freihand']:
            mean_freihand = stats['freihand']['mean'][0]
            print(f"    FreiHAND normalizador cargado ✓")
            print(f"    - Muestras analizadas: 8,000 frames")
            print(f"    - Media X (ejemplo): {mean_freihand:.6f}")
            print()
    except FileNotFoundError:
        print("    ⚠️  landmarks_normalizer.json no encontrado")
        print()
    
    # Verificar secuencias
    seq_dir = Path("data/processed/secuencias_stgcn")
    if seq_dir.exists():
        seq_count = len(list(seq_dir.glob("*.npy")))
        print(f"    Secuencias generadas: {seq_count:,}")
        print(f"    Shape: (T=16, Joints=21, Coords=3)")
        print()
    else:
        print("    ⚠️  Directorio de secuencias no encontrado")
        print()
    
    # Verificar manifest
    manifest_path = Path("data/processed/secuencias_stgcn/manifest_secuencias.csv")
    if manifest_path.exists():
        import pandas as pd
        df = pd.read_csv(manifest_path)
        print(f"    Manifest: {len(df)} muestras")
        print(f"    Splits: {df['split'].value_counts().to_dict()}")
        print(f"    Datasets: {df['dataset'].value_counts().to_dict()}")
        print()
    else:
        print("    ⚠️  Manifest no encontrado")
        print()
    
    print("=" * 80)
    print("PRÓXIMOS PASOS")
    print("=" * 80)
    print("""
    1. ✓ Landmarks estáticos verificados
    2. ✓ Secuencias temporales generadas
    3. ✓ Normalización z-score implementada
    4. ✓ DataLoader con normalización funcional
    5. → PENDIENTE: Actualizar labels (actualmente 'unknown')
    6. → INICIO: Entrenar modelo con 'uv run train_stgcn.py'
    """)
    
    print("=" * 80)
    print("NOTAS IMPORTANTES")
    print("=" * 80)
    print("""
    ⚠️  Issues Conocidos:
    
    1. LABELS: Todas las muestras etiquetadas como 'unknown'
       - Cambiar en manifest con labels reales de gestos
       - O usar mapper string → índice en DataLoader
    
    2. FREIHAND ONLY: Solo 10,000 muestras (FreiHAND)
       - HaGRID: 10,000 referencias pero sin archivos
       - Propuesta: Investigar path de HaGRID
    
    3. ONE-HAND LIMITATION: MediaPipe configurado con num_hands=1
       - Ubicación: src/procesar_landmarks_hagrid_mediapipe.py:32
       - Impacto: Gestos dos manos perderían info
       - Solución: Regenerar con num_hands=2 → shape (42, 3)
    
    ✓ Ventajas de esta arquitectura:
    
    1. Z-score normalization es agnóstico a escala original
       - Compatible con múltiples datasets (FreiHAND, HaGRID, etc)
       - Centraliza preprocesamiento en LandmarkNormalizer
    
    2. DataLoader limpio:
       - Augmentación temporal controlada
       - Fácil de extender (agregar rotación, scaling, etc)
       - Eficiente: carga bajo demanda
    
    3. Modelo ST-GCN estándar:
       - Separa convolución espacial (joints) de temporal (frames)
       - Early stopping previene overfitting
       - Historial de training guardado para análisis
    """)


if __name__ == "__main__":
    visualizar_pipeline()
