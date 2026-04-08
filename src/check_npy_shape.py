"""Script para verificar la shape de los archivos .npy"""
import numpy as np
import os
from pathlib import Path

# Directorio de landmarks
landmarks_dir = Path("data/processed/landmarks")

# Buscar algunos archivos .npy
npy_files = list(landmarks_dir.rglob("*.npy"))[:5]  # Primeros 5 archivos

if not npy_files:
    print("❌ No se encontraron archivos .npy")
else:
    print(f"✅ Se encontraron {len(list(landmarks_dir.rglob('*.npy')))} archivos .npy en total\n")
    print("Verificando shape de algunos archivos:\n")
    
    shapes = set()
    for npy_file in npy_files:
        try:
            data = np.load(npy_file)
            shape = data.shape
            shapes.add(shape)
            print(f"📄 {npy_file.relative_to(landmarks_dir)}")
            print(f"   Shape: {shape}")
            print(f"   Dtype: {data.dtype}")
            print()
        except Exception as e:
            print(f"❌ Error al cargar {npy_file}: {e}\n")
    
    # Resumen
    print("=" * 50)
    print("RESUMEN DE SHAPES ENCONTRADAS:")
    for shape in shapes:
        print(f"  • {shape}")
