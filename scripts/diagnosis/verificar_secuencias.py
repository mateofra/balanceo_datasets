import numpy as np
from pathlib import Path

# Verificar una secuencia generada
secuencias_dir = Path("data/processed/secuencias_stgcn")
secuencias = list(secuencias_dir.glob("*.npy"))[:3]

print("[CHECK] Ejemplos de secuencias generadas:\n")
for seq_file in secuencias:
    data = np.load(seq_file)
    print(f"Archivo: {seq_file.name}")
    print(f"  Shape: {data.shape} (T=frames, landmarks, coords)")
    print(f"  Dtype: {data.dtype}")
    print(f"  Rango: [{data.min():.4f}, {data.max():.4f}]")
    print()

# Verificar manifest
import pandas as pd
manifest = pd.read_csv(secuencias_dir / "manifest_secuencias.csv")
print(f"\n[MANIFEST] Información del manifest:")
print(f"  Registros: {len(manifest)}")
print(f"  Columnas: {', '.join(manifest.columns.tolist())}")
print(f"\n  Primeros registros:")
print(manifest[['sample_id', 'label', 'condition', 'T']].head(3).to_string(index=False))
