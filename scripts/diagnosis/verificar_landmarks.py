from pathlib import Path
import pandas as pd

# Contar archivos reales
landmarks_dir = Path("data/processed/landmarks")
archivos_reales = list(landmarks_dir.rglob("*.npy"))
print(f"✅ Archivos .npy encontrados: {len(archivos_reales)}")

# Muestreo de primeros archivos
print("\nPrimeros 10 archivos encontrados:")
for f in sorted(archivos_reales)[:10]:
    print(f"  - {f.relative_to(landmarks_dir)}")

# Leer CSV y contar referencias
df = pd.read_csv("csv/train_manifest_stgcn.csv")
print(f"\n📋 Referencias en CSV: {len(df)}")
print(f"   Muestras únicas: {df['sample_id'].nunique()}")

# Ver si hay discrepancias
print(f"\n⚠️  Diferencia: {len(df) - len(archivos_reales)} referencias sin archivo")
