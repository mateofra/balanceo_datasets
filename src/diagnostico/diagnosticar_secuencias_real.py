import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

df = pd.read_csv('output/train_manifest_stgcn_real.csv')

# 1. Eliminar duplicados
df = df.drop_duplicates(subset='sample_id', keep='first')
print(f"Sin duplicados: {len(df)}")

# 2. Filtrar HaGRID annotation_2d_projected con label real
df = df[
    (df['dataset'] == 'hagrid') &
    (df['landmark_quality'] == 'annotation_2d_projected') &
    (df['label'] != 'unknown')
].reset_index(drop=True)
print(f"HaGRID válido: {len(df)}")
print(f"Clases: {df['label'].nunique()}")

# 3. Verificar qué columna de ruta apunta a secuencias reales
# Buscar directorio de secuencias HaGRID
seq_dirs = list(Path('data/processed').rglob('secuencias')) + \
           list(Path('data').rglob('secuencias'))
print(f"\nDirectorios de secuencias encontrados: {seq_dirs}")

# Contar .npy en cada uno
for d in seq_dirs:
    npys = list(d.rglob('*.npy'))
    print(f"  {d}: {len(npys)} archivos .npy")

# 4. Ver ejemplo de path_landmarks para entender estructura
print(f"\nEjemplo path_landmarks HaGRID:")
print(df['path_landmarks'].iloc[0])
