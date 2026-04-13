# preparar_entrenamiento.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def generar_secuencia(landmarks, T=16, sigma=0.015):
    """landmarks: (21,3) → secuencia: (T,21,3)"""
    secuencia = np.tile(landmarks, (T, 1, 1))
    ruido = np.random.normal(0, sigma, size=(T, 21, 3))
    for j in range(21):
        for c in range(3):
            ruido[:, j, c] = np.convolve(ruido[:, j, c], np.ones(3)/3, mode='same')
    return (secuencia + ruido).astype(np.float32)

# Cargar manifiesto base
df = pd.read_csv('output/train_manifest_stgcn_real.csv')
df = df.drop_duplicates(subset='sample_id', keep='first')
df = df[
    (df['dataset'] == 'hagrid') &
    (df['landmark_quality'] == 'annotation_2d_projected') &
    (df['label'] != 'unknown')
].reset_index(drop=True)
print(f"Muestras base: {len(df)}")

# Directorio de salida para secuencias
seq_base = Path('data/processed/secuencias/hagrid')

# Generar secuencias
rutas_seq = []
generadas = 0
fallidas  = 0

for _, row in df.iterrows():
    src = Path(row['path_landmarks'])
    if not src.exists():
        rutas_seq.append(None)
        fallidas += 1
        continue

    dst = seq_base / row['label'] / f"{row['sample_id']}.npy"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        lm  = np.load(str(src))          # (21, 3)
        seq = generar_secuencia(lm)      # (T, 21, 3)
        np.save(str(dst), seq)

    rutas_seq.append(str(dst))
    generadas += 1

print(f"Generadas: {generadas} | Fallidas: {fallidas}")

# Añadir ruta de secuencia al dataframe
df['path_secuencia'] = rutas_seq
df = df[df['path_secuencia'].notna()].reset_index(drop=True)

# Split estratificado 70/15/15
train, temp = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
val, test   = train_test_split(temp, test_size=0.50, stratify=temp['label'], random_state=42)

train = train.copy(); train['split'] = 'train'
val   = val.copy();   val['split']   = 'val'
test  = test.copy();  test['split']  = 'test'

canonico = pd.concat([train, val, test]).reset_index(drop=True)
canonico.to_csv('output/manifest_canonico.csv', index=False)

print(f"\nManifiesto canónico guardado: output/manifest_canonico.csv")
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print(f"Clases: {df['label'].nunique()}")
