import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Cargar el manifiesto más completo disponible
df = pd.read_csv('output/train_manifest_stgcn_mst_real.csv')
print(f"Filas originales: {len(df)}")

# 1. Eliminar duplicados
df = df.drop_duplicates(subset='sample_id', keep='first')
print(f"Sin duplicados: {len(df)}")

# 2. Filtrar solo HaGRID con label real y landmarks válidos
df = df[
    (df['dataset'] == 'hagrid') &
    (df['landmark_quality'] == 'annotation_2d_projected') &
    (df['label'] != 'unknown')
].reset_index(drop=True)
print(f"Con label real y calidad válida: {len(df)}")
print(f"Clases únicas: {df['label'].nunique()}")
print(df['label'].value_counts())

# 3. Verificar que los archivos de secuencia existen en disco
df = df[df['path_secuencia'].apply(
    lambda p: Path(str(p).replace('\\','/')).exists()
)].reset_index(drop=True)
print(f"Con secuencia en disco: {len(df)}")

# 4. Split estratificado 70/15/15
train, temp = train_test_split(
    df, test_size=0.30, stratify=df['label'], random_state=42
)
val, test = train_test_split(
    temp, test_size=0.50, stratify=temp['label'], random_state=42
)

train = train.copy(); train['split'] = 'train'
val   = val.copy();   val['split']   = 'val'
test  = test.copy();  test['split']  = 'test'

canonico = pd.concat([train, val, test]).reset_index(drop=True)
canonico.to_csv('output/manifest_canonico.csv', index=False)

print(f"\nManifiesto canónico guardado.")
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
