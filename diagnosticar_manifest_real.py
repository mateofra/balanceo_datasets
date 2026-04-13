import pandas as pd

df = pd.read_csv('output/train_manifest_stgcn_mst_real.csv')
df = df.drop_duplicates(subset='sample_id', keep='first')

print("Columnas disponibles:")
print(df.columns.tolist())

print(f"\nDatasets presentes: {df['dataset'].value_counts().to_dict()}")

if 'landmark_quality' in df.columns:
    print(f"landmark_quality: {df['landmark_quality'].value_counts().to_dict()}")

if 'label' in df.columns:
    print(f"Top labels: {df['label'].value_counts().head(5).to_dict()}")

# Ver una fila de HaGRID completa
hagrid = df[df['dataset'] == 'hagrid']
if len(hagrid):
    print(f"\nEjemplo fila HaGRID:")
    print(hagrid.iloc[0].to_dict())
