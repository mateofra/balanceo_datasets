import pandas as pd
from pathlib import Path

candidatos = [
    'output/training/train_manifest_stgcn_fixed.csv',
    'output/train_manifest_stgcn_activo.csv',
    'output/train_manifest_stgcn_real.csv',
    'csv/train_manifest_stgcn.csv',
]

for ruta in candidatos:
    try:
        df = pd.read_csv(ruta)
        hagrid = df[df['dataset'] == 'hagrid'] if 'dataset' in df.columns else pd.DataFrame()
        
        print(f"\n{'='*50}")
        print(f"Archivo: {ruta}")
        print(f"Filas totales: {len(df)} | Únicos sample_id: {df['sample_id'].nunique()}")
        
        if len(hagrid):
            print(f"HaGRID total: {len(hagrid)}")
            if 'landmark_quality' in df.columns:
                print(f"landmark_quality: {hagrid['landmark_quality'].value_counts().to_dict()}")
            if 'label' in df.columns:
                unknown = (hagrid['label'] == 'unknown').sum()
                print(f"Labels unknown: {unknown} / {len(hagrid)}")
            if 'split' in df.columns:
                print(f"Splits: {df['split'].value_counts().to_dict()}")
            # Verificar columna de ruta a secuencia
            col_ruta = [c for c in df.columns if 'path' in c.lower()]
            print(f"Columnas de ruta: {col_ruta}")
    except Exception as e:
        print(f"{ruta}: ERROR {e}")
