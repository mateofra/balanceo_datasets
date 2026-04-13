import pandas as pd
from pathlib import Path

# Buscar todos los CSV en el proyecto
csvs = list(Path('.').rglob('*.csv'))
for p in sorted(csvs):
    try:
        df = pd.read_csv(p)
        if 'dataset' not in df.columns:
            continue
        hagrid = df[df['dataset'] == 'hagrid']
        if len(hagrid) > 100:
            print(f"\n{p}")
            print(f"  Total filas: {len(df)}")
            print(f"  HaGRID: {len(hagrid)}")
            print(f"  Columnas: {df.columns.tolist()}")
            if 'label' in df.columns:
                labels = hagrid['label'].value_counts().head(3).to_dict()
                print(f"  Top labels HaGRID: {labels}")
            if 'landmark_quality' in df.columns:
                print(f"  landmark_quality: {hagrid['landmark_quality'].value_counts().to_dict()}")
    except:
        continue
