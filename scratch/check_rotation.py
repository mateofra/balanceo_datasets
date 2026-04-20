import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def check_rotation():
    manifest_path = Path("output/train_manifest_stgcn_secuencias.csv")
    df = pd.read_csv(manifest_path)
    mano_df = df[df['dataset'] == 'mano'].head(100)
    
    angles = []
    for idx, row in mano_df.iterrows():
        lms = np.load(row['path_landmarks'])
        wrist = lms[0, :2]
        mid_base = lms[9, :2]
        axis = mid_base - wrist
        angle = np.degrees(np.arctan2(axis[0], -axis[1]))
        angles.append(angle)
        
    print(f"Ángulos detectados (promedio): {np.mean(angles):.2f}°")
    print(f"Desviación estándar: {np.std(angles):.2f}°")
    print(f"Máximo: {np.max(angles):.2f}°, Mínimo: {np.min(angles):.2f}°")

if __name__ == "__main__":
    check_rotation()
