import pandas as pd
import numpy as np
from pathlib import Path

def calculate_finger_extensions(landmarks):
    # landmarks: (21, 3)
    # Wrist is 0
    # Fingers: Thumb (1-4), Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
    
    def get_dist(p1, p2):
        return np.linalg.norm(landmarks[p1] - landmarks[p2])
    
    # Simple heuristic: distance from wrist to tip / sum of segment lengths
    extensions = []
    for tip in [4, 8, 12, 16, 20]:
        wrist_to_tip = get_dist(0, tip)
        extensions.append(wrist_to_tip)
    return np.array(extensions)

MANIFEST = "output/train_manifest_stgcn_secuencias.csv"
df = pd.read_csv(MANIFEST)
mano_df = df[df['dataset'] == 'mano'].copy()

# Extraer el numero de MST del sample_id
mano_df['mst_code'] = mano_df['sample_id'].str.extract(r'MST_(\d+)').fillna('0')

results = []
for mst in sorted(mano_df['mst_code'].unique()):
    subset = mano_df[mano_df['mst_code'] == mst].head(5)
    ext_list = []
    for _, row in subset.iterrows():
        path = Path(row['path_secuencia'])
        if not path.exists():
            path = Path("data/processed/secuencias_stgcn") / path.name
        
        if path.exists():
            seq = np.load(path) # (T, V, C)
            # Analizar el frame medio
            landmarks = seq[len(seq)//2]
            ext = calculate_finger_extensions(landmarks)
            ext_list.append(ext)
    
    if ext_list:
        mean_ext = np.mean(ext_list, axis=0)
        results.append({
            'MST': mst,
            'Count': len(mano_df[mano_df['mst_code'] == mst]),
            'Thumb_Ext': f"{mean_ext[0]:.2f}",
            'Index_Ext': f"{mean_ext[1]:.2f}",
            'Middle_Ext': f"{mean_ext[2]:.2f}",
            'Ring_Ext': f"{mean_ext[3]:.2f}",
            'Pinky_Ext': f"{mean_ext[4]:.2f}"
        })

report = pd.DataFrame(results)
print(report.to_string(index=False))
