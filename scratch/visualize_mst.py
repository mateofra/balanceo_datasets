import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

def draw_skeleton(ax, landmarks, title):
    # Connections for a hand
    connections = [
        (0,1),(1,2),(2,3),(3,4), # Thumb
        (0,5),(5,6),(6,7),(7,8), # Index
        (0,9),(9,10),(10,11),(11,12), # Middle
        (0,13),(13,14),(14,15),(15,16), # Ring
        (0,17),(17,18),(18,19),(19,20) # Pinky
    ]
    for p1, p2 in connections:
        ax.plot([landmarks[p1,0], landmarks[p2,0]], 
                [landmarks[p1,1], landmarks[p2,1]], 'b-')
    ax.scatter(landmarks[:,0], landmarks[:,1], c='r', s=10)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')

MANIFEST = "output/train_manifest_stgcn_secuencias.csv"
df = pd.read_csv(MANIFEST)
mano_df = df[df['dataset'] == 'mano'].copy()
mano_df['mst_code'] = mano_df['sample_id'].str.extract(r'MST_(\d+)').fillna('0')

mst_unique = sorted(mano_df['mst_code'].unique())
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, mst in enumerate(mst_unique[:10]):
    sample = mano_df[mano_df['mst_code'] == mst].iloc[0]
    path = Path(sample['path_secuencia'])
    if not path.exists():
        path = Path("data/processed/secuencias_stgcn") / path.name
    
    if path.exists():
        seq = np.load(path)
        landmarks = seq[len(seq)//2] # Middle frame
        draw_skeleton(axes[i], landmarks, f"MST_{mst}")

plt.tight_layout()
plt.savefig("output/debug_mst_skeletons.png")
print("Imagen guardada en output/debug_mst_skeletons.png")
