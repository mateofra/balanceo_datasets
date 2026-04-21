
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def visualize_mano_audit():
    manifest_path = "/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/output/train_manifest_stgcn_secuencias_fixed.csv"
    df = pd.read_csv(manifest_path)
    
    # Tomar 4 muestras de 'fist' de MANO (heurística)
    samples = df[(df['dataset'] == 'mano') & (df['label'] == 'fist')].sample(n=4, random_state=42)
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, (_, row) in enumerate(samples.iterrows()):
        lm_path = f"/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/{row['path_landmarks']}"
        if not Path(lm_path).exists(): continue
        
        lm = np.load(lm_path) # (21, 3)
        
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        ax.scatter(lm[:, 0], lm[:, 1], lm[:, 2], c='r')
        
        # Conexiones básicas de la mano (simplificadas)
        connections = [
            (0,1), (1,2), (2,3), (3,4), # Pulgar
            (0,5), (5,6), (6,7), (7,8), # Índice
            (0,9), (9,10), (10,11), (11,12), # Medio
            (0,13), (13,14), (14,15), (15,16), # Anular
            (0,17), (17,18), (18,19), (19,20) # Meñique
        ]
        for start, end in connections:
            ax.plot([lm[start,0], lm[end,0]], [lm[start,1], lm[end,1]], [lm[start,2], lm[end,2]], 'blue')
            
        ax.set_title(f"ID: {row['sample_id']}\nLabel: {row['label']}")
        ax.view_init(elev=-90, azim=-90) # Vista palmar
        
    plt.tight_layout()
    plt.savefig("/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/output/auditoria_mano_fist.png")
    print("📸 Auditoría visual guardada en output/auditoria_mano_fist.png")

if __name__ == "__main__":
    visualize_mano_audit()
