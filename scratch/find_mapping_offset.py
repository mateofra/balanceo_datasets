
import pandas as pd
import numpy as np
import os
from pathlib import Path

def find_offset():
    balanced_path = "/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/datasets/synthetic_mst/metadata/manifest_mano_samples_balanced.csv"
    generated_path = "/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/datasets/synthetic_mst/metadata/manifest_synthetic_generated_blocks_qc_adjusted.csv"
    requests_path = "/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/datasets/synthetic_mst/metadata/manifest_synthetic_requests_blocks_qc_adjusted.csv"
    
    b_df = pd.read_csv(balanced_path)
    g_df = pd.read_csv(generated_path)
    # r_df = pd.read_csv(requests_path) # Not needed yet
    
    print(f"Buscando coincidencia para las primeras 5 muestras balanceadas...")
    for i in range(5):
        sid = b_df.iloc[i]['sample_id']
        lm_path = f"/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/data/processed/landmarks/{sid}.npy"
        if not os.path.exists(lm_path): 
            print(f"Landmarks no encontrados: {lm_path}")
            continue
        l1 = np.load(lm_path)
        
        # Search in generated samples (first 5000 for speed)
        for j in range(5000):
            source_id = g_df.iloc[j]['source_sample_id']
            l2_path = f"/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/data/processed/landmarks/{source_id}.npy"
            if not os.path.exists(l2_path): continue
            l2 = np.load(l2_path)
            
            if np.allclose(l1, l2, atol=1e-5):
                print(f"¡COINCIDENCIA ENCONTRADA!")
                print(f"Balanced: {sid} (Index {i})")
                print(f"Generated: {g_df.iloc[j]['sample_id']} (Index {j})")
                print(f"Source: {source_id}")
                return

if __name__ == "__main__":
    find_offset()
