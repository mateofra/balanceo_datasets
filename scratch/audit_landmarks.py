
import pandas as pd
import numpy as np
import os
from pathlib import Path

def audit_landmark_matching():
    # 1. Load the fixed manifest (what we just created)
    fixed_manifest = "/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/output/train_manifest_stgcn_secuencias_fixed.csv"
    df = pd.read_csv(fixed_manifest)
    
    # 2. Filter for MANO samples with heuristic labels
    mano_samples = df[(df['dataset'] == 'mano') & (df['label_source'] == 'heuristic_v3')].sample(n=10, random_state=42)
    
    print(f"🔬 Iniciando Auditoría Forense de Landmarks (10 muestras aleatorias)...")
    
    # 3. Pre-load all Hagrid/Freihand landmarks paths for searching
    # (Actually, let's just use the manifest to find original IDs)
    
    # We suspect the mapping is not direct, so we will use the landmark content as a hash
    
    matches = 0
    total = 0
    
    for _, row in mano_samples.iterrows():
        sid = row['sample_id']
        recovered_label = row['label']
        lm_path = f"/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/data/processed/landmarks/{sid}.npy"
        
        if not os.path.exists(lm_path): continue
        l1 = np.load(lm_path)
        
        # Search for this landmark in the original datasets
        # To speed up, we look at the source_sample_id if we could find it...
        # But we don't have the mapping yet.
        # So let's look at the "original" labeled samples in the SAME manifest!
        
        originals = df[df['label_source'] == 'original']
        
        found = False
        for _, orig_row in originals.iterrows():
            orig_path = f"/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/{orig_row['path_landmarks']}"
            if not os.path.exists(orig_path): continue
            
            l2 = np.load(orig_path)
            
            if np.allclose(l1, l2, atol=1e-5):
                real_label = orig_row['label']
                status = "✅ MATCH" if real_label == recovered_label else "❌ MISMATCH"
                print(f"Sample: {sid} | Recov: {recovered_label} | Real: {real_label} | {status}")
                if real_label == recovered_label: matches += 1
                found = True
                break
        
        if not found:
            print(f"Sample: {sid} | No se encontró el original (pose única)")
        else:
            total += 1

    if total > 0:
        print(f"\n📊 Resultado Final: {matches}/{total} ({100*matches/total:.1f}%) de coincidencia con Ground Truth.")
    else:
        print("\n❌ No se pudieron validar muestras (posiblemente por falta de archivos originales).")

if __name__ == "__main__":
    audit_landmark_matching()
