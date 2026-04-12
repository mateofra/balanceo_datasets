#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (0,17), (17,18), (18,19), (19,20)
]

def load_image(dataset: str, sample_id: str, label: str) -> np.ndarray | None:
    if dataset == "freihand":
        numeric_part = sample_id.replace("freihand_", "")
        img_path = Path(f"datasets/FreiHAND_pub_v2/training/rgb/{numeric_part}.jpg")
    else:
        img_path = Path(f"data/raw/images/{label}/{sample_id}.jpg")
        if not img_path.exists():
            img_path = Path(f"datasets/hagrid_images/{label}/{sample_id}.jpg")
            
    if not img_path.exists():
        # Fallback to PNG?
        png_path = img_path.with_suffix(".png")
        if png_path.exists():
            img_path = png_path
        else:
            return None
            
    try:
        image = Image.open(str(img_path))
        return np.array(image.convert("RGB"))
    except Exception:
        return None

def main():
    manifest_paths = [
        Path("output/test_run_manifest.csv"),
        Path("csv/train_manifest_balanceado_freihand_hagrid.csv"),
        Path("output/train_manifest_stgcn_activo.csv")
    ]
    
    manifest_path = None
    for p in manifest_paths:
        if p.exists():
            manifest_path = p
            break
            
    if not manifest_path:
        print("No se encontró ningún manifiesto.")
        return

    print(f"Cargando {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    # Estandarizar nombre de columna 'dataset' vs 'source'
    source_col = "dataset" if "dataset" in df.columns else "source"
    label_col = "label" if "label" in df.columns else "gesture"
    path_col = "path_landmarks" if "path_landmarks" in df.columns else None
    
    samples = []
    for dataset in ["freihand", "hagrid"]:
        for condition in ["claro", "medio", "oscuro"]:
            def get_cond(mst_val):
                if pd.isna(mst_val): return "unknown"
                v = float(mst_val)
                if v <= 4: return "claro"
                if v <= 7: return "medio"
                return "oscuro"
                
            subset = df[(df[source_col] == dataset) & (df["mst"].apply(get_cond) == condition)]
            
            if not subset.empty:
                chosen = subset.sample(1, random_state=random.randint(0,1000)).iloc[0]
                samples.append(chosen)
                
    if not samples:
        print("No hay samples validos.")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Validación Esquelética Post-Balanceo por Bloque MST")
    
    for ax, row in zip(axes.flatten()[:len(samples)], samples):
        dataset = row[source_col]
        sample_id = row["sample_id"]
        label = row.get(label_col, "unknown")
        m_val = float(row['mst']) if not pd.isna(row['mst']) else 0
        condition = "claro" if m_val <= 4 else ("medio" if m_val <= 7 else "oscuro")
        
        img = load_image(dataset, sample_id, label)
        if img is None:
            ax.set_title(f"IMG MISSING: {dataset} | {condition}\n{sample_id}")
            ax.axis("off")
            continue
            
        h, w = img.shape[:2]
        
        ax.imshow(img)
        ax.set_title(f"{dataset.upper()} | MST: {int(m_val)} ({condition})\nID: {sample_id}")
        ax.axis("off")
        
        npy_path = row[path_col] if path_col and pd.notna(row[path_col]) else None
        if not npy_path or not Path(npy_path).exists():
            if dataset == "freihand":
                numeric_part = str(sample_id).replace("freihand_", "")
                npy_path = f"data/processed/landmarks/freihand/{numeric_part}.npy"
            else:
                npy_path = f"data/processed/landmarks/hagrid/{label}/{sample_id}.npy"
                
        npy_path = Path(npy_path)
        if npy_path.exists():
            lm = np.load(npy_path)
            
            xs = lm[:, 0]
            ys = lm[:, 1]
            
            # Autodetetect if it's normalized 0-1
            if np.max(np.abs(xs)) <= 2.0 and np.max(np.abs(ys)) <= 2.0:
                xs = xs * w
                ys = ys * h
            else:
                # If these are absolute physical coords un-mapped (like FreiHAND xyz)
                # It will cluster tightly near [0,0] if not projected via camera intrinsics
                pass
            
            ax.scatter(xs, ys, s=20, c='red')
            for (i, j) in HAND_CONNECTIONS:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], c='lime', linewidth=2)
        else:
            ax.set_title(ax.get_title() + "\n(NO LMK FILE)")
    
    plt.tight_layout()
    out_path = Path("graficos/validacion_escanometria_mst.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Grafico guardado en {out_path}")

if __name__ == "__main__":
    main()
