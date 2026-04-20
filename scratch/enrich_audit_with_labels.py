import pandas as pd
import json
from pathlib import Path
import sys

def enrich():
    audit_path = Path("csv/mst_real_dataset.csv")
    if not audit_path.exists():
        print(f"Error: No se encuentra {audit_path}")
        return

    print(f"Enriqueciendo {audit_path} con etiquetas...")
    df = pd.read_csv(audit_path)
    
    # 1. Cargar etiquetas de HaGRID
    hagrid_labels = {}
    ann_dir = Path("datasets/ann_subsample")
    if ann_dir.exists():
        print("Cargando etiquetas de HaGRID...")
        for ann_file in ann_dir.glob("*.json"):
            with open(ann_file, "r") as f:
                data = json.load(f)
                for img_id, info in data.items():
                    if "labels" in info and info["labels"]:
                        hagrid_labels[img_id.lower()] = info["labels"][0]
    
    # 2. Aplicar etiquetas
    def get_label(row):
        sid = str(row["sample_id"]).lower()
        if row["dataset"] == "freihand":
            return "hand"
        return hagrid_labels.get(sid, "unknown")

    df["label"] = df.apply(get_label, axis=1)
    
    # 3. Guardar
    df.to_csv(audit_path, index=False)
    print(f"Done! {len(df)} filas actualizadas en {audit_path}")

if __name__ == "__main__":
    enrich()
