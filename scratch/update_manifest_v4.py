
import pandas as pd
from pathlib import Path

def update_manifest():
    manifest_path = "output/train_manifest_stgcn_secuencias_fixed.csv"
    suggestions_path = "output/mano_refined_suggestions.csv"
    
    df = pd.read_csv(manifest_path)
    sug_df = pd.read_csv(suggestions_path)
    
    # Crear mapeo de sample_id -> suggested_label
    mapping = dict(zip(sug_df['sample_id'], sug_df['suggested_label']))
    
    # Actualizar solo para el dataset MANO
    def update_label(row):
        if row['dataset'] == 'mano' and row['sample_id'] in mapping:
            new_label = mapping[row['sample_id']]
            if new_label not in ['unknown', 'bad_pose']:
                return new_label, 'heuristic_v4'
        return row['label'], row['label_source']

    # Aplicar actualización
    updates = df.apply(update_label, axis=1)
    df['label'] = [u[0] for u in updates]
    df['label_source'] = [u[1] for u in updates]
    
    # Guardar manifest refinado
    df.to_csv(manifest_path, index=False)
    print(f"✅ Manifiesto actualizado con heuristic_v4.")
    print(f"Distribución MANO (v4):")
    print(df[df['dataset'] == 'mano']['label'].value_counts())

if __name__ == "__main__":
    update_manifest()
