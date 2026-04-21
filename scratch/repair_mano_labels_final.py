
import pandas as pd
from pathlib import Path

def repair_labels():
    manifest_path = Path("output/train_manifest_stgcn_secuencias.csv")
    suggestions_path = Path("output/mano_refined_suggestions.csv")
    
    if not manifest_path.exists() or not suggestions_path.exists():
        print("❌ Error: Faltan archivos necesarios.")
        return
        
    df = pd.read_csv(manifest_path)
    sugg_df = pd.read_csv(suggestions_path)
    
    print(f"🚀 Iniciando reparación masiva de etiquetas MANO...")
    print(f"Muestras originales: {len(df)}")
    
    # Mapeo de sugerencias a etiquetas válidas
    # Algunos nombres pueden variar un poco
    REMAP = {
        'fist': 'fist',
        'rock': 'rock',
        'two_up': 'two_up',
        'one': 'one',
        'three': 'three',
        'call': 'call',
        'like': 'like',
        'peace': 'peace',
        'four': 'four',
        'palm': 'palm'
    }
    
    # Crear diccionario de búsqueda
    sugg_map = sugg_df.set_index('sample_id')['suggested_label'].to_dict()
    conf_map = sugg_df.set_index('sample_id')['confidence'].to_dict()
    
    updated = 0
    
    for idx, row in df.iterrows():
        sid = row['sample_id']
        # Solo actualizar si es mano y la etiqueta está vacía o es unknown
        if row['dataset'] == 'mano' and (pd.isna(row['label']) or row['label'] == 'unknown'):
            if sid in sugg_map:
                label = sugg_map[sid]
                conf = conf_map.get(sid, 0)
                
                if label in REMAP and conf >= 0.5:
                    df.at[idx, 'label'] = REMAP[label]
                    df.at[idx, 'label_source'] = 'heuristic_v3'
                    updated += 1
                elif label == 'bad_pose':
                    df.at[idx, 'quality_flag'] = 'excluded'
                    df.at[idx, 'quality_detail'] = 'bad_pose_heuristic'
                    updated += 1

    # Guardar resultado
    output_path = Path("output/train_manifest_stgcn_secuencias_fixed.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Reparación completada:")
    print(f"🔹 Etiquetas actualizadas/muestras corregidas: {updated}")
    print(f"📁 Nuevo manifiesto guardado en: {output_path}")
    
    print("\n📊 Nueva distribución de etiquetas (Dataset MANO):")
    print(df[df['dataset'] == 'mano']['label'].value_counts())
    
    print("\n📊 Distribución TOTAL del dataset:")
    print(df['label'].value_counts().head(10))

if __name__ == "__main__":
    repair_labels()
