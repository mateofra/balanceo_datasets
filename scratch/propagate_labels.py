import pandas as pd
from pathlib import Path

def propagate():
    manifest_path = Path("output/train_manifest_stgcn_secuencias.csv")
    suggestions_path = Path("output/mano_refined_suggestions.csv")
    
    if not manifest_path.exists() or not suggestions_path.exists():
        print("❌ Error: Faltan archivos necesarios para la propagación.")
        return
        
    df = pd.read_csv(manifest_path)
    sugg_df = pd.read_csv(suggestions_path)
    
    print("🚀 Iniciando propagación selectiva...")
    
    # Mapeo de sugerencias
    APPROVE = ['rock', 'two_up', 'call']
    EXCLUDE = ['bad_pose']
    
    # Crear un diccionario para búsqueda rápida
    sugg_map = sugg_df.set_index('sample_id')[['suggested_label', 'confidence']].to_dict('index')
    
    count_updated = 0
    count_excluded = 0
    
    # Asegurar columna label_source
    if 'label_source' not in df.columns:
        df['label_source'] = 'original'
        
    for idx, row in df.iterrows():
        sid = row['sample_id']
        if sid in sugg_map:
            sugg = sugg_map[sid]['suggested_label']
            
            if sugg in APPROVE:
                df.at[idx, 'label'] = sugg
                df.at[idx, 'label_source'] = 'heuristic_v2'
                count_updated += 1
            elif sugg in EXCLUDE:
                df.at[idx, 'quality_flag'] = 'excluded'
                df.at[idx, 'quality_detail'] = 'bad_pose_cluster'
                count_excluded += 1
                
    # Guardar
    df.to_csv(manifest_path, index=False)
    
    print(f"\n✅ Propagación completada:")
    print(f"🔹 Etiquetas actualizadas: {count_updated}")
    print(f"🔸 Muestras excluidas (bad_pose): {count_excluded}")
    
    print("\n📊 Nueva distribución de etiquetas (MANO):")
    print(df[df['dataset'] == 'mano']['label'].value_counts())
    
    print(f"\n📁 Manifiesto actualizado en {manifest_path}")

if __name__ == "__main__":
    propagate()
