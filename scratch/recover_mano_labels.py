
import pandas as pd
import os
import re

def recover_labels():
    print("🚀 Iniciando recuperación de etiquetas MANO...")
    
    # 1. Cargar el manifiesto balanceado (Piedra Rosetta)
    # Mapea 'mano_XXXXXXXX' -> 'synth_req_XXXXXXX'
    print("📦 Cargando manifest_mano_samples_balanced.csv...")
    balanced_df = pd.read_csv('/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/datasets/synthetic_mst/metadata/manifest_mano_samples_balanced.csv')
    
    # Extraer el ID de la petición de la ruta
    # path: datasets/synthetic_mst/images_blocks_qc_adjusted/synth_req_0000001.jpg
    balanced_df['request_id'] = balanced_df['path'].apply(lambda x: os.path.basename(x).replace('.jpg', ''))
    
    # 2. Cargar el manifiesto de peticiones (Fuente de etiquetas)
    # Mapea 'req_XXXXXXX' -> 'source_image_path'
    print("📦 Cargando manifest_synthetic_requests_blocks_qc_adjusted.csv...")
    requests_df = pd.read_csv('/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/datasets/synthetic_mst/metadata/manifest_synthetic_requests_blocks_qc_adjusted.csv')
    
    # Normalizar IDs para el merge (synth_req_0000001 vs req_0000001)
    requests_df['request_id_match'] = requests_df['request_id'].apply(lambda x: f"synth_{x}")
    
    # Extraer etiqueta de source_image_path
    # Ej: datasets/hagrid_sample_30k_384p/.../train_val_ok/... -> 'ok'
    def extract_label(path):
        if 'freihand' in str(path).lower():
            return 'hand' # Freihand no tiene gestos específicos en la ruta usualmente
        
        match = re.search(r'train_val_([a-z0-9_]+)', str(path))
        if match:
            label = match.group(1)
            # Limpiar variaciones como peace_inverted -> peace
            label = label.replace('_inverted', '')
            # Normalizar nombres comunes
            label_map = {
                'three2': 'three',
                'two_up': 'two_up'
            }
            return label_map.get(label, label)
        return 'unknown'

    requests_df['recovered_label'] = requests_df['source_image_path'].apply(extract_label)
    
    # 3. Cruzar datos
    print("🔄 Cruzando manifiestos...")
    mapping = pd.merge(
        balanced_df[['sample_id', 'request_id']], 
        requests_df[['request_id_match', 'recovered_label']], 
        left_on='request_id', 
        right_on='request_id_match'
    )
    
    label_map = dict(zip(mapping['sample_id'], mapping['recovered_label']))
    print(f"✅ Mapeo creado para {len(label_map)} muestras MANO.")

    # 4. Actualizar el manifiesto de entrenamiento
    print("📝 Actualizando manifiesto de entrenamiento...")
    train_manifest = pd.read_csv('/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/output/train_manifest_stgcn_secuencias.csv')
    
    count_updated = 0
    for idx, row in train_manifest.iterrows():
        s_id = row['sample_id']
        if s_id in label_map:
            current_label = row['label']
            recovered = label_map[s_id]
            
            # Solo actualizar si es unknown o genérica
            if current_label in ['unknown', 'hand', ''] or pd.isna(current_label):
                train_manifest.at[idx, 'label'] = recovered
                count_updated += 1

    # 5. Guardar resultados
    output_path = '/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/output/train_manifest_stgcn_secuencias_fixed.csv'
    train_manifest.to_csv(output_path, index=False)
    
    print(f"✨ ¡Éxito! Se han actualizado {count_updated} etiquetas.")
    print(f"📄 Nuevo manifiesto guardado en: {output_path}")
    
    # Mostrar distribución
    print("\n📊 Nueva distribución de etiquetas (Top 10):")
    print(train_manifest['label'].value_counts().head(10))

if __name__ == "__main__":
    recover_labels()
