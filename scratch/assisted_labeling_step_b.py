import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from tqdm import tqdm

def run_clustering():
    manifest_path = Path("output/train_manifest_stgcn_secuencias.csv")
    suggestions_path = Path("output/mano_auto_suggestions.csv")
    
    if not suggestions_path.exists():
        print("❌ Error: Ejecuta primero el Paso A (auto-sugerencia).")
        return
        
    df = pd.read_csv(manifest_path)
    sugg_df = pd.read_csv(suggestions_path)
    
    # Unir sugerencias con el manifiesto para tener los paths
    mano_df = pd.merge(sugg_df, df[['sample_id', 'path_landmarks']], on='sample_id')
    
    print(f"🧩 Preparando embeddings para {len(mano_df)} muestras...")
    embeddings = []
    valid_ids = []
    
    for idx, row in tqdm(mano_df.iterrows(), total=len(mano_df)):
        try:
            lms = np.load(row['path_landmarks']) # (21, 3)
            # Normalización básica: centrar en el wrist y escalar por distancia palma
            lms_norm = lms - lms[0]
            scale = np.linalg.norm(lms[5] - lms[0]) + 1e-6
            lms_norm /= scale
            embeddings.append(lms_norm.flatten())
            valid_ids.append(row['sample_id'])
        except:
            continue
            
    embeddings = np.array(embeddings)
    
    print("🚀 Ejecutando K-Means (K=12)...")
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    mano_df = mano_df[mano_df['sample_id'].isin(valid_ids)].copy()
    mano_df['cluster'] = clusters
    
    # Medir coherencia por cluster
    cluster_stats = []
    for c in range(12):
        c_samples = mano_df[mano_df['cluster'] == c]
        dominant = c_samples['suggested_label'].value_counts()
        top_label = dominant.index[0]
        coherence = dominant.iloc[0] / len(c_samples)
        
        cluster_stats.append({
            'cluster': c,
            'dominant_label': top_label,
            'coherence': coherence,
            'size': len(c_samples)
        })
    
    stats_df = pd.DataFrame(cluster_stats)
    print("\n📊 Estadísticas de Clusters:")
    print(stats_df)
    
    # Marcar ambiguos
    mano_df['final_suggestion'] = mano_df.apply(
        lambda r: r['suggested_label'] if stats_df.loc[r['cluster'], 'coherence'] > 0.6 else 'ambiguous', 
        axis=1
    )
    
    output_path = Path("output/mano_clustering_results.csv")
    mano_df.to_csv(output_path, index=False)
    print(f"\n✅ Resultados de clustering guardados en {output_path}")

if __name__ == "__main__":
    run_clustering()
