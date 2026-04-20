import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

def cluster_unknowns():
    csv_path = Path("output/mano_refined_suggestions.csv")
    if not csv_path.exists():
        print("❌ Error: No se encuentra el CSV de sugerencias.")
        return
        
    df = pd.read_csv(csv_path)
    unknown_df = df[df['suggested_label'] == 'unknown'].copy()
    
    if len(unknown_df) < 10:
        print("ℹ️ Muy pocas muestras unknown para clusterizar.")
        return

    print(f"🔬 Analizando clusters para {len(unknown_df)} muestras unknown...")
    
    # Seleccionar features para clustering
    features_cols = ['dist_ok', 'thumb_abduction', 'palm_arc', 'prox_T', 'prox_I', 'prox_M', 'prox_R', 'prox_P']
    X = unknown_df[features_cols].values
    
    # Escalar (opcional pero recomendado)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-6
    X_scaled = (X - X_mean) / X_std
    
    K = 6
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    unknown_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Obtener centroides (en escala original para que sean interpretables)
    centroids = unknown_df.groupby('cluster')[features_cols].mean()
    
    print("\n📊 Centroides de los clusters (Unknowns):")
    print(centroids.round(2))
    
    # Conteo por cluster
    print("\n📦 Distribución de muestras:")
    print(unknown_df['cluster'].value_counts())
    
    # Guardar resultados para inspección
    unknown_df.to_csv("output/unknown_clusters_analysis.csv", index=False)
    print("\n✅ Análisis guardado en output/unknown_clusters_analysis.csv")

if __name__ == "__main__":
    cluster_unknowns()
