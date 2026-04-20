import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

# Landmark indices for MANO (21 points)
# 0: Wrist
# 1-4: Thumb
# 5-8: Index
# 9-12: Middle
# 13-16: Ring
# 17-20: Pinky
FINGERS = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}

def calculate_finger_extension(landmarks_3d):
    """
    Calculates extension score for each finger [0, 1].
    Based on distance from tip to MCP relative to palm size.
    """
    wrist = landmarks_3d[0]
    middle_mcp = landmarks_3d[9]
    palm_size = np.linalg.norm(middle_mcp - wrist)
    
    extensions = {}
    for finger, indices in FINGERS.items():
        mcp = landmarks_3d[indices[0]]
        tip = landmarks_3d[indices[3]]
        # Distance MCP to Tip
        dist = np.linalg.norm(tip - mcp)
        # Simple heuristic: compare to a reference length (approx 0.9 * palm_size for extended)
        extensions[finger] = dist / (palm_size * 0.9)
    
    return extensions

def suggest_label_rules(extensions):
    """
    Rule-based labeling based on finger extensions.
    """
    ext = {f: extensions[f] > 0.6 for f in extensions} # Binary extension threshold
    
    # Core separable classes
    if all(not ext[f] for f in ['index', 'middle', 'ring', 'pinky']):
        return "fist", 0.9
    if all(ext[f] for f in ['index', 'middle', 'ring', 'pinky']):
        if ext['thumb']:
            return "palm", 0.9
        else:
            return "four", 0.8
            
    if ext['index'] and not any(ext[f] for f in ['middle', 'ring', 'pinky']):
        return "one", 0.9
        
    if ext['index'] and ext['middle'] and not any(ext[f] for f in ['ring', 'pinky']):
        return "peace", 0.8
        
    if ext['index'] and ext['pinky'] and not any(ext[f] for f in ['middle', 'ring']):
        return "rock", 0.8
        
    if all(ext[f] for f in ['middle', 'ring', 'pinky']) and not ext['index']:
        return "ok", 0.7
        
    if ext['index'] and ext['middle'] and ext['ring'] and not ext['pinky']:
        return "three", 0.7

    return "ambiguous", 0.3

def process_labeling(manifest_path, output_path):
    print(f"Loading manifest: {manifest_path}")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    
    results = []
    embeddings = []
    
    print("Extracting features and suggesting labels...")
    for idx, row in df.iterrows():
        lm_path = row['path_landmarks']
        lm_path = lm_path.replace('\\', '/')
        
        if not os.path.exists(lm_path):
            continue
            
        landmarks_3d = np.load(lm_path)
        
        # Step A: Geometry
        extensions = calculate_finger_extension(landmarks_3d)
        label, confidence = suggest_label_rules(extensions)
        
        # Step B: Prepare for clustering
        wrist = landmarks_3d[0]
        middle_mcp = landmarks_3d[9]
        palm_size = np.linalg.norm(middle_mcp - wrist)
        if palm_size < 1e-6: palm_size = 1.0
        norm_lm = (landmarks_3d - wrist) / palm_size
        
        results.append({
            'sample_id': row['sample_id'],
            'suggested_label': label,
            'confidence': confidence,
            'ext_index': extensions['index'],
            'ext_middle': extensions['middle'],
            'ext_ring': extensions['ring'],
            'ext_pinky': extensions['pinky'],
            'ext_thumb': extensions['thumb']
        })
        embeddings.append(norm_lm.flatten())
        
    if not results:
        print("No samples processed.")
        return

    results_df = pd.DataFrame(results)
    embeddings = np.array(embeddings)
    
    # Step B: Clustering
    print(f"Running K-Means for {len(embeddings)} samples...")
    n_clusters = min(len(embeddings), 12)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    results_df['cluster'] = clusters
    
    cluster_stats = []
    for c in range(n_clusters):
        c_samples = results_df[results_df['cluster'] == c]
        if len(c_samples) == 0: continue
        
        counts = c_samples['suggested_label'].value_counts()
        top_label = counts.idxmax()
        coherence = counts.max() / len(c_samples)
        
        cluster_stats.append({
            'cluster': int(c),
            'dominant_label': str(top_label),
            'coherence': float(coherence),
            'count': int(len(c_samples))
        })
        
        if coherence < 0.6:
            results_df.loc[results_df['cluster'] == c, 'suggested_label'] = 'ambiguous_cluster'

    results_df.to_csv(output_path, index=False)
    
    summary = {
        'total_samples': int(len(results_df)),
        'label_distribution': results_df['suggested_label'].value_counts().to_dict(),
        'cluster_coherence': cluster_stats
    }
    
    summary_path = output_path.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Results saved to {output_path}")
    print("Label distribution summary:")
    print(json.dumps(summary['label_distribution'], indent=2))

if __name__ == "__main__":
    MANIFEST = "output/manifest_mano_secuencias.csv"
    OUTPUT = "output/mano_assisted_labeling_step_ab.csv"
    process_labeling(MANIFEST, OUTPUT)
