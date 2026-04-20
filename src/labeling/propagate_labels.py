import pandas as pd
import os
from pathlib import Path

def propagate_labels(validated_csv, sequences_manifest, balanced_manifest):
    print(f"Propagating labels from {validated_csv}...")
    if not os.path.exists(validated_csv):
        print(f"Error: Validated CSV not found: {validated_csv}")
        return
        
    val_df = pd.read_csv(validated_csv)
    # Create mapping: sample_id -> label
    label_map = dict(zip(val_df['sample_id'], val_df['suggested_label']))
    
    # 1. Update manifest_mano_secuencias.csv
    if os.path.exists(sequences_manifest):
        seq_df = pd.read_csv(sequences_manifest)
        seq_df['label'] = seq_df['sample_id'].map(label_map)
        # Handle cases not in validated_csv (should not happen if all processed)
        seq_df['label'] = seq_df['label'].fillna('ambiguous')
        seq_df.to_csv(sequences_manifest, index=False)
        print(f"Updated {sequences_manifest}")
    
    # 2. Update manifest_balanced_blocks.csv (global manifest)
    if os.path.exists(balanced_manifest):
        bal_df = pd.read_csv(balanced_manifest)
        # Update labels only for 'mano' dataset
        bal_df.loc[bal_df['dataset'] == 'mano', 'label'] = bal_df.loc[bal_df['dataset'] == 'mano', 'sample_id'].map(label_map)
        bal_df['label'] = bal_df['label'].fillna('ambiguous')
        bal_df.to_csv(balanced_manifest, index=False)
        print(f"Updated {balanced_manifest}")

if __name__ == "__main__":
    VALIDATED = "output/mano_assisted_labeling_step_ab.csv"
    SEQUENCES = "output/manifest_mano_secuencias.csv"
    BALANCED = "output/manifest_balanced_blocks.csv"
    propagate_labels(VALIDATED, SEQUENCES, BALANCED)
