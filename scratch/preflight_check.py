import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Mock CONFIG
CONFIG = {
    "MANIFEST": "output/train_manifest_stgcn_secuencias.csv",
    "SECUENCIAS_DIR": "data/processed/secuencias_stgcn",
}

# Import src
ROOT_DIR = Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.stgcn.stgcn_model import RealSTGCN
from src.stgcn.hand_graph import build_adjacency_matrix

def preflight():
    device = torch.device("cpu")
    
    # 1. Load and Filter
    df = pd.read_csv(CONFIG["MANIFEST"])
    initial_count = len(df)
    df = df[df['quality_flag'] != 'excluded'].copy()
    df['label'] = df['label'].fillna('unknown').astype(str)
    
    print(f"✅ Muestras excluidas: {initial_count - len(df)}")
    print(f"✅ Muestras efectivas: {len(df)}")
    
    # 2. Distribution
    print("\n📊 Distribución de clases en dataset filtrado:")
    print(df['label'].value_counts())
    
    unique_labels = sorted(df['label'].unique())
    num_classes = len(unique_labels)
    
    # 3. Model Init & Forward Pass
    adj = build_adjacency_matrix().to(device)
    model = RealSTGCN(
        num_classes=num_classes,
        adjacency=adj,
        in_channels=3,
        dropout=0.3
    ).to(device)
    
    print("\n🚀 Probando Forward Pass...")
    # Batch size 4, 3 channels, 16 frames, 21 nodes
    dummy_input = torch.randn(4, 3, 16, 21).to(device)
    
    model.eval()
    with torch.no_grad():
        logits, attention = model(dummy_input)
    
    print(f"✅ Forward Pass exitoso!")
    print(f"   Shape de salida (logits): {logits.shape}")
    print(f"   Shape de atención: {attention.shape}")
    
    if logits.shape == (4, num_classes):
        print("\n✨ ¡TODO LISTO PARA EL ENTRENAMIENTO! ✨")
    else:
        print("\n❌ Error en dimensiones de salida.")

if __name__ == "__main__":
    preflight()
