
import os, sys, json, torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

# Root path setup
ROOT_DIR = Path("/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets")
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

from src.stgcn.stgcn_model import RealSTGCN
from src.stgcn.hand_graph import build_adjacency_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Load Data
MANIFEST = ROOT_DIR / "output/train_manifest_stgcn_secuencias_fixed.csv"
df = pd.read_csv(MANIFEST)
df = df[df['quality_flag'] != 'excluded'].copy()

print("--- AUDITORÍA DE ETIQUETAS ---")
df['label'] = df['label'].fillna('unknown')

# EXCLUDE ONLY UNKNOWN/HAND FOR CLEAN DIAGNOSIS
EXCLUDE_CLASSES = ['hand', 'unknown', 'no_gesture']
df = df[~df['label'].isin(EXCLUDE_CLASSES)].copy()

print(f"Total muestras para entrenamiento: {len(df)}")
print(f"Datasets: {df['dataset'].value_counts().to_dict()}")

le = LabelEncoder()
df['label_idx'] = le.fit_transform(df['label'])
unique_labels = le.classes_.tolist()
label_to_idx = {name: i for i, name in enumerate(unique_labels)}

print("Distribución Top 5:")
print(df['label'].value_counts().head(5))

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# 2. Dataset Class
class STGCNDataset(Dataset):
    def __init__(self, manifest_df, label_map):
        self.df = manifest_df
        self.label_map = label_map
        # IMPORTANT: Fix path
        self.base_dir = ROOT_DIR / "data"
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Manifest has 'processed/secuencias_stgcn/...'
        # We need 'data/processed/secuencias_stgcn/...'
        rel_path = row['path_secuencia']
        if not str(rel_path).startswith("data/"):
            path = self.base_dir / rel_path
        else:
            path = ROOT_DIR / rel_path
            
        if not path.exists():
            # Try one more fallback
            path = ROOT_DIR / "data" / "processed" / "secuencias_stgcn" / Path(rel_path).name
            
        if not path.exists():
            raise FileNotFoundError(f"Missing sequence: {path}")
            
        seq = np.load(path).astype(np.float32)
        
        # Normalización relativa a la muñeca
        # seq: (T, V, C)
        seq = seq - seq[:, 0:1, :]
        
        x = torch.from_numpy(np.transpose(seq, (2, 0, 1))).float()
        y = torch.tensor(self.label_map[str(row['label'])], dtype=torch.long)
        return x, y

# Balanced Sampling
class_counts = train_df['label_idx'].value_counts().to_dict()
weights = 1.0 / torch.tensor([class_counts[i] for i in range(len(unique_labels))], dtype=torch.float)
sample_weights = weights[train_df['label_idx'].values]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(STGCNDataset(train_df, label_to_idx), batch_size=128, sampler=sampler)
val_loader = DataLoader(STGCNDataset(val_df, label_to_idx), batch_size=128, shuffle=False)

# 3. Model (MLP for diagnosis)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): 
        return self.net(x), None

# Input dim = 3 (xyz) * 16 (frames) * 21 (nodes) = 1008
model = SimpleMLP(1008, len(unique_labels)).to(device)

# 4. Training Loop
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("\n🚀 Iniciando entrenamiento (MLP Diagnóstico)...")
for epoch in range(20):
    model.train()
    correct = 0; total = 0; train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        
        train_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0); correct += (pred == labels).sum().item()
    
    model.eval()
    v_correct = 0; v_total = 0; v_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            v_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            v_total += labels.size(0); v_correct += (pred == labels).sum().item()
    
    print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {100*correct/total:5.2f}% | Val Acc: {100*v_correct/v_total:5.2f}%")

print("\n✅ Entrenamiento finalizado.")
