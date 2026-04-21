
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Asegurar que el root del repo esté en el path para los imports
REPO_ROOT = Path("/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stgcn.hand_graph import build_adjacency_matrix
from src.stgcn.stgcn_model import RealSTGCN

# --- CONFIGURACIÓN ---
MANIFEST_PATH = REPO_ROOT / "output/train_manifest_stgcn_secuencias_fixed.csv"
BATCH_SIZE = 64 # Reducimos un poco el batch para ST-GCN
EPOCHS = 50
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 15
COMMON_LABELS = ['fist', 'peace', 'one', 'three', 'four', 'palm', 'like']

class STGCNDataset(Dataset):
    def __init__(self, df, base_path):
        self.df = df
        self.base_path = base_path
        self.classes = sorted(df['label'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row['path_secuencia']
        if not rel_path.startswith("data/"):
            rel_path = f"data/{rel_path}"
            
        seq_path = self.base_path / rel_path
        
        # Cargar secuencia (T, V, C) -> (16, 21, 3)
        data = np.load(seq_path).astype(np.float32)
        
        # Normalización relativa a la muñeca (punto 0)
        wrist = data[:, 0:1, :]
        data = data - wrist
        
        # Reorganizar a (C, T, V) para el modelo (3, 16, 21)
        data = data.transpose(2, 0, 1)
        
        label = self.class_to_idx[row['label']]
        source = str(row['label_source']).strip()
        
        return torch.from_numpy(data), torch.tensor(label), source

def train():
    print(f"📡 Usando dispositivo: {DEVICE}")
    df = pd.read_csv(MANIFEST_PATH)
    
    # Limpiar clases ambiguas y asegurar tipos
    df = df.dropna(subset=['label'])
    # Filtrar solo por etiquetas comunes para el experimento 'Clean 7'
    df = df[df['label'].isin(COMMON_LABELS)].copy()
    df['label'] = df['label'].astype(str)
    
    # Split Train/Val (85/15)
    train_df = df.sample(frac=0.85, random_state=42)
    val_df = df.drop(train_df.index)
    
    classes = sorted(df['label'].unique())
    num_classes = len(classes)
    print(f"📦 Dataset: {len(train_df)} train | {len(val_df)} val")
    print(f"🏷️ Clases ({num_classes}): {classes}")
    print(f"🔍 Fuentes en Validación: {val_df['label_source'].value_counts().to_dict()}")

    # Sampler para balanceo de clases
    class_counts = train_df['label'].value_counts()
    weights = 1.0 / class_counts
    sample_weights = train_df['label'].map(weights).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(STGCNDataset(train_df, REPO_ROOT), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(STGCNDataset(val_df, REPO_ROOT), batch_size=BATCH_SIZE)

    # Construir Grafo y Modelo
    A = build_adjacency_matrix()
    model = RealSTGCN(num_classes=num_classes, adjacency=A).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc_original = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for data, target, _ in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        train_acc = 100. * correct / total
        
        # VALIDACIÓN SEGMENTADA
        model.eval()
        val_correct_global = 0
        val_total_global = 0
        source_metrics = {
            'original': {'correct': 0, 'total': 0},
            'heuristic_v2': {'correct': 0, 'total': 0},
            'heuristic_v3': {'correct': 0, 'total': 0}
        }
        
        with torch.no_grad():
            for data, target, sources in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output, _ = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                
                val_correct_global += pred.eq(target.view_as(pred)).sum().item()
                val_total_global += target.size(0)
                
                for i in range(len(sources)):
                    src = sources[i].strip()
                    is_correct = pred[i].item() == target[i].item()
                    if src in source_metrics:
                        source_metrics[src]['total'] += 1
                        if is_correct: source_metrics[src]['correct'] += 1

        val_acc_global = 100. * val_correct_global / val_total_global
        
        # Métrica REAL (Hagrid/FreiHand)
        orig = source_metrics['original']
        val_acc_original = (100. * orig['correct'] / orig['total']) if orig['total'] > 0 else 0
        
        # Métrica Heurística (MANO)
        heur = source_metrics['heuristic_v3']
        val_acc_heur = (100. * heur['correct'] / heur['total']) if heur['total'] > 0 else 0

        print(f"Epoch {epoch+1:02d} | Train: {train_acc:5.2f}% | Val Global: {val_acc_global:5.2f}% | Val REAL: {val_acc_original:5.2f}% | Val MANO: {val_acc_heur:5.2f}%")
        
        # Guardar Checkpoint Basado en REAL
        if val_acc_original > best_val_acc_original:
            best_val_acc_original = val_acc_original
            torch.save(model.state_dict(), REPO_ROOT / "models/best_real_stgcn.pt")
            print(f"  ⭐ Guardado mejor modelo (Acc Real: {val_acc_original:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping disparado en época {epoch+1}")
            break
            
    # MATRIZ DE CONFUSIÓN FINAL
    print("\n📊 Generando Matriz de Confusión Final (Subconjunto REAL)...")
    val_orig_df = val_df[val_df['label_source'] == 'original'].copy()
    val_orig_loader = DataLoader(STGCNDataset(val_orig_df, REPO_ROOT), batch_size=BATCH_SIZE)
    
    model.load_state_dict(torch.load(REPO_ROOT / "models/best_real_stgcn.pt"))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, target, _ in val_orig_loader:
            data = data.to(DEVICE)
            output, _ = model(data)
            pred = output.argmax(dim=1).cpu().numpy()
            y_true.extend(target.numpy())
            y_pred.extend(pred)
            
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('ST-GCN: Matriz de Confusión - Validación HAGRID (Original)')
    plt.savefig(REPO_ROOT / "output/confusion_matrix_stgcn.png")
    print(f"📈 Matriz de confusión guardada en output/confusion_matrix_stgcn.png")
    
    print(f"\n✅ Entrenamiento Finalizado. Mejor Accuracy Real: {best_val_acc_original:.2f}%")

if __name__ == "__main__":
    train()
