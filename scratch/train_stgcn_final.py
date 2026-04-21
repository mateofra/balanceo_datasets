
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- CONFIGURACIÓN ---
MANIFEST_PATH = "output/train_manifest_stgcn_secuencias_fixed.csv"
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10

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
        # Corregir ruta: añadir 'data' si no está
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

class SimpleMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Input: 3 * 16 * 21 = 1008
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(1008, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

def train():
    print(f"📡 Usando dispositivo: {DEVICE}")
    base_path = Path("/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets")
    df = pd.read_csv(MANIFEST_PATH)
    
    # Limpiar clases ambiguas y asegurar tipos
    df = df.dropna(subset=['label'])
    # Mantenemos 'hand' para no perder FreiHand, pero filtramos 'unknown' y 'no_gesture'
    df = df[~df['label'].isin(['unknown', 'no_gesture', ''])].copy()
    df['label'] = df['label'].astype(str)
    
    # Split Train/Val (85/15)
    train_df = df.sample(frac=0.85, random_state=42)
    val_df = df.drop(train_df.index)
    
    classes = sorted(df['label'].unique())
    print(f"📦 Dataset: {len(train_df)} train | {len(val_df)} val")
    print(f"🏷️ Clases ({len(classes)}): {classes}")
    
    # Debug: Ver fuentes en val
    print(f"🔍 Fuentes en Validación: {val_df['label_source'].value_counts().to_dict()}")

    # Sampler para balanceo de clases
    class_counts = train_df['label'].value_counts()
    weights = 1.0 / class_counts
    sample_weights = train_df['label'].map(weights).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(STGCNDataset(train_df, base_path), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(STGCNDataset(val_df, base_path), batch_size=BATCH_SIZE)

    model = SimpleMLP(len(df['label'].unique())).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc_original = 0
    patience_counter = 0
    
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
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
        
        # Métricas por fuente
        source_metrics = {
            'original': {'correct': 0, 'total': 0},
            'heuristic_v2': {'correct': 0, 'total': 0},
            'heuristic_v3': {'correct': 0, 'total': 0}
        }
        
        all_preds = []
        all_targets = []
        
        first_batch = True
        with torch.no_grad():
            for data, target, sources in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                
                if first_batch:
                    # print(f"DEBUG: first batch sources: {sources[:5]}")
                    first_batch = False
                
                # Global
                val_correct_global += pred.eq(target.view_as(pred)).sum().item()
                val_total_global += target.size(0)
                
                # Por fuente
                for i in range(len(sources)):
                    src = str(sources[i]).strip()
                    is_correct = pred[i].item() == target[i].item()
                    if src in source_metrics:
                        source_metrics[src]['total'] += 1
                        if is_correct: source_metrics[src]['correct'] += 1
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

        val_acc_global = 100. * val_correct_global / val_total_global
        
        # Métrica Maestra (Solo original)
        orig_data = source_metrics['original']
        val_acc_original = (100. * orig_data['correct'] / orig_data['total']) if orig_data['total'] > 0 else 0
        
        # Heurísticas
        acc_v3 = (100. * source_metrics['heuristic_v3']['correct'] / source_metrics['heuristic_v3']['total']) if source_metrics['heuristic_v3']['total'] > 0 else 0

        print(f"Epoch {epoch+1:02d} | Train: {train_acc:5.2f}% | Val Global: {val_acc_global:5.2f}% | Val REAL: {val_acc_original:5.2f}% | Val Heur: {acc_v3:5.2f}%")
        
        # Guardar Checkpoint
        if val_acc_original > best_val_acc_original:
            best_val_acc_original = val_acc_original
            torch.save(model.state_with_dict(), "models/best_stgcn_real_val.pt") if hasattr(model, 'state_with_dict') else torch.save(model.state_dict(), "models/best_stgcn_real_val.pt")
            print(f"  ⭐ Guardado mejor modelo (Acc Real: {val_acc_original:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping disparado en época {epoch+1}")
            break
            
    # MATRIZ DE CONFUSIÓN FINAL SOBRE REAL
    print("\n📊 Generando Matriz de Confusión Final (Subconjunto REAL)...")
    # Filtrar solo 'original' para la matriz final
    val_orig_df = val_df[val_df['label_source'] == 'original'].copy()
    val_orig_loader = DataLoader(STGCNDataset(val_orig_df, base_path), batch_size=BATCH_SIZE)
    
    model.load_state_dict(torch.load("models/best_stgcn_real_val.pt"))
    model.eval()
    
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target, _ in val_orig_loader:
            data = data.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1).cpu().numpy()
            y_true.extend(target.numpy())
            y_pred.extend(pred)
            
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(df['label'].unique())
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión - Validación HAGRID (Original)')
    plt.savefig("output/confusion_matrix_final.png")
    print("📈 Matriz de confusión guardada en output/confusion_matrix_final.png")
    
    print("\n✅ Entrenamiento Finalizado.")
    print(f"Mejor Accuracy en Datos Reales: {best_val_acc_original:.2f}%")

if __name__ == "__main__":
    train()
