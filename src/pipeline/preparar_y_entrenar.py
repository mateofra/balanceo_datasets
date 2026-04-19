# preparar_y_entrenar.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from src.stgcn.hand_graph import build_adjacency_matrix
from src.stgcn.stgcn_model import STGCN

MANIFEST_PATH = REPO_ROOT / 'output/manifest_unificado_final.csv'

# ── 1. Cargar manifiesto unificado final ────────────────────────────────
df = pd.read_csv(MANIFEST_PATH)

if 'path_secuencia' not in df.columns and 'path' in df.columns:
    df = df.rename(columns={'path': 'path_secuencia'})
if 'dataset' not in df.columns and 'source' in df.columns:
    df = df.rename(columns={'source': 'dataset'})
if 'condition' not in df.columns:
    if 'mst_imputed' in df.columns:
        df['condition'] = df['mst_imputed']
    else:
        df['condition'] = 'medio'
if 'mst' not in df.columns:
    df['mst'] = 0

train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df = df[df['split'] == 'val'].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)

print(f"Manifiesto: {MANIFEST_PATH}")
print(f"Total: {len(df)} | Clases: {df['label'].nunique()}")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(df['label'].value_counts().head(20))

# ── 2. Dataset ───────────────────────────────────────────────────────────
class GestureDataset(Dataset):
    def __init__(self, frame_df, class_to_idx):
        self.paths = frame_df['path_secuencia'].tolist()
        self.labels = [class_to_idx[l] for l in frame_df['label']]
        self.mst = frame_df['mst'].tolist()
        self.condition = frame_df['condition'].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = REPO_ROOT / self.paths[idx].replace('\\', '/')
        x = np.load(str(p))
        return torch.tensor(x, dtype=torch.float32), self.labels[idx], self.mst[idx]

clases = sorted(train_df['label'].unique())
class_to_idx = {c: i for i, c in enumerate(clases)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

train_set = GestureDataset(train_df, class_to_idx)
val_set = GestureDataset(val_df, class_to_idx)
test_set = GestureDataset(test_df, class_to_idx)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

# ── 3. Modelo ────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

A = build_adjacency_matrix().to(device)
model = STGCN(A, num_classes=len(clases)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# ── 4. Entrenamiento ─────────────────────────────────────────────────────
history = []
best_val = 0.0

for epoch in range(50):
    model.train()
    train_loss = 0.0
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        train_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    val_acc = correct / total
    avg_loss = train_loss / len(train_loader)
    history.append({'epoch': epoch + 1, 'loss': avg_loss, 'val_acc': val_acc})
    print(f"Epoch {epoch+1:03d}/50 | loss: {avg_loss:.4f} | val_acc: {val_acc:.3f}")

    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state': model.state_dict(),
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
        }, REPO_ROOT / 'output/best_stgcn_canonico.pth')

# ── 5. Evaluación final en test bloqueado ────────────────────────────────
ckpt = torch.load(REPO_ROOT / 'output/best_stgcn_canonico.pth', map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()
correct = total = 0
with torch.no_grad():
    for x, y, _ in test_loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

test_acc = correct / total
print(f"\nMejor val_acc: {best_val:.3f}")
print(f"Test accuracy (holdout): {test_acc:.3f}")

with open(REPO_ROOT / 'output/training_history_canonico.json', 'w', encoding='utf-8') as f:
    json.dump({
        'history': history,
        'best_val_acc': best_val,
        'test_acc': test_acc,
        'class_to_idx': class_to_idx,
    }, f, indent=2)

print('Guardado: output/best_stgcn_canonico.pth')
print('Guardado: output/training_history_canonico.json')
