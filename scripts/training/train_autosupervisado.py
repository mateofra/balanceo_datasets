import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stgcn.hand_graph import build_adjacency_matrix
from src.stgcn.stgcn_model import STGCN
import json


# ── Dataset ──────────────────────────────────────────────────────────────
class LandmarkDataset(Dataset):
    def __init__(self, manifest_path, quality_filter='real_3d_freihand'):
        df = pd.read_csv(manifest_path)
        if 'landmark_quality' in df.columns:
            df = df[df['landmark_quality'] == quality_filter]
        self.paths = df['path_secuencia'].tolist()
        self.mst   = df['mst'].tolist()
        print(f"Dataset cargado: {len(self.paths)} muestras")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx].replace('\\', '/'))  # (T, 21, 3)
        return torch.tensor(x, dtype=torch.float32), self.mst[idx]


# ── Tarea de predicción ───────────────────────────────────────────────────
def prepare_prediction_task(x):
    """
    Input:  x de shape (B, T, J, C)
    Tarea:  dado x[:, :-1, :, :] predice x[:, -1, :, :]
    Return: contexto (B, T-1, J, C), target (B, J, C)
    """
    context = x[:, :-1, :, :]   # frames 0..T-2
    target  = x[:, -1,  :, :]   # frame T-1
    return context, target


def reconstruction_loss(pred, target):
    """
    pred:   (B, J*C) salida del decoder
    target: (B, J, C) frame objetivo
    """
    pred = pred.view(pred.shape[0], 21, 3)
    return ((pred - target) ** 2).mean()


# ── Entrenamiento ─────────────────────────────────────────────────────────
def train(manifest_path='data/processed/secuencias_stgcn/manifest_secuencias.csv',
          epochs=30, batch_size=32, lr=1e-3, mask_ratio=0.15):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    training_dir = Path('output/training')
    training_dir.mkdir(parents=True, exist_ok=True)

    dataset    = LandmarkDataset(manifest_path)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    A     = build_adjacency_matrix().to(device)
    model = STGCN(A).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x, mst in dataloader:
            x = x.to(device)                          # (B, T, 21, 3)
            context, target = prepare_prediction_task(x)

            recon, attn = model(context)
            loss = reconstruction_loss(recon, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        history.append({'epoch': epoch+1, 'loss': avg_loss})
        print(f"Epoch {epoch+1:03d}/{epochs} | loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), str(training_dir / 'best_stgcn_autosup.pth'))

    with open(training_dir / 'training_history_autosup.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nMejor loss: {best_loss:.6f}")
    print("Modelo guardado: output/training/best_stgcn_autosup.pth")


if __name__ == '__main__':
    train()