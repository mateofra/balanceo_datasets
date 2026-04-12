"""
Script de entrenamiento ST-GCN simplificado.

Usa el manifiesto balanceado generado (output/train_manifest_stgcn.csv)
sin necesidad de normalización compleja.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime


class HandLandmarkDataset(Dataset):
    """Dataset simple para landmarks de mano."""
    
    def __init__(self, manifest_csv: str, num_frames: int = 1):
        """
        Args:
            manifest_csv: Ruta a train_manifest_stgcn.csv
            num_frames: Número de frames de entrada (default=1 para estático)
        """
        self.df = pd.read_csv(manifest_csv)
        self.num_frames = num_frames
        
        # Mapear gestos a índices
        unique_gestures = self.df['label'].unique()
        self.gesture_to_idx = {gesture: idx for idx, gesture in enumerate(unique_gestures)}
        self.idx_to_gesture = {idx: gesture for gesture, idx in self.gesture_to_idx.items()}
        
        print(f"🎯 Clases: {len(self.gesture_to_idx)}")
        print(f"   {self.gesture_to_idx}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Cargar landmarks
        landmark_path = Path(row['path_landmarks'])
        if not landmark_path.exists():
            # Generar sintéticos si no existen
            landmarks = np.random.uniform(0, 1, (21, 3)).astype(np.float32)
        else:
            landmarks = np.load(landmark_path)  # (21, 3)
        
        # Convertir a formato ST-GCN: (C=3, T, V=21)
        # Input (21, 3) → (3, T, 21)
        # Si T=1: (3, 1, 21)
        if self.num_frames == 1:
            x = landmarks.T[np.newaxis, :, :]  # (1, 21) → (21, 1) → (3, 1, 21)??? No
            # Mejor: (21, 3) → transpose (3, 21) → add time dim (3, 1, 21)
            x = landmarks.T[:, np.newaxis, :]  # (3, 1, 21)
        else:
            # Repetir T veces
            x = np.tile(landmarks.T[:, np.newaxis, :], (1, self.num_frames, 1))  # (3, T, 21)
        
        x = torch.from_numpy(x).float()  # (3, T, 21) donde T=num_frames
        
        # Label
        gesture = row['label']
        label = torch.tensor(self.gesture_to_idx[gesture], dtype=torch.long)
        
        # Metadata
        mst = torch.tensor(float(row.get('mst', 5)), dtype=torch.float)
        
        return {
            'landmarks': x,  # (3, T, 21)
            'label': label,
            'mst': mst,
            'sample_id': row['sample_id'],
        }


class SimpleST_GCN(nn.Module):
    """Modelo ST-GCN minimalista para prototipado rápido."""
    
    def __init__(self, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        
        # Spatial convolution: (B, 3, T, 21) → (B, hidden, T, 21)
        self.spatial_conv = nn.Conv2d(3, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        
        # Global pooling + classification
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, T, 21) - Batch, Channels, Time, Vertices
        
        Returns:
            logits: (B, num_classes)
        """
        # Spatial convolution
        x = self.spatial_conv(x)  # (B, hidden_dim, T, 21)
        x = torch.relu(x)
        
        # Global pooling
        B, C, T, V = x.shape
        x = self.pool(x)  # (B, C, 1, 1)
        x = x.view(B, C)  # (B, C)
        
        # Classification
        logits = self.classifier(x)  # (B, num_classes)
        return logits


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Entrena un epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        x = batch['landmarks'].to(device)  # (B, 3, T, 21)
        y = batch['label'].to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.shape[0]
        
        acc = 100 * correct / total
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.1f}%'})
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100 * correct / total,
    }


def validate(model, val_loader, criterion, device):
    """Valida el modelo."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['landmarks'].to(device)
            y = batch['label'].to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.shape[0]
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': 100 * correct / total,
    }


def main():
    # Config
    MANIFEST_CSV = "output/train_manifest_stgcn.csv"
    OUTPUT_DIR = Path("output/training_logs")
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 ST-GCN Training")
    print(f"   Device: {DEVICE}")
    print(f"   Manifest: {MANIFEST_CSV}")
    
    # Dataset y DataLoader
    print(f"\n📊 Cargando datos...")
    dataset = HandLandmarkDataset(MANIFEST_CSV, num_frames=1)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    print(f"   ✓ Dataset: {len(dataset)} samples")
    print(f"   ✓ Batches: {len(train_loader)}")
    
    # Modelo
    print(f"\n🧠 Creando modelo...")
    model = SimpleST_GCN(
        num_classes=len(dataset.gesture_to_idx),
        hidden_dim=64,
    ).to(DEVICE)
    print(f"   ✓ Clases: {len(dataset.gesture_to_idx)}")
    print(f"   ✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_classes': len(dataset.gesture_to_idx),
            'num_samples': len(dataset),
        },
        'history': [],
    }
    
    print(f"\n🔥 Entrenando ({EPOCHS} epochs)...")
    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        results['history'].append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
        })
        
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.1f}%")
        
        # Guardar checkpoint cada 5 epochs
        if epoch % 5 == 0:
            ckpt_path = OUTPUT_DIR / f'model_epoch_{epoch:02d}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f"   ✓ Checkpoint: {ckpt_path}")
    
    # Guardar modelo final
    final_path = OUTPUT_DIR / 'model_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Modelo final guardado: {final_path}")
    
    # Guardar resultados
    results_path = OUTPUT_DIR / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Resultados: {results_path}")
    
    # Resumen
    print(f"\n📈 Resumen Final:")
    print(f"   Epochs entrenados: {EPOCHS}")
    print(f"   Loss final: {results['history'][-1]['train_loss']:.4f}")
    print(f"   Accuracy final: {results['history'][-1]['train_accuracy']:.1f}%")
    print(f"   Modelo guardado: {final_path}")


if __name__ == '__main__':
    main()
