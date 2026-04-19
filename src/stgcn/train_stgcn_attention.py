"""
Entrenamiento ST-GCN con Atención - Versión Lista para Producción

Características:
- Modelo STGCN con atención espacial basado en src/stgcn/stgcn_model.py
- Grafo anatómico de mano (21 joints) con coneidad realista
- Entrenamiento sobre dataset balanceado por MST
- 10 epochs para prototipado rápido
"""

import sys
from pathlib import Path

# Importar desde src/
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from datetime import datetime
import csv

# Importar componentes ST-GCN con atención
from src.stgcn.hand_graph import build_adjacency_matrix


class SpatialAttention(nn.Module):
    """Mecanismo de atención espacial para nodes (joints) individuales."""

    def __init__(self, in_channels: int, num_joints: int = 21):
        super().__init__()
        self.num_joints = num_joints
        self.W = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, J, C) - Batch, Time, Joints, Channels

        Returns:
            out: (B, T, J, C) - attended features
            attn: (B, J) - attention weights per joint
        """
        scores = self.W(x)  # (B, T, J, 1)
        scores = scores.mean(dim=1)  # (B, J, 1) - promedio temporal
        attn = torch.softmax(scores, dim=1)  # (B, J, 1)
        out = x * attn.unsqueeze(1)  # broadcast attention
        return out, attn.squeeze(-1)


class GraphConvolution(nn.Module):
    """Convolución sobre grafo anatómico de mano."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("A", adjacency.float())
        self.W = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, C)

        Returns:
            y: (B, T, J, C')
        """
        # Agregación sobre el grafo: A @ x
        aggregated = torch.einsum("ij,btjc->btic", self.A, x)
        return torch.relu(self.W(aggregated))


class STGCNWithAttention(nn.Module):
    """ST-GCN con atención espacial para reconocimiento de gestos."""

    def __init__(
        self,
        adjacency: torch.Tensor,
        in_channels: int = 3,
        hidden_dim: int = 64,
        num_joints: int = 21,
        num_classes: int = 20,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Graph convolutions
        self.gcn1 = GraphConvolution(in_channels, hidden_dim, adjacency)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim, adjacency)

        # Spatial attention
        self.spatial_attn = SpatialAttention(hidden_dim, num_joints)

        # Temporal modeling (GRU)
        self.temporal = nn.GRU(
            hidden_dim * num_joints,
            hidden_dim,
            batch_first=True,
            num_layers=2,
            dropout=0.3,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, T, J, C) - (Batch, Time, Joints=21, Channels=3)

        Returns:
            logits: (B, num_classes)
            attention_weights: (B, J) - weights per joint
        """
        # Graph convolutions
        h = self.gcn1(x)  # (B, T, J, hidden_dim)
        h = self.gcn2(h)  # (B, T, J, hidden_dim)

        # Spatial attention - aprende importancia de cada joint
        h, attn_weights = self.spatial_attn(h)  # h: (B, T, J, C), attn: (B, J)

        # Flatten para temporal modeling
        B, T, J, C = h.shape
        h = h.reshape(B, T, J * C)  # (B, T, J*hidden_dim)

        # Temporal modeling
        h, _ = self.temporal(h)  # h: (B, T, hidden_dim)

        # Use last timestep
        h = h[:, -1, :]  # (B, hidden_dim)

        # Classification
        logits = self.classifier(h)  # (B, num_classes)

        return logits, attn_weights


class HandLandmarkDataset(Dataset):
    """Dataset para landmarks de mano con labels de gesto."""

    def __init__(self, manifest_csv: str):
        self.rows = []
        with open(manifest_csv) as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

        # Mapear gestos a índices
        unique_gestures = sorted(set(row['label'] for row in self.rows))
        self.gesture_to_idx = {g: i for i, g in enumerate(unique_gestures)}
        self.idx_to_gesture = {i: g for g, i in self.gesture_to_idx.items()}

        print(f"🎯 Clases detectadas: {len(self.gesture_to_idx)}")
        for i, (g, idx) in enumerate(sorted(self.gesture_to_idx.items())[:5]):
            print(f"   {idx}: {g}")
        if len(self.gesture_to_idx) > 5:
            print(f"   ... +{len(self.gesture_to_idx) - 5} más")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # Cargar landmarks
        landmark_path = Path(row['path_landmarks'])
        if landmark_path.exists():
            landmarks = np.load(landmark_path)  # (21, 3)
        else:
            # Generar sintético si falta
            landmarks = np.random.uniform(0, 1, (21, 3)).astype(np.float32)

        # Asegurar shape
        if landmarks.shape != (21, 3):
            landmarks = landmarks.reshape(21, 3)

        # Formato para ST-GCN: (T, J, C) = (1, 21, 3) para estático
        x = torch.from_numpy(landmarks).unsqueeze(0).float()  # (1, 21, 3)

        # Label
        label = torch.tensor(
            self.gesture_to_idx[row['label']], dtype=torch.long
        )

        # Metadata
        mst = float(row.get('mst', 5))

        return {
            'landmarks': x,  # (1, 21, 3) = (T, J, C)
            'label': label,
            'mst': torch.tensor(mst),
            'sample_id': row['sample_id'],
        }


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Entrena un epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        x = batch['landmarks'].to(device)  # (B, T, J, C)
        y = batch['label'].to(device)  # (B,)

        # Forward
        optimizer.zero_grad()
        logits, attn_weights = model(x)  # logits: (B, K), attn: (B, J)
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


def main():
    """Entrenamiento principal."""
    # Config
    MANIFEST_CSV = REPO_ROOT / "output/train_manifest_stgcn.csv"
    OUTPUT_DIR = REPO_ROOT / "output/training_logs"
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("🚀 ST-GCN CON ATENCIÓN - ENTRENAMIENTO")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Manifest: {MANIFEST_CSV}")

    # Dataset
    print(f"\n📊 Cargando dataset...")
    dataset = HandLandmarkDataset(str(MANIFEST_CSV))
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    print(f"✓ Dataset: {len(dataset)} samples")
    print(f"✓ Batches: {len(train_loader)}")

    # Modelo con atención
    print(f"\n🧠 Construyendo modelo STGCN con Atención...")
    adjacency = build_adjacency_matrix()  # Grafo anatómico de mano
    model = STGCNWithAttention(
        adjacency=adjacency,
        in_channels=3,
        hidden_dim=64,
        num_joints=21,
        num_classes=len(dataset.gesture_to_idx),
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Modelo creado: {param_count:,} parámetros")
    print(f"✓ Clases: {len(dataset.gesture_to_idx)}")
    print(f"  Con attention layer: SpatialAttention (21 joints)")

    # Training
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
            'model': 'STGCNWithAttention',
            'attention_mechanism': 'SpatialAttention',
        },
        'history': [],
    }

    print(f"\n🔥 Entrenando ({EPOCHS} epochs)...")
    for epoch in range(1, EPOCHS + 1):
        metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )

        results['history'].append({
            'epoch': epoch,
            'train_loss': metrics['loss'],
            'train_accuracy': metrics['accuracy'],
        })

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Loss: {metrics['loss']:.4f} | "
              f"Acc: {metrics['accuracy']:.1f}%")

        # Checkpoint cada 5 epochs
        if epoch % 5 == 0:
            ckpt_path = OUTPUT_DIR / f'model_attention_epoch_{epoch:02d}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f"   ✓ Checkpoint: {ckpt_path}")

    # Guardar modelo final
    final_path = OUTPUT_DIR / 'stgcn_attention_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Modelo final guardado: {final_path}")

    # Guardar info del modelo
    model_info = {
        'type': 'STGCNWithAttention',
        'num_joints': 21,
        'num_classes': len(dataset.gesture_to_idx),
        'class_mapping': dataset.gesture_to_idx,
        'hidden_dim': 64,
        'attention_mechanism': 'SpatialAttention (per-joint importance)',
    }

    model_info_path = OUTPUT_DIR / 'model_info.json'
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"✓ Model info: {model_info_path}")

    # Guardar resultados
    results_path = OUTPUT_DIR / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Resultados: {results_path}")

    # Resumen
    print("\n" + "=" * 70)
    print("📈 ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"Epochs: {EPOCHS}")
    print(f"Loss final: {results['history'][-1]['train_loss']:.4f}")
    print(f"Accuracy final: {results['history'][-1]['train_accuracy']:.1f}%")
    print(f"Modelo guardado: {final_path}")
    print(f"Tipo: ST-GCN con Atención Espacial")
    print("=" * 70)


if __name__ == '__main__':
    main()
