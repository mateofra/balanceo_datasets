"""
Entrenamiento ST-GCN COMPATIBLES con DataLoader normalizado.

Input del DataLoader: (secuencia normalizadas, labels)
- secuencia: (B, T=16, Joints=21, Coords=3)
- labels: (B,) índice numérico
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stgcn.st_gcn_dataloader import create_dataloaders


class ST_GCNModel(nn.Module):
    """Modelo ST-GCN para clasificación de gestos.
    
    Architecture:
    - Spatial convolution: Conv2d sobre vertices (21 joints)
    - Temporal convolution: LSTM sobre frames (T=16)
    - Classification head
    """
    
    def __init__(self, num_classes: int, input_dim: int = 3, hidden_dim: int = 128):
        """
        Args:
            num_classes: Número de clases de gesto
            input_dim: Coordenadas por joint (siempre 3: x, y, z)
            hidden_dim: Dimensión de features ocultos
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Spatial convolution: Convoluciona sobre los 21 joints
        # Input: (B, 3, T, 21) [channels=3 coords, 21 verticesqueda]
        # Output: (B, hidden_dim, T, 21)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim//2, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Temporal convolution: LSTM sobre T frames
        # Input: (B, T, hidden_dim * 21)
        lstm_input_dim = hidden_dim * 21
        self.temporal_lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, T, 21, 3) del DataLoader
        
        Returns:
            logits: (B, num_classes)
        """
        # Convertir a formato ST-GCN: (B, T, 21, 3) → (B, 3, T, 21)
        x = x.permute(0, 3, 1, 2)  # (B, 3, T, 21)
        
        batch_size, channels, num_frames, num_vertices = x.shape
        
        # Spatial convolution
        x = self.spatial_conv(x)  # (B, hidden_dim, T, V)
        _, feat_dim, _, _ = x.shape
        
        # Reshape para LSTM: (B, T, hidden_dim * 21)
        x = x.permute(0, 2, 1, 3)  # (B, T, hidden_dim, 21)
        x = x.reshape(batch_size, num_frames, -1)  # (B, T, hidden_dim * 21)
        
        # Temporal LSTM
        lstm_out, (h_n, c_n) = self.temporal_lstm(x)  # (B, T, hidden_dim)
        
        # Usar último timestep
        x = lstm_out[:, -1, :]  # (B, hidden_dim)
        
        # Clasificación
        logits = self.classifier(x)  # (B, num_classes)
        return logits


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict:
    """Entrena un epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [TRAIN]", leave=False)
    
    for batch_idx, (sequences, labels) in enumerate(pbar):
        # Transferir a GPU si está disponible
        sequences = sequences.to(device)  # (B, T, 21, 3)
        labels = labels.to(device)  # (B,)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(sequences)  # (B, num_classes)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100 * correct / total
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{accuracy:.1f}%"
        })
    
    return {
        "loss": total_loss / len(train_loader),
        "accuracy": 100 * correct / total,
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Valida el modelo."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="         [VAL]", leave=False)
        
        for sequences, labels in pbar:
            sequences = sequences.to(device)  # (B, T, 21, 3)
            labels = labels.to(device)  # (B,)
            
            # Forward pass
            logits = model(sequences)  # (B, num_classes)
            loss = criterion(logits, labels)
            
            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            avg_loss = total_loss / (total // len(labels) + 1)
            accuracy = 100 * correct / total
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{accuracy:.1f}%"
            })
    
    return {
        "loss": total_loss / len(val_loader),
        "accuracy": 100 * correct / total,
    }


def main():
    """Script principal de entrenamiento."""
    
    # ================================================
    # CONFIGURACIÓN
    # ================================================
    config = {
        "num_epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hidden_dim": 128,
        "num_classes": 10,  # Cambiar a número real de gestos
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "manifest_path": "data/processed/secuencias_stgcn/manifest_secuencias.csv",
        "normalizer_path": "landmarks_normalizer.json",
    }
    
    print("=" * 70)
    print("ST-GCN TRAINING")
    print("=" * 70)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print()
    
    # ================================================
    # CREAR DATALOADERS
    # ================================================
    print("Cargando dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        manifest_path=config['manifest_path'],
        normalizer_path=config['normalizer_path'],
        batch_size=config['batch_size'],
        num_workers=0,
        augment_temporal=True
    )
    
    if not train_loader:
        print("❌ Error: No se pudo crear train_loader")
        return
    
    # ================================================
    # CREAR MODELO
    # ================================================
    print("\nCreando modelo...")
    model = ST_GCNModel(
        num_classes=config['num_classes'],
        hidden_dim=config['hidden_dim']
    )
    model = model.to(config['device'])
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ================================================
    # OPTIMIZER Y LOSS
    # ================================================
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # ================================================
    # TRAINING LOOP
    # ================================================
    print("\n" + "=" * 70)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 70)
    
    history = {
        "train": [],
        "val": [],
        "test": []
    }
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            config['device'], epoch+1
        )
        history['train'].append(train_metrics)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config['device'])
        history['val'].append(val_metrics)
        
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}%")
        
        # Early stopping
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            # Guardar best checkpoint
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Best model guardado (acc {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏸️  Early stopping (patience={patience} epochs)")
                break
    
    # ================================================
    # EVALUACIÓN FINAL
    # ================================================
    if test_loader:
        print("\n" + "=" * 70)
        print("EVALUACIÓN EN TEST SET")
        print("=" * 70)
        
        # Cargar best model
        model.load_state_dict(torch.load("best_model.pth"))
        
        test_metrics = validate(model, test_loader, criterion, config['device'])
        print(f"Test Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['accuracy']:.2f}%")
    
    # ================================================
    # GUARDAR RESULTADOS
    # ================================================
    results = {
        "config": config,
        "history": history,
        "best_val_acc": best_val_acc,
    }
    
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Resultados guardados en training_results.json")
    print("✓ Mejor modelo en best_model.pth")


if __name__ == "__main__":
    main()
