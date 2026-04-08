#!/usr/bin/env python
"""Script de training ST-GCN con balanceo MST."""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm

# Import dataloader
from src.dataloader import create_dataloaders


class SimpleST_GCN(nn.Module):
    """Modelo ST-GCN simplificado."""
    
    def __init__(self, num_classes: int, num_vertices: int = 21, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        
        # Spatial conv: (B, 3, T, V) → (B, hidden_dim, T, V)
        self.spatial_conv = nn.Conv2d(3, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        
        # Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim * num_vertices,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.5 if hidden_dim > 32 else 0,
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, C=3, T, V=21)
        
        Returns:
            logits: (B, num_classes)
        """
        batch_size = x.shape[0]
        
        # Spatial conv
        x = self.spatial_conv(x)  # (B, hidden_dim, T, V)
        hidden_dim = x.shape[1]
        
        # Reshape para LSTM
        x = x.permute(0, 2, 1, 3)  # (B, T, hidden_dim, V)
        x = x.reshape(batch_size, -1, hidden_dim * 21)  # (B, T, hidden_dim * V)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, 64)
        x = lstm_out[:, -1, :]  # (B, 64) - last frame
        
        # Classify
        logits = self.classifier(x)  # (B, num_classes)
        return logits


def train_epoch(model, loader, optimizer, criterion, device, epoch, log_interval=50):
    """Entrena un epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        landmarks = batch["landmarks"].to(device)  # (B, 21, 3)
        labels = batch["label"].to(device)
        
        # Formato ST-GCN: (B, 3, 1, 21)
        x = landmarks.permute(0, 2, 1).unsqueeze(2)
        
        # Forward
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100 * correct / total
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.1f}%"})
    
    return {
        "loss": total_loss / len(loader),
        "accuracy": 100 * correct / total,
    }


def validate(model, loader, criterion, device):
    """Valida el modelo."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_mst = []
    
    with torch.no_grad():
        for batch in loader:
            landmarks = batch["landmarks"].to(device)
            labels = batch["label"].to(device)
            
            x = landmarks.permute(0, 2, 1).unsqueeze(2)
            logits = model(x)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_mst.extend(batch["mst"].numpy())
    
    return {
        "loss": total_loss / len(loader),
        "accuracy": 100 * correct / total,
        "preds": all_preds,
        "labels": all_labels,
        "mst": all_mst,
    }


def main():
    parser = argparse.ArgumentParser(description="ST-GCN Training")
    parser.add_argument("--manifest", default="../output/train_manifest_stgcn_fixed.csv")
    parser.add_argument("--landmarks-root", default="../data/processed/landmarks")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ST-GCN TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Load data
    print(f"\nCargando datos...")
    loaders = create_dataloaders(
        manifest_csv=args.manifest,
        landmarks_root=args.landmarks_root,
        batch_size=args.batch_size,
        normalize=True,
        balance_by_mst=True,
    )
    
    train_loader = loaders["train"]
    num_classes = loaders["num_classes"]
    
    print(f"✅ Datos cargados:")
    print(f"   Batches: {len(train_loader)}")
    print(f"   Clases: {num_classes}")
    
    # Initialize model
    print(f"\nInitializando modelo...")
    model = SimpleST_GCN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modelo: {total_params:,} parámetros")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"ENTRENANDO")
    print(f"{'='*60}\n")
    
    training_log = []
    
    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch+1
        )
        
        # Val metrics (usar mismo loader para demo)
        val_metrics = validate(model, train_loader, criterion, device)
        
        print(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.1f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.1f}%"
        )
        
        # Log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }
        training_log.append(log_entry)
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
    
    # Save final model
    final_path = checkpoint_dir / "model_final.pth"
    torch.save(model.state_dict(), final_path)
    
    # Save log
    log_path = log_dir / "training_log_final.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"Modelo: {final_path}")
    print(f"Log: {log_path}")
    print(f"\nProximo paso:")
    print(f"  python scripts/analyze_fairness.py {log_path}")


if __name__ == "__main__":
    main()
