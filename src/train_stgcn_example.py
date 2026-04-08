"""
Ejemplo de training ST-GCN con dataset balanceado por MST.

Estructura de datos esperada por ST-GCN:
- Input: (Batch, Channels=3, Frames=T, Vertices=21)
- Labels: (Batch,) - índice de gesto

Referencia: https://github.com/yysijie/st-gcn
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

# Importar dataloader personalizado
import sys
sys.path.insert(0, str(Path(__file__).parent))
from st_gcn_dataloader import create_dataloaders


class SimpleST_GCNModel(nn.Module):
    """Modelo ST-GCN simplificado para demostración.
    
    Estructura:
    - Conv1D por vertices (spatial convolution)
    - LSTM sobre frames (temporal convolution)
    - Classification head
    """

    def __init__(self, num_classes: int, num_vertices: int = 21, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.num_vertices = num_vertices

        # Spatial convolution: aplica conv sobre vertices
        # Input: (B, C, T, V) → Output: (B, hidden_dim, T, V)
        self.spatial_conv = nn.Conv2d(3, hidden_dim, kernel_size=(1, 3), padding=(0, 1))

        # Temporal convolution (LSTM)
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim * num_vertices,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )

        # Classification head
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
        batch_size, channels, num_frames, num_vertices = x.shape

        # Spatial conv
        x = self.spatial_conv(x)  # (B, hidden_dim, T, V)
        _, hidden_dim, _, _ = x.shape

        # Reshape para LSTM: (B, T, hidden_dim * V)
        x = x.permute(0, 2, 1, 3)  # (B, T, hidden_dim, V)
        x = x.reshape(batch_size, num_frames, -1)  # (B, T, hidden_dim * V)

        # Temporal convolution (LSTM)
        lstm_out, (hidden, cell) = self.temporal_lstm(x)  # lstm_out: (B, T, 64)

        # Use last timestep
        x = lstm_out[:, -1, :]  # (B, 64)

        # Classify
        logits = self.classifier(x)  # (B, num_classes)
        return logits


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch: int,
    log_interval: int = 50,
) -> dict:
    """Entrena un epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Cargar datos
        landmarks = batch["landmarks"].to(device)  # (B, 21, 3)
        labels = batch["label"].to(device)  # (B,)
        conditions = batch["condition"]  # Lista de tonos (claro/medio/oscuro)

        # Formato ST-GCN: (B, C=3, T=1, V=21)
        # Input: (B, 21, 3) → (B, 3, 1, 21)
        x = landmarks.permute(0, 2, 1).unsqueeze(2)

        # Forward + backward
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

        # Log
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100 * correct / total
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.1f}%"})

    return {
        "epoch": epoch,
        "avg_loss": total_loss / len(dataloader),
        "accuracy": 100 * correct / total,
    }


def validate(
    model: nn.Module,
    dataloader,
    criterion,
    device,
) -> dict:
    """Valida el modelo."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            landmarks = batch["landmarks"].to(device)
            labels = batch["label"].to(device)

            # Formato ST-GCN
            x = landmarks.permute(0, 2, 1).unsqueeze(2)

            logits = model(x)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        "avg_loss": total_loss / len(dataloader),
        "accuracy": 100 * correct / total,
    }


def main():
    # ======================================================================
    # CONFIG
    # ======================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    MANIFEST_CSV = Path("output/train_manifest_stgcn_fixed.csv")
    OUTPUT_DIR = Path("output/training_logs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Manifest: {MANIFEST_CSV}")
    print(f"Output: {OUTPUT_DIR}")

    # ======================================================================
    #CARGAR DATOS
    # ======================================================================
    print(f"\nCargando dataset...")
    loaders = create_dataloaders(
        manifest_csv=MANIFEST_CSV,
        batch_size=BATCH_SIZE,
        num_workers=0,
        normalize=True,
        balance_by_mst=True,
    )

    train_loader = loaders["train"]
    num_classes = loaders["num_classes"]
    gesture_to_label = loaders["gesture_to_label"]

    print(f"✓ Dataset cargado:")
    print(f"  - Batches: {len(train_loader)}")
    print(f"  - Clases: {num_classes}")
    print(f"  - Gestos: {list(gesture_to_label.keys())[:5]}...")

    # ======================================================================
    # INICIALIZAR MODELO
    # ======================================================================
    print(f"\nInitializing model...")
    model = SimpleST_GCNModel(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard logger
    writer = SummaryWriter(log_dir=str(OUTPUT_DIR / "tensorboard"))

    print(f"✓ Modelo listo (parameters: {sum(p.numel() for p in model.parameters()):,})")

    # ======================================================================
    # TRAINING LOOP
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"{'='*60}\n")

    for epoch in range(NUM_EPOCHS):
        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            epoch,
        )

        # Validate (reutilizamos train_loader para demo; en producción split en train/val)
        val_metrics = validate(model, train_loader, criterion, DEVICE)

        # Print
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_metrics['avg_loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.1f}% | "
            f"Val Loss: {val_metrics['avg_loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.1f}%"
        )

        # TensorBoard
        writer.add_scalar("train/loss", train_metrics["avg_loss"], epoch)
        writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("val/loss", val_metrics["avg_loss"], epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = OUTPUT_DIR / f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Checkpoint guardado: {checkpoint_path}")

    writer.close()

    # ======================================================================
    # FINAL
    # ======================================================================
    final_model_path = OUTPUT_DIR / "model_final.pth"
    torch.save(model.state_dict(), final_model_path)

    print(f"\n{'='*60}")
    print(f"✓ Training completado")
    print(f"  - Modelo: {final_model_path}")
    print(f"  - Logs: {OUTPUT_DIR / 'tensorboard'}")
    print(f"  - Ver con: tensorboard --logdir={OUTPUT_DIR / 'tensorboard'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
