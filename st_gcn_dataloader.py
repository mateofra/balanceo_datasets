"""
DataLoader para ST-GCN con normalización universal de landmarks.

Características:
- Carga secuencias temporales (T, 21, 3)
- Aplica normalización z-score por dataset
- Compatible con torch.DataLoader
- Incluye augmentación temporal (opcional)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from landmarks_normalizer import LandmarkNormalizer


class STGCNDataset(Dataset):
    """Dataset para ST-GCN con normalización."""
    
    def __init__(
        self,
        manifest_path: str,
        normalizer: LandmarkNormalizer,
        split: str = "train",
        augment_temporal: bool = False
    ):
        """
        Args:
            manifest_path: Ruta a manifest_secuencias.csv
            normalizer: Instancia de LandmarkNormalizer fitted
            split: 'train', 'val', o 'test'
            augment_temporal: Si True, aplica temporal dropout (simula oclusión)
        """
        self.manifest = pd.read_csv(manifest_path)

        if "landmark_quality" in self.manifest.columns:
            self.manifest = self.manifest[
                self.manifest["landmark_quality"] == "real_3d_freihand"
            ].reset_index(drop=True)
            print(f"Muestras de entrenamiento: {len(self.manifest)}")
        
        # Filtrar por split
        if split in self.manifest['split'].unique():
            self.manifest = self.manifest[self.manifest['split'] == split].reset_index(drop=True)
        
        self.normalizer = normalizer
        self.augment_temporal = augment_temporal
        self.split = split
        
        print(f"[Dataset] Cargado {len(self)} muestras ({split}, source: {manifest_path})")
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            (secuencia normalizada, label_numérico)
        """
        row = self.manifest.iloc[idx]
        
        # Cargar secuencia
        seq_path = Path(row['path_secuencia'])
        if not seq_path.exists():
            raise FileNotFoundError(f"Secuencia no encontrada: {seq_path}")
        
        seq = np.load(seq_path)  # (T, 21, 3)
        
        # Normalizar
        dataset_source = row['dataset']
        seq_norm = self.normalizer.normalize(seq, dataset_source)
        
        # Augmentación temporal (opcional, solo en train)
        if self.augment_temporal and self.split == "train":
            seq_norm = self._temporal_dropout(seq_norm, dropout_prob=0.1)
        
        # Convertir a tensor
        seq_tensor = torch.from_numpy(seq_norm).float()  # (T, 21, 3)
        
        # Extraer label (debe ser entero)
        # Si la columna 'label' es string, mapear a índice
        label = row['label']
        if isinstance(label, str):
            # Mapear string a índice (simple hash para demostración)
            label = hash(label) % 10
        
        return seq_tensor, int(label)
    
    def _temporal_dropout(self, seq: np.ndarray, dropout_prob: float = 0.1) -> np.ndarray:
        """
        Simula oclusión temporal reemplazando frames ocasionalmente con frame vecino.
        Útil para augmentación durante entrenamiento.
        """
        T = seq.shape[0]
        seq_aug = seq.copy()
        
        for t in range(T):
            if np.random.rand() < dropout_prob:
                # Reemplazar con frame vecino
                neighbor = np.random.choice([max(0, t-1), min(T-1, t+1)])
                seq_aug[t] = seq[neighbor]
        
        return seq_aug
    
    def get_class_distribution(self) -> dict:
        """Retorna distribución de clases en el split."""
        return self.manifest['label'].value_counts().to_dict()


def create_dataloaders(
    manifest_path: str,
    normalizer_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    augment_temporal: bool = False
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Crea dataloaders para train/val/test.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    print("=" * 70)
    print("CREANDO DATALOADERS CON NORMALIZACIÓN")
    print("=" * 70)
    
    # Cargar normalizador
    normalizer = LandmarkNormalizer.load(normalizer_path)
    
    loaders = {}
    
    # Crear dataset por cada split
    for split in ["train", "val", "test"]:
        try:
            dataset = STGCNDataset(
                manifest_path=manifest_path,
                normalizer=normalizer,
                split=split,
                augment_temporal=(augment_temporal and split == "train")
            )
            
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                drop_last=(split == "train"),
                pin_memory=True
            )
            
            print(f"\n✓ {split.upper()} DataLoader creado:")
            print(f"    Muestras: {len(dataset)}")
            print(f"    Lotes: {len(loaders[split])}")
            print(f"    Batch size: {batch_size}")
            
            # Mostrar distribución de clases
            dist = dataset.get_class_distribution()
            if dist:
                print(f"    Clases: {len(dist)} ({dict(sorted(dist.items()))})")
        
        except Exception as e:
            print(f"\n⚠️  {split.upper()}: {e}")
            loaders[split] = None
    
    return loaders.get("train"), loaders.get("val"), loaders.get("test")


def test_dataloader(train_loader: DataLoader, num_batches: int = 3):
    """Valida que el dataloader funciona correctamente."""
    print("\n" + "=" * 70)
    print("VALIDANDO DATALOADER")
    print("=" * 70)
    
    try:
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            
            print(f"\nBatch {batch_idx}:")
            print(f"  Sequences shape: {sequences.shape}")  # (B, T, 21, 3)
            print(f"  Labels shape: {labels.shape}")  # (B,)
            print(f"  Sequence dtype: {sequences.dtype}")
            print(f"  Labels: {labels.tolist()}")
            
            # Validar rango de valores normalizados
            seq_min, seq_max = sequences.min().item(), sequences.max().item()
            seq_mean = sequences.mean().item()
            print(f"  Valor min/max: [{seq_min:.4f}, {seq_max:.4f}]")
            print(f"  Media: {seq_mean:.6f}")
            
            # Después de normalización, deberían estar ~N(0, 1)
            if batch_idx == 0:
                assert sequences.shape[1] == 16, "T debe ser 16"
                assert sequences.shape[2] == 21, "Joints debe ser 21"
                assert sequences.shape[3] == 3, "Coordenadas debe ser 3"
                print("  ✓ Shapes correctas para ST-GCN")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


def main():
    manifest_path = "data/processed/secuencias_stgcn/manifest_secuencias.csv"
    normalizer_path = "landmarks_normalizer.json"
    
    print("\n" + "=" * 70)
    print("ST-GCN DATALOADER CON NORMALIZACIÓN")
    print("=" * 70)
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        manifest_path=manifest_path,
        normalizer_path=normalizer_path,
        batch_size=32,
        num_workers=0,
        augment_temporal=True
    )
    
    # Validar
    if train_loader:
        test_dataloader(train_loader, num_batches=3)
    
    print("\n" + "=" * 70)
    print("LISTO PARA ENTRENAR ST-GCN")
    print("=" * 70)
    print(f"""
    Úsalo así en tu script de entrenamiento:
    
    from st_gcn_dataloader import create_dataloaders
    
    train_loader, val_loader, test_loader = create_dataloaders(
        manifest_path="data/processed/secuencias_stgcn/manifest_secuencias.csv",
        normalizer_path="landmarks_normalizer.json",
        batch_size=32
    )
    
    for epoch in range(num_epochs):
        for sequences, labels in train_loader:  # GPU transfer aquí si aplica
            # sequences shape: (B, T, 21, 3) con T=16
            # Listo para feed al modelo ST-GCN
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            ...
    """)


if __name__ == "__main__":
    main()
