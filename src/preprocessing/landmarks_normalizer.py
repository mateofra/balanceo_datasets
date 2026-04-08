"""
Normalizador universal de landmarks para ST-GCN.

Problema: FreiHAND y HaGRID están en escalas diferentes
- FreiHAND: cámara 3D en metros, Z profundidad absoluta
- MediaPipe: x,y normalizadas [0,1], z profundidad relativa

Solución: Z-score normalization (media=0, std=1)
- Aplica a cada landmark por separado
- Preserva variabilidad temporal (importante para ST-GCN)
- Agnóstico respecto a dataset/escala original
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json


class LandmarkNormalizer:
    """Normaliza landmarks con z-score por dataset."""
    
    def __init__(self):
        self.stats = {
            'freihand': {},
            'hagrid': {}
        }
        self.fitted = False
    
    def fit_from_manifest(self, manifest_path: str, num_samples: int = 500):
        """Calcula estadísticas (media, std) de cada coordenada por dataset."""
        manifest = pd.read_csv(manifest_path)
        
        for dataset in ['freihand', 'hagrid']:
            print(f"[FIT] Calculando estadísticas para {dataset}...")
            
            samples = manifest[manifest['dataset'] == dataset]
            if len(samples) == 0:
                print(f"  ⚠️  No hay muestras de {dataset}")
                continue
            
            # Muestrear para eficiencia
            samples = samples.sample(min(num_samples, len(samples)), random_state=42)
            
            coords_list = []
            
            for _, row in tqdm(samples.iterrows(), total=len(samples)):
                sec_path = Path(row.get('path_secuencia', ''))
                if sec_path.exists():
                    try:
                        seq = np.load(sec_path)  # (T, 21, 3)
                        # Tomar múltiples frames para estadísticas
                        for t in range(seq.shape[0]):
                            coords_list.append(seq[t].flatten())  # (63,)
                    except:
                        pass
            
            if coords_list:
                coords_array = np.array(coords_list)
                
                # Calcular por cada coordenada
                self.stats[dataset]['mean'] = coords_array.mean(axis=0).tolist()
                self.stats[dataset]['std'] = coords_array.std(axis=0).tolist()
                
                print(f"  ✓ {len(coords_list)} frames analizados")
                print(f"    Media global: {np.mean(coords_array):.6f}")
                print(f"    Std global: {np.mean(np.std(coords_array, axis=0)):.6f}")
        
        self.fitted = True
        return self
    
    def normalize(self, landmarks: np.ndarray, dataset: str) -> np.ndarray:
        """
        Normaliza un landmark (o secuencia) según estadísticas del dataset.
        
        Args:
            landmarks: shape (21, 3) para frame, o (T, 21, 3) para secuencia
            dataset: 'freihand' o 'hagrid'
        
        Returns:
            Landmarks normalizados, misma shape
        """
        if not self.fitted:
            raise RuntimeError("Normalizer no está fitted. Llama a fit_from_manifest() primero")
        
        if dataset not in self.stats or not self.stats[dataset]:
            print(f"⚠️  No hay estadísticas para {dataset}, retornando sin normalizar")
            return landmarks
        
        original_shape = landmarks.shape
        mean = np.array(self.stats[dataset]['mean'])
        std = np.array(self.stats[dataset]['std']) + 1e-8  # Evitar división por 0
        
        if len(original_shape) == 2:  # (21, 3)
            flat = landmarks.flatten()
            normalized = (flat - mean) / std
            return normalized.reshape(original_shape)
        
        elif len(original_shape) == 3:  # (T, 21, 3)
            normalized = np.zeros_like(landmarks)
            for t in range(landmarks.shape[0]):
                flat = landmarks[t].flatten()
                normalized[t] = ((flat - mean) / std).reshape(21, 3)
            return normalized
        
        else:
            raise ValueError(f"Shape no soportada: {original_shape}")
    
    def denormalize(self, landmarks: np.ndarray, dataset: str) -> np.ndarray:
        """Revierte la normalización (útil para visualización)."""
        if not self.fitted:
            raise RuntimeError("Normalizer no está fitted")
        
        if dataset not in self.stats or not self.stats[dataset]:
            return landmarks
        
        original_shape = landmarks.shape
        mean = np.array(self.stats[dataset]['mean'])
        std = np.array(self.stats[dataset]['std']) + 1e-8
        
        if len(original_shape) == 2:
            flat = landmarks.flatten()
            denormalized = flat * std + mean
            return denormalized.reshape(original_shape)
        
        elif len(original_shape) == 3:
            denormalized = np.zeros_like(landmarks)
            for t in range(landmarks.shape[0]):
                flat = landmarks[t].flatten()
                denormalized[t] = (flat * std + mean).reshape(21, 3)
            return denormalized
        
        else:
            raise ValueError(f"Shape no soportada: {original_shape}")
    
    def save(self, path: str):
        """Guarda estadísticas para reutilizar después."""
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Normalizador guardado: {path}")
    
    @classmethod
    def load(cls, path: str):
        """Carga estadísticas guardadas."""
        normalizer = cls()
        with open(path, 'r') as f:
            normalizer.stats = json.load(f)
        normalizer.fitted = True
        print(f"Normalizador cargado: {path}")
        return normalizer


def main():
    print("=" * 70)
    print("CREANDO NORMALIZADOR DE LANDMARKS")
    print("=" * 70)
    
    manifest_path = "data/processed/secuencias_stgcn/manifest_secuencias.csv"
    normalizer_path = "landmarks_normalizer.json"
    
    # Crear normalizador
    normalizer = LandmarkNormalizer()
    normalizer.fit_from_manifest(manifest_path, num_samples=500)
    
    # Guardar para usar después
    normalizer.save(normalizer_path)
    
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS DE NORMALIZACIÓN")
    print("=" * 70)
    
    for dataset in ['freihand', 'hagrid']:
        if normalizer.stats[dataset]:
            mean_vec = np.array(normalizer.stats[dataset]['mean'])
            std_vec = np.array(normalizer.stats[dataset]['std'])
            
            print(f"\n{dataset.upper()}:")
            print(f"  Media global: {mean_vec.mean():.6f}")
            print(f"  Std global: {std_vec.mean():.6f}")
            print(f"  Std min/max: [{std_vec.min():.6f}, {std_vec.max():.6f}]")
    
    print("\n" + "=" * 70)
    print("INSTRUCCIONES PARA USAR EN DATALOADER")
    print("=" * 70)
    print("""
    # En tu DataLoader:
    
    from landmarks_normalizer import LandmarkNormalizer
    
    class LandmarkDataset(Dataset):
        def __init__(self, manifest_path):
            self.df = pd.read_csv(manifest_path)
            self.normalizer = LandmarkNormalizer.load("landmarks_normalizer.json")
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            seq = np.load(row['path_secuencia'])  # (T, 21, 3)
            
            # Normalizar según dataset
            seq_norm = self.normalizer.normalize(seq, row['dataset'])
            
            return seq_norm.astype(np.float32)
    """)


if __name__ == "__main__":
    main()
