"""
Genera secuencias temporales sintéticas para ST-GCN.

A partir de landmarks estáticos (21, 3), genera ventanas temporales
aplicando micro-perturbaciones realistas:
- Repetición del frame base
- Ruido gaussiano controlado (σ = 0.015)
- Rotaciones mínimas (1-2 grados) para simular movimiento

Pipeline:
1. Cargar landmarks de data/processed/landmarks/*.npy
2. Para cada landmark, generar T=16 frames con perturbaciones
3. Guardar como data/processed/secuencias_stgcn/*.npy
4. Actualizar manifiesto ST-GCN con paths de secuencias
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.temporal_sequence_utils import generate_temporal_sequence, sample_seed


T = 16  # Duración de la secuencia temporal
SIGMA_NOISE = 0.015  # Desviación estándar del ruido Gaussiano
ROTATION_ANGLE_DEG = 1.5  # Ángulo máximo de rotación (grados)


def generar_secuencia_temporal(
    landmark: np.ndarray,
    T: int = 16,
    sigma_noise: float = 0.015,
    rotation_angle_deg: float = 1.5,
    seed: int | None = None,
) -> np.ndarray:
    """Genera una secuencia temporal sintética a partir de un landmark estático.
    
    Args:
        landmark: Shape (21, 3) - landmark base
        T: Número de frames temporales
        sigma_noise: Desviación estándar del ruido Gaussiano
        rotation_angle_deg: Ángulo máximo de rotación (grados)
        seed: Random seed para reproducibilidad
        
    Returns:
        np.ndarray: Secuencia temporal de shape (T, 21, 3)
    """
    return generate_temporal_sequence(
        landmark,
        T=T,
        sigma=sigma_noise,
        rotation_std_deg=rotation_angle_deg,
        seed=seed,
    )


def generar_secuencias_para_manifest(
    manifest_path: Path,
    landmarks_dir: Path,
    output_dir: Path,
    T: int = 16,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Genera secuencias temporales para cada muestra en el manifiesto.
    
    Args:
        manifest_path: Ruta al manifiesto ST-GCN original
        landmarks_dir: Directorio con landmarks .npy
        output_dir: Directorio de salida para secuencias
        T: Número de frames por secuencia
        seed: Random seed
        verbose: Mostrar progreso
        
    Returns:
        pd.DataFrame: Manifiesto actualizado con rutas de secuencias
    """
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar manifiesto original
    manifest_df = pd.read_csv(manifest_path)
    
    secuencia_data = []
    
    if verbose:
        iterator = tqdm(
            manifest_df.itertuples(),
            total=len(manifest_df),
            desc="Generando secuencias temporales"
        )
    else:
        iterator = manifest_df.itertuples()
    
    for row in iterator:
        sample_id = row.sample_id
        landmark_path = landmarks_dir / f"{sample_id}.npy"
        
        if not landmark_path.exists():
            if verbose:
                print(f"⚠️  No existe landmark: {landmark_path}")
            continue
        
        # Cargar landmark
        try:
            landmark = np.load(landmark_path)  # Shape (21, 3)
            
            # Generar secuencia temporal
            secuencia = generar_secuencia_temporal(
                landmark,
                T=T,
                sigma_noise=SIGMA_NOISE,
                rotation_angle_deg=ROTATION_ANGLE_DEG,
                seed=sample_seed(seed, sample_id),
            )
            
            # Guardar secuencia
            output_path = output_dir / f"{sample_id}.npy"
            np.save(output_path, secuencia)
            
            # Registrar en manifiesto
            secuencia_data.append({
                'sample_id': sample_id,
                'path': getattr(row, 'path', getattr(row, 'image_path', '')),
                'path_secuencia': str(output_path.relative_to(output_dir.parent.parent)),
                'path_landmarks': getattr(row, 'path_landmarks', ''),
                'label': getattr(row, 'label', 'unknown'),
                'condition': getattr(row, 'condition', 'sin_mst'),
                'dataset': getattr(row, 'dataset', 'unknown'),
                'mst': getattr(row, 'mst', ''),
                'mst_origin': getattr(row, 'mst_origin', ''),
                'split': getattr(row, 'split', 'train'),
            })
            
        except Exception as e:
            if verbose:
                print(f"❌ Error procesando {sample_id}: {e}")
            continue
    
    return pd.DataFrame(secuencia_data)


def main():
    parser = argparse.ArgumentParser(
        description="Generar secuencias temporales sintéticas para ST-GCN"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/train_manifest_stgcn.csv"),
        help="Manifiesto ST-GCN con ruta a landmarks",
    )
    parser.add_argument(
        "--landmarks-dir",
        type=Path,
        default=Path("data/processed/landmarks"),
        help="Directorio con landmarks .npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/secuencias_stgcn"),
        help="Directorio de salida para secuencias",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("output/train_manifest_stgcn_secuencias.csv"),
        help="CSV de salida con rutas de secuencias",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=16,
        help="Número de frames por secuencia",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Mostrar progreso",
    )
    
    args = parser.parse_args()
    
    print(f"\n🔄 Generando secuencias temporales sintéticas")
    print(f"   Manifiesto: {args.manifest}")
    print(f"   Landmarks: {args.landmarks_dir}")
    print(f"   Output: {args.output_dir}")
    print(f"   Frames por secuencia (T): {args.T}")
    print(f"   Ruido (σ): {SIGMA_NOISE}")
    print(f"   Rotación máxima: ±{ROTATION_ANGLE_DEG}°")
    
    # Generar secuencias
    manifest_secuencias = generar_secuencias_para_manifest(
        args.manifest,
        args.landmarks_dir,
        args.output_dir,
        T=args.T,
        seed=args.seed,
        verbose=args.verbose,
    )
    
    # Guardar manifiesto actualizado
    manifest_secuencias.to_csv(args.output_manifest, index=False)
    
    print(f"\n✅ Secuencias generadas:")
    print(f"   Total: {len(manifest_secuencias)}")
    print(f"   Manifiesto guardado: {args.output_manifest}")
    print(f"   Directorio secuencias: {args.output_dir}")
    print(f"   Muestras por split:")
    if 'split' in manifest_secuencias.columns:
        for split, count in manifest_secuencias['split'].value_counts().items():
            print(f"     - {split}: {count}")


if __name__ == "__main__":
    main()
