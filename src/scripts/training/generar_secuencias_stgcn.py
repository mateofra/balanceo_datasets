"""
Script para convertir landmarks estáticos (21, 3) en secuencias temporales (T, 21, 3) 
para entrenamiento del ST-GCN.

Cada frame estático se expande en una secuencia con ruido motor suave para simular
movimiento natural de la mano.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.preprocessing.temporal_sequence_utils import generate_temporal_sequence, sample_seed


def generar_secuencia_sintetica(
    landmarks_estaticos: np.ndarray,  # shape (21, 3)
    T: int = 16,
    sigma_ruido: float = 0.015,
    seed: int | None = None,
) -> np.ndarray:
    """
    Construye una secuencia temporal sintética desde un frame estático.
    El ruido es gaussiano por articulación, simulando temblor motor natural.
    
    Args:
        landmarks_estaticos: Array (21, 3) con coordenadas x, y, z de 21 puntos
        T: Número de frames en la secuencia
        sigma_ruido: Desviación estándar del ruido gaussiano
    
    Returns:
        Array (T, 21, 3) con la secuencia temporal
    """
    return generate_temporal_sequence(
        landmarks_estaticos,
        T=T,
        sigma=sigma_ruido,
        seed=seed,
    )


def encontrar_archivo_landmarks(row: pd.Series) -> Path:
    """
    Busca el archivo de landmarks primero en el directorio plano.
    """
    sample_id = row['sample_id']
    
    # Buscar primero en el directorio plano
    ruta_plana = Path('data/processed/landmarks') / f"{sample_id}.npy"
    if ruta_plana.exists():
        return ruta_plana
    
    # Intentar con la ruta del CSV (en caso de que haya subdirectorios)
    landmarks_path = row['path_landmarks']
    if isinstance(landmarks_path, str):
        landmarks_path = landmarks_path.replace('\\', '/')
        ruta_csv = Path(landmarks_path)
        if ruta_csv.exists():
            return ruta_csv
    
    raise FileNotFoundError(f"No se encontró landmark para {sample_id}")



def procesar_manifest(csv_path: str, output_dir: str, T: int = 16, seed: int = 42):
    """
    Procesa un manifest CSV y genera secuencias temporales para cada muestra.
    
    Args:
        csv_path: Ruta al CSV con los landmarks estáticos
        output_dir: Directorio donde guardar las secuencias
        T: Número de frames por secuencia
        seed: Semilla para reproducibilidad
    """
    np.random.seed(seed)
    
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registros = []
    errores = []

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Generando secuencias")):
        try:
            landmarks_path = encontrar_archivo_landmarks(row)
            landmarks = np.load(landmarks_path)  # (21, 3)
            
            if landmarks.shape != (21, 3):
                errores.append(f"Shape inesperado en {landmarks_path}: {landmarks.shape}")
                continue

            secuencia = generar_secuencia_sintetica(
                landmarks,
                T=T,
                seed=sample_seed(42, row.sample_id),
            )  # (T, 21, 3)

            out_path = output_dir / f"{row['sample_id']}.npy"
            np.save(out_path, secuencia.astype(np.float32))

            registros.append({
                **row.to_dict(),
                'path_secuencia': str(out_path),
                'T': T,
                'secuencia_shape': str(secuencia.shape)
            })
            
        except Exception as e:
            errores.append(f"Error procesando {row.get('sample_id', 'unknown')}: {str(e)}")

    df_out = pd.DataFrame(registros)
    output_csv = output_dir / 'manifest_secuencias.csv'
    df_out.to_csv(output_csv, index=False)
    
    print(f"\n[OK] Generadas {len(registros)} secuencias en {output_dir}")
    print(f"[OK] CSV guardado en {output_csv}")
    
    if errores:
        print(f"\n[WARN] {len(errores)} errores encontrados:")
        for error in errores[:5]:  # Mostrar primeros 5
            print(f"       - {error}")
        if len(errores) > 5:
            print(f"       ... y {len(errores) - 5} más")


if __name__ == "__main__":
    # Configuración
    csv_input = "csv/train_manifest_stgcn.csv"
    output_directory = "data/processed/secuencias_stgcn"
    num_frames = 16
    
    print("=" * 60)
    print("GENERADOR DE SECUENCIAS TEMPORALES PARA ST-GCN")
    print("=" * 60)
    print(f"[INPUT] Leyendo: {csv_input}")
    print(f"[OUTPUT] Guardando en: {output_directory}")
    print(f"[CONFIG] Frames por secuencia: {num_frames}")
    print("=" * 60)
    
    procesar_manifest(csv_input, output_directory, T=num_frames)
    
    print("\n[DONE] Proceso completado!")
