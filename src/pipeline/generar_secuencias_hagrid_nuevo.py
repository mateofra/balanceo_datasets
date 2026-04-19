from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.temporal_sequence_utils import generate_temporal_sequence, sample_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / 'data/processed/landmarks/hagrid_nuevo'
OUTPUT_DIR = REPO_ROOT / 'data/processed/secuencias_stgcn/hagrid_nuevo'
T = 16
SIGMA = 0.015


def generar_secuencia(
    landmarks: np.ndarray,
    T: int = T,
    sigma: float = SIGMA,
    seed: int | None = None,
) -> np.ndarray:
    """Genera una secuencia temporal suave a partir de una pose estatica."""
    return generate_temporal_sequence(landmarks, T=T, sigma=sigma, seed=seed)

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f'No existe directorio de landmarks: {INPUT_DIR}')

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    archivos = list(INPUT_DIR.rglob('*.npy'))
    print(f"Transformando {len(archivos)} landmarks en secuencias...")

    datos_manifest = []

    for f in tqdm(archivos):
        # Cargar landmark (21, 3)
        lm = np.load(f)

        # Generar secuencia (16, 21, 3) con variacion temporal coherente
        seq = generar_secuencia(lm, T=T, sigma=SIGMA, seed=sample_seed(42, f.stem))

        # Estructura de salida: .../secuencias_stgcn/hagrid_nuevo/clase/archivo.npy
        clase = f.parent.name
        dest_folder = OUTPUT_DIR / clase
        dest_folder.mkdir(parents=True, exist_ok=True)

        dest_path = dest_folder / f.name
        np.save(dest_path, seq)

        # Guardar para el nuevo manifiesto
        datos_manifest.append({
            'path': str(dest_path.relative_to(REPO_ROOT)),
            'label': clase,
            'source': 'hagrid_nuevo'
        })

    # Guardar un mini-manifiesto temporal para esta tanda
    df = pd.DataFrame(datos_manifest)
    df.to_csv(REPO_ROOT / 'output/manifest_hagrid_nuevo_secuencias.csv', index=False)
    print(f"\nProceso completado. Secuencias guardadas en {OUTPUT_DIR}")
    print(f"Manifiesto temporal creado en output/manifest_hagrid_nuevo_secuencias.csv")

if __name__ == "__main__":
    main()
