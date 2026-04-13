import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE / 'data/processed/landmarks/hagrid_nuevo'
OUTPUT_DIR = BASE / 'data/processed/secuencias_stgcn/hagrid_nuevo'
T = 16
SIGMA = 0.015

def generar_secuencia(landmarks, T=16, sigma=0.015):
    # landmarks shape: (21, 3)
    # Resultado shape: (16, 21, 3)
    secuencia = np.repeat(landmarks[np.newaxis, :, :], T, axis=0)
    ruido = np.random.normal(0, sigma, secuencia.shape)
    return (secuencia + ruido).astype(np.float32)

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

        # Generar secuencia (16, 21, 3)
        seq = generar_secuencia(lm, T=T, sigma=SIGMA)

        # Estructura de salida: .../secuencias_stgcn/hagrid_nuevo/clase/archivo.npy
        clase = f.parent.name
        dest_folder = OUTPUT_DIR / clase
        dest_folder.mkdir(parents=True, exist_ok=True)

        dest_path = dest_folder / f.name
        np.save(dest_path, seq)

        # Guardar para el nuevo manifiesto
        datos_manifest.append({
            'path': str(dest_path.relative_to(BASE)),
            'label': clase,
            'source': 'hagrid_nuevo'
        })

    # Guardar un mini-manifiesto temporal para esta tanda
    df = pd.DataFrame(datos_manifest)
    df.to_csv(BASE / 'output/manifest_hagrid_nuevo_secuencias.csv', index=False)
    print(f"\nProceso completado. Secuencias guardadas en {OUTPUT_DIR}")
    print(f"Manifiesto temporal creado en output/manifest_hagrid_nuevo_secuencias.csv")

if __name__ == "__main__":
    main()
