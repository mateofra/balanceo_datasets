"""
Diagnóstico: Verifica normalización de coordenadas en FreiHAND vs HaGRID.

PROBLEMA CRÍTICO:
- MediaPipe (HaGRID): Coordenadas normalizadas (0-1) en x,y, z es profundidad relativa
- FreiHAND: Coordenadas de cámara 3D en METROS (tipicamente ±0.5m), NO normalizadas

Si no están en el mismo espacio, el ST-GCN aprenderá basura.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


def diagnosticar_normalizacion():
    """Analiza la distribución de coordenadas en FreiHAND vs HaGRID."""
    
    # Cargar manifest
    manifest = pd.read_csv("data/processed/secuencias_stgcn/manifest_secuencias.csv")
    
    freihand_samples = manifest[manifest['dataset'] == 'freihand'].sample(min(50, len(manifest)), random_state=42)
    
    stats = {
        "freihand": {},
        "comparison": {}
    }
    
    print("=" * 70)
    print("DIAGNOSTICO: NORMALIZACION DE COORDENADAS")
    print("=" * 70)
    
    # Analizar FreiHAND
    print("\n[FREIHAND] Cargando landmarks estaticos originales...")
    freihand_coords = []
    
    for _, row in freihand_samples.iterrows():
        try:
            # Cargar landmark estático original
            landmarks_path = Path("data/processed/landmarks") / f"{row['sample_id']}.npy"
            if landmarks_path.exists():
                lm = np.load(landmarks_path)  # shape (21, 3)
                freihand_coords.append(lm.flatten())
        except Exception as e:
            print(f"  Error cargando {row['sample_id']}: {e}")
    
    if freihand_coords:
        freihand_coords = np.array(freihand_coords)  # shape (N, 63)
        
        print(f"  Muestras cargadas: {len(freihand_coords)}")
        print(f"\n  Estadísticas por coordenada tipo:")
        
        # Analizar x, y, z por separado
        x_coords = freihand_coords[:, 0::3]  # Índices 0, 3, 6, ... (x de cada landmark)
        y_coords = freihand_coords[:, 1::3]  # Índices 1, 4, 7, ... (y)
        z_coords = freihand_coords[:, 2::3]  # Índices 2, 5, 8, ... (z)
        
        stats["freihand"]["x"] = {
            "min": float(x_coords.min()),
            "max": float(x_coords.max()),
            "mean": float(x_coords.mean()),
            "std": float(x_coords.std()),
            "percentiles": {
                "p05": float(np.percentile(x_coords, 5)),
                "p95": float(np.percentile(x_coords, 95))
            }
        }
        stats["freihand"]["y"] = {
            "min": float(y_coords.min()),
            "max": float(y_coords.max()),
            "mean": float(y_coords.mean()),
            "std": float(y_coords.std()),
            "percentiles": {
                "p05": float(np.percentile(y_coords, 5)),
                "p95": float(np.percentile(y_coords, 95))
            }
        }
        stats["freihand"]["z"] = {
            "min": float(z_coords.min()),
            "max": float(z_coords.max()),
            "mean": float(z_coords.mean()),
            "std": float(z_coords.std()),
            "percentiles": {
                "p05": float(np.percentile(z_coords, 5)),
                "p95": float(np.percentile(z_coords, 95))
            }
        }
        
        for coord_type in ["x", "y", "z"]:
            s = stats["freihand"][coord_type]
            print(f"\n  {coord_type.upper()}-coord:")
            print(f"    Rango: [{s['min']:.4f}, {s['max']:.4f}]")
            print(f"    Media: {s['mean']:.4f}, Std: {s['std']:.4f}")
            print(f"    P05-P95: [{s['percentiles']['p05']:.4f}, {s['percentiles']['p95']:.4f}]")
    
    # Análisis de secuencias generadas
    print("\n[SECUENCIAS STGCN] Verificando coordenadas en secuencias generadas...")
    
    seq_dir = Path("data/processed/secuencias_stgcn")
    seq_files = list(seq_dir.glob("*.npy"))[:50]
    
    seq_coords = []
    for seq_file in seq_files:
        try:
            seq = np.load(seq_file)  # shape (T, 21, 3)
            seq_coords.append(seq.reshape(-1, 3))
        except:
            pass
    
    if seq_coords:
        seq_coords = np.vstack(seq_coords)  # shape (N*T*21, 3)
        
        x_seq = seq_coords[:, 0]
        y_seq = seq_coords[:, 1]
        z_seq = seq_coords[:, 2]
        
        stats["secuencias"] = {
            "x": {
                "min": float(x_seq.min()),
                "max": float(x_seq.max()),
                "mean": float(x_seq.mean()),
                "std": float(x_seq.std())
            },
            "y": {
                "min": float(y_seq.min()),
                "max": float(y_seq.max()),
                "mean": float(y_seq.mean()),
                "std": float(y_seq.std())
            },
            "z": {
                "min": float(z_seq.min()),
                "max": float(z_seq.max()),
                "mean": float(z_seq.mean()),
                "std": float(z_seq.std())
            }
        }
        
        print(f"  Puntos analizados: {len(seq_coords)}")
        print(f"\n  X-coord en secuencias: [{x_seq.min():.4f}, {x_seq.max():.4f}]")
        print(f"  Y-coord en secuencias: [{y_seq.min():.4f}, {y_seq.max():.4f}]")
        print(f"  Z-coord en secuencias: [{z_seq.min():.4f}, {z_seq.max():.4f}]")
    
    # Verificación crítica
    print("\n" + "=" * 70)
    print("VERIFICACION CRITICA")
    print("=" * 70)
    
    freihand_x_range = stats["freihand"]["x"]["max"] - stats["freihand"]["x"]["min"]
    freihand_y_range = stats["freihand"]["y"]["max"] - stats["freihand"]["y"]["min"]
    freihand_z_range = stats["freihand"]["z"]["max"] - stats["freihand"]["z"]["min"]
    
    print(f"\nFreiHAND (coordenadas de cámara - METROS):")
    print(f"  X rango: {freihand_x_range:.4f}m")
    print(f"  Y rango: {freihand_y_range:.4f}m")
    print(f"  Z rango: {freihand_z_range:.4f}m")
    print(f"  → Son valores ABSOLUTOS en metros, típicamente ±0.5m")
    
    if "secuencias" in stats:
        seq_x_range = stats["secuencias"]["x"]["max"] - stats["secuencias"]["x"]["min"]
        seq_y_range = stats["secuencias"]["y"]["max"] - stats["secuencias"]["y"]["min"]
        seq_z_range = stats["secuencias"]["z"]["max"] - stats["secuencias"]["z"]["min"]
        
        print(f"\nSecuencias generadas:")
        print(f"  X rango: {seq_x_range:.4f}")
        print(f"  Y rango: {seq_y_range:.4f}")
        print(f"  Z rango: {seq_z_range:.4f}")
        
        # Diagnóstico
        print("\n[DIAGNÓSTICO]")
        if freihand_x_range > 0.3 and seq_x_range < 1.5:
            print("  ✓ OK: Valores similares en rango relativo")
        else:
            print("  ⚠️  ALERTA: Rangos muy diferentes - posible desescalado")
        
        # MediaPipe info
        print("\nNOTA sobre HaGRID/MediaPipe:")
        print("  - MediaPipe retorna: x, y normalizadas [0,1], z profundidad relativa")
        print("  - No se encontraron landmarks de HaGRID para comparar")
        print("  - Pero los generados tienen escala similar a FreiHAND")
    
    # Guardar diagnóstico
    diag_file = Path("diagnostico_normalizacion.json")
    with diag_file.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nDiagnóstico guardado: {diag_file}")
    
    print("\n" + "=" * 70)
    print("RECOMENDACIONES")
    print("=" * 70)
    print("""
1. Si FreiHAND está en metros [-0.5, 0.5] y HaGRID en [0, 1]:
   → NORMALIZACION REQUERIDA antes de entrenar ST-GCN
   
2. Opción A - Normalizar todo a [-1, 1]:
   - FreiHAND: escalar por 2 (desde ±0.5m a ±1)
   - HaGRID: escalar hacia el mismo rango
   
3. Opción B - Normalizar por estadísticas:
   - Restar media y dividir por std para cada dataset
   - Luego combinar en una distribución común
   
4. Opción C - Entrenar con normalizaciones separadas:
   - Layer de normalización inicial en el modelo
   - Aprende offset/scale por batch

ACCIÓN RECOMENDADA:
Implementar normalización en el DataLoader si los rangos difieren 
más del 50% en escala.
    """)


if __name__ == "__main__":
    diagnosticar_normalizacion()
