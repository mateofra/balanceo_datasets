#!/usr/bin/env python3
"""
Reorganiza la estructura del repositorio balanceo_datasets
conforme a la nueva organización modular.
"""

import shutil
from pathlib import Path

BASE = Path(__file__).parent

# Mapeamento: (archivo actual, carpeta destino)
MOVES = [
    # Documentación
    ("PIPELINE_MST_STGCN.md", "docs/"),
    ("PIPELINE_STGCN_NORMALIZACION.md", "docs/"),
    ("GUIA_TRAINING_STGCN.md", "docs/"),
    ("consideraciones.md", "docs/"),
    ("REORGANIZACION_A_STGCN.md", "docs/"),
    ("INFORME_USO_STARTER_PACK.md", "docs/"),
    ("README_TRABAJO_COMPLETADO.md", "docs/"),
    ("README_evaluacion_datasets.md", "docs/"),
    
    # Preprocesamiento (de raíz a src/preprocessing)
    ("landmarks_normalizer.py", "src/preprocessing/"),
    ("landmarks_normalizer.json", "src/preprocessing/"),
    ("procesar_landmarks_freihand.py", "src/preprocessing/"),
    ("procesar_landmarks_hagrid_mediapipe.py", "src/preprocessing/"),
    
    # Clasificación
    ("clasificar_mst_mediapipe.py", "src/classification/"),
    
    # Scripts de diagnóstico
    ("diagnosticar_normalizacion.py", "scripts/diagnosis/"),
    ("diagnose_paths.py", "scripts/diagnosis/"),
    ("diagnostico_normalizacion.json", "scripts/diagnosis/"),
    ("verificar_landmarks.py", "scripts/diagnosis/"),
    ("verificar_secuencias.py", "scripts/diagnosis/"),
    
    # Scripts de entrenamiento
    ("generar_secuencias_stgcn.py", "scripts/training/"),
    ("visualizar_pipeline.py", "scripts/training/"),
    ("train_stgcn.py", "scripts/training/"),
    
    # Scripts de reparación
    ("repair_manifest.py", "scripts/repair/"),
    ("check_npy_shape.py", "scripts/repair/"),
    
    # Scripts de generación
    ("generar_graficos_balanceo.py", "scripts/generate/"),
    ("REPORTE_COMPLETITUD.py", "scripts/generate/"),
]

def main():
    moved = 0
    skipped = 0
    
    for src_file, dest_dir in MOVES:
        src_path = BASE / src_file
        dest_path = BASE / dest_dir / src_file.split('/')[-1]
        
        if not src_path.exists():
            print(f"⏭️  Saltando {src_file} (no existe)")
            skipped += 1
            continue
        
        if dest_path.exists() and src_path != dest_path:
            print(f"⚠️  {src_file} ya existe en destino, saltando")
            skipped += 1
            continue
        
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dest_path))
            print(f"✅ {src_file} → {dest_dir}")
            moved += 1
        except Exception as e:
            print(f"❌ Error moviendo {src_file}: {e}")
            skipped += 1
    
    print(f"\n📊 Resumen: {moved} movidos, {skipped} saltados")

if __name__ == "__main__":
    main()
