"""Repara rutas de landmarks en el manifiesto ST-GCN."""

import csv
from pathlib import Path

def repair_landmarks_manifest(
    input_csv: Path,
    output_csv: Path,
    landmarks_dir: Path = Path("data/processed/landmarks"),
) -> None:
    """Busca los archivos reales y actualiza rutas en el manifiesto."""
    
    # Construir índice de archivos que existen
    print(f"Indexando landmarks en {landmarks_dir}...")
    landmark_files = {}
    
    for npy_file in landmarks_dir.glob("*.npy"):
        # Clave: nombre base sin extensión
        key = npy_file.stem  # ej: "freihand_00000000"
        try:
            rel_path = npy_file.relative_to(Path.cwd())
        except ValueError:
            rel_path = npy_file.resolve().relative_to(Path.cwd().resolve())
        landmark_files[key] = str(rel_path).replace("\\", "/")
    
    print(f"Encontrados {len(landmark_files)} archivos .npy")
    
    # Procesar manifiesto
    rows_fixed = []
    not_found = []
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            sample_id = row['sample_id']
            
            # Buscar archivo
            if sample_id in landmark_files:
                row['path_landmarks'] = landmark_files[sample_id]
                rows_fixed.append(row)
            else:
                not_found.append(sample_id)
    
    print(f"\nEstadísticas:")
    print(f"  Rutas reparadas: {len(rows_fixed)}")
    print(f"  No encontrados: {len(not_found)}")
    
    if not_found:
        print(f"  Primeros no encontrados: {not_found[:5]}")
    
    # Guardar manifiesto reparado
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_fixed)
    
    print(f"\n✓ Manifiesto reparado guardado en: {output_csv}")
    print(f"✓ Total muestras: {len(rows_fixed)}")

if __name__ == "__main__":
    repair_landmarks_manifest(
        input_csv=Path("output/train_manifest_stgcn.csv"),
        output_csv=Path("output/train_manifest_stgcn_fixed.csv"),
        landmarks_dir=Path("data/processed/landmarks"),
    )
