"""Repara rutas de landmarks en el manifiesto ST-GCN."""

import csv
import json
from pathlib import Path


def _to_repo_relative_posix(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def _build_landmark_index(landmarks_dir: Path) -> dict[str, str]:
    """Indexa landmarks recursivamente por nombre base (stem en lowercase)."""
    index: dict[str, str] = {}
    for npy_file in landmarks_dir.rglob("*.npy"):
        stem = npy_file.stem.strip().lower()
        if stem:
            index[stem] = _to_repo_relative_posix(npy_file)
    return index


def _load_quality_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out: dict[str, str] = {}
    for k, v in payload.items():
        key = str(k).strip().lower()
        if key:
            out[key] = str(v).strip()
    return out

def repair_landmarks_manifest(
    input_csv: Path,
    output_csv: Path,
    landmarks_dir: Path = Path("data/processed/landmarks"),
    quality_json: Path = Path("csv/hagrid_landmarks_quality.json"),
) -> None:
    """Busca los archivos reales y actualiza rutas en el manifiesto."""
    
    # Construir índice de archivos que existen
    print(f"Indexando landmarks en {landmarks_dir}...")
    landmark_files = _build_landmark_index(landmarks_dir)
    quality_map = _load_quality_mapping(quality_json)
    
    print(f"Encontrados {len(landmark_files)} archivos .npy")
    
    # Procesar manifiesto
    rows_fixed = []
    not_found = []
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("El manifiesto de entrada no contiene encabezados.")
        if "landmark_quality" not in fieldnames:
            fieldnames = list(fieldnames) + ["landmark_quality"]
        
        for row in reader:
            sample_id = row['sample_id'].strip().lower()
            dataset = row.get('dataset', '').strip().lower()
            label = row.get('label', '').strip().lower()
            
            # Buscar archivo con múltiples convenciones de naming.
            candidate_keys = [sample_id]
            if dataset == 'hagrid' and label:
                candidate_keys.append(f"hagrid_{label}_{sample_id}")

            fixed_path = None
            for key in candidate_keys:
                if key in landmark_files:
                    fixed_path = landmark_files[key]
                    break

            if fixed_path is not None:
                row['path_landmarks'] = fixed_path

                if dataset == 'freihand':
                    row['landmark_quality'] = 'real_3d_freihand'
                elif dataset == 'hagrid':
                    row['landmark_quality'] = quality_map.get(sample_id, 'unknown_hagrid_quality')
                else:
                    row['landmark_quality'] = 'unknown_quality'

                rows_fixed.append(row)
            else:
                not_found.append(row['sample_id'])
    
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
        input_csv=Path("output/training/train_manifest_stgcn.csv"),
        output_csv=Path("output/training/train_manifest_stgcn_fixed.csv"),
        landmarks_dir=Path("data/processed/landmarks"),
    )
