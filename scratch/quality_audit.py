import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def run_quality_audit():
    manifest_path = Path("output/train_manifest_stgcn_secuencias.csv")
    if not manifest_path.exists():
        print(f"❌ Error: No se encuentra el manifiesto {manifest_path}")
        return

    print(f"🔍 Iniciando auditoría de calidad sobre {manifest_path}...")
    df = pd.read_csv(manifest_path)
    
    results = []
    
    # Cache para evitar leer el mismo archivo de landmarks varias veces si se repite
    # (aunque en este manifiesto las secuencias son únicas, apuntan a landmarks compartidos)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Auditando muestras"):
        lm_path = Path(row['path_landmarks'])
        if not lm_path.exists():
            results.append('excluded_missing')
            continue
            
        try:
            # Los landmarks suelen ser (21, 3)
            lms = np.load(lm_path)
            
            # 1. Detectar NaNs o Infs
            if np.any(np.isnan(lms)) or np.any(np.isinf(lms)):
                results.append('excluded_invalid_values')
                continue
                
            # 2. Detectar fuera de rango [0, 1]
            # Contamos cuántas coordenadas (de las 63 totales) están fuera
            out_of_range = np.logical_or(lms < 0.0, lms > 1.0)
            ratio_out = np.sum(out_of_range) / lms.size
            
            if ratio_out > 0.3:
                results.append(f'excluded_out_of_range_{ratio_out:.2f}')
            else:
                results.append('ok')
                
        except Exception as e:
            results.append(f'excluded_error_{type(e).__name__}')

    df['quality_flag'] = ['ok' if r == 'ok' else 'excluded' for r in results]
    df['quality_detail'] = results
    
    # Guardar manifiesto actualizado
    df.to_csv(manifest_path, index=False)
    
    # Reporte
    print("\n📊 Reporte de Auditoría de Calidad:")
    summary = df.groupby(['dataset', 'quality_flag']).size().unstack(fill_value=0)
    print(summary)
    
    total_ok = (df['quality_flag'] == 'ok').sum()
    total_ex = (df['quality_flag'] == 'excluded').sum()
    print(f"\n✅ Muestras OK: {total_ok}")
    print(f"❌ Muestras Excluidas: {total_ex} ({(total_ex/len(df))*100:.2f}%)")

if __name__ == "__main__":
    run_quality_audit()
