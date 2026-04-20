import pandas as pd
import json
from pathlib import Path

def generate_confidence_report():
    csv_path = Path("output/mano_refined_suggestions.csv")
    if not csv_path.exists():
        print("❌ Error: No se encuentra el CSV de sugerencias.")
        return
        
    df = pd.read_csv(csv_path)
    
    new_classes = ['rock', 'two_up', 'call', 'like', 'bad_pose']
    report_data = []
    
    print("📊 Reporte de Confianza por Clase Reclasificada:\n")
    print(f"{'Clase':<10} | {'N':<6} | {'Media':<6} | {'% High':<8} | {'% Low':<8} | {'Estado'}")
    print("-" * 65)

    for cls in new_classes:
        cls_df = df[df['suggested_label'] == cls]
        n = len(cls_df)
        
        if n == 0:
            continue
            
        conf_mean = cls_df['confidence'].mean()
        conf_std = cls_df['confidence'].std()
        pct_high = (cls_df['confidence'] >= 0.70).sum() / n * 100
        pct_low = (cls_df['confidence'] < 0.55).sum() / n * 100
        
        # Criterios de aprobación
        status = "✅ OK"
        if conf_mean < 0.65:
            status = "⚠️ REVISAR (Media Baja)"
        elif pct_low > 30:
            status = "⚠️ ADVERTENCIA (Inestable)"
            
        report_item = {
            'clase': cls,
            'n_muestras': int(n),
            'confidence_media': float(conf_mean),
            'confidence_std': float(conf_std) if not pd.isna(conf_std) else 0.0,
            'pct_alta_confianza': float(pct_high),
            'pct_baja_confianza': float(pct_low),
            'muestra_ids_baja': cls_df[cls_df['confidence'] < 0.55]['sample_id'].head(10).tolist(),
            'status': status
        }
        report_data.append(report_item)
        
        print(f"{cls:<10} | {n:<6} | {conf_mean:<6.2f} | {pct_high:<7.1f}% | {pct_low:<7.1f}% | {status}")

    # Guardar JSON
    output_path = Path("output/confidence_report.json")
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=4)
        
    print(f"\n✅ Reporte completo guardado en {output_path}")

if __name__ == "__main__":
    generate_confidence_report()
