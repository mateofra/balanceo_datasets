import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE = Path(__file__).resolve().parent
PATH_ORIGINAL = BASE / 'output/manifest_canonico.csv'
PATH_NUEVO = BASE / 'output/manifest_hagrid_nuevo_secuencias.csv'
PATH_SALIDA = BASE / 'output/manifest_unificado_final.csv'

def main():
    # 1. Cargar manifiestos
    df_orig = pd.read_csv(PATH_ORIGINAL)
    df_nuevo = pd.read_csv(PATH_NUEVO)

    print(f"Original: {len(df_orig)} muestras")
    print(f"Nuevo (HaGRID 30k): {len(df_nuevo)} muestras")

    # 2. Estandarizar columnas para que encajen
    # Normalizar columnas del manifiesto original real
    if 'path' not in df_orig.columns and 'path_secuencia' in df_orig.columns:
        df_orig = df_orig.rename(columns={'path_secuencia': 'path'})
    if 'source' not in df_orig.columns and 'dataset' in df_orig.columns:
        df_orig = df_orig.rename(columns={'dataset': 'source'})
    if 'mst_imputed' not in df_orig.columns:
        if 'condition' in df_orig.columns:
            df_orig['mst_imputed'] = df_orig['condition']
        else:
            df_orig['mst_imputed'] = 'medio'
    if 'landmark_quality' not in df_orig.columns:
        df_orig['landmark_quality'] = 'annotation_2d_projected'

    # Manifiesto nuevo: completar metadatos mínimos
    if 'landmark_quality' not in df_nuevo.columns:
        df_nuevo['landmark_quality'] = 'annotation_2d_projected'
    if 'mst_imputed' not in df_nuevo.columns:
        df_nuevo['mst_imputed'] = 'medio'  # Imputación por defecto para auditoría posterior

    # Seleccionar solo lo necesario del original para evitar conflictos
    cols = ['path', 'label', 'source', 'landmark_quality', 'mst_imputed']
    df_final = pd.concat([df_orig[cols], df_nuevo[cols]], ignore_index=True)

    print(f"Total combinado: {len(df_final)} muestras")

    # 3. Split Estratificado (70/15/15)
    # Primero: 70% train, 30% resto
    train_df, resto_df = train_test_split(
        df_final,
        test_size=0.30,
        random_state=42,
        stratify=df_final['label']
    )

    # Segundo: Dividir el 30% restante a la mitad (15% val, 15% test)
    val_df, test_df = train_test_split(
        resto_df,
        test_size=0.50,
        random_state=42,
        stratify=resto_df['label']
    )

    # Asignar etiquetas de split
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Unir todo de nuevo
    df_split = pd.concat([train_df, val_df, test_df])

    # 4. Guardar
    df_split.to_csv(PATH_SALIDA, index=False)

    print("\n--- RESUMEN DEL NUEVO DATASET ---")
    print(df_split.groupby(['split', 'source']).size().unstack(fill_value=0))
    print(f"\nManifiesto guardado en: {PATH_SALIDA}")

if __name__ == "__main__":
    main()
