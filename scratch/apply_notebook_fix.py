
import json
import os

def modify_nb(nb_path, cell_id, new_source):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    found = False
    for cell in nb['cells']:
        if cell.get('id') == cell_id:
            cell['source'] = new_source
            found = True
            break
    
    if found:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"✅ Celda {cell_id} actualizada.")
    else:
        print(f"❌ No se encontró la celda {cell_id}.")

if __name__ == "__main__":
    NB_PATH = "/home/mateo/cosas_de_clase/4/PRACTICAS/aaa/balanceo_datasets/colab/entrenar_stgcn_colab.ipynb"
    
    # Actualizar MANIFEST
    NEW_SOURCE = [
        "from sklearn.preprocessing import LabelEncoder\n",
        "MANIFEST = \"output/train_manifest_stgcn_secuencias_fixed.csv\"\n",
        "df = pd.read_csv(MANIFEST)\n",
        "df = df[df['quality_flag'] != 'excluded'].copy()\n",
        "\n",
        "print(\"--- AUDITORÍA DE ETIQUETAS ---\")\n",
        "print(f\"Total muestras: {len(df)}\")\n",
        "print(f\"Muestras con etiqueta vacía: {df['label'].isna().sum()}\")\n",
        "\n",
        "df['label'] = df['label'].fillna('unknown')\n",
        "le = LabelEncoder()\n",
        "df['label_idx'] = le.fit_transform(df['label'])\n",
        "unique_labels = le.classes_.tolist()\n",
        "label_to_idx = {name: i for i, name in enumerate(unique_labels)}\n",
        "\n",
        "print(\"Distribución Top 5:\")\n",
        "print(df['label'].value_counts().head(5))\n",
        "\n",
        "train_df = df.sample(frac=0.8, random_state=42)\n",
        "val_df = df.drop(train_df.index)"
    ]
    modify_nb(NB_PATH, "stgcn-load-data", NEW_SOURCE)
