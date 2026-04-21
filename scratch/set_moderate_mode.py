import json
from pathlib import Path

def setup_moderate_mode_notebooks():
    paths = ["entrenar_stgcn.ipynb", "colab/entrenar_stgcn_colab.ipynb"]
    
    for p in paths:
        path = Path(p)
        if not path.exists(): continue
        
        with open(path, "r") as f:
            nb = json.load(f)

        for cell in nb['cells']:
            # 1. Update Config (LR remains 0.0001 for safety)
            if cell['id'] == 'stgcn-params':
                source = "".join(cell['source'])
                source = source.replace('"EPOCHS": 20', '"EPOCHS": 40') # More epochs for learning rare classes
                cell['source'] = [l + "\n" for l in source.split("\n") if l]

            # 2. Update Data Loading (Smooth weights)
            if cell['id'] == 'stgcn-load-data':
                # Use sqrt weights for better stability
                new_source = [
                    "from sklearn.preprocessing import LabelEncoder\n",
                    "\n",
                    "df = pd.read_csv(CONFIG[\"MANIFEST\"])\n",
                    "df = df[df['quality_flag'] != 'excluded'].copy()\n",
                    "df['label'] = df['label'].fillna('unknown').astype(str)\n",
                    "\n",
                    "le = LabelEncoder()\n",
                    "df['label_idx'] = le.fit_transform(df['label'])\n",
                    "unique_labels = le.classes_.tolist()\n",
                    "label_to_idx = {name: i for i, name in enumerate(unique_labels)}\n",
                    "\n",
                    "# MODERATE MODE: Pesos suavizados con raíz cuadrada\n",
                    "counts = df['label_idx'].value_counts().sort_index().values\n",
                    "weights = 1.0 / np.sqrt(counts)\n",
                    "weights = weights / weights.sum() * len(counts) # Normalizar para que la media sea 1.0\n",
                    "weights_tensor = torch.FloatTensor(weights).to(device)\n",
                    "\n",
                    "train_df = df.sample(frac=0.8, random_state=CONFIG[\"SEED\"])\n",
                    "val_df = df.drop(train_df.index)\n",
                    "print(f\"Moderate Mode: Pesos calculados (max: {weights.max():.2f}, min: {weights.min():.2f})\")\n"
                ]
                cell['source'] = new_source

            # 3. Re-enable Augmentation (p=0.05) and Weighted Loss
            if cell['id'] == 'stgcn-train-exec':
                source = "".join(cell['source'])
                # Enable augmentation back with lower probability
                source = source.replace('mask = np.random.rand(num_nodes) > 0.1', 'mask = np.random.rand(num_nodes) > 0.05')
                source = source.replace('augment=False', 'augment=True')
                # Use the new weights_tensor
                source = source.replace('nn.CrossEntropyLoss()', 'nn.CrossEntropyLoss(weight=weights_tensor)')
                cell['source'] = [l + "\n" for l in source.split("\n") if l]

        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        
        print(f"✅ Moderate Mode aplicado a {p}")

if __name__ == "__main__":
    setup_moderate_mode_notebooks()
