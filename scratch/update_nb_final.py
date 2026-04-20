import json
from pathlib import Path

def update_notebook():
    notebook_path = Path("entrenar_stgcn.ipynb")
    with open(notebook_path, "r") as f:
        nb = json.load(f)

    for cell in nb['cells']:
        # Update Data Loading Cell
        if cell['id'] == 'stgcn-load-data':
            cell['source'] = [
                "from sklearn.utils.class_weight import compute_class_weight\n",
                "from sklearn.preprocessing import LabelEncoder\n",
                "\n",
                "df = pd.read_csv(CONFIG[\"MANIFEST\"])\n",
                "\n",
                "# 1. Exclusión de bad_pose\n",
                "initial_count = len(df)\n",
                "df = df[df['quality_flag'] != 'excluded'].copy()\n",
                "print(f\"Muestras excluidas por calidad: {initial_count - len(df)}\")\n",
                "\n",
                "df['label'] = df['label'].fillna('unknown').astype(str)\n",
                "\n",
                "# 2. Configurar Label Encoder\n",
                "le = LabelEncoder()\n",
                "df['label_idx'] = le.fit_transform(df['label'])\n",
                "unique_labels = le.classes_.tolist()\n",
                "label_to_idx = {name: i for i, name in enumerate(unique_labels)}\n",
                "idx_to_label = {i: name for name, i in label_to_idx.items()}\n",
                "\n",
                "# 3. Calcular pesos para balanceo de clases\n",
                "class_weights = compute_class_weight(\n",
                "    class_weight='balanced',\n",
                "    classes=np.unique(df['label_idx']),\n",
                "    y=df['label_idx']\n",
                ")\n",
                "weights_tensor = torch.FloatTensor(class_weights).to(device)\n",
                "\n",
                "print(f\"Total de secuencias efectivas: {len(df)}\")\n",
                "print(f\"Clases detectadas ({len(unique_labels)}): {unique_labels}\")\n",
                "\n",
                "# Split Train/Val (80/20)\n",
                "train_df = df.sample(frac=0.8, random_state=CONFIG[\"SEED\"])\n",
                "val_df = df.drop(train_df.index)\n",
                "\n",
                "print(f\"Train samples: {len(train_df)} | Val samples: {len(val_df)}\")"
            ]
        
        # Update Training Exec Cell (Dataset + Criterion)
        if cell['id'] == 'stgcn-train-exec':
            cell['source'] = [
                "class STGCNDataset(Dataset):\n",
                "    def __init__(self, manifest_df, base_dir, label_map, augment=False):\n",
                "        self.df = manifest_df\n",
                "        self.base_dir = Path(base_dir)\n",
                "        self.label_map = label_map\n",
                "        self.augment = augment\n",
                "        \n",
                "    def __len__(self): return len(self.df)\n",
                "    \n",
                "    def __getitem__(self, idx):\n",
                "        row = self.df.iloc[idx]\n",
                "        path = Path(row['path_secuencia'])\n",
                "        if not path.exists(): path = self.base_dir / path.name\n",
                "        \n",
                "        seq = np.load(path).astype(np.float32) # (16, 21, 3)\n",
                "        \n",
                "        # 4. Augmentation: Node Dropping (p=0.1)\n",
                "        if self.augment:\n",
                "            num_nodes = seq.shape[1]\n",
                "            mask = np.random.rand(num_nodes) > 0.1\n",
                "            seq[:, ~mask, :] = 0.0\n",
                "            \n",
                "        x = torch.from_numpy(np.transpose(seq, (2, 0, 1))).float() # (3, 16, 21)\n",
                "        y = torch.tensor(self.label_map[str(row['label'])], dtype=torch.long)\n",
                "        return x, y\n",
                "\n",
                "train_ds = STGCNDataset(train_df, CONFIG[\"SECUENCIAS_DIR\"], label_to_idx, augment=True)\n",
                "val_ds = STGCNDataset(val_df, CONFIG[\"SECUENCIAS_DIR\"], label_to_idx, augment=False)\n",
                "train_loader = DataLoader(train_ds, batch_size=CONFIG[\"BATCH_SIZE\"], shuffle=True)\n",
                "val_loader = DataLoader(val_ds, batch_size=CONFIG[\"BATCH_SIZE\"], shuffle=False)\n",
                "\n",
                "criterion = nn.CrossEntropyLoss(weight=weights_tensor)\n",
                "optimizer = optim.Adam(model.parameters(), lr=CONFIG[\"LR\"])\n",
                "\n",
                "history = {'train_loss': [], 'val_acc': []}\n",
                "best_acc = 0.0\n",
                "for epoch in range(CONFIG['EPOCHS']):\n",
                "    model.train()\n",
                "    running_loss = 0.0\n",
                "    for inputs, labels in train_loader:\n",
                "        inputs, labels = inputs.to(device), labels.to(device)\n",
                "        optimizer.zero_grad()\n",
                "        outputs, _ = model(inputs)\n",
                "        loss = criterion(outputs, labels)\n",
                "        loss.backward(); optimizer.step()\n",
                "        running_loss += loss.item()\n",
                "    \n",
                "    model.eval()\n",
                "    correct = 0; total = 0\n",
                "    with torch.no_grad():\n",
                "        for inputs, labels in val_loader:\n",
                "            inputs, labels = inputs.to(device), labels.to(device)\n",
                "            outputs, _ = model(inputs)\n",
                "            _, predicted = torch.max(outputs.data, 1)\n",
                "            total += labels.size(0); correct += (predicted == labels).sum().item()\n",
                "    val_acc = 100 * correct / total; train_loss = running_loss / len(train_loader)\n",
                "    history['train_loss'].append(train_loss); history['val_acc'].append(val_acc)\n",
                "    print(f\"Epoch [{epoch+1}/{CONFIG['EPOCHS']}] - Loss: {train_loss:.4f} - Val Acc: {val_acc:.2f}%\")\n",
                "    if val_acc > best_acc:\n",
                "        best_acc = val_acc\n",
                "        torch.save(model.state_dict(), CONFIG[\"MODEL_OUTPUT\"])\n",
                "        print(f\"  --> Guardado mejor modelo ({best_acc:.2f}%)\")"
            ]

    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=1)
    
    print("✅ Notebook actualizado correctamente.")

if __name__ == "__main__":
    update_notebook()
