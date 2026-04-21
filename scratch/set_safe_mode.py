import json
from pathlib import Path

def setup_safe_mode_notebooks():
    paths = ["entrenar_stgcn.ipynb", "colab/entrenar_stgcn_colab.ipynb"]
    
    for p in paths:
        path = Path(p)
        if not path.exists(): continue
        
        with open(path, "r") as f:
            nb = json.load(f)

        for cell in nb['cells']:
            # 1. Lower LR in Config
            if cell['id'] == 'stgcn-params':
                source = "".join(cell['source'])
                source = source.replace('"LR": 0.001', '"LR": 0.0001')
                source = source.replace('"EPOCHS": 15', '"EPOCHS": 20')
                source = source.replace('"EPOCHS": 30', '"EPOCHS": 20')
                cell['source'] = [l + "\n" for l in source.split("\n") if l]

            # 2. Disable Augmentation and Weights in Training Cell
            if cell['id'] == 'stgcn-train-exec':
                source = "".join(cell['source'])
                # Disable augment in loaders
                source = source.replace('augment=True', 'augment=False')
                # Disable weights in criterion
                source = source.replace('nn.CrossEntropyLoss(weight=weights_tensor)', 'nn.CrossEntropyLoss()')
                cell['source'] = [l + "\n" for l in source.split("\n") if l]

        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        
        print(f"✅ Safe Mode aplicado a {p}")

if __name__ == "__main__":
    setup_safe_mode_notebooks()
