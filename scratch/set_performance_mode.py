import json
from pathlib import Path

def setup_performance_mode_notebooks():
    paths = ["entrenar_stgcn.ipynb", "colab/entrenar_stgcn_colab.ipynb"]
    
    for p in paths:
        path = Path(p)
        if not path.exists(): continue
        
        with open(path, "r") as f:
            nb = json.load(f)

        for cell in nb['cells']:
            # 1. Update Config (Aggressive LR and Batch Size)
            if cell['id'] == 'stgcn-params':
                source = "".join(cell['source'])
                source = source.replace('"LR": 0.0001', '"LR": 0.001')
                source = source.replace('"BATCH_SIZE": 64', '"BATCH_SIZE": 128')
                cell['source'] = [l + "\n" for l in source.split("\n") if l]

            # 2. Update Training Loop (Clipping and Smoothing)
            if cell['id'] == 'stgcn-train-exec':
                source = "".join(cell['source'])
                # Add Label Smoothing
                source = source.replace('nn.CrossEntropyLoss(weight=weights_tensor)', 'nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)')
                
                # Add Gradient Clipping before optimizer.step()
                if "optimizer.step()" in source and "clip_grad_norm_" not in source:
                    source = source.replace('loss.backward()', 'loss.backward()\n        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)')
                
                cell['source'] = [l + "\n" for l in source.split("\n") if l]

        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        
        print(f"✅ Performance Mode aplicado a {p}")

if __name__ == "__main__":
    setup_performance_mode_notebooks()
