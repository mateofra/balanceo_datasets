import json
from pathlib import Path

def fix_notebook():
    notebook_path = Path("entrenar_stgcn.ipynb")
    with open(notebook_path, "r") as f:
        nb = json.load(f)

    for cell in nb['cells']:
        # Fix Imports/Init Cell to define device
        if cell['id'] == 'stgcn-imports':
            cell['source'] = [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import cv2\n",
                "import mediapipe as mp\n",
                "from mediapipe.tasks import python\n",
                "from mediapipe.tasks.python import vision\n",
                "from pathlib import Path\n",
                "from torch.utils.data import DataLoader, Dataset\n",
                "from datetime import datetime\n",
                "\n",
                "# Asegurar que el directorio raiz esta en el path para importar src\n",
                "ROOT_DIR = Path.cwd()\n",
                "if str(ROOT_DIR) not in sys.path:\n",
                "    sys.path.insert(0, str(ROOT_DIR))\n",
                "\n",
                "from src.stgcn.stgcn_model import RealSTGCN\n",
                "from src.stgcn.hand_graph import build_adjacency_matrix\n",
                "\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "print(f\"PyTorch version: {torch.__version__}\")\n",
                "print(f\"Device: {device}\")"
            ]
        
        # Remove redundant device definition in Cell 4
        if cell['id'] == 'stgcn-model-code':
             cell['source'] = [
                "adj = build_adjacency_matrix().to(device)\n",
                "\n",
                "model = RealSTGCN(\n",
                "    num_classes=len(unique_labels),\n",
                "    adjacency=adj,\n",
                "    in_channels=3,\n",
                "    dropout=0.3\n",
                ").to(device)\n",
                "\n",
                "print(\"Modelo ST-GCN con Atención Espacial inicializado.\")"
            ]

    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=1)
    
    print("✅ Notebook corregido (device definido al inicio).")

if __name__ == "__main__":
    fix_notebook()
