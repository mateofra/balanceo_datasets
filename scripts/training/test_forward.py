from __future__ import annotations

import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stgcn.hand_graph import build_adjacency_matrix
from src.stgcn.stgcn_model import STGCN


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible en este entorno.")

    device = torch.device("cuda")
    adjacency = build_adjacency_matrix().to(device)
    model = STGCN(adjacency).to(device)

    x = torch.randn(8, 16, 21, 3, device=device)
    recon, attn = model(x)

    print(f"recon shape: {tuple(recon.shape)}")
    print(f"attn shape:  {tuple(attn.shape)}")
    print(f"attn suma:   {attn[0].sum().item():.4f}")
    print("Forward pass OK")


if __name__ == "__main__":
    main()