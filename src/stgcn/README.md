# ST-GCN

Contiene los componentes del modelo ST-GCN y utilidades de entrenamiento de ejemplo.

## Contenido

- `hand_graph.py`: topologia anatomica (21 nodos) y adyacencia normalizada.
- `stgcn_model.py`: modelo ST-GCN con atencion espacial por nodo.
- `st_gcn_dataloader.py`: DataLoader para manifiestos ST-GCN.
- `train_stgcn_example.py`: ejemplo de entrenamiento basico.

## Uso rapido

```bash
uv run python src/stgcn/train_stgcn_example.py
```

## Integracion

Para usar el modelo desde otros scripts:

```python
from src.stgcn.hand_graph import build_adjacency_matrix
from src.stgcn.stgcn_model import STGCN
```
