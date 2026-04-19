from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.stgcn.hand_graph import build_adjacency_matrix
from src.stgcn.stgcn_model import STGCN

MANIFEST = REPO_ROOT / 'output' / 'manifest_unificado_final.csv'
CHECKPOINT = REPO_ROOT / 'output' / 'best_stgcn_canonico.pth'


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.paths = self.df['path_secuencia'].tolist()
        self.labels = [class_to_idx[label] for label in self.df['label']]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        x = np.load(str(REPO_ROOT / self.paths[idx].replace('\\', '/')))
        y = self.labels[idx]
        condition = self.df.loc[idx, 'condition']
        label = self.df.loc[idx, 'label']
        sample_id = self.df.loc[idx, 'sample_id']
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            condition,
            label,
            sample_id,
        )


def collate_fn(batch):
    xs, ys, conditions, labels, sample_ids = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(conditions), list(labels), list(sample_ids)


def canonical_tvd_by_pairs(df_res: pd.DataFrame):
    blocks = ['claro', 'medio', 'oscuro']
    all_labels = sorted(df_res['label'].unique())

    error_distributions = {}
    for block in blocks:
        sub = df_res[df_res['condition'] == block]
        errors = sub[sub['correcto'] == 0]['label'].value_counts(normalize=True)
        error_distributions[block] = errors.reindex(all_labels, fill_value=0.0)

    pairs = [('claro', 'medio'), ('claro', 'oscuro'), ('medio', 'oscuro')]
    tvd_rows = []
    for b1, b2 in pairs:
        p = error_distributions[b1]
        q = error_distributions[b2]
        tvd = 0.5 * (p - q).abs().sum()
        tvd_rows.append((f'{b1}_vs_{b2}', float(tvd)))
    return tvd_rows


def main() -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(f'No existe manifiesto: {MANIFEST}')
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f'No existe checkpoint: {CHECKPOINT}')

    df = pd.read_csv(MANIFEST)
    if 'path_secuencia' not in df.columns and 'path' in df.columns:
        df = df.rename(columns={'path': 'path_secuencia'})
    if 'dataset' not in df.columns and 'source' in df.columns:
        df = df.rename(columns={'source': 'dataset'})
    if 'condition' not in df.columns and 'mst_imputed' in df.columns:
        df = df.rename(columns={'mst_imputed': 'condition'})
    if 'condition' not in df.columns:
        df['condition'] = 'medio'
    if 'sample_id' not in df.columns:
        df['sample_id'] = df['path_secuencia'].apply(lambda p: Path(str(p)).stem)

    df_test = df[df['split'] == 'test'].copy().reset_index(drop=True)

    print(f'Manifiesto: {MANIFEST}')
    print(f'Checkpoint: {CHECKPOINT}')
    print(f'Muestras test: {len(df_test)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    class_to_idx = checkpoint['class_to_idx']

    dataset = TestDataset(df_test, class_to_idx)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)

    A = build_adjacency_matrix().to(device)
    model = STGCN(A, num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    results = []
    with torch.no_grad():
        for x, y, conditions, labels, sample_ids in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            correct = (preds == y).cpu().numpy().tolist()

            for i in range(len(correct)):
                results.append(
                    {
                        'sample_id': sample_ids[i],
                        'label': labels[i],
                        'condition': conditions[i],
                        'correcto': int(correct[i]),
                    }
                )

    df_res = pd.DataFrame(results)
    overall_acc = float(df_res['correcto'].mean())

    print('\n=== Accuracy por bloque MST (test) ===')
    by_block = {}
    for cond, sub in df_res.groupby('condition'):
        acc = float(sub['correcto'].mean())
        by_block[cond] = acc
        print(f'  {cond:8s}: {acc:.3f} (n={len(sub)})')

    dpr = min(by_block.values()) / max(by_block.values()) if by_block else 0.0
    print(f'\nDPR (min/max): {dpr:.3f}')

    tvd_rows = canonical_tvd_by_pairs(df_res)
    print('\n=== TVD canónico por par ===')
    for pair, tvd in tvd_rows:
        print(f'  {pair}: {tvd:.3f}')

    out_csv = REPO_ROOT / 'output' / 'auditoria_final_test.csv'
    df_res.to_csv(out_csv, index=False)

    print(f'\nOverall test acc: {overall_acc:.3f}')
    print(f'Resultados guardados: {out_csv}')


if __name__ == '__main__':
    main()
