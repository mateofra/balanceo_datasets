from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stgcn.hand_graph import build_adjacency_matrix
from src.stgcn.stgcn_model import STGCN


def evaluar_por_mst(
    manifest_path: str = "output/training/train_manifest_stgcn_fixed.csv",
    modelo_path: str = "output/training/best_stgcn_supervisado.pth",
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(modelo_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    adjacency = build_adjacency_matrix().to(device)
    model = STGCN(adjacency, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    df = pd.read_csv(manifest_path)
    df = df[
        (df["dataset"] == "hagrid")
        & (df["landmark_quality"] == "annotation_2d_projected")
    ].copy()

    seq_dir = Path("data/processed/secuencias_stgcn")
    df["path_seq"] = df["sample_id"].apply(lambda sid: seq_dir / f"{sid}.npy")
    df = df[df["path_seq"].apply(lambda p: p.exists())].reset_index(drop=True)

    print(f"Muestras evaluadas: {len(df)}")
    print(f"Batch size: {batch_size}")

    resultados = []

    with torch.no_grad():
        for _, row in df.iterrows():
            x = np.load(str(row["path_seq"]))
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            logits, attn = model(x)
            pred = logits.argmax(1).item()
            correcto = int(pred == class_to_idx[row["label"]])
            resultados.append(
                {
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "mst": row["mst"],
                    "condition": row["condition"],
                    "correcto": correcto,
                    "attn_max_nodo": attn[0].argmax().item(),
                }
            )

    df_res = pd.DataFrame(resultados)

    print("\n=== Accuracy por bloque MST ===")
    for bloque, grupo in df_res.groupby("condition"):
        acc = grupo["correcto"].mean()
        print(f"  {bloque:8s}: {acc:.3f}  (n={len(grupo)})")

    accs = df_res.groupby("condition")["correcto"].mean()
    dpr = accs.min() / accs.max()
    print(f"\nDPR (min/max accuracy): {dpr:.3f}")
    print("  → Ideal: > 0.8  |  Crítico: < 0.6")

    distribuciones = df_res.groupby(["condition", "label"])["correcto"].mean().unstack(fill_value=0)
    tvd_legacy = 0.5 * (distribuciones.max() - distribuciones.min()).abs().sum()
    print(f"TVD legacy (proxy global): {tvd_legacy:.3f}")
    print("  → Nota: esta metrica se mantiene solo para compatibilidad historica")
    print("  → Referencia principal: TVD canonico por pares (ver bloque inferior)")

    auditoria_dir = Path("output/auditoria")
    auditoria_dir.mkdir(parents=True, exist_ok=True)
    output_csv = auditoria_dir / "auditoria_dpr_resultados.csv"
    df_res.to_csv(output_csv, index=False)
    print(f"\nResultados guardados: {output_csv}")

    tvds, tvd_max = calcular_tvd_correcto(df_res)
    print("\n=== TVD canónico por par ===")
    for item in tvds:
        print(f"  {item['par']}: {item['tvd']:.3f}")
    print(f"TVD máximo canónico: {tvd_max:.3f}")

    return df_res, dpr, tvd_legacy


def calcular_tvd_correcto(df_res):
    """
    TVD canónica entre distribuciones de error por bloque MST.
    Compara claro vs medio, claro vs oscuro, medio vs oscuro.
    Devuelve el TVD máximo entre pares (caso más conservador).
    """
    bloques = ['claro', 'medio', 'oscuro']

    # Distribución de errores por clase en cada bloque
    # (qué clases falla cada bloque, normalizado)
    distros = {}
    for bloque in bloques:
        sub = df_res[df_res['condition'] == bloque]
        errores = sub[sub['correcto'] == 0]['label'].value_counts(normalize=True)
        distros[bloque] = errores

    # Alinear índices entre bloques
    todas_clases = sorted(df_res['label'].unique())
    tvds = []
    pares = [('claro', 'medio'), ('claro', 'oscuro'), ('medio', 'oscuro')]

    for b1, b2 in pares:
        p = distros[b1].reindex(todas_clases, fill_value=0)
        q = distros[b2].reindex(todas_clases, fill_value=0)
        tvd = 0.5 * (p - q).abs().sum()
        tvds.append({'par': f"{b1}_vs_{b2}", 'tvd': round(float(tvd), 3)})
        print(f"  TVD {b1} vs {b2}: {tvd:.3f}")

    tvd_max = max(t['tvd'] for t in tvds)
    print(f"\nTVD máximo entre pares: {tvd_max:.3f}")
    print("  → Ideal: < 0.2  |  Aceptable: < 0.4")
    return tvds, tvd_max


if __name__ == "__main__":
    evaluar_por_mst()