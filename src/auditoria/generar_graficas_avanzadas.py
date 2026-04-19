import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "output" / "graficas_avanzadas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.stgcn.hand_graph import build_adjacency_matrix
from src.stgcn.stgcn_model import STGCN

MANIFEST_PATH = REPO_ROOT / "output" / "manifest_unificado_final.csv"
HISTORY_PATH = REPO_ROOT / "output" / "training_history_canonico.json"
CHECKPOINT_PATH = REPO_ROOT / "output" / "best_stgcn_canonico.pth"


class GestureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str, int], base: Path):
        self.base = base
        self.paths = df["path_secuencia"].tolist()
        self.labels_text = df["label"].tolist()
        self.labels = [class_to_idx[l] for l in self.labels_text]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.base / self.paths[idx].replace("\\", "/")
        x = np.load(str(p)).astype(np.float32)
        return torch.tensor(x, dtype=torch.float32), self.labels[idx], self.labels_text[idx], str(p)


def normalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    if "path_secuencia" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "path_secuencia"})
    if "condition" not in df.columns and "mst_imputed" in df.columns:
        df = df.rename(columns={"mst_imputed": "condition"})
    if "condition" not in df.columns:
        df["condition"] = "medio"
    return df


def load_model(device: torch.device):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    A = build_adjacency_matrix().to(device)
    model = STGCN(A, num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, class_to_idx, idx_to_class


def graph_07_kine_kinemorfo(df_test: pd.DataFrame):
    first_path = REPO_ROOT / df_test.iloc[0]["path_secuencia"].replace("\\", "/")
    seq = np.load(str(first_path))
    idx_tip = seq[:, 8, :]  # nodo 8

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(idx_tip[:, 0], idx_tip[:, 1], idx_tip[:, 2], color="#1f77b4", linewidth=2.5, label="Kinemorfo")
    ax.scatter(idx_tip[:, 0], idx_tip[:, 1], idx_tip[:, 2], c=np.arange(len(idx_tip)), cmap="viridis", s=30)

    for t in range(len(idx_tip) - 1):
        x0, y0, z0 = idx_tip[t]
        x1, y1, z1 = idx_tip[t + 1]
        ax.quiver(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0, color="#ff7f0e", arrow_length_ratio=0.2, linewidth=1.2)

    ax.set_title("07 - Jerarquia Birdwhistell: Kine vs Kinemorfo (nodo 8)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.text(idx_tip[0, 0], idx_tip[0, 1], idx_tip[0, 2], "Kine", color="#ff7f0e")
    ax.legend(loc="upper right")

    out = OUT_DIR / "07_jerarquia_birdwhistell.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"OK {out.name}")


def graph_08_heatmap_importance(df_test: pd.DataFrame):
    sample_paths = df_test["path_secuencia"].tolist()
    stack = []
    for p in sample_paths[:1200]:
        seq = np.load(str(REPO_ROOT / p.replace("\\", "/")))  # (T,21,3)
        stack.append(seq)
    arr = np.stack(stack, axis=0)  # (N,T,21,3)

    node_var = arr.var(axis=(0, 1, 3))  # (21,)
    node_imp = (node_var - node_var.min()) / (node_var.max() - node_var.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 2.8))
    im = ax.imshow(node_imp.reshape(1, -1), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(21))
    ax.set_xticklabels([str(i) for i in range(21)])
    ax.set_yticks([0])
    ax.set_yticklabels(["Importancia"])
    fig.colorbar(im, ax=ax)
    plt.title("08 - Mapa de Calor de Atencion Espacial por Nodo (proxy por varianza)")
    plt.xlabel("Nodo de mano")
    out = OUT_DIR / "08_mapa_calor_atencion_nodos.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"OK {out.name}")


def collect_predictions(model, loader, device, idx_to_class):
    y_true = []
    y_pred = []
    logits_all = []
    labels_text = []
    with torch.no_grad():
        for x, y, label_text, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(1)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            logits_all.append(logits.cpu().numpy())
            labels_text.extend(label_text)

    logits_all = np.concatenate(logits_all, axis=0)
    true_names = [idx_to_class[i] for i in y_true]
    pred_names = [idx_to_class[i] for i in y_pred]
    return np.array(y_true), np.array(y_pred), true_names, pred_names, logits_all, labels_text


def graph_09_top_fp_matrix(true_names, pred_names):
    dfp = pd.DataFrame({"true": true_names, "pred": pred_names})
    err = dfp[dfp["true"] != dfp["pred"]].copy()

    fp_counts = err["pred"].value_counts()
    top5_pred = fp_counts.head(5).index.tolist()
    rows = err[err["pred"].isin(top5_pred)]["true"].value_counts().head(5).index.tolist()

    mtx = pd.crosstab(err["true"], err["pred"]).reindex(index=rows, columns=top5_pred, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(mtx.values, cmap="Reds", aspect="auto")
    ax.set_xticks(np.arange(len(mtx.columns)))
    ax.set_xticklabels(mtx.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mtx.index)))
    ax.set_yticklabels(mtx.index)
    for i in range(mtx.shape[0]):
        for j in range(mtx.shape[1]):
            ax.text(j, i, str(int(mtx.values[i, j])), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax)
    plt.title("09 - Matriz simplificada: Top 5 clases con mas falsos positivos")
    plt.xlabel("Clase predicha (FP)")
    plt.ylabel("Clase real")
    out = OUT_DIR / "09_matriz_confusion_top_errores.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"OK {out.name}")


def graph_10_tsne(logits_all, labels_text):
    n = min(2500, len(logits_all))
    idx = np.random.RandomState(42).choice(len(logits_all), size=n, replace=False)
    feats = logits_all[idx]
    lab = np.array(labels_text)[idx]

    emb = PCA(n_components=2, random_state=42).fit_transform(feats)

    class_to_num = {c: i for i, c in enumerate(sorted(set(lab.tolist())))}
    cvals = np.array([class_to_num[c] for c in lab])

    plt.figure(figsize=(9, 7))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=cvals, cmap="tab20", s=10, alpha=0.8)
    plt.title("10 - PCA de salidas de ultima capa (espacio latente)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(sc, label="Indice de clase")
    out = OUT_DIR / "10_tsne_clusters_gestos.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"OK {out.name}")


def evaluate_accuracy_with_noise(model, loader, device, sigma: float, max_batches: int = 8) -> float:
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (x, y, _, _) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            if sigma > 0:
                x = x + torch.randn_like(x) * sigma
            logits, _ = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if batch_idx + 1 >= max_batches:
                break
    return correct / max(1, total)


def graph_11_noise_robustness(model, loader, device):
    sigmas = [0.0, 0.005, 0.01, 0.02, 0.03]
    accs = [evaluate_accuracy_with_noise(model, loader, device, s) for s in sigmas]

    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, accs, marker="o", linewidth=2, color="#2a9d8f")
    plt.title("11 - Analisis de ruido y robustez")
    plt.xlabel("Sigma de ruido gaussiano")
    plt.ylabel("Accuracy en test")
    plt.grid(alpha=0.3)
    out = OUT_DIR / "11_analisis_ruido_robustez.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"OK {out.name}")


def benchmark_device(model, sample: torch.Tensor, device_name: str, iters: int = 30) -> float:
    sample = sample.to(device_name)
    model = model.to(device_name)
    model.eval()

    warmup = 10
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)
        if device_name.startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(sample)
        if device_name.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters


def graph_12_latency(model, sample: torch.Tensor):
    cpu_ms = benchmark_device(model, sample.clone(), "cpu")
    labels = ["CPU"]
    vals = [cpu_ms]

    if torch.cuda.is_available():
        gpu_ms = benchmark_device(model, sample.clone(), "cuda")
        labels.append("GPU RTX 4050")
        vals.append(gpu_ms)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, vals, color=["#f4a261", "#264653"][: len(vals)])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{v:.2f} ms", ha="center")
    plt.title("12 - Latencia de inferencia por hardware")
    plt.ylabel("Tiempo medio por secuencia (ms)")
    out = OUT_DIR / "12_latencia_inferencia_hardware.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"OK {out.name}")


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"No existe manifiesto: {MANIFEST_PATH}")
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"No existe historial: {HISTORY_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"No existe checkpoint: {CHECKPOINT_PATH}")

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        _ = json.load(f)

    df = pd.read_csv(MANIFEST_PATH)
    df = normalize_manifest(df)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_to_idx, idx_to_class = load_model(device)

    dataset = GestureDataset(df_test, class_to_idx, REPO_ROOT)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    graph_07_kine_kinemorfo(df_test)
    graph_08_heatmap_importance(df_test)

    y_true, y_pred, true_names, pred_names, logits_all, labels_text = collect_predictions(model, loader, device, idx_to_class)
    graph_09_top_fp_matrix(true_names, pred_names)
    graph_10_tsne(logits_all, labels_text)
    graph_11_noise_robustness(model, loader, device)

    sample, _, _, _ = dataset[0]
    sample = sample.unsqueeze(0)
    graph_12_latency(model, sample)

    print(f"\nGraficas avanzadas guardadas en: {OUT_DIR}")


if __name__ == "__main__":
    main()
