from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stgcn.hand_graph import HAND_EDGES, build_adjacency_matrix
from src.stgcn.stgcn_model import STGCN


DEFAULT_TONES = ("claro", "medio", "oscuro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera una visualizacion por tono MST con esqueleto de mano y brillo "
            "de nodos segun atencion espacial del ST-GCN entrenado."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/training/best_stgcn_supervisado.pth"),
        help="Checkpoint entrenado del ST-GCN supervisado",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("output/training/train_manifest_stgcn_fixed.csv"),
        help="Manifiesto ST-GCN usado para seleccionar muestras por tono",
    )
    parser.add_argument(
        "--seq-dir",
        type=Path,
        default=Path("data/processed/secuencias_stgcn"),
        help="Directorio con secuencias .npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("graficos/stgcn_atencion_mst"),
        help="Carpeta de salida para imagenes y metadata",
    )
    parser.add_argument(
        "--tones",
        nargs="+",
        default=list(DEFAULT_TONES),
        help="Tonos MST a renderizar (por defecto: claro medio oscuro)",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Frame a visualizar (default -1 = ultimo frame)",
    )
    parser.add_argument(
        "--image-roots",
        nargs="+",
        default=[
            "datasets/FreiHAND_pub_v2/training/rgb",
            "data/raw/images",
            "datasets/hagrid_dataset",
        ],
        help=(
            "Directorios base donde buscar imagenes originales por sample_id "
            "(por defecto: FreiHAND rgb, data/raw/images y datasets/hagrid_dataset)"
        ),
    )
    parser.add_argument(
        "--allow-missing-originals",
        action="store_true",
        help=(
            "Permite continuar cuando no se encuentra imagen original. "
            "Por defecto el script falla para forzar composiciones con imagen real."
        ),
    )
    parser.add_argument(
        "--tone-strategy",
        choices=["manifest", "image"],
        default="image",
        help=(
            "Estrategia para elegir muestras por tono: 'manifest' usa condition del CSV; "
            "'image' infiere tono desde la imagen original (recomendado cuando MST fue imputado)."
        ),
    )
    parser.add_argument(
        "--freihand-k-json",
        type=Path,
        default=Path("datasets/FreiHAND_pub_v2/training_K.json"),
        help="Ruta a training_K.json para proyectar XYZ de FreiHAND en pixeles.",
    )
    return parser.parse_args()


def load_filtered_manifest(manifest_csv: Path, seq_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)

    seq_paths = [str(seq_dir / f"{sid}.npy") for sid in df["sample_id"].tolist()]
    df = df.assign(seq_path=seq_paths)
    df = df[df["seq_path"].map(lambda p: Path(str(p)).exists())].reset_index(drop=True)

    return df


def build_model(checkpoint_path: Path, device: torch.device) -> tuple[STGCN, dict[int, str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError(
            "Checkpoint no compatible: se esperaba un dict con 'model_state'."
        )

    idx_to_class_raw = checkpoint.get("idx_to_class")
    if not isinstance(idx_to_class_raw, dict) or not idx_to_class_raw:
        raise RuntimeError("Checkpoint sin mapeo idx_to_class; no se puede etiquetar prediccion.")

    idx_to_class = {int(k): str(v) for k, v in idx_to_class_raw.items()}

    adjacency = build_adjacency_matrix().to(device)
    model = STGCN(adjacency, num_classes=len(idx_to_class)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, idx_to_class


def normalize_attention(attn: np.ndarray) -> np.ndarray:
    min_val = float(attn.min())
    max_val = float(attn.max())
    if max_val - min_val < 1e-9:
        return np.full_like(attn, 0.5, dtype=np.float32)
    return ((attn - min_val) / (max_val - min_val)).astype(np.float32)


def draw_skeleton_attention(
    coords_xy: np.ndarray,
    attn_norm: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    for src, dst in HAND_EDGES:
        edge_strength = float((attn_norm[src] + attn_norm[dst]) / 2.0)
        edge_color = (edge_strength, edge_strength, edge_strength)
        ax.plot(
            [coords_xy[src, 0], coords_xy[dst, 0]],
            [coords_xy[src, 1], coords_xy[dst, 1]],
            color=edge_color,
            linewidth=2.0 + 2.0 * edge_strength,
            alpha=0.95,
            zorder=1,
        )

    node_sizes = 100 + 500 * attn_norm
    scatter = ax.scatter(
        coords_xy[:, 0],
        coords_xy[:, 1],
        c=attn_norm,
        cmap="gray",
        s=node_sizes,
        edgecolors="#202020",
        linewidths=0.7,
        zorder=2,
    )

    for node_idx, (x, y) in enumerate(coords_xy):
        ax.text(
            float(x),
            float(y),
            str(node_idx),
            color="#4ad7a8",
            fontsize=7,
            ha="center",
            va="center",
            zorder=3,
        )

    ax.set_title(title, color="white", fontsize=10, pad=10)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Atencion normalizada", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_skeleton_attention_on_axis(
    ax,
    coords_xy: np.ndarray,
    attn_norm: np.ndarray,
    title: str,
):
    ax.set_facecolor("#111111")

    for src, dst in HAND_EDGES:
        edge_strength = float((attn_norm[src] + attn_norm[dst]) / 2.0)
        edge_color = (edge_strength, edge_strength, edge_strength)
        ax.plot(
            [coords_xy[src, 0], coords_xy[dst, 0]],
            [coords_xy[src, 1], coords_xy[dst, 1]],
            color=edge_color,
            linewidth=2.0 + 2.0 * edge_strength,
            alpha=0.95,
            zorder=1,
        )

    node_sizes = 100 + 500 * attn_norm
    scatter = ax.scatter(
        coords_xy[:, 0],
        coords_xy[:, 1],
        c=attn_norm,
        cmap="gray",
        s=node_sizes,
        edgecolors="#202020",
        linewidths=0.7,
        zorder=2,
    )

    for node_idx, (x, y) in enumerate(coords_xy):
        ax.text(
            float(x),
            float(y),
            str(node_idx),
            color="#4ad7a8",
            fontsize=7,
            ha="center",
            va="center",
            zorder=3,
        )

    ax.set_title(title, color="white", fontsize=10, pad=10)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    return scatter


def find_original_image(
    sample_id: str,
    label: str,
    image_roots: list[Path],
) -> Path | None:
    candidates: list[Path] = []
    extensions = [".jpg", ".jpeg", ".png", ".webp"]
    stems = [sample_id]
    if sample_id.lower().startswith("freihand_"):
        numeric = sample_id.split("_", 1)[1]
        stems.insert(0, numeric)

    for root in image_roots:
        for stem in stems:
            for ext in extensions:
                candidates.append(root / label / f"{stem}{ext}")
                candidates.append(root / f"{stem}{ext}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback costoso: busqueda recursiva por sample_id.
    for root in image_roots:
        if not root.exists():
            continue
        for ext in extensions:
            matches = list(root.rglob(f"{sample_id}{ext}"))
            if matches:
                return matches[0]

    return None


def draw_original_plus_skeleton(
    original_image_path: Path | None,
    coords_xy: np.ndarray,
    attn_norm: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=160)
    fig.patch.set_facecolor("#111111")

    ax_img = axes[0]
    ax_skel = axes[1]

    ax_img.set_facecolor("#111111")
    if original_image_path is not None and original_image_path.exists():
        image = plt.imread(original_image_path)
        ax_img.imshow(image)
        ax_img.set_title(
            f"Imagen original\n{original_image_path.name}",
            color="white",
            fontsize=10,
            pad=10,
        )
    else:
        ax_img.text(
            0.5,
            0.5,
            "Imagen original\nno disponible localmente",
            color="white",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax_img.set_title("Imagen original", color="white", fontsize=10, pad=10)
    ax_img.axis("off")

    scatter = draw_skeleton_attention_on_axis(
        ax=ax_skel,
        coords_xy=coords_xy,
        attn_norm=attn_norm,
        title="Esqueleto + atencion",
    )

    cbar = fig.colorbar(scatter, ax=ax_skel, fraction=0.046, pad=0.04)
    cbar.set_label("Atencion normalizada", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")

    fig.suptitle(title, color="white", fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def select_row_for_tone(df: pd.DataFrame, tone: str) -> pd.Series:
    subset = df[df["condition"] == tone].copy()
    if subset.empty:
        raise RuntimeError(f"No hay muestras disponibles para tono '{tone}'.")
    subset["dataset_priority"] = subset["dataset"].map(lambda value: 0 if str(value).lower() == "freihand" else 1)
    subset = subset.sort_values(["dataset_priority", "sample_id"]).reset_index(drop=True)
    return subset.iloc[0]


def _sample_idx_from_freihand_id(sample_id: str) -> int | None:
    if not sample_id.startswith("freihand_"):
        return None
    raw_idx = sample_id.split("_", 1)[1]
    if not raw_idx.isdigit():
        return None
    return int(raw_idx)


def _load_freihand_k(path: Path) -> list[np.ndarray] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return None
    mats: list[np.ndarray] = []
    for item in payload:
        arr = np.asarray(item, dtype=np.float32)
        if arr.shape == (3, 3):
            mats.append(arr)
    return mats if mats else None


def _project_xyz_freihand(frame_xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = np.clip(frame_xyz[:, 2], 1e-6, None)
    u = k[0, 0] * (frame_xyz[:, 0] / z) + k[0, 2]
    v = k[1, 1] * (frame_xyz[:, 1] / z) + k[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32)


def _project_generic_xy(frame_xyz: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
    coords = frame_xyz[:, :2].astype(np.float32).copy()
    if float(coords.max()) <= 2.0 and float(coords.min()) >= -1.0:
        coords[:, 0] *= image_w
        coords[:, 1] *= image_h
    return coords


def _infer_tone_from_image(
    sample_id: str,
    seq: np.ndarray,
    image: np.ndarray,
    freihand_k: list[np.ndarray] | None,
) -> str:
    h, w = image.shape[:2]
    frame_xyz = seq[-1].astype(np.float32)

    idx = _sample_idx_from_freihand_id(sample_id)
    if idx is not None and freihand_k is not None and idx < len(freihand_k):
        coords = _project_xyz_freihand(frame_xyz, freihand_k[idx])
    else:
        coords = _project_generic_xy(frame_xyz, h, w)

    vals: list[float] = []
    for x, y in coords:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            continue
        x0 = max(0, xi - 2)
        x1 = min(w, xi + 3)
        y0 = max(0, yi - 2)
        y1 = min(h, yi + 3)
        patch = image[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        patch = patch.astype(np.float32)
        if patch.max() > 1.5:
            patch = patch / 255.0
        luma = 0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2]
        vals.append(float(np.median(luma)))

    if not vals:
        return "medio"

    score = float(np.median(np.asarray(vals, dtype=np.float32)))
    if score >= 0.58:
        return "claro"
    if score >= 0.44:
        return "medio"
    return "oscuro"


def frame_from_index(seq: np.ndarray, frame_index: int) -> np.ndarray:
    idx = frame_index
    if idx < 0:
        idx = seq.shape[0] + idx
    idx = max(0, min(idx, seq.shape[0] - 1))
    return seq[idx]


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = (ROOT / args.manifest_csv).resolve()
    seq_dir = (ROOT / args.seq_dir).resolve()
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    image_roots = [(ROOT / Path(p)).resolve() for p in args.image_roots]
    freihand_k = _load_freihand_k((ROOT / args.freihand_k_json).resolve())

    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe manifiesto: {manifest_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_filtered_manifest(manifest_path, seq_dir)
    if df.empty:
        raise RuntimeError("No hay muestras validas luego del filtrado de manifest + secuencias.")

    model, idx_to_class = build_model(checkpoint_path, device)

    metadata: list[dict[str, object]] = []

    for tone in args.tones:
        row = select_row_for_tone(df, tone)

        if args.tone_strategy == "image":
            candidates = df.copy()
            candidates["dataset_priority"] = candidates["dataset"].map(
                lambda value: 0 if str(value).lower() == "freihand" else 1
            )
            candidates = candidates.sort_values(["dataset_priority", "sample_id"]).reset_index(drop=True)

            inspected = 0
            selected = None
            for _, cand in candidates.iterrows():
                seq_path_c = Path(str(cand["seq_path"]))
                if not seq_path_c.exists():
                    continue
                image_path_c = find_original_image(
                    sample_id=str(cand["sample_id"]),
                    label=str(cand["label"]),
                    image_roots=image_roots,
                )
                if image_path_c is None or not image_path_c.exists():
                    continue

                seq_c = np.load(seq_path_c).astype(np.float32)
                img_c = plt.imread(image_path_c)
                inferred = _infer_tone_from_image(
                    sample_id=str(cand["sample_id"]),
                    seq=seq_c,
                    image=img_c,
                    freihand_k=freihand_k,
                )
                inspected += 1
                if inferred == tone:
                    selected = cand
                    break
                if inspected >= 2500:
                    break

            if selected is not None:
                row = selected

        seq_path = Path(str(row["seq_path"]))
        seq = np.load(seq_path).astype(np.float32)  # (T, 21, 3)

        x = torch.from_numpy(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, attn = model(x)

        pred_idx = int(logits.argmax(dim=1).item())
        pred_label = idx_to_class.get(pred_idx, str(pred_idx))
        true_label = str(row["label"])

        attn_np = attn.squeeze(0).detach().cpu().numpy().astype(np.float32)
        attn_norm = normalize_attention(attn_np)

        frame_xyz = frame_from_index(seq, args.frame_index)
        coords_xy = frame_xyz[:, :2]

        out_path = output_dir / f"{tone}_stgcn_atencion.png"
        out_comp_path = output_dir / f"{tone}_stgcn_composicion.png"
        title = (
            f"Tono {tone} | sample {row['sample_id']}\n"
            f"true={true_label} | pred={pred_label}"
        )

        original_image_path = find_original_image(
            sample_id=str(row["sample_id"]),
            label=true_label,
            image_roots=image_roots,
        )
        if original_image_path is None and not args.allow_missing_originals:
            roots_txt = ", ".join(str(p) for p in image_roots)
            raise RuntimeError(
                "No se encontró imagen original real para "
                f"sample_id={row['sample_id']} (tone={tone}, label={true_label}). "
                f"Buscado en: {roots_txt}. "
                "Usa --image-roots para apuntar a la ruta correcta o "
                "--allow-missing-originals para permitir fallback."
            )

        draw_skeleton_attention(coords_xy, attn_norm, title, out_path)
        draw_original_plus_skeleton(
            original_image_path=original_image_path,
            coords_xy=coords_xy,
            attn_norm=attn_norm,
            title=title,
            output_path=out_comp_path,
        )

        metadata.append(
            {
                "tone": tone,
                "tone_strategy": args.tone_strategy,
                "sample_id": str(row["sample_id"]),
                "seq_path": str(seq_path),
                "original_image_path": str(original_image_path) if original_image_path else None,
                "true_label": true_label,
                "pred_label": pred_label,
                "output_image": str(out_path),
                "output_composition": str(out_comp_path),
                "attention": [float(v) for v in attn_np],
                "attention_normalized": [float(v) for v in attn_norm],
            }
        )

        print(f"OK {tone}: {out_path}")
        print(f"OK {tone} composicion: {out_comp_path}")

    metadata_path = output_dir / "metadata_atencion_stgcn.json"
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "manifest_csv": str(manifest_path),
                "seq_dir": str(seq_dir),
                "device": str(device),
                "tones": args.tones,
                "items": metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
