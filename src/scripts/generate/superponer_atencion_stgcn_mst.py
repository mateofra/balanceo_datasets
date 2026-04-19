from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stgcn.hand_graph import HAND_EDGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Superpone el esqueleto con pesos de atencion ST-GCN sobre la imagen "
            "original para cada tono MST (claro/medio/oscuro)."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("graficos/stgcn_atencion_mst/metadata_atencion_stgcn.json"),
        help="Metadata generado por visualizar_atencion_stgcn_mst.py",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Frame a usar de la secuencia (default -1 = ultimo frame)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("graficos/stgcn_atencion_mst"),
        help="Directorio de salida para overlays",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=0.9,
        help="Transparencia de lineas del esqueleto",
    )
    parser.add_argument(
        "--node-alpha",
        type=float,
        default=0.95,
        help="Transparencia de nodos de atencion",
    )
    parser.add_argument(
        "--freihand-k-json",
        type=Path,
        default=Path("datasets/FreiHAND_pub_v2/training_K.json"),
        help=(
            "Ruta al training_K.json de FreiHAND para proyectar landmarks 3D "
            "a coordenadas de imagen"
        ),
    )
    return parser.parse_args()


def frame_from_index(seq: np.ndarray, frame_index: int) -> np.ndarray:
    idx = frame_index
    if idx < 0:
        idx = seq.shape[0] + idx
    idx = max(0, min(idx, seq.shape[0] - 1))
    return seq[idx]


def normalize_attention(attn: np.ndarray) -> np.ndarray:
    min_val = float(attn.min())
    max_val = float(attn.max())
    if max_val - min_val < 1e-9:
        return np.full_like(attn, 0.5, dtype=np.float32)
    return ((attn - min_val) / (max_val - min_val)).astype(np.float32)


def map_coords_to_image(coords_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    coords = coords_xy.astype(np.float32).copy()

    # Si parece normalizado (0..1 o aprox), escalar a pixeles.
    if float(coords.max()) <= 2.0 and float(coords.min()) >= -1.0:
        coords[:, 0] *= width
        coords[:, 1] *= height

    return coords


def load_freihand_k_matrices(k_json_path: Path) -> list[np.ndarray] | None:
    if not k_json_path.exists():
        return None

    payload = json.loads(k_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return None

    matrices: list[np.ndarray] = []
    for item in payload:
        arr = np.asarray(item, dtype=np.float32)
        if arr.shape == (3, 3):
            matrices.append(arr)
    return matrices if matrices else None


def sample_index_from_freihand_id(sample_id: str) -> int | None:
    prefix = "freihand_"
    if not sample_id.startswith(prefix):
        return None
    raw_idx = sample_id[len(prefix):]
    if not raw_idx.isdigit():
        return None
    return int(raw_idx)


def project_xyz_with_k(frame_xyz: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    xyz = frame_xyz.astype(np.float32)
    z = np.clip(xyz[:, 2], 1e-6, None)
    x = xyz[:, 0]
    y = xyz[:, 1]

    fx = float(k_matrix[0, 0])
    fy = float(k_matrix[1, 1])
    cx = float(k_matrix[0, 2])
    cy = float(k_matrix[1, 2])

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return np.stack([u, v], axis=1).astype(np.float32)


def draw_overlay(
    image_path: Path,
    coords_xy: np.ndarray,
    attn_norm: np.ndarray,
    title: str,
    output_path: Path,
    line_alpha: float,
    node_alpha: float,
) -> None:
    image = plt.imread(image_path)
    height, width = image.shape[:2]

    px_coords = map_coords_to_image(coords_xy, width=width, height=height)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    ax.imshow(image)

    # Esqueleto con grosor/intensidad segun atencion media del borde.
    for src, dst in HAND_EDGES:
        strength = float((attn_norm[src] + attn_norm[dst]) / 2.0)
        color = (1.0, 0.95 * strength, 0.1 * (1.0 - strength))
        ax.plot(
            [px_coords[src, 0], px_coords[dst, 0]],
            [px_coords[src, 1], px_coords[dst, 1]],
            color=color,
            linewidth=1.5 + 4.0 * strength,
            alpha=line_alpha,
            zorder=2,
        )

    node_sizes = 60 + 360 * attn_norm
    scatter = ax.scatter(
        px_coords[:, 0],
        px_coords[:, 1],
        c=attn_norm,
        cmap="inferno",
        s=node_sizes,
        alpha=node_alpha,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

    for node_idx, (x, y) in enumerate(px_coords):
        ax.text(
            float(x),
            float(y),
            str(node_idx),
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            zorder=4,
        )

    ax.set_title(title, fontsize=10)
    ax.axis("off")

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Atencion normalizada")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    metadata_path = (ROOT / args.metadata).resolve()
    output_dir = (ROOT / args.output_dir).resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe metadata: {metadata_path}")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not items:
        raise RuntimeError("Metadata sin items para renderizar.")

    freihand_k_path = (ROOT / args.freihand_k_json).resolve()
    freihand_k = load_freihand_k_matrices(freihand_k_path)

    generated = 0

    for item in items:
        tone = str(item.get("tone", "tono"))
        sample_id = str(item.get("sample_id", "unknown"))
        seq_path = Path(str(item.get("seq_path", "")))
        img_path_raw = item.get("original_image_path")
        pred_label = str(item.get("pred_label", "?"))
        true_label = str(item.get("true_label", "?"))

        if not seq_path.exists():
            print(f"SKIP {tone}: secuencia no existe ({seq_path})")
            continue

        if not img_path_raw:
            print(f"SKIP {tone}: metadata sin original_image_path")
            continue

        img_path = Path(str(img_path_raw))
        if not img_path.exists():
            print(f"SKIP {tone}: imagen no existe ({img_path})")
            continue

        seq = np.load(seq_path).astype(np.float32)
        frame_xyz = frame_from_index(seq, args.frame_index)

        # FreiHAND usa XYZ en espacio de camara; para overlay correcto
        # hay que proyectar con la matriz intrinseca K.
        sample_idx = sample_index_from_freihand_id(sample_id)
        if sample_idx is not None and freihand_k is not None and sample_idx < len(freihand_k):
            coords_xy = project_xyz_with_k(frame_xyz, freihand_k[sample_idx])
        else:
            coords_xy = frame_xyz[:, :2]

        attn = np.asarray(item.get("attention_normalized", []), dtype=np.float32)
        if attn.shape != (21,):
            raw_attn = np.asarray(item.get("attention", []), dtype=np.float32)
            if raw_attn.shape != (21,):
                print(f"SKIP {tone}: atencion invalida para sample {sample_id}")
                continue
            attn = normalize_attention(raw_attn)

        out_path = output_dir / f"{tone}_stgcn_overlay.png"
        title = f"{tone} | {sample_id} | true={true_label} pred={pred_label}"

        draw_overlay(
            image_path=img_path,
            coords_xy=coords_xy,
            attn_norm=attn,
            title=title,
            output_path=out_path,
            line_alpha=args.line_alpha,
            node_alpha=args.node_alpha,
        )

        print(f"OK overlay {tone}: {out_path}")
        generated += 1

    if generated == 0:
        raise RuntimeError(
            "No se genero ningun overlay. Verifica metadata y disponibilidad de imagenes."
        )

    print(f"Overlays generados: {generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
