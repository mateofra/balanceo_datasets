from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.balancer.balancear_freihand_hagrid import (
    _compute_summary,
)
from src.balancer.imputer import MstImputer
from src.balancer.loaders import DataLoaders, MstLoader


def augment_extreme_batch(
    x: torch.Tensor,
    mst_batch: torch.Tensor,
    *,
    extreme_levels: set[int],
    prob: float,
    noise_std: float,
    rot_deg: float,
    scale_jitter: float,
) -> torch.Tensor:
    if prob <= 0.0:
        return x

    x_aug = x.clone()
    for b in range(x_aug.shape[0]):
        mst_level = int(mst_batch[b].item())
        if mst_level not in extreme_levels:
            continue
        if random.random() > prob:
            continue

        sample = x_aug[b]
        scale = 1.0 + random.uniform(-scale_jitter, scale_jitter)
        sample = sample * scale

        ang = np.deg2rad(random.uniform(-rot_deg, rot_deg))
        c = float(np.cos(ang))
        s = float(np.sin(ang))
        x_old = sample[..., 0].clone()
        y_old = sample[..., 1].clone()
        sample[..., 0] = c * x_old - s * y_old
        sample[..., 1] = s * x_old + c * y_old

        if noise_std > 0:
            sample = sample + torch.randn_like(sample) * noise_std

        x_aug[b] = sample

    return x_aug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera una grafica resumen de medidas de balanceo activas con comparativa "
            "antes/despues y ejemplos."
        )
    )
    parser.add_argument(
        "--summary-after",
        type=Path,
        default=Path("csv/resumen_balanceo_freihand_hagrid_activo.json"),
        help="Resumen de balanceo activo final.",
    )
    parser.add_argument(
        "--freihand-training-xyz",
        type=Path,
        default=Path("datasets/FreiHAND_pub_v2/training_xyz.json"),
        help="FreiHAND training_xyz usado para reconstruir distribucion original.",
    )
    parser.add_argument(
        "--freihand-canonical-rgb-manifest",
        type=Path,
        default=Path("output/auditoria/freihand_rgb_canonical_manifest.csv"),
        help=(
            "CSV canónico de FreiHAND (sample_id) para reconstruir medidas "
            "sobre el mismo subset usado en balanceo."
        ),
    )
    parser.add_argument(
        "--hagrid-annotations-dir",
        type=Path,
        default=Path("datasets/hagrid_kaggle_raw/ann_subsample"),
        help="Directorio de anotaciones HaGRID usado para reconstruir distribucion original.",
    )
    parser.add_argument(
        "--mst-csv",
        type=Path,
        default=Path("csv/mst_real_dataset_actualizado.csv"),
        help="CSV de MST para asignar/imputar niveles en distribucion original.",
    )
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=[
            "call", "dislike", "fist", "four", "like", "mute", "ok", "one",
            "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
            "three", "three2", "two_up", "two_up_inverted",
        ],
        help="Gestos HaGRID a considerar en la reconstruccion original.",
    )
    parser.add_argument(
        "--impute-missing-mst",
        action="store_true",
        default=True,
        help="Imputa MST faltante para obtener distribucion original completa.",
    )
    parser.add_argument(
        "--no-impute-missing-mst",
        action="store_false",
        dest="impute_missing_mst",
        help="No imputar MST faltante en la distribucion original.",
    )
    parser.add_argument(
        "--manifest-active",
        type=Path,
        default=Path("csv/train_manifest_balanceado_freihand_hagrid_activo.csv"),
        help="Manifest activo para extraer ejemplos de sample_id.",
    )
    parser.add_argument(
        "--seq-dir",
        type=Path,
        default=Path("data/processed/secuencias_stgcn"),
        help="Directorio de secuencias ST-GCN (.npy) para ejemplo de augmentation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("graficos/stgcn_atencion_mst/medidas_balanceo_con_ejemplos.png"),
        help="Ruta de salida de la figura.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_block_counts(summary: dict) -> dict[str, int]:
    return {k: int(v) for k, v in summary["summary"]["by_mst_block"].items()}


def get_level_counts(summary: dict) -> dict[int, int]:
    by_level = summary["summary"]["by_mst_level"]
    return {int(k): int(v) for k, v in by_level.items()}


def compute_original_summary(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)

    freihand_records = DataLoaders.load_freihand_records(
        args.freihand_training_xyz,
        canonical_rgb_manifest_csv=args.freihand_canonical_rgb_manifest,
    )
    hagrid_records = DataLoaders.load_hagrid_records(args.hagrid_annotations_dir, args.gestures)
    mst_map = MstLoader.load_mst_map(args.mst_csv)

    freihand_records = MstLoader.attach_mst(freihand_records, mst_map)
    hagrid_records = MstLoader.attach_mst(hagrid_records, mst_map)

    if args.impute_missing_mst:
        freihand_records = MstImputer.impute_missing_mst(freihand_records, rng)
        hagrid_records = MstImputer.impute_missing_mst(hagrid_records, rng)

    combined = freihand_records + hagrid_records
    return {"summary": _compute_summary(combined)}


def select_example_ids(manifest_path: Path, per_level: int = 2) -> dict[int, list[str]]:
    df = pd.read_csv(manifest_path)
    levels = [1, 2, 3, 10]
    picks: dict[int, list[str]] = {}
    for level in levels:
        subset = df[df["mst"] == level]
        ids = subset["sample_id"].astype(str).drop_duplicates().tolist()
        picks[level] = ids[:per_level]
    return picks


def find_sequence_example(manifest_path: Path, seq_dir: Path) -> tuple[np.ndarray, int, str] | None:
    df = pd.read_csv(manifest_path)
    df = df[df["mst"].isin([1, 2, 3, 10])].copy()
    if df.empty:
        return None

    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        seq_path = seq_dir / f"{sid}.npy"
        if seq_path.exists():
            arr = np.load(seq_path).astype(np.float32)
            if arr.ndim == 3 and arr.shape[1:] == (21, 3):
                return arr, int(row["mst"]), sid
    return None


def draw_landmarks(ax: plt.Axes, frame: np.ndarray, title: str) -> None:
    x = frame[:, 0]
    y = frame[:, 1]
    ax.scatter(x, y, s=18, c="#0f766e", alpha=0.9)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    for i, j in edges:
        ax.plot([x[i], x[j]], [y[i], y[j]], color="#115e59", linewidth=1)

    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(alpha=0.2, linestyle="--")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    before = compute_original_summary(args)
    after = load_json(args.summary_after)

    block_before = get_block_counts(before)
    block_after = get_block_counts(after)
    level_before = get_level_counts(before)
    level_after = get_level_counts(after)

    cfg = after.get("sampling_config", {})
    examples = select_example_ids(args.manifest_active, per_level=2)
    seq_example = find_sequence_example(args.manifest_active, args.seq_dir)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_blocks = fig.add_subplot(gs[0, 0])
    ax_levels = fig.add_subplot(gs[0, 1])
    ax_measures = fig.add_subplot(gs[0, 2])
    ax_original = fig.add_subplot(gs[1, 0])
    ax_aug = fig.add_subplot(gs[1, 1])
    ax_examples = fig.add_subplot(gs[1, 2])

    blocks = ["claro", "medio", "oscuro"]
    x = np.arange(len(blocks))
    w = 0.35
    y_before_blocks = [block_before.get(b, 0) for b in blocks]
    y_after_blocks = [block_after.get(b, 0) for b in blocks]
    ax_blocks.bar(x - w / 2, y_before_blocks, width=w, color="#94a3b8", label="original")
    ax_blocks.bar(x + w / 2, y_after_blocks, width=w, color="#0ea5e9", label="lograda")
    ax_blocks.set_xticks(x, blocks)
    ax_blocks.set_title("Distribucion MST por bloque (original vs lograda)")
    ax_blocks.set_ylabel("muestras")
    ax_blocks.legend()
    ax_blocks.grid(axis="y", alpha=0.2)
    for i, (b0, b1) in enumerate(zip(y_before_blocks, y_after_blocks)):
        ax_blocks.text(i - w / 2, b0 + 40, str(b0), ha="center", va="bottom", fontsize=8)
        ax_blocks.text(i + w / 2, b1 + 40, str(b1), ha="center", va="bottom", fontsize=8)

    levels = list(range(1, 11))
    y_before_levels = [level_before.get(i, 0) for i in levels]
    y_after_levels = [level_after.get(i, 0) for i in levels]
    ax_levels.plot(levels, y_before_levels, marker="o", color="#64748b", label="original")
    ax_levels.plot(levels, y_after_levels, marker="o", color="#0284c7", label="lograda")
    ax_levels.set_xticks(levels)
    ax_levels.set_title("Distribucion original y final por nivel MST")
    ax_levels.set_xlabel("nivel MST")
    ax_levels.set_ylabel("muestras")
    ax_levels.legend()
    ax_levels.grid(alpha=0.2)
    for lvl in [1, 2, 3, 10]:
        idx = lvl - 1
        ax_levels.annotate(
            f"{y_before_levels[idx]}->{y_after_levels[idx]}",
            (lvl, y_after_levels[idx]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    total_before = int(before["summary"].get("total_samples", 0))
    total_after = int(after["summary"].get("total_samples", 0))
    osc_before = int(block_before.get("oscuro", 0))
    osc_after = int(block_after.get("oscuro", 0))
    lvl10_before = int(level_before.get(10, 0))
    lvl10_after = int(level_after.get(10, 0))

    ax_measures.axis("off")
    lines = [
        "MEDIDAS IMPLEMENTADAS (BALANCE ACTIVO)",
        f"extreme_mst_levels: {cfg.get('extreme_mst_levels', [1, 2, 3, 10])}",
        f"extreme_factor: {cfg.get('extreme_factor', 'n/a')}",
        f"dark_jitter_factor: {cfg.get('dark_jitter_factor', 'n/a')}",
        "",
        "Impacto observado:",
        f"total_samples: {total_before} -> {total_after} (delta {total_after - total_before:+d})",
        f"bloque oscuro: {osc_before} -> {osc_after} (delta {osc_after - osc_before:+d})",
        f"MST nivel 10: {lvl10_before} -> {lvl10_after} (delta {lvl10_after - lvl10_before:+d})",
        "",
        "Medidas en entrenamiento:",
        "1) oversampling por sampler ponderado de extremos",
        "2) augmentation geometrica en extremos (ruido/rot/escala)",
        "3) candidatos MST 8-9 para dark jitter en train",
    ]
    ax_measures.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#ecfeff", "edgecolor": "#0891b2"},
    )

    if seq_example is not None:
        arr, mst_level, sid = seq_example
        frame_orig = arr[0]
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        mst_t = torch.tensor([mst_level], dtype=torch.long)
        aug = augment_extreme_batch(
            tensor,
            mst_batch=mst_t,
            extreme_levels={1, 2, 3, 10},
            prob=1.0,
            noise_std=0.006,
            rot_deg=6.0,
            scale_jitter=0.04,
        )[0].numpy()
        frame_aug = aug[0]

        draw_landmarks(ax_original, frame_orig, f"Ejemplo landmarks original (mst={mst_level})")
        draw_landmarks(ax_aug, frame_aug, "Ejemplo tras augmentation extrema")
        ax_aug.text(
            0.01,
            0.01,
            f"sample_id: {sid[:24]}...",
            transform=ax_aug.transAxes,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
        )
    else:
        ax_original.axis("off")
        ax_aug.axis("off")
        ax_original.text(0.5, 0.5, "Sin secuencias para ejemplo", ha="center", va="center")

    ax_examples.axis("off")
    ex_lines = ["EJEMPLOS DE SAMPLE_ID REFORZADOS"]
    for lvl in [1, 2, 3, 10]:
        ids = examples.get(lvl, [])
        if ids:
            ex_lines.append(f"MST {lvl}: {ids[0][:18]}..., {ids[1][:18]}...")
        else:
            ex_lines.append(f"MST {lvl}: sin ejemplos")

    ax_examples.text(
        0.02,
        0.98,
        "\n".join(ex_lines),
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f8fafc", "edgecolor": "#334155"},
    )

    fig.suptitle("MST: distribucion original, medidas aplicadas y distribucion lograda", fontsize=15, weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"[OK] Figura generada: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())