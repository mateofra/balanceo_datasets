from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera una grafica final con resumen de balanceo, entrenamiento, "
            "auditoria y representacion de atencion ST-GCN."
        )
    )
    parser.add_argument(
        "--balance-summary",
        type=Path,
        default=Path("output/resumen_balanceo_freihand_hagrid.json"),
        help="JSON de resumen de balanceo.",
    )
    parser.add_argument(
        "--training-history",
        type=Path,
        default=Path("output/training/training_history_supervisado.json"),
        help="Historial JSON del entrenamiento supervisado.",
    )
    parser.add_argument(
        "--auditoria-csv",
        type=Path,
        default=Path("output/auditoria/auditoria_dpr_resultados.csv"),
        help="CSV de resultados de auditoria DPR.",
    )
    parser.add_argument(
        "--attention-metadata",
        type=Path,
        default=Path("graficos/stgcn_atencion_mst/metadata_atencion_stgcn.json"),
        help="Metadata JSON de visualizacion de atencion ST-GCN.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("graficos/stack_resultados_final.png"),
        help="Ruta de salida para la grafica final.",
    )
    return parser.parse_args()


def _read_balance_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("summary", {})


def _read_training_history(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    return pd.DataFrame(history)


def _read_attention_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("items", [])


def _format_counts(d: dict[str, int]) -> tuple[list[str], list[int]]:
    labels = list(d.keys())
    values = [int(d[k]) for k in labels]
    return labels, values


def main() -> int:
    args = parse_args()

    balance_summary_path = ROOT / args.balance_summary
    training_history_path = ROOT / args.training_history
    auditoria_csv_path = ROOT / args.auditoria_csv
    attention_metadata_path = ROOT / args.attention_metadata
    output_image_path = ROOT / args.output_image

    required_paths = [
        balance_summary_path,
        training_history_path,
        auditoria_csv_path,
        attention_metadata_path,
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan artefactos para la grafica final: " + ", ".join(missing))

    summary = _read_balance_summary(balance_summary_path)
    history_df = _read_training_history(training_history_path)
    audit_df = pd.read_csv(auditoria_csv_path)
    attention_items = _read_attention_metadata(attention_metadata_path)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    # 1) Balanceo
    ax_balance = axes[0, 0]
    source_labels, source_values = _format_counts(summary.get("by_source", {}))
    block_labels, block_values = _format_counts(summary.get("by_mst_block", {}))

    all_labels = [f"src:{l}" for l in source_labels] + [f"mst:{l}" for l in block_labels]
    all_values = source_values + block_values
    colors = ["#4C78A8"] * len(source_values) + ["#F58518"] * len(block_values)

    ax_balance.bar(all_labels, all_values, color=colors)
    ax_balance.set_title("Etapa 1: Balanceo")
    ax_balance.set_ylabel("Muestras")
    ax_balance.tick_params(axis="x", rotation=20)
    total_samples = int(summary.get("total_samples", 0))
    ax_balance.text(
        0.02,
        0.95,
        f"total={total_samples}",
        transform=ax_balance.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    # 2) Entrenamiento
    ax_train = axes[0, 1]
    if not history_df.empty:
        ax_train.plot(history_df["epoch"], history_df["val_acc"], color="#54A24B", label="val_acc")
        ax_train.set_ylim(0.0, 1.0)
        ax_train.set_xlabel("Epoca")
        ax_train.set_ylabel("val_acc")
        ax_train.set_title("Etapa 2: Entrenamiento")
        best_idx = history_df["val_acc"].idxmax()
        best_epoch = int(history_df.loc[best_idx, "epoch"])
        best_acc = float(history_df.loc[best_idx, "val_acc"])
        ax_train.scatter([best_epoch], [best_acc], color="#54A24B", s=45)
        ax_train.text(
            best_epoch,
            best_acc,
            f" best={best_acc:.3f}",
            va="bottom",
            ha="left",
        )

        ax_loss = ax_train.twinx()
        ax_loss.plot(history_df["epoch"], history_df["loss"], color="#E45756", alpha=0.6, label="loss")
        ax_loss.set_ylabel("loss")
    else:
        ax_train.set_title("Etapa 2: Entrenamiento")
        ax_train.text(0.5, 0.5, "Sin historial", ha="center", va="center")

    # 3) Auditoria
    ax_audit = axes[1, 0]
    acc_by_condition = (
        audit_df.groupby("condition", as_index=True)["correcto"].mean().sort_index()
    )
    if not acc_by_condition.empty:
        ax_audit.bar(acc_by_condition.index.tolist(), acc_by_condition.values.tolist(), color="#72B7B2")
        ax_audit.set_ylim(0.0, 1.0)
        ax_audit.set_ylabel("accuracy")
    ax_audit.set_title("Etapa 3: Auditoria DPR")

    min_acc = float(acc_by_condition.min()) if not acc_by_condition.empty else 0.0
    max_acc = float(acc_by_condition.max()) if not acc_by_condition.empty else 0.0
    dpr = (min_acc / max_acc) if max_acc > 0 else 0.0

    dist_by_condition = (
        audit_df.groupby(["condition", "label"]).size().unstack(fill_value=0)
    )
    pairs = [("claro", "medio"), ("claro", "oscuro"), ("medio", "oscuro")]
    tvd_values: list[float] = []
    for a, b in pairs:
        if a in dist_by_condition.index and b in dist_by_condition.index:
            pa = dist_by_condition.loc[a].to_numpy(dtype=float)
            pb = dist_by_condition.loc[b].to_numpy(dtype=float)
            pa = pa / pa.sum() if pa.sum() > 0 else pa
            pb = pb / pb.sum() if pb.sum() > 0 else pb
            tvd_values.append(0.5 * float(abs(pa - pb).sum()))
    max_tvd = max(tvd_values) if tvd_values else 0.0

    ax_audit.text(
        0.02,
        0.95,
        f"DPR={dpr:.3f}\nmax_TVD={max_tvd:.3f}",
        transform=ax_audit.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    # 4) Representacion
    ax_repr = axes[1, 1]
    tones: list[str] = []
    top_nodes: list[int] = []
    top_scores: list[float] = []
    for item in attention_items:
        tone = str(item.get("tone", "na"))
        attn = item.get("attention_normalized", [])
        if not attn:
            continue
        max_idx = int(max(range(len(attn)), key=lambda i: attn[i]))
        tones.append(tone)
        top_nodes.append(max_idx)
        top_scores.append(float(attn[max_idx]))

    if tones:
        bars = ax_repr.bar(tones, top_scores, color="#B279A2")
        ax_repr.set_ylim(0.0, 1.05)
        ax_repr.set_ylabel("max attn normalized")
        for idx, bar in enumerate(bars):
            ax_repr.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"nodo {top_nodes[idx]}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax_repr.text(0.5, 0.5, "Sin metadata de atencion", ha="center", va="center")
    ax_repr.set_title("Etapa 4: Representacion")

    fig.suptitle("Resumen final del stack: balanceo, entrenamiento, auditoria y representacion")
    fig.savefig(output_image_path, dpi=180)
    plt.close(fig)

    print(f"Grafica final guardada en: {output_image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())