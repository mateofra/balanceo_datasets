from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def _read_manifest(manifest_csv: Path) -> list[dict[str, str]]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_summary(summary_json: Path | None) -> dict[str, object] | None:
    if summary_json is None:
        return None
    with summary_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_bar(counter: Counter[str], title: str, xlabel: str, out_path: Path) -> None:
    labels = list(counter.keys())
    values = [counter[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    ax.tick_params(axis="x", rotation=30)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_pie(counter: Counter[str], title: str, out_path: Path) -> None:
    labels = list(counter.keys())
    values = [counter[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _compute_block_from_mst(mst_value: str) -> str:
    if mst_value == "":
        return "sin_mst"
    mst_int = int(float(mst_value))
    if 1 <= mst_int <= 4:
        return "claro"
    if 5 <= mst_int <= 7:
        return "medio"
    return "oscuro"


def generate_plots(
    manifest_rows: list[dict[str, str]],
    output_dir: Path,
) -> dict[str, Counter[str]]:
    source_counter: Counter[str] = Counter()
    gesture_counter: Counter[str] = Counter()
    block_counter: Counter[str] = Counter()
    mst_counter: Counter[str] = Counter()

    for row in manifest_rows:
        source = row.get("source", "unknown") or "unknown"
        gesture = row.get("gesture", "unknown") or "unknown"
        mst = row.get("mst", "")

        source_counter[source] += 1
        gesture_counter[gesture] += 1
        block_counter[_compute_block_from_mst(mst)] += 1
        if mst != "":
            mst_counter[mst] += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    _save_bar(
        source_counter,
        title="Composicion por fuente (FreiHAND vs HaGRID)",
        xlabel="Fuente",
        out_path=output_dir / "01_fuentes_barras.png",
    )
    _save_pie(
        source_counter,
        title="Proporcion por fuente",
        out_path=output_dir / "02_fuentes_pie.png",
    )
    _save_bar(
        block_counter,
        title="Composicion por bloque MST",
        xlabel="Bloque MST",
        out_path=output_dir / "03_bloques_mst_barras.png",
    )

    if mst_counter:
        mst_sorted = Counter(dict(sorted(mst_counter.items(), key=lambda item: int(item[0]))))
        _save_bar(
            mst_sorted,
            title="Distribucion por nivel MST",
            xlabel="Nivel MST",
            out_path=output_dir / "04_nivel_mst_barras.png",
        )

    # Limita a top-12 para evitar grafico ilegible con muchos gestos.
    top_gestures = Counter(dict(gesture_counter.most_common(12)))
    _save_bar(
        top_gestures,
        title="Top gestos en entrenamiento",
        xlabel="Gesto",
        out_path=output_dir / "05_gestos_top_barras.png",
    )

    return {
        "source": source_counter,
        "gesture": gesture_counter,
        "block": block_counter,
        "mst": mst_counter,
    }


def write_report(
    output_dir: Path,
    counters: dict[str, Counter[str]],
    summary_payload: dict[str, object] | None,
) -> None:
    report_path = output_dir / "reporte_graficos_balanceo.md"

    lines = [
        "# Reporte de graficos de balanceo",
        "",
        "## Conteos calculados desde manifiesto",
        "",
        f"- Por fuente: {dict(counters['source'])}",
        f"- Por bloque MST: {dict(counters['block'])}",
        f"- Top gestos: {dict(counters['gesture'].most_common(12))}",
    ]

    if counters["mst"]:
        lines.append(f"- Por nivel MST: {dict(sorted(counters['mst'].items(), key=lambda item: int(item[0])))}")
    else:
        lines.append("- Por nivel MST: sin datos MST en manifiesto")

    if summary_payload is not None:
        lines.extend(
            [
                "",
                "## Resumen JSON asociado",
                "",
                "```json",
                json.dumps(summary_payload, ensure_ascii=True, indent=2),
                "```",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera graficos para validar que el balanceo quede alineado con el objetivo."
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        required=True,
        help="Ruta del manifiesto CSV balanceado.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Ruta opcional al resumen JSON de balanceo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("graficos/balanceo"),
        help="Carpeta de salida para PNG y reporte.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = _read_manifest(args.manifest_csv)
    summary_payload = _read_summary(args.summary_json)

    counters = generate_plots(rows, args.output_dir)
    write_report(args.output_dir, counters, summary_payload)

    print("Graficos generados correctamente.")
    print(f"Salida: {args.output_dir}")


if __name__ == "__main__":
    main()
