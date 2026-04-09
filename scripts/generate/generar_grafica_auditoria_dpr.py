from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUDIT_DIR = ROOT / "scripts" / "auditoria"
if str(AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIT_DIR))

from auditoria_dpr import calcular_tvd_correcto


def main() -> None:
    resultados_path = Path("output/auditoria/auditoria_dpr_resultados.csv")
    output_dir = Path("graficos/auditoria_dpr")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_res = pd.read_csv(resultados_path)

    acc_por_condicion = df_res.groupby("condition")["correcto"].mean().reindex(["claro", "medio", "oscuro"])
    tvds, tvd_max = calcular_tvd_correcto(df_res)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Auditoria DPR y TVD", fontsize=14, fontweight="bold")

    barras = axes[0].bar(acc_por_condicion.index, acc_por_condicion.values, color=["#6baed6", "#74c476", "#fd8d3c"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Accuracy por bloque MST")
    axes[0].set_ylabel("Accuracy")
    for barra, valor in zip(barras, acc_por_condicion.values):
        axes[0].text(barra.get_x() + barra.get_width() / 2, valor + 0.01, f"{valor:.3f}", ha="center", va="bottom", fontsize=9)

    pares = [item["par"] for item in tvds]
    valores = [item["tvd"] for item in tvds]
    barras = axes[1].bar(pares, valores, color="#9e9ac8")
    axes[1].set_ylim(0, max(0.4, max(valores) * 1.25))
    axes[1].set_title(f"TVD canónico por par (max={tvd_max:.3f})")
    axes[1].set_ylabel("TVD")
    axes[1].tick_params(axis="x", rotation=20)
    for barra, valor in zip(barras, valores):
        axes[1].text(barra.get_x() + barra.get_width() / 2, valor + 0.01, f"{valor:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    output_path = output_dir / "auditoria_dpr_resumen.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafica guardada en: {output_path}")


if __name__ == "__main__":
    main()