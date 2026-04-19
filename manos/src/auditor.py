"""src/auditor.py — Modulo de auditoria estadistica de sesgo.

Este auditor asume que el generador exporta el layout canonico de 21 puntos,
no el antiguo esqueleto MANO de 16 articulaciones.

Por muestra espera:
    - *_landmarks.npy      → landmarks 2-D con forma (21, 2)
    - *_landmarks3d.npy    → landmarks 3-D pareados con forma (21, 3), si existen

Calcula RMSE invariante a escala, DPR y TVD por grupos de tono MST,
y exporta:
    reports/audit_metrics.json    : metricas crudas legibles por maquina
    reports/bias_report.png       : comparativa de boxplots por nivel MST

Uso:
    uv run src/auditor.py
    uv run src/auditor.py --sample-dir data/synthetic_samples --report-dir reports
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")  # renderizado sin interfaz grafica
import matplotlib.pyplot as plt
import numpy as np

# ── Orden de niveles MST (canonico, en sincronizacion con generator.py) ─────
MST_GROUPS = [f"MST_{i}" for i in range(1, 11)]
EXPECTED_LANDMARKS_2D = (21, 2)
EXPECTED_LANDMARKS_3D = (21, 3)

# ── Desvio estandar de ruido simulado por grupo (pixeles, imagen de 128px)
# Los niveles MST mas oscuros reciben mayor ruido para modelar brecha WEIRD.
# Reemplazar por salidas reales del detector para una auditoria autentica.
_BIAS_STD_PX: dict[str, float] = {
    "MST_1":  0.010 * 128,
    "MST_2":  0.012 * 128,
    "MST_3":  0.014 * 128,
    "MST_4":  0.017 * 128,
    "MST_5":  0.020 * 128,
    "MST_6":  0.024 * 128,
    "MST_7":  0.028 * 128,
    "MST_8":  0.034 * 128,
    "MST_9":  0.041 * 128,
    "MST_10": 0.048 * 128,
}

_LEGACY_TYPE_TO_MST = {
    "Type_I": "MST_1",
    "Type_II": "MST_2",
    "Type_III": "MST_4",
    "Type_IV": "MST_6",
    "Type_V": "MST_8",
    "Type_VI": "MST_10",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Primitivas de metricas
# ─────────────────────────────────────────────────────────────────────────────

def bbox_diagonal(points: np.ndarray) -> float:
    """Diagonal de la caja contenedora para un conjunto de puntos 2-D o 3-D."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def rmse_normalized(pred: np.ndarray, gt: np.ndarray, diag: float) -> float:
    """RMSE invariante a escala normalizado por ``diag`` (diagonal de bbox).

    Args:
        pred:  Landmarks predichos, forma (N, 2).
        gt:    Landmarks de referencia, forma (N, 2).
        diag:  Factor de normalizacion (diagonal de bbox de ``gt``).

    Returns:
        RMSE normalizado (adimensional).
    """
    raw = float(np.sqrt(np.mean((pred - gt) ** 2)))
    return raw / max(diag, 1e-8)


def demographic_parity_ratio(errors: dict[str, float]) -> float:
    """DPR = RMSE_min_grupo / RMSE_max_grupo. Objetivo: ≥ 0.80 (1.0 = paridad perfecta)."""
    vals = list(errors.values())
    return float(min(vals) / max(vals))


def total_variation_distance(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    """TVD = 0.5 · Σ |p_i − q_i| sobre bins de histograma normalizados."""
    a = dist_a / max(dist_a.sum(), 1e-8)
    b = dist_b / max(dist_b.sum(), 1e-8)
    return float(0.5 * np.sum(np.abs(a - b)))


def _histogram(errors: list[float], bins: int = 20) -> np.ndarray:
    counts, _ = np.histogram(errors, bins=bins, range=(0.0, 1.0))
    return counts.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
#  Simulacion de sesgo
# ─────────────────────────────────────────────────────────────────────────────

def simulate_predictions(gt_landmarks: np.ndarray, mst_group: str) -> np.ndarray:
    """Agrega ruido gaussiano especifico por grupo a landmarks 2-D de referencia.

    Simula un detector que rinde peor en tonos mas oscuros (sesgo WEIRD).
    Sustituye esta funcion por salidas reales para una auditoria genuina.

    Args:
        gt_landmarks:    Landmarks de referencia, forma (N, 2), coords de pixel.
        mst_group: Clave de MST_GROUPS, por ejemplo ``'MST_10'``.

    Returns:
        Predicciones simuladas con ruido gaussiano aditivo por tipo.
    """
    std = _BIAS_STD_PX.get(mst_group, 0.02 * 128)
    noise = np.random.normal(0.0, std, gt_landmarks.shape)
    return gt_landmarks + noise


# ─────────────────────────────────────────────────────────────────────────────
#  Helper para parseo de nombres de archivo
# ─────────────────────────────────────────────────────────────────────────────

def _parse_mst_group(stem: str) -> str | None:
    """Extrae el grupo MST desde el stem de nombre de archivo.

    Patrones soportados:
    - ``sample_XXXXX_MST_N`` con N en 1..10
    - ``sample_XXXXX_Type_N`` (legacy) mapeado a MST aproximado.
    Devuelve ``None`` si el patron no aparece.
    """
    parts = stem.split("_")
    try:
        idx = parts.index("MST")
        candidate = f"MST_{parts[idx + 1]}"
        if candidate in MST_GROUPS:
            return candidate
    except (ValueError, IndexError):
        pass

    try:
        idx = parts.index("Type")
        legacy = f"Type_{parts[idx + 1]}"
        return _LEGACY_TYPE_TO_MST.get(legacy)
    except (ValueError, IndexError):
        return None


def _validate_landmark_shapes(
    stem: str,
    landmarks_2d: np.ndarray,
    landmarks_3d: np.ndarray | None,
) -> None:
    """Hace cumplir el contrato de 21 landmarks durante la auditoria.

    Los archivos antiguos de 16 articulaciones se rechazan explicitamente para
    evitar mezclar datasets incompatibles de forma silenciosa.
    """
    if landmarks_2d.shape != EXPECTED_LANDMARKS_2D:
        raise ValueError(
            f"{stem}: se esperaban landmarks 2-D con forma {EXPECTED_LANDMARKS_2D}, "
            f"pero se obtuvo {landmarks_2d.shape}. Regenera el dataset con el generador de 21 puntos."
        )

    if landmarks_3d is not None and landmarks_3d.shape != EXPECTED_LANDMARKS_3D:
        raise ValueError(
            f"{stem}: se esperaban landmarks 3-D con forma {EXPECTED_LANDMARKS_3D}, "
            f"pero se obtuvo {landmarks_3d.shape}."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Funcion principal de auditoria
# ─────────────────────────────────────────────────────────────────────────────

def audit(
    sample_dir: str = "data/synthetic_samples",
    report_dir: str = "reports",
) -> dict[str, Any]:
    """Carga landmarks de referencia, calcula metricas de sesgo y escribe reportes.

    Args:
        sample_dir: Carpeta con archivos ``*_landmarks.npy`` del generador.
        report_dir: Carpeta destino para ``audit_metrics.json`` y
                    ``bias_report.png``.

    Returns:
        Diccionario con claves ``DPR``, ``TVD``, ``RMSE``, ``n_samples`` y
        metadatos de formato de 21 landmarks, ademas de resumenes geometricos
        3-D cuando existan archivos ``*_landmarks3d.npy`` pareados.
    """
    os.makedirs(report_dir, exist_ok=True)

    errors_by_group: dict[str, list[float]] = {t: [] for t in MST_GROUPS}
    geom_diag_3d_by_group: dict[str, list[float]] = {t: [] for t in MST_GROUPS}
    paired_3d_counts: dict[str, int] = {t: 0 for t in MST_GROUPS}

    for fname in sorted(os.listdir(sample_dir)):
        if not fname.endswith("_landmarks.npy"):
            continue

        stem = fname[: -len("_landmarks.npy")]
        mst_group = _parse_mst_group(stem)
        if mst_group is None:
            continue

        gt = np.load(os.path.join(sample_dir, fname))  # (21, 2): coordenadas de pixel
        landmarks3d_path = os.path.join(sample_dir, f"{stem}_landmarks3d.npy")
        gt_3d = np.load(landmarks3d_path) if os.path.exists(landmarks3d_path) else None
        _validate_landmark_shapes(stem, gt, gt_3d)

        pred = simulate_predictions(gt, mst_group)

        diag = bbox_diagonal(gt)
        err = rmse_normalized(pred, gt, diag)
        errors_by_group[mst_group].append(err)

        if gt_3d is not None:
            geom_diag_3d_by_group[mst_group].append(bbox_diagonal(gt_3d))
            paired_3d_counts[mst_group] += 1

    active = {t: v for t, v in errors_by_group.items() if v}
    if not active:
        raise RuntimeError(f"No se encontraron archivos de landmarks en '{sample_dir}'.  "
                           "Ejecuta primero generator.py.")

    # ── RMSE medio por grupo ───────────────────────────────────────────────
    mean_rmse: dict[str, float] = {t: float(np.mean(v)) for t, v in active.items()}

    # ── DPR ────────────────────────────────────────────────────────────────
    dpr = demographic_parity_ratio(mean_rmse)

    # ── TVD (maximo entre todas las combinaciones por pares) ───────────────
    types  = list(active.keys())
    tvd_values: list[float] = []
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            hi = _histogram(active[types[i]])
            hj = _histogram(active[types[j]])
            tvd_values.append(total_variation_distance(hi, hj))
    max_tvd = float(max(tvd_values)) if tvd_values else 0.0

    metrics: dict[str, Any] = {
        "landmark_format": {
            "2d_shape": list(EXPECTED_LANDMARKS_2D),
            "3d_shape": list(EXPECTED_LANDMARKS_3D),
        },
        "DPR": round(dpr, 4),
        "TVD": round(max_tvd, 4),
        "RMSE": {t: round(v, 6) for t, v in mean_rmse.items()},
        "n_samples": {t: len(v) for t, v in errors_by_group.items()},
        "paired_3d_samples": paired_3d_counts,
        "mean_3d_bbox_diagonal": {
            t: round(float(np.mean(v)), 6) for t, v in geom_diag_3d_by_group.items() if v
        },
    }

    # ── Reporte JSON ───────────────────────────────────────────────────────
    json_path = os.path.join(report_dir, "audit_metrics.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[audit] Metricas guardadas  → {json_path}")

    # ── Reporte visual (boxplots) ──────────────────────────────────────────
    _write_boxplot(active, mean_rmse, dpr, max_tvd, report_dir)

    # ── Resumen en terminal ────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  DPR  = {dpr:.4f}  (objetivo ≥ 0.80)")
    print(f"  TVD  = {max_tvd:.4f}  (maximo por pares)")
    print("  RMSE (normalizado) por grupo MST:")
    for t in MST_GROUPS:
        if t in mean_rmse:
            n = len(errors_by_group[t])
            print(f"    {t:<10} {mean_rmse[t]:.6f}   (n={n})")
    print(f"{'─'*50}\n")

    return metrics


def _write_boxplot(
    errors_by_group: dict[str, list[float]],
    mean_rmse: dict[str, float],
    dpr: float,
    max_tvd: float,
    report_dir: str,
) -> None:
    """Renderiza y guarda el boxplot de distribucion de sesgo."""
    # Colores representativos MST (RGB → tupla matplotlib 0-1)
    _MST_COLORS = {
        "MST_1": (246 / 255, 237 / 255, 228 / 255),
        "MST_2": (243 / 255, 231 / 255, 219 / 255),
        "MST_3": (247 / 255, 234 / 255, 208 / 255),
        "MST_4": (234 / 255, 218 / 255, 186 / 255),
        "MST_5": (215 / 255, 189 / 255, 150 / 255),
        "MST_6": (160 / 255, 126 / 255,  86 / 255),
        "MST_7": (130 / 255,  92 / 255,  67 / 255),
        "MST_8": ( 96 / 255,  65 / 255,  52 / 255),
        "MST_9": ( 58 / 255,  49 / 255,  42 / 255),
        "MST_10": (41 / 255, 36 / 255, 32 / 255),
    }

    types  = [t for t in MST_GROUPS if t in errors_by_group]
    data   = [errors_by_group[t] for t in types]
    colors = [_MST_COLORS[t] for t in types]

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(data, tick_labels=types, patch_artist=True, notch=False, vert=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    baseline = mean_rmse.get("MST_1", 0.0)
    ax.axhline(
        y=baseline, color="steelblue", linestyle="--", linewidth=1.0,
        label=f"Linea base MST_1 ({baseline:.4f})",
    )

    ax.set_title(
        f"RMSE de landmarks de mano por nivel MST\n"
        f"DPR = {dpr:.3f}  (objetivo ≥ 0.80)  |  TVD maximo por pares = {max_tvd:.3f}",
        fontsize=12,
    )
    ax.set_xlabel("Nivel MST", fontsize=11)
    ax.set_ylabel("RMSE normalizado (unidades de diagonal de bbox)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()

    png_path = os.path.join(report_dir, "bias_report.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[audit] Reporte visual  → {png_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditor de sesgo para landmarks MANO")
    parser.add_argument(
        "--sample-dir", default="data/synthetic_samples",
        help="Carpeta con archivos *_landmarks.npy (por defecto: data/synthetic_samples)",
    )
    parser.add_argument(
        "--report-dir", default="reports",
        help="Carpeta de salida para audit_metrics.json y bias_report.png (por defecto: reports)",
    )
    args = parser.parse_args()
    audit(sample_dir=args.sample_dir, report_dir=args.report_dir)
