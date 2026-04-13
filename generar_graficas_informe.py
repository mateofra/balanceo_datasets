# generar_graficas_informe.py
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'src' / 'stgcn'))

import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'

OUT_DIR = BASE / 'output/graficas_informe'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paleta consistente ────────────────────────────────────────────────────
COLOR_CLARO  = '#F4A261'
COLOR_MEDIO  = '#2A9D8F'
COLOR_OSCURO = '#264653'
COLOR_ACC    = '#E76F51'
COLOR_LOSS   = '#457B9D'

# ════════════════════════════════════════════════════════════════════════
# FASE BALANCEO — Distribución MST antes y después
# ════════════════════════════════════════════════════════════════════════
df = pd.read_csv(BASE / 'output/manifest_unificado_final.csv')
if 'condition' not in df.columns and 'mst_imputed' in df.columns:
    df = df.rename(columns={'mst_imputed': 'condition'})
if 'condition' not in df.columns:
    df['condition'] = 'medio'
if 'mst' not in df.columns:
    df['mst'] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Fase Balanceo: Distribución MST en el Dataset', fontsize=14, fontweight='bold')

# Antes (distribución cruda, usando el manifiesto fuente si existe)
manifest_fuente = BASE / 'data/processed/secuencias_stgcn/manifest_secuencias.csv'
if manifest_fuente.exists():
    fuente = pd.read_csv(manifest_fuente)
    if 'dataset' in fuente.columns:
        fuente = fuente[fuente['dataset'] == 'hagrid']
    if 'sample_id' in fuente.columns:
        fuente = fuente.drop_duplicates('sample_id')
    mst_counts_antes = fuente['mst'].value_counts().sort_index() if 'mst' in fuente.columns else pd.Series(dtype=float)
else:
    mst_counts_antes = pd.Series(dtype=float)

if len(mst_counts_antes) > 0:
    axes[0].bar(mst_counts_antes.index, mst_counts_antes.values, color=COLOR_MEDIO, edgecolor='white')
else:
    axes[0].text(0.5, 0.5, 'Sin datos MST crudos disponibles', ha='center', va='center', transform=axes[0].transAxes)
axes[0].set_title('Distribución MST — Dataset Crudo')
axes[0].set_xlabel('Nivel MST (Monk Skin Tone)')
axes[0].set_ylabel('Número de muestras')
axes[0].axvline(x=4.5, color='red', linestyle='--', alpha=0.5, label='Límite claro/medio')
axes[0].axvline(x=7.5, color='orange', linestyle='--', alpha=0.5, label='Límite medio/oscuro')
axes[0].legend(fontsize=8)

# Después (por bloques)
bloques = df['condition'].value_counts()
colores = [COLOR_CLARO, COLOR_MEDIO, COLOR_OSCURO]
orden   = ['claro', 'medio', 'oscuro']
vals    = [bloques.get(b, 0) for b in orden]
bars    = axes[1].bar(orden, vals, color=colores, edgecolor='white')
axes[1].set_title('Distribución por Bloques MST — Dataset Balanceado')
axes[1].set_xlabel('Bloque MST')
axes[1].set_ylabel('Número de muestras')
for bar, val in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(val), ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / '01_balanceo_mst.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 01_balanceo_mst.png")

# ════════════════════════════════════════════════════════════════════════
# FASE SECUENCIAS — Visualización de secuencia sintética
# ════════════════════════════════════════════════════════════════════════
seq_ejemplo = list((BASE / 'data/processed/secuencias_stgcn').rglob('*.npy'))[0]
seq         = np.load(str(seq_ejemplo))  # (16, 21, 3)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Fase Secuencias Sintéticas: Estructura Temporal (T=16 frames)', 
             fontsize=13, fontweight='bold')

# Trayectoria de 3 nodos clave a lo largo del tiempo
nodos_clave = {'Muñeca (0)': 0, 'Base índice (5)': 5, 'Punta índice (8)': 8}
colores_nodos = ['#E63946', '#2A9D8F', '#F4A261']
for ax_idx, (nombre, nodo_idx) in enumerate(nodos_clave.items()):
    for c, coord, label in zip(colores_nodos, range(3), ['X', 'Y', 'Z']):
        axes[ax_idx].plot(seq[:, nodo_idx, coord], color=colores_nodos[coord],
                          label=f'{label}', linewidth=1.5)
    axes[ax_idx].set_title(f'Nodo: {nombre}')
    axes[ax_idx].set_xlabel('Frame')
    axes[ax_idx].set_ylabel('Coordenada normalizada')
    axes[ax_idx].legend(fontsize=8)
    axes[ax_idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / '02_secuencias_sinteticas.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 02_secuencias_sinteticas.png")

# ════════════════════════════════════════════════════════════════════════
# FASE STGCN — Arquitectura del grafo anatómico
# ════════════════════════════════════════════════════════════════════════
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Posiciones 2D aproximadas de los 21 landmarks para visualización
LANDMARK_POS = {
    0:  (0.5,  0.0),   # muñeca
    1:  (0.2,  0.15),  2: (0.15, 0.3),  3: (0.1, 0.45),  4: (0.05, 0.6),
    5:  (0.35, 0.2),   6: (0.32, 0.4),  7: (0.3, 0.55),  8: (0.28, 0.7),
    9:  (0.5,  0.22),  10:(0.5,  0.42), 11:(0.5, 0.58),  12:(0.5,  0.72),
    13: (0.65, 0.2),   14:(0.68, 0.4),  15:(0.7, 0.55),  16:(0.72, 0.7),
    17: (0.8,  0.18),  18:(0.85, 0.35), 19:(0.88,0.5),   20:(0.9,  0.62),
}

fig, ax = plt.subplots(figsize=(6, 8))
fig.suptitle('Fase ST-GCN: Grafo Anatómico (21 nodos, 24 aristas)', 
             fontsize=13, fontweight='bold')

for i, j in HAND_EDGES:
    x = [LANDMARK_POS[i][0], LANDMARK_POS[j][0]]
    y = [LANDMARK_POS[i][1], LANDMARK_POS[j][1]]
    ax.plot(x, y, 'k-', linewidth=1.5, alpha=0.4, zorder=1)

dedos_colores = {
    'Pulgar':   ([0,1,2,3,4],    '#E63946'),
    'Índice':   ([5,6,7,8],      '#2A9D8F'),
    'Corazón':  ([9,10,11,12],   '#F4A261'),
    'Anular':   ([13,14,15,16],  '#A8DADC'),
    'Meñique':  ([17,18,19,20],  '#6D6875'),
    'Muñeca':   ([0],            '#264653'),
}

for nombre, (nodos, color) in dedos_colores.items():
    xs = [LANDMARK_POS[n][0] for n in nodos]
    ys = [LANDMARK_POS[n][1] for n in nodos]
    ax.scatter(xs, ys, c=color, s=100, zorder=3, label=nombre)
    for n in nodos:
        ax.annotate(str(n), LANDMARK_POS[n], textcoords='offset points',
                    xytext=(4, 4), fontsize=7, color='#333333')

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.1, 0.85)
ax.set_aspect('equal')
ax.axis('off')
ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / '03_grafo_anatomico.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 03_grafo_anatomico.png")

# ════════════════════════════════════════════════════════════════════════
# FASE ENTRENAMIENTO — Curvas de loss y val_acc
# ════════════════════════════════════════════════════════════════════════
with open(BASE / 'output/training_history_canonico.json') as f:
    hist = json.load(f)

epochs   = [h['epoch']   for h in hist['history']]
losses   = [h['loss']    for h in hist['history']]
val_accs = [h['val_acc'] for h in hist['history']]
best_ep  = val_accs.index(max(val_accs)) + 1
baseline_random = 1 / max(1, len(hist.get('class_to_idx', {})))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Fase ST-GCN: Curvas de Entrenamiento (50 épocas)', 
             fontsize=13, fontweight='bold')

ax1.plot(epochs, losses, color=COLOR_LOSS, linewidth=2)
ax1.set_title('Loss de Entrenamiento')
ax1.set_xlabel('Época')
ax1.set_ylabel('CrossEntropy Loss')
ax1.grid(alpha=0.3)

ax2.plot(epochs, val_accs, color=COLOR_ACC, linewidth=2)
ax2.axhline(y=baseline_random, color='gray', linestyle='--', alpha=0.7,
            label=f'Baseline azar ({baseline_random:.3f})')
ax2.axhline(y=hist['test_acc'], color='green', linestyle='--', 
            alpha=0.7, label=f"Test holdout ({hist['test_acc']:.3f})")
ax2.axvline(x=best_ep, color='red', linestyle=':', alpha=0.5,
            label=f'Mejor época ({best_ep})')
ax2.set_title('Accuracy de Validación')
ax2.set_xlabel('Época')
ax2.set_ylabel('Accuracy')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / '04_curvas_entrenamiento.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 04_curvas_entrenamiento.png")

# ════════════════════════════════════════════════════════════════════════
# FASE AUDITORÍA — DPR y TVD
# ════════════════════════════════════════════════════════════════════════
df_audit = pd.read_csv(BASE / 'output/auditoria_final_test.csv')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Fase Auditoria: Equidad Algoritmica por Bloque MST', 
             fontsize=13, fontweight='bold')

# Accuracy por bloque
acc_bloque = df_audit.groupby('condition')['correcto'].mean()
orden      = ['claro', 'medio', 'oscuro']
colores    = [COLOR_CLARO, COLOR_MEDIO, COLOR_OSCURO]
vals       = [acc_bloque.get(b, 0) for b in orden]
bars       = axes[0].bar(orden, vals, color=colores, edgecolor='white')
axes[0].axhline(y=0.8 * max(vals), color='red', linestyle='--', 
                alpha=0.7, label='Umbral DPR=0.8')
axes[0].set_title('Accuracy por Bloque MST')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, min(1.0, max(vals) + 0.08))
axes[0].axhline(y=min(vals) / max(vals) * max(vals),
                color='gray', linestyle=':', alpha=0.4)
axes[0].legend(fontsize=8)
for bar, val in zip(bars, vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

# DPR visual
dpr = (min(vals) / max(vals)) if max(vals) > 0 else 0.0
axes[1].barh(['DPR'], [dpr], color='#E76F51' if dpr < 0.8 else '#2A9D8F', height=0.4)
axes[1].axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='Umbral (0.8)')
axes[1].axvline(x=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Paridad perfecta')
axes[1].set_xlim(0, 1.1)
axes[1].set_title(f'DPR = {dpr:.3f}')
axes[1].legend(fontsize=8)
axes[1].text(dpr + 0.02, 0, f'{dpr:.3f}', va='center', fontsize=12, fontweight='bold',
             color='#E76F51' if dpr < 0.8 else '#2A9D8F')

# TVD por par (calculado desde errores reales)
all_labels = sorted(df_audit['label'].unique())
error_distributions = {}
for block in ['claro', 'medio', 'oscuro']:
    sub = df_audit[df_audit['condition'] == block]
    errors = sub[sub['correcto'] == 0]['label'].value_counts(normalize=True)
    error_distributions[block] = errors.reindex(all_labels, fill_value=0.0)

pair_defs = [('claro', 'medio'), ('claro', 'oscuro'), ('medio', 'oscuro')]
pares = [f'{a}\nvs\n{b}' for a, b in pair_defs]
tvds = []
for a, b in pair_defs:
    p = error_distributions[a]
    q = error_distributions[b]
    tvds.append(float(0.5 * (p - q).abs().sum()))

colores_tvd = ['#E9C46A', '#F4A261', '#E76F51']
bars_tvd = axes[2].bar(pares, tvds, color=colores_tvd, edgecolor='white')
axes[2].axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Ideal (<0.2)')
axes[2].axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='Aceptable (<0.4)')
axes[2].set_title('TVD Canónico entre Pares MST')
axes[2].set_ylabel('Total Variation Distance')
axes[2].legend(fontsize=8)
for bar, val in zip(bars_tvd, tvds):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / '05_auditoria_dpr_tvd.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 05_auditoria_dpr_tvd.png")

# ════════════════════════════════════════════════════════════════════════
# FASE AUDITORÍA — Accuracy por clase y bloque MST
# ════════════════════════════════════════════════════════════════════════
pivot = df_audit.groupby(['label', 'condition'])['correcto'].mean().unstack(fill_value=0)
pivot = pivot.reindex(columns=['claro', 'medio', 'oscuro'])
pivot = pivot.sort_values('claro', ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle('Auditoría: Accuracy por Clase y Bloque MST', 
             fontsize=13, fontweight='bold')

x     = np.arange(len(pivot))
width = 0.28
ax.barh(x - width, pivot['claro'],  width, label='Claro',  color=COLOR_CLARO)
ax.barh(x,         pivot['medio'],  width, label='Medio',  color=COLOR_MEDIO)
ax.barh(x + width, pivot['oscuro'], width, label='Oscuro', color=COLOR_OSCURO)
ax.set_yticks(x)
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_xlabel('Accuracy')
ax.set_title('Disparidad por clase entre bloques MST')
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / '06_accuracy_por_clase_mst.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 06_accuracy_por_clase_mst.png")

print(f"\nTodas las gráficas guardadas en: {OUT_DIR}")
