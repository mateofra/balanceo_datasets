# limpiar_repo.py
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parents[2]

print("=== LIMPIEZA DEL REPO ===\n")

# 1. Eliminar secuencias huérfanas (no referenciadas en manifest_canonico)
canonico = pd.read_csv(REPO_ROOT / 'output/manifest_canonico.csv')
paths_validos = set(
    str(REPO_ROOT / p.replace('\\', '/'))
    for p in canonico['path_secuencia']
)

seq_dir = REPO_ROOT / 'data/processed/secuencias_stgcn'
todas_seq = list(seq_dir.rglob('*.npy'))
huerfanas = [p for p in todas_seq if str(p) not in paths_validos]

print(f"Secuencias totales en disco: {len(todas_seq)}")
print(f"Referenciadas en manifest_canonico: {len(paths_validos)}")
print(f"Huérfanas a eliminar: {len(huerfanas)}")

for p in huerfanas:
    p.unlink()
print("Secuencias huérfanas eliminadas.")

# 2. Eliminar manifiestos obsoletos — conservar solo los útiles
DRY_RUN = True
manifiestos_conservar = {
    'data/processed/secuencias_stgcn/manifest_secuencias.csv',  # fuente original
    'output/manifest_canonico.csv',                              # canónico activo
}

todos_csv = list(REPO_ROOT.rglob('*.csv'))
eliminados = 0
for csv in todos_csv:
    rel = str(csv.relative_to(REPO_ROOT)).replace('\\', '/')
    if rel not in manifiestos_conservar:
        print(f"  {'[DRY-RUN] ' if DRY_RUN else ''}Eliminando CSV: {rel}")
        if not DRY_RUN:
            csv.unlink()
        eliminados += 1
print(f"\nCSVs obsoletos eliminados: {eliminados}")

# 3. Eliminar scripts temporales de diagnóstico
scripts_tmp = [
    'diagnosticar_manifest_real.py',
    'diagnosticar_candidatos.py',
    'diagnosticar_secuencias_real.py',
    'diagnosticar_path_secuencia.py',
    'diagnostico_rutas_npys.py',
    'buscar_manifest_hagrid.py',
    'preparar_canonico_final_tmp.py',
    'reentrenar_canonico_3ep.py',
    'fix_manifest_canonico.py',
]

for nombre in scripts_tmp:
    p = REPO_ROOT / nombre
    if p.exists():
        p.unlink()
        print(f"  Eliminado script tmp: {nombre}")

print("\n=== ESTADO FINAL ===")
seq_restantes = list(seq_dir.rglob('*.npy'))
print(f"Secuencias en disco: {len(seq_restantes)}")
print(f"Manifiestos activos: manifest_secuencias.csv + manifest_canonico.csv")
print("Repo limpio.")
