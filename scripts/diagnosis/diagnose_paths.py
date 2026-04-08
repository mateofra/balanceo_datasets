"""Diagnostica y repara rutas de landmarks en el manifiesto ST-GCN."""

import csv
from pathlib import Path
from collections import defaultdict

# Verificar qué rutas espera vs. qué existe
with open('output/train_manifest_stgcn.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f'Total filas: {len(rows)}')

# Contar rutas por patrón
path_patterns = defaultdict(int)
wrong_paths = 0

for row in rows:
    path = Path(row['path_landmarks'])
    if path.exists():
        path_patterns['EXISTE'] += 1
    else:
        path_patterns['FALTA'] += 1
        wrong_paths += 1

print(f'\nEstadísticas de rutas:')
print(f'  EXISTE: {path_patterns["EXISTE"]}')
print(f'  FALTA: {path_patterns["FALTA"]}')

# Mostrar ejemplos de rutas que faltan
print(f'\nEjemplos de rutas que faltan:')
for i, row in enumerate(rows):
    if i >= 5:
        break
    path = Path(row['path_landmarks'])
    exists = 'YES' if path.exists() else 'NO'
    print(f'  [{exists}] {path}')

# Mostrar qué sí existe
print(f'\nEjemplos de archivos que existen en data/processed/landmarks/:')
landmarks_dir = Path('data/processed/landmarks/')
existing_files = list(landmarks_dir.glob('*.npy'))
print(f'  Total archivos .npy: {len(existing_files)}')
for p in existing_files[:5]:
    print(f'    {p}')
