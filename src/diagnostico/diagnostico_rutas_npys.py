from pathlib import Path

# Buscar todos los .npy en el disco dentro del proyecto
repos = [
    Path('C:/Users/usuario/Mateo/balanceo_datasets'),
    Path('C:/Users/usuario/Mateo/balanceo_2/balanceo_datasets'),
]

for repo in repos:
    if repo.exists():
        npys = list(repo.rglob('*.npy'))
        print(f"{repo}: {len(npys)} archivos .npy")
        if npys:
            print(f"  Ejemplo: {npys[0]}")
    else:
        print(f"{repo}: no existe")
