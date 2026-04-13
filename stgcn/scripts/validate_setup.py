#!/usr/bin/env python
"""Valida que el setup esté correcto antes de training."""

import sys
from pathlib import Path
import importlib

def check_python_version():
    """Verifica versión de Python."""
    if sys.version_info < (3, 13):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detectado")
        print(f"   Requiere: Python 3.13+")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def check_packages():
    """Verifica paquetes instalados."""
    packages = ["torch", "numpy"]
    all_ok = True
    
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"✅ {pkg} ({version})")
        except ImportError:
            print(f"❌ {pkg} no instalado")
            print(f"   Instala: pip install {pkg}")
            all_ok = False
    
    return all_ok

def check_manifest():
    """Verifica que manifiesto CSV sea accesible."""
    # Buscar en ubicaciones comunes
    candidates = [
        Path("data/train_manifest_stgcn_fixed.csv"),
        Path("../output/train_manifest_stgcn_fixed.csv"),
        Path("../../output/train_manifest_stgcn_fixed.csv"),
    ]
    
    for path in candidates:
        if path.exists():
            print(f"✅ Manifiesto encontrado: {path}")
            # Contar filas
            with open(path) as f:
                lines = len(f.readlines())
            print(f"   ({lines-1} muestras)")
            return True
    
    print(f"❌ Manifiesto no encontrado en:")
    for p in candidates:
        print(f"   - {p}")
    return False

def check_landmarks():
    """Verifica que landmarks .npy sea accesibles."""
    candidates = [
        Path("data"),
        Path("../data/processed/landmarks"),
        Path("../../data/processed/landmarks"),
    ]
    
    for base_path in candidates:
        if base_path.exists():
            npy_files = list(base_path.glob("**/*.npy"))
            if npy_files:
                print(f"✅ Landmarks encontrados: {len(npy_files)} archivos")
                print(f"   Ubicación: {base_path}")
                return True
    
    print(f"❌ Landmarks .npy no encontrados en:")
    for p in candidates:
        print(f"   - {p}")
    return False

def main():
    print("\n" + "="*60)
    print("VALIDACIÓN DE SETUP ST-GCN")
    print("="*60 + "\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Packages", check_packages),
        ("Manifiesto CSV", check_manifest),
        ("Landmarks .npy", check_landmarks),
    ]
    
    results = []
    for name, check_fn in checks:
        print(f"\n{name}:")
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("✅ SETUP VALIDADO - Listo para training")
        print("="*60)
        print("\nEjecuta: uv run python scripts/train.py\n")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ Fallos detectados - Revisa arriba")
        print("="*60 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
