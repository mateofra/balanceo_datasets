#!/usr/bin/env python
"""Analiza fairness del modelo por tono MST."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Análisis de Fairness por MST")
    parser.add_argument("log_file", help="Archivo de log de training (JSON)")
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    
    if not log_path.exists():
        print(f"❌ Archivo no encontrado: {log_path}")
        return 1
    
    with open(log_path) as f:
        logs = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"FAIRNESS REPORT")
    print(f"{'='*60}\n")
    
    # Get final epoch
    final = logs[-1]
    
    print(f"Final Metrics (Epoch {final['epoch']}):")
    print(f"  Train Accuracy: {final['train_accuracy']:.1f}%")
    print(f"  Val Accuracy: {final['val_accuracy']:.1f}%")
    print(f"  Val Loss: {final['val_loss']:.4f}")
    
    print(f"\n⚠️ NOTA: Este reporte es básico.")
    print(f"Para análisis completo de fairness por MST,")
    print(f"requiere modelo entrenado + predicciones.")
    print(f"\nVer GUIA_RAPIDA.md para detalles.")
    
    print(f"\n{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
