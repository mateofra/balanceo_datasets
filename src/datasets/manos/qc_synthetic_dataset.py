from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "manos" / "src" / "qc_synthetic_dataset.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")