from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audita el mapeo de FreiHAND training/rgb contra el indice canonico "
            "0..N-1 definido por training_xyz.json y genera listados limpios."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/FreiHAND_pub_v2"),
        help="Raiz de FreiHAND_pub_v2.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("output/auditoria/freihand_rgb_mapping_summary.json"),
        help="JSON de resumen de auditoria.",
    )
    parser.add_argument(
        "--output-canonical-csv",
        type=Path,
        default=Path("output/auditoria/freihand_rgb_canonical_manifest.csv"),
        help="CSV con subset canonico 1:1 con training_xyz.",
    )
    parser.add_argument(
        "--output-extra-csv",
        type=Path,
        default=Path("output/auditoria/freihand_rgb_extra_manifest.csv"),
        help="CSV con imagenes RGB extra fuera del indice canonico.",
    )
    return parser.parse_args()


def _read_json_len(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return len(payload)


def _collect_numeric_jpg_indices(path: Path) -> set[int]:
    indices: set[int] = set()
    for img in path.glob("*.jpg"):
        stem = img.stem
        if stem.isdigit():
            indices.add(int(stem))
    return indices


def main() -> int:
    args = parse_args()

    root = args.dataset_root
    xyz_json = root / "training_xyz.json"
    rgb_dir = root / "training" / "rgb"
    mask_dir = root / "training" / "mask"

    if not xyz_json.exists():
        raise FileNotFoundError(f"No existe {xyz_json}")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"No existe {rgb_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"No existe {mask_dir}")

    n_xyz = _read_json_len(xyz_json)
    expected = set(range(n_xyz))

    rgb_idx = _collect_numeric_jpg_indices(rgb_dir)
    mask_idx = _collect_numeric_jpg_indices(mask_dir)

    canonical_idx = sorted(expected & rgb_idx)
    missing_rgb = sorted(expected - rgb_idx)
    missing_mask = sorted(expected - mask_idx)
    extra_rgb = sorted(rgb_idx - expected)
    extra_mask = sorted(mask_idx - expected)

    canonical_rows: list[dict[str, object]] = []
    for idx in canonical_idx:
        canonical_rows.append(
            {
                "sample_id": f"freihand_{idx:08d}",
                "index": idx,
                "rgb_path": str((rgb_dir / f"{idx:08d}.jpg").as_posix()),
                "mask_path": str((mask_dir / f"{idx:08d}.jpg").as_posix()),
                "has_mask": idx in mask_idx,
            }
        )

    extra_rows: list[dict[str, object]] = []
    for idx in extra_rgb:
        extra_rows.append(
            {
                "index": idx,
                "rgb_path": str((rgb_dir / f"{idx:08d}.jpg").as_posix()),
                "has_matching_mask": idx in mask_idx,
            }
        )

    summary = {
        "dataset_root": str(root.as_posix()),
        "training_xyz_count": n_xyz,
        "rgb_count": len(rgb_idx),
        "mask_count": len(mask_idx),
        "canonical_rgb_count": len(canonical_idx),
        "missing_rgb_count": len(missing_rgb),
        "missing_mask_count": len(missing_mask),
        "extra_rgb_count": len(extra_rgb),
        "extra_mask_count": len(extra_mask),
        "missing_rgb_first20": missing_rgb[:20],
        "missing_mask_first20": missing_mask[:20],
        "extra_rgb_first20": extra_rgb[:20],
        "extra_mask_first20": extra_mask[:20],
        "output_canonical_csv": str(args.output_canonical_csv.as_posix()),
        "output_extra_csv": str(args.output_extra_csv.as_posix()),
    }

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_canonical_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_extra_csv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(canonical_rows).to_csv(args.output_canonical_csv, index=False)
    pd.DataFrame(extra_rows).to_csv(args.output_extra_csv, index=False)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("AUDITORIA_MAPEO_OK")
    print(f"training_xyz_count={n_xyz}")
    print(f"canonical_rgb_count={len(canonical_idx)}")
    print(f"extra_rgb_count={len(extra_rgb)}")
    print(f"missing_rgb_count={len(missing_rgb)}")
    print(f"missing_mask_count={len(missing_mask)}")
    print(f"summary={args.output_summary}")
    print(f"canonical_csv={args.output_canonical_csv}")
    print(f"extra_csv={args.output_extra_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
