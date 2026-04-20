from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


BLOCK_TO_LEVELS = {
    "claro": [1, 2, 3, 4],
    "medio": [5, 6, 7],
    "oscuro": [8, 9, 10],
}


def _mst_block(mst_level: int) -> str:
    if 1 <= mst_level <= 4:
        return "claro"
    if 5 <= mst_level <= 7:
        return "medio"
    if 8 <= mst_level <= 10:
        return "oscuro"
    raise ValueError(f"Nivel MST invalido: {mst_level}")


def _parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


@dataclass
class RealSample:
    sample_id: str
    image_path: str
    dataset: str
    mst_level: int
    mst_block: str


def _load_real_samples(csv_paths: list[Path]) -> list[RealSample]:
    rows: list[RealSample] = []
    for csv_path in csv_paths:
        if not csv_path.is_file():
            raise FileNotFoundError(f"No existe real-csv: {csv_path}")

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                status = str(row.get("status", "ok")).strip().lower()
                if status and status != "ok":
                    continue

                mst_level = _parse_int(str(row.get("mst_level", "")))
                if mst_level is None:
                    continue

                image_path = str(row.get("image_path", "")).strip()
                if not image_path:
                    continue

                rows.append(
                    RealSample(
                        sample_id=str(row.get("sample_id", "")).strip() or "unknown",
                        image_path=image_path,
                        dataset=str(row.get("dataset", "real")).strip() or "real",
                        mst_level=mst_level,
                        mst_block=_mst_block(mst_level),
                    )
                )
    return rows


def _load_existing_accepted(path: Path | None) -> Counter:
    counts: Counter = Counter()
    if path is None or not path.is_file():
        return counts

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mst_level = _parse_int(str(row.get("mst_level", "")))
            if mst_level is None:
                continue
            counts[_mst_block(mst_level)] += 1
    return counts


def _target_counts(args: argparse.Namespace, current: Counter) -> dict[str, int]:
    explicit = {
        "claro": args.target_block_count_claro,
        "medio": args.target_block_count_medio,
        "oscuro": args.target_block_count_oscuro,
    }

    if any(value > 0 for value in explicit.values()):
        return {block: max(0, explicit[block]) for block in BLOCK_TO_LEVELS}

    baseline = max([current.get(block, 0) for block in BLOCK_TO_LEVELS] + [0])
    return {block: baseline for block in BLOCK_TO_LEVELS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye solicitudes sinteticas por deficit de bloques MST"
    )
    parser.add_argument(
        "--real-csv",
        action="append",
        required=True,
        help="CSV real de entrada. Puede repetirse varias veces.",
    )
    parser.add_argument(
        "--existing-accepted-csv",
        default="",
        help="CSV opcional de sinteticos aceptados previamente.",
    )
    parser.add_argument("--target-block-count-claro", type=int, default=0)
    parser.add_argument("--target-block-count-medio", type=int, default=0)
    parser.add_argument("--target-block-count-oscuro", type=int, default=0)
    parser.add_argument(
        "--expected-qc-acceptance-rate",
        type=float,
        default=0.5,
        help="Tasa esperada de aceptacion de QC para inflar solicitudes.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="CSV de solicitudes sinteticas de salida.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        required=True,
        help="Resumen JSON de balanceo y solicitudes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    real_paths = [Path(p) for p in args.real_csv]
    real_samples = _load_real_samples(real_paths)
    if not real_samples:
        raise RuntimeError("No hay muestras reales validas para construir solicitudes sinteticas")

    acceptance = max(0.01, min(1.0, float(args.expected_qc_acceptance_rate)))
    existing_counts = _load_existing_accepted(Path(args.existing_accepted_csv) if args.existing_accepted_csv else None)

    real_counts = Counter(sample.mst_block for sample in real_samples)
    current_counts = Counter(real_counts)
    current_counts.update(existing_counts)
    target_counts = _target_counts(args, current_counts)

    deficits = {
        block: max(0, target_counts[block] - current_counts.get(block, 0))
        for block in BLOCK_TO_LEVELS
    }
    requested_counts = {
        block: (int(math.ceil(deficits[block] / acceptance)) if deficits[block] > 0 else 0)
        for block in BLOCK_TO_LEVELS
    }

    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    request_index = 0
    for block in ("claro", "medio", "oscuro"):
        request_count = requested_counts[block]
        if request_count <= 0:
            continue

        pool = [sample for sample in real_samples if sample.mst_block == block]
        if not pool:
            pool = list(real_samples)

        levels = BLOCK_TO_LEVELS[block]
        for i in range(request_count):
            source = pool[rng.randrange(len(pool))]
            target_level = levels[i % len(levels)]
            request_index += 1
            rows.append(
                {
                    "request_id": f"req_{request_index:07d}",
                    "source_sample_id": source.sample_id,
                    "source_image_path": source.image_path,
                    "source_dataset": source.dataset,
                    "target_mst_level": target_level,
                    "mst_block": block,
                    "deficit_block_count": deficits[block],
                    "expected_qc_acceptance_rate": acceptance,
                }
            )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "request_id",
                "source_sample_id",
                "source_image_path",
                "source_dataset",
                "target_mst_level",
                "mst_block",
                "deficit_block_count",
                "expected_qc_acceptance_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "inputs": {
            "real_csv": [str(p) for p in real_paths],
            "existing_accepted_csv": args.existing_accepted_csv or None,
        },
        "acceptance_rate": acceptance,
        "real_counts": {block: int(real_counts.get(block, 0)) for block in BLOCK_TO_LEVELS},
        "existing_accepted_counts": {block: int(existing_counts.get(block, 0)) for block in BLOCK_TO_LEVELS},
        "current_counts": {block: int(current_counts.get(block, 0)) for block in BLOCK_TO_LEVELS},
        "target_counts": {block: int(target_counts.get(block, 0)) for block in BLOCK_TO_LEVELS},
        "deficits": {block: int(deficits.get(block, 0)) for block in BLOCK_TO_LEVELS},
        "requested_counts": {block: int(requested_counts.get(block, 0)) for block in BLOCK_TO_LEVELS},
        "total_requests": len(rows),
    }

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[synthetic-request] Manifest guardado en {args.output_manifest}")
    print(f"[synthetic-request] Resumen guardado en {args.output_summary}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())