from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BlockRecord:
    sample_id: str
    path: str
    dataset: str
    mst_level: int
    mst_block: str
    source_kind: str
    input_csv: str
    extra_data: dict[str, str]


def _mst_block(mst_level: int) -> str:
    if 1 <= mst_level <= 4:
        return "claro"
    if 5 <= mst_level <= 7:
        return "medio"
    if 8 <= mst_level <= 10:
        return "oscuro"
    raise ValueError(f"Nivel MST invalido: {mst_level}")


def _parse_mst_level(row: dict[str, str]) -> int:
    for key in ("mst_level", "mst", "mst_block_level"):
        raw_value = str(row.get(key, "")).strip()
        if raw_value:
            level = int(float(raw_value))
            if 1 <= level <= 10:
                return level
    raise ValueError("La fila no contiene un nivel MST valido")


def _load_records(csv_path: Path) -> list[BlockRecord]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    records: list[BlockRecord] = []
    for row in rows:
        if "sample_id" in row:
            sample_id = str(row.get("sample_id", "")).strip()
        else:
            sample_id = str(row.get("image_id", "")).strip()

        if not sample_id:
            continue

        path = str(row.get("path", row.get("image_path", ""))).strip()
        if not path:
            continue

        dataset = str(row.get("dataset", "unknown")).strip() or "unknown"
        source_kind = str(row.get("source_kind", "real" if dataset in {"freihand", "hagrid"} else "synthetic")).strip()
        input_csv = str(row.get("input_csv", str(csv_path))).strip() or str(csv_path)
        try:
            mst_level = _parse_mst_level(row)
        except ValueError:
            # Si no hay nivel MST valido, se omite la fila del balanceo
            continue

        # Capturar todas las demás columnas
        reserved = {"sample_id", "image_id", "path", "image_path", "dataset", "source_kind", "input_csv", "mst_level", "mst", "mst_block_level", "mst_block"}
        extra_data = {k: str(v).strip() for k, v in row.items() if k not in reserved}

        records.append(
            BlockRecord(
                sample_id=sample_id,
                path=path,
                dataset=dataset,
                mst_level=mst_level,
                mst_block=_mst_block(mst_level),
                source_kind=source_kind,
                input_csv=input_csv,
                extra_data=extra_data,
            )
        )

    return records


def _balance_blocks(records: list[BlockRecord], seed: int) -> tuple[list[BlockRecord], dict[str, int]]:
    grouped: dict[str, list[BlockRecord]] = defaultdict(list)
    for record in records:
        grouped[record.mst_block].append(record)

    active_blocks = {block: rows for block, rows in grouped.items() if rows}
    if not active_blocks:
        return [], {}

    target_block_count = min(len(rows) for rows in active_blocks.values())
    rng = random.Random(seed)
    selected: list[BlockRecord] = []

    for block in ("claro", "medio", "oscuro"):
        pool = grouped.get(block, [])
        if not pool:
            continue
        if len(pool) <= target_block_count:
            chosen = pool[:]
        else:
            chosen = rng.sample(pool, target_block_count)
        selected.extend(chosen)

    rng.shuffle(selected)
    counts = Counter(record.mst_block for record in selected)
    return selected, dict(counts)


def build_balanced_block_manifest(
    input_csvs: list[Path],
    output_manifest: Path,
    output_summary: Path,
    seed: int,
) -> None:
    records: list[BlockRecord] = []
    for csv_path in input_csvs:
        if not csv_path.exists():
            raise FileNotFoundError(f"No existe el CSV de entrada: {csv_path} (CWD: {Path.cwd()}, Abs: {csv_path.resolve()})")
        records.extend(_load_records(csv_path))

    selected, selected_counts = _balance_blocks(records, seed)
    original_counts = Counter(record.mst_block for record in records)

    # Determinar todos los fieldnames (base + extras de todos los registros)
    all_extra_keys = set()
    for r in selected:
        all_extra_keys.update(r.extra_data.keys())
    
    base_fields = ["sample_id", "path", "dataset", "mst_level", "mst_block", "source_kind", "input_csv"]
    fieldnames = base_fields + sorted(list(all_extra_keys))

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in selected:
            row_dict = {
                "sample_id": record.sample_id,
                "path": record.path,
                "dataset": record.dataset,
                "mst_level": record.mst_level,
                "mst_block": record.mst_block,
                "source_kind": record.source_kind,
                "input_csv": record.input_csv,
            }
            row_dict.update(record.extra_data)
            writer.writerow(row_dict)

    summary = {
        "inputs": [str(path.resolve()) for path in input_csvs],
        "seed": seed,
        "target_block_count": min(selected_counts.values()) if selected_counts else 0,
        "original_block_counts": dict(original_counts),
        "selected_block_counts": selected_counts,
        "total_original_rows": len(records),
        "total_selected_rows": len(selected),
        "output_manifest": str(output_manifest.resolve()),
    }

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Balancear un manifest por bloques MST claro/medio/oscuro")
    parser.add_argument(
        "--input-csv",
        action="append",
        type=Path,
        required=True,
        help="CSV de entrada. Se puede repetir.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("output/manifest_balanced_blocks.csv"),
        help="Ruta del manifest balanceado de salida.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("output/auditoria/manifest_balanced_blocks_summary.json"),
        help="Ruta del resumen JSON de salida.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para el muestreo reproducible.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    print(f"[balanced-blocks] Iniciando balanceo (v1.1 - try/except fix)...")
    build_balanced_block_manifest(args.input_csv, args.output_manifest, args.output_summary, args.seed)
    print(f"[balanced-blocks] Manifest guardado en {args.output_manifest}")


if __name__ == "__main__":
    main()