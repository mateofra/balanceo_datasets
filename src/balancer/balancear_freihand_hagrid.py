from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SEED = 42
DEFAULT_TARGET_SIZE = 20000
DEFAULT_HAGRID_RATIO = 0.5
DEFAULT_EXTREME_MST_LEVELS = (1, 2, 3, 10)
DEFAULT_EXTREME_FACTOR = 2.0
DEFAULT_JITTER_FACTOR = 0.0
DEFAULT_IMPUTE_MISSING_MST = True


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    source: str
    gesture: str
    mst: int | None = None
    mst_origin: str = "missing"


def _to_repo_relative_posix(path: Path) -> str:
    """Convierte una ruta absoluta/relativa a relativa al repo con separador POSIX."""
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def _load_landmark_mapping(mapping_json: Path | None) -> dict[str, str]:
    """Carga mapeo sample_id -> path_landmarks; retorna vacio si no existe."""
    if mapping_json is None:
        return {}
    if not mapping_json.exists():
        return {}

    with mapping_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    mapping: dict[str, str] = {}
    for raw_key, raw_path in payload.items():
        key = str(raw_key).strip().lower()
        if not key:
            continue
        mapping[key] = str(raw_path).replace("\\", "/")
    return mapping


def _load_quality_mapping(mapping_json: Path | None) -> dict[str, str]:
    """Carga mapeo sample_id -> landmark_quality; retorna vacio si no existe."""
    if mapping_json is None:
        return {}
    if not mapping_json.exists():
        return {}

    with mapping_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    mapping: dict[str, str] = {}
    for raw_key, raw_quality in payload.items():
        key = str(raw_key).strip().lower()
        if not key:
            continue
        mapping[key] = str(raw_quality).strip()
    return mapping


def _load_freihand_records(training_xyz_path: Path) -> list[SampleRecord]:
    with training_xyz_path.open("r", encoding="utf-8") as f:
        xyz = json.load(f)

    records: list[SampleRecord] = []
    for idx, _ in enumerate(xyz):
        records.append(
            SampleRecord(
                sample_id=f"freihand_{idx:08d}",
                source="freihand",
                gesture="unknown",
            )
        )
    return records


def _parse_hagrid_sample_id(raw_id: str) -> str:
    # Homogeniza IDs para facilitar cruces con auditorias externas.
    return raw_id.strip().lower()


def _pick_primary_label(labels: Iterable[str], fallback_gesture: str) -> str:
    for label in labels:
        label_norm = label.strip().lower()
        if label_norm and label_norm != "no_gesture":
            return label_norm
    return fallback_gesture


def _load_hagrid_records(annotations_dir: Path, gestures: list[str]) -> list[SampleRecord]:
    records: list[SampleRecord] = []

    for gesture in gestures:
        annotation_path = annotations_dir / f"{gesture}.json"
        if not annotation_path.exists():
            continue

        try:
            with annotation_path.open("r", encoding="utf-8") as f:
                ann_data = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"Advertencia: se omite {annotation_path} por JSON invalido ({exc}).")
            continue

        for image_id, payload in ann_data.items():
            labels = payload.get("labels", [])
            primary_label = _pick_primary_label(labels, fallback_gesture=gesture)
            sample_id = _parse_hagrid_sample_id(image_id)
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    source="hagrid",
                    gesture=primary_label,
                )
            )

    return records


def _load_mst_map(mst_csv_path: Path) -> dict[str, int]:
    mst_map: dict[str, int] = {}

    with mst_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = {c.lower(): c for c in (reader.fieldnames or [])}

        id_col = None
        for candidate in ("sample_id", "image_id", "id"):
            if candidate in columns:
                id_col = columns[candidate]
                break

        mst_col = None
        for candidate in ("mst", "mst_level", "monk_skin_tone"):
            if candidate in columns:
                mst_col = columns[candidate]
                break

        if id_col is None or mst_col is None:
            raise ValueError(
                "El CSV de MST debe incluir columnas sample_id/image_id/id y mst/mst_level/monk_skin_tone."
            )

        for row in reader:
            raw_id = str(row[id_col]).strip().lower()
            if not raw_id:
                continue
            try:
                mst_value = int(float(row[mst_col]))
            except (TypeError, ValueError):
                continue
            if 1 <= mst_value <= 10:
                mst_map[raw_id] = mst_value

    return mst_map


def _attach_mst(records: list[SampleRecord], mst_map: dict[str, int]) -> list[SampleRecord]:
    with_mst: list[SampleRecord] = []
    for record in records:
        key = record.sample_id.lower()
        mst_value = mst_map.get(key)
        mst_origin = "original" if mst_value is not None else "missing"
        with_mst.append(
            SampleRecord(
                sample_id=record.sample_id,
                source=record.source,
                gesture=record.gesture,
                mst=mst_value,
                mst_origin=mst_origin,
            )
        )
    return with_mst


def _choose_level_for_block(block_name: str, rng: random.Random) -> int:
    if block_name == "claro":
        return rng.choice([1, 2, 3, 4])
    if block_name == "medio":
        return rng.choice([5, 6, 7])
    return rng.choice([8, 9, 10])


def _impute_missing_mst(records: list[SampleRecord], rng: random.Random) -> list[SampleRecord]:
    total = len(records)
    if total == 0:
        return records

    current_blocks = Counter()
    missing_indices: list[int] = []
    imputed: list[SampleRecord] = []

    for idx, record in enumerate(records):
        if record.mst is None:
            missing_indices.append(idx)
            imputed.append(record)
            continue
        if 1 <= record.mst <= 4:
            current_blocks["claro"] += 1
        elif 5 <= record.mst <= 7:
            current_blocks["medio"] += 1
        else:
            current_blocks["oscuro"] += 1
        imputed.append(record)

    if not missing_indices:
        return records

    target_blocks = {
        "claro": total // 3,
        "medio": total // 3,
        "oscuro": total - 2 * (total // 3),
    }

    for idx in missing_indices:
        deficits = {
            block_name: target_blocks[block_name] - current_blocks[block_name]
            for block_name in ("claro", "medio", "oscuro")
        }
        best_block = max(deficits, key=lambda b: deficits[b])
        if deficits[best_block] <= 0:
            best_block = rng.choice(["claro", "medio", "oscuro"])

        mst_value = _choose_level_for_block(best_block, rng)
        current_blocks[best_block] += 1

        original = imputed[idx]
        imputed[idx] = SampleRecord(
            sample_id=original.sample_id,
            source=original.source,
            gesture=original.gesture,
            mst=mst_value,
            mst_origin="imputed",
        )

    return imputed


def _sample_with_replacement(
    pool: list[SampleRecord],
    target_size: int,
    rng: random.Random,
) -> list[SampleRecord]:
    if not pool:
        return []
    if target_size <= len(pool):
        return rng.sample(pool, target_size)
    return [rng.choice(pool) for _ in range(target_size)]


def _mst_sampling_weight(
    record: SampleRecord,
    extreme_mst_levels: set[int],
    extreme_factor: float,
) -> float:
    if record.mst is not None and record.mst in extreme_mst_levels:
        return extreme_factor
    return 1.0


def _sample_with_mst_priority(
    pool: list[SampleRecord],
    target_size: int,
    rng: random.Random,
    extreme_mst_levels: set[int],
    extreme_factor: float,
) -> list[SampleRecord]:
    if not pool:
        return []

    has_mst = any(record.mst is not None for record in pool)
    if not has_mst:
        return _sample_with_replacement(pool, target_size, rng)

    weights = [
        _mst_sampling_weight(record, extreme_mst_levels, extreme_factor)
        for record in pool
    ]
    return rng.choices(pool, weights=weights, k=target_size)


def _group_by_mst_block(records: list[SampleRecord]) -> dict[str, list[SampleRecord]]:
    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        if record.mst is None:
            grouped["sin_mst"].append(record)
        elif 1 <= record.mst <= 4:
            grouped["claro"].append(record)
        elif 5 <= record.mst <= 7:
            grouped["medio"].append(record)
        else:
            grouped["oscuro"].append(record)
    return grouped


def _balance_with_optional_blocks(
    records: list[SampleRecord],
    target_size: int,
    rng: random.Random,
    extreme_mst_levels: set[int],
    extreme_factor: float,
) -> list[SampleRecord]:
    grouped = _group_by_mst_block(records)

    if set(grouped.keys()) <= {"sin_mst"}:
        return _sample_with_replacement(records, target_size, rng)

    # Objetivo por bloques 33/33/33 cuando exista MST.
    block_targets = {
        "claro": target_size // 3,
        "medio": target_size // 3,
        "oscuro": target_size - 2 * (target_size // 3),
    }

    selected: list[SampleRecord] = []
    for block_name in ("claro", "medio", "oscuro"):
        selected.extend(
            _sample_with_mst_priority(
                grouped.get(block_name, []),
                block_targets[block_name],
                rng,
                extreme_mst_levels,
                extreme_factor,
            )
        )

    return selected


def _build_balanced_manifest(
    freihand_records: list[SampleRecord],
    hagrid_records: list[SampleRecord],
    target_size: int,
    hagrid_ratio: float,
    rng: random.Random,
    extreme_mst_levels: set[int],
    extreme_factor: float,
) -> list[SampleRecord]:
    hagrid_target = int(round(target_size * hagrid_ratio))
    freihand_target = target_size - hagrid_target

    selected_hagrid = _balance_with_optional_blocks(
        hagrid_records,
        hagrid_target,
        rng,
        extreme_mst_levels,
        extreme_factor,
    )
    selected_freihand = _balance_with_optional_blocks(
        freihand_records,
        freihand_target,
        rng,
        extreme_mst_levels,
        extreme_factor,
    )

    merged = selected_hagrid + selected_freihand
    rng.shuffle(merged)
    return merged


def _compute_summary(records: list[SampleRecord]) -> dict[str, object]:
    source_counts = Counter(r.source for r in records)
    gesture_counts = Counter(r.gesture for r in records)
    mst_level_counts: Counter[int] = Counter(r.mst for r in records if r.mst is not None)
    mst_origin_counts = Counter(r.mst_origin for r in records)

    block_counts: Counter[str] = Counter()
    for record in records:
        if record.mst is None:
            block_counts["sin_mst"] += 1
        elif 1 <= record.mst <= 4:
            block_counts["claro"] += 1
        elif 5 <= record.mst <= 7:
            block_counts["medio"] += 1
        else:
            block_counts["oscuro"] += 1

    return {
        "total_samples": len(records),
        "by_source": dict(source_counts),
        "by_gesture": dict(gesture_counts),
        "by_mst_block": dict(block_counts),
        "by_mst_level": dict(sorted(mst_level_counts.items())),
        "by_mst_origin": dict(mst_origin_counts),
    }


def _is_dark_aug_candidate(record: SampleRecord) -> bool:
    return record.mst in (8, 9)


def _expand_with_dark_jitter_candidates(
    records: list[SampleRecord],
    jitter_factor: float,
    rng: random.Random,
) -> list[SampleRecord]:
    if jitter_factor <= 0.0:
        return records

    base_candidates = [record for record in records if _is_dark_aug_candidate(record)]
    if not base_candidates:
        return records

    # Replica entradas de candidatos MST 8-9 para representar augmentacion cromatica
    # de entrenamiento (sin tocar validacion/test).
    n_extra = int(round(len(base_candidates) * jitter_factor))
    if n_extra <= 0:
        return records

    extra_records = [rng.choice(base_candidates) for _ in range(n_extra)]
    return records + extra_records


def _write_manifest_csv(
    output_csv: Path,
    records: list[SampleRecord],
    extreme_mst_levels: set[int],
    extreme_factor: float,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "source",
                "gesture",
                "mst",
                "mst_origin",
                "split",
                "sampling_weight",
                "augmentation_hint",
            ],
        )
        writer.writeheader()
        for record in records:
            weight = _mst_sampling_weight(record, extreme_mst_levels, extreme_factor)
            augmentation_hint = ""
            if record.mst in (8, 9):
                augmentation_hint = "color_jitter_dark_candidate"
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "source": record.source,
                    "gesture": record.gesture,
                    "mst": "" if record.mst is None else record.mst,
                    "mst_origin": record.mst_origin,
                    "split": "train",
                    "sampling_weight": weight,
                    "augmentation_hint": augmentation_hint,
                }
            )


def _write_tone_sets(output_dir: Path, records: list[SampleRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tone_sets: dict[str, list[SampleRecord]] = {
        "claro": [r for r in records if r.mst is not None and 1 <= r.mst <= 4],
        "medio": [r for r in records if r.mst is not None and 5 <= r.mst <= 7],
        "oscuro": [r for r in records if r.mst is not None and 8 <= r.mst <= 10],
    }

    for tone_name, tone_records in tone_sets.items():
        out_csv = output_dir / f"train_set_{tone_name}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "source", "gesture", "mst", "split"],
            )
            writer.writeheader()
            for record in tone_records:
                writer.writerow(
                    {
                        "sample_id": record.sample_id,
                        "source": record.source,
                        "gesture": record.gesture,
                        "mst": record.mst,
                        "split": "train",
                    }
                )


def _write_landmark_training_dirs(output_dir: Path, records: list[SampleRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tone_sets: dict[str, list[SampleRecord]] = {
        "claro": [r for r in records if r.mst is not None and 1 <= r.mst <= 4],
        "medio": [r for r in records if r.mst is not None and 5 <= r.mst <= 7],
        "oscuro": [r for r in records if r.mst is not None and 8 <= r.mst <= 10],
    }

    for tone_name, tone_records in tone_sets.items():
        tone_dir = output_dir / tone_name
        tone_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = tone_dir / "train_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "source", "gesture", "mst", "split", "mst_origin"],
            )
            writer.writeheader()
            for record in tone_records:
                writer.writerow(
                    {
                        "sample_id": record.sample_id,
                        "source": record.source,
                        "gesture": record.gesture,
                        "mst": record.mst,
                        "split": "train",
                        "mst_origin": record.mst_origin,
                    }
                )

        stats = {
            "tone": tone_name,
            "total_samples": len(tone_records),
            "by_source": dict(Counter(r.source for r in tone_records)),
            "by_gesture": dict(Counter(r.gesture for r in tone_records)),
            "by_mst_level": dict(sorted(Counter(r.mst for r in tone_records if r.mst is not None).items())),
        }
        stats_path = tone_dir / "stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=True, indent=2)

    readme_lines = [
        "# training_tono",
        "",
        "Directorios de entrenamiento por tono para el modelo de landmarks.",
        "",
        "## Estructura",
        "",
        "- claro/train_manifest.csv y claro/stats.json",
        "- medio/train_manifest.csv y medio/stats.json",
        "- oscuro/train_manifest.csv y oscuro/stats.json",
        "",
        "Cada manifesto contiene solo split=train y muestras del bloque MST correspondiente.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def _extract_freihand_numeric_id(sample_id: str) -> str:
    prefix = "freihand_"
    if sample_id.startswith(prefix):
        return sample_id[len(prefix):]
    return sample_id


def _mst_to_condition(mst: int | None) -> str:
    if mst is None:
        return "sin_mst"
    if 1 <= mst <= 4:
        return "claro"
    if 5 <= mst <= 7:
        return "medio"
    return "oscuro"


def _build_landmark_path(
    record: SampleRecord,
    landmarks_root: Path,
    hagrid_mapping: dict[str, str] | None = None,
    landmarks_index: dict[str, str] | None = None,
) -> str:
    hagrid_mapping = hagrid_mapping or {}
    landmarks_index = landmarks_index or {}

    if record.source == "hagrid":
        sample_id_key = record.sample_id.strip().lower()
        prefixed_key = f"hagrid_{record.gesture}_{sample_id_key}"

        # 1) Prioridad a mapping explícito generado por el pipeline de extracción.
        mapped_path = hagrid_mapping.get(sample_id_key)
        if mapped_path:
            return mapped_path

        # 2) Resolver por índice real de archivos en disco.
        indexed = landmarks_index.get(sample_id_key)
        if indexed:
            return indexed

        indexed_prefixed = landmarks_index.get(prefixed_key)
        if indexed_prefixed:
            return indexed_prefixed

        # 3) Fallback compatible con manifests históricos.
        return str(landmarks_root / "hagrid" / record.gesture / f"{record.sample_id}.npy")

    freihand_id = _extract_freihand_numeric_id(record.sample_id)
    return str(landmarks_root / "freihand" / f"{freihand_id}.npy")


def _infer_landmark_quality(
    record: SampleRecord,
    hagrid_quality_mapping: dict[str, str] | None = None,
) -> str:
    """Etiqueta calidad de landmarks para auditoria y filtrado."""
    if record.source == "freihand":
        return "real_3d_freihand"

    hagrid_quality_mapping = hagrid_quality_mapping or {}
    quality = hagrid_quality_mapping.get(record.sample_id.strip().lower())
    if quality:
        return quality
    return "unknown_hagrid_quality"


def _write_stgcn_manifest_csv(
    output_csv: Path,
    records: list[SampleRecord],
    landmarks_root: Path,
    include_missing_mst: bool,
    hagrid_mapping: dict[str, str] | None = None,
    hagrid_quality_mapping: dict[str, str] | None = None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Indexa landmarks existentes una sola vez para reparar rutas de forma robusta.
    landmarks_index: dict[str, str] = {}
    if landmarks_root.exists():
        for npy_path in landmarks_root.rglob("*.npy"):
            key = npy_path.stem.strip().lower()
            if not key:
                continue
            landmarks_index[key] = _to_repo_relative_posix(npy_path)

    missing_by_source = Counter()
    written_rows = 0

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "path_landmarks",
                "label",
                "condition",
                "dataset",
                "landmark_quality",
                "mst",
                "mst_origin",
                "split",
            ],
        )
        writer.writeheader()

        for record in records:
            condition = _mst_to_condition(record.mst)
            if condition == "sin_mst" and not include_missing_mst:
                continue

            path_landmarks = _build_landmark_path(
                record,
                landmarks_root,
                hagrid_mapping=hagrid_mapping,
                landmarks_index=landmarks_index,
            )

            if not Path(path_landmarks).exists():
                missing_by_source[record.source] += 1

            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "path_landmarks": path_landmarks,
                    "label": record.gesture,
                    "condition": condition,
                    "dataset": record.source,
                    "landmark_quality": _infer_landmark_quality(
                        record,
                        hagrid_quality_mapping=hagrid_quality_mapping,
                    ),
                    "mst": "" if record.mst is None else record.mst,
                    "mst_origin": record.mst_origin,
                    "split": "train",
                }
            )
            written_rows += 1

    if missing_by_source:
        missing_total = sum(missing_by_source.values())
        print(
            "Aviso: el manifiesto ST-GCN incluye rutas de landmarks faltantes "
            f"({missing_total}/{written_rows})."
        )
        print(f"- faltantes por fuente: {dict(missing_by_source)}")


def _compute_mst_match_report(
    freihand_records: list[SampleRecord],
    hagrid_records: list[SampleRecord],
    mst_map: dict[str, int],
) -> dict[str, object]:
    freihand_keys = [record.sample_id.lower() for record in freihand_records]
    hagrid_keys = [record.sample_id.lower() for record in hagrid_records]

    freihand_matches = sum(1 for key in freihand_keys if key in mst_map)
    hagrid_matches = sum(1 for key in hagrid_keys if key in mst_map)
    total_records = len(freihand_keys) + len(hagrid_keys)
    total_matches = freihand_matches + hagrid_matches

    def pct(matches: int, total: int) -> float:
        if total == 0:
            return 0.0
        return round((matches / total) * 100.0, 2)

    return {
        "mst_rows": len(mst_map),
        "freihand": {
            "records": len(freihand_keys),
            "matches": freihand_matches,
            "match_pct": pct(freihand_matches, len(freihand_keys)),
        },
        "hagrid": {
            "records": len(hagrid_keys),
            "matches": hagrid_matches,
            "match_pct": pct(hagrid_matches, len(hagrid_keys)),
        },
        "total": {
            "records": total_records,
            "matches": total_matches,
            "match_pct": pct(total_matches, total_records),
        },
    }


def _print_mst_match_report(report: dict[str, object]) -> None:
    freihand = report["freihand"]
    hagrid = report["hagrid"]
    total = report["total"]

    print("Cobertura de match MST (antes de balancear):")
    print(f"- filas_unicas_mst_csv: {report['mst_rows']}")
    print(
        "- freihand: "
        f"{freihand['matches']}/{freihand['records']} "
        f"({freihand['match_pct']}%)"
    )
    print(
        "- hagrid: "
        f"{hagrid['matches']}/{hagrid['records']} "
        f"({hagrid['match_pct']}%)"
    )
    print(
        "- total: "
        f"{total['matches']}/{total['records']} "
        f"({total['match_pct']}%)"
    )

    if float(total["match_pct"]) < 20.0:
        print(
            "Aviso: cobertura de MST muy baja (<20%). "
            "El CSV podria ser mock o estar mal alineado con sample_id/image_id."
        )


def _write_summary_json(
    output_json: Path,
    summary: dict[str, object],
    seed: int,
    extreme_mst_levels: set[int],
    extreme_factor: float,
    jitter_factor: float,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "sampling_config": {
            "extreme_mst_levels": sorted(extreme_mst_levels),
            "extreme_factor": extreme_factor,
            "dark_jitter_factor": jitter_factor,
        },
        "summary": summary,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye un manifiesto de entrenamiento balanceado FreiHAND + HaGRID "
            "con opcion de cuotas por bloque MST si se entrega CSV de auditoria."
        )
    )
    parser.add_argument(
        "--freihand-training-xyz",
        type=Path,
        default=Path("datasets/training_xyz.json"),
        help="Ruta a datasets/training_xyz.json de FreiHAND.",
    )
    parser.add_argument(
        "--hagrid-annotations-dir",
        type=Path,
        default=Path("datasets/ann_subsample"),
        help="Directorio de anotaciones HaGRID para entrenamiento.",
    )
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"],
        help="Gestos HaGRID a incluir.",
    )
    parser.add_argument(
        "--mst-csv",
        type=Path,
        default=None,
        help=(
            "CSV opcional de auditoria previa con MST por sample_id/image_id. "
            "Si existe, activa balance por bloques claro/medio/oscuro."
        ),
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help="Cantidad total objetivo para el manifiesto de entrenamiento.",
    )
    parser.add_argument(
        "--hagrid-ratio",
        type=float,
        default=DEFAULT_HAGRID_RATIO,
        help="Fraccion objetivo de HaGRID en el entrenamiento (0.0-1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semilla para reproducibilidad del muestreo.",
    )
    parser.add_argument(
        "--extreme-mst-levels",
        nargs="+",
        type=int,
        default=list(DEFAULT_EXTREME_MST_LEVELS),
        help="Niveles MST a sobrerrepresentar durante el muestreo cuando hay MST.",
    )
    parser.add_argument(
        "--extreme-factor",
        type=float,
        default=DEFAULT_EXTREME_FACTOR,
        help="Factor de peso para niveles MST extremos (>= 1.0).",
    )
    parser.add_argument(
        "--dark-jitter-factor",
        type=float,
        default=DEFAULT_JITTER_FACTOR,
        help=(
            "Factor de replicacion virtual para candidatos MST 8-9 en entrenamiento "
            "(0.0 desactiva; por ejemplo 0.5 agrega 50%% de candidatos extra)."
        ),
    )
    parser.add_argument(
        "--output-tone-sets-dir",
        type=Path,
        default=None,
        help=(
            "Directorio opcional para exportar tres sets de entrenamiento por tono "
            "(claro, medio, oscuro). Requiere MST disponible."
        ),
    )
    parser.add_argument(
        "--output-landmark-train-dir",
        type=Path,
        default=None,
        help=(
            "Directorio opcional para exportar datos de entrenamiento del modelo "
            "de landmarks separados en tres subdirectorios por tono "
            "(claro/medio/oscuro). Requiere MST disponible."
        ),
    )
    parser.add_argument(
        "--impute-missing-mst",
        action="store_true",
        default=DEFAULT_IMPUTE_MISSING_MST,
        help="Imputa MST faltante para clasificar todos los objetos del dataset.",
    )
    parser.add_argument(
        "--no-impute-missing-mst",
        action="store_false",
        dest="impute_missing_mst",
        help="Desactiva imputacion de MST faltante.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("csv/train_manifest_balanceado_freihand_hagrid.csv"),
        help="Ruta del manifiesto CSV de salida.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("csv/resumen_balanceo_freihand_hagrid.json"),
        help="Ruta del resumen JSON de salida.",
    )
    parser.add_argument(
        "--output-stgcn-manifest-csv",
        type=Path,
        default=None,
        help=(
            "Ruta opcional para exportar manifiesto ST-GCN (CSV) con columnas "
            "sample_id,path_landmarks,label,condition,dataset,mst,mst_origin,split."
        ),
    )
    parser.add_argument(
        "--landmarks-root-dir",
        type=Path,
        default=Path("data/processed/landmarks"),
        help=(
            "Directorio base de landmarks preprocesados (.npy). "
            "Se usa para construir path_landmarks en manifiesto ST-GCN."
        ),
    )
    parser.add_argument(
        "--hagrid-landmarks-mapping-json",
        type=Path,
        default=Path("csv/hagrid_landmarks_mapping.json"),
        help=(
            "JSON opcional sample_id->path_landmarks generado al procesar HaGRID. "
            "Si existe, se usa para resolver rutas de HaGRID de forma exacta."
        ),
    )
    parser.add_argument(
        "--hagrid-landmarks-quality-json",
        type=Path,
        default=Path("csv/hagrid_landmarks_quality.json"),
        help=(
            "JSON opcional sample_id->landmark_quality para etiquetar calidad de "
            "landmarks HaGRID (por ejemplo annotation_2d_projected/synthetic_gesture_mean)."
        ),
    )
    parser.add_argument(
        "--include-missing-mst-in-stgcn",
        action="store_true",
        default=False,
        help=(
            "Incluye filas sin MST en el manifiesto ST-GCN con condition=sin_mst. "
            "Por defecto se excluyen para mantener claro/medio/oscuro."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.target_size <= 0:
        raise ValueError("--target-size debe ser mayor que 0.")
    if not (0.0 <= args.hagrid_ratio <= 1.0):
        raise ValueError("--hagrid-ratio debe estar entre 0.0 y 1.0.")
    if args.extreme_factor < 1.0:
        raise ValueError("--extreme-factor debe ser >= 1.0.")
    if args.dark_jitter_factor < 0.0:
        raise ValueError("--dark-jitter-factor debe ser >= 0.0.")

    extreme_mst_levels = {int(level) for level in args.extreme_mst_levels if 1 <= int(level) <= 10}
    if not extreme_mst_levels:
        raise ValueError("--extreme-mst-levels debe incluir al menos un valor entre 1 y 10.")

    rng = random.Random(args.seed)

    freihand_records = _load_freihand_records(args.freihand_training_xyz)
    hagrid_records = _load_hagrid_records(args.hagrid_annotations_dir, args.gestures)

    if args.mst_csv is not None:
        if not args.mst_csv.exists():
            raise FileNotFoundError(
                f"No existe --mst-csv en la ruta indicada: {args.mst_csv}"
            )

        mst_map = _load_mst_map(args.mst_csv)
        match_report = _compute_mst_match_report(freihand_records, hagrid_records, mst_map)
        _print_mst_match_report(match_report)
        freihand_records = _attach_mst(freihand_records, mst_map)
        hagrid_records = _attach_mst(hagrid_records, mst_map)

    if args.impute_missing_mst:
        freihand_records = _impute_missing_mst(freihand_records, rng)
        hagrid_records = _impute_missing_mst(hagrid_records, rng)

    balanced_records = _build_balanced_manifest(
        freihand_records=freihand_records,
        hagrid_records=hagrid_records,
        target_size=args.target_size,
        hagrid_ratio=args.hagrid_ratio,
        rng=rng,
        extreme_mst_levels=extreme_mst_levels,
        extreme_factor=args.extreme_factor,
    )
    balanced_records = _expand_with_dark_jitter_candidates(
        balanced_records,
        jitter_factor=args.dark_jitter_factor,
        rng=rng,
    )

    summary = _compute_summary(balanced_records)
    _write_manifest_csv(
        args.output_csv,
        balanced_records,
        extreme_mst_levels=extreme_mst_levels,
        extreme_factor=args.extreme_factor,
    )
    _write_summary_json(
        args.output_summary,
        summary,
        seed=args.seed,
        extreme_mst_levels=extreme_mst_levels,
        extreme_factor=args.extreme_factor,
        jitter_factor=args.dark_jitter_factor,
    )

    if args.output_tone_sets_dir is not None:
        has_mst = any(record.mst is not None for record in balanced_records)
        if not has_mst:
            print("Aviso: no se exportan sets por tono porque no hay MST en el manifiesto.")
        else:
            _write_tone_sets(args.output_tone_sets_dir, balanced_records)
            print(f"Sets por tono: {args.output_tone_sets_dir}")

    if args.output_landmark_train_dir is not None:
        has_mst = any(record.mst is not None for record in balanced_records)
        if not has_mst:
            print("Aviso: no se exporta entrenamiento de landmarks por tono porque no hay MST.")
        else:
            _write_landmark_training_dirs(args.output_landmark_train_dir, balanced_records)
            print(f"Entrenamiento landmarks por tono: {args.output_landmark_train_dir}")

    if args.output_stgcn_manifest_csv is not None:
        hagrid_mapping = _load_landmark_mapping(args.hagrid_landmarks_mapping_json)
        hagrid_quality_mapping = _load_quality_mapping(args.hagrid_landmarks_quality_json)
        _write_stgcn_manifest_csv(
            args.output_stgcn_manifest_csv,
            balanced_records,
            landmarks_root=args.landmarks_root_dir,
            include_missing_mst=args.include_missing_mst_in_stgcn,
            hagrid_mapping=hagrid_mapping,
            hagrid_quality_mapping=hagrid_quality_mapping,
        )
        print(f"Manifiesto ST-GCN: {args.output_stgcn_manifest_csv}")

    print("Balanceo completado.")
    print(f"Manifiesto: {args.output_csv}")
    print(f"Resumen: {args.output_summary}")


if __name__ == "__main__":
    main()
