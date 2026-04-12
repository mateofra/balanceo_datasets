from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from src.balancer.core import SampleRecord


class ManifestWriters:
    @staticmethod
    def _to_repo_relative_posix(path: Path) -> str:
        try:
            rel = path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            rel = path
        return str(rel).replace("\\", "/")

    @staticmethod
    def _extract_freihand_numeric_id(sample_id: str) -> str:
        prefix = "freihand_"
        if sample_id.startswith(prefix):
            return sample_id[len(prefix):]
        return sample_id

    @staticmethod
    def _mst_to_condition(mst: int | None) -> str:
        if mst is None:
            return "sin_mst"
        if 1 <= mst <= 4:
            return "claro"
        if 5 <= mst <= 7:
            return "medio"
        return "oscuro"

    @classmethod
    def _build_landmark_path(
        cls,
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

            mapped_path = hagrid_mapping.get(sample_id_key)
            if mapped_path:
                return mapped_path

            indexed = landmarks_index.get(sample_id_key)
            if indexed:
                return indexed

            indexed_prefixed = landmarks_index.get(prefixed_key)
            if indexed_prefixed:
                return indexed_prefixed

            return str(landmarks_root / "hagrid" / record.gesture / f"{record.sample_id}.npy")

        freihand_id = cls._extract_freihand_numeric_id(record.sample_id)
        return str(landmarks_root / "freihand" / f"{freihand_id}.npy")

    @staticmethod
    def _infer_landmark_quality(
        record: SampleRecord,
        hagrid_quality_mapping: dict[str, str] | None = None,
    ) -> str:
        if record.source == "freihand":
            return "real_3d_freihand"

        hagrid_quality_mapping = hagrid_quality_mapping or {}
        quality = hagrid_quality_mapping.get(record.sample_id.strip().lower())
        if quality:
            return quality
        return "unknown_hagrid_quality"

    @staticmethod
    def _mst_sampling_weight(record: SampleRecord, extreme_mst_levels: set[int], extreme_factor: float) -> float:
        if record.mst is not None and record.mst in extreme_mst_levels:
            return extreme_factor
        return 1.0

    @classmethod
    def write_manifest_csv(
        cls,
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
                    "sample_id", "source", "gesture", "mst", "mst_origin",
                    "split", "sampling_weight", "augmentation_hint",
                ],
            )
            writer.writeheader()
            for record in records:
                weight = cls._mst_sampling_weight(record, extreme_mst_levels, extreme_factor)
                augmentation_hint = "color_jitter_dark_candidate" if record.mst in (8, 9) else ""
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

    @classmethod
    def write_stgcn_manifest_csv(
        cls,
        output_csv: Path,
        records: list[SampleRecord],
        landmarks_root: Path,
        include_missing_mst: bool,
        hagrid_mapping: dict[str, str] | None = None,
        hagrid_quality_mapping: dict[str, str] | None = None,
    ) -> None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        landmarks_index: dict[str, str] = {}
        if landmarks_root.exists():
            for npy_path in landmarks_root.rglob("*.npy"):
                key = npy_path.stem.strip().lower()
                if not key:
                    continue
                landmarks_index[key] = cls._to_repo_relative_posix(npy_path)

        missing_by_source: Counter[str] = Counter()
        written_rows = 0

        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id", "path_landmarks", "label", "condition",
                    "dataset", "landmark_quality", "mst", "mst_origin", "split",
                ],
            )
            writer.writeheader()

            for record in records:
                condition = cls._mst_to_condition(record.mst)
                if condition == "sin_mst" and not include_missing_mst:
                    continue

                path_landmarks = cls._build_landmark_path(
                    record, landmarks_root, hagrid_mapping, landmarks_index
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
                        "landmark_quality": cls._infer_landmark_quality(
                            record, hagrid_quality_mapping
                        ),
                        "mst": "" if record.mst is None else record.mst,
                        "mst_origin": record.mst_origin,
                        "split": "train",
                    }
                )
                written_rows += 1

        if missing_by_source:
            missing_total = sum(missing_by_source.values())
            print(f"Aviso: el manifiesto ST-GCN incluye rutas de landmarks faltantes ({missing_total}/{written_rows}).")
            print(f"- faltantes por fuente: {dict(missing_by_source)}")

    @staticmethod
    def compute_summary(records: list[SampleRecord]) -> dict[str, object]:
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

    @staticmethod
    def write_summary_json(
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

    @staticmethod
    def write_tone_sets(output_dir: Path, records: list[SampleRecord]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        tone_sets: dict[str, list[SampleRecord]] = {
            "claro": [r for r in records if r.mst is not None and 1 <= r.mst <= 4],
            "medio": [r for r in records if r.mst is not None and 5 <= r.mst <= 7],
            "oscuro": [r for r in records if r.mst is not None and 8 <= r.mst <= 10],
        }

        for tone_name, tone_records in tone_sets.items():
            out_csv = output_dir / f"train_set_{tone_name}.csv"
            with out_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["sample_id", "source", "gesture", "mst", "split"])
                writer.writeheader()
                for record in tone_records:
                    writer.writerow({
                        "sample_id": record.sample_id,
                        "source": record.source,
                        "gesture": record.gesture,
                        "mst": record.mst,
                        "split": "train",
                    })

    @staticmethod
    def write_landmark_training_dirs(output_dir: Path, records: list[SampleRecord]) -> None:
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
                    f, fieldnames=["sample_id", "source", "gesture", "mst", "split", "mst_origin"]
                )
                writer.writeheader()
                for record in tone_records:
                    writer.writerow({
                        "sample_id": record.sample_id,
                        "source": record.source,
                        "gesture": record.gesture,
                        "mst": record.mst,
                        "split": "train",
                        "mst_origin": record.mst_origin,
                    })

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
            "# training_tono", "",
            "Directorios de entrenamiento por tono para el modelo de landmarks.", "",
            "## Estructura", "",
            "- claro/train_manifest.csv y claro/stats.json",
            "- medio/train_manifest.csv y medio/stats.json",
            "- oscuro/train_manifest.csv y oscuro/stats.json", "",
            "Cada manifesto contiene solo split=train y muestras del bloque MST correspondiente.",
        ]
        (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

