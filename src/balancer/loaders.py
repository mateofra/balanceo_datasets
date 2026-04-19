from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from src.balancer.core import SampleRecord


class DataLoaders:
    @staticmethod
    def load_landmark_mapping(mapping_json: Path | None) -> dict[str, str]:
        if not mapping_json or not mapping_json.exists():
            return {}
        with mapping_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        mapping: dict[str, str] = {}
        for raw_key, raw_path in payload.items():
            key = str(raw_key).strip().lower()
            if key:
                mapping[key] = str(raw_path).replace("\\", "/")
        return mapping

    @staticmethod
    def load_quality_mapping(mapping_json: Path | None) -> dict[str, str]:
        if not mapping_json or not mapping_json.exists():
            return {}
        with mapping_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        mapping: dict[str, str] = {}
        for raw_key, raw_quality in payload.items():
            key = str(raw_key).strip().lower()
            if key:
                mapping[key] = str(raw_quality).strip()
        return mapping

    @staticmethod
    def load_freihand_records(
        training_xyz_path: Path,
        canonical_rgb_manifest_csv: Path | None = None,
    ) -> list[SampleRecord]:
        with training_xyz_path.open("r", encoding="utf-8") as f:
            xyz = json.load(f)

        canonical_indices: list[int] | None = None
        if canonical_rgb_manifest_csv is not None and canonical_rgb_manifest_csv.exists():
            expected_ids = {f"freihand_{idx:08d}" for idx in range(len(xyz))}
            indices: set[int] = set()

            with canonical_rgb_manifest_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sample_id = str(row.get("sample_id", "")).strip().lower()
                    if sample_id in expected_ids:
                        try:
                            indices.add(int(sample_id.split("_", 1)[1]))
                        except (IndexError, ValueError):
                            continue

            if indices:
                canonical_indices = sorted(indices)
                print(
                    "FreiHAND: usando manifiesto canónico "
                    f"{canonical_rgb_manifest_csv} ({len(canonical_indices)} muestras)."
                )
            else:
                print(
                    "Aviso: manifiesto canónico FreiHAND sin filas válidas para training_xyz; "
                    "se usa rango completo 0..N-1."
                )

        selected_indices = canonical_indices if canonical_indices is not None else list(range(len(xyz)))

        records: list[SampleRecord] = []
        for idx in selected_indices:
            records.append(
                SampleRecord(
                    sample_id=f"freihand_{idx:08d}",
                    source="freihand",
                    gesture="unknown",
                )
            )
        return records

    @staticmethod
    def _parse_hagrid_sample_id(raw_id: str) -> str:
        return raw_id.strip().lower()

    @staticmethod
    def _pick_primary_label(labels: Iterable[str], fallback_gesture: str) -> str:
        for label in labels:
            label_norm = label.strip().lower()
            if label_norm and label_norm != "no_gesture":
                return label_norm
        return fallback_gesture

    @classmethod
    def load_hagrid_records(cls, annotations_dir: Path, gestures: list[str]) -> list[SampleRecord]:
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
                primary_label = cls._pick_primary_label(labels, fallback_gesture=gesture)
                sample_id = cls._parse_hagrid_sample_id(image_id)
                records.append(
                    SampleRecord(
                        sample_id=sample_id,
                        source="hagrid",
                        gesture=primary_label,
                    )
                )

        return records


class MstLoader:
    @staticmethod
    def load_mst_map(mst_csv_path: Path) -> dict[str, int]:
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

    @staticmethod
    def attach_mst(records: list[SampleRecord], mst_map: dict[str, int]) -> list[SampleRecord]:
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

    @staticmethod
    def compute_match_report(
        freihand_records: list[SampleRecord],
        hagrid_records: list[SampleRecord],
        mst_map: dict[str, int],
    ) -> dict[str, object]:
        freihand_keys = [r.sample_id.lower() for r in freihand_records]
        hagrid_keys = [r.sample_id.lower() for r in hagrid_records]

        freihand_matches = sum(1 for key in freihand_keys if key in mst_map)
        hagrid_matches = sum(1 for key in hagrid_keys if key in mst_map)
        total_records = len(freihand_keys) + len(hagrid_keys)
        total_matches = freihand_matches + hagrid_matches

        def pct(matches: int, total: int) -> float:
            return round((matches / total) * 100.0, 2) if total > 0 else 0.0

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

    @staticmethod
    def print_match_report(report: dict[str, object]) -> None:
        fh = report["freihand"]
        hg = report["hagrid"]
        tot = report["total"]

        print("Cobertura de match MST (antes de balancear):")
        print(f"- filas_unicas_mst_csv: {report['mst_rows']}")
        print(f"- freihand: {fh['matches']}/{fh['records']} ({fh['match_pct']}%)")
        print(f"- hagrid: {hg['matches']}/{hg['records']} ({hg['match_pct']}%)")
        print(f"- total: {tot['matches']}/{tot['records']} ({tot['match_pct']}%)")

        if float(tot["match_pct"]) < 20.0:  # type: ignore
            print(
                "Aviso: cobertura de MST muy baja (<20%). "
                "El CSV podria ser mock o estar mal alineado con sample_id/image_id."
            )
