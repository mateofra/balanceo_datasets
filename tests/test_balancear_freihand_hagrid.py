import random
import tempfile
import unittest
import csv
import io
from contextlib import redirect_stdout
from pathlib import Path

from src.balancer.balancear_freihand_hagrid import (
    SampleRecord,
    _build_landmark_path,
    _build_balanced_manifest,
    _compute_mst_match_report,
    _compute_summary,
    _expand_with_dark_jitter_candidates,
    _impute_missing_mst,
    _mst_to_condition,
    _print_mst_match_report,
    _sample_with_mst_priority,
    _write_stgcn_manifest_csv,
    _write_landmark_training_dirs,
    _write_tone_sets,
)


class BalancearFreihandHagridTests(unittest.TestCase):
    def test_balanceo_respeta_ratio_fuente(self) -> None:
        freihand = [
            SampleRecord(sample_id=f"freihand_{i}", source="freihand", gesture="unknown")
            for i in range(200)
        ]
        hagrid = [
            SampleRecord(sample_id=f"hagrid_{i}", source="hagrid", gesture="palm")
            for i in range(200)
        ]

        records = _build_balanced_manifest(
            freihand_records=freihand,
            hagrid_records=hagrid,
            target_size=100,
            hagrid_ratio=0.4,
            rng=random.Random(42),
            extreme_mst_levels={1, 2, 3, 10},
            extreme_factor=2.0,
        )

        summary = _compute_summary(records)
        self.assertEqual(summary["total_samples"], 100)
        self.assertEqual(summary["by_source"]["hagrid"], 40)
        self.assertEqual(summary["by_source"]["freihand"], 60)

    def test_oversampling_extremos_mst_incrementa_frecuencia(self) -> None:
        pool = []
        for i in range(20):
            pool.append(SampleRecord(sample_id=f"e_{i}", source="hagrid", gesture="palm", mst=1))
        for i in range(20):
            pool.append(SampleRecord(sample_id=f"m_{i}", source="hagrid", gesture="palm", mst=5))

        sampled = _sample_with_mst_priority(
            pool=pool,
            target_size=400,
            rng=random.Random(42),
            extreme_mst_levels={1, 2, 3, 10},
            extreme_factor=3.0,
        )

        mst_counts = {}
        for record in sampled:
            mst_counts[record.mst] = mst_counts.get(record.mst, 0) + 1

        self.assertGreater(mst_counts.get(1, 0), mst_counts.get(5, 0))

    def test_dark_jitter_factor_agrega_registros_mst_8_9(self) -> None:
        records = [
            SampleRecord(sample_id="a", source="freihand", gesture="unknown", mst=8),
            SampleRecord(sample_id="b", source="freihand", gesture="unknown", mst=9),
            SampleRecord(sample_id="c", source="freihand", gesture="unknown", mst=5),
        ]

        expanded = _expand_with_dark_jitter_candidates(
            records,
            jitter_factor=1.0,
            rng=random.Random(42),
        )

        self.assertEqual(len(expanded), 5)
        self.assertGreaterEqual(sum(1 for r in expanded if r.mst in (8, 9)), 4)

    def test_exporta_tres_sets_por_tono(self) -> None:
        records = [
            SampleRecord(sample_id="c1", source="hagrid", gesture="palm", mst=2),
            SampleRecord(sample_id="m1", source="hagrid", gesture="palm", mst=6),
            SampleRecord(sample_id="o1", source="hagrid", gesture="palm", mst=9),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _write_tone_sets(out_dir, records)

            self.assertTrue((out_dir / "train_set_claro.csv").exists())
            self.assertTrue((out_dir / "train_set_medio.csv").exists())
            self.assertTrue((out_dir / "train_set_oscuro.csv").exists())

    def test_exporta_directorios_para_entrenamiento_landmarks_por_tono(self) -> None:
        records = [
            SampleRecord(sample_id="c1", source="hagrid", gesture="palm", mst=2),
            SampleRecord(sample_id="m1", source="hagrid", gesture="palm", mst=6),
            SampleRecord(sample_id="o1", source="freihand", gesture="unknown", mst=9),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "landmarks_train"
            _write_landmark_training_dirs(out_dir, records)

            self.assertTrue((out_dir / "README.md").exists())
            self.assertTrue((out_dir / "claro" / "train_manifest.csv").exists())
            self.assertTrue((out_dir / "medio" / "train_manifest.csv").exists())
            self.assertTrue((out_dir / "oscuro" / "train_manifest.csv").exists())
            self.assertTrue((out_dir / "claro" / "stats.json").exists())
            self.assertTrue((out_dir / "medio" / "stats.json").exists())
            self.assertTrue((out_dir / "oscuro" / "stats.json").exists())

    def test_imputacion_mst_clasifica_todos_los_objetos(self) -> None:
        records = [
            SampleRecord(sample_id="k1", source="hagrid", gesture="palm", mst=None),
            SampleRecord(sample_id="k2", source="hagrid", gesture="palm", mst=None),
            SampleRecord(sample_id="k3", source="hagrid", gesture="palm", mst=2, mst_origin="original"),
        ]

        imputed = _impute_missing_mst(records, random.Random(7))

        self.assertTrue(all(r.mst is not None for r in imputed))
        self.assertTrue(all(1 <= int(r.mst) <= 10 for r in imputed))
        self.assertGreaterEqual(sum(1 for r in imputed if r.mst_origin == "imputed"), 2)

    def test_condition_por_mst(self) -> None:
        self.assertEqual(_mst_to_condition(2), "claro")
        self.assertEqual(_mst_to_condition(6), "medio")
        self.assertEqual(_mst_to_condition(9), "oscuro")
        self.assertEqual(_mst_to_condition(None), "sin_mst")

    def test_path_landmarks_hagrid_y_freihand(self) -> None:
        root = Path("data/processed/landmarks")
        hagrid_record = SampleRecord(sample_id="abc123", source="hagrid", gesture="call", mst=8)
        freihand_record = SampleRecord(sample_id="freihand_00000050", source="freihand", gesture="unknown", mst=3)

        self.assertEqual(
            _build_landmark_path(hagrid_record, root),
            str(Path("data/processed/landmarks/hagrid/call/abc123.npy")),
        )
        self.assertEqual(
            _build_landmark_path(freihand_record, root),
            str(Path("data/processed/landmarks/freihand/00000050.npy")),
        )

    def test_exporta_manifest_stgcn(self) -> None:
        records = [
            SampleRecord(sample_id="h1", source="hagrid", gesture="call", mst=9, mst_origin="original"),
            SampleRecord(sample_id="freihand_00000012", source="freihand", gesture="unknown", mst=2, mst_origin="imputed"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "train_manifest_stgcn.csv"
            _write_stgcn_manifest_csv(
                output_csv=out_csv,
                records=records,
                landmarks_root=Path("data/processed/landmarks"),
                include_missing_mst=False,
            )

            self.assertTrue(out_csv.exists())
            with out_csv.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["dataset"], "hagrid")
            self.assertEqual(rows[0]["label"], "call")
            self.assertEqual(rows[0]["condition"], "oscuro")
            self.assertEqual(rows[1]["dataset"], "freihand")
            self.assertEqual(rows[1]["label"], "unknown")
            self.assertEqual(rows[1]["condition"], "claro")

    def test_reporte_match_mst(self) -> None:
        freihand = [
            SampleRecord(sample_id="freihand_00000001", source="freihand", gesture="unknown"),
            SampleRecord(sample_id="freihand_00000002", source="freihand", gesture="unknown"),
        ]
        hagrid = [
            SampleRecord(sample_id="abc", source="hagrid", gesture="call"),
            SampleRecord(sample_id="def", source="hagrid", gesture="call"),
        ]
        mst_map = {
            "freihand_00000001": 3,
            "abc": 8,
        }

        report = _compute_mst_match_report(freihand, hagrid, mst_map)
        self.assertEqual(report["freihand"]["matches"], 1)
        self.assertEqual(report["hagrid"]["matches"], 1)
        self.assertEqual(report["total"]["matches"], 2)

        output = io.StringIO()
        with redirect_stdout(output):
            _print_mst_match_report(report)
        printed = output.getvalue()

        self.assertIn("Cobertura de match MST", printed)
        self.assertIn("freihand", printed)
        self.assertIn("hagrid", printed)


if __name__ == "__main__":
    unittest.main()
