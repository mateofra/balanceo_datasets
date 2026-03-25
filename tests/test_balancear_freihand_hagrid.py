import random
import tempfile
import unittest
from pathlib import Path

from src.balancear_freihand_hagrid import (
    SampleRecord,
    _build_balanced_manifest,
    _compute_summary,
    _expand_with_dark_jitter_candidates,
    _impute_missing_mst,
    _sample_with_mst_priority,
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


if __name__ == "__main__":
    unittest.main()
