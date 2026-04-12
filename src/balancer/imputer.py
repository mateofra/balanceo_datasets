from __future__ import annotations

import random
from collections import Counter

from src.balancer.core import SampleRecord


class MstImputer:
    @staticmethod
    def _choose_level_for_block(block_name: str, rng: random.Random) -> int:
        if block_name == "claro":
            return rng.choice([1, 2, 3, 4])
        if block_name == "medio":
            return rng.choice([5, 6, 7])
        return rng.choice([8, 9, 10])

    @classmethod
    def impute_missing_mst(cls, records: list[SampleRecord], rng: random.Random) -> list[SampleRecord]:
        total = len(records)
        if total == 0:
            return records

        current_blocks: Counter[str] = Counter()
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

            mst_value = cls._choose_level_for_block(best_block, rng)
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
