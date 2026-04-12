from __future__ import annotations

import random
from collections import defaultdict

from src.balancer.core import SampleRecord


class DatasetBalancer:
    def __init__(self, rng: random.Random, extreme_mst_levels: set[int], extreme_factor: float):
        self.rng = rng
        self.extreme_mst_levels = extreme_mst_levels
        self.extreme_factor = extreme_factor

    def _sample_with_replacement(self, pool: list[SampleRecord], target_size: int) -> list[SampleRecord]:
        if not pool:
            return []
        if target_size <= len(pool):
            return self.rng.sample(pool, target_size)
        return [self.rng.choice(pool) for _ in range(target_size)]

    def _mst_sampling_weight(self, record: SampleRecord) -> float:
        if record.mst is not None and record.mst in self.extreme_mst_levels:
            return self.extreme_factor
        return 1.0

    def _sample_with_mst_priority(self, pool: list[SampleRecord], target_size: int) -> list[SampleRecord]:
        if not pool:
            return []

        has_mst = any(record.mst is not None for record in pool)
        if not has_mst:
            return self._sample_with_replacement(pool, target_size)

        weights = [self._mst_sampling_weight(record) for record in pool]
        return self.rng.choices(pool, weights=weights, k=target_size)

    def _group_by_mst_block(self, records: list[SampleRecord]) -> dict[str, list[SampleRecord]]:
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

    def _balance_with_optional_blocks(self, records: list[SampleRecord], target_size: int) -> list[SampleRecord]:
        grouped = self._group_by_mst_block(records)

        if set(grouped.keys()) <= {"sin_mst"}:
            return self._sample_with_replacement(records, target_size)

        # Objetivo por bloques 33/33/33 cuando exista MST.
        block_targets = {
            "claro": target_size // 3,
            "medio": target_size // 3,
            "oscuro": target_size - 2 * (target_size // 3),
        }

        selected: list[SampleRecord] = []
        for block_name in ("claro", "medio", "oscuro"):
            selected.extend(
                self._sample_with_mst_priority(
                    grouped.get(block_name, []),
                    block_targets[block_name],
                )
            )

        return selected

    def build_balanced_manifest(self, freihand_records: list[SampleRecord], hagrid_records: list[SampleRecord], target_size: int, hagrid_ratio: float) -> list[SampleRecord]:
        hagrid_target = int(round(target_size * hagrid_ratio))
        freihand_target = target_size - hagrid_target

        selected_hagrid = self._balance_with_optional_blocks(hagrid_records, hagrid_target)
        selected_freihand = self._balance_with_optional_blocks(freihand_records, freihand_target)

        merged = selected_hagrid + selected_freihand
        self.rng.shuffle(merged)
        return merged

    def expand_with_dark_jitter_candidates(self, records: list[SampleRecord], jitter_factor: float) -> list[SampleRecord]:
        if jitter_factor <= 0.0:
            return records

        def _is_dark_aug_candidate(record: SampleRecord) -> bool:
            return record.mst in (8, 9)

        base_candidates = [record for record in records if _is_dark_aug_candidate(record)]
        if not base_candidates:
            return records

        n_extra = int(round(len(base_candidates) * jitter_factor))
        if n_extra <= 0:
            return records

        extra_records = [self.rng.choice(base_candidates) for _ in range(n_extra)]
        return records + extra_records
