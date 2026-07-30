"""Content-aware batch samplers that never change partition membership."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class UniqueTargetBatchSampler(Sampler[list[int]]):
    """Visit every sample while preventing duplicate target IDs within a batch."""

    def __init__(
        self,
        target_ids: Sequence[str],
        *,
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2 for contrastive learning")
        if not target_ids:
            raise ValueError("target_ids cannot be empty")
        self.target_ids = list(target_ids)
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, target_id in enumerate(self.target_ids):
            grouped[target_id].append(index)
        if len(grouped) < batch_size and drop_last:
            raise ValueError("Not enough unique targets to form one full batch")
        self._grouped = {
            target_id: grouped[target_id] for target_id in sorted(grouped)
        }

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(f"{self.seed}\0{self.epoch}")
        queues = {
            target_id: rng.sample(indices, k=len(indices))
            for target_id, indices in self._grouped.items()
        }
        while queues:
            active_targets = list(queues)
            rng.shuffle(active_targets)
            round_indices = [queues[target_id].pop() for target_id in active_targets]
            for target_id in active_targets:
                if not queues[target_id]:
                    del queues[target_id]
            for start in range(0, len(round_indices), self.batch_size):
                batch = round_indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        # Each target-round flushes its partial batch to preserve uniqueness.
        remaining = {
            target_id: len(indices) for target_id, indices in self._grouped.items()
        }
        batch_count = 0
        while remaining:
            active_count = len(remaining)
            full, partial = divmod(active_count, self.batch_size)
            batch_count += full + int(bool(partial) and not self.drop_last)
            remaining = {
                target_id: count - 1
                for target_id, count in remaining.items()
                if count > 1
            }
        return batch_count
