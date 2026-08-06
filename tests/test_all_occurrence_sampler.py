from training.samplers import AllOccurrenceBatchSampler


def test_all_occurrence_sampler_visits_every_index_and_keeps_remainder() -> None:
    sampler = AllOccurrenceBatchSampler(
        11, batch_size=4, seed=42, drop_last=False
    )
    first = list(sampler)
    assert sorted(index for batch in first for index in batch) == list(range(11))
    assert [len(batch) for batch in first] == [4, 4, 3]
    assert list(sampler) == first
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_all_occurrence_sampler_drop_last_is_explicit() -> None:
    sampler = AllOccurrenceBatchSampler(11, batch_size=4, seed=42, drop_last=True)
    batches = list(sampler)
    assert len(batches) == 2
    assert sum(len(batch) for batch in batches) == 8
