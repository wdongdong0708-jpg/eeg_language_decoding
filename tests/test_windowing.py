from data.windowing import WindowSpec, assert_windows_within_block, make_block_windows


def test_windows_inherit_split_and_do_not_cross_block() -> None:
    windows = make_block_windows(
        block_id="block-1",
        content_id="content-1",
        split="test",
        block_start_sample=100,
        block_stop_sample=350,
        spec=WindowSpec(size_samples=100, stride_samples=50, tail_policy="drop"),
    )
    assert [(window.start_sample, window.stop_sample) for window in windows] == [
        (100, 200),
        (150, 250),
        (200, 300),
        (250, 350),
    ]
    assert {window.split for window in windows} == {"test"}
    assert_windows_within_block(
        windows,
        block_start_sample=100,
        block_stop_sample=350,
        expected_split="test",
    )


def test_padding_is_explicit_and_remains_inside_block() -> None:
    windows = make_block_windows(
        block_id="short",
        content_id="content-short",
        split="valid",
        block_start_sample=0,
        block_stop_sample=60,
        spec=WindowSpec(size_samples=100, stride_samples=50, tail_policy="pad"),
    )
    assert len(windows) == 1
    assert windows[0].valid_samples == 60
    assert windows[0].padded_samples == 40
    assert windows[0].stop_sample == 60

