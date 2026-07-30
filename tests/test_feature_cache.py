from features.cache import safe_artifact_filename


def test_artifact_filename_is_stable_and_windows_safe() -> None:
    first = safe_artifact_filename("dataset:book/chapter\\row")
    second = safe_artifact_filename("dataset:book/chapter\\row")
    assert first == second
    assert first.endswith(".npz")
    assert ":" not in first
    assert "/" not in first
    assert "\\" not in first


def test_similar_sanitized_ids_remain_distinct() -> None:
    assert safe_artifact_filename("a:b") != safe_artifact_filename("a/b")
