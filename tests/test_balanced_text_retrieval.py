from evaluation.balanced_text_retrieval import macro_metrics_from_ranks


def test_balanced_recall_macro_weights_text_types_equally() -> None:
    metrics = macro_metrics_from_ranks(
        [1, 1, 20, 20],
        ["frequent", "frequent", "frequent", "rare"],
    )
    # frequent R@10=2/3, rare R@10=0; macro is their unweighted mean.
    assert metrics.recall_at_10 == (2 / 3) / 2
