"""Evidence-gated approximate character timing for ChineseEEG1.

The experiment has ROWS/ROWE markers, but no per-character EEG marker.  This
module therefore separates two claims that are easy to conflate:

* the configured visual dwell time (0.35 s) is verified by the official code;
* exact character onsets are *not* observed and must remain approximate.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

OFFICIAL_REPOSITORY = (
    "https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing"
)
OFFICIAL_COMMIT = "279b2a6d5b1ea9d58a7fdb15a7b54fb7d8f72baf"
OFFICIAL_PRESENTATION_FILE = "experiment/play_novel.py"
OFFICIAL_CHARACTER_DWELL_SEC = 0.35

# This is deliberately the program's explicit list, not a Unicode-category
# approximation.  Characters such as em dash and single quotation marks are
# consequently presentation-clock characters in the released program.
OFFICIAL_SKIPPED_CHARACTERS = frozenset(
    {
        "\n",
        "。",
        "，",
        "！",
        "？",
        "：",
        "；",
        "“",
        "”",
        "、",
        "《",
        "》",
        ".",
        "（",
        "）",
        "…",
        "·",
    }
)
TIMELINE_AUDIT_SCHEMA_VERSION = "ce1-character-timeline-audit-v1"


def is_chineseeeg1_clock_character(character: str) -> bool:
    """Return the exact presentation-program decision for one character."""

    if len(character) != 1:
        raise ValueError("Expected exactly one character")
    return character not in OFFICIAL_SKIPPED_CHARACTERS


def chineseeeg1_clock_positions(text: str) -> tuple[int, ...]:
    """Raw Python-string offsets of characters that advance the visual clock."""

    return tuple(
        index
        for index, character in enumerate(text)
        if is_chineseeeg1_clock_character(character)
    )


@dataclass(frozen=True, slots=True)
class OfficialCodeEvidence:
    source_path: str
    source_sha256: str
    shift_time_default_verified: bool
    rows_marker_verified: bool
    rowe_marker_verified: bool
    timed_loop_verified: bool
    explicit_punctuation_list_verified: bool
    rows_precedes_timed_loop: bool
    rowe_follows_timed_loop: bool
    per_character_marker_present: bool

    @property
    def configured_clock_verified(self) -> bool:
        return all(
            (
                self.shift_time_default_verified,
                self.rows_marker_verified,
                self.rowe_marker_verified,
                self.timed_loop_verified,
                self.explicit_punctuation_list_verified,
                self.rows_precedes_timed_loop,
                self.rowe_follows_timed_loop,
            )
        )


@dataclass(frozen=True, slots=True)
class TimelineAuditThresholds:
    minimum_duration_count_correlation: float = 0.98
    minimum_run_correlation_p05: float = 0.95
    maximum_regression_absolute_error_p95_sec: float = 0.35
    minimum_effective_stride_sec: float = 0.35
    maximum_effective_stride_sec: float = 0.50


def _literal_punctuation_lists(tree: ast.AST) -> list[frozenset[str]]:
    lists: list[frozenset[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "punctuations" for target in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, (list, tuple, set)) and all(
            isinstance(item, str) for item in value
        ):
            lists.append(frozenset(value))
    return lists


def inspect_official_presentation_code(path: str | Path) -> OfficialCodeEvidence:
    """Inspect a local checkout of the pinned official presentation file."""

    source_path = Path(path)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    punctuation_lists = _literal_punctuation_lists(tree)
    compact = "".join(source.split())
    rows_tokens = ("send_event(event_type='ROWS')", 'send_event(event_type="ROWS")')
    rowe_tokens = ("send_event(event_type='ROWE')", 'send_event(event_type="ROWE")')
    loop_tokens = (
        "whileroutineTimer.getTime()<args.shift_time:",
        "whileroutineTimer.getTime()<=args.shift_time:",
    )
    rows_positions = [compact.find(token) for token in rows_tokens if token in compact]
    rowe_positions = [compact.find(token) for token in rowe_tokens if token in compact]
    loop_positions = [compact.find(token) for token in loop_tokens if token in compact]
    rows_position = min(rows_positions, default=-1)
    rowe_position = min(rowe_positions, default=-1)
    loop_position = min(loop_positions, default=-1)
    per_character_markers = any(
        marker in source
        for marker in ("CHAR_START", "CHAR_END", "CHRS", "CHRE", "WORD_START")
    )
    return OfficialCodeEvidence(
        source_path=str(source_path.resolve()),
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        shift_time_default_verified=(
            "add_argument('--shift_time',type=float,default=0.35" in compact
            or 'add_argument("--shift_time",type=float,default=0.35' in compact
        ),
        rows_marker_verified=bool(rows_positions),
        rowe_marker_verified=bool(rowe_positions),
        timed_loop_verified=bool(loop_positions),
        explicit_punctuation_list_verified=(
            OFFICIAL_SKIPPED_CHARACTERS in punctuation_lists
        ),
        rows_precedes_timed_loop=(
            rows_position >= 0 and loop_position >= 0 and rows_position < loop_position
        ),
        rowe_follows_timed_loop=(
            rowe_position >= 0 and loop_position >= 0 and rowe_position > loop_position
        ),
        per_character_marker_present=per_character_markers,
    )


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 2 or np.unique(x).size < 2:
        raise ValueError("A duration fit needs at least two distinct character counts")
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x
    residual = y - prediction
    return {
        "slope_sec_per_character": float(slope),
        "intercept_sec": float(intercept),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "median_absolute_error_sec": float(np.median(np.abs(residual))),
        "p95_absolute_error_sec": float(np.quantile(np.abs(residual), 0.95)),
    }


def audit_chineseeeg1_timeline(
    rows: pd.DataFrame,
    *,
    code_evidence: OfficialCodeEvidence,
    manifest_path: str | Path,
    thresholds: TimelineAuditThresholds = TimelineAuditThresholds(),
) -> dict[str, Any]:
    """Audit code and event-duration evidence without inventing onsets."""

    required = {
        "dataset_version",
        "paradigm",
        "raw_text",
        "highlight_char_count",
        "eeg_duration_sec",
        "subject_id",
        "session_id",
        "run_id",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Timeline audit is missing columns: {sorted(missing)}")
    frame = rows.loc[
        (rows["dataset_version"] == "ChineseEEG1")
        & (rows["paradigm"] == "silent_reading")
    ].copy()
    if frame.empty:
        raise ValueError("No ChineseEEG1 silent-reading rows were found")
    frame = frame.loc[frame["raw_text"].notna() & frame["eeg_duration_sec"].notna()].copy()
    frame["official_clock_character_count"] = frame["raw_text"].map(
        lambda value: len(chineseeeg1_clock_positions(str(value)))
    )
    frame = frame.loc[frame["official_clock_character_count"] > 0].copy()
    counts = frame["official_clock_character_count"].to_numpy(dtype=float)
    durations = frame["eeg_duration_sec"].to_numpy(dtype=float)
    global_fit = _linear_fit(counts, durations)
    configured_residual = durations - counts * OFFICIAL_CHARACTER_DWELL_SEC

    run_fits: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["subject_id", "session_id", "run_id"]):
        group_counts = group["official_clock_character_count"].to_numpy(dtype=float)
        if len(group_counts) < 2 or np.unique(group_counts).size < 2:
            continue
        fit = _linear_fit(
            group_counts,
            group["eeg_duration_sec"].to_numpy(dtype=float),
        )
        fit.update(
            {
                "subject_id": str(keys[0]),
                "session_id": str(keys[1]),
                "run_id": str(keys[2]),
                "row_count": int(len(group)),
            }
        )
        run_fits.append(fit)

    run_correlations = np.asarray([fit["pearson_r"] for fit in run_fits])
    run_slopes = np.asarray(
        [fit["slope_sec_per_character"] for fit in run_fits]
    )
    data_gate = bool(
        global_fit["pearson_r"] >= thresholds.minimum_duration_count_correlation
        and np.quantile(run_correlations, 0.05)
        >= thresholds.minimum_run_correlation_p05
        and global_fit["p95_absolute_error_sec"]
        <= thresholds.maximum_regression_absolute_error_p95_sec
        and thresholds.minimum_effective_stride_sec
        <= global_fit["slope_sec_per_character"]
        <= thresholds.maximum_effective_stride_sec
    )
    configured_pace_verified = bool(
        code_evidence.configured_clock_verified and data_gate
    )
    exact_onsets_verified = bool(code_evidence.per_character_marker_present)
    verdict = (
        "verified_approximate_only"
        if configured_pace_verified and not exact_onsets_verified
        else "exact_character_events_verified"
        if configured_pace_verified and exact_onsets_verified
        else "weak_supervision_only"
    )
    manifest_path = Path(manifest_path)
    code_payload = asdict(code_evidence)
    # Local checkout paths are machine-specific and must not enter a versioned
    # scientific artifact. The canonical repository/commit/file triple below is
    # the reproducible provenance.
    code_payload.pop("source_path", None)
    report: dict[str, Any] = {
        "schema_version": TIMELINE_AUDIT_SCHEMA_VERSION,
        "verdict": verdict,
        "configured_pace_verified": configured_pace_verified,
        "exact_character_onsets_verified": exact_onsets_verified,
        "allowed_timeline_methods": (
            ["event_affine", "fixed_dwell_sensitivity", "sentence_weak"]
            if configured_pace_verified
            else ["sentence_weak"]
        ),
        "scientific_interpretation": (
            "The 0.35 s visual dwell is verified, but ROWS/ROWE do not observe "
            "per-character onset. Event-affine boundaries are approximate visual "
            "timing and are not word or speech timestamps."
        ),
        "official_source": {
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "file": OFFICIAL_PRESENTATION_FILE,
            **code_payload,
        },
        "presentation_rule": {
            "configured_character_dwell_sec": OFFICIAL_CHARACTER_DWELL_SEC,
            "skipped_characters": sorted(OFFICIAL_SKIPPED_CHARACTERS),
            "line_break_advances_clock": False,
            "explicit_leading_static_interval": False,
            "explicit_trailing_static_interval": False,
            "transition_rendering_overhead_inside_rows_rowe": True,
        },
        "manifest": {
            "path": manifest_path.as_posix(),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "audited_row_count": int(len(frame)),
            "recording_count": int(
                frame[["subject_id", "session_id", "run_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "existing_highlight_count_disagreement_rows": int(
                (
                    frame["official_clock_character_count"].astype(float)
                    != frame["highlight_char_count"].astype(float)
                ).sum()
            ),
        },
        "event_duration_evidence": {
            "global_fit": global_fit,
            "duration_minus_0p35n_sec": {
                "median": float(np.median(configured_residual)),
                "p05": float(np.quantile(configured_residual, 0.05)),
                "p95": float(np.quantile(configured_residual, 0.95)),
            },
            "run_fit_summary": {
                "run_count": int(len(run_fits)),
                "pearson_r_p05": float(np.quantile(run_correlations, 0.05)),
                "pearson_r_median": float(np.median(run_correlations)),
                "slope_sec_per_character_p05": float(np.quantile(run_slopes, 0.05)),
                "slope_sec_per_character_median": float(np.median(run_slopes)),
                "slope_sec_per_character_p95": float(np.quantile(run_slopes, 0.95)),
            },
        },
        "thresholds": asdict(thresholds),
        "run_fits": run_fits,
    }
    return report


def write_timeline_audit(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_target.write_text(
        render_timeline_audit_markdown(report),
        encoding="utf-8",
        newline="\n",
    )


def render_timeline_audit_markdown(report: Mapping[str, Any]) -> str:
    fit = report["event_duration_evidence"]["global_fit"]
    run_summary = report["event_duration_evidence"]["run_fit_summary"]
    manifest = report["manifest"]
    rule = report["presentation_rule"]
    source = report["official_source"]
    return f"""# ChineseEEG1 character-timeline audit

## Verdict

- Verdict: `{report['verdict']}`
- Configured 0.35 s visual pace verified: `{report['configured_pace_verified']}`
- Exact per-character onset verified: `{report['exact_character_onsets_verified']}`
- Audited rows: {manifest['audited_row_count']:,}
- Audited recordings: {manifest['recording_count']:,}

The released events support an **approximate visual character clock**, not true
word timestamps and not speech timestamps. `ROWS`/`ROWE` only bracket a displayed
row. The presentation loop rebuilds PsychoPy text objects between characters, so
the event interval contains rendering overhead in addition to the configured
0.35 s dwell.

## Official-code evidence

- Repository: {source['repository']}
- Pinned commit: `{source['commit']}`
- File: `{source['file']}`
- Audited source SHA-256: `{source['source_sha256']}`
- Explicit leading static interval: `{rule['explicit_leading_static_interval']}`
- Explicit trailing static interval: `{rule['explicit_trailing_static_interval']}`
- Per-character event marker present: `{source['per_character_marker_present']}`
- Newline advances the visual clock: `{rule['line_break_advances_clock']}`

The exact skip list is stored in the JSON audit. It is intentionally used instead
of generic Unicode punctuation categories; the current manifest differs on
{manifest['existing_highlight_count_disagreement_rows']:,} rows, mostly because
some special punctuation is advanced by the actual experiment program.

## Event-duration evidence

The global fit is
`ROWE-ROWS = {fit['intercept_sec']:.6f} + {fit['slope_sec_per_character']:.6f} × N`
seconds, with Pearson `r={fit['pearson_r']:.6f}`. Median absolute regression error
is {fit['median_absolute_error_sec']:.6f} s and P95 error is
{fit['p95_absolute_error_sec']:.6f} s. Across recordings, the median slope is
{run_summary['slope_sec_per_character_median']:.6f} s/character and the P05
correlation is {run_summary['pearson_r_p05']:.6f}.

## Permitted use

The primary local-span index may use event-affine boundaries, storing the method,
estimated disagreement from the configured clock, and a confidence flag. Model
inputs are resampled to a fixed length per character span, so source duration and
padding masks are unavailable to the model. `fixed_dwell_sensitivity` is retained
only as an alignment sensitivity analysis. If this audit fails on another data
release, code must fall back to sentence-level weak supervision.
"""


def assert_timeline_method_allowed(report: Mapping[str, Any], method: str) -> None:
    allowed: Iterable[str] = report.get("allowed_timeline_methods", ())
    if method not in allowed:
        raise ValueError(
            f"Timeline method {method!r} is not allowed by audit verdict "
            f"{report.get('verdict')!r}; allowed={list(allowed)}"
        )
