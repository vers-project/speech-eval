"""Contrasts that apply to any measure, whatever produced it.

Everything here takes a column of numbers — an F0 median, an alpha ratio, a MOS,
a cosine similarity — and says how it moved between conditions.  None of it
knows which metric wrote the column, which is what lets one implementation serve
every metric family.

The two contrasts
-----------------
**Recovery rate** (paired).  Needs the same content at both the source and the
target condition, so it applies where a corpus records the same material at
several vocal efforts.  It answers: of the distance a real speaker travels
between two efforts, what fraction did the model travel?

**Slope against the target** (unpaired).  Fits how a measure changes per dB, on
real speech and on converted speech, and compares the two slopes.  It needs no
content matching and works on spontaneous corpora, which is what makes it the
fallback whenever pairing is impossible.

Neither is a metric.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# Below this, the source and target references are too close for a ratio: the
# denominator carries more noise than signal and the rate explodes.  Expressed
# in the unit of whatever measure is passed in, so callers measuring in Hz and
# callers measuring in dB both have to think about it — hence no default that
# silently fits one and not the other.
DEFAULT_MIN_GAP = 1e-6


def array_features(
    artifacts: dict[str, np.ndarray],
    column: str,
    utt_ids: list[str] | pd.Series | None = None,
) -> dict[str, np.ndarray]:
    """Pull one array-valued feature out of ``artifacts.npz``, keyed by ``utt_id``.

    The runner stores arrays too large for a CSV cell under ``"<utt_id>|<metric
    name>.<key>"`` — speaker embeddings, F0 contours, spectra.  This undoes that
    keying for one column, so a comparison can work in terms of utterances
    instead of in terms of the archive's flat namespace.

    ``column`` is the full prefixed name, e.g. ``"speaker_ecapa.embedding"``.
    ``utt_ids`` restricts the result and is otherwise ignored; utterances with no
    entry are simply absent from the returned mapping, since a metric that
    skipped an utterance is the normal case rather than an error.

    Raises
    ------
    KeyError
        If no key in the archive carries this column at all — that is a typo or
        a run that never computed the metric, and returning an empty dict would
        turn it into a comparison over zero pairs.
    """
    suffix = f"|{column}"
    found = {
        key[: -len(suffix)]: value
        for key, value in artifacts.items()
        if key.endswith(suffix)
    }
    if not found:
        available = sorted({key.split("|", 1)[1] for key in artifacts if "|" in key})
        raise KeyError(
            f"No artifact holds column {column!r}. Available: {available}"
        )
    if utt_ids is None:
        return found
    wanted = {str(u) for u in utt_ids}
    return {k: v for k, v in found.items() if k in wanted}


# ---------------------------------------------------------------------------
# Shared plumbing for comparisons that read one metric column
# ---------------------------------------------------------------------------
#
# Both the WER and the speaker-similarity families exclude rows their metric
# flagged, and both can score a condition against another condition of the same
# group.  Those two steps live here rather than in each family, so that "which
# rows were dropped, and what was printed about it" is one behaviour and cannot
# drift into two.


def status_column_for(value_col: str, frame: pd.DataFrame) -> str | None:
    """The ``status`` column belonging to ``value_col``'s metric, if it exists.

    Columns are named ``<metric name>.<key>``, so the status of
    ``asr_whisper.hypothesis`` is ``asr_whisper.status``.  Returns ``None`` when
    the metric emits no status, which is the normal case for metrics that cannot
    fail per utterance.
    """
    prefix = value_col.rsplit(".", 1)[0]
    candidate = f"{prefix}.status"
    return candidate if candidate in frame.columns else None


def exclude_flagged_rows(
    frame: pd.DataFrame,
    status_col: str | None,
    ok_values: tuple[str, ...] = ("ok",),
    label: str = "",
) -> pd.DataFrame:
    """Drop rows whose metric flagged them, printing what went and why.

    Silent exclusion is the failure mode worth avoiding: a manifest that lost
    most of its rows would otherwise look like a clean run on a tenth of the
    data.

    ``ok_values`` is a tuple because not every flag is fatal.  A transcript past
    Whisper's 30 s window is wrong and must go; a speaker embedding from a short
    segment is merely noisy and is usually worth keeping, so the speaker family
    passes ``("ok", "short")``.
    """
    if status_col is None:
        return frame
    if status_col not in frame.columns:
        raise KeyError(f"Results table has no column {status_col!r}.")

    flagged = ~frame[status_col].astype(str).isin(ok_values)
    if flagged.any():
        counts = frame.loc[flagged, status_col].value_counts().to_dict()
        print(
            f"[{label}] {int(flagged.sum())} of {len(frame)} rows excluded by the "
            f"metric: {counts}"
        )
        frame = frame[~flagged]
    return frame


def resolve_reference_condition(
    frame: pd.DataFrame,
    reference_condition: str,
    source_col: str,
    group_col: str = "group_id",
    condition_col: str = "condition",
    label: str = "",
) -> tuple[pd.DataFrame, pd.Series]:
    """Give every row the value ``source_col`` takes at another condition.

    The pseudo-reference pattern: within each ``group_id``, find the row whose
    condition is ``reference_condition`` and hand its ``source_col`` value to
    every row of that group.  What the value *is* differs by family — a
    hypothesis string for WER, a ``utt_id`` for speaker similarity — so it is
    named by column rather than assumed.

    Returns the surviving rows and their reference values, aligned.  Rows whose
    group has no reference are dropped and counted aloud.

    Raises
    ------
    ValueError
        If the reference condition appears more than once in a group. A
        pseudo-reference must be unique, or the pairing is ambiguous and any
        choice would be arbitrary.
    """
    for column in (group_col, condition_col, source_col):
        if column not in frame.columns:
            raise KeyError(f"Results table has no column {column!r}.")

    references = frame.loc[
        frame[condition_col].astype(str) == str(reference_condition),
        [group_col, source_col],
    ]
    duplicated = references[group_col].duplicated()
    if duplicated.any():
        offending = sorted(references.loc[duplicated, group_col].unique())[:5]
        raise ValueError(
            f"Condition {reference_condition!r} appears more than once in group(s) "
            f"{offending}. A pseudo-reference must be unique within its group."
        )

    mapping = dict(zip(references[group_col], references[source_col]))
    resolved = frame[group_col].map(mapping)
    missing = resolved.isna()
    if missing.any():
        print(
            f"[{label}] {int(missing.sum())} of {len(frame)} rows dropped: their "
            f"group has no {reference_condition!r} row to serve as the reference."
        )
        frame = frame[~missing]
        resolved = resolved[~missing]
    return frame, resolved


def to_wide(
    results: pd.DataFrame,
    value_columns: list[str],
    group_col: str = "group_id",
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Pivot the long results table so one row holds all conditions of a group.

    Output columns are ``<condition>.<value column>``, plus ``group_col``.  This
    is the shape every paired comparison wants, and building it from
    ``group_id`` rather than from filenames is why the manifest carries that
    column at all.

    Groups whose conditions are not unique raise, because a silently averaged
    duplicate would corrupt every pair built from it.
    """
    missing = [c for c in value_columns if c not in results.columns]
    if missing:
        raise KeyError(f"Results table has no column(s) {missing}.")

    duplicated = results.duplicated(subset=[group_col, condition_col])
    if duplicated.any():
        offending = results.loc[duplicated, [group_col, condition_col]]
        raise ValueError(
            f"{len(offending)} duplicate ({group_col}, {condition_col}) pairs, e.g. "
            f"{offending.head(3).to_dict(orient='records')}. Each condition may "
            "appear at most once per group."
        )

    wide = results.pivot(
        index=group_col, columns=condition_col, values=value_columns
    )
    # pivot gives a (value, condition) MultiIndex; flip it so the condition
    # leads, which is how the comparisons below name their inputs.
    wide.columns = [f"{condition}.{value}" for value, condition in wide.columns]
    return wide.reset_index()


def recovery_rate(
    source: np.ndarray | pd.Series,
    converted: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
    min_gap: float = DEFAULT_MIN_GAP,
) -> np.ndarray:
    """Fraction of the real source→target change that the conversion reproduced.

        R = (converted - source) / (target - source)

    ``R = 1`` reaches the target, ``R = 0`` changes nothing, ``R < 0`` moves the
    wrong way, ``R > 1`` overshoots.

    Comparing the conversion against the *source* alone — the intuitive contrast
    — cannot distinguish a correct change from a wrong one, because it has no
    third point saying where the measure should have landed.  That is what
    ``target`` supplies: real speech from the same speaker at the target
    condition.

    Pairs whose references differ by less than ``min_gap`` yield ``nan``: when
    real speech barely moves between the two conditions there is nothing to
    recover, and the ratio is dominated by measurement noise in the denominator.
    Set ``min_gap`` in the unit of the measure being compared.

    Note this wants a measure on a **linear coordinate** — dB, Hz, semitones.
    A cosine similarity is not one, and the analogous rescaling for it lives in
    :func:`speech_eval.compare.speaker.calibrated_similarity`.
    """
    source = np.asarray(source, dtype=float)
    converted = np.asarray(converted, dtype=float)
    target = np.asarray(target, dtype=float)

    denominator = target - source
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = (converted - source) / denominator
    return np.where(np.abs(denominator) < min_gap, np.nan, rate)


def slope_vs_target(
    results: pd.DataFrame,
    value_col: str,
    target_col: str,
    group_col: str | None = "subject_id",
    min_points: int = 3,
) -> pd.DataFrame:
    """Per-group least-squares slope of ``value_col`` against ``target_col``.

    Fit per speaker rather than pooled: speakers differ in mean F0, mean tilt
    and mean level, so a pooled fit measures between-speaker variation as much
    as the within-speaker effect the model is supposed to reproduce.  Run it
    once on real speech and once on converted speech, then compare the two slope
    distributions — that comparison is the unpaired form of
    :func:`recovery_rate`.

    Returns one row per group with ``slope``, ``intercept``, ``r`` (Pearson
    correlation, a fit-quality check that costs nothing) and ``n``.  Groups with
    fewer than ``min_points`` usable points, or with no variation in
    ``target_col``, are dropped — a slope is undefined there.

    ``group_col=None`` fits a single pooled slope, reported under group
    ``"all"``.
    """
    for column in (value_col, target_col):
        if column not in results.columns:
            raise KeyError(f"Results table has no column {column!r}.")

    frame = results.copy()
    if group_col is None:
        frame["__group"] = "all"
        group_col = "__group"
    elif group_col not in frame.columns:
        raise KeyError(f"Results table has no grouping column {group_col!r}.")

    rows = []
    for name, part in frame.groupby(group_col, dropna=True):
        usable = part[[value_col, target_col]].dropna()
        x = usable[target_col].to_numpy(dtype=float)
        y = usable[value_col].to_numpy(dtype=float)
        if len(usable) < min_points or np.ptp(x) == 0:
            continue
        slope, intercept = np.polyfit(x, y, deg=1)
        r = float(np.corrcoef(x, y)[0, 1]) if np.ptp(y) > 0 else np.nan
        rows.append({
            group_col: name,
            "slope": float(slope),
            "intercept": float(intercept),
            "r": r,
            "n": int(len(usable)),
        })
    return pd.DataFrame(rows, columns=[group_col, "slope", "intercept", "r", "n"])


def cluster_bootstrap_ci(
    values: np.ndarray | pd.Series,
    clusters: np.ndarray | pd.Series,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> dict[str, float]:
    """Confidence interval for a statistic, resampling **clusters**, not values.

    Utterances from one speaker are not independent observations: a corpus of
    50 speakers carries roughly 50 independent units regardless of how many
    segments were cut from them.  Bootstrapping over utterances would therefore
    produce an interval several times too narrow and turn speaker-level noise
    into apparent significance.  Resampling whole speakers with replacement
    keeps the interval honest.

    Returns the point estimate, the percentile interval, and the number of
    clusters — report that count, since it, not the utterance count, is the
    effective sample size.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    keep = ~np.isnan(values)
    values, clusters = values[keep], clusters[keep]
    if values.size == 0:
        raise ValueError("No non-nan values to bootstrap.")

    unique = np.unique(clusters)
    by_cluster = [values[clusters == c] for c in unique]

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.integers(0, len(unique), size=len(unique))
        estimates[i] = statistic(np.concatenate([by_cluster[j] for j in draw]))

    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "estimate": float(statistic(values)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_clusters": int(len(unique)),
        "n": int(values.size),
    }
