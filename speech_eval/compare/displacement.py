"""Did the transformation move the measure by the amount it was asked to?

The paired form of :func:`speech_eval.compare.core.recovery_rate`.  That function collapses
three points into a ratio; this one keeps them as a pair, because the pair is what can be
plotted, fitted and read as three separate quantities:

    requested = reference@target − reference@source
    achieved  = transformed      − baseline@source

``requested`` is how far the measure *should* have moved, taken from two real recordings —
so it describes the corpus, not the system.  ``achieved`` is how far it *did* move,
measured from the system's own starting point.

Why ``baseline`` is a parameter
-------------------------------
``recovery_rate`` fixes the achieved-side origin at the source recording.  That is right
when the system's output is comparable to its input, and wrong when everything passes
through a lossy stage first: a codec round trip, a vocoder, a resynthesis.  Then the
origin has to be *the source through that same stage*, or the stage's damage is charged to
the transformation.  ``baseline_condition=None`` restores the source as origin and reduces
exactly to ``recovery_rate``'s numerator.

Why regress rather than average
-------------------------------
A set of transformations toward different targets is not a set of samples of one quantity.
Moving up is a different task from moving down, one step is different from three, and the
extreme target is usually hardest whatever the source.  A mean over them describes none of
them.  Plotting achieved against requested keeps them apart — downward on the left, upward
on the right, identity at the origin — and the line through them carries three separately
interpretable numbers: slope (right amount?), intercept (systematic offset), residual
spread (consistency).

Nothing here is specific to one kind of transformation.  Intensity conversion, F0
conversion, accent or emotion transfer, speaking-rate transfer — anything whose corpus
holds real examples at both ends of the change — is the same three points under different
column names.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def reference_maps(
    results: pd.DataFrame,
    source_col: str,
    condition_col: str,
    source_condition: str,
    baseline_condition: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The per-source reference rows: ``(source, baseline)``, indexed by ``source_col``.

    Both are looked up by the key each row already carries rather than by rewriting
    identifiers, so an id that happens to contain the condition's name cannot break the
    join.  With ``baseline_condition=None`` the baseline *is* the source.
    """
    source = results[results[condition_col] == source_condition].set_index(source_col)
    if source.index.has_duplicates:
        duplicated = source.index[source.index.duplicated()].unique()[:5].tolist()
        raise ValueError(
            f"{source_col!r} does not identify a unique {source_condition!r} row; "
            f"duplicated for {duplicated}."
        )
    if baseline_condition is None:
        return source, source

    baseline = results[results[condition_col] == baseline_condition].set_index(source_col)
    if baseline.index.has_duplicates:
        raise ValueError(
            f"{source_col!r} does not identify a unique {baseline_condition!r} row."
        )
    return source, baseline


def displacement_table(
    results: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    source_col: str,
    target_col: str,
    key_col: str = "utt_id",
    condition_col: str = "condition",
    source_condition: str = "real",
    baseline_condition: str | None = None,
    subject_conditions: Iterable[str] | None = None,
    carry: Iterable[str] = (),
) -> pd.DataFrame:
    """One row per (transformed item, measure): requested, achieved, error.

    Parameters
    ----------
    results            : per-utterance measurements, one row per item.
    value_cols         : the measures to displace.  Columns absent from ``results`` are
                         skipped rather than raising, so one config can name a superset.
    source_col         : column naming, on every row, the key of its source reference.
    target_col         : column naming, on a transformed row, the key of the reference it
                         was aimed at.
    key_col            : the column ``target_col`` refers into.
    source_condition   : which condition is the untouched reference.
    baseline_condition : which condition is the achieved-side origin.  ``None`` uses the
                         source itself — see the module docstring.
    subject_conditions : which rows to displace.  ``None`` means everything that is
                         neither the source nor the baseline condition.
    carry              : extra columns copied onto each output row.

    Long form — one row per measure rather than one column per measure — because every
    downstream step groups by measure, and because measures with different missingness
    then do not have to share a row.
    """
    source, baseline = reference_maps(
        results, source_col, condition_col, source_condition, baseline_condition
    )
    by_key = results.set_index(key_col)

    if subject_conditions is None:
        excluded = {source_condition, baseline_condition} - {None}
        subjects = results[~results[condition_col].isin(excluded)]
    else:
        subjects = results[results[condition_col].isin(list(subject_conditions))]

    carry = [c for c in carry if c in subjects.columns]
    frames = []
    for measure in value_cols:
        if measure not in results.columns:
            continue
        source_value = subjects[source_col].map(source[measure])
        baseline_value = subjects[source_col].map(baseline[measure])
        target_value = subjects[target_col].map(by_key[measure])

        frame = pd.DataFrame({
            key_col: subjects[key_col].to_numpy(),
            "metric": measure,
            "requested": (target_value - source_value).to_numpy(),
            "achieved": (subjects[measure] - baseline_value).to_numpy(),
            "value": subjects[measure].to_numpy(),
            "value_source": source_value.to_numpy(),
            "value_target": target_value.to_numpy(),
            "value_baseline": baseline_value.to_numpy(),
        })
        for column in carry:
            frame[column] = subjects[column].to_numpy()
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    # Signed, and it stays signed: folding to an absolute value merges "moved too little"
    # with "moved too far", which usually have different causes.
    table["error"] = table["achieved"] - table["requested"]
    return table


def baseline_displacement_table(
    results: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    source_col: str,
    target_col: str,
    key_col: str = "utt_id",
    condition_col: str = "condition",
    source_condition: str = "real",
    baseline_condition: str = "codec",
    subject_conditions: Iterable[str] | None = None,
    carry: Iterable[str] = (),
) -> pd.DataFrame:
    """The same requested-vs-achieved table, with the lossy stage as the subject.

    A transformation is asked to move a measure by ``requested``.  Before it is
    asked anything, part of that movement may already be unavailable: if the
    encoder discards the cue, or the decoder cannot render it, then no
    transformation living between them can produce it.  This fits the same
    regression on the stage alone::

        requested = reference@target − reference@source     (unchanged)
        achieved  = baseline@target  − baseline@source

    so its slope is an upper bound on the transformation's, on the same axes and
    in the same units.  A slope of 1 means the round trip preserves the whole
    continuum and the transformation is free to realise all of it; 0.5 means half
    of it is gone before training begins, and a transformation scoring 0.5 there
    has in fact realised everything available to it.

    Reading a transformation's slope against unity rather than against this is the
    commonest way to mistake a representation's limit for a model's failure.

    The subject rows only supply the pairing: what is measured is the baseline
    condition at each end, looked up through the same reference map
    :func:`displacement_table` uses.  ``subject_conditions`` therefore defaults to
    whatever is neither the source nor the baseline — the transformed rows — so
    that the two tables cover exactly the same (source, target) pairs and their
    slopes are comparable pair by pair.
    """
    source, baseline = reference_maps(
        results, source_col, condition_col, source_condition, baseline_condition
    )
    if subject_conditions is None:
        excluded = {source_condition, baseline_condition} - {None}
        subjects = results[~results[condition_col].isin(excluded)]
    else:
        subjects = results[results[condition_col].isin(list(subject_conditions))]

    carry = [c for c in carry if c in subjects.columns]
    frames = []
    for measure in value_cols:
        if measure not in results.columns:
            continue
        source_value = subjects[source_col].map(source[measure])
        target_value = subjects[target_col].map(source[measure])
        # The lossy stage at each end.  A converted row's target_col names the
        # *reference* recording at the target level, and a reference row's
        # source_col is its own key, so the baseline map -- which is indexed by
        # source_col -- resolves the stage's output for that same source.
        base_source = subjects[source_col].map(baseline[measure])
        base_target = subjects[target_col].map(baseline[measure])

        frame = pd.DataFrame({
            key_col: subjects[key_col].to_numpy(),
            "metric": measure,
            "requested": (target_value - source_value).to_numpy(),
            "achieved": (base_target - base_source).to_numpy(),
            "value": base_target.to_numpy(),
            "value_source": source_value.to_numpy(),
            "value_target": target_value.to_numpy(),
            "value_baseline": base_source.to_numpy(),
        })
        for column in carry:
            frame[column] = subjects[column].to_numpy()
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    table["error"] = table["achieved"] - table["requested"]
    return table


def per_group_slopes(
    table: pd.DataFrame,
    group_col: str = "subject_id",
    min_points: int = 3,
) -> pd.DataFrame:
    """Least-squares achieved-vs-requested fit, per measure and per group.

    Per group rather than pooled: groups (speakers, usually) differ in their baseline
    level of most measures, so a pooled fit measures between-group variation as much as
    the within-group effect the system is supposed to produce.
    """
    from speech_eval.compare.core import slope_vs_target

    rows = []
    for measure, part in table.groupby("metric", sort=False):
        fitted = slope_vs_target(
            part, value_col="achieved", target_col="requested",
            group_col=group_col, min_points=min_points,
        )
        if fitted.empty:
            continue
        fitted.insert(0, "metric", measure)
        rows.append(fitted)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarise(
    slopes: pd.DataFrame,
    table: pd.DataFrame,
    group_col: str = "subject_id",
    n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Headline per measure: mean slope with a cluster-bootstrap interval, and spread.

    The interval resamples **groups**, not rows.  Items from one speaker are not
    independent evidence, and bootstrapping over them yields an interval several times too
    narrow — turning group-level noise into apparent significance.  With few groups the
    interval is wide, and that width is the honest one.
    """
    from speech_eval.compare.core import cluster_bootstrap_ci

    rows = []
    for measure, part in slopes.groupby("metric", sort=False):
        interval = cluster_bootstrap_ci(
            part["slope"].to_numpy(), part[group_col].to_numpy(),
            statistic=np.mean, n_boot=n_boot, seed=seed,
        )
        errors = table.loc[table["metric"] == measure, "error"]
        rows.append({
            "metric": measure,
            "n_groups": int(part[group_col].nunique()),
            "n_items": int((table["metric"] == measure).sum()),
            "slope_mean": float(part["slope"].mean()),
            "slope_lo": float(interval["ci_low"]),
            "slope_hi": float(interval["ci_high"]),
            "slope_sd_across_groups": float(part["slope"].std()),
            "intercept_mean": float(part["intercept"].mean()),
            # A fixable offset and a consistency figure: different problems, reported apart.
            "error_mean": float(errors.mean()),
            "error_sd": float(errors.std()),
        })
    return pd.DataFrame(rows)


def cell_means(
    table: pd.DataFrame,
    row_col: str,
    col_col: str,
    value: str = "error",
) -> pd.DataFrame:
    """Mean of ``value`` per (measure, row, col) — the source × target grid.

    The view a scatter blurs: "transforming *to* the extreme target fails" is a bright
    band here and an indistinct smear there.
    """
    grouped = (
        table.groupby(["metric", row_col, col_col])[value]
        .agg(["mean", "std", "size"])
        .reset_index()
        .rename(columns={"size": "n"})
    )
    return grouped
