"""Speaker identity: similarity between stored embeddings, and its calibration.

A cosine is not a result
-----------------------
:mod:`speech_eval.metrics.speaker` writes one embedding per utterance and no
similarity at all.  Everything that needs two utterances is here — but a raw
cosine is still not reportable on its own.  Same-speaker ECAPA pairs land
anywhere between roughly 0.5 and 0.85 depending on corpus, microphone, language
and segment length, so "0.62" says nothing until the corpus itself has been
asked what its own same-speaker and different-speaker distances look like.

That is the job of :func:`verification_trials` and :func:`calibrate`.  From the
**real** audio alone they build two distributions:

*target*
    two utterances of the same speaker, different content;
*nontarget*
    two utterances of different speakers.

Those give an equal error rate, its threshold, and the two means that
:func:`calibrated_similarity` rescales onto — 0 is "as similar as two different
speakers", 1 is "as similar as two real recordings of this speaker".  A number
on that axis is comparable across corpora and across embedders; a bare cosine is
not.

The two questions this answers
------------------------------
Both are :func:`paired_similarity` with a different reference.  Given a manifest
where one ``group_id`` holds several versions of the same utterance::

    group_id            condition          what the file is
    avid_s07_sent12     real_normal        original recording, normal effort
    avid_s07_sent12     real_loud          same speaker and sentence, high effort
    avid_s07_sent12     converted_loud     real_normal driven to the loud target
    avid_s07_sent12     codec_roundtrip    real_normal through encode/decode only

*Does vocal effort by itself move a speaker embedding?*  Score everything
against ``real_normal`` and read the ``real_loud`` rows.  This is a **baseline**,
and it has to be measured before any conversion is judged: whatever identity
drift effort alone causes is not the converter's fault.

*Is generated speech at a given level as much "this speaker" as real speech at
that level?*  Score against ``real_loud`` and compare the ``converted_loud``
rows with what ``codec_roundtrip`` scores — the round trip is the ceiling, since
it is the same performance through the same synthesis path with no conversion
applied.

Why target trials exclude same-group pairs
------------------------------------------
Two recordings of the *same content* are unusually easy to match: they share
phonetic material, and on a paired corpus they often share a take and a channel.
Counting them as target trials inflates the target distribution and yields a
yardstick that flatters every subsequent comparison.  :func:`verification_trials`
therefore excludes them by default, so "same speaker" means *the same speaker
saying something else*.

What is deliberately not here
-----------------------------
Adaptive score normalisation (AS-Norm) and PLDA back-ends.  Both are standard in
speaker-verification systems and both would change the numbers; neither is
needed to compare conditions *within* one corpus, which is all this module is
for.  Calibration against the corpus's own distributions does that job with no
extra parameters to fit.

References are collected in ``references.bib`` at the repository root.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from speech_eval.compare.core import (
    array_features,
    exclude_flagged_rows,
    resolve_reference_condition,
    status_column_for,
)

#: Statuses whose embeddings are still usable.  Unlike an over-long ASR
#: hypothesis, which is silently truncated and therefore wrong, an embedding from
#: a short segment is merely noisier than usual — see
#: :mod:`speech_eval.metrics.speaker`.  Pass ``("ok",)`` to be strict.
DEFAULT_SPEAKER_OK_STATUSES = ("ok", "short")

#: Trial caps.  An equal error rate is already stable well below this many
#: trials, and the nontarget side grows with the square of the corpus.
DEFAULT_MAX_TRIALS = 50_000


# ---------------------------------------------------------------------------
# Cosine
# ---------------------------------------------------------------------------


def cosine(a: np.ndarray, b: np.ndarray) -> float | np.ndarray:
    """Cosine similarity, row-wise over the last axis.

    Accepts two vectors, or two equally shaped stacks of vectors.  A zero vector
    has no direction, so it yields ``nan`` rather than a division warning and a
    meaningless 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = (a * b).sum(axis=-1) / (norm_a * norm_b)
    similarity = np.where((norm_a == 0) | (norm_b == 0), np.nan, similarity)
    return float(similarity) if similarity.ndim == 0 else similarity


def _unit(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise along the last axis, leaving zero vectors as they are."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


# ---------------------------------------------------------------------------
# Assembling the usable rows
# ---------------------------------------------------------------------------


def _usable_rows(
    results: pd.DataFrame,
    artifacts: dict[str, np.ndarray],
    embedding_col: str,
    status_col: str | None,
    ok_statuses: Sequence[str],
    min_duration_s: float | None,
    label: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Rows that carry a usable embedding, plus the embeddings themselves.

    Three filters, each reported: the metric's own status flag, an optional
    duration floor, and the presence of an embedding in the archive.  The last
    is not redundant — a resumed run can leave a results row whose artifact was
    never written.
    """
    frame = results.copy()
    if "utt_id" not in frame.columns:
        raise KeyError("Results table has no 'utt_id' column.")
    frame["utt_id"] = frame["utt_id"].astype(str)

    if status_col == "auto":
        status_col = status_column_for(embedding_col, frame)
    frame = exclude_flagged_rows(frame, status_col, tuple(ok_statuses), label=label)

    if min_duration_s is not None:
        duration_col = f"{embedding_col.rsplit('.', 1)[0]}.duration_s"
        if duration_col not in frame.columns:
            raise KeyError(
                f"Results table has no column {duration_col!r}, so min_duration_s "
                "cannot be applied. Pass min_duration_s=None, or re-run the metric."
            )
        too_short = frame[duration_col].astype(float) < min_duration_s
        if too_short.any():
            print(
                f"[{label}] {int(too_short.sum())} of {len(frame)} rows dropped for "
                f"being shorter than {min_duration_s} s."
            )
            frame = frame[~too_short]

    embeddings = array_features(artifacts, embedding_col, frame["utt_id"])
    missing = ~frame["utt_id"].isin(embeddings)
    if missing.any():
        print(
            f"[{label}] {int(missing.sum())} of {len(frame)} rows dropped for having "
            f"no {embedding_col!r} in artifacts.npz."
        )
        frame = frame[~missing]

    return frame.reset_index(drop=True), embeddings


# ---------------------------------------------------------------------------
# Paired similarity — the two questions
# ---------------------------------------------------------------------------


def paired_similarity(
    results: pd.DataFrame,
    artifacts: dict[str, np.ndarray],
    embedding_col: str,
    reference_condition: str,
    group_col: str = "group_id",
    condition_col: str = "condition",
    status_col: str | None = "auto",
    ok_statuses: Sequence[str] = DEFAULT_SPEAKER_OK_STATUSES,
    min_duration_s: float | None = None,
    include_reference: bool = False,
    carry: list[str] | None = None,
) -> pd.DataFrame:
    """Cosine of every condition against one reference condition of its group.

    The speaker-identity counterpart of
    :func:`speech_eval.compare.wer.transcription_errors`' pseudo-reference mode,
    and deliberately the same shape: same status handling, same "this group has
    no reference row" report, same carried columns.

    Returns one row per scored utterance with ``cosine``, the
    ``reference_utt_id`` it was scored against, and whatever ``carry`` named.

    ``include_reference`` keeps the reference rows themselves.  Their cosine is
    1.0 by construction, so they are dropped by default and are useful only as a
    sanity check that the right embeddings were paired.

    Both sides must be comparable in **duration**: embedding similarity rises
    with utterance length, so scoring a 2-second conversion against a 10-second
    original measures length as much as identity.  For a frame-synchronous
    converter the durations match by construction; where they do not, pass
    ``min_duration_s`` and check the metric's own ``duration_s`` column.
    """
    frame, embeddings = _usable_rows(
        results, artifacts, embedding_col, status_col, ok_statuses,
        min_duration_s, label=embedding_col,
    )
    if frame.empty:
        return _empty_paired(group_col, condition_col, carry)

    frame, reference_ids = resolve_reference_condition(
        frame, reference_condition, "utt_id",
        group_col=group_col, condition_col=condition_col, label=embedding_col,
    )
    if not include_reference:
        is_reference = frame[condition_col].astype(str) == str(reference_condition)
        frame, reference_ids = frame[~is_reference], reference_ids[~is_reference]
    if frame.empty:
        return _empty_paired(group_col, condition_col, carry)

    left = np.stack([embeddings[u] for u in frame["utt_id"]])
    right = np.stack([embeddings[u] for u in reference_ids])

    identifying = [c for c in ("utt_id", group_col, condition_col) if c in frame.columns]
    carried = [c for c in (carry or []) if c in frame.columns and c not in identifying]

    output = frame[identifying + carried].copy()
    output["reference_condition"] = str(reference_condition)
    output["reference_utt_id"] = reference_ids.to_numpy()
    output["cosine"] = cosine(left, right)
    return output.reset_index(drop=True)


def _empty_paired(
    group_col: str, condition_col: str, carry: list[str] | None
) -> pd.DataFrame:
    columns = ["utt_id", group_col, condition_col, *(carry or []),
               "reference_condition", "reference_utt_id", "cosine"]
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


# ---------------------------------------------------------------------------
# Calibration: what "same speaker" looks like in this corpus
# ---------------------------------------------------------------------------


def _subsample(
    rng: np.random.Generator, pairs: list[tuple[int, int]], max_pairs: int | None
) -> list[tuple[int, int]]:
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    picked = rng.choice(len(pairs), size=max_pairs, replace=False)
    return [pairs[i] for i in sorted(picked)]


def _target_pairs(
    rng: np.random.Generator,
    by_speaker: dict[Any, np.ndarray],
    group_of: np.ndarray,
    exclude_same_group: bool,
    max_pairs: int | None,
) -> list[tuple[int, int]]:
    """Same-speaker pairs, enumerated when that is cheap and drawn when it is not.

    Enumerating is exact and is used whenever the full pair set is within a
    factor of two of what was asked for.  Beyond that the pairs are drawn
    directly, with each speaker weighted by how many pairs it contributes, which
    samples uniformly over the same space without ever building it.
    """
    weights = {s: len(ix) * (len(ix) - 1) // 2 for s, ix in by_speaker.items()
               if len(ix) >= 2}
    total = sum(weights.values())
    if total == 0:
        return []

    if max_pairs is None or total <= 2 * max_pairs:
        pairs = [
            (int(a), int(b))
            for indices in by_speaker.values()
            for a, b in combinations(sorted(indices), 2)
        ]
        if exclude_same_group:
            pairs = [p for p in pairs if group_of[p[0]] != group_of[p[1]]]
        return _subsample(rng, pairs, max_pairs)

    speakers = list(weights)
    probabilities = np.array([weights[s] for s in speakers], dtype=float)
    probabilities /= probabilities.sum()
    drawn: set[tuple[int, int]] = set()
    # Bounded so a corpus where almost every pair shares a group cannot spin
    # here forever; the shortfall is reported by the caller through n_target.
    for _ in range(100 * max_pairs + 1000):
        if len(drawn) >= max_pairs:
            break
        speaker = speakers[int(rng.choice(len(speakers), p=probabilities))]
        a, b = rng.choice(by_speaker[speaker], size=2, replace=False)
        if exclude_same_group and group_of[a] == group_of[b]:
            continue
        drawn.add((int(min(a, b)), int(max(a, b))))
    return sorted(drawn)


def _nontarget_pairs(
    rng: np.random.Generator,
    speaker_of: np.ndarray,
    max_pairs: int | None,
) -> list[tuple[int, int]]:
    """Different-speaker pairs, by the same enumerate-or-draw rule."""
    n = len(speaker_of)
    if n < 2:
        return []
    counts = pd.Series(speaker_of).value_counts().to_numpy()
    total = n * (n - 1) // 2 - int((counts * (counts - 1) // 2).sum())
    if total == 0:
        return []

    if max_pairs is None or total <= 2 * max_pairs:
        pairs = [(i, j) for i, j in combinations(range(n), 2)
                 if speaker_of[i] != speaker_of[j]]
        return _subsample(rng, pairs, max_pairs)

    drawn: set[tuple[int, int]] = set()
    for _ in range(100 * max_pairs + 1000):
        if len(drawn) >= max_pairs:
            break
        i, j = rng.choice(n, size=2, replace=False)
        if speaker_of[i] == speaker_of[j]:
            continue
        drawn.add((int(min(i, j)), int(max(i, j))))
    return sorted(drawn)


def verification_trials(
    results: pd.DataFrame,
    artifacts: dict[str, np.ndarray],
    embedding_col: str,
    speaker_col: str = "subject_id",
    conditions: Iterable[str] | None = None,
    group_col: str = "group_id",
    condition_col: str = "condition",
    exclude_same_group: bool = True,
    max_target: int | None = DEFAULT_MAX_TRIALS,
    max_nontarget: int | None = DEFAULT_MAX_TRIALS,
    status_col: str | None = "auto",
    ok_statuses: Sequence[str] = DEFAULT_SPEAKER_OK_STATUSES,
    min_duration_s: float | None = None,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Build the target/nontarget trial list this corpus will be calibrated on.

    ``conditions`` is the important argument and has no safe default: pass the
    **real** conditions only.  Calibrating on converted audio would fold the very
    artifacts under test into the yardstick, and a converter that damaged every
    speaker equally would then appear to have damaged none of them.

    ``exclude_same_group`` drops target pairs built from the same underlying
    utterance — see the module docstring.

    Returns one row per trial: ``utt_a``, ``utt_b``, the speaker, condition and
    group of each side, ``cosine``, and ``label`` (``True`` for target).  The
    trial counts are printed, since the number of *speakers* rather than the
    number of trials is what an equal error rate's precision actually rests on.
    """
    frame, embeddings = _usable_rows(
        results, artifacts, embedding_col, status_col, ok_statuses,
        min_duration_s, label=embedding_col,
    )
    if speaker_col not in frame.columns:
        raise KeyError(
            f"Results table has no speaker column {speaker_col!r}. It must be in the "
            "runner's carry_columns for identity trials to be built."
        )
    if conditions is not None:
        wanted = {str(c) for c in conditions}
        frame = frame[frame[condition_col].astype(str).isin(wanted)]
    frame = frame[frame[speaker_col].notna()].reset_index(drop=True)

    if len(frame) < 2:
        raise ValueError(
            f"Only {len(frame)} usable utterance(s) after filtering; a trial list "
            "needs at least two."
        )

    speaker_of = frame[speaker_col].to_numpy()
    group_of = (
        frame[group_col].to_numpy() if group_col in frame.columns
        else np.arange(len(frame))
    )
    by_speaker = {
        speaker: np.flatnonzero(speaker_of == speaker)
        for speaker in pd.unique(speaker_of)
    }

    rng = np.random.default_rng(seed)
    targets = _target_pairs(rng, by_speaker, group_of, exclude_same_group, max_target)
    nontargets = _nontarget_pairs(rng, speaker_of, max_nontarget)
    if not targets:
        raise ValueError(
            "No target trials could be built: no speaker has two usable utterances "
            f"at different {group_col!r} values. Widen `conditions`, or pass "
            "exclude_same_group=False if the corpus records each utterance once."
        )
    print(
        f"[{embedding_col}] {len(targets)} target and {len(nontargets)} nontarget "
        f"trials over {len(by_speaker)} speakers and {len(frame)} utterances."
    )

    matrix = _unit(np.stack([embeddings[u] for u in frame["utt_id"]]))
    pairs = targets + nontargets
    left = np.array([p[0] for p in pairs])
    right = np.array([p[1] for p in pairs])

    return pd.DataFrame({
        "utt_a": frame["utt_id"].to_numpy()[left],
        "utt_b": frame["utt_id"].to_numpy()[right],
        "speaker_a": speaker_of[left],
        "speaker_b": speaker_of[right],
        "condition_a": frame[condition_col].to_numpy()[left],
        "condition_b": frame[condition_col].to_numpy()[right],
        "group_a": group_of[left],
        "group_b": group_of[right],
        "cosine": (matrix[left] * matrix[right]).sum(axis=-1),
        "label": np.concatenate([
            np.ones(len(targets), dtype=bool), np.zeros(len(nontargets), dtype=bool),
        ]),
    })


def equal_error_rate(
    scores: np.ndarray | pd.Series, labels: np.ndarray | pd.Series
) -> dict[str, float]:
    """Equal error rate and the threshold that achieves it.

    A trial is accepted when its score is **at or above** the threshold, so the
    false-acceptance rate is the fraction of nontargets at or above it and the
    false-rejection rate the fraction of targets below it.  The reported EER is
    the mean of the two at the candidate threshold where they are closest — the
    discrete minimiser over the observed scores, not an interpolated ROC
    crossing, so it is exactly reproducible from the trial list.

    Returns ``eer``, ``threshold``, the two rates there, and the trial counts.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    keep = ~np.isnan(scores)
    scores, labels = scores[keep], labels[keep]

    target = np.sort(scores[labels])
    nontarget = np.sort(scores[~labels])
    if target.size == 0 or nontarget.size == 0:
        raise ValueError(
            f"Need both target and nontarget trials to compute an EER; got "
            f"{target.size} target and {nontarget.size} nontarget."
        )

    thresholds = np.unique(scores)
    frr = np.searchsorted(target, thresholds, side="left") / target.size
    far = (nontarget.size - np.searchsorted(nontarget, thresholds, side="left")) \
        / nontarget.size

    best = int(np.argmin(np.abs(far - frr)))
    return {
        "eer": float((far[best] + frr[best]) / 2),
        "threshold": float(thresholds[best]),
        "far": float(far[best]),
        "frr": float(frr[best]),
        "n_target": int(target.size),
        "n_nontarget": int(nontarget.size),
    }


def calibrate(trials: pd.DataFrame) -> dict[str, float]:
    """Summarise a trial list into the yardstick every other number is read on.

    Returns the two class means and standard deviations, ``d_prime`` (their
    separation in pooled standard deviations), and everything
    :func:`equal_error_rate` reports.  Feed the whole dict to
    :func:`calibrated_similarity` and :func:`accept_rate`.

    Report ``n_speakers`` alongside the EER when there is one: with 12 speakers
    a corpus supplies 12 independent units no matter how many trials were built
    from them, and the EER's precision follows the speakers.
    """
    for column in ("cosine", "label"):
        if column not in trials.columns:
            raise KeyError(
                f"Trial table has no column {column!r}. Build it with "
                "verification_trials()."
            )
    label = trials["label"].astype(bool)
    target = trials.loc[label, "cosine"].to_numpy(dtype=float)
    nontarget = trials.loc[~label, "cosine"].to_numpy(dtype=float)

    summary = equal_error_rate(trials["cosine"], label)
    mu_target, mu_nontarget = float(np.nanmean(target)), float(np.nanmean(nontarget))
    sd_target, sd_nontarget = float(np.nanstd(target)), float(np.nanstd(nontarget))
    pooled = np.sqrt((sd_target**2 + sd_nontarget**2) / 2)

    speakers = pd.unique(
        np.concatenate([trials["speaker_a"], trials["speaker_b"]])
    ) if "speaker_a" in trials.columns else []

    return {
        **summary,
        "mu_target": mu_target,
        "sd_target": sd_target,
        "mu_nontarget": mu_nontarget,
        "sd_nontarget": sd_nontarget,
        "d_prime": float((mu_target - mu_nontarget) / pooled) if pooled > 0 else np.nan,
        "n_speakers": int(len(speakers)),
    }


def calibrated_similarity(
    similarity: np.ndarray | pd.Series, calibration: dict[str, float]
) -> np.ndarray:
    """Put a cosine on the corpus's own same-speaker / different-speaker axis.

        c = (cosine - mu_nontarget) / (mu_target - mu_nontarget)

    ``c = 1`` is as similar as two real recordings of one speaker; ``c = 0`` is
    as similar as two different speakers.  Values outside [0, 1] are ordinary
    and meaningful — trials do overlap both means.

    This is **not** a probability and not bounded, and it is only comparable
    between two runs whose ``calibration`` came from the same corpus and the same
    embedder.  It is the similarity analogue of
    :func:`speech_eval.compare.core.recovery_rate`, which cannot be used here
    because a cosine is not a linear coordinate.
    """
    for key in ("mu_target", "mu_nontarget"):
        if key not in calibration:
            raise KeyError(
                f"Calibration has no {key!r}. Build it with calibrate()."
            )
    span = calibration["mu_target"] - calibration["mu_nontarget"]
    if not np.isfinite(span) or abs(span) < 1e-12:
        raise ValueError(
            "Calibration's target and nontarget means coincide, so there is no axis "
            "to project onto. The embedder is not separating speakers on this corpus."
        )
    return (np.asarray(similarity, dtype=float) - calibration["mu_nontarget"]) / span


def accept_rate(
    similarity: np.ndarray | pd.Series, threshold: float
) -> dict[str, float]:
    """Fraction of scores at or above a threshold — usually the corpus EER's.

    The one number a reader of a verification table expects: *"converted speech
    is accepted as the same speaker 89% of the time, at a threshold where this
    corpus's own real speech errs 2.1% either way."*  Quote the EER with it, or
    the rate is unanchored.
    """
    values = np.asarray(similarity, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"accept_rate": float("nan"), "n": 0, "threshold": float(threshold)}
    return {
        "accept_rate": float((values >= threshold).mean()),
        "n": int(values.size),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# Centroid enrolment — the unpaired route
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Enrolment:
    """Per-speaker centroids and the utterances behind them.

    ``centroids`` is what most callers want.  ``sums`` and ``members`` exist so
    that a speaker's centroid can be rebuilt without one of its own members, in
    constant time — see :meth:`centroid_excluding`.
    """

    centroids: dict[Any, np.ndarray]
    sums: dict[Any, np.ndarray] = field(repr=False, default_factory=dict)
    members: dict[Any, list[str]] = field(repr=False, default_factory=dict)

    def __len__(self) -> int:
        return len(self.centroids)

    def centroid_excluding(
        self, speaker: Any, utt_id: str, embeddings: dict[str, np.ndarray]
    ) -> np.ndarray | None:
        """This speaker's centroid without ``utt_id``, or ``None`` if nothing is left.

        An utterance that contributed to a centroid is closer to it for that
        reason alone.  Leaving it out is what keeps a real utterance and a
        converted one scored against the *same* enrolment.
        """
        if speaker not in self.sums:
            return None
        members = self.members.get(speaker, [])
        if utt_id not in members:
            return self.centroids[speaker]
        if len(members) < 2:
            return None
        remainder = self.sums[speaker] - _unit(np.asarray(embeddings[utt_id], float))
        norm = np.linalg.norm(remainder)
        return None if norm == 0 else remainder / norm


def speaker_centroids(
    results: pd.DataFrame,
    artifacts: dict[str, np.ndarray],
    embedding_col: str,
    speaker_col: str = "subject_id",
    conditions: Iterable[str] | None = None,
    condition_col: str = "condition",
    min_utterances: int = 2,
    status_col: str | None = "auto",
    ok_statuses: Sequence[str] = DEFAULT_SPEAKER_OK_STATUSES,
    min_duration_s: float | None = None,
) -> Enrolment:
    """Enrol each speaker as the mean of their embeddings, re-normalised.

    ``conditions`` should name the **real** conditions, for the same reason
    :func:`verification_trials` needs it: enrol from converted audio and the
    reference contains the artifacts under test.

    Averaging several utterances is standard verification practice and buys two
    things here.  It cuts the variance a single noisy reference utterance
    injects, and — the reason it earns its place in this suite — it needs no
    content matching, so the same analysis runs on a spontaneous corpus where no
    two recordings share a sentence.  It is to :func:`paired_similarity` what
    :func:`speech_eval.compare.core.slope_vs_target` is to
    :func:`speech_eval.compare.core.recovery_rate`.

    Speakers with fewer than ``min_utterances`` are left out and reported.
    """
    frame, embeddings = _usable_rows(
        results, artifacts, embedding_col, status_col, ok_statuses,
        min_duration_s, label=embedding_col,
    )
    if speaker_col not in frame.columns:
        raise KeyError(f"Results table has no speaker column {speaker_col!r}.")
    if conditions is not None:
        wanted = {str(c) for c in conditions}
        frame = frame[frame[condition_col].astype(str).isin(wanted)]
    frame = frame[frame[speaker_col].notna()]

    centroids: dict[Any, np.ndarray] = {}
    sums: dict[Any, np.ndarray] = {}
    members: dict[Any, list[str]] = {}
    skipped = 0
    for speaker, part in frame.groupby(speaker_col, dropna=True):
        utt_ids = part["utt_id"].tolist()
        if len(utt_ids) < min_utterances:
            skipped += 1
            continue
        total = _unit(np.stack([embeddings[u] for u in utt_ids])).sum(axis=0)
        norm = np.linalg.norm(total)
        if norm == 0:
            skipped += 1
            continue
        centroids[speaker] = total / norm
        sums[speaker] = total
        members[speaker] = utt_ids

    if skipped:
        print(
            f"[{embedding_col}] {skipped} speaker(s) not enrolled: fewer than "
            f"{min_utterances} usable utterances."
        )
    if not centroids:
        raise ValueError(
            "No speaker could be enrolled. Check `conditions` and `speaker_col`, or "
            "lower min_utterances."
        )
    return Enrolment(centroids=centroids, sums=sums, members=members)


def similarity_to_centroid(
    results: pd.DataFrame,
    artifacts: dict[str, np.ndarray],
    embedding_col: str,
    enrolment: Enrolment,
    speaker_col: str = "subject_id",
    condition_col: str = "condition",
    group_col: str = "group_id",
    exclude_self: bool = True,
    status_col: str | None = "auto",
    ok_statuses: Sequence[str] = DEFAULT_SPEAKER_OK_STATUSES,
    min_duration_s: float | None = None,
    carry: list[str] | None = None,
) -> pd.DataFrame:
    """Score every utterance against its own speaker's enrolled centroid.

    ``exclude_self`` rebuilds the centroid without the utterance being scored
    whenever that utterance helped build it.  Leave it on: real speech would
    otherwise be compared with a reference it contributed to and converted speech
    with one it did not, which biases the whole comparison toward real by an
    amount that grows as the enrolment shrinks.

    Returns one row per utterance with ``cosine`` and ``n_enrolment``, the number
    of utterances actually behind the reference it was scored against.  Rows
    whose speaker was never enrolled are dropped and counted.
    """
    frame, embeddings = _usable_rows(
        results, artifacts, embedding_col, status_col, ok_statuses,
        min_duration_s, label=embedding_col,
    )
    if speaker_col not in frame.columns:
        raise KeyError(f"Results table has no speaker column {speaker_col!r}.")

    unenrolled = ~frame[speaker_col].isin(enrolment.centroids)
    if unenrolled.any():
        print(
            f"[{embedding_col}] {int(unenrolled.sum())} of {len(frame)} rows dropped: "
            "their speaker has no enrolment."
        )
        frame = frame[~unenrolled]

    identifying = [c for c in ("utt_id", group_col, condition_col, speaker_col)
                   if c in frame.columns]
    carried = [c for c in (carry or []) if c in frame.columns and c not in identifying]

    similarities: list[float] = []
    sizes: list[int] = []
    for utt_id, speaker in zip(frame["utt_id"], frame[speaker_col]):
        members = enrolment.members.get(speaker, [])
        contributed = exclude_self and utt_id in members
        reference = (
            enrolment.centroid_excluding(speaker, utt_id, embeddings) if contributed
            else enrolment.centroids[speaker]
        )
        if reference is None:
            similarities.append(float("nan"))
            sizes.append(0)
            continue
        similarities.append(float(cosine(embeddings[utt_id], reference)))
        sizes.append(len(members) - 1 if contributed else len(members))

    output = frame[identifying + carried].copy()
    output["cosine"] = similarities
    output["n_enrolment"] = sizes
    return output.reset_index(drop=True)
