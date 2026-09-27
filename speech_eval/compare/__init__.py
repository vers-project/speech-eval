"""Comparisons over the per-utterance results table.

Everything here is a pure function of what the runner already wrote —
``metrics.csv`` and, for array-valued features, ``artifacts.npz``.  No audio, no
models, no GPU.  That is the point of keeping metrics per-utterance: a change of
mind about how conditions should be contrasted costs seconds, not a re-run.

Where things live
-----------------
:mod:`~speech_eval.compare.core`
    Contrasts that apply to any measure — ``recovery_rate``, ``slope_vs_target``,
    ``cluster_bootstrap_ci``, ``to_wide`` — plus the plumbing the families share:
    reading an array feature out of the archive, excluding rows a metric flagged,
    and resolving one condition against another within a group.
:mod:`~speech_eval.compare.wer`
    Word error rate from the stored ASR hypotheses.
:mod:`~speech_eval.compare.speaker`
    Speaker-identity similarity from the stored embeddings, and the calibration
    that makes a cosine mean something.

Every public name is re-exported here, so ``from speech_eval.compare import
pooled_wer`` works regardless of which submodule it came to live in.
"""
from __future__ import annotations

from speech_eval.compare.core import (
    DEFAULT_MIN_GAP,
    array_features,
    cluster_bootstrap_ci,
    exclude_flagged_rows,
    recovery_rate,
    resolve_reference_condition,
    slope_vs_target,
    status_column_for,
    to_wide,
)
from speech_eval.compare.speaker import (
    DEFAULT_SPEAKER_OK_STATUSES,
    accept_rate,
    calibrate,
    calibrated_similarity,
    cosine,
    equal_error_rate,
    paired_similarity,
    similarity_to_centroid,
    speaker_centroids,
    verification_trials,
)
from speech_eval.compare.wer import (
    DEFAULT_NORMALIZER_MODEL_ID,
    ERROR_COUNT_COLUMNS,
    basic_normalize,
    build_normalizer,
    pooled_wer,
    transcription_errors,
    word_errors,
)

__all__ = [
    # core
    "DEFAULT_MIN_GAP",
    "array_features",
    "cluster_bootstrap_ci",
    "exclude_flagged_rows",
    "recovery_rate",
    "resolve_reference_condition",
    "slope_vs_target",
    "status_column_for",
    "to_wide",
    # wer
    "DEFAULT_NORMALIZER_MODEL_ID",
    "ERROR_COUNT_COLUMNS",
    "basic_normalize",
    "build_normalizer",
    "pooled_wer",
    "transcription_errors",
    "word_errors",
    # speaker
    "DEFAULT_SPEAKER_OK_STATUSES",
    "accept_rate",
    "calibrate",
    "calibrated_similarity",
    "cosine",
    "equal_error_rate",
    "paired_similarity",
    "similarity_to_centroid",
    "speaker_centroids",
    "verification_trials",
]
