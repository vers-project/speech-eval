"""Word error rate, from the hypotheses the ASR metrics stored.

WER lives here rather than in ``speech_eval.metrics.asr`` because it needs two
texts, and the ASR metrics store only one each.  Normalisation, the choice of
reference, and the aggregation are therefore all decided after the GPU work is
finished, and can be changed without repeating it.

Two reference modes:

``reference_column="text"``
    Ground truth, where the corpus has it.  ``text`` must be in the runner's
    ``carry_columns`` to reach the results table.
``reference_condition="real"``
    Pseudo-reference: score every condition against the named condition's own
    hypothesis, from the same backend.  Note what this measures — *divergence
    from the original decode*, not intelligibility.  It reads 0 when the
    recogniser makes the identical error on both, and it is 0 by construction
    for the reference condition itself.

The headline number is then a difference of two pooled WERs::

    errors = transcription_errors(results, "asr_whisper.hypothesis",
                                  reference_column="text")
    pooled = pooled_wer(errors, by="condition").set_index("condition")
    delta  = pooled.loc["converted", "wer"] - pooled.loc["real_src", "wer"]

computed once per backend.  The gap between the autoregressive and the CTC
delta is the intelligibility evidence; see the ``asr`` module docstring.
"""
from __future__ import annotations

import re
import string
from typing import Callable

import numpy as np
import pandas as pd

from speech_eval.compare.core import (
    exclude_flagged_rows,
    resolve_reference_condition,
    status_column_for,
)

#: Whose tokenizer supplies the normaliser's British-American spelling map.  Any
#: Whisper checkpoint carries the same one; this is simply the one already
#: downloaded for ``asr_whisper``.
DEFAULT_NORMALIZER_MODEL_ID = "openai/whisper-large-v3"

#: Columns produced per utterance by :func:`transcription_errors`, and summed by
#: :func:`pooled_wer`.
ERROR_COUNT_COLUMNS = ["n_sub", "n_del", "n_ins", "n_err", "n_ref"]


def basic_normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace.  Nothing else.

    The dependency-free fallback, so that results can be read on a machine
    without the ``asr`` extra.  It does **not** expand numbers or contractions,
    so a ground-truth "20" scored against a hypothesis "twenty" counts as an
    error here and does not under :func:`build_normalizer`'s default.
    """
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def build_normalizer(
    kind: str = "whisper_en", model_id: str = DEFAULT_NORMALIZER_MODEL_ID
) -> Callable[[str], str]:
    """Return the text normaliser applied to both sides of every comparison.

    ``"whisper_en"`` is Whisper's own ``EnglishTextNormalizer`` — Appendix C of
    the Whisper paper, and the normaliser every published Whisper WER is
    measured through.  Three of its behaviours are worth knowing before reading
    a number produced with it:

    * It **strips fillers**, on ``\\b(hmm|mm|mhm|mmm|uh|um)\\b``.  On spontaneous
      speech that removes a real part of what was said.  Mostly welcome — it
      stops two backends disagreeing about how to transcribe a disfluency from
      counting as errors — but it does mean disfluencies are not scored at all.
    * It converts spelled-out numbers **to digits**, expands contractions, and
      applies a British-to-American spelling map.  That map lives in the Whisper
      *tokenizer*, not the model, so an offline node needs the tokenizer files
      too — see ``scripts/download_asr_models.py``.
    * It is English-only.

    ``"basic"`` is :func:`basic_normalize`, which needs no extra installed.

    Applied identically to both references and hypotheses and to both backends,
    which is what keeps the AR/CTC contrast fair despite one emitting cased,
    punctuated text and the other bare upper case.
    """
    if callable(kind):
        return kind
    if kind == "basic":
        return basic_normalize
    if kind != "whisper_en":
        raise ValueError(
            f"Unknown normalizer {kind!r}. Available: 'whisper_en', 'basic', or a "
            "callable taking and returning a string."
        )

    try:
        from transformers import WhisperTokenizer
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    except ImportError as e:
        raise ImportError(
            "transformers is required for the 'whisper_en' normalizer. Install it "
            "with:  uv sync --extra asr   (or pass normalizer='basic')"
        ) from e

    tokenizer = WhisperTokenizer.from_pretrained(model_id)
    # Located by attribute so a rename between transformers versions surfaces
    # here rather than as a silently weaker normaliser.
    mapping = getattr(tokenizer, "english_spelling_normalizer", None)
    if mapping is None:
        raise RuntimeError(
            f"{model_id}'s tokenizer exposes no english_spelling_normalizer, so the "
            "British-American spelling map cannot be built. Pass normalizer='basic' "
            "to proceed without it."
        )
    return EnglishTextNormalizer(mapping)


def word_errors(reference: str, hypothesis: str) -> dict[str, float]:
    """Substitution, deletion and insertion counts for one pair of texts.

    Returns the raw counts rather than a rate, because the corpus number is
    ``sum(n_err) / sum(n_ref)`` and cannot be recovered from per-utterance rates
    — see :func:`pooled_wer`.  ``wer`` is included for inspecting single rows.

    An empty reference yields ``n_ref = 0`` and a NaN rate, with the hypothesis's
    words counted as insertions: undefined per utterance, but still correct
    arithmetic once pooled.
    """
    reference_words = str(reference).split()
    hypothesis_words = str(hypothesis).split()

    if not reference_words:
        n_ins = len(hypothesis_words)
        return {"n_sub": 0, "n_del": 0, "n_ins": n_ins, "n_err": n_ins, "n_ref": 0,
                "wer": float("nan")}
    if not hypothesis_words:
        n_del = len(reference_words)
        return {"n_sub": 0, "n_del": n_del, "n_ins": 0, "n_err": n_del,
                "n_ref": n_del, "wer": 1.0}

    try:
        import jiwer
    except ImportError as e:
        raise ImportError(
            "jiwer is required to compute word error rates. Install it with:  "
            "uv sync --extra asr"
        ) from e

    output = jiwer.process_words(" ".join(reference_words), " ".join(hypothesis_words))
    n_err = output.substitutions + output.deletions + output.insertions
    return {
        "n_sub": output.substitutions,
        "n_del": output.deletions,
        "n_ins": output.insertions,
        "n_err": n_err,
        "n_ref": len(reference_words),
        "wer": n_err / len(reference_words),
    }


def transcription_errors(
    results: pd.DataFrame,
    hypothesis_col: str,
    reference_column: str | None = None,
    reference_condition: str | None = None,
    normalizer: str | Callable[[str], str] = "whisper_en",
    normalizer_model_id: str = DEFAULT_NORMALIZER_MODEL_ID,
    status_col: str | None = "auto",
    group_col: str = "group_id",
    condition_col: str = "condition",
    carry: list[str] | None = None,
) -> pd.DataFrame:
    """Per-utterance error counts for one ASR column, against one reference.

    Exactly one reference must be named:

    ``reference_column``
        a column of ground-truth text, e.g. ``"text"``.
    ``reference_condition``
        a condition label; each row is scored against the hypothesis **the same
        backend** produced for that condition within the same ``group_id``.

    Rows the recogniser flagged — ``status`` other than ``"ok"`` — are excluded,
    as are rows with no reference, and both counts are printed.  Silent
    exclusion is the failure mode worth avoiding here: a manifest that lost most
    of its rows to a missing ``text`` column would otherwise look like a clean
    run on a tenth of the data.

    ``status_col="auto"`` derives the column from ``hypothesis_col``'s metric
    prefix; pass ``None`` to skip the check entirely.
    """
    if (reference_column is None) == (reference_condition is None):
        raise ValueError(
            "Name exactly one of reference_column (ground truth) or "
            "reference_condition (pseudo-reference)."
        )
    if hypothesis_col not in results.columns:
        raise KeyError(f"Results table has no column {hypothesis_col!r}.")

    frame = results.copy()
    if status_col == "auto":
        status_col = status_column_for(hypothesis_col, frame)
    frame = exclude_flagged_rows(frame, status_col, ("ok",), label=hypothesis_col)

    frame["__hypothesis"] = frame[hypothesis_col].fillna("").astype(str)

    if reference_column is not None:
        if reference_column not in frame.columns:
            raise KeyError(
                f"Results table has no column {reference_column!r}. Ground-truth "
                "scoring needs it in the runner's carry_columns."
            )
        frame["__reference"] = frame[reference_column].fillna("").astype(str)
        missing = frame["__reference"].str.strip() == ""
        if missing.any():
            print(
                f"[{hypothesis_col}] {int(missing.sum())} of {len(frame)} rows "
                f"dropped for having no {reference_column!r}."
            )
            frame = frame[~missing]
    else:
        frame, reference = resolve_reference_condition(
            frame, reference_condition, "__hypothesis",
            group_col=group_col, condition_col=condition_col, label=hypothesis_col,
        )
        frame["__reference"] = reference

    normalize = (
        normalizer if callable(normalizer)
        else build_normalizer(normalizer, normalizer_model_id)
    )

    identifying = [c for c in ("utt_id", group_col, condition_col) if c in frame.columns]
    carried = [c for c in (carry or []) if c in frame.columns and c not in identifying]

    rows = []
    for record in frame.to_dict(orient="records"):
        reference_text = normalize(record["__reference"])
        hypothesis = normalize(record["__hypothesis"])
        rows.append({
            **{c: record[c] for c in identifying + carried},
            "reference_norm": reference_text,
            "hypothesis_norm": hypothesis,
            **word_errors(reference_text, hypothesis),
        })
    return pd.DataFrame(
        rows,
        columns=identifying + carried + ["reference_norm", "hypothesis_norm",
                                         *ERROR_COUNT_COLUMNS, "wer"],
    )


def pooled_wer(
    errors: pd.DataFrame, by: str | list[str] | None = None
) -> pd.DataFrame:
    """Corpus word error rate: ``sum(n_err) / sum(n_ref)``, optionally per group.

    Pooled, **not** the mean of the per-utterance ``wer`` column.  On VAD
    segments the difference is large and systematic: a mean lets a two-word
    segment weigh as much as a thirty-word one, so it measures the short
    segments' error rate more than the corpus's, and it is not the definition
    anyone reports.  The per-utterance rate stays in ``errors`` for inspecting
    individual rows, which is what it is good for.

    ``by=None`` pools everything into one row labelled ``"all"``.
    """
    missing = [c for c in ERROR_COUNT_COLUMNS if c not in errors.columns]
    if missing:
        raise KeyError(
            f"Error table has no column(s) {missing}. Build it with "
            "transcription_errors()."
        )

    frame = errors.copy()
    if by is None:
        frame["__group"] = "all"
        keys: list[str] = ["__group"]
    else:
        keys = [by] if isinstance(by, str) else list(by)
        for key in keys:
            if key not in frame.columns:
                raise KeyError(f"Error table has no grouping column {key!r}.")

    grouped = frame.groupby(keys, dropna=False)
    pooled = grouped[ERROR_COUNT_COLUMNS].sum()
    pooled["n_utterances"] = grouped.size()
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled["wer"] = pooled["n_err"] / pooled["n_ref"].replace(0, np.nan)
    return pooled.reset_index().rename(columns={"__group": "group"})
