"""Tests for the predicted-naturalness metric.

The contract worth protecting here is not the score — that belongs to UTMOS —
but the *shape of the calls*: one call per utterance, each on unpadded audio.
``UTMOS22Strong.forward`` takes no length mask and rates silence higher than
noise, so a padded batch would silently shift every score except the longest
utterance's (measured: -0.36 MOS for one second padded to two).  A stub model
pins that without downloading 392 MB.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import speech_eval.metrics  # noqa: F401  -- registers the metric types
from speech_eval.core import Utterance, UtteranceBatch, build_metric
from speech_eval.metrics.naturalness import MIN_SAMPLES, MOS_LEVEL_NORM_DBFS

SAMPLE_RATE = 16_000


class RecordingStub:
    """Stands in for the hub model, recording the shape of every call."""

    def __init__(self, value: float = 3.5):
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.value = value

    def __call__(self, wave: torch.Tensor, sr: int) -> torch.Tensor:
        self.calls.append((tuple(wave.shape), sr))
        return torch.full((wave.shape[0],), self.value)


def stubbed(value: float = 3.5):
    metric = build_metric({"type": "mos_utmos"})
    stub = RecordingStub(value)
    metric._model = stub
    metric._loaded = True
    metric._device = torch.device("cpu")
    return metric, stub


def utterance(utt_id: str, n_samples: int, seed: int = 0) -> Utterance:
    rng = np.random.default_rng(seed)
    wav = torch.from_numpy((0.05 * rng.standard_normal(n_samples)).astype("float32"))
    return Utterance(utt_id, wav.unsqueeze(0), SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Registration and declared requirements
# ---------------------------------------------------------------------------


def test_registered_under_its_config_name():
    metric = build_metric({"type": "mos_utmos"})
    assert metric.name == "mos_utmos"


def test_level_normalisation_is_declared_and_matches_the_asr_group():
    """Conversion changes level by design, so an unnormalised MOS measures gain.

    The value equalling ASR's is what puts this metric in an existing
    requirements group, so it costs no extra preparation pass.
    """
    from speech_eval.metrics.asr import ASR_LEVEL_NORM_DBFS

    metric = build_metric({"type": "mos_utmos"})
    assert metric.requirements.level_norm_dbfs == MOS_LEVEL_NORM_DBFS
    assert MOS_LEVEL_NORM_DBFS == ASR_LEVEL_NORM_DBFS


def test_requirements_fix_the_rate_rather_than_leaving_it_native():
    metric = build_metric({"type": "mos_utmos"})
    assert metric.requirements.sample_rate == SAMPLE_RATE
    assert metric.requirements.mono is True


def test_level_normalisation_can_be_disabled_explicitly():
    metric = build_metric({"type": "mos_utmos", "level_norm_dbfs": None})
    assert metric.requirements.level_norm_dbfs is None


def test_compute_before_load_raises_rather_than_returning_nonsense():
    metric = build_metric({"type": "mos_utmos"})
    with pytest.raises(RuntimeError, match="load"):
        metric.compute(UtteranceBatch(utterances=[utterance("a", SAMPLE_RATE)]))


# ---------------------------------------------------------------------------
# The no-padding contract
# ---------------------------------------------------------------------------


def test_one_call_per_utterance_never_a_padded_batch():
    metric, stub = stubbed()
    batch = UtteranceBatch(utterances=[
        utterance("a", SAMPLE_RATE),
        utterance("b", 2 * SAMPLE_RATE),
        utterance("c", SAMPLE_RATE // 2),
    ])
    metric.compute(batch)
    assert len(stub.calls) == 3, "batched into one call; padding would corrupt scores"
    widths = [shape[-1] for shape, _ in stub.calls]
    assert widths == [SAMPLE_RATE, 2 * SAMPLE_RATE, SAMPLE_RATE // 2], (
        "every call must carry the utterance's own length, unpadded"
    )
    assert all(shape[0] == 1 for shape, _ in stub.calls)


def test_the_declared_sample_rate_is_passed_through():
    metric, stub = stubbed()
    metric.compute(UtteranceBatch(utterances=[utterance("a", SAMPLE_RATE)]))
    assert stub.calls[0][1] == SAMPLE_RATE


def test_results_are_in_batch_order_one_per_utterance():
    metric, stub = stubbed(value=4.25)
    batch = UtteranceBatch(utterances=[
        utterance("a", SAMPLE_RATE), utterance("b", 2 * SAMPLE_RATE)
    ])
    out = metric.compute(batch)
    assert [set(f) for f in out] == [{"score"}, {"score"}]
    assert out == [{"score": 4.25}, {"score": 4.25}]


# ---------------------------------------------------------------------------
# Unmeasurable input is data, not an error
# ---------------------------------------------------------------------------


def test_too_short_to_rate_yields_nan_without_calling_the_model():
    metric, stub = stubbed()
    out = metric.compute(UtteranceBatch(utterances=[utterance("tiny", MIN_SAMPLES - 1)]))
    assert np.isnan(out[0]["score"])
    assert stub.calls == [], "the backbone raises below MIN_SAMPLES; guard before calling"


def test_a_short_utterance_does_not_prevent_the_others_being_scored():
    metric, stub = stubbed(value=2.0)
    out = metric.compute(UtteranceBatch(utterances=[
        utterance("tiny", MIN_SAMPLES - 1),
        utterance("ok", SAMPLE_RATE),
    ]))
    assert np.isnan(out[0]["score"])
    assert out[1]["score"] == 2.0
    assert len(stub.calls) == 1


def test_multichannel_input_is_mixed_to_mono_before_scoring():
    metric, stub = stubbed()
    stereo = torch.zeros(2, SAMPLE_RATE)
    metric.compute(UtteranceBatch(
        utterances=[Utterance("s", stereo, SAMPLE_RATE)]
    ))
    shape, _ = stub.calls[0]
    assert shape == (1, SAMPLE_RATE), f"expected mono (1, T), got {shape}"


# ---------------------------------------------------------------------------
# Real backend — opt-in, downloads ~392 MB on first run
# ---------------------------------------------------------------------------

REAL_BACKEND = pytest.mark.skipif(
    not os.environ.get("SPEECH_EVAL_TEST_MOS_BACKEND"),
    reason="set SPEECH_EVAL_TEST_MOS_BACKEND=1 to download and run UTMOS22",
)


@REAL_BACKEND
def test_real_backend_returns_a_score_on_the_mos_scale():
    metric = build_metric({"type": "mos_utmos"})
    metric.load(torch.device("cpu"))
    out = metric.compute(UtteranceBatch(utterances=[utterance("a", SAMPLE_RATE)]))
    assert 1.0 <= out[0]["score"] <= 5.0


@REAL_BACKEND
def test_real_backend_scores_a_mixed_length_batch_exactly_as_solo():
    """The property the stub tests enforce structurally, verified on the model."""
    metric = build_metric({"type": "mos_utmos"})
    metric.load(torch.device("cpu"))
    utts = [utterance("a", SAMPLE_RATE, seed=1), utterance("b", 2 * SAMPLE_RATE, seed=2)]
    together = [f["score"] for f in metric.compute(UtteranceBatch(utterances=utts))]
    alone = [
        metric.compute(UtteranceBatch(utterances=[u]))[0]["score"] for u in utts
    ]
    assert together == pytest.approx(alone, abs=1e-6)
