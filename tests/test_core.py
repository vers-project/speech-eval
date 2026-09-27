"""Core contracts: requirements grouping, the registry, and batch views."""
from __future__ import annotations

import pytest
import torch

from speech_eval.core import (
    Metric,
    Requirements,
    Utterance,
    UtteranceBatch,
    build_metric,
    build_metrics,
)
from speech_eval.metrics.level import LevelMetric


def utterance(utt_id: str, n: int = 1000, sr: int = 16_000) -> Utterance:
    return Utterance(utt_id=utt_id, wav=torch.zeros(1, n), sample_rate=sr)


def test_requirements_are_hashable_and_group():
    # The runner buckets metrics on this, so equal requirements must collide.
    assert Requirements(sample_rate=16_000) == Requirements(sample_rate=16_000)
    assert len({Requirements(), Requirements(), Requirements(mono=False)}) == 2


def test_build_metric_defaults_name_to_type():
    metric = build_metric({"type": "level"})
    assert isinstance(metric, LevelMetric)
    assert metric.name == "level"


def test_build_metric_honours_name_override():
    # The override is what lets two backends of one metric type coexist.
    assert build_metric({"type": "level", "name": "level_raw"}).name == "level_raw"


def test_build_metric_rejects_unknown_type():
    with pytest.raises(KeyError, match="Unknown metric type"):
        build_metric({"type": "not_a_metric"})


def test_build_metrics_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate metric names"):
        build_metrics([{"type": "level"}, {"type": "level"}])


def test_batch_pads_lazily_and_keeps_true_lengths():
    batch = UtteranceBatch([utterance("a", 1000), utterance("b", 400)])
    audio = batch.audio
    assert audio.data.shape == (2, 1, 1000)
    assert audio.lengths.tolist() == [1000, 400]
    # The mask must not declare the padding valid.
    assert audio.padding_mask[1].sum().item() == 400


def test_batch_refuses_to_pad_mixed_sample_rates():
    batch = UtteranceBatch([utterance("a", sr=16_000), utterance("b", sr=44_100)])
    with pytest.raises(ValueError, match="mixed sample rates"):
        _ = batch.audio


def test_metric_is_abstract():
    with pytest.raises(TypeError):
        Metric()  # type: ignore[abstract]
