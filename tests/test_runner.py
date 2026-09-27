"""End-to-end: grouping, array artifacts, resume, and the results table."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from speech_eval.core import Features, Metric, Requirements, UtteranceBatch
from speech_eval.io import load_manifest
from speech_eval.metrics.level import LevelMetric
from speech_eval.runner import (
    group_by_requirements,
    load_artifacts,
    run_metrics,
)


class ArrayMetric(Metric):
    """Emits one scalar and one array, and counts how often it ran."""

    metric_type = "array_test"
    requirements = Requirements(sample_rate=16_000, level_norm_dbfs=-27.0)

    def __init__(self, name: str | None = None):
        super().__init__(name or "array_test")
        self.calls: list[str] = []

    def compute(self, batch: UtteranceBatch) -> list[Features]:
        self.calls.extend(batch.utt_ids)
        return [
            {"n_samples": u.wav.shape[-1], "envelope": np.arange(4, dtype="float32")}
            for u in batch.utterances
        ]


class MetaMetric(Metric):
    """Needs a manifest column to compute, and reports what it was given."""

    metric_type = "meta_test"
    requirements = Requirements(
        sample_rate=16_000, level_norm_dbfs=-27.0, meta_columns=("subject_id",)
    )

    def __init__(self, name: str | None = None):
        super().__init__(name or "meta_test")

    def compute(self, batch: UtteranceBatch) -> list[Features]:
        return [{"saw": u.meta.get("subject_id", "MISSING")} for u in batch.utterances]


class BadCountMetric(Metric):
    metric_type = "bad_count"
    requirements = Requirements()

    def __init__(self):
        super().__init__("bad_count")

    def compute(self, batch: UtteranceBatch) -> list[Features]:
        return [{"x": 1.0}]  # wrong length whenever the batch is not size 1


@pytest.fixture
def metadata(manifest_path, corpus):
    return load_manifest(manifest_path, dataset_roots={"test": corpus})


def test_group_by_requirements_buckets_on_the_declared_audio():
    groups = group_by_requirements([LevelMetric(), ArrayMetric(), LevelMetric("l2")])
    assert len(groups) == 2
    assert [m.name for m in groups[LevelMetric.requirements]] == ["level", "l2"]


def test_meta_columns_do_not_split_a_requirements_group():
    """They name manifest columns, not audio, so a metric that reads `sex` must
    still share one preparation pass with one that does not."""
    groups = group_by_requirements([ArrayMetric(), MetaMetric()])
    assert len(groups) == 1


def test_a_metric_gets_the_columns_it_needs_without_them_being_reported(
    metadata, tmp_path
):
    """What a metric needs to compute and what the results table shows are two
    different lists: `subject_id` reaches the metric, but nothing asked for it to
    be carried, so it stays out of the output."""
    results = run_metrics([MetaMetric()], metadata, tmp_path / "out")
    assert results["meta_test.saw"].tolist() == ["s1", "s1", "s2"]
    assert "subject_id" not in results.columns


def test_run_writes_one_row_per_utterance_with_prefixed_columns(metadata, tmp_path):
    out = tmp_path / "out"
    results = run_metrics([LevelMetric()], metadata, out, batch_size=2)

    assert len(results) == 3
    assert list(results.columns[:3]) == ["utt_id", "group_id", "condition"]
    assert "level.rms_dbfs" in results.columns
    assert results.set_index("utt_id").loc["a_real", "level.duration_s"] == pytest.approx(1.0)
    # Level metric declares no normalisation, so the 12 dB difference survives.
    levels = results.set_index("utt_id")["level.rms_dbfs"]
    assert levels["a_conv"] - levels["a_real"] == pytest.approx(12.0, abs=0.1)
    assert (out / "metrics.csv").exists()


def test_run_carries_requested_metadata_columns(metadata, tmp_path):
    results = run_metrics(
        [LevelMetric()], metadata, tmp_path / "out", carry_columns=["subject_id"]
    )
    assert results["subject_id"].tolist() == ["s1", "s1", "s2"]


def test_arrays_go_to_the_archive_not_the_csv(metadata, tmp_path):
    out = tmp_path / "out"
    results = run_metrics([ArrayMetric()], metadata, out, batch_size=3)

    assert "array_test.n_samples" in results.columns
    assert "array_test.envelope" not in results.columns
    artifacts = load_artifacts(out)
    assert artifacts["a_real|array_test.envelope"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert len(artifacts) == 3


def test_two_requirement_groups_both_run(metadata, tmp_path):
    results = run_metrics([LevelMetric(), ArrayMetric()], metadata, tmp_path / "out")
    assert "level.rms_dbfs" in results.columns
    assert "array_test.n_samples" in results.columns
    assert results["array_test.n_samples"].notna().all()


def test_resume_skips_completed_utterances(metadata, tmp_path):
    out = tmp_path / "out"
    run_metrics([ArrayMetric()], metadata, out)

    second = ArrayMetric()
    results = run_metrics([second], metadata, out)
    assert second.calls == []                       # nothing recomputed
    assert results["array_test.n_samples"].notna().all()  # results still present
    assert len(load_artifacts(out)) == 3            # and artifacts survived


def test_resume_computes_only_the_missing_rows(metadata, tmp_path):
    out = tmp_path / "out"
    run_metrics([ArrayMetric()], metadata, out, batch_size=1)

    # Simulate a run killed after two of three utterances.
    csv = pd.read_csv(out / "metrics.csv")
    csv.loc[csv["utt_id"] == "b_real", "array_test.n_samples"] = np.nan
    csv.to_csv(out / "metrics.csv", index=False)

    resumed = ArrayMetric()
    results = run_metrics([resumed], metadata, out, batch_size=1)
    assert resumed.calls == ["b_real"]
    assert results["array_test.n_samples"].notna().all()


def test_resume_disabled_recomputes_everything(metadata, tmp_path):
    out = tmp_path / "out"
    run_metrics([ArrayMetric()], metadata, out)
    second = ArrayMetric()
    run_metrics([second], metadata, out, resume=False)
    assert sorted(second.calls) == ["a_conv", "a_real", "b_real"]


def test_metric_returning_the_wrong_number_of_results_is_caught(metadata, tmp_path):
    with pytest.raises(RuntimeError, match="must return one per"):
        run_metrics([BadCountMetric()], metadata, tmp_path / "out", batch_size=3)


def test_needs_intervals_fails_loudly_until_alignment_lands(metadata, tmp_path):
    class NeedsIntervals(ArrayMetric):
        requirements = Requirements(needs_intervals=True)

    with pytest.raises(NotImplementedError, match="forced-alignment"):
        run_metrics([NeedsIntervals()], metadata, tmp_path / "out")
