"""The three comparisons: pivoting, recovery rate, slopes, clustered CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from speech_eval.compare import (
    cluster_bootstrap_ci,
    recovery_rate,
    slope_vs_target,
    to_wide,
)


@pytest.fixture
def results():
    return pd.DataFrame([
        {"group_id": "g1", "condition": "real_src", "f0": 100.0, "subject_id": "s1"},
        {"group_id": "g1", "condition": "converted", "f0": 130.0, "subject_id": "s1"},
        {"group_id": "g1", "condition": "real_tgt", "f0": 140.0, "subject_id": "s1"},
        {"group_id": "g2", "condition": "real_src", "f0": 200.0, "subject_id": "s2"},
        {"group_id": "g2", "condition": "converted", "f0": 200.0, "subject_id": "s2"},
        {"group_id": "g2", "condition": "real_tgt", "f0": 240.0, "subject_id": "s2"},
    ])


def test_to_wide_puts_every_condition_on_one_row(results):
    wide = to_wide(results, ["f0"]).set_index("group_id")
    assert wide.loc["g1", "real_src.f0"] == 100.0
    assert wide.loc["g1", "converted.f0"] == 130.0
    assert wide.loc["g2", "real_tgt.f0"] == 240.0


def test_to_wide_rejects_a_repeated_condition(results):
    doubled = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        to_wide(doubled, ["f0"])


def test_to_wide_rejects_missing_columns(results):
    with pytest.raises(KeyError):
        to_wide(results, ["not_measured"])


def test_recovery_rate_spans_no_change_to_full_recovery(results):
    wide = to_wide(results, ["f0"]).set_index("group_id")
    rate = recovery_rate(wide["real_src.f0"], wide["converted.f0"], wide["real_tgt.f0"])
    assert rate[0] == pytest.approx(30.0 / 40.0)  # g1 covered three quarters
    assert rate[1] == pytest.approx(0.0)          # g2 did not move at all


def test_recovery_rate_signs_overshoot_and_wrong_direction():
    rate = recovery_rate([100.0, 100.0], [160.0, 90.0], [140.0, 140.0])
    assert rate[0] > 1.0   # overshoot
    assert rate[1] < 0.0   # moved away from the target


def test_recovery_rate_is_undefined_when_there_is_nothing_to_recover():
    # Real speech barely differs between the conditions: the ratio would be
    # noise divided by noise.
    rate = recovery_rate([100.0], [105.0], [100.2], min_gap=1.0)
    assert np.isnan(rate[0])


def test_slope_vs_target_fits_per_group():
    frame = pd.DataFrame({
        "subject_id": ["s1"] * 3 + ["s2"] * 3,
        "tau_tgt_db": [60.0, 70.0, 80.0] * 2,
        "f0": [100.0, 110.0, 120.0, 200.0, 220.0, 240.0],
    })
    slopes = slope_vs_target(frame, "f0", "tau_tgt_db").set_index("subject_id")
    assert slopes.loc["s1", "slope"] == pytest.approx(1.0)
    assert slopes.loc["s2", "slope"] == pytest.approx(2.0)
    assert slopes.loc["s1", "r"] == pytest.approx(1.0)
    assert slopes["n"].tolist() == [3, 3]


def test_slope_vs_target_drops_groups_it_cannot_fit():
    frame = pd.DataFrame({
        "subject_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
        "tau_tgt_db": [60.0, 70.0, 80.0, 70.0, 70.0, 70.0],  # s2 has no variation
        "f0": [100.0, 110.0, 120.0, 200.0, 220.0, 240.0],
    })
    assert slope_vs_target(frame, "f0", "tau_tgt_db")["subject_id"].tolist() == ["s1"]


def test_slope_vs_target_can_pool():
    frame = pd.DataFrame({
        "tau_tgt_db": [60.0, 70.0, 80.0],
        "f0": [100.0, 110.0, 120.0],
    })
    pooled = slope_vs_target(frame, "f0", "tau_tgt_db", group_col=None)
    assert len(pooled) == 1
    assert pooled["slope"].iloc[0] == pytest.approx(1.0)


def test_cluster_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, 200)
    clusters = np.repeat(np.arange(20), 10)
    ci = cluster_bootstrap_ci(values, clusters, n_boot=500)
    assert ci["ci_low"] < ci["estimate"] < ci["ci_high"]
    assert ci["n_clusters"] == 20
    assert ci["n"] == 200


def test_cluster_bootstrap_is_wider_than_resampling_utterances():
    """The reason the function exists: within-cluster correlation is real.

    Ten speakers with a large between-speaker offset and no within-speaker
    variation carry ten independent observations, not two hundred.  Treating
    each utterance as its own cluster would understate the interval.
    """
    speaker_means = np.arange(10, dtype=float) * 10
    values = np.repeat(speaker_means, 20)
    clusters = np.repeat(np.arange(10), 20)

    by_speaker = cluster_bootstrap_ci(values, clusters, n_boot=1000)
    by_utterance = cluster_bootstrap_ci(values, np.arange(values.size), n_boot=1000)
    speaker_width = by_speaker["ci_high"] - by_speaker["ci_low"]
    utterance_width = by_utterance["ci_high"] - by_utterance["ci_low"]
    assert speaker_width > 2 * utterance_width


def test_cluster_bootstrap_ignores_nan_and_is_seeded():
    values = np.array([1.0, 2.0, np.nan, 4.0])
    clusters = np.array(["a", "a", "b", "b"])
    first = cluster_bootstrap_ci(values, clusters, n_boot=200, seed=7)
    second = cluster_bootstrap_ci(values, clusters, n_boot=200, seed=7)
    assert first == second
    assert first["n"] == 3


def test_cluster_bootstrap_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="No non-nan values"):
        cluster_bootstrap_ci([np.nan], ["a"])
