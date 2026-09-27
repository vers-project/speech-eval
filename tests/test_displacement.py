"""Tests for the displacement formulation and its figures.

The arithmetic is three subtractions, so what is worth testing is which three values get
subtracted: a displacement computed against the wrong reference is a plausible number
that answers a different question, and nothing downstream can tell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from speech_eval.compare.displacement import (
    cell_means,
    displacement_table,
    per_group_slopes,
    reference_maps,
    summarise,
)


def make_results() -> pd.DataFrame:
    """One speaker, one group, two levels, with a lossy baseline stage.

    real@A = 10, real@B = 20  -> requested A->B is +10
    codec@A = 11 (the stage adds +1), converted A->B = 19 -> achieved is +8
    """
    return pd.DataFrame([
        {"utt_id": "a_real", "src": "a_real", "ref": None, "condition": "real",
         "spk": "S1", "x": 10.0, "from": 1, "to": None},
        {"utt_id": "b_real", "src": "b_real", "ref": None, "condition": "real",
         "spk": "S1", "x": 20.0, "from": 2, "to": None},
        {"utt_id": "a_codec", "src": "a_real", "ref": None, "condition": "codec",
         "spk": "S1", "x": 11.0, "from": 1, "to": None},
        {"utt_id": "b_codec", "src": "b_real", "ref": None, "condition": "codec",
         "spk": "S1", "x": 21.0, "from": 2, "to": None},
        {"utt_id": "a_to_b", "src": "a_real", "ref": "b_real", "condition": "converted",
         "spk": "S1", "x": 19.0, "from": 1, "to": 2},
        {"utt_id": "b_to_a", "src": "b_real", "ref": "a_real", "condition": "converted",
         "spk": "S1", "x": 12.0, "from": 2, "to": 1},
    ])


def table(**kwargs) -> pd.DataFrame:
    defaults = dict(
        source_col="src", target_col="ref", condition_col="condition",
        source_condition="real", baseline_condition="codec",
        subject_conditions=["converted"], carry=["spk", "from", "to"],
    )
    return displacement_table(make_results(), ["x"], **{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# Which three values
# ---------------------------------------------------------------------------

def test_requested_is_real_target_minus_real_source():
    row = table().set_index("utt_id").loc["a_to_b"]
    assert row["requested"] == pytest.approx(10.0)          # 20 - 10, both real
    assert row["value_source"] == pytest.approx(10.0)
    assert row["value_target"] == pytest.approx(20.0)


def test_achieved_is_measured_from_the_baseline_not_the_source():
    """The lossy stage's own effect must not be charged to the transformation."""
    row = table().set_index("utt_id").loc["a_to_b"]
    assert row["value_baseline"] == pytest.approx(11.0)     # codec@A, not real@A
    assert row["achieved"] == pytest.approx(8.0)            # 19 - 11, not 19 - 10
    assert row["error"] == pytest.approx(-2.0)


def test_no_baseline_reduces_to_the_source_as_origin():
    """Which is recovery_rate's numerator — the function subsumes it, not duplicates it."""
    row = table(baseline_condition=None).set_index("utt_id").loc["a_to_b"]
    assert row["value_baseline"] == pytest.approx(10.0)
    assert row["achieved"] == pytest.approx(9.0)            # 19 - 10


def test_the_downward_direction_is_kept_signed():
    rows = table().set_index("utt_id")
    assert rows.loc["a_to_b", "requested"] > 0
    assert rows.loc["b_to_a", "requested"] < 0
    # Pooling these by magnitude is exactly what the formulation exists to prevent.
    assert rows.loc["a_to_b", "requested"] == pytest.approx(
        -rows.loc["b_to_a", "requested"])


def test_only_subject_rows_are_displaced():
    out = table()
    assert set(out["utt_id"]) == {"a_to_b", "b_to_a"}


def test_subject_conditions_default_to_everything_else():
    out = table(subject_conditions=None)
    # `real` is the source and `codec` the baseline, so only `converted` is left.
    assert set(out["utt_id"]) == {"a_to_b", "b_to_a"}


def test_carried_columns_survive():
    row = table().set_index("utt_id").loc["a_to_b"]
    assert row["spk"] == "S1" and row["from"] == 1 and row["to"] == 2


def test_absent_measures_are_skipped_not_fatal():
    """One config can name a superset of what a given run measured."""
    out = displacement_table(
        make_results(), ["x", "never_measured"], source_col="src", target_col="ref",
        baseline_condition="codec", subject_conditions=["converted"],
    )
    assert set(out["metric"]) == {"x"}


def test_a_duplicated_source_key_is_refused():
    """Two reference rows per source would silently pick one and look fine."""
    results = pd.concat([make_results(), make_results().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="does not identify a unique"):
        displacement_table(results, ["x"], source_col="src", target_col="ref",
                           baseline_condition="codec", subject_conditions=["converted"])


def test_reference_maps_returns_the_source_twice_without_a_baseline():
    source, baseline = reference_maps(make_results(), "src", "condition", "real", None)
    assert source.equals(baseline)


# ---------------------------------------------------------------------------
# Fits and summaries
# ---------------------------------------------------------------------------

def _many_speakers(slope: float, n_speakers=4, n_points=6) -> pd.DataFrame:
    rows = []
    for s in range(n_speakers):
        for i in range(n_points):
            requested = float(i - n_points // 2)
            rows.append({"metric": "x", "spk": f"S{s}", "requested": requested,
                         "achieved": slope * requested, "error": 0.0})
    return pd.DataFrame(rows)


def test_a_perfect_system_has_slope_one():
    slopes = per_group_slopes(_many_speakers(1.0), group_col="spk")
    assert slopes["slope"].to_numpy() == pytest.approx(1.0)


def test_a_system_that_ignores_the_request_has_slope_zero():
    slopes = per_group_slopes(_many_speakers(0.0), group_col="spk")
    assert slopes["slope"].to_numpy() == pytest.approx(0.0)


def test_the_summary_interval_comes_from_resampling_groups():
    data = _many_speakers(0.6)
    slopes = per_group_slopes(data, group_col="spk")
    summary = summarise(slopes, data, group_col="spk", n_boot=200, seed=0)
    row = summary.iloc[0]
    assert row["metric"] == "x"
    assert row["n_groups"] == 4
    assert row["slope_mean"] == pytest.approx(0.6)
    assert row["slope_lo"] <= row["slope_mean"] <= row["slope_hi"]


def test_cell_means_form_the_source_by_target_grid():
    cells = cell_means(table(), row_col="from", col_col="to")
    assert set(cells.columns) >= {"metric", "from", "to", "mean", "std", "n"}
    assert len(cells) == 2
    assert cells["n"].tolist() == [1, 1]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_the_scatter_draws_the_diagonal_and_a_fit(tmp_path):
    from speech_eval import figures

    ax = figures.displacement_scatter(_many_speakers(0.5), metric="x", unit="dB")
    assert "requested displacement (dB)" == ax.get_xlabel()
    # Equal limits, or the diagonal is not at 45 degrees and the eye is lied to.
    assert ax.get_xlim() == pytest.approx(ax.get_ylim())
    assert any("slope" in t.get_text() for t in ax.texts)
    figures.save(ax, tmp_path / "s.png")
    assert (tmp_path / "s.png").exists()


def test_the_scatter_refuses_a_fourth_categorical_series():
    """Past three slots the palette cannot hold its separation floors in a scatter."""
    from speech_eval import figures

    data = _many_speakers(1.0)
    data["level"] = np.tile(["a", "b", "c", "d"], len(data) // 4 + 1)[:len(data)]
    with pytest.raises(ValueError, match="at most 3 categorical slots"):
        figures.displacement_scatter(data, metric="x", hue="level")


def test_the_heatmap_is_symmetric_about_zero():
    """Zero must sit at the neutral midpoint, or a cell with no error looks like one."""
    from speech_eval import figures

    cells = pd.DataFrame([
        {"metric": "x", "from": 1, "to": 1, "mean": 0.0},
        {"metric": "x", "from": 1, "to": 2, "mean": -4.0},
        {"metric": "x", "from": 2, "to": 1, "mean": 1.0},
        {"metric": "x", "from": 2, "to": 2, "mean": 0.0},
    ])
    ax = figures.cell_heatmap(cells, metric="x", row_col="from", col_col="to")
    image = ax.images[0]
    assert image.get_clim() == pytest.approx((-4.0, 4.0))
    # Every finite cell is labelled: the relief for readers who cannot separate the hues.
    assert len([t for t in ax.texts if t.get_text()]) == 4


def test_the_slope_plot_marks_perfect_and_ignored():
    from speech_eval import figures

    slopes = per_group_slopes(_many_speakers(0.7), group_col="spk")
    ax = figures.group_slopes(slopes, group_col="spk")
    verticals = [line.get_xdata()[0] for line in ax.lines]
    assert 1.0 in verticals and 0.0 in verticals
    assert len(ax.collections) == 2 * slopes["metric"].nunique()   # dots + mean marker
