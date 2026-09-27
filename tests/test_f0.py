"""F0 metrics: the three trackers held to the same contract.

Almost everything here is parametrised over all three backends, which is the
point: a column named ``f0_praat.median_hz`` and one named
``f0_swiftf0.median_hz`` are only comparable if the two agree on what a voiced
frame is, what a semitone is measured from, and what an unmeasurable utterance
returns.  The tolerances are analytic — a synthetic complex at a known F0 — so
these tests survive a backend upgrade that moves a decimal.

Pure tones are fine here, unlike in ``test_egemaps``: pitch tracking is exactly
the measurement a periodic signal is a valid probe for.  A harmonic complex is
used anyway, since a single sinusoid has no fundamental to distinguish from its
own first harmonic.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_eval.core import Requirements, Utterance, UtteranceBatch, build_metric
from speech_eval.metrics.f0 import (
    DEFAULT_PITCH_RANGE,
    SEMITONE_REFERENCE_HZ,
    F0Metric,
    OpenSMILEF0Metric,
    PraatF0Metric,
    SwiftF0Metric,
)

pytest.importorskip("parselmouth", reason="needs the `phonetics` extra")
pytest.importorskip("opensmile", reason="needs the `phonetics` extra")
pytest.importorskip("swift_f0", reason="needs the `phonetics` extra")

SAMPLE_RATE = 16_000
TRACKERS = ("f0_praat", "f0_opensmile", "f0_swiftf0")

# Every backend lands within ~1 Hz of truth on these signals; 2 Hz leaves room
# for one to change its interpolation without the test becoming a tripwire.
HZ_TOLERANCE = 2.0


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def complex_tone(f0: float = 120.0, duration_s: float = 2.0, amplitude: float = 0.3):
    """A harmonic complex at `f0`: 19 harmonics with 1/k amplitudes."""
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    signal = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 20))
    return (amplitude * signal / np.abs(signal).max()).astype("float32")


def sweep(f_start: float, f_end: float, duration_s: float = 2.0):
    """A phase-continuous glide, for checking the contour and not just a median."""
    n = int(duration_s * SAMPLE_RATE)
    phase = 2 * np.pi * np.cumsum(np.linspace(f_start, f_end, n)) / SAMPLE_RATE
    signal = sum(np.sin(k * phase) / k for k in range(1, 20))
    return (0.3 * signal / np.abs(signal).max()).astype("float32")


def voiced_then_silence(f0: float = 150.0):
    """One second of voice, one of digital silence."""
    return np.concatenate([
        complex_tone(f0, duration_s=1.0), np.zeros(SAMPLE_RATE, dtype="float32")
    ]).astype("float32")


def batch_of(*signals: np.ndarray, meta: dict | None = None) -> UtteranceBatch:
    return UtteranceBatch([
        Utterance(
            utt_id=f"u{i}",
            wav=torch.from_numpy(signal).unsqueeze(0),
            sample_rate=SAMPLE_RATE,
            meta=dict(meta or {}),
        )
        for i, signal in enumerate(signals)
    ])


@pytest.fixture(scope="module")
def loaded():
    """One loaded instance of each tracker, shared by the whole module."""
    built = {}
    for tracker in TRACKERS:
        metric = build_metric({"type": tracker})
        metric.load(torch.device("cpu"))
        built[tracker] = metric
    return built


@pytest.fixture(params=TRACKERS)
def tracker(request, loaded) -> F0Metric:
    return loaded[request.param]


def measure(metric: F0Metric, signal: np.ndarray, meta: dict | None = None):
    return metric.compute(batch_of(signal, meta=meta))[0]


# ---------------------------------------------------------------------------
# Accuracy — the same contract for every backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("f0", [120.0, 220.0])
def test_every_tracker_recovers_a_known_fundamental(tracker, f0):
    features = measure(tracker, complex_tone(f0))
    assert features["median_hz"] == pytest.approx(f0, abs=HZ_TOLERANCE)
    assert features["voiced_fraction"] > 0.9


def test_semitones_are_measured_from_the_same_reference_as_opensmile(tracker):
    """12*log2(f/27.5), so the three columns are directly comparable and speakers
    of different sexes sit on one scale."""
    features = measure(tracker, complex_tone(220.0))
    recovered = SEMITONE_REFERENCE_HZ * 2 ** (features["mean_semitones"] / 12)
    assert recovered == pytest.approx(features["median_hz"], abs=HZ_TOLERANCE)


def test_a_steady_tone_has_almost_no_semitone_spread(tracker):
    assert measure(tracker, complex_tone(150.0))["sd_semitones"] < 0.1


def test_the_contour_follows_a_glide(tracker):
    """The summary alone cannot tell a rising contour from a flat one at the same
    median, and for vocal effort the movement is the evidence."""
    features = measure(tracker, sweep(120.0, 200.0))
    contour = features["contour.f0_hz"]
    quarter = contour.size // 4
    assert np.nanmedian(contour[:quarter]) == pytest.approx(130.0, abs=10.0)
    assert np.nanmedian(contour[-quarter:]) == pytest.approx(190.0, abs=10.0)


def test_f0_is_invariant_to_gain(tracker):
    """The level normalisation in Requirements must not be doing the work here:
    pitch is a rate, and a converter that only amplified must not move it."""
    loud = measure(tracker, complex_tone(150.0, amplitude=0.9))
    soft = measure(tracker, complex_tone(150.0, amplitude=0.05))
    assert loud["median_hz"] == pytest.approx(soft["median_hz"], abs=0.5)
    assert loud["mean_semitones"] == pytest.approx(soft["mean_semitones"], abs=0.05)


def test_the_three_trackers_agree_on_clean_synthetic_speech(loaded):
    """Not a redundancy check — it establishes the baseline against which
    disagreement on real converted audio is later read."""
    signal = complex_tone(180.0)
    medians = [measure(loaded[t], signal)["median_hz"] for t in TRACKERS]
    assert max(medians) - min(medians) < HZ_TOLERANCE


# ---------------------------------------------------------------------------
# Voicing
# ---------------------------------------------------------------------------


def test_voiced_fraction_finds_the_voiced_half(tracker):
    features = measure(tracker, voiced_then_silence())
    assert features["voiced_fraction"] == pytest.approx(0.5, abs=0.1)
    assert features["n_voiced_frames"] < features["n_frames"]


def test_silence_yields_nan_rather_than_zero(tracker):
    """A pitch of 0 Hz is not a measurement, and averaging it into a mean would
    pull every aggregate toward zero in proportion to the silence."""
    features = measure(tracker, np.zeros(SAMPLE_RATE, dtype="float32"))
    assert np.isnan(features["median_hz"])
    assert np.isnan(features["mean_semitones"])
    assert features["voiced_fraction"] == 0.0
    assert features["n_voiced_frames"] == 0


def test_a_signal_too_short_to_analyse_yields_nan_not_an_exception(tracker):
    """Praat's window is 3/pitch_floor and it raises below that; an utterance it
    cannot measure is data, so the metric must return NaN and keep going."""
    features = measure(tracker, complex_tone(120.0, duration_s=0.02))
    assert np.isnan(features["median_hz"])


def test_an_empty_waveform_is_survivable(tracker):
    features = measure(tracker, np.zeros(0, dtype="float32"))
    assert features["n_frames"] == 0
    assert np.isnan(features["median_hz"])
    assert np.isnan(features["voiced_fraction"])


# ---------------------------------------------------------------------------
# The analysis range — provenance, not configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sex, expected", [("M", (60.0, 400.0)), ("female", (90.0, 600.0))]
)
def test_the_range_is_sex_conditional(sex, expected, loaded):
    features = measure(loaded["f0_praat"], complex_tone(), meta={"sex": sex})
    assert (features["fmin_hz"], features["fmax_hz"]) == expected
    assert features["range_source"] == "sex"


@pytest.mark.parametrize("meta", [{}, {"sex": None}, {"sex": float("nan")}, {"sex": "x"}])
def test_an_unusable_sex_value_falls_back_to_the_wide_default(meta, loaded):
    features = measure(loaded["f0_swiftf0"], complex_tone(), meta=meta)
    assert (features["fmin_hz"], features["fmax_hz"]) == DEFAULT_PITCH_RANGE
    assert features["range_source"] == "default"


def test_the_same_speaker_gets_the_same_range_in_every_condition(loaded):
    """The property the whole design turns on: an instrument that adapted to the
    signal would move with the thing it is measuring.  Two conditions differing
    in level and in pitch must still be analysed identically."""
    metric = loaded["f0_praat"]
    source = measure(metric, complex_tone(140.0, amplitude=0.1), meta={"sex": "M"})
    converted = measure(metric, complex_tone(190.0, amplitude=0.8), meta={"sex": "M"})
    assert (source["fmin_hz"], source["fmax_hz"]) == (
        converted["fmin_hz"], converted["fmax_hz"]
    )


def test_opensmile_reports_the_range_it_is_stuck_with(loaded):
    """55-1000 Hz is fixed inside the eGeMAPS configuration, so `sex` cannot
    apply.  Constant across conditions is the property that matters, and holds."""
    features = measure(loaded["f0_opensmile"], complex_tone(), meta={"sex": "M"})
    assert (features["fmin_hz"], features["fmax_hz"]) == OpenSMILEF0Metric.BACKEND_RANGE
    assert features["range_source"] == "backend_fixed"


def test_a_configured_range_is_honoured_and_reported():
    metric = PraatF0Metric(pitch_ranges={"M": [70.0, 300.0]})
    metric.load(torch.device("cpu"))
    features = measure(metric, complex_tone(), meta={"sex": "m"})
    assert (features["fmin_hz"], features["fmax_hz"]) == (70.0, 300.0)


def test_an_impossible_range_is_rejected_at_construction():
    with pytest.raises(ValueError, match="not a valid interval"):
        PraatF0Metric(pitch_ranges={"m": [400.0, 60.0]})


def test_swiftf0_rejects_a_range_its_model_cannot_resolve_at_load():
    """Better at startup than a thousand utterances into a run."""
    metric = SwiftF0Metric(default_range=[20.0, 600.0])
    with pytest.raises(ValueError, match="below model minimum"):
        metric.load(torch.device("cpu"))


# ---------------------------------------------------------------------------
# Frame grids and contours
# ---------------------------------------------------------------------------


def test_each_tracker_reports_its_own_grid(tracker):
    """The grids genuinely differ, so alignment is compare.py's problem and the
    two numbers it needs must be on every row."""
    features = measure(tracker, complex_tone())
    assert features["time_step_s"] > 0
    assert features["t0_s"] >= 0
    assert features["contour.f0_hz"].size == features["n_frames"]


def test_swiftf0_runs_on_its_fixed_sixteen_millisecond_hop(loaded):
    features = measure(loaded["f0_swiftf0"], complex_tone())
    assert features["time_step_s"] == pytest.approx(0.016)


@pytest.mark.parametrize("name", ["f0_praat", "f0_opensmile"])
def test_the_classical_trackers_run_at_ten_milliseconds(name, loaded):
    assert measure(loaded[name], complex_tone())["time_step_s"] == pytest.approx(0.01)


def test_unvoiced_frames_are_nan_in_the_contour(tracker):
    """Each backend fills unvoiced frames differently — Praat writes 0.0, SwiftF0
    leaves an unconstrained estimate — and a contour usable as pitch by accident
    is a trap."""
    features = measure(tracker, voiced_then_silence())
    contour, voiced = features["contour.f0_hz"], features["contour.voiced"]
    assert np.isnan(contour[~voiced]).all()
    assert np.isfinite(contour[voiced]).all()
    assert voiced.dtype == bool


def test_contours_can_be_turned_off():
    metric = PraatF0Metric(contours=False)
    metric.load(torch.device("cpu"))
    features = measure(metric, complex_tone())
    assert not any(key.startswith("contour.") for key in features)


def test_the_tracker_is_recorded_on_every_row(tracker):
    """The instance name is a config's choice; this says which backend ran."""
    assert measure(tracker, complex_tone())["tracker"] in {
        "praat_ac", "opensmile_shs", "swiftf0"
    }


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_all_three_share_one_audio_preparation():
    """They differ only in meta_columns, which names manifest columns rather than
    a property of the waveform, so the runner must not load the audio twice."""
    from speech_eval.runner import group_by_requirements

    metrics = [build_metric({"type": t}) for t in TRACKERS]
    assert len(group_by_requirements(metrics)) == 1


def test_requirements_ask_for_the_sex_column_where_it_is_used():
    assert PraatF0Metric().requirements.meta_columns == ("sex",)
    assert SwiftF0Metric().requirements.meta_columns == ("sex",)
    # openSMILE's range is not ours to choose, so it asks for nothing.
    assert OpenSMILEF0Metric().requirements.meta_columns == ()


def test_requirements_match_the_egemaps_family():
    requirements = PraatF0Metric().requirements
    assert requirements.sample_rate == 16_000
    assert requirements.level_norm_dbfs == -27.0
    assert requirements.mono is True


@pytest.mark.parametrize("name", TRACKERS)
def test_registered_and_buildable_from_config(name):
    metric = build_metric({"type": name, "name": f"{name}_v2", "contours": False})
    assert isinstance(metric, F0Metric)
    assert metric.name == f"{name}_v2"


def test_compute_before_load_is_an_error():
    with pytest.raises(RuntimeError, match="load\\(\\) must be called"):
        PraatF0Metric().compute(batch_of(complex_tone()))


def test_a_rate_the_metric_was_not_configured_for_is_refused(loaded):
    metric = loaded["f0_praat"]
    wrong_rate = UtteranceBatch([
        Utterance("u", torch.from_numpy(complex_tone()).unsqueeze(0), 8_000)
    ])
    with pytest.raises(ValueError, match="configured for 16000 Hz"):
        metric.compute(wrong_rate)
