"""eGeMAPS metric: analytic spectral checks, frame-set rules, and guards.

The band measures are checked against closed-form expectations on synthetic
spectra rather than against stored numbers, so the tests state what the
quantities *mean* and would survive an openSMILE upgrade that changed a decimal.

Shaped noise, not tones: a pure tone under a 20 ms Hamming window leaks across
the whole spectrum, so a two-tone signal cannot isolate a frequency band. Noise
with a prescribed magnitude response can.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_eval.core import Requirements, Utterance, UtteranceBatch, build_metric
from speech_eval.metrics.egemaps import EGeMAPSMetric

opensmile = pytest.importorskip("opensmile", reason="needs the `phonetics` extra")

SAMPLE_RATE = 16_000
DURATION_S = 4.0


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def shaped_noise(response, seed: int = 0, peak: float = 0.5) -> np.ndarray:
    """Gaussian noise given a prescribed magnitude response, scaled to `peak`."""
    n = int(DURATION_S * SAMPLE_RATE)
    rng = np.random.default_rng(seed)
    spectrum = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    signal = np.fft.irfft(spectrum * response(freqs), n)
    return (signal / np.abs(signal).max() * peak).astype("float32")


def harmonic_then_silence(f0: float = 120.0) -> np.ndarray:
    """One second of a voiced-like harmonic complex, then one second of silence."""
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    voiced = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 20))
    voiced = 0.3 * voiced / np.abs(voiced).max()
    return np.concatenate([voiced, np.zeros(SAMPLE_RATE)]).astype("float32")


def batch_of(*signals: np.ndarray, sample_rate: int = SAMPLE_RATE) -> UtteranceBatch:
    return UtteranceBatch([
        Utterance(utt_id=f"u{i}", wav=torch.from_numpy(s).unsqueeze(0),
                  sample_rate=sample_rate)
        for i, s in enumerate(signals)
    ])


@pytest.fixture(scope="module")
def metric():
    # level_norm_dbfs=None: these tests drive the level deliberately.
    m = EGeMAPSMetric(level_norm_dbfs=None)
    m.load(torch.device("cpu"))
    return m


def measure(metric: EGeMAPSMetric, signal: np.ndarray, key: str) -> float:
    return metric.compute(batch_of(signal))[0][key]


# ---------------------------------------------------------------------------
# Band measures against closed-form expectations
# ---------------------------------------------------------------------------


def test_alpha_ratio_on_a_flat_spectrum_matches_the_band_width_ratio(metric):
    """Flat spectrum: the alpha ratio is fixed by the two bands' widths alone.

    |10*log10(950/4000)| = 6.24 dB, since the bands are 50-1000 Hz and 1-5 kHz.
    """
    flat = shaped_noise(lambda f: np.ones_like(f))
    value = measure(metric, flat, "alphaRatio_sma3_all_mean")
    assert abs(value) == pytest.approx(6.24, abs=0.5)


def test_alpha_ratio_is_high_over_low_as_the_1975_original_defines_it(metric):
    """alphaRatio is E(1-5 kHz) / E(50-1000 Hz), so it rises with vocal effort.

    That is the orientation of the original definition (Frokjaer-Jensen & Prytz
    1975) and of the vocal-effort literature that followed it; the GeMAPS paper's
    prose states the reciprocal, and is the odd one out.  Boosting the low band
    must therefore *lower* this value.  The test guards against a silent
    re-orientation on upgrade, which would flip the sign of a reported result.
    """
    flat = shaped_noise(lambda f: np.ones_like(f))
    low_boosted = shaped_noise(lambda f: np.where(f < 1000, 10.0, 1.0))

    assert measure(metric, flat, "alphaRatio_sma3_all_mean") > 0
    assert measure(metric, low_boosted, "alphaRatio_sma3_all_mean") < -5


def test_hammarberg_index_follows_its_published_orientation(metric):
    """Strongest peak in 0-2 kHz minus strongest in 2-5 kHz: positive when low wins."""
    low_boosted = shaped_noise(lambda f: np.where(f < 1000, 10.0, 1.0))
    high_boosted = shaped_noise(lambda f: np.where(f > 2000, 10.0, 1.0))

    assert measure(metric, low_boosted, "hammarbergIndex_sma3_all_mean") > 10
    assert measure(metric, high_boosted, "hammarbergIndex_sma3_all_mean") < 0


def test_the_two_indices_move_in_opposite_directions(metric):
    """Which is the point: they are not interchangeable, and a paper must say
    which direction each takes with vocal effort."""
    low_boosted = shaped_noise(lambda f: np.where(f < 1000, 10.0, 1.0))
    high_boosted = shaped_noise(lambda f: np.where(f > 2000, 10.0, 1.0))
    features_low = metric.compute(batch_of(low_boosted))[0]
    features_high = metric.compute(batch_of(high_boosted))[0]

    alpha_change = (features_high["alphaRatio_sma3_all_mean"]
                    - features_low["alphaRatio_sma3_all_mean"])
    hammarberg_change = (features_high["hammarbergIndex_sma3_all_mean"]
                         - features_low["hammarbergIndex_sma3_all_mean"])
    assert alpha_change > 0 and hammarberg_change < 0


# ---------------------------------------------------------------------------
# Level behaviour — the property the whole family is valued for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("peak", [0.01, 0.1, 0.5, 1.0])
def test_tilt_indices_are_level_invariant_within_full_scale(metric, peak):
    """Gain must not move the tilt: this is what a gain-only baseline cannot fake."""
    signal = shaped_noise(lambda f: np.where(f < 1000, 10.0, 1.0), peak=1.0)
    reference = metric.compute(batch_of(signal * 0.5))[0]
    scaled = metric.compute(batch_of(signal * peak))[0]

    assert scaled["alphaRatio_sma3_all_mean"] == pytest.approx(
        reference["alphaRatio_sma3_all_mean"], abs=0.3
    )
    assert scaled["hammarbergIndex_sma3_all_mean"] == pytest.approx(
        reference["hammarbergIndex_sma3_all_mean"], abs=0.3
    )


def test_above_full_scale_the_indices_are_corrupted_and_flagged(metric):
    """openSMILE's documented input domain is [-1, +1]; beyond it, results are
    wrong by many dB and it says nothing.  The metric must flag the condition."""
    signal = shaped_noise(lambda f: np.where(f < 1000, 10.0, 1.0), peak=1.0)
    in_range = metric.compute(batch_of(signal * 0.5))[0]
    clipped = metric.compute(batch_of(signal * 5.0))[0]

    assert in_range["above_full_scale"] is False
    assert clipped["above_full_scale"] is True
    # Not a subtle drift — the measurement is destroyed.
    assert abs(clipped["alphaRatio_sma3_all_mean"]
               - in_range["alphaRatio_sma3_all_mean"]) > 5.0


def test_level_normalisation_keeps_the_signal_inside_full_scale():
    """The default requirements are what prevent the failure above."""
    assert EGeMAPSMetric().requirements.level_norm_dbfs == -27.0
    assert EGeMAPSMetric().requirements.sample_rate == 16_000
    assert EGeMAPSMetric(level_norm_dbfs=None).requirements.level_norm_dbfs is None


# ---------------------------------------------------------------------------
# Frame sets
# ---------------------------------------------------------------------------


def test_voicing_mask_finds_the_voiced_half(metric):
    features = metric.compute(batch_of(harmonic_then_silence()))[0]
    assert features["voiced_fraction"] == pytest.approx(0.5, abs=0.1)
    assert features["n_voiced_frames"] > 0
    assert features["n_frames"] > features["n_voiced_frames"]


def test_f0_is_recovered_from_the_harmonic_complex(metric):
    """A sanity check on the instrument: 120 Hz is semitone 25.5 above 27.5 Hz."""
    features = metric.compute(batch_of(harmonic_then_silence(f0=120.0)))[0]
    semitones = features["F0semitoneFrom27.5Hz_sma3nz_voiced_mean"]
    assert 27.5 * 2 ** (semitones / 12) == pytest.approx(120.0, abs=3.0)


def test_voiced_only_descriptors_get_no_all_frame_aggregate(metric):
    """openSMILE writes a literal 0.0 into _sma3nz descriptors on unvoiced frames.

    Aggregating those over every frame would not thin the sample, it would drag
    the mean toward zero in proportion to the silence in the segment.  So those
    columns must not exist at all.
    """
    features = metric.compute(batch_of(harmonic_then_silence()))[0]
    assert "jitterLocal_sma3nz_voiced_mean" in features
    assert "jitterLocal_sma3nz_all_mean" not in features
    assert "jitterLocal_sma3nz_unvoiced_mean" not in features
    # …while descriptors defined everywhere keep all three.
    for frame_set in ("voiced", "unvoiced", "all"):
        assert f"alphaRatio_sma3_{frame_set}_mean" in features


def test_voiced_aggregate_excludes_the_unvoiced_zeros(metric):
    """The mean over voiced frames must not be pulled toward zero by silence."""
    features = metric.compute(batch_of(harmonic_then_silence()))[0]
    hnr = features["HNRdBACF_sma3nz_voiced_mean"]
    assert np.isfinite(hnr) and hnr != 0.0


# ---------------------------------------------------------------------------
# Guards and configuration
# ---------------------------------------------------------------------------


def test_digital_silence_yields_nan_not_zero(metric):
    """openSMILE reports alphaRatio = 0.0 for silence, which is a number, not a
    measurement; letting it through would bias every mean computed over it."""
    features = metric.compute(batch_of(np.zeros(2 * SAMPLE_RATE, dtype="float32")))[0]
    assert np.isnan(features["alphaRatio_sma3_all_mean"])
    assert np.isnan(features["voiced_fraction"])


def test_signal_too_short_to_frame_yields_nan(metric):
    """Under one 20 ms window openSMILE fills with NaN and warns."""
    tone = 0.3 * np.sin(2 * np.pi * 200 * np.arange(SAMPLE_RATE // 100) / SAMPLE_RATE)
    features = metric.compute(batch_of(tone.astype("float32")))[0]
    assert np.isnan(features["alphaRatio_sma3_all_mean"])


def test_batch_results_are_one_per_utterance_and_in_order(metric):
    flat = shaped_noise(lambda f: np.ones_like(f))
    low = shaped_noise(lambda f: np.where(f < 1000, 10.0, 1.0))
    results = metric.compute(batch_of(flat, low))
    assert len(results) == 2
    assert results[0]["alphaRatio_sma3_all_mean"] > results[1]["alphaRatio_sma3_all_mean"]


def test_descriptor_subset_limits_the_emitted_columns():
    metric = EGeMAPSMetric(descriptors=["alphaRatio_sma3"], level_norm_dbfs=None)
    metric.load(torch.device("cpu"))
    features = metric.compute(batch_of(shaped_noise(lambda f: np.ones_like(f))))[0]
    assert "alphaRatio_sma3_all_mean" in features
    assert "hammarbergIndex_sma3_all_mean" not in features


def test_contours_are_emitted_as_arrays_for_the_artifact_archive():
    metric = EGeMAPSMetric(contours=["alphaRatio_sma3"], level_norm_dbfs=None)
    metric.load(torch.device("cpu"))
    features = metric.compute(batch_of(shaped_noise(lambda f: np.ones_like(f))))[0]
    contour = features["contour.alphaRatio_sma3"]
    assert isinstance(contour, np.ndarray)
    assert contour.size == features["n_frames"]


def test_provenance_is_recorded_per_row(metric):
    """v01a, v01b and v02 differ, so the version is part of the measurement."""
    assert metric.compute(batch_of(shaped_noise(lambda f: np.ones_like(f))))[0][
        "feature_set"
    ] == "eGeMAPSv02"


def test_unknown_statistic_is_rejected_at_construction():
    with pytest.raises(ValueError, match="Unknown statistic"):
        EGeMAPSMetric(statistics=["mean", "kurtosis"])


def test_unknown_descriptor_is_rejected_at_load():
    metric = EGeMAPSMetric(descriptors=["notADescriptor"])
    with pytest.raises(ValueError, match="not in eGeMAPSv02"):
        metric.load(torch.device("cpu"))


def test_registered_and_buildable_from_config():
    metric = build_metric({"type": "egemaps", "name": "tilt", "statistics": ["mean"]})
    assert isinstance(metric, EGeMAPSMetric)
    assert metric.name == "tilt"
    assert metric.statistics == ("mean",)


def test_compute_before_load_is_an_error():
    with pytest.raises(RuntimeError, match="load\\(\\) must be called"):
        EGeMAPSMetric().compute(batch_of(np.zeros(1000, dtype="float32")))
