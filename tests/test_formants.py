"""Praat formant analysis, against vowels whose formants are known by construction.

The signals here are full source-filter syntheses — pulse train, source tilt,
resonator cascade, lip radiation — rather than the pure tones ``test_f0`` can get
away with.  That is not fussiness: without lip radiation the signal is a glottal
spectrum rather than a speech spectrum, and *every* LPC analyser spends its poles
modelling the rolloff instead of the formants.  A probe that omits it measures
the probe, not the tracker.

The tolerances are analytic — the resonator frequencies are inputs — so these
tests survive a Praat upgrade that moves a decimal.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_eval.core import Utterance, UtteranceBatch, build_metric
from speech_eval.metrics.formants import (
    DEFAULT_FORMANT_CEILING,
    FORMANTS,
    SEX_FORMANT_CEILINGS,
)

pytest.importorskip("parselmouth", reason="needs the `phonetics` extra")

SAMPLE_RATE = 16_000

# Praat lands within ~3% of truth on these signals; 10% leaves room for a version
# to move without the test becoming a tripwire, while still failing loudly on the
# formant-relabelling failure this metric exists to avoid (which is a factor of 2+).
RELATIVE_TOLERANCE = 0.10

# Five poles below the ceiling, because that is what an LPC analyser configured
# for five formants expects to find in speech.
VOWELS = {
    "schwa": (500.0, 1500.0, 2500.0, 3500.0, 4500.0),
    "open_a": (700.0, 1100.0, 2600.0, 3600.0, 4600.0),
    "close_i": (350.0, 2200.0, 2900.0, 3700.0, 4600.0),
}
BANDWIDTHS = (60.0, 90.0, 120.0, 180.0, 250.0)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def _resonator(frequency: float, bandwidth: float, freqs: np.ndarray) -> np.ndarray:
    """Frequency response of a 2-pole resonator, unity gain at DC."""
    r = np.exp(-np.pi * bandwidth / SAMPLE_RATE)
    theta = 2 * np.pi * frequency / SAMPLE_RATE
    a1, a2 = -2 * r * np.cos(theta), r ** 2
    z = np.exp(-2j * np.pi * freqs / SAMPLE_RATE)
    return (1 + a1 + a2) / (1 + a1 * z + a2 * z ** 2)


def vowel(
    formants=VOWELS["schwa"],
    f0: float = 120.0,
    tilt_db_per_octave: float = 12.0,
    duration_s: float = 1.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """A synthetic vowel with exactly the formants asked for.

    ``tilt_db_per_octave`` is the *source* tilt: 12 is a modal voice, 6 the sort
    of flattening vocal effort produces.  It is a parameter because a formant
    analyser that moves with it cannot separate the vocal tract from the glottal
    source, which is the whole argument for preferring Praat here.
    """
    n = int(duration_s * SAMPLE_RATE)
    source = np.zeros(n)
    source[:: int(round(SAMPLE_RATE / f0))] = 1.0

    spectrum = np.fft.rfft(source)
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    octaves = np.maximum(np.log2(np.maximum(freqs, 1.0) / 100.0), 0.0)
    spectrum = spectrum * 10 ** (-tilt_db_per_octave * octaves / 20.0)
    for frequency, bandwidth in zip(formants, BANDWIDTHS):
        spectrum = spectrum * _resonator(frequency, bandwidth, freqs)

    signal = np.fft.irfft(spectrum, n)
    # Lip radiation: a first difference, +6 dB/octave.  See the module docstring.
    signal = np.diff(signal, prepend=0.0)
    return (amplitude * signal / np.abs(signal).max()).astype("float32")


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
def metric():
    built = build_metric({"type": "formants_praat"})
    built.load(torch.device("cpu"))
    return built


def measure(metric, signal: np.ndarray, meta: dict | None = None):
    return metric.compute(batch_of(signal, meta=meta))[0]


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(VOWELS))
def test_recovers_known_formants(metric, name):
    truth = VOWELS[name]
    features = measure(metric, vowel(truth), meta={"sex": "M"})
    for number in FORMANTS:
        assert features[f"f{number}_median_hz"] == pytest.approx(
            truth[number - 1], rel=RELATIVE_TOLERANCE
        ), f"F{number} of {name}"


def test_bandwidths_are_emitted_and_plausible(metric):
    features = measure(metric, vowel(), meta={"sex": "M"})
    for number in FORMANTS:
        bandwidth = features[f"f{number}_bandwidth_hz"]
        # Not asserted against the synthesis bandwidths: LPC bandwidth estimates
        # are biased by the analysis window and are not the quantity this probe
        # knows exactly.  What is asserted is that they are real and finite.
        assert np.isfinite(bandwidth) and 0 < bandwidth < 1000


# ---------------------------------------------------------------------------
# The reason this metric exists: it must not move with vocal effort
# ---------------------------------------------------------------------------


def test_formants_do_not_follow_the_source_tilt(metric):
    """A flatter source is what vocal effort does, and the vocal tract here never
    moves.  openSMILE's tracker reads F1 at 336 Hz modal and 1197 Hz flat on this
    signal; Praat must not, or every effort comparison measures the source."""
    truth = VOWELS["schwa"]
    measured = {
        tilt: measure(metric, vowel(truth, tilt_db_per_octave=tilt), meta={"sex": "M"})
        for tilt in (16.0, 12.0, 8.0, 6.0)
    }
    for number in FORMANTS:
        values = [f[f"f{number}_median_hz"] for f in measured.values()]
        assert np.ptp(values) < 0.15 * truth[number - 1], (
            f"F{number} moved by {np.ptp(values):.0f} Hz across the source-tilt "
            f"sweep: {dict(zip(measured, np.round(values)))}"
        )
        for value in values:
            assert value == pytest.approx(truth[number - 1], rel=0.15)


def test_is_level_invariant(metric):
    """Praat's formant analysis does not depend on gain, so two takes of one
    signal at different levels must not differ — the suite compares conditions
    that differ in level by construction."""
    signal = vowel()
    loud = measure(metric, signal, meta={"sex": "M"})
    quiet = measure(metric, (signal * 10 ** (-20 / 20)).astype("float32"),
                    meta={"sex": "M"})
    for number in FORMANTS:
        assert loud[f"f{number}_median_hz"] == pytest.approx(
            quiet[f"f{number}_median_hz"], rel=1e-6
        )


# ---------------------------------------------------------------------------
# Settings are provenance, and depend on metadata rather than on the audio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sex, expected_ceiling, expected_source",
    [
        ("M", SEX_FORMANT_CEILINGS["m"], "sex"),
        ("female", SEX_FORMANT_CEILINGS["f"], "sex"),
        ("", DEFAULT_FORMANT_CEILING, "default"),
        (None, DEFAULT_FORMANT_CEILING, "default"),
        (float("nan"), DEFAULT_FORMANT_CEILING, "default"),
    ],
)
def test_ceiling_is_resolved_from_sex_alone(metric, sex, expected_ceiling,
                                            expected_source):
    features = measure(metric, vowel(), meta={"sex": sex})
    assert features["ceiling_hz"] == expected_ceiling
    assert features["ceiling_source"] == expected_source


def test_same_speaker_is_analysed_identically_whatever_the_signal(metric):
    """The ceiling must come from metadata only: a quiet take and a loud take of
    one speaker have to be measured with the same settings or the comparison
    between them is not a comparison."""
    modal = measure(metric, vowel(tilt_db_per_octave=12.0), meta={"sex": "F"})
    loud = measure(metric, vowel(tilt_db_per_octave=6.0, f0=240.0), meta={"sex": "F"})
    assert modal["ceiling_hz"] == loud["ceiling_hz"] == SEX_FORMANT_CEILINGS["f"]


def test_tracking_is_reported_per_row(metric):
    features = measure(metric, vowel(), meta={"sex": "M"})
    assert features["tracked"] is True


def test_untracked_instance_says_so():
    untracked = build_metric({"type": "formants_praat", "track": False})
    untracked.load(torch.device("cpu"))
    features = measure(untracked, vowel(), meta={"sex": "M"})
    assert features["tracked"] is False
    # Both settings must still measure the same vowel, or the flag would be
    # hiding two different instruments rather than two settings of one.
    for number in FORMANTS:
        assert features[f"f{number}_median_hz"] == pytest.approx(
            VOWELS["schwa"][number - 1], rel=RELATIVE_TOLERANCE
        )


def test_track_references_scale_with_the_ceiling(metric):
    """Praat's own published defaults, at Praat's own default 5500 Hz ceiling."""
    assert metric._references(5500.0) == pytest.approx(
        (550.0, 1650.0, 2750.0, 3850.0, 4950.0)
    )
    assert metric._references(5000.0) == pytest.approx(
        (500.0, 1500.0, 2500.0, 3500.0, 4500.0)
    )


# ---------------------------------------------------------------------------
# Unmeasurable input is data, not an error
# ---------------------------------------------------------------------------


def test_digital_silence_yields_nan_rather_than_a_number(metric):
    features = measure(metric, np.zeros(SAMPLE_RATE, dtype="float32"), meta={"sex": "M"})
    for number in FORMANTS:
        assert np.isnan(features[f"f{number}_median_hz"])
    assert features["n_voiced_frames"] == 0


def test_a_signal_too_short_to_analyse_does_not_raise(metric):
    features = measure(metric, vowel(duration_s=0.01), meta={"sex": "M"})
    for number in FORMANTS:
        assert np.isnan(features[f"f{number}_median_hz"])


def test_unvoiced_noise_is_not_summarised_as_formants(metric):
    """Praat's LPC fits poles to noise happily.  The voicing mask is what stops
    those poles being reported as though they were vowel formants."""
    rng = np.random.default_rng(0)
    noise = (rng.normal(0, 0.1, SAMPLE_RATE)).astype("float32")
    features = measure(metric, noise, meta={"sex": "M"})
    assert features["voiced_fraction"] < 0.2


# ---------------------------------------------------------------------------
# Contours
# ---------------------------------------------------------------------------


def test_contours_share_one_grid_and_mask_unvoiced_frames(metric):
    features = measure(metric, vowel(), meta={"sex": "M"})
    voiced = features["contour.voiced"]
    assert voiced.shape == (features["n_frames"],)
    for number in FORMANTS:
        contour = features[f"contour.f{number}_hz"]
        assert contour.shape == voiced.shape
        # Unvoiced frames carry NaN rather than Praat's own filler, so a contour
        # can never be read as a formant by accident.
        assert np.all(np.isnan(contour[~voiced]))


def test_frame_times_are_reconstructible_from_the_emitted_grid(metric):
    features = measure(metric, vowel(), meta={"sex": "M"})
    assert np.isfinite(features["t0_s"])
    assert features["time_step_s"] == pytest.approx(0.01, abs=1e-6)
    # Frame k sits at t0_s + k*time_step_s, which is what makes alignment with
    # the F0 metrics one explicit step downstream rather than an assumption.
    last = features["t0_s"] + (features["n_frames"] - 1) * features["time_step_s"]
    assert last < 1.0
