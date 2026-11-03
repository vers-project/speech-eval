"""Frame-level spectral tilt: framing, known answers, and openSMILE as the reference."""
from __future__ import annotations

import numpy as np
import pytest

from audio_utils.data.frame_grid import FrameGrid
from speech_eval.frame_features.core import build_frame_feature
from speech_eval.frame_features.framing import frame_signal
from speech_eval.frame_features.spectral_tilt import SpectralTilt, hamming

SR = 16_000
GRID = FrameGrid(hop_size=320, sample_rate=SR)  # P_φ's grid
BIN_HZ = SR / 512


def tones(*components: tuple[float, float], duration_s: float = 1.0) -> np.ndarray:
    t = np.arange(int(duration_s * SR)) / SR
    return sum(a * np.cos(2 * np.pi * f * t + 0.3 * i) for i, (f, a) in enumerate(components))


def interior(values: np.ndarray, column: int) -> np.ndarray:
    """Drop the two edge frames, whose windows are partly zero padding."""
    return values[2:-2, column]


@pytest.fixture(scope="module")
def tilt():
    return SpectralTilt()


# -- framing ----------------------------------------------------------------


def test_frame_count_follows_the_grid(tilt):
    assert tilt.compute(np.ones(16_000), GRID).shape == (50, 2)
    assert tilt.compute(np.ones(16_319), GRID).shape == (50, 2)


@pytest.mark.parametrize("s", [0, 199, 200, 5_000, 15_999])
def test_non_causal_frame_i_covers_i_hops_plus_minus_half_a_window(s):
    wave = np.zeros(16_000)
    wave[s] = 1.0
    hit = np.flatnonzero(frame_signal(wave, GRID, 400).any(-1))
    expected = [i for i in range(50) if i * 320 - 200 <= s < i * 320 + 200]
    assert hit.tolist() == expected


@pytest.mark.parametrize("s", [0, 159, 160, 5_000])
def test_causal_frame_i_starts_at_i_hops_like_opensmile_left(s):
    wave = np.zeros(16_000)
    wave[s] = 1.0
    grid = FrameGrid(hop_size=160, sample_rate=SR, causal=True)
    hit = np.flatnonzero(frame_signal(wave, grid, 320).any(-1))
    expected = [i for i in range(100) if i * 160 <= s < i * 160 + 320]
    assert hit.tolist() == expected


# -- window and bands ---------------------------------------------------------


def test_window_is_opensmiles_symmetric_hamming(tilt):
    n = np.arange(400)
    np.testing.assert_allclose(tilt.window, 0.54 - 0.46 * np.cos(2 * np.pi * n / 399))
    np.testing.assert_allclose(hamming(400), np.hamming(400))
    assert tilt.n_fft == 512


def test_band_edges_follow_opensmiles_loop(tilt):
    f = tilt.freqs
    low, high = tilt._masks["alpha_low"], tilt._masks["alpha_high"]
    assert low[f == 0.0].all()                       # DC is in the low band
    assert high[f == 1000.0].all() and not low[f == 1000.0].any()
    assert high[f == 5000.0].all() and not high[f > 5000.0].any()
    assert low.sum() == 32 and high.sum() == 129
    assert tilt._masks["hammarberg_high"][f == 2000.0].all()


def test_low_band_lower_edge_is_a_parameter():
    tilt50 = SpectralTilt(alpha_low_band_hz=(50, 1000))
    assert not tilt50._masks["alpha_low"][tilt50.freqs < 50].any()
    assert tilt50._masks["alpha_low"].sum() == 30   # bins at 0 and 31.25 Hz dropped


def test_a_band_without_any_bin_is_refused():
    with pytest.raises(ValueError, match="no FFT bin"):
        SpectralTilt(alpha_low_band_hz=(1, 20))


# -- known answers ----------------------------------------------------------------


def test_alpha_ratio_of_two_tones_is_their_level_difference(tilt):
    a, b = 0.5, 0.05
    out = tilt.compute(tones((8 * BIN_HZ, a), (80 * BIN_HZ, b)), GRID)  # 250 Hz, 2500 Hz
    np.testing.assert_allclose(interior(out, 0), 20 * np.log10(b / a), atol=0.1)


@pytest.mark.parametrize("k", [16, 96])
def test_an_on_bin_tone_peaks_at_its_windowed_amplitude(tilt, k):
    power = tilt.power_spectrum(tones((k * BIN_HZ, 0.3)), GRID)[2:-2]
    assert (power.argmax(-1) == k).all()
    np.testing.assert_allclose(power[:, k], (0.3 * tilt.window.sum() / 2) ** 2, rtol=1e-4)


def test_hammarberg_of_two_on_bin_tones_is_their_level_difference(tilt):
    """Exact up to the stronger tone's window side-lobes in the weaker tone's bin.

    That interference depends on the tones' relative phase: measured ±0.052 dB at
    a 28 dB level gap, ±0.01 dB at the 14 dB gap used here.
    """
    a, b = 0.5, 0.1
    out = tilt.compute(tones((16 * BIN_HZ, a), (96 * BIN_HZ, b)), GRID)  # 500 Hz, 3000 Hz
    np.testing.assert_allclose(interior(out, 1), 20 * np.log10(a / b), atol=0.02)


def test_raw_bin_maximum_scallops_by_about_one_db_half_way_between_bins(tilt):
    on = tilt.compute(tones((96 * BIN_HZ, 1.0), (16 * BIN_HZ, 1.0)), GRID)
    off = tilt.compute(tones((96.5 * BIN_HZ, 1.0), (16 * BIN_HZ, 1.0)), GRID)
    loss = np.median(interior(off, 1) - interior(on, 1))   # Hammarberg rises as the 3 kHz peak drops
    assert loss == pytest.approx(1.06, abs=0.05)


def test_flat_spectrum_alpha_is_the_bin_count_ratio(tilt):
    noise = np.random.default_rng(0).standard_normal(10 * SR)
    alpha = tilt.compute(noise, GRID)[:, 0]
    assert np.median(alpha) == pytest.approx(10 * np.log10(129 / 32), abs=0.3)


def test_boosting_the_high_band_raises_alpha_and_lowers_hammarberg(tilt):
    dull = tilt.compute(tones((300, 0.5), (3000, 0.01)), GRID)
    bright = tilt.compute(tones((300, 0.5), (3000, 0.1)), GRID)
    assert (interior(bright, 0) > interior(dull, 0)).all()
    assert (interior(bright, 1) < interior(dull, 1)).all()


@pytest.mark.parametrize("gain", [0.01, 10.0])
def test_both_indices_are_gain_invariant(tilt, gain):
    x = tones((200, 0.3), (1500, 0.1), (3500, 0.02)) + 0.001 * np.random.default_rng(1).standard_normal(SR)
    np.testing.assert_allclose(tilt.compute(gain * x, GRID), tilt.compute(x, GRID), atol=1e-4)


def test_digital_silence_is_nan(tilt):
    assert np.isnan(tilt.compute(np.zeros(SR), GRID)).all()


def test_only_all_zero_frames_are_nan(tilt):
    x = tones((200, 0.3), (2500, 0.1))
    x[4_000:12_000] = 0.0
    out = tilt.compute(x, GRID)
    silent = ~frame_signal(x, GRID, 400).any(-1)
    assert silent.any() and (~silent).any()
    assert np.isnan(out[silent]).all()
    assert np.isfinite(out[~silent]).all()


# -- FrameFeature contract --------------------------------------------------------


def test_output_is_float32_with_one_row_per_frame(tilt):
    out = tilt.compute(np.random.default_rng(2).standard_normal(SR), GRID)
    assert out.dtype == np.float32 and out.shape == (50, len(tilt.columns))


def test_params_rebuild_an_identical_feature(tilt):
    x = np.random.default_rng(3).standard_normal(SR)
    rebuilt = build_frame_feature({"type": "spectral_tilt", **tilt.params()})
    np.testing.assert_array_equal(rebuilt.compute(x, GRID), tilt.compute(x, GRID))


def test_grid_at_another_sample_rate_is_refused(tilt):
    with pytest.raises(ValueError, match="sample rate"):
        tilt.compute(np.zeros(SR), FrameGrid(hop_size=441, sample_rate=22_050))


# -- tilt from a given power spectrum -------------------------------------------------


def two_bin_power(tilt, low: float, high: float) -> np.ndarray:
    """A spectrum with power only at 250 Hz (``low``) and 2500 Hz (``high``)."""
    power = np.zeros(tilt.freqs.size)
    power[8], power[80] = low, high
    return power


def test_tilt_from_frame_spectra_is_the_frame_feature(tilt):
    x = tones((200, 0.3), (1500, 0.1), (3500, 0.02)) + 0.001 * np.random.default_rng(4).standard_normal(SR)
    from_power = tilt.tilt_from_power(tilt.power_spectrum(x, GRID))
    np.testing.assert_array_equal(from_power.astype(np.float32), tilt.compute(x, GRID))


def test_a_one_dimensional_spectrum_gives_one_alpha_and_one_hammarberg(tilt):
    out = tilt.tilt_from_power(two_bin_power(tilt, 1.0, 0.01))
    assert out.shape == (2,)
    np.testing.assert_allclose(out, [-20.0, 20.0])


def test_averaging_identical_frames_keeps_their_tilt(tilt):
    power = tilt.power_spectrum(tones((300, 0.5), (3000, 0.05)), GRID)[10]
    averaged = np.tile(power, (7, 1)).mean(0)
    np.testing.assert_allclose(tilt.tilt_from_power(averaged), tilt.tilt_from_power(power))


def test_tilt_of_the_averaged_spectrum_is_not_the_mean_of_frame_tilts(tilt):
    """High band at −20 dB in one frame and 0 dB in the other: the frames' tilts average
    to −10 dB, the averaged spectrum (high band 0.505) reads 10·log10(0.505) = −2.97 dB."""
    frames = np.stack([two_bin_power(tilt, 1.0, 0.01), two_bin_power(tilt, 1.0, 1.0)])
    per_frame = tilt.tilt_from_power(frames)
    np.testing.assert_allclose(per_frame.mean(0), [-10.0, 10.0])
    expected = 10 * np.log10(0.505)
    np.testing.assert_allclose(tilt.tilt_from_power(frames.mean(0)), [expected, -expected])


def test_a_spectrum_with_the_wrong_bin_count_is_refused(tilt):
    with pytest.raises(ValueError, match="bins"):
        tilt.tilt_from_power(np.ones(tilt.freqs.size - 1))


def test_an_all_zero_spectrum_is_nan(tilt):
    assert np.isnan(tilt.tilt_from_power(np.zeros(tilt.freqs.size))).all()


# -- openSMILE reference ------------------------------------------------------------


def test_matches_opensmile_egemaps_when_configured_like_it():
    """Our feature with eGeMAPS' framing + its 3-frame moving average = openSMILE's LLDs.

    Measured when written: median |Δ| 0.0005 dB (alpha) / 0.0008 dB (Hammarberg),
    max 0.007 dB — float32 precision.  A one-frame lag already costs 0.09 dB.
    """
    opensmile = pytest.importorskip("opensmile", reason="needs the `phonetics` extra")
    rng = np.random.default_rng(0)
    t = np.arange(3 * SR) / SR
    phase = 2 * np.pi * np.cumsum(140 + 30 * np.sin(2 * np.pi * 0.7 * t)) / SR
    x = sum(k ** -1.5 * np.sin(k * phase) for k in range(1, 40))
    x = x * (0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * t)) + 0.01 * rng.standard_normal(len(t))
    x = (0.5 * x / np.abs(x).max()).astype(np.float32)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    reference = smile.process_signal(x, SR)[["alphaRatio_sma3", "hammarbergIndex_sma3"]].to_numpy()

    ours = SpectralTilt(window_length=320).compute(
        x, FrameGrid(hop_size=160, sample_rate=SR, causal=True)
    ).astype(np.float64)
    smoothed = (ours[:-2] + ours[1:-1] + ours[2:]) / 3          # frame i ↔ ours[i+1]
    n = min(len(reference), len(ours)) - 2
    diff = np.abs(smoothed[1 : n - 1] - reference[2:n])
    assert np.median(diff) < 0.002
    assert diff.max() < 0.02
