"""Frame-level spectral tilt: alpha ratio and Hammarberg index.

Definitions
-----------
* **Alpha ratio** — Frøkjær-Jensen & Prytz (1975), ``frokjaer1975registration``:
  ``log α = log A(above 1000 Hz) − log A(below 1000 Hz)``.  Their analog chain
  (1 kHz high-/low-pass filters, rectifier, integrator, log) sampled it 25 times
  per second on voiced speech.  It rises with vocal effort.
* **Hammarberg index** — Hammarberg et al. (1980), ``hammarberg1980perceptual``:
  the maximum level in 0–2 kHz minus the maximum level in 2–5 kHz, read off a
  ~40 s long-term average spectrum (51 filters, 250 Hz wide).  It falls with
  vocal effort.

Neither paper fixes a frame-level, FFT-domain recipe.  This module computes
openSMILE's (``eyben2016geneva``), read from its source
(``src/lldcore/spectral.cpp``, ``src/dspcore/transformFft.cpp``,
``smileDsp_winHam``):

* symmetric Hamming window ``0.54 − 0.46·cos(2πn/(N−1))``, no pre-emphasis,
  no DC removal;
* FFT zero-padded to the next power of two (400 samples → 512 at 16 kHz), power
  spectrum ``|X|²``, bin ``k`` at ``k·sr/n_fft`` Hz;
* ``alpha_ratio_db = 10·log10(Σ P[1000 ≤ f ≤ 5000] / Σ P[0 ≤ f < 1000])`` —
  bins are **summed**, not averaged, so a flat spectrum reads about +6 dB
  (129 high-band bins against 32 low-band bins at 31.25 Hz spacing), and the
  low band includes the DC bin;
* ``hammarberg_db = 10·log10(max P[0 ≤ f < 2000] / max P[2000 ≤ f ≤ 5000])``
  over raw FFT bins.

Band membership follows openSMILE's loop: a band ``(lo, hi)`` passed as the
*low* band means ``lo ≤ f < hi``, as the *high* band ``lo ≤ f ≤ hi``.

Deviations from openSMILE's eGeMAPS output
------------------------------------------
* The frame grid is the caller's (25 ms windows on P_φ's 20 ms grid, frame ``i``
  centred on sample ``i·hop``) instead of 20 ms windows every 10 ms starting at
  ``t = 0``; the framing is :func:`speech_eval.frame_features.framing.frame_signal`.
* No 3-frame moving average (eGeMAPS' ``sma3``).
* A band holding zero power (an all-zero frame) gives ``NaN``, where openSMILE
  emits ``0.0`` or substitutes its spectral floor — a number that is not a
  measurement.
* Zero padding is appended at the end instead of openSMILE's symmetric padding;
  the magnitudes are identical, only the phase differs.

The lower edge of the alpha low band is a parameter
(``alpha_low_band_hz=(50, 1000)`` drops the bins below 50 Hz); its default is
openSMILE's.

Known limitation of the raw-bin maximum
---------------------------------------
A harmonic falling between two bins is read low (scalloping).  With the
Hamming window zero-padded from 400 to 512 points the worst case is
−1.06 dB (half-bin offset), measured in ``tests/test_frame_spectral_tilt.py``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from audio_utils.data.frame_grid import FrameGrid
from speech_eval.frame_features.core import FrameFeature, register_frame_feature
from speech_eval.frame_features.framing import frame_signal

Band = tuple[float, float]


def hamming(n: int) -> np.ndarray:
    """openSMILE's ``smileDsp_winHam``: ``0.54 − 0.46·cos(2πn/(N−1))``."""
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def next_power_of_two(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def band_mask(freqs: np.ndarray, band: Band, *, upper_inclusive: bool) -> np.ndarray:
    lo, hi = band
    upper = freqs <= hi if upper_inclusive else freqs < hi
    return (freqs >= lo) & upper


@register_frame_feature("spectral_tilt")
class SpectralTilt(FrameFeature):
    """``alpha_ratio_db`` and ``hammarberg_db`` per frame (see module docstring)."""

    columns = ("alpha_ratio_db", "hammarberg_db")

    def __init__(
        self,
        name: str | None = None,
        sample_rate: int = 16_000,
        window_length: int = 400,
        n_fft: int | None = None,
        alpha_low_band_hz: Band = (0.0, 1000.0),
        alpha_high_band_hz: Band = (1000.0, 5000.0),
        hammarberg_low_band_hz: Band = (0.0, 2000.0),
        hammarberg_high_band_hz: Band = (2000.0, 5000.0),
    ):
        super().__init__(name)
        self.sample_rate = int(sample_rate)
        self.window_length = int(window_length)
        self.n_fft = int(n_fft) if n_fft is not None else next_power_of_two(self.window_length)
        if self.n_fft < self.window_length:
            raise ValueError(f"n_fft={self.n_fft} is shorter than window_length={self.window_length}.")
        self.alpha_low_band_hz = tuple(float(x) for x in alpha_low_band_hz)
        self.alpha_high_band_hz = tuple(float(x) for x in alpha_high_band_hz)
        self.hammarberg_low_band_hz = tuple(float(x) for x in hammarberg_low_band_hz)
        self.hammarberg_high_band_hz = tuple(float(x) for x in hammarberg_high_band_hz)

        self.window = hamming(self.window_length)
        self.freqs = np.arange(self.n_fft // 2 + 1) * self.sample_rate / self.n_fft
        nyquist = self.sample_rate / 2
        self._masks = {}
        for key, band, upper_inclusive in (
            ("alpha_low", self.alpha_low_band_hz, False),
            ("alpha_high", self.alpha_high_band_hz, True),
            ("hammarberg_low", self.hammarberg_low_band_hz, False),
            ("hammarberg_high", self.hammarberg_high_band_hz, True),
        ):
            lo, hi = band
            if not 0.0 <= lo < hi <= nyquist:
                raise ValueError(f"{key} band {band} Hz is not inside [0, {nyquist}] Hz.")
            mask = band_mask(self.freqs, band, upper_inclusive=upper_inclusive)
            if not mask.any():
                raise ValueError(f"{key} band {band} Hz contains no FFT bin at n_fft={self.n_fft}.")
            self._masks[key] = mask

    def params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample_rate": self.sample_rate,
            "window_length": self.window_length,
            "n_fft": self.n_fft,
            "alpha_low_band_hz": self.alpha_low_band_hz,
            "alpha_high_band_hz": self.alpha_high_band_hz,
            "hammarberg_low_band_hz": self.hammarberg_low_band_hz,
            "hammarberg_high_band_hz": self.hammarberg_high_band_hz,
        }

    def power_spectrum(self, wave: np.ndarray, grid: FrameGrid) -> np.ndarray:
        """``(n_frames, n_fft // 2 + 1)`` power spectrum of the windowed frames."""
        frames = frame_signal(wave, grid, self.window_length) * self.window
        return np.abs(np.fft.rfft(frames, n=self.n_fft, axis=-1)) ** 2

    def tilt_from_power(self, power: np.ndarray) -> np.ndarray:
        """``(..., 2)`` alpha ratio and Hammarberg index of ``(..., n_fft // 2 + 1)`` power spectra.

        Works on one frame's spectrum or on a long-term average spectrum: the mean power
        of several frames, as in Hammarberg et al. (1980).  The tilt of an averaged
        spectrum is not the mean of the frames' tilts (log of a mean ≠ mean of logs).
        """
        power = np.asarray(power)
        if power.shape[-1] != self.freqs.size:
            raise ValueError(f"expected {self.freqs.size} bins on the last axis, got {power.shape[-1]}.")
        m = self._masks
        alpha = log_ratio(power[..., m["alpha_high"]].sum(-1), power[..., m["alpha_low"]].sum(-1))
        hammarberg = log_ratio(
            power[..., m["hammarberg_low"]].max(-1, initial=0.0),
            power[..., m["hammarberg_high"]].max(-1, initial=0.0),
        )
        return np.stack([alpha, hammarberg], axis=-1)

    def _compute(self, wave: np.ndarray, grid: FrameGrid) -> np.ndarray:
        return self.tilt_from_power(self.power_spectrum(wave, grid))


def log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``10·log10(numerator / denominator)``, NaN wherever either side holds no power."""
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    valid =(numerator > 0.0) & (denominator > 0.0)
    out = np.full(numerator.shape, np.nan)
    out[valid] = 10.0 * np.log10(numerator[valid] / denominator[valid])
    return out
