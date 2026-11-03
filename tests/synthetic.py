"""Source-filter vowel synthesis shared by the formant and spectral-tilt tests.

The signals are full source-filter syntheses — pulse train, source tilt,
resonator cascade, lip radiation — so the vocal tract and the glottal source are
both known by construction.  Without lip radiation the signal is a glottal
spectrum rather than a speech spectrum, and *every* LPC analyser spends its poles
modelling the rolloff instead of the formants.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000

# Five poles below the ceiling, because that is what an LPC analyser configured
# for five formants expects to find in speech.
VOWELS = {
    "schwa": (500.0, 1500.0, 2500.0, 3500.0, 4500.0),
    "open_a": (700.0, 1100.0, 2600.0, 3600.0, 4600.0),
    "close_i": (350.0, 2200.0, 2900.0, 3700.0, 4600.0),
}
BANDWIDTHS = (60.0, 90.0, 120.0, 180.0, 250.0)


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
    of flattening vocal effort produces.
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
