"""Shared fixtures: a tiny on-disk corpus and a manifest describing it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

SAMPLE_RATE = 16_000


def write_tone(path, duration_s: float, amplitude: float, sr: int = SAMPLE_RATE):
    """A 200 Hz sine of known duration and amplitude, so levels are predictable."""
    t = np.arange(int(duration_s * sr)) / sr
    sf.write(path, (amplitude * np.sin(2 * np.pi * 200 * t)).astype("float32"), sr)


@pytest.fixture
def corpus(tmp_path):
    """Three files, two conditions of one group plus one of another."""
    root = tmp_path / "audio"
    root.mkdir()
    write_tone(root / "a_real.wav", 1.0, 0.10)
    write_tone(root / "a_conv.wav", 1.0, 0.40)
    write_tone(root / "b_real.wav", 0.5, 0.20)
    return root


@pytest.fixture
def manifest_path(tmp_path, corpus):
    path = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"utt_id": "a_real", "signal_path": "a_real.wav", "group_id": "a",
         "condition": "real", "corpus": "test", "subject_id": "s1", "text": "hello"},
        {"utt_id": "a_conv", "signal_path": "a_conv.wav", "group_id": "a",
         "condition": "converted", "corpus": "test", "subject_id": "s1", "text": ""},
        {"utt_id": "b_real", "signal_path": "b_real.wav", "group_id": "b",
         "condition": "real", "corpus": "test", "subject_id": "s2", "text": "world"},
    ]).to_csv(path, index=False)
    return path
