"""FrameFeature base class and registry."""
from __future__ import annotations

import numpy as np
import pytest

import speech_eval  # noqa: F401  (registers the built-in frame features)
from audio_utils.data.frame_grid import FrameGrid
from speech_eval.frame_features import core
from speech_eval.frame_features.core import (
    FrameFeature,
    available_frame_features,
    build_frame_feature,
    build_frame_features,
    register_frame_feature,
)

GRID = FrameGrid(hop_size=320, sample_rate=16_000)


class _WrongShape(FrameFeature):
    columns = ("a",)
    feature_type = "_wrong_shape"

    def params(self):
        return {}

    def _compute(self, wave, grid):
        return np.zeros((1, 2))


def test_import_speech_eval_registers_spectral_tilt():
    assert "spectral_tilt" in available_frame_features()


def test_unknown_type_lists_the_available_ones():
    with pytest.raises(KeyError, match="spectral_tilt"):
        build_frame_feature({"type": "nope"})


def test_empty_registry_names_the_bypassed_import(monkeypatch):
    monkeypatch.setattr(core, "_REGISTRY", {})
    with pytest.raises(KeyError, match="import was bypassed"):
        build_frame_feature({"type": "spectral_tilt"})


def test_missing_type_key_is_reported():
    with pytest.raises(KeyError, match="no 'type' key"):
        build_frame_feature({"name": "x"})


def test_registering_a_type_twice_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        register_frame_feature("spectral_tilt")(_WrongShape)


def test_duplicate_names_are_refused():
    with pytest.raises(ValueError, match="Duplicate"):
        build_frame_features([{"type": "spectral_tilt"}, {"type": "spectral_tilt"}])
    built = build_frame_features(
        [{"type": "spectral_tilt"}, {"type": "spectral_tilt", "name": "tilt_50hz",
                                     "alpha_low_band_hz": (50, 1000)}]
    )
    assert [f.name for f in built] == ["spectral_tilt", "tilt_50hz"]


def test_multichannel_input_is_refused():
    with pytest.raises(ValueError, match="mono"):
        build_frame_feature({"type": "spectral_tilt"}).compute(np.zeros((2, 16_000)), GRID)


def test_a_subclass_returning_the_wrong_shape_is_caught():
    with pytest.raises(RuntimeError, match="expected"):
        _WrongShape().compute(np.zeros(16_000), GRID)
