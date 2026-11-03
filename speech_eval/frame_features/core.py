"""Core types for frame-level features: what a frame feature is, and its registry.

Design
------
A :class:`FrameFeature` turns one utterance into one row per frame of a
:class:`~audio_utils.data.frame_grid.FrameGrid` imposed from outside — the grid
of the model the features will be fed to.  That is what separates it from a
:class:`speech_eval.core.Metric`, which summarises an utterance and, when it
emits a contour at all, does so on its backend's own grid.

Each feature keeps its own analysis window; only the frame centres are shared.
A value that cannot be measured on a frame (no voicing, digital silence) is
``NaN``: what to impute is the consumer's decision, not the extractor's.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from audio_utils.data.frame_grid import FrameGrid


class FrameFeature(ABC):
    """Base class for every frame-level feature.

    Subclasses set :attr:`columns` and :attr:`sample_rate`, implement
    :meth:`params` and :meth:`_compute`, and optionally override :meth:`load`.
    """

    #: Names of the output columns, in order.  Overridden by subclasses.
    columns: tuple[str, ...] = ()
    #: Sample rate the feature is defined at.  The caller resamples.
    sample_rate: int = 16_000

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.feature_type  # type: ignore[attr-defined]
        self._loaded = False

    def load(self) -> None:
        """Build a backend.  Called at most once, through :meth:`ensure_loaded`."""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Every parameter that changes the output, as constructor keyword arguments.

        ``build_frame_feature({"type": ..., **params()})`` must rebuild an
        identical feature: this dict is the provenance stored with the values.
        """

    def compute(self, wave: np.ndarray, grid: FrameGrid) -> np.ndarray:
        """Return a ``(grid.n_frames(len(wave)), len(columns))`` float32 array."""
        wave = np.asarray(wave)
        if wave.ndim != 1:
            raise ValueError(f"{self.name}: expected a mono 1-D waveform, got shape {wave.shape}.")
        if grid.sample_rate != self.sample_rate:
            raise ValueError(
                f"{self.name}: grid sample rate {grid.sample_rate} Hz differs from the "
                f"feature's {self.sample_rate} Hz; resample the audio and build the grid "
                "at the feature's rate."
            )
        self.ensure_loaded()
        out = np.asarray(self._compute(wave.astype(np.float64), grid), dtype=np.float32)
        expected = (grid.n_frames(len(wave)), len(self.columns))
        if out.shape != expected:
            raise RuntimeError(f"{self.name}: produced shape {out.shape}, expected {expected}.")
        return out

    @abstractmethod
    def _compute(self, wave: np.ndarray, grid: FrameGrid) -> np.ndarray:
        """Feature values for a validated float64 mono waveform."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[FrameFeature]] = {}


def register_frame_feature(feature_type: str) -> Callable[[type[FrameFeature]], type[FrameFeature]]:
    """Class decorator registering a frame feature under a config-visible type name."""

    def decorator(cls: type[FrameFeature]) -> type[FrameFeature]:
        if feature_type in _REGISTRY:
            raise ValueError(f"Frame feature type {feature_type!r} is already registered.")
        cls.feature_type = feature_type  # type: ignore[attr-defined]
        _REGISTRY[feature_type] = cls
        return cls

    return decorator


def available_frame_features() -> list[str]:
    return sorted(_REGISTRY)


def build_frame_feature(spec: dict[str, Any]) -> FrameFeature:
    """Instantiate one frame feature from ``{"type": ..., "name": ..., **kwargs}``."""
    spec = dict(spec)
    try:
        feature_type = spec.pop("type")
    except KeyError:
        raise KeyError(f"Frame feature config entry has no 'type' key: {spec}") from None
    if feature_type not in _REGISTRY:
        if not _REGISTRY:
            raise KeyError(
                f"Unknown frame feature type {feature_type!r} because NO frame feature is "
                "registered. The registry is populated by importing "
                "`speech_eval.frame_features` (which `import speech_eval` does for you), "
                "so an empty list means that import was bypassed."
            )
        raise KeyError(
            f"Unknown frame feature type {feature_type!r}. "
            f"Available: {available_frame_features()}"
        )
    return _REGISTRY[feature_type](**spec)


def build_frame_features(specs: list[dict[str, Any]]) -> list[FrameFeature]:
    """Instantiate a list of frame features, rejecting duplicate names."""
    features = [build_frame_feature(s) for s in specs]
    names = [f.name for f in features]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(
            f"Duplicate frame feature names {sorted(duplicates)}. Give each entry a "
            "distinct 'name' when using the same feature type more than once."
        )
    return features
