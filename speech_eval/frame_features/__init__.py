"""Frame-level features on an imposed FrameGrid (see :mod:`speech_eval.frame_features.core`).

Importing this package registers every built-in frame feature, so
``build_frame_feature`` can resolve a config's ``type``.
"""
from __future__ import annotations

from speech_eval.frame_features import spectral_tilt  # noqa: F401
