"""Cut a waveform into analysis windows sitting on a FrameGrid.

Reproduces the framing of VIC's frame-level SPL labels
(``vic.features.spl.FrameLevelTransform``) sample for sample, so a feature and
the label it is paired with are computed on the same samples:

* non-causal grid — zero-pad ``window_length // 2`` on both sides, so frame
  ``i`` covers ``[i·hop − window_length//2, i·hop + window_length − window_length//2)``;
* causal grid — pad ``(0, window_length − 1)``, so frame ``i`` covers
  ``[i·hop, i·hop + window_length)``, which is openSMILE's
  ``frameCenterSpecial = left``;

then keep the first ``grid.n_frames(len(wave))`` windows.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from audio_utils.data.frame_grid import FrameGrid


def frame_signal(wave: np.ndarray, grid: FrameGrid, window_length: int) -> np.ndarray:
    """Return a ``(grid.n_frames(len(wave)), window_length)`` array of windows (a copy)."""
    if wave.ndim != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {wave.shape}")
    if grid.causal:
        pad_left, pad_right = 0, window_length - 1
    else:
        pad_left = pad_right = window_length // 2
    padded = np.pad(wave, (pad_left, pad_right))
    n_frames = grid.n_frames(len(wave))
    if n_frames == 0:
        return np.empty((0, window_length), dtype=wave.dtype)
    windows = sliding_window_view(padded, window_length)[:: grid.hop_size][:n_frames]
    return np.ascontiguousarray(windows)
