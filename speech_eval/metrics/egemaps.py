"""eGeMAPS low-level descriptors, via openSMILE, aggregated here.

Reference
---------
F. Eyben, K. R. Scherer, B. W. Schuller, J. Sundberg, E. André, C. Busso,
L. Y. Devillers, J. Epps, P. Laukka, S. S. Narayanan, K. P. Truong,
"The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and
Affective Computing", IEEE Transactions on Affective Computing 7(2):190-202, 2016.

openSMILE is the authors' own implementation, so it *is* the reference for these
definitions — hence delegating to it rather than reimplementing.  What is not
delegated is the aggregation: openSMILE's Functionals level would hand back 88
pre-summarised numbers, whereas the LowLevelDescriptors level returns one row per
20 ms frame with timestamps.  Taking the per-frame form means this suite decides
which frames count, and leaves the door open to restricting a measurement to
forced-aligned vowel intervals later without recomputing anything.

Frame settings are openSMILE's, per the paper's §6.1: 20 ms windows 10 ms apart,
Hamming, zero-padded to the next power of two, for the spectral descriptors; 60 ms
windows for F0, harmonic differences, HNR, jitter and shimmer.  Every LLD is
smoothed by a symmetric 3-frame moving average, which is what the ``_sma3`` suffix
records.  These are not configurable and should be reported as they are.

The formant descriptors here should not be read
------------------------------------------------
``F1frequency_sma3nz`` and its eight siblings are emitted like any other
descriptor, and they are the wrong instrument for this suite's question: their
error is coupled to the *source spectral tilt*, so they move when vocal effort
flattens the spectrum even though the vocal tract has not.  Use
:mod:`speech_eval.metrics.formants`, which measures the same quantity with Praat
and carries the evidence.  They stay emitted rather than being filtered out
because the ``egemaps.`` column prefix already tells the two families apart, and
because dropping descriptors from a published parameter set would make this
metric something other than eGeMAPS.

Voiced and unvoiced frames
--------------------------
Descriptors ending ``_sma3nz`` (F0, jitter, shimmer, HNR, the harmonic
differences, the formant tracks) are defined only where the voice is periodic,
and openSMILE writes a literal **0.0** into them on unvoiced frames.  Averaging
those over every frame does not average a smaller sample — it averages in
structural zeros and silently pulls the mean toward zero in proportion to how much
silence the segment contains.  They are therefore aggregated over voiced frames
only.  The plain ``_sma3`` descriptors are defined everywhere and get all three
frame sets, which also mirrors the paper: functionals over voiced regions, plus
the spectral-balance means over unvoiced ones.

Level normalisation is on by default, and that is not a stylistic choice
-----------------------------------------------------------------------
Measured on this openSMILE build (2.6.0, eGeMAPSv02), with one waveform scaled
across its level range:

* ``alphaRatio`` and ``hammarbergIndex`` are level-invariant to within 0.15 dB
  while the signal stays inside [-1, +1] — the property that makes spectral tilt
  the measure a gain-only baseline cannot fake.
* Above full scale they collapse: the same signal reads -12.3 dB at peak 1.0,
  -5.4 at peak 2.0 and +4.5 at peak 5.0.  No warning is raised.  Converted audio
  at a high target intensity is precisely the case that can exceed full scale.
* ``slope0-500`` and ``slope500-1500`` drift monotonically with level even inside
  the valid range (+0.040 to -0.009 dB/Hz over 60 dB), because bins fall below the
  spectral floor as the signal quietens.  They are the level-sensitive members of
  the family.

Normalising to a fixed RMS fixes all three at once, and costs nothing: absolute
level is measured exactly by the ``level`` metric, so it is not this metric's job.
Set ``level_norm_dbfs: null`` to defeat it, and read ``Loudness_sma3`` as an
absolute quantity — at the price of the three effects above.

alphaRatio is high-over-low, and the GeMAPS paper's prose is what is wrong
-------------------------------------------------------------------------
The GeMAPS paper describes the alpha ratio as the "ratio of the summed energy
from 50-1000 Hz and 1-5 kHz", i.e. low over high.  openSMILE computes the
reciprocal, and **openSMILE is the one that matches the original definition**:
Frokjaer-Jensen & Prytz (1975), where the measure is introduced as the ratio of
energy above to below 1 kHz.  Sundberg & Nordenberg (2006) use that same
orientation.  So this is a slip in the GeMAPS prose, not a bug in its toolkit.

Confirmed by measurement here: flat-spectrum noise gives +6.14 dB against the
analytic |10*log10(950/4000)| = 6.24 dB, and boosting everything below 1 kHz by
20 dB drives it to -12.3 dB.  The emitted quantity is therefore

    alphaRatio = 10*log10( E(1-5 kHz) / E(50-1000 Hz) )

so it **increases** with vocal effort, while ``hammarbergIndex`` — strongest peak
in 0-2 kHz minus strongest in 2-5 kHz, an orientation openSMILE and its source
agree on — **decreases** with it.  Report the direction, not just the number, and
cite the 1975 original for the definition rather than the 2016 restatement of it.

References are collected in ``references.bib`` at the repository root.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from audio_utils.data.transforms import peak_dbfs

from speech_eval.core import (
    Features,
    Metric,
    Requirements,
    UtteranceBatch,
    register_metric,
)

# Descriptors defined only on voiced frames carry this suffix in every
# GeMAPS-family set.
VOICED_ONLY_SUFFIX = "_sma3nz"

# Frame sets a plain _sma3 descriptor is summarised over.
ALL_FRAME_SETS = ("voiced", "unvoiced", "all")

STATISTICS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.nanmean,
    "std": lambda v: float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan,
    "median": np.nanmedian,
    "min": np.nanmin,
    "max": np.nanmax,
}


@register_metric("egemaps")
class EGeMAPSMetric(Metric):
    """Per-utterance summaries of the eGeMAPS low-level descriptors.

    Emits ``<descriptor>_<frame set>_<statistic>`` for every descriptor, keeping
    openSMILE's own names (``alphaRatio_sma3``, ``logRelF0-H1-H2_sma3nz``) so a
    column can always be traced back to the toolkit that produced it.

    Parameters
    ----------
    feature_set     : any GeMAPS-family set name on ``opensmile.FeatureSet``.
                      Pinned and emitted as a column: v01a, v01b and v02 differ,
                      so the version is part of the measurement, not packaging.
    sample_rate     : rate the audio is delivered at.  Fixed rather than native
                      because the band edges (1 kHz, 2 kHz, 5 kHz) must fall on
                      the same bins for every corpus being compared.
    level_norm_dbfs : RMS level to normalise to before measuring; ``None``
                      disables it.  See the module docstring — the default is
                      load-bearing.
    statistics      : which of mean/std/median/min/max to emit per frame set.
    descriptors     : restrict to these openSMILE descriptor names.  ``None``
                      emits all of them.
    contours        : descriptors whose full per-frame track is saved to the
                      artifact archive, for anything that needs the time course
                      rather than a summary.
    """

    def __init__(
        self,
        name: str | None = None,
        feature_set: str = "eGeMAPSv02",
        sample_rate: int = 16_000,
        level_norm_dbfs: float | None = -27.0,
        statistics: Sequence[str] = ("mean", "std"),
        descriptors: Sequence[str] | None = None,
        contours: Sequence[str] = (),
    ):
        super().__init__(name)
        unknown = set(statistics) - set(STATISTICS)
        if unknown:
            raise ValueError(
                f"Unknown statistic(s) {sorted(unknown)}. "
                f"Available: {sorted(STATISTICS)}."
            )
        self.feature_set = feature_set
        self.sample_rate = sample_rate
        self.statistics = tuple(statistics)
        self.descriptors = tuple(descriptors) if descriptors else None
        self.contours = tuple(contours)
        # Per-instance, not class-level: the audio this metric needs depends on
        # how it was configured.
        self.requirements = Requirements(
            sample_rate=sample_rate, mono=True, level_norm_dbfs=level_norm_dbfs
        )
        self._smile: Any = None
        self._f0_column: str | None = None

    # ------------------------------------------------------------------

    def load(self, device) -> None:  # noqa: ANN001 — openSMILE is CPU-only
        """Build the openSMILE extractor.  ``device`` is ignored: it is C++ on CPU."""
        if self._smile is not None:
            return
        try:
            import opensmile
        except ImportError as e:
            raise ImportError(
                "openSMILE is required for the eGeMAPS metric. Install it with:  "
                "uv sync --extra phonetics"
            ) from e

        try:
            feature_set = opensmile.FeatureSet[self.feature_set]
        except KeyError:
            raise ValueError(
                f"Unknown feature set {self.feature_set!r}. Available: "
                f"{[f.name for f in opensmile.FeatureSet]}"
            ) from None

        self._smile = opensmile.Smile(
            feature_set=feature_set,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            # A no-op given the Requirements above, kept as a guard so the metric
            # stays correct if someone changes the requested rate.
            sampling_rate=self.sample_rate,
            resample=True,
        )
        names = list(self._smile.feature_names)
        if self.descriptors:
            missing = set(self.descriptors) - set(names)
            if missing:
                raise ValueError(
                    f"Descriptor(s) {sorted(missing)} are not in {self.feature_set}. "
                    f"Available: {names}"
                )
            names = [n for n in names if n in self.descriptors]
        self._names = names

        missing_contours = set(self.contours) - set(self._smile.feature_names)
        if missing_contours:
            raise ValueError(
                f"Contour descriptor(s) {sorted(missing_contours)} are not in "
                f"{self.feature_set}."
            )

        # The voicing mask comes from the F0 track, which openSMILE zeroes on
        # unvoiced frames.  Located by prefix so the metric survives a rename
        # between feature-set versions; absent means no mask is available.
        self._f0_column = next(
            (n for n in self._smile.feature_names if n.startswith("F0semitone")), None
        )
        self._loaded = True

    # ------------------------------------------------------------------

    def _frame_masks(self, frame: Any) -> dict[str, np.ndarray]:
        """Boolean masks selecting the frames each frame set summarises."""
        n = len(frame)
        everything = np.ones(n, dtype=bool)
        if self._f0_column is None or self._f0_column not in frame.columns:
            return {"all": everything}
        voiced = frame[self._f0_column].to_numpy() > 0
        return {"voiced": voiced, "unvoiced": ~voiced, "all": everything}

    def _summarise(self, frame: Any) -> Features:
        masks = self._frame_masks(frame)
        features: Features = {}
        for descriptor in self._names:
            values = frame[descriptor].to_numpy(dtype=float)
            # Voiced-only descriptors are zero, not absent, on unvoiced frames:
            # summarising them anywhere but the voiced mask averages in zeros.
            frame_sets = (
                ("voiced",) if descriptor.endswith(VOICED_ONLY_SUFFIX)
                else ALL_FRAME_SETS
            )
            for frame_set in frame_sets:
                mask = masks.get(frame_set)
                selected = values[mask] if mask is not None else np.empty(0)
                selected = selected[np.isfinite(selected)]
                for statistic in self.statistics:
                    key = f"{descriptor}_{frame_set}_{statistic}"
                    features[key] = (
                        float(STATISTICS[statistic](selected))
                        if selected.size else float("nan")
                    )
        return features

    def _empty_features(self) -> Features:
        """All-NaN result, for input openSMILE cannot measure."""
        features: Features = {}
        for descriptor in self._names:
            frame_sets = (
                ("voiced",) if descriptor.endswith(VOICED_ONLY_SUFFIX)
                else ALL_FRAME_SETS
            )
            for frame_set in frame_sets:
                for statistic in self.statistics:
                    features[f"{descriptor}_{frame_set}_{statistic}"] = float("nan")
        return features

    def compute(self, batch: UtteranceBatch) -> list[Features]:
        if self._smile is None:
            raise RuntimeError("load() must be called before compute().")

        results: list[Features] = []
        for utterance in batch.utterances:
            # Unpadded, one at a time: openSMILE frames whatever it is given, so a
            # padded batch would hand it fabricated silence to measure.
            wav = utterance.wav.detach().cpu().numpy().astype("float32")
            peak = peak_dbfs(utterance.wav)

            frame = self._smile.process_signal(wav, utterance.sample_rate)
            n_frames = len(frame)
            # Digital silence yields frames of literal 0.0 rather than NaN, which
            # would otherwise enter the means as a real measurement.
            silent = not np.any(wav)

            if n_frames == 0 or silent:
                features = self._empty_features()
                masks: dict[str, np.ndarray] = {}
            else:
                features = self._summarise(frame)
                masks = self._frame_masks(frame)

            voiced = masks.get("voiced")
            features.update({
                "feature_set": self.feature_set,
                "n_frames": n_frames,
                "n_voiced_frames": int(voiced.sum()) if voiced is not None else -1,
                "voiced_fraction": (
                    float(voiced.mean()) if voiced is not None and voiced.size
                    else float("nan")
                ),
                # Audit trail for the full-scale failure described in the module
                # docstring: above 0 dBFS the spectral measures are corrupted, so
                # this column says whether any result can be trusted.
                "peak_dbfs": peak,
                "above_full_scale": bool(peak > 0.0),
            })
            for descriptor in self.contours:
                features[f"contour.{descriptor}"] = (
                    frame[descriptor].to_numpy(dtype="float32") if n_frames
                    else np.zeros(0, dtype="float32")
                )
            results.append(features)
        return results
