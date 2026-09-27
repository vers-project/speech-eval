"""Fundamental frequency, measured three times by three different trackers.

Why three
---------
F0 is the most reliable single acoustic correlate of vocal effort: a converter
that raises level without raising pitch has learned gain, not effort.  That makes
the pitch track load-bearing evidence, and load-bearing evidence deserves more
than one instrument.

The three registered here are ``f0_praat`` (autocorrelation, via parselmouth),
``f0_opensmile`` (subharmonic summation, the eGeMAPS F0) and ``f0_swiftf0`` (a
small convolutional network on the STFT).  All three are emitted; **no consensus
rule is applied here**, because adjudication needs more than one utterance and
therefore belongs in :mod:`speech_eval.compare`.

The two classical trackers are not independent witnesses
--------------------------------------------------------
Praat's autocorrelation and openSMILE's subharmonic summation are both
harmonic/periodicity estimators and share the same failure modes — octave
halving or doubling when the fundamental is weak relative to the upper harmonics,
and unreliability on creak.  Two of them agreeing against the neural tracker is
therefore *not* a 2-of-3 majority among independent witnesses.

This matters here specifically.  High vocal effort weakens the fundamental
relative to the upper harmonics — the same spectral flattening the tilt measures
in :mod:`speech_eval.metrics.egemaps` are there to capture — so the loud
condition this project cares most about is exactly the regime where the
classical pair may *both* octave-halve and the neural tracker be the correct one.
The converse worry is equally live: a neural tracker can break on conversion
artifacts that are inaudible but out of its training distribution, where a
signal-processing method would not.

Neither fear should be settled by assertion, and both are testable with data this
suite already produces: compare inter-tracker disagreement rates on real against
converted audio from the same speakers.  Equal rates mean the out-of-distribution
worry is unfounded for this material; rates elevated only on converted audio
demonstrate it.  That computation is a ``compare.py`` job.

Analysis range: sex-conditional, widened, and constant per speaker
-----------------------------------------------------------------
Male 60-400 Hz, female 90-600 Hz, unknown 60-600 Hz.  The ceilings are above the
300/500 Hz Praat usually suggests because high vocal effort pushes F0 up, and a
ceiling below the true F0 is a direct cause of octave halving.

The range depends only on the ``sex`` column, never on the audio, which is what
guarantees the property that actually matters: **the same speaker is analysed
with the same range in every condition**.  A range that adapted to the signal
would let the instrument move with the thing it is measuring.  Because a
different range gives a different number, ``fmin_hz``, ``fmax_hz`` and
``range_source`` are emitted per row as provenance rather than left in a config.

openSMILE is the exception: its range is fixed at 55-1000 Hz inside the eGeMAPS
configuration and is not settable.  It reports that range with
``range_source = "backend_fixed"``.  Constant across conditions is the property
that matters, and it holds.

Frame grids differ, and are reported rather than resampled
----------------------------------------------------------
Praat and openSMILE run at a 10 ms hop; SwiftF0's is fixed by its ONNX graph at
256 samples of 16 kHz audio, i.e. 16 ms, and the first frame centres differ too
(25 ms, 10 ms, 7.97 ms respectively).  Rather than silently nearest-neighbour a
contour onto a shared grid, each tracker emits its own frames plus ``t0_s`` and
``time_step_s``, from which frame ``k`` sits at ``t0_s + k * time_step_s``.
Alignment is then one explicit step in ``compare.py``, where the choice of
interpolation is visible and can be made once for every pair of trackers.

Reported quantities
-------------------
``median_hz`` over voiced frames — a median, not a mean, because a single
octave error is a factor-of-two outlier — and ``mean_semitones`` /
``sd_semitones`` on the ``12*log2(f/27.5)`` scale, which is openSMILE's own
reference so all three trackers are directly comparable and speakers of
different sexes are on one scale.  Semitone SD is the standard measure of pitch
variability; a Hz SD would grow with a speaker's mean pitch on its own.

Contours go to the artifact archive by default: for F0 the time course *is* the
point, and it costs about 2 kB per utterance per tracker.  Unvoiced frames are
written as NaN in ``contour.f0_hz`` — the backends disagree on what to put there
(Praat writes 0.0, SwiftF0 leaves an unconstrained estimate), and a contour whose
unvoiced frames are usable by accident is a trap.  ``contour.voiced`` keeps the
voicing decision itself, which is what a voicing-agreement diagnostic needs.

References are collected in ``references.bib`` at the repository root.
"""
from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from speech_eval.core import (
    Features,
    Metric,
    Requirements,
    Utterance,
    UtteranceBatch,
    register_metric,
)

# openSMILE's semitone reference, adopted for every tracker so the semitone
# columns are comparable across all three.
SEMITONE_REFERENCE_HZ = 27.5

# Widened deliberately; see the module docstring.  Keyed by the first letter of
# the manifest's `sex` value, so "M", "male" and "Male" all resolve.
SEX_PITCH_RANGES: dict[str, tuple[float, float]] = {
    "m": (60.0, 400.0),
    "f": (90.0, 600.0),
}
DEFAULT_PITCH_RANGE = (60.0, 600.0)


class Track(NamedTuple):
    """One tracker's raw output on its own frame grid.

    ``f0_hz`` and ``voiced`` are per frame and the same length; ``t0_s`` is the
    centre of the first frame and ``time_step_s`` the hop between them, both in
    seconds.  Empty arrays mean the tracker could not analyse the signal at all.
    """

    f0_hz: np.ndarray
    voiced: np.ndarray
    t0_s: float
    time_step_s: float

    @classmethod
    def empty(cls, time_step_s: float = float("nan")) -> "Track":
        return cls(
            np.zeros(0, dtype=float), np.zeros(0, dtype=bool),
            float("nan"), time_step_s,
        )


def _is_missing(value: Any) -> bool:
    """True for None and for the float NaN pandas puts in an empty CSV cell."""
    return value is None or (isinstance(value, float) and np.isnan(value))


class F0Metric(Metric):
    """Shared behaviour of the pitch trackers: range, aggregation, contours.

    Subclasses implement :meth:`_track` and nothing else.  Everything a reader
    of the results table compares across trackers — how the analysis range was
    chosen, which frames counted as voiced, what a semitone is measured from —
    is decided once, here, so that a difference between two columns is a
    difference between two trackers rather than between two conventions.

    Parameters
    ----------
    sample_rate     : rate the audio is delivered at.  Fixed rather than native:
                      every tracker here has an internal frame size in samples,
                      so the analysis window is only constant across a corpus if
                      the rate is.
    level_norm_dbfs : RMS level to normalise to.  F0 itself is level-invariant,
                      but voicing decisions are not — Praat's silence threshold
                      and SwiftF0's confidence both move with level — and this
                      suite compares signals that differ in level by
                      construction.  It also keeps this family in one
                      preparation pass with ``egemaps``.
    pitch_ranges    : ``{sex key: (fmin, fmax)}``, matched on the lower-cased
                      first letter of the manifest's ``sex`` value.
    default_range   : used when ``sex`` is absent, empty or unrecognised.
    sex_column      : manifest column holding the speaker's sex.
    contours        : emit the per-frame track to the artifact archive.
    """

    #: Identifies the backend in the results table, independent of the instance
    #: name a config may have given it.
    tracker: str = ""

    #: Manifest columns this tracker needs to *compute*.  Cleared by the
    #: openSMILE subclass, whose range is not ours to choose.
    meta_columns: tuple[str, ...] = ("sex",)

    def __init__(
        self,
        name: str | None = None,
        sample_rate: int = 16_000,
        level_norm_dbfs: float | None = -27.0,
        pitch_ranges: Mapping[str, Sequence[float]] | None = None,
        default_range: Sequence[float] = DEFAULT_PITCH_RANGE,
        sex_column: str = "sex",
        contours: bool = True,
    ):
        super().__init__(name)
        self.sample_rate = sample_rate
        self.pitch_ranges = {
            key.strip().lower()[:1]: (float(low), float(high))
            for key, (low, high) in (pitch_ranges or SEX_PITCH_RANGES).items()
        }
        self.default_range = (float(default_range[0]), float(default_range[1]))
        for low, high in [*self.pitch_ranges.values(), self.default_range]:
            if not 0 < low < high:
                raise ValueError(
                    f"Pitch range ({low}, {high}) is not a valid interval above 0 Hz."
                )
        self.sex_column = sex_column
        self.contours = contours
        self.requirements = Requirements(
            sample_rate=sample_rate,
            mono=True,
            level_norm_dbfs=level_norm_dbfs,
            meta_columns=self.meta_columns,
        )

    # -- subclass hook --------------------------------------------------

    def _track(
        self, wav: np.ndarray, sample_rate: int, fmin: float, fmax: float
    ) -> Track:
        """Run the backend on one mono signal within ``[fmin, fmax]``.

        Return :meth:`Track.empty` — never raise — when the signal cannot be
        analysed: too short for the analysis window, or digitally silent.  An
        unmeasurable utterance is data, not an error.
        """
        raise NotImplementedError

    # -- range ----------------------------------------------------------

    def _resolve_range(self, utterance: Utterance) -> tuple[float, float, str]:
        """``(fmin, fmax, source)`` for one utterance, from its metadata alone."""
        sex = utterance.meta.get(self.sex_column)
        if not _is_missing(sex):
            key = str(sex).strip().lower()[:1]
            if key in self.pitch_ranges:
                return (*self.pitch_ranges[key], "sex")
        return (*self.default_range, "default")

    # -- aggregation ----------------------------------------------------

    def _summarise(self, track: Track) -> tuple[Features, np.ndarray]:
        """Per-utterance summary, and the voicing mask it was computed over.

        The mask comes back so the emitted contour and the emitted statistics
        cannot disagree about which frames were voiced.
        """
        # A frame counts as voiced only if the tracker says so *and* left a
        # usable frequency there: Praat writes 0.0 into unvoiced frames and
        # SwiftF0 leaves whatever its argmax picked.
        voiced = track.voiced & np.isfinite(track.f0_hz) & (track.f0_hz > 0)
        values = track.f0_hz[voiced]
        n_frames = int(track.f0_hz.size)
        semitones = (
            12.0 * np.log2(values / SEMITONE_REFERENCE_HZ) if values.size
            else np.zeros(0)
        )
        return {
            "median_hz": float(np.median(values)) if values.size else float("nan"),
            "mean_semitones": (
                float(semitones.mean()) if semitones.size else float("nan")
            ),
            "sd_semitones": (
                float(semitones.std(ddof=1)) if semitones.size > 1 else float("nan")
            ),
            "n_frames": n_frames,
            "n_voiced_frames": int(voiced.sum()),
            "voiced_fraction": (
                float(voiced.mean()) if n_frames else float("nan")
            ),
        }, voiced

    def compute(self, batch: UtteranceBatch) -> list[Features]:
        if not self._loaded:
            raise RuntimeError("load() must be called before compute().")

        results: list[Features] = []
        for utterance in batch.utterances:
            if utterance.sample_rate != self.sample_rate:
                raise ValueError(
                    f"{self.name!r} was given {utterance.sample_rate} Hz audio but is "
                    f"configured for {self.sample_rate} Hz. The runner delivers "
                    "Requirements.sample_rate; do not override one without the other."
                )
            wav = utterance.wav.detach().cpu().numpy().astype("float32")
            wav = wav.mean(axis=0) if wav.ndim > 1 else wav

            fmin, fmax, range_source = self._resolve_range(utterance)
            track = (
                self._track(wav, utterance.sample_rate, fmin, fmax) if wav.size
                else Track.empty()
            )
            features, voiced = self._summarise(track)
            features.update({
                "tracker": self.tracker,
                "fmin_hz": fmin,
                "fmax_hz": fmax,
                "range_source": range_source,
                "t0_s": track.t0_s,
                "time_step_s": track.time_step_s,
            })
            if self.contours:
                # Unvoiced frames are NaN rather than each backend's own filler,
                # so a contour cannot be read as pitch by accident.
                contour = np.where(voiced, track.f0_hz, np.nan)
                features["contour.f0_hz"] = contour.astype("float32")
                features["contour.voiced"] = voiced
            results.append(features)
        return results


# ---------------------------------------------------------------------------
# Praat — autocorrelation
# ---------------------------------------------------------------------------


@register_metric("f0_praat")
class PraatF0Metric(F0Metric):
    """Praat's ``To Pitch (ac)``, through parselmouth.

    The published autocorrelation method (Boersma 1993), and the tracker most
    phonetics results in the literature are expressed in.  Its analysis window is
    ``3 / pitch_floor``, so a 60 Hz floor implies a 50 ms window and any signal
    shorter than that cannot be analysed at all — parselmouth raises, and this
    metric turns that into NaN.

    The five path-search costs are Praat's own defaults, exposed because they are
    the knobs an octave-error investigation would reach for, and pinned in the
    results by nothing: change them and the column means something else, so
    change them deliberately.
    """

    tracker = "praat_ac"

    def __init__(
        self,
        name: str | None = None,
        time_step_s: float = 0.01,
        very_accurate: bool = False,
        silence_threshold: float = 0.03,
        voicing_threshold: float = 0.45,
        octave_cost: float = 0.01,
        octave_jump_cost: float = 0.35,
        voiced_unvoiced_cost: float = 0.14,
        **base_kwargs: Any,
    ):
        super().__init__(name, **base_kwargs)
        self.time_step_s = time_step_s
        self.very_accurate = very_accurate
        self.silence_threshold = silence_threshold
        self.voicing_threshold = voicing_threshold
        self.octave_cost = octave_cost
        self.octave_jump_cost = octave_jump_cost
        self.voiced_unvoiced_cost = voiced_unvoiced_cost
        self._parselmouth: Any = None

    def load(self, device) -> None:  # noqa: ANN001 — Praat is CPU-only
        """Import parselmouth.  ``device`` is ignored: Praat is C++ on CPU."""
        try:
            import parselmouth
        except ImportError as e:
            raise ImportError(
                "parselmouth is required for the Praat F0 metric. Install it with:  "
                "uv sync --extra phonetics"
            ) from e
        self._parselmouth = parselmouth
        self._loaded = True

    def _track(
        self, wav: np.ndarray, sample_rate: int, fmin: float, fmax: float
    ) -> Track:
        sound = self._parselmouth.Sound(
            wav.astype("float64"), sampling_frequency=sample_rate
        )
        try:
            pitch = sound.to_pitch_ac(
                time_step=self.time_step_s,
                pitch_floor=fmin,
                very_accurate=self.very_accurate,
                silence_threshold=self.silence_threshold,
                voicing_threshold=self.voicing_threshold,
                octave_cost=self.octave_cost,
                octave_jump_cost=self.octave_jump_cost,
                voiced_unvoiced_cost=self.voiced_unvoiced_cost,
                pitch_ceiling=fmax,
            )
        except self._parselmouth.PraatError:
            # Raised when the signal is shorter than the 3/pitch_floor window.
            return Track.empty(self.time_step_s)
        f0 = np.asarray(pitch.selected_array["frequency"], dtype=float)
        return Track(f0, f0 > 0, float(pitch.t1), float(pitch.dt))


# ---------------------------------------------------------------------------
# openSMILE — subharmonic summation
# ---------------------------------------------------------------------------


@register_metric("f0_opensmile")
class OpenSMILEF0Metric(F0Metric):
    """The eGeMAPS F0: subharmonic summation (Hermes 1988), read from the LLDs.

    Per GeMAPS §6.1: 60 ms windows at a 10 ms hop, the spectrum octave
    shift-added 15 times with compression 0.85, up to six candidates in
    **55-1000 Hz**, and an on-line Viterbi path search with voicing threshold
    0.7.  None of that is configurable, which is the point — it is the F0 the
    published parameter set is defined on, and it is what
    ``F0semitoneFrom27.5Hz_sma3nz`` in the ``egemaps`` columns already contains.

    Two consequences worth knowing before reading its column.  The 60 ms window
    smooths real pitch movement, so this tracker is adequate for a per-utterance
    median and poor for contour work.  And its frame times come from openSMILE's
    20 ms spectral frame index, not from the 60 ms pitch window, so ``t0_s`` is
    that frame's midpoint — accurate to the grid, approximate as the centre of
    what was actually analysed.

    Running this alongside ``egemaps`` calls openSMILE twice on the same audio.
    That is accepted: it is fast, and sharing the extractor would couple two
    metrics that should be independently configurable.
    """

    tracker = "opensmile_shs"
    # The range is the backend's, so there is nothing for `sex` to decide.
    meta_columns: tuple[str, ...] = ()
    #: Fixed inside the eGeMAPS configuration; reported, not chosen.
    BACKEND_RANGE = (55.0, 1000.0)

    def __init__(
        self, name: str | None = None, feature_set: str = "eGeMAPSv02", **base_kwargs: Any
    ):
        super().__init__(name, **base_kwargs)
        self.feature_set = feature_set
        self._smile: Any = None
        self._f0_column: str | None = None

    def load(self, device) -> None:  # noqa: ANN001 — openSMILE is CPU-only
        """Build the openSMILE extractor.  ``device`` is ignored: C++ on CPU."""
        try:
            import opensmile
        except ImportError as e:
            raise ImportError(
                "openSMILE is required for the openSMILE F0 metric. Install it with:  "
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
            sampling_rate=self.sample_rate,
            resample=True,
        )
        # Located by prefix so a rename between feature-set versions surfaces
        # here rather than as a column of NaN.
        self._f0_column = next(
            (n for n in self._smile.feature_names if n.startswith("F0semitone")), None
        )
        if self._f0_column is None:
            raise ValueError(
                f"{self.feature_set} exposes no F0 descriptor, so it cannot be used "
                "as a pitch tracker."
            )
        self._loaded = True

    def _resolve_range(self, utterance: Utterance) -> tuple[float, float, str]:
        """Always the backend's own range — see the class docstring."""
        return (*self.BACKEND_RANGE, "backend_fixed")

    def _track(
        self, wav: np.ndarray, sample_rate: int, fmin: float, fmax: float
    ) -> Track:
        frame = self._smile.process_signal(wav, sample_rate)
        if not len(frame):
            return Track.empty(0.01)
        starts = frame.index.get_level_values("start").total_seconds().to_numpy()
        ends = frame.index.get_level_values("end").total_seconds().to_numpy()
        semitones = frame[self._f0_column].to_numpy(dtype=float)
        # openSMILE writes a literal 0.0 into unvoiced frames of every _sma3nz
        # descriptor, F0 included.
        voiced = semitones > 0
        f0 = np.where(voiced, SEMITONE_REFERENCE_HZ * 2 ** (semitones / 12.0), 0.0)
        time_step = float(starts[1] - starts[0]) if len(starts) > 1 else float("nan")
        return Track(f0, voiced, float((starts[0] + ends[0]) / 2), time_step)


# ---------------------------------------------------------------------------
# SwiftF0 — convolutional network on the STFT
# ---------------------------------------------------------------------------


@register_metric("f0_swiftf0")
class SwiftF0Metric(F0Metric):
    """SwiftF0, the neural cross-check.

    Chosen over PENN, which scores 0.6 points higher on PTDB-TUG's laryngograph
    ground truth but imports ``torbi``, whose prebuilt Viterbi binaries lag the
    current torch by two minor versions and fail at import.  SwiftF0 needs only
    onnxruntime, ships its model inside the wheel — so it works on an offline
    compute node and can actually be tested here — and still ranks above Praat
    and well above CREPE, which is music-trained and underperforms Praat on
    speech.

    Note what the published rankings do and do not cover: they are all clean read
    speech.  Nobody has benchmarked a pitch tracker on high vocal effort or on
    codec-decoded audio, which is this project's regime.  That is the reason all
    three trackers are emitted rather than one being trusted.

    Its frame grid is fixed by the ONNX graph at a 256-sample hop, so 16 ms at
    16 kHz, and the model resolves only 46.875-2093.75 Hz.  The configured range
    is applied as SwiftF0's own voicing constraint, alongside
    ``confidence_threshold``; ranges outside what the model can resolve are
    rejected at load rather than mid-run.
    """

    tracker = "swiftf0"

    def __init__(
        self, name: str | None = None, confidence_threshold: float = 0.9,
        **base_kwargs: Any,
    ):
        super().__init__(name, **base_kwargs)
        self.confidence_threshold = confidence_threshold
        self._swift_f0: Any = None
        self._detectors: dict[tuple[float, float], Any] = {}

    def load(self, device) -> None:  # noqa: ANN001 — the ONNX session is CPU-only
        """Build one detector per configured range.  ``device`` is ignored.

        Constructing them all here rather than lazily is what makes an
        unresolvable pitch range fail at startup instead of a thousand
        utterances into a run.
        """
        try:
            import swift_f0
        except ImportError as e:
            raise ImportError(
                "swift-f0 is required for the SwiftF0 metric. Install it with:  "
                "uv sync --extra phonetics"
            ) from e
        self._swift_f0 = swift_f0
        for pitch_range in [*self.pitch_ranges.values(), self.default_range]:
            self._detector(pitch_range)
        self._loaded = True

    def _detector(self, pitch_range: tuple[float, float]) -> Any:
        if pitch_range not in self._detectors:
            fmin, fmax = pitch_range
            self._detectors[pitch_range] = self._swift_f0.SwiftF0(
                confidence_threshold=self.confidence_threshold, fmin=fmin, fmax=fmax
            )
        return self._detectors[pitch_range]

    def _track(
        self, wav: np.ndarray, sample_rate: int, fmin: float, fmax: float
    ) -> Track:
        if sample_rate != self._swift_f0.SwiftF0.TARGET_SAMPLE_RATE:
            # Its own resampling path needs librosa, which this suite does not
            # install; the runner is already delivering a fixed rate.
            raise ValueError(
                f"SwiftF0 analyses {self._swift_f0.SwiftF0.TARGET_SAMPLE_RATE} Hz "
                f"audio; got {sample_rate} Hz."
            )
        result = self._detector((fmin, fmax)).detect_from_array(wav, sample_rate)
        timestamps = np.asarray(result.timestamps, dtype=float)
        time_step = (
            float(timestamps[1] - timestamps[0]) if timestamps.size > 1
            else self._swift_f0.SwiftF0.HOP_LENGTH / sample_rate
        )
        return Track(
            np.asarray(result.pitch_hz, dtype=float),
            np.asarray(result.voicing, dtype=bool),
            float(timestamps[0]) if timestamps.size else float("nan"),
            time_step,
        )
