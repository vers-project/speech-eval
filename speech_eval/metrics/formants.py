"""Formant frequencies and bandwidths, by Praat's Burg LPC analysis.

Why a separate metric when ``egemaps`` already emits formants
--------------------------------------------------------------
eGeMAPSv02 carries nine formant descriptors (``F1frequency_sma3nz`` and friends)
and :mod:`speech_eval.metrics.egemaps` aggregates them like any other, so
``egemaps.F1frequency_sma3nz_voiced_mean`` already exists in the results table.
**Those columns should not be used for this project**, and the reason is
specific rather than general.

Measured here against synthesised vowels with exactly known formants — a pulse
train, a source tilt, a five-resonator cascade and lip radiation, so the truth is
known by construction — mean absolute error on F1/F2/F3 over three vowels, three
F0s and two source tilts:

=================  ==================
Praat (this)       3.6 %
openSMILE          48.2 %
=================  ==================

The mean is not the reason to prefer Praat.  This is, sweeping the source tilt on
one vowel whose vocal tract never moves::

    source tilt |   praat F1/F2/F3    |  openSMILE F1/F2/F3
        16      |  488   1469   2444  |   418   1418   2585
        12      |  488   1485   2488  |   379   1339   2423
        10      |  496   1486   2487  |   336   1312   2384
         8      |  508   1494   2484  |   544   1631   2734
         6      |  527   1505   2487  |  1197   2017   2984
         4      |  556   1517   2493  |  1115   2118   3005
      TRUTH     |  500   1500   2500  |

Flattening the source spectrum is *what vocal effort does*.  Somewhere below
-8 dB/octave openSMILE loses F1 and promotes F2 into its place, so its F1 reads
336 Hz at a modal tilt and 1197 Hz at a flat one on a signal with one fixed
vocal tract.  Averaged over the sweep its error grows by 59 percentage points
from modal to flat; Praat's grows by 3.3.

For a suite whose whole purpose is measuring displacement between a quiet and a
loud condition, that is the worst available failure: the instrument moves with
the variable under test and would manufacture a large, clean, entirely spurious
formant displacement out of a spectral-tilt change.  It is the same class of
problem as the octave-halving argument in :mod:`speech_eval.metrics.f0`.

Praat is additionally exactly level-invariant here — bit-identical output from
+6 dB down to -40 dB of applied gain — where openSMILE returns NaN at -40 dB.

The probe is ideal synthetic source-filter audio, so what it establishes is the
*direction* of the argument (openSMILE's error is coupled to tilt, Praat's is
small and roughly not), not the exact breakdown point on real loud speech.

Praat's residual tilt bias lands on F1, and is not negligible against the effect
---------------------------------------------------------------------------------
"Roughly not" is not "not".  Read the F1 row of the sweep above again: 488 Hz at
a 16 dB/octave source and 556 Hz at 4 dB/octave, on a vocal tract that never
moved.  End to end through the runner, a 12 -> 6 dB/octave flattening moves the
measured F1 by +28 to +43 Hz across three vowels.  Call it roughly 5-10 Hz per
dB/octave of source flattening.

Two things follow, and both matter when reading a displacement:

* **The bias points the same way as the real effect.**  Vocal effort raises F1
  *and* flattens the source, and the artifact adds to the true rise rather than
  opposing it.  A measured F1 displacement is therefore an upper bound on the
  articulatory one.  Against a true effort effect in the tens to low hundreds of
  Hz, a 30-40 Hz artifact is a minority of the signal but not a rounding error,
  so an F1 displacement close to that size is not evidence of anything on its own.
* **F2 and F3 are clean.**  Over the same sweep F2 moved 1469 -> 1517 Hz and F3
  2444 -> 2493 Hz — within the analysis's own resolution, and without the
  monotone trend F1 shows.  Where a result has to rest on one formant, rest it on
  F2 or F3 and treat F1 as corroboration.

The honest way to settle the residual on real audio is the route ``f0.py``
prescribes for its trackers: measure the disagreement on material where the
answer is known independently.  Comparing F1 displacement against
``egemaps.alphaRatio_sma3_voiced_mean`` displacement across the corpus would show
whether the two co-vary more than articulation alone explains.  That is a
``compare.py`` job and is not done here.

A formant mean is mostly a statement about phonetic content
------------------------------------------------------------
This is the trap that matters more than the choice of backend.  Averaged over an
utterance, F1 is dominated by *which vowels were spoken*, not by voice quality:
an utterance rich in /a/ has a higher mean F1 than one rich in /i/, in any voice,
at any effort.

These columns are therefore interpretable **only in a content-matched paired
comparison** — the same speaker reading the same words, which is what
``group_id`` identifies.  Aggregated across conditions without that pairing they
measure the text.  The eventual fix is to restrict the measurement to
forced-aligned vowel intervals, which is what ``Requirements.needs_intervals``
is reserved for and which the runner does not yet support.

What the literature expects, for orientation and not as a claim checked here:
vocal effort raises F1 substantially (a jaw- and mouth-opening effect), F2
modestly and F3 little.  Traunmueller & Eriksson (2000) and Huber et al. (1999)
are the canonical measurements; verify either against the primary source before
citing it.

The analysis ceiling is sex-conditional, and is provenance
-----------------------------------------------------------
Male 5000 Hz, female 5500 Hz, unknown 5500 Hz, decided from the ``sex`` column
alone and never from the audio — the same guarantee :mod:`speech_eval.metrics.f0`
makes about its pitch range, and for the same reason: **one speaker is analysed
identically in every condition**, so a difference between conditions cannot be a
difference between measurement settings.

The ceiling is load-bearing.  Measured on the synthetic vowels, a 4500 Hz ceiling
loses F3 entirely (1979 Hz read for a true 2500); 5000, 5500 and 6000 all land
within about 1 %.  Because a different ceiling gives a different number,
``ceiling_hz`` and ``ceiling_source`` are emitted per row rather than left in a
config.

Praat resamples internally to twice the ceiling, so 16 kHz input is downsampled
to 10-11 kHz for the analysis.  That is a downsample at every supported ceiling,
which is why :attr:`Requirements.sample_rate` is fixed rather than native.

Viterbi tracking, and an honest note about it
-----------------------------------------------
``track=True`` follows the raw Burg analysis with Praat's ``Track``, which picks
one path through the per-frame formant candidates by minimising

    frequency_cost * |F - reference| + bandwidth_cost * B
        + transition_cost * |dF/dt|

The reference frequencies default to ``(2k+1) * ceiling / 10``, the resonances of
a uniform tube whose formants fill the analysis band.  That rule is not invented
here: at Praat's own default 5500 Hz ceiling it reproduces Praat's own published
defaults (550, 1650, 2750, 3850, 4950 Hz) exactly, and it scales them coherently
when the ceiling changes with the speaker's sex — which a fixed reference list
would not.

**The known-answer probe could not show tracking to help, and showed it mildly
hurting.** On synthesised diphthongs with moving formants and additive noise,
mean RMSE over F1/F2/F3 was 68 Hz raw and 81 Hz tracked; on the hardest case — a
/w/-like onset where F1 and F2 start 350 Hz apart, at 20 dB SNR — 288 Hz raw
against 364 Hz tracked.  The frequency term is what makes ``Track`` work at all
rather than what breaks it: setting ``frequency_cost=0`` collapses it to 866 Hz
RMSE, because the reference is the only thing anchoring a pole to a track slot.
Praat's own 1/1/1 is the best tracking setting tested, and still lost to not
tracking.

That result should be read for what it covers.  The probe synthesises exactly
five well-separated poles and hands the analyser exactly five formants to find,
which is the regime in which raw Burg makes almost no labelling errors (17 Hz
RMSE at 40 dB SNR) and tracking therefore has nothing to repair.  Real speech —
nasals, formant-zero pairs, creak, genuine mergers — has many more ways to
mislabel a frame, and that is the case tracking exists for and that this probe
cannot reach.  Tracking is the configured default here as a deliberate choice;
``track: false`` is a one-line config change, and ``tracked`` is emitted per row
so the two are never silently mixed in one table.

References are collected in ``references.bib`` at the repository root.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import numpy as np

from speech_eval.core import (
    Features,
    Metric,
    Requirements,
    Utterance,
    UtteranceBatch,
    register_metric,
)
from speech_eval.metrics.f0 import (
    DEFAULT_PITCH_RANGE,
    SEX_PITCH_RANGES,
    _is_missing,
)

#: Analysis ceiling by the lower-cased first letter of the manifest's `sex`.
#: Praat's long-standing convention, and the shorter female vocal tract is the
#: reason the two differ.
SEX_FORMANT_CEILINGS: dict[str, float] = {
    "m": 5000.0,
    "f": 5500.0,
}
DEFAULT_FORMANT_CEILING = 5500.0

#: Formants summarised and emitted.  F4 and up are not reported: their estimates
#: depend far more on the ceiling than F1-F3 do, and nothing downstream reads them.
FORMANTS = (1, 2, 3)


class FormantTrack(NamedTuple):
    """One analysis on its own frame grid.

    ``frequencies`` and ``bandwidths`` are ``(n_formants, n_frames)`` in Hz with
    NaN where Praat left a formant undefined; ``voiced`` is per frame.  ``t0_s``
    is the centre of the first frame and ``time_step_s`` the hop.  Empty arrays
    mean the signal could not be analysed at all.
    """

    frequencies: np.ndarray
    bandwidths: np.ndarray
    voiced: np.ndarray
    t0_s: float
    time_step_s: float
    tracked: bool

    @classmethod
    def empty(cls, time_step_s: float = float("nan"), tracked: bool = False):
        return cls(
            np.zeros((len(FORMANTS), 0)), np.zeros((len(FORMANTS), 0)),
            np.zeros(0, dtype=bool), float("nan"), time_step_s, tracked,
        )


@register_metric("formants_praat")
class PraatFormantMetric(Metric):
    """F1-F3 frequency and bandwidth from Praat's ``To Formant (burg)``.

    Emits ``f{n}_median_hz``, ``f{n}_sd_hz`` and ``f{n}_bandwidth_hz`` over voiced
    frames, plus the frame grid and the analysis settings as provenance.  Medians
    rather than means, for the reason F0 uses one: a single mislabelled frame is
    an outlier of formant-spacing size, not of measurement-noise size.

    Parameters
    ----------
    sample_rate       : rate the audio is delivered at.  Fixed, not native: Praat
                        resamples to twice the ceiling internally, and a corpus
                        analysed at mixed rates would mix resampling filters.
    level_norm_dbfs   : RMS level to normalise to.  Praat's formant analysis is
                        exactly level-invariant, so this changes no number; it is
                        kept at the suite's -27 dBFS so this metric shares one
                        preparation pass with ``egemaps`` and the F0 family
                        instead of forcing a second.
    ceilings          : ``{sex key: ceiling_hz}``, matched on the lower-cased
                        first letter of the manifest's ``sex`` value.
    default_ceiling   : used when ``sex`` is absent, empty or unrecognised.
    n_formants        : formants Praat fits below the ceiling.  Praat takes a
                        real number here; 5.0 is its default and 5.5 is the
                        usual remedy when five poles do not fit a speaker.
    track             : run Praat's Viterbi ``Track`` over the Burg analysis.
                        See the module docstring before changing it.
    track_references  : reference frequencies for the tracker, or ``None`` for
                        ``(2k+1) * ceiling / 10`` — Praat's own defaults, scaled
                        with the ceiling.
    frequency_cost, bandwidth_cost, transition_cost :
                        ``Track``'s three cost weights, at Praat's defaults.
                        ``frequency_cost=0`` does not mean "no reference bias";
                        it means the tracker has no anchor and fails.
    window_length_s, time_step_s, pre_emphasis_from_hz :
                        Burg analysis settings, at Praat's defaults.
    contours          : emit per-frame frequency tracks to the artifact archive.
    """

    def __init__(
        self,
        name: str | None = None,
        sample_rate: int = 16_000,
        level_norm_dbfs: float | None = -27.0,
        ceilings: dict[str, float] | None = None,
        default_ceiling: float = DEFAULT_FORMANT_CEILING,
        n_formants: float = 5.0,
        track: bool = True,
        track_references: Sequence[float] | None = None,
        frequency_cost: float = 1.0,
        bandwidth_cost: float = 1.0,
        transition_cost: float = 1.0,
        window_length_s: float = 0.025,
        time_step_s: float = 0.01,
        pre_emphasis_from_hz: float = 50.0,
        sex_column: str = "sex",
        pitch_ranges: dict[str, Sequence[float]] | None = None,
        contours: bool = True,
    ):
        super().__init__(name)
        self.sample_rate = sample_rate
        self.ceilings = {
            key.strip().lower()[:1]: float(value)
            for key, value in (ceilings or SEX_FORMANT_CEILINGS).items()
        }
        self.default_ceiling = float(default_ceiling)
        for ceiling in [*self.ceilings.values(), self.default_ceiling]:
            if not 0 < ceiling <= sample_rate / 2:
                raise ValueError(
                    f"Formant ceiling {ceiling} Hz is not inside (0, Nyquist] for "
                    f"{sample_rate} Hz audio."
                )
        if n_formants < len(FORMANTS):
            raise ValueError(
                f"n_formants={n_formants} cannot yield the {len(FORMANTS)} formants "
                "this metric reports."
            )
        self.n_formants = float(n_formants)
        self.track = bool(track)
        self.track_references = (
            tuple(float(f) for f in track_references) if track_references else None
        )
        self.frequency_cost = float(frequency_cost)
        self.bandwidth_cost = float(bandwidth_cost)
        self.transition_cost = float(transition_cost)
        self.window_length_s = float(window_length_s)
        self.time_step_s = float(time_step_s)
        self.pre_emphasis_from_hz = float(pre_emphasis_from_hz)
        self.sex_column = sex_column
        # The voicing mask reuses the pitch ranges the F0 metrics analyse with,
        # so "voiced" means the same thing in both families' columns.
        self.pitch_ranges = {
            key.strip().lower()[:1]: (float(low), float(high))
            for key, (low, high) in (pitch_ranges or SEX_PITCH_RANGES).items()
        }
        self.contours = bool(contours)
        self.requirements = Requirements(
            sample_rate=sample_rate,
            mono=True,
            level_norm_dbfs=level_norm_dbfs,
            meta_columns=(sex_column,),
        )
        self._parselmouth: Any = None
        self._call: Any = None

    # -- lifecycle ------------------------------------------------------

    def load(self, device) -> None:  # noqa: ANN001 — Praat is C++ on CPU
        """Import parselmouth.  ``device`` is ignored."""
        try:
            import parselmouth
            from parselmouth.praat import call
        except ImportError as e:
            raise ImportError(
                "parselmouth is required for the Praat formant metric. Install it "
                "with:  uv sync --extra phonetics"
            ) from e
        self._parselmouth = parselmouth
        self._call = call
        self._loaded = True

    # -- settings -------------------------------------------------------

    def _resolve_ceiling(self, utterance: Utterance) -> tuple[float, str]:
        """``(ceiling_hz, source)`` for one utterance, from its metadata alone."""
        sex = utterance.meta.get(self.sex_column)
        if not _is_missing(sex):
            key = str(sex).strip().lower()[:1]
            if key in self.ceilings:
                return self.ceilings[key], "sex"
        return self.default_ceiling, "default"

    def _resolve_pitch_range(self, utterance: Utterance) -> tuple[float, float]:
        sex = utterance.meta.get(self.sex_column)
        if not _is_missing(sex):
            key = str(sex).strip().lower()[:1]
            if key in self.pitch_ranges:
                return self.pitch_ranges[key]
        return DEFAULT_PITCH_RANGE

    def _references(self, ceiling: float) -> tuple[float, ...]:
        """Track reference frequencies: the uniform-tube resonances for ``ceiling``.

        ``(2k+1) * ceiling / 10`` reproduces Praat's published defaults exactly at
        Praat's own default 5500 Hz ceiling, and scales with a ceiling chosen per
        speaker.  Always five values: ``Track`` takes five regardless of how many
        tracks it is asked for.
        """
        if self.track_references is not None:
            return self.track_references
        return tuple((2 * k + 1) * ceiling / 10.0 for k in range(5))

    # -- analysis -------------------------------------------------------

    def _analyse(
        self, wav: np.ndarray, sample_rate: int, ceiling: float, pitch_range: tuple
    ) -> FormantTrack:
        """Burg analysis, optional Viterbi tracking, and a voicing mask."""
        sound = self._parselmouth.Sound(
            wav.astype("float64"), sampling_frequency=sample_rate
        )
        try:
            formant = sound.to_formant_burg(
                time_step=self.time_step_s,
                max_number_of_formants=self.n_formants,
                maximum_formant=ceiling,
                window_length=self.window_length_s,
                pre_emphasis_from=self.pre_emphasis_from_hz,
            )
        except self._parselmouth.PraatError:
            # Signal shorter than the analysis window, or otherwise unanalysable.
            return FormantTrack.empty(self.time_step_s, tracked=False)

        tracked = False
        if self.track:
            try:
                formant = self._call(
                    formant, "Track", len(FORMANTS), *self._references(ceiling),
                    self.frequency_cost, self.bandwidth_cost, self.transition_cost,
                )
                tracked = True
            except self._parselmouth.PraatError:
                # Raised when some frame holds fewer candidates than tracks asked
                # for.  The untracked analysis is still a measurement, and
                # `tracked` on the row says which one this is.
                tracked = False

        n_frames = formant.get_number_of_frames()
        if n_frames == 0:
            return FormantTrack.empty(self.time_step_s, tracked)

        # Praat's pitch grid does not share the formant grid's first frame centre
        # (the windows differ), so voicing is sampled BY TIME rather than by index.
        try:
            pitch = sound.to_pitch_ac(
                time_step=self.time_step_s,
                pitch_floor=pitch_range[0],
                pitch_ceiling=pitch_range[1],
            )
        except self._parselmouth.PraatError:
            pitch = None

        times = np.array(
            [formant.get_time_from_frame_number(k) for k in range(1, n_frames + 1)]
        )
        frequencies = np.full((len(FORMANTS), n_frames), np.nan)
        bandwidths = np.full((len(FORMANTS), n_frames), np.nan)
        for row, number in enumerate(FORMANTS):
            for column, time in enumerate(times):
                frequencies[row, column] = formant.get_value_at_time(number, time)
                bandwidths[row, column] = formant.get_bandwidth_at_time(number, time)

        if pitch is None:
            voiced = np.zeros(n_frames, dtype=bool)
        else:
            f0 = np.array([pitch.get_value_at_time(t) for t in times])
            voiced = np.isfinite(f0) & (f0 > 0)

        return FormantTrack(
            frequencies, bandwidths, voiced,
            float(times[0]), float(formant.dt), tracked,
        )

    # -- aggregation ----------------------------------------------------

    def _summarise(self, track: FormantTrack) -> Features:
        """Per-formant medians over voiced frames with a defined estimate.

        Formants are undefined outside voiced speech: Praat's LPC still fits poles
        to silence and to fricative noise, and summarising those would measure the
        room.  A frame counts only where the voice is periodic *and* Praat left a
        real number there.
        """
        features: Features = {}
        for row, number in enumerate(FORMANTS):
            frequency = track.frequencies[row]
            usable = track.voiced & np.isfinite(frequency) & (frequency > 0)
            values = frequency[usable]
            bandwidth = track.bandwidths[row][usable]
            bandwidth = bandwidth[np.isfinite(bandwidth)]
            features[f"f{number}_median_hz"] = (
                float(np.median(values)) if values.size else float("nan")
            )
            features[f"f{number}_sd_hz"] = (
                float(values.std(ddof=1)) if values.size > 1 else float("nan")
            )
            features[f"f{number}_bandwidth_hz"] = (
                float(np.median(bandwidth)) if bandwidth.size else float("nan")
            )
            features[f"f{number}_n_frames"] = int(usable.sum())
        return features

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

            ceiling, ceiling_source = self._resolve_ceiling(utterance)
            pitch_range = self._resolve_pitch_range(utterance)
            track = (
                self._analyse(wav, utterance.sample_rate, ceiling, pitch_range)
                if wav.size and np.any(wav)
                else FormantTrack.empty(self.time_step_s)
            )

            features = self._summarise(track)
            n_frames = int(track.voiced.size)
            features.update({
                "ceiling_hz": ceiling,
                "ceiling_source": ceiling_source,
                "n_formants": self.n_formants,
                # Per row, because an utterance whose tracking failed is still
                # measured and must not be read as if it had been tracked.
                "tracked": track.tracked,
                "n_frames": n_frames,
                "n_voiced_frames": int(track.voiced.sum()),
                "voiced_fraction": (
                    float(track.voiced.mean()) if n_frames else float("nan")
                ),
                "t0_s": track.t0_s,
                "time_step_s": track.time_step_s,
            })
            if self.contours:
                # Unvoiced frames are NaN, so a contour cannot be read as a
                # formant by accident — the same rule f0.py applies.
                for row, number in enumerate(FORMANTS):
                    frequency = track.frequencies[row]
                    features[f"contour.f{number}_hz"] = np.where(
                        track.voiced, frequency, np.nan
                    ).astype("float32")
                features["contour.voiced"] = track.voiced
            results.append(features)
        return results
