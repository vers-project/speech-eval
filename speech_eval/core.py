"""Core types: what a metric is, what it is given, and what it returns.

Design
------
A metric is a **per-utterance feature extractor**, never a comparison.  It sees
one utterance at a time (batched for efficiency) and emits features describing
*that* utterance: a transcription hypothesis, a speaker embedding, a median F0,
an alpha ratio.  Anything requiring two utterances — WER against a reference,
cosine similarity against an enrolment, a recovery rate — is computed downstream
in :mod:`speech_eval.compare`, from the features this layer produced.

Three consequences make this worth the indirection:

* A file that serves as the reference for several pairs is embedded **once**.
  With five conditions per utterance, the pairwise formulation would re-run the
  model on the same audio four extra times.
* Re-designing the comparison — a different pairing, an added baseline, another
  text normaliser — costs no GPU time, because the expensive per-utterance stage
  is already on disk.
* ``speech_eval`` never learns what a "condition" is, which is what keeps it
  reusable across experiments.

Requirements drive preprocessing
--------------------------------
Each metric declares a :class:`Requirements` describing the audio it needs
rather than preparing that audio itself.  The runner groups metrics by
requirements and prepares each distinct form once, so ten metrics wanting 16 kHz
mono cause one resample, not ten.  It also puts level normalisation somewhere
visible: ASR, MOS and speaker embedders are all level-sensitive, and a suite
comparing signals that *differ in level by construction* will silently measure
gain instead of degradation unless normalisation is declared per metric.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor

from audio_utils.data.audio_batch import AudioBatch

# What a metric emits per utterance.  Scalars and strings land in the results
# CSV; ndarray values (embeddings, contours) go to the artifact archive.
Features = dict[str, Any]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Requirements:
    """The form of audio a metric needs.  Frozen so the runner can group on it.

    Parameters
    ----------
    sample_rate     : rate to resample to.  ``None`` keeps the file's native
                      rate — correct for measurements that must not be touched
                      by a resampler, and for metrics whose backend resamples
                      internally.
    mono            : average all channels down to one.  ``False`` passes the
                      selected channel(s) through unchanged.
    level_norm_dbfs : RMS-normalise each utterance to this level in dBFS before
                      the metric sees it, or ``None`` to leave amplitudes alone.
                      Set it for every metric that is level-sensitive but whose
                      *question* is not about level (ASR, MOS, speaker identity);
                      leave it ``None`` for metrics that measure level itself,
                      and for level-invariant ratios where it makes no
                      difference.
    needs_text      : the metric requires a reference transcript; rows without
                      one are skipped rather than passed a ``None``.
    needs_intervals : the metric requires phone/word intervals from forced
                      alignment.  Not yet supported — see
                      :mod:`speech_eval.runner`.
    meta_columns    : manifest columns the metric needs in order to *compute* —
                      ``("sex",)`` for a pitch tracker whose analysis range is
                      sex-conditional.  They arrive in :attr:`Utterance.meta`.
                      Deliberately not the same thing as the runner's
                      ``carry_columns``, which say what the results table
                      *shows*: a metric may need a column it does not report,
                      and a report may carry a column no metric reads.  Names
                      audio the metric never sees, so the runner drops it when
                      grouping — see :func:`speech_eval.runner.group_by_requirements`.
    """

    sample_rate: int | None = 16_000
    mono: bool = True
    level_norm_dbfs: float | None = None
    needs_text: bool = False
    needs_intervals: bool = False
    meta_columns: tuple[str, ...] = ()


@dataclass
class Utterance:
    """One audio item, unpadded.

    ``wav`` is (C, T) and carries its true length: padding is the batch's job,
    never the dataset's, so that a mask can never declare invented silence to be
    valid audio.
    """

    utt_id: str
    wav: Tensor
    sample_rate: int
    text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.wav.shape[-1] / self.sample_rate


@dataclass
class UtteranceBatch:
    """A batch of utterances, exposed in both shapes metrics actually want.

    Neural metrics take :attr:`audio` — a padded :class:`AudioBatch` with the
    mask needed to ignore the padding.  Phonetics backends (Praat, LPC) run one
    signal at a time and take :attr:`utterances`, whose waveforms are unpadded,
    so no analysis window ever straddles fabricated silence.
    """

    utterances: list[Utterance]
    _audio: AudioBatch | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.utterances)

    @property
    def utt_ids(self) -> list[str]:
        return [u.utt_id for u in self.utterances]

    @property
    def sample_rate(self) -> int:
        return self.utterances[0].sample_rate

    @property
    def audio(self) -> AudioBatch:
        """Padded (B, C, T) view.  Built on first access, then cached.

        Only valid when every utterance shares a sample rate, which the runner
        guarantees within a requirements group unless ``sample_rate=None`` was
        requested over a corpus of mixed rates.
        """
        if self._audio is None:
            rates = {u.sample_rate for u in self.utterances}
            if len(rates) > 1:
                raise ValueError(
                    f"Cannot batch utterances at mixed sample rates {sorted(rates)}. "
                    "Set Requirements.sample_rate to a fixed value, or consume "
                    "`batch.utterances` one at a time."
                )
            self._audio = AudioBatch.from_list(
                [u.wav for u in self.utterances], self.sample_rate
            )
        return self._audio


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class Metric(ABC):
    """Base class for every metric.

    Subclasses set :attr:`requirements`, optionally override :meth:`load` to
    build a model, and implement :meth:`compute`.

    ``name`` prefixes every column the metric writes (``asr_whisper.hypothesis``),
    so the same metric class can be instantiated twice under different names —
    two ASR backends, two speaker embedders — and their outputs coexist in one
    table.  Reading two independent instruments and reporting their disagreement
    is the intended use, not an edge case.
    """

    #: Audio this metric needs.  Overridden by subclasses.
    requirements: Requirements = Requirements()

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.metric_type  # type: ignore[attr-defined]
        self._loaded = False

    # -- lifecycle ------------------------------------------------------

    def load(self, device: torch.device) -> None:
        """Build models and move them to ``device``.

        Loading here rather than in ``__init__`` keeps construction free of
        downloads, so a config can be validated, and a run can fail on a bad
        manifest, without touching the network or a GPU.

        Subclasses override this and are **not** required to be idempotent.
        Callers should use :meth:`ensure_loaded`.
        """
        self._loaded = True

    def ensure_loaded(self, device: torch.device) -> None:
        """Call :meth:`load` at most once per instance.

        Subclass ``load`` implementations rebuild their backend unconditionally —
        they fetch from a hub, construct a model and move it to the device — so a
        caller that loads twice pays the download and the construction twice and
        leaves the first copy on the GPU until it is collected.  That never
        happened while a metric was built, used and dropped inside one script, and
        it happens every time a long-lived owner (a training callback evaluating
        every N epochs) reuses its metric instances.  The guard lives here rather
        than in each subclass so no new metric has to remember it.
        """
        if self._loaded:
            return
        self.load(device)
        self._loaded = True

    # -- computation ----------------------------------------------------

    @abstractmethod
    def compute(self, batch: UtteranceBatch) -> list[Features]:
        """Return one :data:`Features` dict per utterance, in batch order.

        Keys are bare (``"median_f0_hz"``); the runner prefixes them with the
        metric's name.  A metric that cannot measure an utterance should emit
        ``float("nan")`` for the affected keys rather than omitting them or
        raising — a short or unvoiced signal is data, not an error.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Metric]] = {}


def register_metric(metric_type: str) -> Callable[[type[Metric]], type[Metric]]:
    """Class decorator registering a metric under a config-visible type name."""

    def decorator(cls: type[Metric]) -> type[Metric]:
        if metric_type in _REGISTRY:
            raise ValueError(f"Metric type {metric_type!r} is already registered.")
        cls.metric_type = metric_type  # type: ignore[attr-defined]
        _REGISTRY[metric_type] = cls
        return cls

    return decorator


def available_metrics() -> list[str]:
    return sorted(_REGISTRY)


def build_metric(spec: dict[str, Any]) -> Metric:
    """Instantiate one metric from a config entry.

    ``spec`` is ``{"type": ..., "name": ..., **kwargs}``; ``name`` defaults to
    ``type`` and every other key is passed to the constructor.
    """
    spec = dict(spec)
    try:
        metric_type = spec.pop("type")
    except KeyError:
        raise KeyError(f"Metric config entry has no 'type' key: {spec}") from None
    if metric_type not in _REGISTRY:
        if not _REGISTRY:
            # An empty Available list is not a misspelling, it is a bypassed
            # import: the registry is populated by importing `speech_eval.metrics`,
            # which `import speech_eval` does.  Said here because the bare
            # "Available: []" cost three separate debugging sessions.
            raise KeyError(
                f"Unknown metric type {metric_type!r} because NO metric is "
                "registered. The registry is populated by importing "
                "`speech_eval.metrics` (which `import speech_eval` does for you), "
                "so an empty list means that import was bypassed."
            )
        raise KeyError(
            f"Unknown metric type {metric_type!r}. Available: {available_metrics()}"
        )
    return _REGISTRY[metric_type](**spec)


def build_metrics(specs: list[dict[str, Any]]) -> list[Metric]:
    """Instantiate a list of metrics, rejecting duplicate names.

    Duplicate names would silently overwrite each other's columns, which is
    exactly the failure mode the ``name`` override exists to avoid.
    """
    metrics = [build_metric(s) for s in specs]
    names = [m.name for m in metrics]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(
            f"Duplicate metric names {sorted(duplicates)}. Give each entry a "
            "distinct 'name' when running the same metric type more than once."
        )
    return metrics
