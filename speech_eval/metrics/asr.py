"""Speech recognition: the hypothesis, not the error rate.

What this module emits, and what it deliberately does not
---------------------------------------------------------
An ASR metric transcribes one utterance and stores the **raw hypothesis
string**.  It computes no WER.  Word error rate needs a second text — a ground
truth, or another condition's hypothesis — and anything needing two utterances
belongs in :mod:`speech_eval.compare`.

The consequence worth having: the text normaliser is applied at comparison time,
not at transcription time.  Changing normalisers, or scoring the same run
against ground truth *and* against a pseudo-reference, then costs no GPU.

Two backends, because the disagreement is the measurement
---------------------------------------------------------
``asr_whisper`` is autoregressive: an encoder-decoder whose decoder is a
language model.  On degraded audio it repairs what it hears into fluent English,
so its WER *understates* intelligibility loss.  ``asr_wav2vec2`` is CTC with
greedy decoding and no language model at all — every frame is scored
independently, so a destroyed phoneme stays destroyed.

Neither number is the truth on its own.  The AR model reports what a listener
with strong expectations would understand; the CTC model reports what is
acoustically present.  A converter that damages speech while leaving it
guessable moves the CTC WER and not the Whisper WER, and that gap is the
diagnostic.  Both are emitted; nothing here adjudicates between them.

Level normalisation is -20 dBFS here, not the -27 the phonetics metrics use
--------------------------------------------------------------------------
This is a Whisper constraint, checked against its source rather than inherited.
``whisper/audio.py`` ends its feature extraction with::

    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)   # relative
    log_spec = (log_spec + 4.0) / 4.0                          # absolute

The floor is relative to the utterance's own peak, so the dynamic-range window
is level-invariant.  The affine that follows is not.  Magnitudes are squared, so
a gain of X dB shifts **every input feature by X/40 units**, on a scale whose
whole span after the floor is 8/4 = 2 units.  At -27 dBFS RMS with speech's
12-15 dB crest factor, peaks sit at -12 to -15 dBFS and the features land
0.30-0.38 units low — 15-19% of the entire feature range, systematically and in
one direction.  Whisper's author states the model was trained on log magnitudes
"mostly ranging from -80 dB to 0 dB", i.e. audio filling full scale.

-20 dBFS RMS is as close to full scale as this suite can go without a peak
limiter: ``rms_normalize`` does not limit, deliberately, and clipping would
distort the signal under measurement.  Peak-normalising would match the "scale
to [-1, 1]" advice more literally, but peak level is set by a single sample and
is a poor loudness proxy — and loudness is what has to be constant across
conditions that differ in level by construction.

Two things follow.  This family gets its own preparation pass, separate from
``egemaps``/``f0`` at -27 — one extra decode over the corpus, paid so that both
ASR backends see byte-identical audio.  And ``do_normalize=True`` on
``WhisperFeatureExtractor`` is **not** the fix: it is absent from ``__init__``,
appears only in ``__call__`` defaulting to ``None``, and z-scores the *log-mel
features* rather than the waveform, which makes the model level-invariant by
destroying the feature scale it was trained on.

wav2vec2 needs none of this: its processor z-scores the raw waveform, so it is
exactly level-invariant.  It is given the same audio anyway, so that a
difference between the two columns is a difference between two models.

Unmeasurable utterances are flagged, never raised
-------------------------------------------------
Whisper's feature extractor silently truncates past 30 s, which would turn every
word after the cut into a deletion and produce a plausible-looking, badly
inflated WER.  So over-long utterances are not transcribed at all: ``status``
becomes ``"too_long"`` and the hypothesis is empty.  Raising instead would kill
a ten-hour run over one 31-second segment, and :class:`speech_eval.core.Metric`
already says an unmeasurable utterance is data rather than an error.

``status`` is also what makes resume work.  It is the one column guaranteed
non-null on every path: pandas reads an empty CSV cell back as NaN, so an
utterance that legitimately decodes to ``""`` — silence, or a conversion that
destroyed the speech — would otherwise look unfinished and be re-transcribed on
every resume, forever.

Language is configured, never detected
--------------------------------------
Whisper auto-detects language by default.  On degraded or converted audio that
detection can flip, and the instrument would then move with the signal it is
measuring — the same argument that keeps the F0 analysis range keyed to ``sex``
and never to the audio.  ``language`` is therefore a config value, optionally
overridden per row by a manifest column, and reported with ``language_source``.

Decoding is pinned deterministic for the same reason: greedy, temperature 0, no
temperature fallback, no timestamps, and **no initial prompt** — a prompt is
additional language-model prior, which is precisely the variable this pair of
backends exists to separate.

References are collected in ``references.bib`` at the repository root.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch

from speech_eval.core import (
    Features,
    Metric,
    Requirements,
    Utterance,
    UtteranceBatch,
    register_metric,
)

#: RMS level every ASR backend is fed at.  See the module docstring — this is a
#: Whisper feature-scale argument, not the -27 dBFS the phonetics metrics use.
ASR_LEVEL_NORM_DBFS = -20.0

# `status` values.  Exactly one is written on every row, always non-null.
STATUS_OK = "ok"
STATUS_TOO_LONG = "too_long"
STATUS_EMPTY = "empty_audio"


class Transcript(NamedTuple):
    """One backend's decode of one utterance.

    ``mean_logprob`` is the mean per-unit log probability the decoder assigned —
    over generated tokens for an autoregressive model, over valid frames for
    CTC.  It is a *reference-free* degradation signal, and on Whisper it also
    flags the repetition loops that degraded audio provokes.  NaN when the
    backend cannot supply one; the two backends' values are not on a common
    scale and must never be compared across columns.
    """

    text: str
    mean_logprob: float


class ASRMetric(Metric):
    """Shared behaviour of the ASR backends: guards, provenance, batch order.

    Subclasses implement :meth:`_transcribe` and nothing else.  Everything a
    reader of the results table compares across backends — what counts as an
    unmeasurable utterance, what level the audio arrived at, how a hypothesis is
    counted into words — is decided once, here, so a difference between two
    columns is a difference between two models rather than two conventions.

    Parameters
    ----------
    model_id        : backend checkpoint.  Recorded on every row: a WER is
                      meaningless without it.
    sample_rate     : rate the audio is delivered at.  Both backends are
                      16 kHz-only and will refuse anything else.
    level_norm_dbfs : RMS level to normalise to.  See the module docstring
                      before changing it.
    language        : forced decoding language, never detected.
    language_column : manifest column allowed to override it per row.  ``None``
                      disables the lookup entirely.
    max_duration_s  : utterances longer than this are flagged ``"too_long"``
                      and not transcribed.  ``None`` disables the guard.
    dtype           : compute dtype.  ``float16`` roughly halves Whisper's
                      memory and can change a hypothesis at the margin, which is
                      why it is recorded nowhere and should be held fixed across
                      a run.
    """

    #: Names the model family in the results table, independent of the instance
    #: name a config may have given it.
    backend: str = ""

    #: The architectural class, which is the thing the two-backend design is
    #: actually contrasting: "autoregressive" against "ctc_greedy".
    decoder: str = ""

    #: Default guard, overridden per backend: 30 s is Whisper's architectural
    #: limit, while CTC's is only a memory ceiling.
    default_max_duration_s: float | None = None

    #: Whether this backend's language is ours to choose.
    language_is_configurable: bool = True

    #: Checkpoint used when a config names none.
    default_model_id: str = ""

    def __init__(
        self,
        name: str | None = None,
        model_id: str | None = None,
        sample_rate: int = 16_000,
        level_norm_dbfs: float | None = ASR_LEVEL_NORM_DBFS,
        language: str = "en",
        language_column: str | None = "language",
        max_duration_s: float | None = -1.0,
        dtype: str = "float32",
        batch_size: int | None = None,
    ):
        super().__init__(name)
        self.model_id = model_id or self.default_model_id
        self.sample_rate = sample_rate
        self.language = language
        self.language_column = language_column if self.language_is_configurable else None
        # -1.0 is the "not given" sentinel, so that an explicit None can mean
        # "no guard" and still be distinguishable from the default.
        self.max_duration_s = (
            self.default_max_duration_s if max_duration_s == -1.0 else max_duration_s
        )
        self.dtype = dtype
        self.batch_size = batch_size
        self._model: Any = None
        self._processor: Any = None
        self._device = torch.device("cpu")
        self.requirements = Requirements(
            sample_rate=sample_rate,
            mono=True,
            level_norm_dbfs=level_norm_dbfs,
            meta_columns=(self.language_column,) if self.language_column else (),
        )

    @property
    def torch_dtype(self) -> torch.dtype:
        try:
            return getattr(torch, self.dtype)
        except AttributeError:
            raise ValueError(f"Unknown dtype {self.dtype!r}.") from None

    # -- subclass hook --------------------------------------------------

    def _transcribe(
        self, waveforms: list[np.ndarray], language: str
    ) -> list[Transcript]:
        """Decode a list of mono signals, returning one transcript each, in order.

        Only utterances that passed the guards reach this method, and they all
        share one ``language`` — the base class groups by it, because both
        backends decode a whole batch under a single language setting.
        """
        raise NotImplementedError

    # -- guards and provenance ------------------------------------------

    def _resolve_language(self, utterance: Utterance) -> tuple[str, str]:
        """``(language, source)`` for one utterance, from metadata alone."""
        if self.language_column:
            value = utterance.meta.get(self.language_column)
            if value is not None and not (
                isinstance(value, float) and np.isnan(value)
            ) and str(value).strip():
                return str(value).strip(), "meta"
        if not self.language_is_configurable:
            return self.language, "backend_fixed"
        return self.language, "configured"

    def _status(self, utterance: Utterance, wav: np.ndarray) -> str:
        if wav.size == 0:
            return STATUS_EMPTY
        if self.max_duration_s is not None and utterance.duration_s > self.max_duration_s:
            return STATUS_TOO_LONG
        return STATUS_OK

    def compute(self, batch: UtteranceBatch) -> list[Features]:
        if not self._loaded:
            raise RuntimeError("load() must be called before compute().")

        results: list[Features] = []
        # position in the batch → waveform, bucketed by language because both
        # backends decode under one language setting per call.
        pending: dict[str, list[tuple[int, np.ndarray]]] = {}

        for index, utterance in enumerate(batch.utterances):
            if utterance.sample_rate != self.sample_rate:
                raise ValueError(
                    f"{self.name!r} was given {utterance.sample_rate} Hz audio but is "
                    f"configured for {self.sample_rate} Hz. The runner delivers "
                    "Requirements.sample_rate; do not override one without the other."
                )
            wav = utterance.wav.detach().cpu().numpy().astype("float32")
            wav = wav.mean(axis=0) if wav.ndim > 1 else wav

            language, language_source = self._resolve_language(utterance)
            status = self._status(utterance, wav)
            results.append({
                "hypothesis": "",
                "n_words": 0,
                "mean_logprob": float("nan"),
                "status": status,
                "backend": self.backend,
                "model_id": self.model_id,
                "decoder": self.decoder,
                "language": language,
                "language_source": language_source,
                "duration_s": utterance.duration_s,
            })
            if status == STATUS_OK:
                pending.setdefault(language, []).append((index, wav))

        for language, items in pending.items():
            for start in range(0, len(items), self.batch_size or len(items)):
                chunk = items[start : start + (self.batch_size or len(items))]
                transcripts = self._transcribe([wav for _, wav in chunk], language)
                if len(transcripts) != len(chunk):
                    raise RuntimeError(
                        f"{self.name!r} decoded {len(transcripts)} of {len(chunk)} "
                        "utterances; _transcribe must return one per input, in order."
                    )
                for (index, _), transcript in zip(chunk, transcripts):
                    text = transcript.text.strip()
                    results[index].update({
                        "hypothesis": text,
                        "n_words": len(text.split()),
                        "mean_logprob": float(transcript.mean_logprob),
                    })
        return results


# ---------------------------------------------------------------------------
# Whisper — autoregressive encoder-decoder
# ---------------------------------------------------------------------------


@register_metric("asr_whisper")
class WhisperASRMetric(ASRMetric):
    """Whisper, decoded greedily with every source of randomness removed.

    The decoder is a language model, which is the entire reason this backend is
    paired with a CTC one: it will render degraded audio as fluent English and
    report a WER that flatters the system under test.  That is not a defect to
    correct — it is the quantity being measured, against ``asr_wav2vec2``.

    Generation is pinned: ``num_beams=1``, ``temperature=0.0`` (a tuple would
    re-enable Whisper's temperature *fallback*, which is stochastic and would
    make this metric unresumable and un-diffable), ``condition_on_prev_tokens``
    off, no timestamps, no initial prompt.

    ``max_duration_s`` defaults to 30 s because that is architectural: the
    feature extractor pads or truncates to a fixed 30 s window, silently.
    """

    backend = "whisper"
    decoder = "autoregressive"
    default_model_id = "openai/whisper-large-v3"
    default_max_duration_s = 30.0

    def load(self, device: torch.device) -> None:
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required for the Whisper ASR metric. Install it "
                "with:  uv sync --extra asr"
            ) from e

        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        # dtype is applied through .to() rather than from_pretrained's keyword,
        # which was renamed across transformers versions.
        self._model = model.to(device=device, dtype=self.torch_dtype).eval()
        self._device = device
        self._loaded = True

    @torch.inference_mode()
    def _transcribe(
        self, waveforms: list[np.ndarray], language: str
    ) -> list[Transcript]:
        features = self._processor.feature_extractor(
            waveforms, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features.to(device=self._device, dtype=self.torch_dtype)

        outputs = self._model.generate(
            features,
            language=language,
            task="transcribe",
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            condition_on_prev_tokens=False,
            return_timestamps=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        texts = self._processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        return [
            Transcript(text, logprob)
            for text, logprob in zip(texts, self._mean_logprobs(outputs))
        ]

    def _mean_logprobs(self, outputs: Any) -> list[float]:
        """Mean log probability per generated token, one per sequence.

        Sequences that finished early are padded, and their padded positions
        carry no information, so they are masked out rather than averaged in.
        """
        scores = self._model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        generated = outputs.sequences[:, -scores.shape[1]:]
        pad_id = self._model.generation_config.pad_token_id
        valid = generated != pad_id if pad_id is not None else torch.ones_like(
            generated, dtype=torch.bool
        )
        # compute_transition_scores leaves -inf at masked positions; zero them
        # before summing, or 0 * -inf makes the whole mean NaN.
        scores = torch.where(valid & torch.isfinite(scores), scores.float(), 0.0)
        counts = valid.sum(dim=-1).clamp(min=1)
        return (scores.sum(dim=-1) / counts).tolist()


# ---------------------------------------------------------------------------
# wav2vec2 — CTC, greedy, no language model
# ---------------------------------------------------------------------------


@register_metric("asr_wav2vec2")
class Wav2Vec2ASRMetric(ASRMetric):
    """wav2vec2 with a CTC head, decoded by argmax and nothing else.

    No beam search, no n-gram, no shallow fusion: every frame is scored
    independently of every other, so this backend cannot repair a phoneme the
    converter destroyed.  That is the point of running it beside Whisper.

    ``facebook/wav2vec2-large-960h-lv60-self`` is the Large architecture,
    pretrained on Libri-Light's 60k unlabelled hours (``lv60``), fine-tuned on
    LibriSpeech 960 h, then self-trained on pseudo-labels (``-self``).  The
    ``lv60`` checkpoints are the ones to batch: their preprocessor sets
    ``return_attention_mask=true``, whereas ``wav2vec2-base-960h`` was trained
    without masking and gives wrong answers on padded batches.

    Its processor also sets ``do_normalize=true``, which z-scores the **raw
    waveform** — unlike Whisper's, which acts after the mel filterbank — so this
    model is exactly level-invariant.  The waveforms are handed over unpadded
    and the processor does its own padding, so that the per-utterance
    normalisation is computed over real samples and never over invented silence.

    English-only, and the language is not settable: it reports
    ``language_source = "backend_fixed"``.

    ``max_duration_s`` defaults to 60 s.  Unlike Whisper's 30 s this is not
    architectural — it is a memory guard, since self-attention over a 50 fps
    frame sequence grows quadratically and an OOM cannot be turned into a
    per-utterance flag.
    """

    backend = "wav2vec2"
    decoder = "ctc_greedy"
    default_model_id = "facebook/wav2vec2-large-960h-lv60-self"
    default_max_duration_s = 60.0
    language_is_configurable = False

    def load(self, device: torch.device) -> None:
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as e:
            raise ImportError(
                "transformers is required for the wav2vec2 ASR metric. Install it "
                "with:  uv sync --extra asr"
            ) from e

        self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        if not self._processor.feature_extractor.return_attention_mask:
            raise ValueError(
                f"{self.model_id} was trained without an attention mask, so batched "
                "inference over padded audio gives wrong results. Use an `lv60` "
                "checkpoint, or run this metric with batch_size=1."
            )
        model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        self._model = model.to(device=device, dtype=self.torch_dtype).eval()
        self._device = device
        self._loaded = True

    @torch.inference_mode()
    def _transcribe(
        self, waveforms: list[np.ndarray], language: str
    ) -> list[Transcript]:
        inputs = self._processor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        attention_mask = inputs.attention_mask.to(self._device)
        logits = self._model(
            inputs.input_values.to(device=self._device, dtype=self.torch_dtype),
            attention_mask=attention_mask,
        ).logits.float()

        # Frames beyond an utterance's own length are padding, and decoding or
        # averaging over them would mix in the neighbour that set the batch width.
        frame_counts = self._model._get_feat_extract_output_lengths(
            attention_mask.sum(dim=-1)
        ).tolist()
        log_probs = logits.log_softmax(dim=-1)
        predicted = logits.argmax(dim=-1)

        transcripts = []
        for row, n_frames in enumerate(frame_counts):
            n_frames = max(int(n_frames), 1)
            text = self._processor.decode(predicted[row, :n_frames])
            confidence = log_probs[row, :n_frames].max(dim=-1).values.mean()
            transcripts.append(Transcript(text, float(confidence)))
        return transcripts
