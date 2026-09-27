"""Speaker identity: the embedding, not the similarity.

What this module emits, and what it deliberately does not
---------------------------------------------------------
A speaker metric embeds one utterance and stores the **vector**.  It computes no
cosine.  Similarity needs a second utterance, and anything needing two
utterances belongs in :mod:`speech_eval.compare` — where the corpus's own
same-speaker and different-speaker distributions turn a bare cosine into a
number that can be reported.  See :mod:`speech_eval.compare.speaker`.

The consequence worth having is the same one the ASR family buys: an utterance
that serves as the reference for four other conditions is embedded once, and
re-deciding the pairing, the enrolment or the calibration afterwards costs no
GPU at all.

Two backends, chosen to be genuinely independent
------------------------------------------------
``speaker_ecapa`` is ECAPA-TDNN (Desplanques et al. 2020): an 80-band log-mel
filterbank into a time-delay CNN with squeeze-excitation, multi-scale feature
aggregation and attentive statistics pooling, trained on VoxCeleb1+2 with AAM
softmax.  192 dimensions.

``speaker_wavlm`` is ``WavLMForXVector``: a self-supervised transformer
(Chen et al. 2022) over the raw waveform, with an x-vector head (Snyder et al.
2018) on top, fine-tuned on VoxCeleb1.  512 dimensions.

They share no frontend, no architecture and no training set, which is the point.
Two checkpoints differing only in their SSL encoder would agree with each other
for reasons that have nothing to do with the speech under test.  Neither is
adjudicated here; both columns are emitted and their disagreement is data.

Neither has ever seen high vocal effort
---------------------------------------
Both are VoxCeleb-trained — celebrity interview speech — and both were trained
with a criterion that makes every within-speaker variation, including speaking
style, something to be *invariant to*.  Two consequences for this project.

A small effect of vocal effort on embedding distance is a plausible and
informative result rather than a broken measurement: it would say the embedders
discard the very variation the converter manipulates.  And because that
invariance was learned rather than guaranteed, it may hold on real loud speech
and fail on a converter's artifacts, which is a different distribution again.
Both readings are testable from what this suite produces, by comparing the
effort-only contrast against the conversion contrast on the same speakers.  That
is a ``compare.py`` job, and it is why the raw cosine distributions are worth
reporting alongside any threshold-based accept rate.

Level normalisation is on, and the two backends need it unequally
-----------------------------------------------------------------
Checked against each checkpoint's own configuration rather than assumed.

``microsoft/wavlm-base-plus-sv`` ships ``do_normalize: false`` in its
``preprocessor_config.json``.  Unlike the wav2vec2 ASR checkpoint — whose
processor z-scores the raw waveform and is exactly level-invariant — this one
passes amplitudes straight through, so it **is** level-sensitive and the
normalisation is load-bearing.

``speechbrain/spkrec-ecapa-voxceleb`` is the opposite.  Its ``hyperparams.yaml``
sets ``mean_var_norm: InputNormalization(norm_type: sentence, std_norm: False)``,
which subtracts each log-mel bin's own mean over time.  A gain of *g* multiplies
every energy by *g²* and so adds the constant ``2·ln g`` to every bin at every
frame; subtracting the per-bin time mean removes it exactly.  ECAPA is therefore
analytically level-invariant, down to where the filterbank's log floor starts to
bite.  ``tests/test_speaker.py`` probes this rather than trusting the reasoning.

So the level is set by the backend that cares, and the one that does not is fed
the same audio anyway, so that a difference between the two columns is a
difference between two models.  The **value**, -20 dBFS, has no embedder-side
argument behind it: it is the ASR family's level, chosen so the runner prepares
one audio pass for ASR and speaker together instead of two.  Say so rather than
inventing a justification.

Duration is the confound to watch
---------------------------------
Speaker embeddings get better with length, steeply below a few seconds.  A
comparison between conditions of *different* duration therefore measures length
as much as identity.  For a frame-synchronous converter the durations match by
construction and this is a non-issue; where it is not, it is fatal and silent.

Short utterances are flagged and still embedded
-----------------------------------------------
``status`` is ``"short"`` below ``min_duration_s`` — and the embedding is
computed anyway.  This is deliberately unlike the ASR family, where ``too_long``
means Whisper silently truncated the audio and the hypothesis is *wrong*.  A
short embedding is not wrong, only noisier, and throwing it away here would
force the decision on every downstream reader.  So the decision is theirs:
:mod:`speech_eval.compare.speaker` accepts ``("ok", "short")`` by default and
takes ``ok_statuses=("ok",)`` or ``min_duration_s`` to be stricter.

Only ``empty_audio`` yields no usable vector; those rows carry an all-NaN
embedding, which propagates to NaN through any cosine rather than quietly
scoring as an orthogonal direction.

As in the ASR family, ``status`` is also what makes resume terminate: it is the
one column guaranteed non-null on every path.

References are collected in ``references.bib`` at the repository root.
"""
from __future__ import annotations

from typing import Any

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

#: RMS level every speaker embedder is fed at.  Shared with the ASR family so
#: the runner prepares the audio once; see the module docstring — no
#: embedder-side argument selects this value.
SPEAKER_LEVEL_NORM_DBFS = -20.0

#: Utterances shorter than this are flagged ``"short"`` and embedded anyway.
#: Verification accuracy falls off steeply below a few seconds, and 2 s is where
#: the VoxCeleb evaluation conditions themselves stop being representative.
DEFAULT_MIN_DURATION_S = 2.0

# `status` values.  Exactly one is written on every row, always non-null.
STATUS_OK = "ok"
STATUS_SHORT = "short"
STATUS_EMPTY = "empty_audio"

#: Length of the signal used to read a backend's embedding dimension at load
#: time.  Noise rather than silence, so a log-domain frontend is not asked for
#: the logarithm of zero.
PROBE_SECONDS = 0.5


class SpeakerEmbeddingMetric(Metric):
    """Shared behaviour of the speaker embedders: guards, provenance, batch order.

    Subclasses implement :meth:`_load_backend` and :meth:`_embed` and nothing
    else.  Everything a reader compares across backends — what counts as an
    unusable utterance, what level the audio arrived at, whether the stored
    vector is normalised — is decided once, here, so a difference between two
    columns is a difference between two models rather than two conventions.

    Parameters
    ----------
    model_id        : backend checkpoint.  Recorded on every row: a similarity
                      is meaningless without it.
    sample_rate     : rate the audio is delivered at.  Both backends are
                      16 kHz-only.
    level_norm_dbfs : RMS level to normalise to.  See the module docstring
                      before changing it.
    min_duration_s  : below this, ``status`` becomes ``"short"``.  The embedding
                      is still computed.  ``None`` disables the flag.
    dtype           : compute dtype.  Hold it fixed across a run.
    batch_size      : inner batch, if the loader's is too large for the model.
    """

    #: Names the model family in the results table, independent of the instance
    #: name a config may have given it.
    backend: str = ""

    #: The architectural contrast the two-backend design exists to exploit:
    #: "log_mel" against "ssl_transformer".
    frontend: str = ""

    #: Checkpoint used when a config names none.
    default_model_id: str = ""

    def __init__(
        self,
        name: str | None = None,
        model_id: str | None = None,
        sample_rate: int = 16_000,
        level_norm_dbfs: float | None = SPEAKER_LEVEL_NORM_DBFS,
        min_duration_s: float | None = DEFAULT_MIN_DURATION_S,
        dtype: str = "float32",
        batch_size: int | None = None,
    ):
        super().__init__(name)
        self.model_id = model_id or self.default_model_id
        self.sample_rate = sample_rate
        self.min_duration_s = min_duration_s
        self.dtype = dtype
        self.batch_size = batch_size
        self._model: Any = None
        self._processor: Any = None
        self._device = torch.device("cpu")
        self._dim: int | None = None
        self.requirements = Requirements(
            sample_rate=sample_rate,
            mono=True,
            level_norm_dbfs=level_norm_dbfs,
        )

    @property
    def torch_dtype(self) -> torch.dtype:
        try:
            return getattr(torch, self.dtype)
        except AttributeError:
            raise ValueError(f"Unknown dtype {self.dtype!r}.") from None

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("load() must be called before the dimension is known.")
        return self._dim

    # -- subclass hooks -------------------------------------------------

    def _load_backend(self, device: torch.device) -> None:
        """Build the model on ``device``.  Called by :meth:`load`."""
        raise NotImplementedError

    def _embed(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Embed a list of mono signals, returning ``(len(waveforms), dim)``.

        Signals arrive unpadded and at their true lengths; padding for the
        backend, and telling the backend where the padding is, is the subclass's
        responsibility.  Getting that wrong is the characteristic bug of this
        family — it embeds silence as if it were speech, and produces plausible
        vectors that are quietly wrong.
        """
        raise NotImplementedError

    # -- lifecycle ------------------------------------------------------

    def load(self, device: torch.device) -> None:
        """Build the backend, then read its embedding dimension from a forward pass.

        The dimension is measured rather than declared as a constant: a
        checkpoint swapped in through ``model_id`` may not have the dimension
        this class was written against, and a wrong constant would only surface
        as a shape error much later, inside the archive.
        """
        self._load_backend(device)
        self._device = device
        rng = np.random.default_rng(0)
        probe = (1e-2 * rng.standard_normal(
            int(self.sample_rate * PROBE_SECONDS)
        )).astype("float32")
        self._dim = int(np.asarray(self._embed([probe])).shape[-1])
        self._loaded = True

    # -- computation ----------------------------------------------------

    def _status(self, utterance: Utterance, wav: np.ndarray) -> str:
        if wav.size == 0:
            return STATUS_EMPTY
        if self.min_duration_s is not None and utterance.duration_s < self.min_duration_s:
            return STATUS_SHORT
        return STATUS_OK

    def compute(self, batch: UtteranceBatch) -> list[Features]:
        if not self._loaded:
            raise RuntimeError("load() must be called before compute().")

        results: list[Features] = []
        pending: list[tuple[int, np.ndarray]] = []

        for index, utterance in enumerate(batch.utterances):
            if utterance.sample_rate != self.sample_rate:
                raise ValueError(
                    f"{self.name!r} was given {utterance.sample_rate} Hz audio but is "
                    f"configured for {self.sample_rate} Hz. The runner delivers "
                    "Requirements.sample_rate; do not override one without the other."
                )
            wav = utterance.wav.detach().cpu().numpy().astype("float32")
            wav = wav.mean(axis=0) if wav.ndim > 1 else wav

            status = self._status(utterance, wav)
            results.append({
                "embedding": np.full(self.embedding_dim, np.nan, dtype="float32"),
                "embedding_norm": float("nan"),
                "dim": self.embedding_dim,
                "status": status,
                "backend": self.backend,
                "frontend": self.frontend,
                "model_id": self.model_id,
                "duration_s": utterance.duration_s,
            })
            # "short" is advisory, not fatal: it is embedded like any other.
            if status != STATUS_EMPTY:
                pending.append((index, wav))

        for start in range(0, len(pending), self.batch_size or max(len(pending), 1)):
            chunk = pending[start : start + (self.batch_size or len(pending))]
            embeddings = np.asarray(self._embed([wav for _, wav in chunk]))
            if embeddings.shape[0] != len(chunk):
                raise RuntimeError(
                    f"{self.name!r} embedded {embeddings.shape[0]} of {len(chunk)} "
                    "utterances; _embed must return one row per input, in order."
                )
            for (index, _), vector in zip(chunk, embeddings):
                vector = np.asarray(vector, dtype="float32")
                norm = float(np.linalg.norm(vector))
                # Stored as a unit vector so the archive holds one convention and
                # every cosine downstream is a plain dot product.  The length is
                # kept as a scalar rather than discarded: it varies with duration
                # and signal quality, and is occasionally a useful diagnostic.
                results[index].update({
                    "embedding": vector / norm if norm > 0 else vector,
                    "embedding_norm": norm,
                })
        return results


# ---------------------------------------------------------------------------
# ECAPA-TDNN — log-mel filterbank into a time-delay CNN
# ---------------------------------------------------------------------------


@register_metric("speaker_ecapa")
class ECAPASpeakerMetric(SpeakerEmbeddingMetric):
    """ECAPA-TDNN via speechbrain, in the configuration speechbrain verifies with.

    Two details of the toolkit are load-bearing and both are pinned here rather
    than left at a default.

    ``encode_batch`` takes ``wav_lens`` as **relative** lengths — its docstring
    says "lengths of the waveforms relative to the longest one in the batch […]
    the longest one should have relative length 1.0".  They default to ``None``,
    which means all ones, which tells the model that every utterance fills the
    padded width.  The attentive statistics pooling would then average over the
    zero padding of every utterance that is not the longest, weighting it by
    however much padding it happens to carry.  The lengths are therefore always
    passed explicitly.

    ``normalize=False`` on ``encode_batch`` leaves the checkpoint's
    ``mean_var_norm_emb`` unapplied.  That is not a preference: it is what
    speechbrain's own ``SpeakerRecognition.verify_batch`` does, so it is the
    configuration the published verification numbers are measured in.

    Offline nodes need ``savedir`` pointed at a pre-populated directory — see
    ``scripts/download_speaker_models.py``.
    """

    backend = "ecapa_tdnn"
    frontend = "log_mel"
    default_model_id = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, *args: Any, savedir: str | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.savedir = savedir

    def _load_backend(self, device: torch.device) -> None:
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError as e:
            raise ImportError(
                "speechbrain is required for the ECAPA speaker metric. Install it "
                "with:  uv sync --extra speaker"
            ) from e

        self._model = EncoderClassifier.from_hparams(
            source=self.model_id,
            savedir=self.savedir,
            run_opts={"device": str(device)},
        )
        self._model.eval()

    @torch.inference_mode()
    def _embed(self, waveforms: list[np.ndarray]) -> np.ndarray:
        lengths = np.array([len(w) for w in waveforms], dtype=np.int64)
        width = int(lengths.max())
        padded = np.zeros((len(waveforms), width), dtype="float32")
        for row, wav in enumerate(waveforms):
            padded[row, : len(wav)] = wav

        embeddings = self._model.encode_batch(
            torch.from_numpy(padded).to(self._device),
            # Relative, not absolute.  See the class docstring.
            torch.from_numpy(lengths / width).float().to(self._device),
            normalize=False,
        )
        # ECAPA emits (batch, 1, dim): the pooling collapses time to a single
        # frame rather than dropping the axis.
        return embeddings.squeeze(1).float().cpu().numpy()


# ---------------------------------------------------------------------------
# WavLM + x-vector head — self-supervised transformer over the raw waveform
# ---------------------------------------------------------------------------


@register_metric("speaker_wavlm")
class WavLMSpeakerMetric(SpeakerEmbeddingMetric):
    """WavLM with an x-vector head, through transformers.

    ``microsoft/wavlm-base-plus-sv`` is the base-plus SSL encoder — 94k hours of
    Libri-Light, VoxPopuli and GigaSpeech — with the TDNN x-vector head
    fine-tuned on VoxCeleb1.  Its processor sets ``return_attention_mask: true``,
    so batching over padded audio is safe **provided the mask is passed**, which
    it is; and ``do_normalize: false``, which is why the level normalisation in
    ``Requirements`` matters for this backend and not for ECAPA.

    Nothing about the SSL frontend is fine-tuned per condition, and no layer
    weighting is configured here beyond the checkpoint's own.
    """

    backend = "wavlm_xvector"
    frontend = "ssl_transformer"
    default_model_id = "microsoft/wavlm-base-plus-sv"

    def _load_backend(self, device: torch.device) -> None:
        try:
            from transformers import AutoFeatureExtractor, WavLMForXVector
        except ImportError as e:
            raise ImportError(
                "transformers is required for the WavLM speaker metric. Install it "
                "with:  uv sync --extra speaker"
            ) from e

        self._processor = AutoFeatureExtractor.from_pretrained(self.model_id)
        if not self._processor.return_attention_mask:
            raise ValueError(
                f"{self.model_id}'s feature extractor returns no attention mask, so "
                "batched inference over padded audio would embed the padding. Run "
                "this metric with batch_size=1, or use a checkpoint that provides one."
            )
        model = WavLMForXVector.from_pretrained(self.model_id)
        self._model = model.to(device=device, dtype=self.torch_dtype).eval()

    @torch.inference_mode()
    def _embed(self, waveforms: list[np.ndarray]) -> np.ndarray:
        inputs = self._processor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        embeddings = self._model(
            inputs.input_values.to(device=self._device, dtype=self.torch_dtype),
            attention_mask=inputs.attention_mask.to(self._device),
        ).embeddings
        return embeddings.float().cpu().numpy()
