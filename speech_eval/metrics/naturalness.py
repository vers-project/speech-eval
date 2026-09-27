"""Predicted naturalness (MOS), for detecting degradation no other metric sees.

Every other metric here measures whether a *specific* property moved the way it
was asked to: a level, an F0, a formant, a transcription.  None of them notices
a signal that hits all of those targets and sounds wrong, which is the failure
mode a vocoder driven off its training distribution actually has.  That is what
this module is for.

UTMOS22
-------
``mos_utmos`` wraps UTMOS22 (strong), the winning system of the VoiceMOS
Challenge 2022.  It is the de-facto naturalness number in TTS and voice
conversion papers, and it was trained on BVCC — listener ratings of speech
*synthesised* by TTS and VC systems — which is the domain of a vocoder's output
rather than of recorded speech.

Read it as a relative ranking, not an absolute MOS
--------------------------------------------------
Two independent reasons, both of which bite in a vocal-effort experiment:

* The training ratings come from TTS/VC samples of ordinary conversational
  loudness.  Shouted speech is out of that distribution, so the absolute value
  on loud material is not a calibrated opinion score.
* The rating is a single scalar with no notion of what the signal was *supposed*
  to sound like, so it cannot separate "the vocoder is buzzing" from "this
  speaker has a rough voice".

Both are harmless when the comparison holds the utterance set fixed and varies
only the system under test — tracking one model across training, or ranking
systems on the same material.  Neither is harmless when comparing corpora.

ZERO PADDING CHANGES THE SCORE, SO THIS METRIC NEVER BATCHES
------------------------------------------------------------
``UTMOS22Strong.forward(wave, sr)`` takes no length mask, so padding is scored
as if it were signal — and the model rates silence *higher* than noise (2.92 vs
1.75 on one second of each), so the padding does not merely dilute the score, it
pulls it toward a different answer.  Measured on one second of noise: padding it
to two seconds moves the score by **-0.36 MOS**, which is larger than the
differences this metric is used to detect.  In a mixed-length batch only the
longest utterance escapes.

So :meth:`compute` iterates ``batch.utterances`` — whose waveforms are unpadded
by construction — and calls the model once per utterance.  Do not "optimise"
this into a padded batch through ``batch.audio``; it silently changes results.
The cost is real but small: the backbone is wav2vec 2.0 base sized, so a few
hundred utterances take seconds on a GPU.

Offline nodes
-------------
The checkpoint arrives through ``torch.hub``, not the HuggingFace cache, so it
needs ``TORCH_HOME`` pointed at a pre-populated directory rather than ``HF_HOME``
— see ``scripts/download_mos_models.py``.  No extra dependency: the hub module
needs only torch.
"""
from __future__ import annotations

from typing import Any

import torch

from speech_eval.core import (
    Features,
    Metric,
    Requirements,
    UtteranceBatch,
    register_metric,
)

#: Same value as the ASR and speaker metrics, so this metric joins their
#: requirements group and costs no additional preparation pass.  It has to be
#: set at all because conversion changes level by design: scoring unnormalised
#: audio would let loudness leak into a number that is supposed to be about
#: quality.
MOS_LEVEL_NORM_DBFS = -20.0

#: Below this the wav2vec 2.0 convolutional front end has fewer frames than its
#: kernel and raises.  Measured: 160 samples raises, 400 succeeds.
MIN_SAMPLES = 400


@register_metric("mos_utmos")
class UTMOSMetric(Metric):
    """Predicted naturalness from UTMOS22 (strong).

    Emits one key, ``score``, on a MOS-like 1--5 scale; the column is
    ``mos_utmos.score``.  An utterance shorter than :data:`MIN_SAMPLES` gets
    ``nan`` rather than an exception, since a signal too short to rate is data
    and not an error.

    Parameters
    ----------
    hub_repo : ``torch.hub`` specifier, pinned to a tag so a silent upstream
               change cannot move every score in a results table.
    entry    : the hub entry point to build.
    """

    def __init__(
        self,
        name: str | None = None,
        hub_repo: str = "tarepan/SpeechMOS:v1.2.0",
        entry: str = "utmos22_strong",
        level_norm_dbfs: float | None = MOS_LEVEL_NORM_DBFS,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.hub_repo = hub_repo
        self.entry = entry
        self.requirements = Requirements(
            sample_rate=16_000,
            mono=True,
            level_norm_dbfs=level_norm_dbfs,
        )
        self._model = None
        self._device = torch.device("cpu")

    def load(self, device: torch.device) -> None:
        if self._loaded:
            return
        self._device = device
        self._model = torch.hub.load(
            self.hub_repo, self.entry, trust_repo=True
        )
        self._model.eval()
        self._model.to(device)
        self._loaded = True

    @torch.inference_mode()
    def compute(self, batch: UtteranceBatch) -> list[Features]:
        if self._model is None:
            raise RuntimeError(
                f"{type(self).__name__}.load() must be called before compute()."
            )
        sample_rate = batch.sample_rate
        results: list[Features] = []
        # One utterance at a time, unpadded.  See the module docstring.
        for utterance in batch.utterances:
            wav = utterance.wav
            if wav.dim() == 2:
                wav = wav.mean(dim=0)
            if wav.shape[-1] < MIN_SAMPLES:
                results.append({"score": float("nan")})
                continue
            score = self._model(
                wav.unsqueeze(0).to(self._device), sample_rate
            )
            results.append({"score": float(score.reshape(-1)[0])})
        return results
