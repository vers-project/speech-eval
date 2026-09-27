"""Pre-download the speaker embedders so an offline compute node can run them.

Jean-Zay's compute nodes have no route to the model hub, so every checkpoint has
to be in the cache before the job starts.  Run this on a login node:

    uv run --extra cpu --extra speaker scripts/download_speaker_models.py \\
        --ecapa-savedir ~/.cache/speechbrain/spkrec-ecapa-voxceleb

What it fetches:

* **ECAPA-TDNN** through speechbrain.  Unlike a transformers checkpoint this one
  does not live in the HuggingFace cache by default — speechbrain copies it into
  a ``savedir`` of its own and loads from there.  Pass the same ``--ecapa-savedir``
  here and as the metric's ``savedir:`` in the run config, or the compute node
  will try to fetch it again and fail.
* **WavLM + x-vector head** through transformers, which does use the normal
  ``HF_HOME`` cache.

It then runs both models on half a second of noise and prints the embedding
dimension each returned, which is the only way to find out that the cache is
complete before a job does.  Plain argparse rather than the launcher's
``@parse_args``: this is a one-shot helper with no sweep, no SLURM and no
experiment config.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
WAVLM_MODEL_ID = "microsoft/wavlm-base-plus-sv"
SAMPLE_RATE = 16_000
PROBE_SECONDS = 0.5


def probe_signal(seed: int = 0) -> np.ndarray:
    """Noise rather than silence, so a log-domain frontend stays finite."""
    rng = np.random.default_rng(seed)
    return (1e-2 * rng.standard_normal(int(SAMPLE_RATE * PROBE_SECONDS))).astype(
        "float32"
    )


def fetch_ecapa(model_id: str, savedir: str | None, verify: bool) -> None:
    from speechbrain.inference.classifiers import EncoderClassifier

    print(f"[download] {model_id}")
    if savedir is None:
        print(
            "[download]   WARNING: no --ecapa-savedir given, so speechbrain will use "
            "its default location. Name it explicitly if this cache has to be reused "
            "on an offline node."
        )
    model = EncoderClassifier.from_hparams(
        source=model_id, savedir=savedir, run_opts={"device": "cpu"}
    )
    model.eval()
    if savedir:
        print(f"[download]   savedir: {savedir}")

    if verify:
        wav = torch.from_numpy(probe_signal()).unsqueeze(0)
        with torch.inference_mode():
            # normalize=False matches speechbrain's own verify_batch, which is
            # the configuration speaker_ecapa runs in.
            embedding = model.encode_batch(
                wav, torch.ones(1), normalize=False
            ).squeeze(1)
        print(f"[download]   forward pass ok: {tuple(embedding.shape)}")


def fetch_wavlm(model_id: str, verify: bool) -> None:
    from transformers import AutoFeatureExtractor, WavLMForXVector

    print(f"[download] {model_id}")
    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = WavLMForXVector.from_pretrained(model_id).eval()

    if not processor.return_attention_mask:
        # Not fatal here, but the metric refuses to batch such a checkpoint, so
        # say it now rather than after the audio has been staged.
        print(
            f"[download]   WARNING: {model_id}'s feature extractor returns no "
            "attention mask; speaker_wavlm will refuse to batch it."
        )
    if processor.do_normalize:
        # Worth knowing if a future checkpoint changes it: the level-normalisation
        # argument in speech_eval/metrics/speaker.py rests on this being False.
        print(
            f"[download]   note: {model_id} sets do_normalize=True, so it is level-"
            "invariant — unlike the checkpoint this metric was written against."
        )

    if verify:
        inputs = processor(
            [probe_signal()], sampling_rate=SAMPLE_RATE, return_tensors="pt",
            padding=True,
        )
        with torch.inference_mode():
            embedding = model(
                inputs.input_values, attention_mask=inputs.attention_mask
            ).embeddings
        print(f"[download]   forward pass ok: {tuple(embedding.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecapa-model-id", default=ECAPA_MODEL_ID)
    parser.add_argument("--wavlm-model-id", default=WAVLM_MODEL_ID)
    parser.add_argument(
        "--ecapa-savedir", default=None,
        help="where speechbrain copies the ECAPA checkpoint. Pass the same path "
             "as the metric's `savedir:` so an offline node finds it.",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="download only, without a forward pass over half a second of noise.",
    )
    args = parser.parse_args()

    fetch_ecapa(args.ecapa_model_id, args.ecapa_savedir, verify=not args.skip_verify)
    fetch_wavlm(args.wavlm_model_id, verify=not args.skip_verify)
    print("[download] done. Both speaker embedders are cached.")


if __name__ == "__main__":
    main()
