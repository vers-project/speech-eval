"""Pre-download the ASR backends so an offline compute node can run them.

Jean-Zay's compute nodes have no route to the model hub, so every checkpoint has
to be in the cache before the job starts.  Run this on a login node:

    uv run --extra cpu --extra asr scripts/download_asr_models.py

What it fetches, and why each piece is needed:

* the Whisper **model and processor** — ``asr_whisper``;
* the Whisper **tokenizer** — not only for decoding, but because the
  British-American spelling map that ``EnglishTextNormalizer`` requires ships
  inside the tokenizer rather than the model.  Forgetting it makes the *scoring*
  step fail on an offline node, long after transcription succeeded;
* the wav2vec2 **model and processor** — ``asr_wav2vec2``.

It then builds the normaliser and runs both models on one second of silence,
which is the only way to find out that the cache is complete before a job does.
Plain argparse rather than the launcher's ``@parse_args``: this is a one-shot
helper with no sweep, no SLURM and no experiment config.
"""
from __future__ import annotations

import argparse

import torch

WHISPER_MODEL_ID = "openai/whisper-large-v3"
WAV2VEC2_MODEL_ID = "facebook/wav2vec2-large-960h-lv60-self"
SAMPLE_RATE = 16_000


def fetch_whisper(model_id: str, verify: bool) -> None:
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    print(f"[download] {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    tokenizer = WhisperTokenizer.from_pretrained(model_id)
    mapping = getattr(tokenizer, "english_spelling_normalizer", None)
    if mapping is None:
        raise RuntimeError(
            f"{model_id}'s tokenizer exposes no english_spelling_normalizer; the "
            "'whisper_en' normaliser in compare.py cannot be built from it."
        )
    normalizer = EnglishTextNormalizer(mapping)
    print(f"[download]   normaliser ok: {normalizer('He said twenty-three, uh, colours')!r}")

    if verify:
        silence = torch.zeros(SAMPLE_RATE).numpy()
        features = processor.feature_extractor(
            silence, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        tokens = model.generate(
            features, language="en", task="transcribe", num_beams=1, do_sample=False,
            temperature=0.0, condition_on_prev_tokens=False, return_timestamps=False,
        )
        text = processor.batch_decode(tokens, skip_special_tokens=True)[0]
        print(f"[download]   forward pass ok: {text.strip()!r}")


def fetch_wav2vec2(model_id: str, verify: bool) -> None:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print(f"[download] {model_id}")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    if not processor.feature_extractor.return_attention_mask:
        # Not fatal here, but the metric refuses to batch such a checkpoint, so
        # say it now rather than after the audio has been staged.
        print(
            f"[download]   WARNING: {model_id} was trained without an attention "
            "mask; asr_wav2vec2 will refuse to batch it."
        )

    if verify:
        silence = torch.zeros(SAMPLE_RATE).numpy()
        inputs = processor(silence, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs.input_values).logits
        text = processor.decode(logits.argmax(dim=-1)[0])
        print(f"[download]   forward pass ok: {text.strip()!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whisper-model-id", default=WHISPER_MODEL_ID)
    parser.add_argument("--wav2vec2-model-id", default=WAV2VEC2_MODEL_ID)
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="download only, without a forward pass over one second of silence.",
    )
    args = parser.parse_args()

    fetch_whisper(args.whisper_model_id, verify=not args.skip_verify)
    fetch_wav2vec2(args.wav2vec2_model_id, verify=not args.skip_verify)
    print("[download] done. Both backends and the normaliser are cached.")


if __name__ == "__main__":
    main()
