"""Manifest validation, path resolution, and requirement-driven loading."""
from __future__ import annotations

import pandas as pd
import pytest

from audio_utils.data.transforms import rms_dbfs

from speech_eval.core import Requirements
from speech_eval.io import (
    UtteranceDataset,
    collate_utterances,
    load_manifest,
    select_rows,
)


def test_load_manifest_resolves_per_corpus_roots(manifest_path, corpus):
    df = load_manifest(manifest_path, dataset_roots={"test": corpus})
    assert df["signal_path"].iloc[0] == str(corpus / "a_real.wav")


def test_load_manifest_resolves_single_root(manifest_path, corpus):
    df = load_manifest(manifest_path, dataset_root=corpus)
    assert df["signal_path"].iloc[0] == str(corpus / "a_real.wav")


def test_load_manifest_rejects_both_root_forms(manifest_path, corpus):
    with pytest.raises(ValueError, match="not both"):
        load_manifest(manifest_path, dataset_roots={"test": corpus}, dataset_root=corpus)


def test_load_manifest_requires_the_identifying_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"utt_id": "a", "signal_path": "a.wav"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_manifest(path)


def test_load_manifest_rejects_duplicate_utt_ids(tmp_path):
    # A duplicate would make one row's results silently overwrite another's.
    path = tmp_path / "dup.csv"
    row = {"utt_id": "a", "signal_path": "a.wav", "group_id": "g", "condition": "real"}
    pd.DataFrame([row, row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate utt_id"):
        load_manifest(path)


def test_select_rows_drops_rows_without_text(manifest_path, corpus):
    df = load_manifest(manifest_path, dataset_roots={"test": corpus})
    kept = select_rows(df, Requirements(needs_text=True))
    # The converted row has an empty text field, which is not a transcript.
    assert set(kept["utt_id"]) == {"a_real", "b_real"}
    assert len(select_rows(df, Requirements(needs_text=False))) == 3


def test_dataset_reads_whole_segment_at_requested_rate(manifest_path, corpus):
    df = load_manifest(manifest_path, dataset_roots={"test": corpus})
    dataset = UtteranceDataset(df, Requirements(sample_rate=8000))
    utterance = dataset[0]
    assert utterance.utt_id == "a_real"
    assert utterance.sample_rate == 8000
    assert utterance.wav.shape[0] == 1
    assert utterance.duration_s == pytest.approx(1.0, abs=0.01)


def test_dataset_keeps_native_rate_when_asked(manifest_path, corpus):
    df = load_manifest(manifest_path, dataset_roots={"test": corpus})
    dataset = UtteranceDataset(df, Requirements(sample_rate=None))
    assert dataset[0].sample_rate == 16_000


def test_dataset_applies_level_normalisation(manifest_path, corpus):
    """The two conditions of group 'a' differ by 12 dB on disk and must not after."""
    df = load_manifest(manifest_path, dataset_roots={"test": corpus})
    raw = UtteranceDataset(df, Requirements(sample_rate=None, level_norm_dbfs=None))
    assert rms_dbfs(raw[1].wav) - rms_dbfs(raw[0].wav) == pytest.approx(12.0, abs=0.1)

    normed = UtteranceDataset(df, Requirements(sample_rate=None, level_norm_dbfs=-27.0))
    assert rms_dbfs(normed[0].wav) == pytest.approx(-27.0, abs=0.01)
    assert rms_dbfs(normed[1].wav) == pytest.approx(-27.0, abs=0.01)


def test_dataset_carries_requested_metadata_only(manifest_path, corpus):
    df = load_manifest(manifest_path, dataset_roots={"test": corpus})
    dataset = UtteranceDataset(df, Requirements(), carry_columns=["subject_id", "nope"])
    assert dataset[0].meta == {"subject_id": "s1"}


def test_collate_does_not_pad(manifest_path, corpus):
    df = load_manifest(manifest_path, dataset_roots={"test": corpus})
    dataset = UtteranceDataset(df, Requirements(sample_rate=None))
    batch = collate_utterances([dataset[0], dataset[2]])
    # 1.0 s and 0.5 s: the utterances themselves keep their true lengths.
    assert [u.wav.shape[-1] for u in batch.utterances] == [16_000, 8_000]


# ---------------------------------------------------------------------------
# prepare_utterance — the in-memory path
# ---------------------------------------------------------------------------

def test_prepare_utterance_applies_every_step():
    import torch
    from audio_utils.data.transforms import rms_dbfs
    from speech_eval.core import Requirements
    from speech_eval.io import prepare_utterance

    wav = torch.randn(2, 8000) * 0.01                      # stereo, 8 kHz, quiet
    u = prepare_utterance(
        "u1", wav, 8000,
        Requirements(sample_rate=16000, mono=True, level_norm_dbfs=-20.0),
        text="hello", meta={"sex": "F"},
    )
    assert u.sample_rate == 16000
    assert u.wav.shape[0] == 1                             # mono
    assert u.wav.shape[-1] == 16000                        # resampled
    assert rms_dbfs(u.wav) == pytest.approx(-20.0, abs=0.1)
    assert u.text == "hello" and u.meta == {"sex": "F"}


def test_prepare_utterance_leaves_level_alone_when_not_asked():
    """The level metric measures amplitude; normalising it would erase the answer."""
    import torch
    from audio_utils.data.transforms import rms_dbfs
    from speech_eval.core import Requirements
    from speech_eval.io import prepare_utterance

    wav = torch.randn(1, 16000) * 0.01
    u = prepare_utterance("u1", wav, 16000,
                          Requirements(sample_rate=None, mono=True, level_norm_dbfs=None))
    assert u.sample_rate == 16000
    assert rms_dbfs(u.wav) == pytest.approx(rms_dbfs(wav), abs=1e-6)


def test_prepare_utterance_matches_the_dataset_path(tmp_path):
    """The two must agree, which is the whole point of sharing the function."""
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    from speech_eval.core import Requirements
    from speech_eval.io import UtteranceDataset, prepare_utterance

    signal = np.random.default_rng(0).normal(size=16000).astype("float32") * 0.05
    # FLOAT, not the default 16-bit PCM: otherwise the file's own quantisation (~3e-5)
    # dominates and the test measures soundfile rather than the two code paths.
    sf.write(tmp_path / "a.wav", signal, 16000, subtype="FLOAT")
    frame = pd.DataFrame([{"utt_id": "a", "signal_path": str(tmp_path / "a.wav"),
                           "group_id": "g", "condition": "real"}])
    req = Requirements(sample_rate=16000, mono=True, level_norm_dbfs=-27.0)

    from_disk = UtteranceDataset(frame, req)[0]
    in_memory = prepare_utterance("a", torch.from_numpy(signal).unsqueeze(0), 16000, req)
    assert torch.allclose(from_disk.wav, in_memory.wav, atol=1e-6)
