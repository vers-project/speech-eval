"""Manifest loading and the dataset that turns manifest rows into utterances.

Manifest schema
---------------
One row per audio item, long form.  The path/segment columns are deliberately
the same ones every other repo here already writes (``signal_path``,
``channel``, ``start_s``, ``end_s``), so an existing segment index — AVID's
``segments.csv``, a Mixer 7 split — becomes a manifest by adding three columns
rather than by being rewritten.

    utt_id       required   unique key for the row
    signal_path  required   path to the audio, relative to a dataset root
    group_id     required   ties together the rows describing the *same*
                            underlying utterance across conditions
    condition    required   free-form label; speech_eval never interprets it
    text         optional   reference transcript
    corpus       optional   selects the dataset root for this row
    channel      optional   0-based; -1/NaN/absent → mono-mix
    start_s/end_s optional  restrict the row to a region of its file
    …            optional   any other column is carried through to the results

``group_id`` is what makes comparison a ``groupby`` instead of a filename join.
Whether a group holds two rows or five — source, target, converted, codec round
trip, gain baseline — is the experiment's business, decided when the manifest is
written and again in :mod:`speech_eval.compare`, never here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

from audio_utils.data.dataset import AudioDataset
from audio_utils.data.metadata import resolve_paths
from audio_utils.data.transforms import (
    load_audio,
    mono_mix,
    resample,
    rms_normalize,
    select_channel,
)

from speech_eval.core import Requirements, Utterance, UtteranceBatch

REQUIRED_COLUMNS = ("utt_id", "signal_path", "group_id", "condition")


def load_manifest(
    path: str | Path,
    dataset_roots: dict[str, str | Path] | None = None,
    dataset_root: str | Path | None = None,
) -> pd.DataFrame:
    """Read a manifest CSV and resolve its paths to absolute ones.

    Paths are stored relative and joined here so that one manifest works on a
    laptop and on a cluster.  Two ways to say where the audio is:

    ``dataset_roots``
        ``{corpus: root}``; requires a ``corpus`` column and resolves per row.
        Use it whenever a manifest mixes corpora.
    ``dataset_root``
        a single root prepended to every row.

    Passing neither means the manifest already holds absolute paths.

    Raises
    ------
    ValueError
        If a required column is missing, or ``utt_id`` is not unique — a
        duplicate would make results silently overwrite each other.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Manifest {path} is missing required column(s) {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}."
        )

    df["utt_id"] = df["utt_id"].astype(str)
    duplicated = df["utt_id"][df["utt_id"].duplicated()].unique()
    if len(duplicated):
        raise ValueError(
            f"Manifest {path} has duplicate utt_id values: {sorted(duplicated)[:10]}"
            f"{' …' if len(duplicated) > 10 else ''}. utt_id must be unique."
        )

    if dataset_roots and dataset_root:
        raise ValueError("Pass dataset_roots or dataset_root, not both.")
    if dataset_roots:
        df = resolve_paths(df, dataset_roots, path_columns=["signal_path"])
    elif dataset_root:
        root = Path(dataset_root)
        df["signal_path"] = df["signal_path"].map(lambda p: str(root / p))

    return df


def select_rows(metadata: pd.DataFrame, requirements: Requirements) -> pd.DataFrame:
    """Drop the rows a metric with these requirements cannot be given.

    Currently only ``needs_text``: a metric asking for a reference transcript is
    better run on the subset that has one than handed a ``None`` it has to guess
    the meaning of.  The runner reports how many rows this removed, so a
    manifest that lost 90% of its rows to a missing ``text`` column is visible
    rather than quietly halving the sample size.
    """
    if not requirements.needs_text:
        return metadata
    if "text" not in metadata.columns:
        return metadata.iloc[:0]
    text = metadata["text"]
    return metadata[text.notna() & (text.astype(str).str.strip() != "")]


def prepare_utterance(
    utt_id: str,
    wav: Tensor,
    sample_rate: int,
    requirements: Requirements,
    text: Any = None,
    meta: dict[str, Any] | None = None,
) -> Utterance:
    """Apply a :class:`Requirements` to an in-memory waveform.

    The preparation contract in one place: resample, mono-mix, level-normalise — in that
    order, because RMS normalisation has to be measured on the signal the metric will
    actually see.

    Exists because audio does not always come from a file.  A conversion system evaluating
    its own output holds tensors it has no reason to write to disk, and reproducing these
    three steps at the call site is how such a caller ends up silently skipping the level
    normalisation that ASR and MOS metrics declare — measuring gain instead of
    degradation, with nothing raised.  :class:`UtteranceDataset` routes through here too,
    so the two paths cannot drift.
    """
    if requirements.sample_rate is not None and requirements.sample_rate != sample_rate:
        wav = resample(wav, sample_rate, requirements.sample_rate)
        sample_rate = requirements.sample_rate
    if requirements.mono:
        wav = mono_mix(wav)
    if requirements.level_norm_dbfs is not None:
        wav = rms_normalize(wav, requirements.level_norm_dbfs)
    return Utterance(
        utt_id=utt_id,
        wav=wav,
        sample_rate=sample_rate,
        text=None if text is None or pd.isna(text) else str(text),
        meta=dict(meta or {}),
    )


class UtteranceDataset(AudioDataset):
    """Yields whole :class:`Utterance` objects prepared for one requirements set.

    Subclasses ``AudioDataset`` to inherit its seek-based reader: only the
    ``[start_s, end_s]`` region is decoded, so cost tracks the segment rather
    than the recording, which is what makes evaluating segments of hour-long
    sessions affordable.  ``AudioDataset._load`` is the documented hook for
    exactly this — read and resample without normalising, then apply whatever
    amplitude handling the caller needs.

    ``chunk_s=None`` and ``train=False``: evaluation reads the whole segment,
    deterministically.  Cropping here would silently change what every metric
    measures.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        requirements: Requirements,
        carry_columns: list[str] | None = None,
    ):
        # target_sr is only consulted on the resampling path; the native-rate
        # path below bypasses the parent loader entirely, so the value passed
        # here is unused in that case.
        super().__init__(
            metadata,
            target_sr=requirements.sample_rate or 0,
            chunk_s=None,
            train=False,
            pad_to_multiple_of=None,
            normalize_sequence=False,
        )
        self.requirements = requirements
        self.utt_ids = metadata["utt_id"].astype(str).tolist()
        self.texts = (
            metadata["text"].tolist() if "text" in metadata.columns
            else [None] * len(metadata)
        )
        cols = [c for c in (carry_columns or []) if c in metadata.columns]
        self.meta_records: list[dict[str, Any]] = (
            metadata[cols].to_dict(orient="records") if cols
            else [{} for _ in range(len(metadata))]
        )

    def _load_native(self, index: int) -> tuple[Tensor, int]:
        """Read one region at the file's own sample rate.

        The parent's ``_load`` always resamples to ``target_sr``; there is no
        value of it meaning "leave the rate alone".  Rather than mutate the
        parent's state per item, this repeats its two-step read for the native
        case — the seek-based read is still ``audio_utils``', only the resample
        is dropped.
        """
        start_s, end_s = self.starts[index], self.ends[index]
        duration_s = None if end_s is None else end_s - start_s
        wav, sr = load_audio(self.paths[index], start_s, duration_s)
        return select_channel(wav, self.channels[index]), sr

    def load_utterance_audio(self, index: int) -> tuple[Tensor, int]:
        """Waveform and its sample rate, resampled or native per requirements."""
        if self.requirements.sample_rate is None:
            return self._load_native(index)
        return super()._load(index), self.requirements.sample_rate

    def __getitem__(self, index: int) -> Utterance:  # type: ignore[override]
        wav, sample_rate = self.load_utterance_audio(index)
        return prepare_utterance(
            utt_id=self.utt_ids[index],
            wav=wav,
            sample_rate=sample_rate,
            requirements=self.requirements,
            text=self.texts[index],
            meta=self.meta_records[index],
        )


def collate_utterances(items: list[Utterance]) -> UtteranceBatch:
    """Collate into an :class:`UtteranceBatch` without padding anything.

    Padding happens lazily in ``UtteranceBatch.audio``, for the metrics that ask
    for a padded tensor.  Metrics that iterate ``batch.utterances`` see the true
    signals, so no Praat window and no LPC frame ever lands on invented silence.
    """
    return UtteranceBatch(utterances=items)


class ListDataset(Dataset):
    """Trivial in-memory dataset, for tests and for callers holding tensors."""

    def __init__(self, utterances: list[Utterance]):
        self.utterances = utterances

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, index: int) -> Utterance:
        return self.utterances[index]
