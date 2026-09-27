"""Runs metrics over a manifest and writes the per-utterance results.

What the runner is responsible for
----------------------------------
Preparing audio once per *distinct* :class:`Requirements` rather than once per
metric, loading each model once, batching, resuming, and splitting the output
into a table of scalars and an archive of arrays.  Metrics stay free of all of
it.

Outputs (in ``output_dir``)
---------------------------
``metrics.csv``
    One row per utterance: the manifest's identifying columns, the carried
    metadata columns, and one column per emitted scalar, named
    ``<metric name>.<key>``.  Long form on purpose — every aggregation this
    suite needs is a ``groupby`` over it, so aggregation choices never have to
    be made before the expensive stage runs.
``artifacts.npz``
    Arrays too large for a cell: speaker embeddings, F0 contours, spectra.
    Keyed ``<utt_id>|<metric name>.<key>``.

Resume
------
A metric's results for an utterance are skipped when a previous run already
wrote a non-null value under that metric's prefix.  Long GPU runs over big
corpora do get interrupted, and re-transcribing 100 h of audio to recover from a
walltime kill is not a reasonable cost.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from speech_eval.core import Features, Metric, Requirements
from speech_eval.io import UtteranceDataset, collate_utterances, select_rows

METRICS_CSV = "metrics.csv"
ARTIFACTS_NPZ = "artifacts.npz"

# Manifest columns that identify a row.  Always carried into the results, on top
# of whatever `carry_columns` asks for, because a results table without them
# cannot be grouped, paired or joined back to the manifest.
ID_COLUMNS = ["utt_id", "group_id", "condition"]


def group_by_requirements(metrics: list[Metric]) -> dict[Requirements, list[Metric]]:
    """Bucket metrics by the audio they need, preserving declaration order.

    ``meta_columns`` is cleared from the key: it names manifest columns, not a
    property of the waveform, so two metrics that differ only in it still share
    one preparation pass.  The group's union of them is supplied to the dataset
    instead, and handing a metric a metadata column it does not read costs
    nothing.
    """
    groups: dict[Requirements, list[Metric]] = defaultdict(list)
    for metric in metrics:
        groups[replace(metric.requirements, meta_columns=())].append(metric)
    return dict(groups)


def _meta_columns(group: list[Metric], carry_columns: list[str] | None) -> list[str]:
    """Manifest columns to place in ``Utterance.meta`` for one requirements group."""
    columns = list(carry_columns or [])
    for metric in group:
        columns.extend(metric.requirements.meta_columns)
    return list(dict.fromkeys(columns))


def _split_features(
    utt_id: str, name: str, features: Features
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Separate a metric's output into CSV cells and archive entries."""
    scalars: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for key, value in features.items():
        column = f"{name}.{key}"
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray) and value.ndim > 0:
            arrays[f"{utt_id}|{column}"] = value
        else:
            scalars[column] = value.item() if isinstance(value, np.ndarray) else value
    return scalars, arrays


def _already_done(previous: pd.DataFrame | None, name: str) -> set[str]:
    """utt_ids that already carry a non-null value for this metric."""
    if previous is None:
        return set()
    columns = [c for c in previous.columns if c.startswith(f"{name}.")]
    if not columns:
        return set()
    done = previous.loc[previous[columns].notna().any(axis=1), "utt_id"]
    return set(done.astype(str))


def run_metrics(
    metrics: list[Metric],
    metadata: pd.DataFrame,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 8,
    num_workers: int = 0,
    carry_columns: list[str] | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Compute every metric over every manifest row and write the results.

    Returns the merged results table, which is also written to
    ``output_dir/metrics.csv``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    previous = None
    csv_path = output_dir / METRICS_CSV
    if resume and csv_path.exists():
        previous = pd.read_csv(csv_path)
        previous["utt_id"] = previous["utt_id"].astype(str)

    # utt_id → {column: value}, filled in across all metrics then written once.
    # Metric columns only: identifying and carried columns are re-read from the
    # manifest at write time, so a resumed run cannot end up with two sources of
    # truth for them.  Metric columns are exactly those carrying the
    # ``<name>.<key>`` prefix, which no manifest column has.
    scalar_records: dict[str, dict[str, Any]] = defaultdict(dict)
    array_records: dict[str, np.ndarray] = {}
    if previous is not None:
        for record in previous.to_dict(orient="records"):
            scalar_records[str(record["utt_id"])].update({
                k: v for k, v in record.items() if "." in k and pd.notna(v)
            })
    npz_path = output_dir / ARTIFACTS_NPZ
    if resume and npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as archive:
            array_records.update({k: archive[k] for k in archive.files})

    for requirements, group in group_by_requirements(metrics).items():
        if requirements.needs_intervals:
            raise NotImplementedError(
                f"Metrics {[m.name for m in group]} ask for forced-alignment "
                "intervals, which the runner does not yet supply. Intervals "
                "arrive with the formant metric; until then, no metric may set "
                "Requirements.needs_intervals."
            )

        rows = select_rows(metadata, requirements)
        if len(rows) < len(metadata):
            print(
                f"[{'/'.join(m.name for m in group)}] {len(metadata) - len(rows)} of "
                f"{len(metadata)} rows dropped for lacking a reference transcript."
            )
        if rows.empty:
            continue

        pending = [m for m in group if not _already_done(previous, m.name).issuperset(
            set(rows["utt_id"].astype(str))
        )]
        if not pending:
            print(f"[{'/'.join(m.name for m in group)}] already complete, skipping.")
            continue

        for metric in pending:
            metric.ensure_loaded(device)

        # Each metric may be at a different point of completion, so the skip set
        # is per metric and the loader still visits every row.
        done_by_metric = {m.name: _already_done(previous, m.name) for m in pending}

        dataset = UtteranceDataset(
            rows, requirements, carry_columns=_meta_columns(group, carry_columns)
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_utterances,
        )

        for batch in loader:
            for metric in pending:
                done = done_by_metric[metric.name]
                todo = [u for u in batch.utterances if u.utt_id not in done]
                if not todo:
                    continue
                sub = batch if len(todo) == len(batch) else type(batch)(utterances=todo)
                results = metric.compute(sub)
                if len(results) != len(sub):
                    raise RuntimeError(
                        f"Metric {metric.name!r} returned {len(results)} results "
                        f"for {len(sub)} utterances; it must return one per "
                        "utterance, in batch order."
                    )
                for utterance, features in zip(sub.utterances, results):
                    scalars, arrays = _split_features(
                        utterance.utt_id, metric.name, features
                    )
                    scalar_records[utterance.utt_id].update(scalars)
                    array_records.update(arrays)

        # Flush after each requirements group: an interrupted run then resumes
        # from the last completed group instead of from nothing.
        _write(output_dir, metadata, scalar_records, array_records, carry_columns)

    return _write(output_dir, metadata, scalar_records, array_records, carry_columns)


def _write(
    output_dir: Path,
    metadata: pd.DataFrame,
    scalar_records: dict[str, dict[str, Any]],
    array_records: dict[str, np.ndarray],
    carry_columns: list[str] | None,
) -> pd.DataFrame:
    """Merge results onto the manifest's identifying columns and write to disk."""
    carried = [
        c for c in (carry_columns or [])
        if c in metadata.columns and c not in ID_COLUMNS
    ]
    base = metadata[ID_COLUMNS + carried].copy()
    base["utt_id"] = base["utt_id"].astype(str)

    results = pd.DataFrame.from_dict(scalar_records, orient="index")
    results.index.name = "utt_id"
    results = results.reset_index()

    merged = base.merge(results, on="utt_id", how="left")
    metric_columns = sorted(c for c in merged.columns if c not in base.columns)
    merged = merged[list(base.columns) + metric_columns]

    merged.to_csv(output_dir / METRICS_CSV, index=False)
    if array_records:
        np.savez_compressed(output_dir / ARTIFACTS_NPZ, **array_records)
    return merged


def load_artifacts(output_dir: str | Path) -> dict[str, np.ndarray]:
    """Read ``artifacts.npz`` back as ``{"<utt_id>|<column>": array}``."""
    path = Path(output_dir) / ARTIFACTS_NPZ
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
