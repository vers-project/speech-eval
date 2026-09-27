"""Run a set of metrics over a manifest.

Usage
-----
    uv run --extra cpu scripts/run_eval.py \\
        --config     configs/run_eval/local.yaml \\
        --output-dir outputs/eval/avid_smoke

The config names the manifest, the dataset roots and the metrics; everything
about *which* utterances get compared to which is deliberately absent, because
that decision belongs to the manifest that produced the rows and to
``speech_eval.compare`` reading the results afterwards.
"""
from __future__ import annotations

from pathlib import Path

import torch

from experiment_launcher import parse_args

from speech_eval import metrics as _metrics  # noqa: F401  (registers built-ins)
from speech_eval.core import available_metrics, build_metrics
from speech_eval.io import load_manifest
from speech_eval.runner import run_metrics


def resolve_device(requested: str | None) -> torch.device:
    """Pick the device, falling back to CPU with a warning rather than crashing.

    A metric suite is often re-run on a laptop to inspect a result; failing hard
    on a config that says ``cuda`` would make that awkward for no benefit.
    """
    if requested in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[run_eval] {requested} requested but no CUDA available — using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


@parse_args
def main(config: dict, output_dir: Path) -> None:
    data_cfg = config.get("data", {})
    runner_cfg = config.get("runner", {})

    metric_specs = config.get("metrics")
    if not metric_specs:
        raise ValueError(
            f"Config has no 'metrics' list. Available types: {available_metrics()}"
        )
    metric_list = build_metrics(metric_specs)

    metadata = load_manifest(
        data_cfg["manifest"],
        dataset_roots=data_cfg.get("dataset_roots"),
        dataset_root=data_cfg.get("dataset_root"),
    )
    if data_cfg.get("limit"):
        metadata = metadata.head(int(data_cfg["limit"]))

    device = resolve_device(config.get("device"))
    print(
        f"[run_eval] {len(metadata)} utterances × "
        f"{len(metric_list)} metrics ({', '.join(m.name for m in metric_list)}) "
        f"on {device}"
    )

    results = run_metrics(
        metric_list,
        metadata,
        output_dir=output_dir,
        device=device,
        batch_size=int(runner_cfg.get("batch_size", 8)),
        num_workers=int(runner_cfg.get("num_workers", 0)),
        carry_columns=data_cfg.get("carry_columns"),
        resume=bool(runner_cfg.get("resume", True)),
    )
    print(f"[run_eval] wrote {len(results)} rows to {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
