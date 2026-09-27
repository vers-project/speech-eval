"""Pre-download the naturalness predictor so an offline compute node can run it.

Jean-Zay's compute nodes have no route to the model hub, so the checkpoint has to
be in the cache before the job starts.  Run this on a login node:

    uv run --extra cpu scripts/download_mos_models.py \\
        --hub-dir /path/to/torch/hub

**This one does not use ``HF_HOME``.**  UTMOS22 arrives through ``torch.hub``,
which has its own cache: the repository archive lands in ``<hub-dir>/`` and the
392 MB checkpoint in ``<hub-dir>/checkpoints/``.  Setting ``HF_HOME`` does
nothing for it, and the two other download scripts here do not cover it.  Export
``TORCH_HOME=<parent of hub-dir>`` in the launcher ``setup`` so the job finds
what this script fetched, or pass the same directory again via ``torch.hub``.

No extra dependency: the hub module needs only torch, so the ``cpu`` extra is
enough.

It then scores half a second of noise and prints the result, which is the only
way to find out that the cache is complete before a job does.  Plain argparse
rather than the launcher's ``@parse_args``: a one-shot helper with no sweep, no
SLURM and no experiment config.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

SAMPLE_RATE = 16_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hub-dir",
        default=None,
        help="torch.hub cache directory to populate (default: torch's own, "
             "normally ~/.cache/torch/hub)",
    )
    parser.add_argument(
        "--hub-repo",
        default="tarepan/SpeechMOS:v1.2.0",
        help="hub specifier, pinned to a tag (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.hub_dir:
        torch.hub.set_dir(args.hub_dir)
    print(f"torch.hub cache: {torch.hub.get_dir()}")

    # The backend directly, not through the metric wrapper: this script's job is
    # to prove the *cache* is complete, and it should keep doing that even if the
    # wrapper's API moves.  Same reason the speaker and ASR scripts here call
    # their toolkits directly.
    print(f"fetching {args.hub_repo} ...")
    model = torch.hub.load(args.hub_repo, "utmos22_strong", trust_repo=True)
    model.eval()

    rng = np.random.default_rng(0)
    wav = torch.from_numpy(
        (0.05 * rng.standard_normal(SAMPLE_RATE // 2)).astype("float32")
    ).unsqueeze(0)
    with torch.inference_mode():
        score = float(model(wav, SAMPLE_RATE).reshape(-1)[0])

    print(f"mos_utmos scored half a second of noise at {score:.3f}")
    print(
        "Cache is complete. Remember to point TORCH_HOME at the parent of the "
        "hub directory in the job's setup, not HF_HOME."
    )


if __name__ == "__main__":
    main()
