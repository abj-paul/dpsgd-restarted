"""Launch ten sequential, independently seeded WRN-16-1 collection runs.

Requires an explicitly approved private rclone destination. This script never
starts a partially completed run over: exact DataLoader-worker resume is not
implemented, so a partial run must be investigated before retrying.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/collection_10"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--ntfy-topic", default="")
    parser.add_argument("--projection-seed", type=int, default=4275)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--collection-batch-size", type=int, default=512)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.runs <= 10:
        raise ValueError("runs must be between 1 and 10")
    if not args.remote_base.startswith("rclone_s3:"):
        raise ValueError("remote base must be an explicitly approved rclone_s3 path")

    first_projection = args.output_root / "run_00" / "collection" / "projection.pt"
    for run in range(args.runs):
        output_dir = args.output_root / f"run_{run:02d}"
        summary = output_dir / "summary.json"
        if summary.exists():
            print(f"run {run:02d} already complete; skipping", flush=True)
            continue
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(f"partial run at {output_dir}; refusing to overwrite")
        if run:
            if not first_projection.exists():
                raise RuntimeError(f"shared projection missing: {first_projection}")
            (output_dir / "collection").mkdir(parents=True, exist_ok=True)
            os.link(first_projection, output_dir / "collection" / "projection.pt")

        command = [
            sys.executable, "dp_train.py",
            "--data-dir", str(args.data_dir),
            "--output-dir", str(output_dir),
            "--steps", "510",
            "--logical-batch-size", "16667",
            "--physical-batch-size", "2048",
            "--noise-multiplier", "4.888246618",
            "--delta", "1e-5",
            "--depth", "16", "--width", "1", "--groups", "16",
            "--seed", str(args.seed_start + run),
            "--collect-gradients",
            "--collection-batch-size", str(args.collection_batch_size),
            "--collection-remote", f"{args.remote_base.rstrip('/')}/run_{run:02d}",
            "--projection-seed", str(args.projection_seed),
            "--ntfy-topic", args.ntfy_topic,
        ]
        print(f"starting run {run + 1}/{args.runs}: seed={args.seed_start + run}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
