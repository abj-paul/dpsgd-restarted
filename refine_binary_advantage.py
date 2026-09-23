"""Derive the binary C=0.25 versus C=1 advantage dataset.

The source experiment contains baseline-relative labels for C=0.25 and C=0.75.
This derivation keeps only C=0.25 and makes the baseline C=1 action explicit.
It emits one row per anchor rather than duplicating the state for each action.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np


SIGN_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path("runs/counterfactual_advantage_40800_v1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/counterfactual_advantage_binary_20400_v1"),
    )
    parser.add_argument(
        "--output-remote",
        default=(
            "rclone_s3:dpsgd-as-a-optimization-problem/current-work/"
            "adaptive-clipping-wrn/counterfactual_advantage_binary_20400/"
            "v1_20260923"
        ),
    )
    parser.add_argument("--c-low", type=float, default=0.25)
    parser.add_argument("--c-high", type=float, default=1.0)
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
    ).strip()


def preferred_action(advantage: float) -> str:
    if advantage > SIGN_TOLERANCE:
        return "low"
    if advantage < -SIGN_TOLERANCE:
        return "high"
    return "neutral"


def describe(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def group_statistics(connection: sqlite3.Connection, group_column: str) -> list[dict]:
    if group_column not in {"run_id", "dataset_split"}:
        raise ValueError("invalid grouping column")
    rows = connection.execute(
        f"""
        SELECT {group_column}, COUNT(*), AVG(advantage_low_vs_high),
               AVG(CASE WHEN preferred_action='low' THEN 1.0 ELSE 0.0 END),
               AVG(gradient_norm), AVG(baseline_high_probe_ce)
        FROM binary_advantages GROUP BY {group_column} ORDER BY {group_column}
        """
    ).fetchall()
    return [
        {
            group_column: group,
            "rows": int(count),
            "mean_advantage": float(mean),
            "low_preferred_fraction": float(low_fraction),
            "mean_gradient_norm": float(norm),
            "mean_baseline_probe_ce": float(loss),
        }
        for group, count, mean, low_fraction, norm, loss in rows
    ]


def create_database(source: sqlite3.Connection, destination_path: Path) -> tuple[sqlite3.Connection, float]:
    source_rows = source.execute(
        """
        SELECT run_id, checkpoint_step, batch_replicate, anchor_slot,
               example_id, baseline_probe_ce, candidate_probe_ce,
               advantage_raw, gradient_norm, baseline_factor,
               candidate_factor, parameter_delta_norm, logical_batch_size,
               batch_seed, augmentation_seed, noise_seed, checkpoint_uri,
               history_remote, history_end_step
        FROM advantages
        WHERE action_name='low' AND ABS(candidate_clip-0.25)<1e-15
        ORDER BY run_id, checkpoint_step, batch_replicate, anchor_slot
        """
    ).fetchall()
    if len(source_rows) != 20_400:
        raise RuntimeError(f"expected 20,400 low-action rows, found {len(source_rows):,}")
    advantages = np.asarray([row[7] for row in source_rows], dtype=np.float64)
    scale = float(np.median(np.abs(advantages)))

    destination = sqlite3.connect(destination_path)
    destination.execute("PRAGMA journal_mode=WAL")
    destination.execute("PRAGMA synchronous=FULL")
    destination.execute(
        """
        CREATE TABLE binary_advantages (
            run_id INTEGER NOT NULL,
            checkpoint_step INTEGER NOT NULL,
            batch_replicate INTEGER NOT NULL,
            anchor_slot INTEGER NOT NULL,
            example_id INTEGER NOT NULL,
            dataset_split TEXT NOT NULL,
            c_low REAL NOT NULL CHECK(c_low=0.25),
            c_high REAL NOT NULL CHECK(c_high=1.0),
            baseline_high_probe_ce REAL NOT NULL,
            candidate_low_probe_ce REAL NOT NULL,
            reward_high REAL NOT NULL CHECK(reward_high=0.0),
            reward_low REAL NOT NULL,
            advantage_low_vs_high REAL NOT NULL,
            advantage_scaled REAL NOT NULL,
            absolute_advantage REAL NOT NULL,
            preferred_action TEXT NOT NULL
                CHECK(preferred_action IN ('low','high','neutral')),
            gradient_norm REAL NOT NULL,
            high_clipping_factor REAL NOT NULL,
            low_clipping_factor REAL NOT NULL,
            parameter_delta_norm REAL NOT NULL,
            logical_batch_size INTEGER NOT NULL,
            batch_seed INTEGER NOT NULL,
            augmentation_seed INTEGER NOT NULL,
            noise_seed INTEGER NOT NULL,
            checkpoint_uri TEXT NOT NULL,
            history_remote TEXT NOT NULL,
            history_end_step INTEGER NOT NULL,
            PRIMARY KEY (
                run_id, checkpoint_step, batch_replicate, anchor_slot
            )
        )
        """
    )
    destination.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    insert = """
        INSERT INTO binary_advantages VALUES(
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
    """
    with destination:
        for row in source_rows:
            (
                run_id, step, replicate, anchor_slot, example_id,
                baseline_loss, candidate_loss, advantage, norm,
                high_factor, low_factor, delta_norm, batch_size,
                batch_seed, augmentation_seed, noise_seed, checkpoint_uri,
                history_remote, history_end_step,
            ) = row
            split = "train" if run_id <= 7 else ("validation" if run_id == 8 else "test")
            destination.execute(
                insert,
                (
                    run_id, step, replicate, anchor_slot, example_id, split,
                    0.25, 1.0, baseline_loss, candidate_loss,
                    0.0, advantage, advantage, advantage / (scale + 1e-12),
                    abs(advantage), preferred_action(advantage), norm,
                    high_factor, low_factor, delta_norm, batch_size,
                    batch_seed, augmentation_seed, noise_seed, checkpoint_uri,
                    history_remote, history_end_step,
                ),
            )
        for key, value in {
            "c_low": "0.25",
            "c_high": "1.0",
            "sign_tolerance": repr(SIGN_TOLERANCE),
            "advantage_median_abs": repr(scale),
            "advantage_multiplier": repr(1.0 / (scale + 1e-12)),
            "train_runs": "0-7",
            "validation_run": "8",
            "test_run": "9",
        }.items():
            destination.execute("INSERT INTO metadata VALUES(?,?)", (key, value))
    destination.execute("CREATE INDEX idx_binary_split ON binary_advantages(dataset_split)")
    destination.execute("CREATE INDEX idx_binary_example ON binary_advantages(example_id)")
    destination.execute("CREATE INDEX idx_binary_preference ON binary_advantages(preferred_action)")
    destination.commit()
    return destination, scale


def build_summary(connection: sqlite3.Connection, scale: float) -> dict:
    rows = connection.execute(
        "SELECT advantage_low_vs_high, gradient_norm, example_id FROM binary_advantages"
    ).fetchall()
    advantages = np.asarray([row[0] for row in rows], dtype=np.float64)
    norms = np.asarray([row[1] for row in rows], dtype=np.float64)
    example_ids = np.asarray([row[2] for row in rows], dtype=np.int64)
    counts = dict(connection.execute(
        "SELECT preferred_action,COUNT(*) FROM binary_advantages GROUP BY preferred_action"
    ).fetchall())
    stage_rows = connection.execute(
        """
        SELECT CAST((checkpoint_step-1)/50 AS INTEGER), MIN(checkpoint_step),
               MAX(checkpoint_step), COUNT(*), AVG(advantage_low_vs_high),
               AVG(CASE WHEN preferred_action='low' THEN 1.0 ELSE 0.0 END),
               AVG(gradient_norm), AVG(baseline_high_probe_ce)
        FROM binary_advantages GROUP BY CAST((checkpoint_step-1)/50 AS INTEGER)
        ORDER BY 1
        """
    ).fetchall()
    norm_regimes = []
    for lower, upper, name in (
        (-math.inf, 0.25, "norm_le_c_low"),
        (0.25, 1.0, "c_low_lt_norm_le_c_high"),
        (1.0, math.inf, "norm_gt_c_high"),
    ):
        mask = (norms > lower) & (norms <= upper)
        norm_regimes.append({
            "regime": name,
            "rows": int(mask.sum()),
            "fraction": float(mask.mean()),
            "mean_advantage": float(advantages[mask].mean()) if mask.any() else None,
        })
    unique, observed_counts = np.unique(example_ids, return_counts=True)
    run_means = np.asarray([
        value for (value,) in connection.execute(
            "SELECT AVG(advantage_low_vs_high) FROM binary_advantages GROUP BY run_id ORDER BY run_id"
        )
    ])
    return {
        "status": "complete",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": int(len(rows)),
        "action_definition": {"low": 0.25, "high": 1.0},
        "advantage_definition": "CE(theta_next_high) - CE(theta_next_low)",
        "advantage": describe(advantages),
        "advantage_median_abs": scale,
        "advantage_multiplier": 1.0 / (scale + 1e-12),
        "preferences": {
            "low": int(counts.get("low", 0)),
            "high": int(counts.get("high", 0)),
            "neutral": int(counts.get("neutral", 0)),
        },
        "binary_oracle_mean_advantage": float(np.maximum(advantages, 0).mean()),
        "gradient_norm": describe(norms),
        "norm_regimes": norm_regimes,
        "coverage": {
            "unique_update_examples": int(unique.size),
            "unobserved_update_examples": int(4_000 - unique.size),
            "mean_occurrences_among_observed": float(observed_counts.mean()),
            "median_occurrences_among_observed": float(np.median(observed_counts)),
            "max_occurrences": int(observed_counts.max()),
        },
        "by_split": group_statistics(connection, "dataset_split"),
        "by_run": group_statistics(connection, "run_id"),
        "by_stage": [
            {
                "stage": int(stage), "step_min": int(step_min),
                "step_max": int(step_max), "rows": int(count),
                "mean_advantage": float(mean),
                "low_preferred_fraction": float(low_fraction),
                "mean_gradient_norm": float(norm),
                "mean_baseline_probe_ce": float(loss),
            }
            for stage, step_min, step_max, count, mean, low_fraction, norm, loss
            in stage_rows
        ],
        "run_mean_advantage": {
            "mean": float(run_means.mean()),
            "sample_std": float(run_means.std(ddof=1)),
            "standard_error": float(run_means.std(ddof=1) / math.sqrt(len(run_means))),
        },
    }


def export_jsonl(connection: sqlite3.Connection, path: Path) -> None:
    cursor = connection.execute(
        "SELECT * FROM binary_advantages "
        "ORDER BY run_id,checkpoint_step,batch_replicate,anchor_slot"
    )
    names = [description[0] for description in cursor.description]
    with gzip.open(path, "wt", encoding="utf-8") as destination:
        for values in cursor:
            destination.write(json.dumps(dict(zip(names, values, strict=True))) + "\n")


def main() -> None:
    args = parse_args()
    if args.c_low != 0.25 or args.c_high != 1.0:
        raise ValueError("the registered binary derivation is C_low=0.25, C_high=1.0")
    if not args.output_remote.startswith("rclone_s3:"):
        raise ValueError("output remote must use rclone_s3")
    source_database = args.source_dir / "advantages.sqlite3"
    source_manifest_path = args.source_dir / "manifest.json"
    source_split_path = args.source_dir / "public_split.json"
    for path in (source_database, source_manifest_path, source_split_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads(source_manifest_path.read_text())
    with sqlite3.connect(source_database) as source:
        source.row_factory = sqlite3.Row
        source_count = source.execute("SELECT COUNT(*) FROM advantages").fetchone()[0]
        if source_count != 40_800:
            raise RuntimeError(f"source row count is {source_count}, expected 40,800")
        destination, scale = create_database(
            source, args.output_dir / "binary_advantages.sqlite3"
        )
    summary = build_summary(destination, scale)
    integrity = destination.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"destination integrity check failed: {integrity}")
    export_jsonl(destination, args.output_dir / "binary_advantages.jsonl.gz")
    destination.close()

    shutil.copy2(source_split_path, args.output_dir / "public_split.json")
    manifest = {
        "schema_version": 1,
        "description": "binary C_low=0.25 versus C_high=1.0 clipping advantage dataset",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "source": {
            "directory": str(args.source_dir),
            "manifest_sha256": source_manifest["manifest_sha256"],
            "database_sha256": sha256_file(source_database),
            "rows": 40_800,
        },
        "rows": 20_400,
        "c_low": 0.25,
        "c_high": 1.0,
        "advantage_definition": "CE(theta_next_high) - CE(theta_next_low)",
        "preference_definition": {
            "low": f"advantage > {SIGN_TOLERANCE}",
            "high": f"advantage < -{SIGN_TOLERANCE}",
            "neutral": f"abs(advantage) <= {SIGN_TOLERANCE}",
        },
        "splits": {"train": "runs 0-7", "validation": "run 8", "test": "run 9"},
        "output_remote": args.output_remote.rstrip("/"),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "summary.json", summary)

    if not args.no_upload:
        subprocess.run(
            ["rclone", "copy", str(args.output_dir), args.output_remote.rstrip("/"),
             "--transfers", "4", "--checkers", "8"],
            check=True,
        )
        subprocess.run(
            ["rclone", "check", str(args.output_dir), args.output_remote.rstrip("/"),
             "--size-only"],
            check=True,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
