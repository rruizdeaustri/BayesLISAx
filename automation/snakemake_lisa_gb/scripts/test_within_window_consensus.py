#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from within_window_consensus import main


def make_bundle(path: Path, seed: int, upper: float) -> None:
    rng = np.random.default_rng(seed)
    n = 600
    dim_per = 7
    k = 4
    samples = np.zeros((n, k, dim_per), dtype=float)
    centers = np.array([1.83450e-3, 1.84086e-3, upper, upper + 1.4e-6])
    for comp, center in enumerate(centers):
        samples[:, comp, 0] = rng.normal(center, 1.2e-8, size=n)
        samples[:, comp, 1] = rng.normal(0.0, 1e-15, size=n)
        samples[:, comp, 6] = 1.0
    np.savez_compressed(
        path,
        samples=samples.reshape(n, k * dim_per),
        weights=np.empty(0),
        seed=np.asarray(str(seed)),
        diagnostics_json=np.asarray("{}"),
    )


def run_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs = []
        for seed, upper in [(0, 1.8420e-3), (11, 1.8438e-3), (22, 1.8449e-3)]:
            path = root / f"seed{seed}.npz"
            make_bundle(path, seed, upper)
            inputs.append(str(path))

        main(
            [
                "--inputs",
                *inputs,
                "--window-id",
                "synthetic",
                "--tobs",
                "31457280",
                "--cluster-eps-bins",
                "5",
                "--min-mean-inclusion",
                "0.05",
                "--out-clusters",
                str(root / "clusters.csv"),
                "--out-cooccurrence",
                str(root / "cooccurrence.csv"),
                "--out-modes",
                str(root / "modes.json"),
                "--out-summary",
                str(root / "summary.json"),
            ]
        )
        summary = json.loads((root / "summary.json").read_text())
        assert summary["n_input_seeds"] == 3
        assert summary["n_consensus_clusters"] >= 4
        assert summary["quality_counts"].get("robust", 0) >= 2
        assert summary["recommended_action"] == "subtract_or_condition_and_rescan"
        print("within-window consensus synthetic test: PASS")


if __name__ == "__main__":
    run_test()
