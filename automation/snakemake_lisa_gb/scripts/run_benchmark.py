#!/usr/bin/env python3

import argparse
import ast
import json
import os
import re
import subprocess
import time
from pathlib import Path


NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def find_num(pattern, text):
    matches = re.findall(
        pattern,
        text,
        re.MULTILINE,
    )

    return (
        float(matches[-1])
        if matches
        else None
    )


def find_list(pattern, text):
    matches = re.findall(
        pattern,
        text,
        re.MULTILINE,
    )

    if not matches:
        return None

    try:
        return ast.literal_eval(
            matches[-1]
        )

    except Exception:
        return None


def parse_numeric_array(expr):
    expr = expr.strip()

    try:
        return ast.literal_eval(
            expr
        )

    except Exception:
        nums = re.findall(
            NUM_RE,
            expr,
        )

        if not nums:
            return None

        return [
            float(x)
            for x in nums
        ]


def parse_ns_f0_arrays(text):
    """
    Parse f0 arrays from NS summary lines.

    Supports numpy-style arrays with spaces,
    brackets, and scientific notation.
    """

    f0_best = None
    f0_mean = None
    f0_std = None

    patterns = {
        "f0_best": (
            r"\[NS\]\s+best\s+f0\s*=\s*(\[.*\])"
        ),
        "f0_mean": (
            r"\[NS summary\]\s+"
            r"f0\s+mean\s+per\s+source\s*=\s*(\[.*\])"
        ),
        "f0_std": (
            r"\[NS summary\]\s+"
            r"f0\s+std\s+per\s+source\s*=\s*(\[.*\])"
        ),
    }

    for line in text.splitlines():
        for key, pattern in patterns.items():
            match = re.search(
                pattern,
                line,
            )

            if not match:
                continue

            array = parse_numeric_array(
                match.group(1)
            )

            if key == "f0_best":
                f0_best = array

            elif key == "f0_mean":
                f0_mean = array

            else:
                f0_std = array

    return (
        f0_best,
        f0_mean,
        f0_std,
    )


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--python",
        required=True,
    )

    parser.add_argument(
        "--repo-root",
        required=True,
    )

    parser.add_argument(
        "--problem-factory",
        required=True,
    )

    parser.add_argument(
        "--config-json",
        required=True,
    )

    parser.add_argument(
        "--algo",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--n-live",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--tol",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--num-inner-steps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--num-delete-ratio",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--ggns-step-size",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--ggns-num-inner-steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--initial-num-steps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--refinement-num-steps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--ess-min",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--bad-logz-min",
        type=float,
        default=-1e99,
    )

    parser.add_argument(
        "--local-mode-f0-sigma-max",
        type=float,
        default=1e-7,
    )

    parser.add_argument(
        "--skip-plots",
        action="store_true",
    )

    parser.add_argument(
        "--benchmark-preset",
        default="custom",
    )

    parser.add_argument(
        "--out-json",
        required=True,
    )

    return parser


def main():
    args = (
        build_parser()
        .parse_args()
    )

    cmd = [
        args.python,
        "-m",
        "jax_samplers.cli",
        "--algo",
        args.algo,
        "--problem",
        "factory",
        "--problem-factory",
        args.problem_factory,
        "--n-live",
        str(args.n_live),
        "--tol",
        str(args.tol),
        "--num-inner-steps",
        str(args.num_inner_steps),
        "--num-delete-ratio",
        str(args.num_delete_ratio),
        "--seed",
        str(args.seed),
        "--initial-num-steps",
        str(args.initial_num_steps),
        "--refinement-num-steps",
        str(args.refinement_num_steps),
        "--max-batches",
        str(args.max_batches),
    ]

    if args.ggns_step_size is not None:
        cmd.extend(
            [
                "--ggns-step-size",
                str(args.ggns_step_size),
            ]
        )

    if args.ggns_num_inner_steps is not None:
        cmd.extend(
            [
                "--ggns-num-inner-steps",
                str(
                    args.ggns_num_inner_steps
                ),
            ]
        )

    if args.skip_plots:
        cmd.append(
            "--skip-plots"
        )

    # ---------------------------------------------------------
    # Output paths
    # ---------------------------------------------------------

    out_path = Path(
        args.out_json
    ).resolve()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_log_path = (
        out_path.parent
        / "run.log"
    )

    posterior_path = (
        out_path.parent
        / "posterior.npz"
    )

    # ---------------------------------------------------------
    # Environment for jax_samplers
    # ---------------------------------------------------------

    env = os.environ.copy()

    env[
        "JAX_SAMPLERS_CONFIG"
    ] = str(
        Path(
            args.config_json
        ).resolve()
    )

    env[
        "PYTHONPATH"
    ] = (
        f"{args.repo_root}/src:"
        + env.get(
            "PYTHONPATH",
            "",
        )
    )

    # Ask SamplerResult to persist the physical posterior.
    #
    # The produced NPZ is compatible with
    # scripts/within_window_consensus.py.
    env[
        "JAX_SAMPLERS_POSTERIOR_PATH"
    ] = str(
        posterior_path
    )

    # ---------------------------------------------------------
    # Execute sampler
    # ---------------------------------------------------------

    start_time = time.time()

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    if completed.returncode == 0:
        ensure_posterior_metadata(
            posterior_path,
            seed=args.seed,
            algo=args.algo,
            config_path=args.config_json,
        )
    
    runtime = (
        time.time()
        - start_time
    )

    text = (
        (completed.stdout or "")
        + "\n"
        + (completed.stderr or "")
    )

    run_log_path.write_text(
        text,
        encoding="utf-8",
        errors="ignore",
    )

    # ---------------------------------------------------------
    # Parse sampler diagnostics
    # ---------------------------------------------------------

    diagnostics = find_list(
        r"^\[post\] diagnostics\s*=\s*(\{.*\})$",
        text,
    )

    logz = find_num(
        r"logZ\s*=\s*([-+0-9.eE]+)",
        text,
    )

    logz_std = find_num(
        r"logZ\s*=\s*"
        r"[-+0-9.eE]+"
        r"±([-+0-9.eE]+)",
        text,
    )

    ess = find_num(
        r"ESS\s*=\s*([-+0-9.eE]+)",
        text,
    )

    best_logl = find_num(
        r"best logL\s*=\s*([-+0-9.eE]+)",
        text,
    )

    if (
        diagnostics
        and isinstance(
            diagnostics,
            dict,
        )
    ):
        logz = diagnostics.get(
            "logZ",
            logz,
        )

        logz_std = diagnostics.get(
            "logZ_std",
            logz_std,
        )

        ess = diagnostics.get(
            "ESS",
            ess,
        )

        best_logl = diagnostics.get(
            "best_logL",
            best_logl,
        )

    # ---------------------------------------------------------
    # Parse source-frequency diagnostics
    # ---------------------------------------------------------

    (
        f0_best,
        f0_mean,
        f0_std,
    ) = parse_ns_f0_arrays(
        text
    )

    if f0_mean is None:
        f0_mean = find_list(
            r"f0_mean['\"]?\s*[:=]\s*"
            r"(\[[^\n]*\])",
            text,
        )

    if f0_std is None:
        f0_std = find_list(
            r"f0_std['\"]?\s*[:=]\s*"
            r"(\[[^\n]*\])",
            text,
        )

    # ---------------------------------------------------------
    # Parse posterior shapes
    # ---------------------------------------------------------

    samples_shape = find_list(
        r"^\[post\] samples\.shape\s*=\s*(\(.*\))$",
        text,
    )

    weights_shape = find_list(
        r"^\[post\] weights\.shape\s*=\s*(\(.*\))$",
        text,
    )

    pk_vals = find_list(
        r"pK_vals['\"]?\s*[:=]\s*"
        r"(\[[^\]]*\])",
        text,
    )

    pk_probs = find_list(
        r"pK_probs['\"]?\s*[:=]\s*"
        r"(\[[^\]]*\])",
        text,
    )

    if (
        diagnostics
        and isinstance(
            diagnostics,
            dict,
        )
    ):
        pk_vals = diagnostics.get(
            "pK_vals",
            pk_vals,
        )

        pk_probs = diagnostics.get(
            "pK_probs",
            pk_probs,
        )

    # ---------------------------------------------------------
    # Determine run status
    # ---------------------------------------------------------

    status = (
        "pass"
        if completed.returncode == 0
        else "crash"
    )

    labels = [
        status
    ]

    if completed.returncode == 0:

        if (
            ess is not None
            and ess < args.ess_min
        ):
            labels.append(
                "low_ESS"
            )

        if (
            logz is None
            or logz
            <= args.bad_logz_min
        ):
            labels.append(
                "bad_logZ"
            )

        if f0_std is not None:
            f0_std_values = [
                float(x)
                for x in re.findall(
                    NUM_RE,
                    str(f0_std),
                )
            ]

            if (
                f0_std_values
                and max(
                    f0_std_values
                )
                > args.local_mode_f0_sigma_max
            ):
                labels.append(
                    "local_mode_suspected"
                )

    if (
        str(
            args.benchmark_preset
        ).lower()
        == "smoke"
    ):
        labels.append(
            "workflow_smoke_only"
        )

    if (
        args.algo
        in {
            "ggns",
            "dynamic_ggns",
        }
        and status == "crash"
    ):
        labels.append(
            "experimental"
        )

    # ---------------------------------------------------------
    # Posterior persistence status
    # ---------------------------------------------------------

    posterior_exists = (
        posterior_path.exists()
    )

    posterior_size_bytes = (
        posterior_path.stat().st_size
        if posterior_exists
        else None
    )

    if (
        completed.returncode == 0
        and not posterior_exists
    ):
        labels.append(
            "posterior_not_saved"
        )

    # ---------------------------------------------------------
    # Summary JSON
    # ---------------------------------------------------------

    output = {
        "algo": args.algo,
        "seed": args.seed,
        "n_live": args.n_live,
        "tol": args.tol,
        "num_inner_steps": (
            args.num_inner_steps
        ),
        "num_delete_ratio": (
            args.num_delete_ratio
        ),
        "runtime_seconds": runtime,
        "return_code": (
            completed.returncode
        ),
        "status": status,
        "status_labels": labels,
        "benchmark_preset": (
            args.benchmark_preset
        ),
        "log_path": str(
            run_log_path
        ),
        "posterior_path": str(
            posterior_path
        ),
        "posterior_exists": (
            posterior_exists
        ),
        "posterior_size_bytes": (
            posterior_size_bytes
        ),
        "logZ": logz,
        "logZ_std": logz_std,
        "ESS": ess,
        "best_logL": best_logl,
        "pK_vals": pk_vals,
        "pK_probs": pk_probs,
        "f0_best": f0_best,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "samples_shape": (
            samples_shape
        ),
        "weights_shape": (
            weights_shape
        ),
        "ggns_step_size": (
            args.ggns_step_size
        ),
        "ggns_num_inner_steps": (
            args.ggns_num_inner_steps
        ),
        "initial_num_steps": (
            args.initial_num_steps
        ),
        "refinement_num_steps": (
            args.refinement_num_steps
        ),
        "max_batches": (
            args.max_batches
        ),
    }

    out_path.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        completed.returncode
    )

def ensure_posterior_metadata(
    posterior_path: Path,
    *,
    seed: int,
    algo: str,
    config_path: str,
) -> None:
    """
    Ensure posterior.npz contains correct identifying metadata.

    SamplerResult writes the posterior arrays, but older/current
    implementations may leave the scalar seed field empty.
    """
    if not posterior_path.exists():
        return

    import numpy as np

    with np.load(
        posterior_path,
        allow_pickle=False,
    ) as npz:
        payload = {
            key: npz[key]
            for key in npz.files
        }

    payload["seed"] = np.asarray(
        str(seed)
    )

    payload["algo"] = np.asarray(
        str(algo)
    )

    payload["config_path"] = np.asarray(
        str(
            Path(config_path).resolve()
        )
    )

    np.savez_compressed(
        posterior_path,
        **payload,
    )

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
