#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class RunResult:
    k: int
    seed: int
    logz: float
    logz_err: float
    summary_path: str
    fidelity: str


@dataclass
class EvidencePoint:
    k: int
    seeds: list[int]
    logz_values: list[float]
    logz_err_values: list[float]
    robust_logz: float
    robust_method: str
    agreeing_seeds: list[int]
    pair_spread: float
    median_logz: float
    mean_logz: float
    std_logz: float
    min_logz: float
    max_logz: float
    n_outliers: int
    status: str
    confidence: str


@dataclass
class ControllerDecision:
    status: str
    selected_k: int | None
    current_best_k: int | None
    next_k: int | None
    next_action: str
    reason: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Adaptive model-order controller"
    )

    p.add_argument("--window-id", required=True)
    p.add_argument("--f-min", required=True, type=float)
    p.add_argument("--f-max", required=True, type=float)

    p.add_argument("--base-json", required=True)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--workflow-root", default=".")
    p.add_argument("--python", default=sys.executable)

    p.add_argument(
        "--problem-factory",
        default=(
            "jax_samplers.problems."
            "lisa_gb_transdim_problem:make"
        ),
    )

    p.add_argument(
        "--output-root",
        default="results/model_order",
    )

    p.add_argument(
        "--reuse-root",
        action="append",
        default=[],
    )

    p.add_argument(
        "--force-runs",
        action="store_true",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    # Search/model-order settings.
    p.add_argument(
        "--initial-k-max",
        type=int,
        default=4,
    )

    p.add_argument(
        "--safety-k-max",
        type=int,
        default=10,
    )

    p.add_argument(
        "--confirm-lower-points",
        type=int,
        default=2,
    )

    p.add_argument(
        "--min-peak-drop-logz",
        type=float,
        default=20.0,
    )

    p.add_argument(
        "--pair-agreement-logz",
        type=float,
        default=150.0,
    )

    # Search fidelity.
    p.add_argument(
        "--search-seeds",
        default="0,11,22",
    )

    p.add_argument(
        "--search-n-live",
        type=int,
        default=500,
    )

    p.add_argument(
        "--search-tol",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--search-num-inner-steps",
        type=int,
        default=64,
    )

    p.add_argument(
        "--search-num-delete-ratio",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--search-initial-num-steps",
        type=int,
        default=16,
    )

    p.add_argument(
        "--search-refinement-num-steps",
        type=int,
        default=8,
    )

    p.add_argument(
        "--search-max-batches",
        type=int,
        default=8,
    )

    # Confirmation fidelity.
    p.add_argument(
        "--skip-highres-confirmation",
        action="store_true",
    )

    p.add_argument(
        "--confirmation-seeds",
        default="0,11,22",
    )

    p.add_argument(
        "--highres-n-live",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--highres-tol",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--highres-num-inner-steps",
        type=int,
        default=128,
    )

    p.add_argument(
        "--highres-num-delete-ratio",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--highres-initial-num-steps",
        type=int,
        default=32,
    )

    p.add_argument(
        "--highres-refinement-num-steps",
        type=int,
        default=16,
    )

    p.add_argument(
        "--highres-max-batches",
        type=int,
        default=12,
    )

    p.add_argument(
        "--max-confirmation-extensions",
        type=int,
        default=1,
    )

    p.add_argument(
        "--min-confirmation-rise-logz",
        type=float,
        default=20.0,
    )

    return p


def parse_seed_list(raw: str) -> list[int]:
    values = [
        int(x.strip())
        for x in raw.split(",")
        if x.strip()
    ]

    if not values:
        raise ValueError(
            f"Invalid seed list: {raw}"
        )

    if len(values) != len(set(values)):
        raise ValueError(
            f"Seed list contains duplicates: {raw}"
        )

    return values


def run_command(
    command: Sequence[str],
    dry_run: bool,
) -> None:
    print(
        "+",
        " ".join(command),
        flush=True,
    )

    if not dry_run:
        subprocess.run(
            command,
            check=True,
        )


def nested_get(
    data: dict[str, Any],
    candidates: Iterable[Sequence[str]],
) -> Any:
    for path in candidates:
        value: Any = data

        for key in path:
            if (
                not isinstance(value, dict)
                or key not in value
            ):
                break

            value = value[key]

        else:
            return value

    return None


def extract_logz(
    path: Path,
) -> tuple[float, float]:
    data = json.loads(
        path.read_text()
    )

    logz = nested_get(
        data,
        [
            ("logZ",),
            ("logz",),
            ("results", "logZ"),
            ("results", "logz"),
            ("summary", "logZ"),
            ("summary", "logz"),
        ],
    )

    err = nested_get(
        data,
        [
            ("logZ_err",),
            ("logz_err",),
            ("results", "logZ_err"),
            ("results", "logz_err"),
            ("summary", "logZ_err"),
            ("summary", "logz_err"),
        ],
    )

    if logz is None:
        raise KeyError(
            f"Could not find logZ in {path}"
        )

    logz_value = float(logz)

    if not math.isfinite(logz_value):
        raise ValueError(
            f"Non-finite logZ in {path}: "
            f"{logz_value}"
        )

    if err is None:
        err_value = math.nan
    else:
        err_value = float(err)

    return (
        logz_value,
        err_value,
    )


def infer_run_identity(
    path: Path,
) -> tuple[
    str | None,
    int | None,
    int | None,
    str,
]:
    try:
        data = json.loads(
            path.read_text()
        )
    except Exception:
        data = {}

    metadata = (
        data.get("metadata", {})
        if isinstance(data, dict)
        else {}
    )

    text = str(path)

    window = metadata.get(
        "window_id"
    )

    if window is None:
        match = re.search(
            r"(gb_\d+|manual_[A-Za-z0-9_]+)",
            text,
        )
        window = (
            match.group(1)
            if match
            else None
        )

    k = metadata.get("K")

    if k is None:
        match = re.search(
            r"(?:^|/)[kK](\d+)(?:/|$)",
            text,
        )
        k = (
            int(match.group(1))
            if match
            else None
        )

    seed = metadata.get("seed")

    if seed is None:
        match = re.search(
            r"(?:^|/)seed(-?\d+)(?:/|$)",
            text,
        )
        seed = (
            int(match.group(1))
            if match
            else None
        )

    fidelity = str(
        metadata.get("fidelity")
        or metadata.get("phase")
        or (
            "highres"
            if "highres" in text
            else "search"
        )
    )

    return (
        window,
        None if k is None else int(k),
        None if seed is None else int(seed),
        fidelity,
    )


def build_reuse_index(
    roots: Sequence[str],
) -> dict[
    tuple[str, int, int, str],
    Path,
]:
    index: dict[
        tuple[str, int, int, str],
        Path,
    ] = {}

    for raw in roots:
        root = (
            Path(raw)
            .expanduser()
            .resolve()
        )

        if not root.exists():
            continue

        for path in root.rglob(
            "summary.json"
        ):
            (
                window,
                k,
                seed,
                fidelity,
            ) = infer_run_identity(path)

            if (
                window is None
                or k is None
                or seed is None
            ):
                continue

            try:
                extract_logz(path)

            except (
                KeyError,
                ValueError,
                json.JSONDecodeError,
                OSError,
            ) as exc:
                print(
                    "[reuse-index] "
                    "skipping invalid summary "
                    f"{path}: {exc}",
                    flush=True,
                )
                continue

            key = (
                window,
                k,
                seed,
                fidelity,
            )

            old = index.get(key)

            if (
                old is None
                or path.stat().st_mtime
                > old.stat().st_mtime
            ):
                index[key] = path

    return index


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    k: int,
    seed: int,
    fidelity: str,
) -> None:
    data = json.loads(
        path.read_text()
    )

    old = (
        data.get("metadata", {})
        if isinstance(
            data.get("metadata"),
            dict,
        )
        else {}
    )

    data["metadata"] = {
        **old,
        "window_id": args.window_id,
        "f_min": args.f_min,
        "f_max": args.f_max,
        "K": k,
        "seed": seed,
        "fidelity": fidelity,
    }

    path.write_text(
        json.dumps(
            data,
            indent=2,
        )
    )


def ensure_run(
    args: argparse.Namespace,
    k: int,
    seed: int,
    fidelity: str,
    output_dir: Path,
    reuse_index: dict[
        tuple[str, int, int, str],
        Path,
    ],
) -> RunResult:
    keys = [
        (
            args.window_id,
            k,
            seed,
            fidelity,
        )
    ]

    if fidelity == "search":
        keys += [
            (
                args.window_id,
                k,
                seed,
                alias,
            )
            for alias in (
                "adaptive_scan",
                "static",
                "scan",
            )
        ]

    if not args.force_runs:
        for key in keys:
            if key not in reuse_index:
                continue

            path = reuse_index[key]

            try:
                logz, err = extract_logz(
                    path
                )

            except (
                KeyError,
                ValueError,
                json.JSONDecodeError,
                OSError,
            ) as exc:
                print(
                    "[reuse-invalid] "
                    f"ignoring {path}: {exc}",
                    flush=True,
                )
                continue

            print(
                f"[reuse] {fidelity} "
                f"K={k} seed={seed}: "
                f"{path}",
                flush=True,
            )

            return RunResult(
                k=k,
                seed=seed,
                logz=logz,
                logz_err=err,
                summary_path=str(path),
                fidelity=fidelity,
            )

    config_path = (
        output_dir
        / "configs"
        / f"k{k}.json"
    )

    summary_path = (
        output_dir
        / fidelity
        / f"k{k}"
        / f"seed{seed}"
        / "summary.json"
    )

    config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    workflow_root = (
        Path(args.workflow_root)
        .resolve()
    )

    if (
        args.force_runs
        or not config_path.exists()
    ):
        run_command(
            [
                args.python,
                str(
                    workflow_root
                    / "scripts"
                    / "make_band_config.py"
                ),
                "--base-json",
                str(
                    Path(
                        args.base_json
                    ).resolve()
                ),
                "--out-json",
                str(config_path),
                "--f-min",
                repr(args.f_min),
                "--f-max",
                repr(args.f_max),
                "--kmax",
                str(k),
            ],
            args.dry_run,
        )

    prefix = (
        "search"
        if fidelity == "search"
        else "highres"
    )

    settings = {
        "n_live": getattr(
            args,
            f"{prefix}_n_live",
        ),
        "tol": getattr(
            args,
            f"{prefix}_tol",
        ),
        "num_inner_steps": getattr(
            args,
            f"{prefix}_num_inner_steps",
        ),
        "num_delete_ratio": getattr(
            args,
            f"{prefix}_num_delete_ratio",
        ),
        "initial_num_steps": getattr(
            args,
            f"{prefix}_initial_num_steps",
        ),
        "refinement_num_steps": getattr(
            args,
            f"{prefix}_refinement_num_steps",
        ),
        "max_batches": getattr(
            args,
            f"{prefix}_max_batches",
        ),
    }

    summary_is_valid = False

    if (
        summary_path.exists()
        and not args.force_runs
    ):
        try:
            extract_logz(
                summary_path
            )

        except (
            KeyError,
            ValueError,
            json.JSONDecodeError,
            OSError,
        ) as exc:
            print(
                "[local-invalid] "
                "removing incomplete output "
                f"{summary_path}: {exc}",
                flush=True,
            )

            summary_path.unlink(
                missing_ok=True
            )

        else:
            summary_is_valid = True

    if (
        args.force_runs
        or not summary_is_valid
    ):
        command = [
            args.python,
            str(
                workflow_root
                / "scripts"
                / "run_benchmark.py"
            ),
            "--python",
            args.python,
            "--repo-root",
            str(
                Path(
                    args.repo_root
                ).resolve()
            ),
            "--problem-factory",
            args.problem_factory,
            "--config-json",
            str(config_path),
            "--algo",
            "ns",
            "--seed",
            str(seed),
            "--n-live",
            str(settings["n_live"]),
            "--tol",
            str(settings["tol"]),
            "--num-inner-steps",
            str(
                settings[
                    "num_inner_steps"
                ]
            ),
            "--num-delete-ratio",
            str(
                settings[
                    "num_delete_ratio"
                ]
            ),
            "--initial-num-steps",
            str(
                settings[
                    "initial_num_steps"
                ]
            ),
            "--refinement-num-steps",
            str(
                settings[
                    "refinement_num_steps"
                ]
            ),
            "--max-batches",
            str(
                settings[
                    "max_batches"
                ]
            ),
            "--skip-plots",
            "--out-json",
            str(summary_path),
        ]

        run_command(
            command,
            args.dry_run,
        )

        if not args.dry_run:
            write_metadata(
                summary_path,
                args,
                k,
                seed,
                fidelity,
            )

    if args.dry_run:
        return RunResult(
            k=k,
            seed=seed,
            logz=math.nan,
            logz_err=math.nan,
            summary_path=str(
                summary_path
            ),
            fidelity=fidelity,
        )

    logz, err = extract_logz(
        summary_path
    )

    return RunResult(
        k=k,
        seed=seed,
        logz=logz,
        logz_err=err,
        summary_path=str(
            summary_path
        ),
        fidelity=fidelity,
    )


def choose_agreeing_pair(
    values: dict[int, float],
    tolerance: float,
) -> tuple[
    list[int],
    float,
    float,
] | None:
    candidates = []

    for a, b in itertools.combinations(
        sorted(values),
        2,
    ):
        spread = abs(
            values[a] - values[b]
        )

        if spread <= tolerance:
            candidates.append(
                (
                    0.5
                    * (
                        values[a]
                        + values[b]
                    ),
                    spread,
                    [a, b],
                )
            )

    if not candidates:
        return None

    mean, spread, pair = max(
        candidates,
        key=lambda x: (
            x[0],
            -x[1],
        ),
    )

    return (
        pair,
        mean,
        spread,
    )


def summarize_evidence(
    results: Sequence[RunResult],
    pair_tolerance: float,
) -> EvidencePoint:
    if not results:
        raise ValueError(
            "Cannot summarize empty "
            "run collection."
        )

    values = {
        result.seed: result.logz
        for result in results
    }

    errors = {
        result.seed: result.logz_err
        for result in results
    }

    pair = choose_agreeing_pair(
        values,
        pair_tolerance,
    )

    median_logz = float(
        statistics.median(
            values.values()
        )
    )

    mean_logz = float(
        statistics.fmean(
            values.values()
        )
    )

    std_logz = (
        float(
            statistics.stdev(
                values.values()
            )
        )
        if len(values) > 1
        else 0.0
    )

    if pair is None:
        agreeing_seeds: list[int] = []
        robust_logz = median_logz
        pair_spread = math.nan
        robust_method = (
            "median_no_agreeing_pair"
        )
        confidence = "low"

    else:
        (
            agreeing_seeds,
            robust_logz,
            pair_spread,
        ) = pair

        robust_method = (
            "highest_agreeing_pair_mean"
        )

        provisional_outliers = sum(
            abs(
                value
                - robust_logz
            )
            > pair_tolerance
            for value
            in values.values()
        )

        confidence = (
            "high"
            if provisional_outliers == 0
            else "moderate"
        )

    n_outliers = sum(
        abs(
            value
            - robust_logz
        )
        > pair_tolerance
        for value
        in values.values()
    )

    return EvidencePoint(
        k=results[0].k,
        seeds=sorted(values),
        logz_values=[
            values[seed]
            for seed in sorted(values)
        ],
        logz_err_values=[
            errors[seed]
            for seed in sorted(values)
        ],
        robust_logz=float(
            robust_logz
        ),
        robust_method=(
            robust_method
        ),
        agreeing_seeds=(
            agreeing_seeds
        ),
        pair_spread=float(
            pair_spread
        ),
        median_logz=median_logz,
        mean_logz=mean_logz,
        std_logz=std_logz,
        min_logz=float(
            min(values.values())
        ),
        max_logz=float(
            max(values.values())
        ),
        n_outliers=n_outliers,
        status="available",
        confidence=confidence,
    )


def find_bracketed_peak(
    points: dict[
        int,
        EvidencePoint,
    ],
    confirm_lower_points: int,
    min_drop: float,
) -> int | None:
    if len(points) < 3:
        return None

    tested = sorted(points)

    best_k = max(
        tested,
        key=lambda k: (
            points[k].robust_logz
        ),
    )

    # Search-stage best point must
    # be interior.
    if (
        best_k == tested[0]
        or best_k == tested[-1]
    ):
        return None

    later_k = [
        k
        for k in tested
        if k > best_k
    ]

    if (
        len(later_k)
        < confirm_lower_points
    ):
        return None

    best_logz = (
        points[
            best_k
        ].robust_logz
    )

    for k in later_k[
        :confirm_lower_points
    ]:
        drop = (
            best_logz
            - points[k].robust_logz
        )

        if drop < min_drop:
            return None

    return best_k


def select_best(
    points: dict[
        int,
        EvidencePoint,
    ],
) -> int:
    if not points:
        raise ValueError(
            "Cannot select K from an "
            "empty evidence table."
        )

    return max(
        points,
        key=lambda k: (
            points[k].robust_logz
        ),
    )


def per_seed_model_order_diagnostic(
    runs: Sequence[RunResult],
    ks: Sequence[int],
) -> dict:
    by_seed: dict[
        int,
        dict[int, float],
    ] = {}

    allowed_ks = set(ks)

    for run in runs:
        if run.k not in allowed_ks:
            continue

        by_seed.setdefault(
            run.seed,
            {},
        )

        by_seed[
            run.seed
        ][run.k] = run.logz

    seed_best_k: dict[
        int,
        int,
    ] = {}

    seed_delta_logz: dict[
        int,
        dict[int, float],
    ] = {}

    for (
        seed,
        values,
    ) in by_seed.items():
        if not values:
            continue

        best_k = max(
            values,
            key=values.get,
        )

        best_logz = values[
            best_k
        ]

        seed_best_k[
            seed
        ] = best_k

        seed_delta_logz[
            seed
        ] = {
            k: value - best_logz
            for (
                k,
                value,
            )
            in sorted(
                values.items()
            )
        }

    if not seed_best_k:
        return {
            "seed_best_k": {},
            "seed_delta_logz": {},
            "majority_best_k": None,
            "n_support": 0,
            "n_seeds": 0,
            "support_fraction": 0.0,
        }

    counts: dict[
        int,
        int,
    ] = {}

    for best_k in (
        seed_best_k.values()
    ):
        counts[best_k] = (
            counts.get(
                best_k,
                0,
            )
            + 1
        )

    majority_best_k = max(
        counts,
        key=counts.get,
    )

    n_support = counts[
        majority_best_k
    ]

    n_seeds = len(
        seed_best_k
    )

    return {
        "seed_best_k": (
            seed_best_k
        ),
        "seed_delta_logz": (
            seed_delta_logz
        ),
        "majority_best_k": (
            majority_best_k
        ),
        "n_support": (
            n_support
        ),
        "n_seeds": (
            n_seeds
        ),
        "support_fraction": (
            n_support / n_seeds
        ),
    }


def select_best_from_seed_deltas(
    ranking: dict,
) -> tuple[
    int,
    dict[int, float],
]:
    seed_delta_logz = ranking[
        "seed_delta_logz"
    ]

    if not seed_delta_logz:
        raise ValueError(
            "Cannot select model order "
            "without per-seed evidence "
            "differences."
        )

    all_ks = sorted(
        {
            int(k)
            for seed_values
            in seed_delta_logz.values()
            for k
            in seed_values
        }
    )

    median_delta_by_k: dict[
        int,
        float,
    ] = {}

    for k in all_ks:
        values = [
            seed_values[k]
            for seed_values
            in seed_delta_logz.values()
            if k in seed_values
        ]

        if values:
            median_delta_by_k[
                k
            ] = float(
                statistics.median(
                    values
                )
            )

    if not median_delta_by_k:
        raise ValueError(
            "No usable per-seed "
            "delta-logZ values."
        )

    best_k = max(
        median_delta_by_k,
        key=median_delta_by_k.get,
    )

    return (
        best_k,
        median_delta_by_k,
    )


def write_points_csv(
    path: Path,
    args: argparse.Namespace,
    points: dict[
        int,
        EvidencePoint,
    ],
) -> None:
    fields = [
        "window_id",
        "f_min",
        "f_max",
        "K",
        "n_seeds",
        "seeds",
        "robust_logZ",
        "robust_method",
        "agreeing_seeds",
        "pair_spread",
        "median_logZ",
        "mean_logZ",
        "std_logZ",
        "min_logZ",
        "max_logZ",
        "n_outliers",
        "status",
        "confidence",
    ]

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for (
            k,
            point,
        ) in sorted(
            points.items()
        ):
            writer.writerow(
                {
                    "window_id": (
                        args.window_id
                    ),
                    "f_min": (
                        args.f_min
                    ),
                    "f_max": (
                        args.f_max
                    ),
                    "K": k,
                    "n_seeds": (
                        len(
                            point.seeds
                        )
                    ),
                    "seeds": (
                        " ".join(
                            map(
                                str,
                                point.seeds,
                            )
                        )
                    ),
                    "robust_logZ": (
                        point.robust_logz
                    ),
                    "robust_method": (
                        point.robust_method
                    ),
                    "agreeing_seeds": (
                        " ".join(
                            map(
                                str,
                                point.agreeing_seeds,
                            )
                        )
                    ),
                    "pair_spread": (
                        point.pair_spread
                    ),
                    "median_logZ": (
                        point.median_logz
                    ),
                    "mean_logZ": (
                        point.mean_logz
                    ),
                    "std_logZ": (
                        point.std_logz
                    ),
                    "min_logZ": (
                        point.min_logz
                    ),
                    "max_logZ": (
                        point.max_logz
                    ),
                    "n_outliers": (
                        point.n_outliers
                    ),
                    "status": (
                        point.status
                    ),
                    "confidence": (
                        point.confidence
                    ),
                }
            )


def write_state(
    output_dir: Path,
    args: argparse.Namespace,
    decision: ControllerDecision,
    search_points: dict[
        int,
        EvidencePoint,
    ],
    confirmation_points: dict[
        int,
        EvidencePoint,
    ],
    search_runs: Sequence[
        RunResult
    ],
    confirmation_runs: Sequence[
        RunResult
    ],
    ranking_diagnostic: (
        dict | None
    ) = None,
) -> None:
    state = {
        "window_id": (
            args.window_id
        ),
        "f_min": args.f_min,
        "f_max": args.f_max,
        "decision": (
            asdict(decision)
        ),
        "policy": {
            "initial_k_max": (
                args.initial_k_max
            ),
            "safety_k_max": (
                args.safety_k_max
            ),
            "confirm_lower_points": (
                args.confirm_lower_points
            ),
            "min_peak_drop_logz": (
                args.min_peak_drop_logz
            ),
            "pair_agreement_logz": (
                args.pair_agreement_logz
            ),
            "max_confirmation_extensions": (
                args.max_confirmation_extensions
            ),
            "min_confirmation_rise_logz": (
                args.min_confirmation_rise_logz
            ),
            "search_seeds": (
                parse_seed_list(
                    args.search_seeds
                )
            ),
            "confirmation_seeds": (
                parse_seed_list(
                    args.confirmation_seeds
                )
            ),
        },
        "search_points": {
            str(k): asdict(v)
            for (
                k,
                v,
            )
            in sorted(
                search_points.items()
            )
        },
        "confirmation_points": {
            str(k): asdict(v)
            for (
                k,
                v,
            )
            in sorted(
                confirmation_points.items()
            )
        },
        "search_runs": [
            asdict(x)
            for x in search_runs
        ],
        "confirmation_runs": [
            asdict(x)
            for x
            in confirmation_runs
        ],
        "ranking_diagnostic": (
            ranking_diagnostic
        ),
    }

    (
        output_dir
        / "state.json"
    ).write_text(
        json.dumps(
            state,
            indent=2,
        )
    )

    selected_path = (
        output_dir
        / "selected_k.csv"
    )

    with selected_path.open(
        "w",
        newline="",
    ) as handle:
        fields = [
            "window_id",
            "f_min",
            "f_max",
            "K_best",
            "status",
            "next_action",
            "reason",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerow(
            {
                "window_id": (
                    args.window_id
                ),
                "f_min": (
                    args.f_min
                ),
                "f_max": (
                    args.f_max
                ),
                "K_best": (
                    ""
                    if (
                        decision.selected_k
                        is None
                    )
                    else (
                        decision.selected_k
                    )
                ),
                "status": (
                    decision.status
                ),
                "next_action": (
                    decision.next_action
                ),
                "reason": (
                    decision.reason
                ),
            }
        )

    write_points_csv(
        output_dir
        / "search_evidence.csv",
        args,
        search_points,
    )

    write_points_csv(
        output_dir
        / "confirmation_evidence.csv",
        args,
        confirmation_points,
    )


def main() -> int:
    args = (
        build_parser()
        .parse_args()
    )

    search_seeds = (
        parse_seed_list(
            args.search_seeds
        )
    )

    confirmation_seeds = (
        parse_seed_list(
            args.confirmation_seeds
        )
    )

    output_dir = (
        Path(
            args.output_root
        ).resolve()
        / args.window_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reuse_index = (
        build_reuse_index(
            [
                *args.reuse_root,
                str(output_dir),
            ]
        )
    )

    search_runs: list[
        RunResult
    ] = []

    confirmation_runs: list[
        RunResult
    ] = []

    search_points: dict[
        int,
        EvidencePoint,
    ] = {}

    confirmation_points: dict[
        int,
        EvidencePoint,
    ] = {}

    current_max = (
        args.initial_k_max
    )

    peak: int | None = None

    # -------------------------------------------------
    # Search stage
    # -------------------------------------------------

    for k in range(
        1,
        args.safety_k_max + 1,
    ):
        if k > current_max:
            break

        runs = [
            ensure_run(
                args,
                k,
                seed,
                "search",
                output_dir,
                reuse_index,
            )
            for seed
            in search_seeds
        ]

        search_runs.extend(
            runs
        )

        if args.dry_run:
            if (
                k == current_max
                and current_max
                < args.safety_k_max
            ):
                current_max += 1

            continue

        search_points[k] = (
            summarize_evidence(
                runs,
                args.pair_agreement_logz,
            )
        )

        peak = (
            find_bracketed_peak(
                search_points,
                args.confirm_lower_points,
                args.min_peak_drop_logz,
            )
        )

        if peak is not None:
            break

        if (
            k == current_max
            and current_max
            < args.safety_k_max
        ):
            current_max += 1

        write_state(
            output_dir,
            args,
            ControllerDecision(
                status="searching",
                selected_k=None,
                current_best_k=(
                    select_best(
                        search_points
                    )
                ),
                next_k=(
                    current_max
                    if (
                        current_max
                        <= args.safety_k_max
                    )
                    else None
                ),
                next_action=(
                    "run_next_k"
                ),
                reason=(
                    "No bracketed "
                    "interior evidence "
                    "maximum yet."
                ),
            ),
            search_points,
            confirmation_points,
            search_runs,
            confirmation_runs,
        )

    if args.dry_run:
        return 0

    # -------------------------------------------------
    # No search-stage bracket
    # -------------------------------------------------

    if peak is None:
        best = select_best(
            search_points
        )

        write_state(
            output_dir,
            args,
            ControllerDecision(
                status=(
                    "model_order_unresolved"
                ),
                selected_k=None,
                current_best_k=best,
                next_k=None,
                next_action=(
                    "review_or_raise_safety_cap"
                ),
                reason=(
                    "No bracketed maximum "
                    f"before K="
                    f"{args.safety_k_max}; "
                    f"current best K={best}."
                ),
            ),
            search_points,
            confirmation_points,
            search_runs,
            confirmation_runs,
        )

        return 0

    # -------------------------------------------------
    # Optional search-only result
    # -------------------------------------------------

    if (
        args.skip_highres_confirmation
    ):
        peak_point = (
            search_points[peak]
        )

        next_action = (
            "run_additional_stability_seeds"
            if (
                peak_point.confidence
                == "low"
            )
            else (
                "run_stability_and_consensus"
            )
        )

        write_state(
            output_dir,
            args,
            ControllerDecision(
                status=(
                    "search_peak_selected"
                ),
                selected_k=peak,
                current_best_k=peak,
                next_k=None,
                next_action=(
                    next_action
                ),
                reason=(
                    f"Bracketed search "
                    f"peak K={peak}; "
                    f"confidence="
                    f"{peak_point.confidence}; "
                    "high-resolution "
                    "confirmation skipped."
                ),
            ),
            search_points,
            confirmation_points,
            search_runs,
            confirmation_runs,
        )

        return 0

    # -------------------------------------------------
    # High-resolution confirmation stage
    # -------------------------------------------------

    confirmation_ks = {
        max(
            1,
            peak - 1,
        ),
        peak,
        min(
            args.safety_k_max,
            peak + 1,
        ),
    }

    evaluated_highres: set[
        int
    ] = set()

    n_confirmation_extensions = 0

    latest_ranking: dict | None = (
        None
    )

    latest_median_delta_by_k: dict[
        int,
        float,
    ] = {}

    while True:
        for k in sorted(
            confirmation_ks
        ):
            if (
                k
                in evaluated_highres
            ):
                continue

            runs = [
                ensure_run(
                    args,
                    k,
                    seed,
                    "highres",
                    output_dir,
                    reuse_index,
                )
                for seed
                in confirmation_seeds
            ]

            confirmation_runs.extend(
                runs
            )

            confirmation_points[k] = (
                summarize_evidence(
                    runs,
                    args.pair_agreement_logz,
                )
            )

            evaluated_highres.add(
                k
            )

        latest_ranking = (
            per_seed_model_order_diagnostic(
                confirmation_runs,
                sorted(
                    confirmation_points
                ),
            )
        )

        (
            confirmed,
            latest_median_delta_by_k,
        ) = (
            select_best_from_seed_deltas(
                latest_ranking
            )
        )

        latest_ranking[
            "median_delta_logz_by_k"
        ] = (
            latest_median_delta_by_k
        )

        tested = sorted(
            confirmation_points
        )

        if len(tested) < 2:
            break

        # ---------------------------------------------
        # Upper-boundary extension
        # ---------------------------------------------

        if (
            confirmed == tested[-1]
            and confirmed
            < args.safety_k_max
        ):
            previous_k = tested[-2]

            boundary_rise = (
                latest_median_delta_by_k[
                    confirmed
                ]
                - latest_median_delta_by_k[
                    previous_k
                ]
            )

            if (
                boundary_rise
                < args.min_confirmation_rise_logz
            ):
                print(
                    "[confirmation-stop] "
                    f"best K={confirmed} "
                    "is upper boundary, "
                    f"but rise from "
                    f"K={previous_k} is "
                    f"{boundary_rise:.3f} "
                    "< "
                    f"{args.min_confirmation_rise_logz:.3f}",
                    flush=True,
                )
                break

            if (
                n_confirmation_extensions
                >= (
                    args.max_confirmation_extensions
                )
            ):
                print(
                    "[confirmation-stop] "
                    f"best K={confirmed} "
                    "remains at upper "
                    "boundary after "
                    f"{n_confirmation_extensions} "
                    "extension(s); "
                    f"boundary rise="
                    f"{boundary_rise:.3f}",
                    flush=True,
                )
                break

            new_k = (
                confirmed + 1
            )

            print(
                "[confirmation-extend] "
                f"best K={confirmed} "
                "is upper boundary; "
                f"rise from K="
                f"{previous_k} is "
                f"{boundary_rise:.3f}; "
                f"adding K={new_k}",
                flush=True,
            )

            confirmation_ks.add(
                new_k
            )

            n_confirmation_extensions += 1

            continue

        # ---------------------------------------------
        # Lower-boundary extension
        # ---------------------------------------------

        if (
            confirmed == tested[0]
            and confirmed > 1
        ):
            next_k = tested[1]

            boundary_rise = (
                latest_median_delta_by_k[
                    confirmed
                ]
                - latest_median_delta_by_k[
                    next_k
                ]
            )

            if (
                boundary_rise
                < args.min_confirmation_rise_logz
            ):
                print(
                    "[confirmation-stop] "
                    f"best K={confirmed} "
                    "is lower boundary, "
                    f"but rise relative "
                    f"to K={next_k} is "
                    f"{boundary_rise:.3f} "
                    "< "
                    f"{args.min_confirmation_rise_logz:.3f}",
                    flush=True,
                )
                break

            if (
                n_confirmation_extensions
                >= (
                    args.max_confirmation_extensions
                )
            ):
                print(
                    "[confirmation-stop] "
                    f"best K={confirmed} "
                    "remains at lower "
                    "boundary after "
                    f"{n_confirmation_extensions} "
                    "extension(s); "
                    f"boundary rise="
                    f"{boundary_rise:.3f}",
                    flush=True,
                )
                break

            new_k = (
                confirmed - 1
            )

            print(
                "[confirmation-extend] "
                f"best K={confirmed} "
                "is lower boundary; "
                f"rise relative to "
                f"K={next_k} is "
                f"{boundary_rise:.3f}; "
                f"adding K={new_k}",
                flush=True,
            )

            confirmation_ks.add(
                new_k
            )

            n_confirmation_extensions += 1

            continue

        break

    # -------------------------------------------------
    # Final confirmation decision
    # -------------------------------------------------

    ranking = (
        per_seed_model_order_diagnostic(
            confirmation_runs,
            sorted(
                confirmation_points
            ),
        )
    )

    (
        confirmed,
        median_delta_by_k,
    ) = (
        select_best_from_seed_deltas(
            ranking
        )
    )

    ranking[
        "median_delta_logz_by_k"
    ] = (
        median_delta_by_k
    )

    confirmed_point = (
        confirmation_points[
            confirmed
        ]
    )

    ranking_supports_confirmed = (
        ranking[
            "majority_best_k"
        ]
        == confirmed
    )

    strong_ranking_support = (
        ranking_supports_confirmed
        and (
            ranking[
                "support_fraction"
            ]
            >= 2.0 / 3.0
        )
    )

    tested = sorted(
        confirmation_points
    )

    boundary_selected = (
        confirmed == tested[0]
        or confirmed == tested[-1]
    )

    if boundary_selected:
        status = (
            "confirmation_unresolved_boundary"
        )

        next_action = (
            "extend_confirmation_range"
        )

    elif confirmed != peak:
        status = (
            "confirmation_shifted_peak"
        )

        next_action = (
            "run_stability_and_consensus"
            if strong_ranking_support
            else (
                "run_additional_stability_seeds"
            )
        )

    elif strong_ranking_support:
        status = (
            "complete_model_order_consistent"
        )

        next_action = (
            "run_stability_and_consensus"
        )

    elif (
        confirmed_point.confidence
        == "low"
    ):
        status = (
            "selected_low_confidence"
        )

        next_action = (
            "run_additional_stability_seeds"
        )

    else:
        status = "complete"

        next_action = (
            "run_stability_and_consensus"
        )

    reason = (
        f"Search peak K={peak}; "
        "high-resolution "
        "seed-normalized selection "
        f"K={confirmed}; "
        "median-delta-logZ="
        f"{median_delta_by_k.get(confirmed)}; "
        "absolute-confidence="
        f"{confirmed_point.confidence}; "
        "seed-majority-K="
        f"{ranking['majority_best_k']}; "
        "seed-support="
        f"{ranking['n_support']}/"
        f"{ranking['n_seeds']}."
    )

    write_state(
        output_dir,
        args,
        ControllerDecision(
            status=status,
            selected_k=confirmed,
            current_best_k=confirmed,
            next_k=None,
            next_action=(
                next_action
            ),
            reason=reason,
        ),
        search_points,
        confirmation_points,
        search_runs,
        confirmation_runs,
        ranking_diagnostic=(
            ranking
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
