#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class RunResult:
    k: int
    seed: int
    logz: float
    logz_err: float
    summary_path: str
    fidelity: str
    origin: str


@dataclass
class RankingDiagnostic:
    ks: list[int]
    seeds: list[int]
    seed_best_k: dict[int, int]
    seed_delta_logz: dict[int, dict[int, float]]
    median_delta_logz_by_k: dict[int, float]
    best_k: int
    runner_up_k: int | None
    margin_to_runner_up: float
    best_k_counts: dict[int, int]
    selected_k_support: int
    selected_k_support_fraction: float
    n_seeds: int


@dataclass
class ConsensusDecision:
    status: str
    selected_k: int
    new_seed_best_k: int
    combined_best_k: int
    next_action: str
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stability and consensus validation for model order"
    )

    parser.add_argument(
        "--window-id",
        required=True,
    )

    parser.add_argument(
        "--model-order-root",
        default="results/model_order",
    )

    parser.add_argument(
        "--output-root",
        default="results/stability_consensus",
    )

    parser.add_argument(
        "--base-json",
        required=True,
    )

    parser.add_argument(
        "--repo-root",
        required=True,
    )

    parser.add_argument(
        "--workflow-root",
        default=".",
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
    )

    parser.add_argument(
        "--problem-factory",
        default=(
            "jax_samplers.problems."
            "lisa_gb_transdim_problem:make"
        ),
    )

    parser.add_argument(
        "--seeds",
        default="33,44,55",
    )

    parser.add_argument(
        "--n-live",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--num-inner-steps",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-delete-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--initial-num-steps",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--refinement-num-steps",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--min-support-fraction",
        type=float,
        default=2.0 / 3.0,
    )

    parser.add_argument(
        "--min-median-delta-margin",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--force-runs",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser


def parse_seed_list(raw: str) -> list[int]:
    seeds = [
        int(value.strip())
        for value in raw.split(",")
        if value.strip()
    ]

    if not seeds:
        raise ValueError(
            f"Invalid seed list: {raw}"
        )

    if len(seeds) != len(set(seeds)):
        raise ValueError(
            f"Duplicate seeds in: {raw}"
        )

    return seeds


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
    paths: Sequence[Sequence[str]],
) -> Any:
    for path in paths:
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

    logz_err = nested_get(
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

    if logz_err is None:
        err_value = math.nan
    else:
        err_value = float(logz_err)

    return (
        logz_value,
        err_value,
    )


def write_metadata(
    path: Path,
    window_id: str,
    f_min: float,
    f_max: float,
    k: int,
    seed: int,
) -> None:
    data = json.loads(
        path.read_text()
    )

    old_metadata = (
        data.get("metadata", {})
        if isinstance(
            data.get("metadata"),
            dict,
        )
        else {}
    )

    data["metadata"] = {
        **old_metadata,
        "window_id": window_id,
        "f_min": f_min,
        "f_max": f_max,
        "K": k,
        "seed": seed,
        "fidelity": "stability",
        "origin": "stability_consensus",
    }

    path.write_text(
        json.dumps(
            data,
            indent=2,
        )
    )


def load_model_order_state(
    args: argparse.Namespace,
) -> dict[str, Any]:
    state_path = (
        Path(args.model_order_root).resolve()
        / args.window_id
        / "state.json"
    )

    if not state_path.exists():
        raise FileNotFoundError(
            f"Model-order state does not exist: "
            f"{state_path}"
        )

    state = json.loads(
        state_path.read_text()
    )

    decision = state.get(
        "decision",
        {},
    )

    selected_k = decision.get(
        "selected_k"
    )

    if selected_k is None:
        raise RuntimeError(
            f"{args.window_id} has no resolved selected K. "
            f"Model-order status="
            f"{decision.get('status')!r}"
        )

    if decision.get("next_action") in {
        "review_or_raise_safety_cap",
        "extend_confirmation_range",
    }:
        raise RuntimeError(
            f"{args.window_id} is not ready for stability "
            f"validation: next_action="
            f"{decision.get('next_action')!r}"
        )

    return state


def original_confirmation_runs(
    state: dict[str, Any],
    ks: Sequence[int],
) -> list[RunResult]:
    allowed_ks = set(ks)

    runs: list[RunResult] = []

    for item in state.get(
        "confirmation_runs",
        [],
    ):
        k = int(
            item["k"]
        )

        if k not in allowed_ks:
            continue

        runs.append(
            RunResult(
                k=k,
                seed=int(
                    item["seed"]
                ),
                logz=float(
                    item["logz"]
                ),
                logz_err=float(
                    item.get(
                        "logz_err",
                        math.nan,
                    )
                ),
                summary_path=str(
                    item.get(
                        "summary_path",
                        "",
                    )
                ),
                fidelity=str(
                    item.get(
                        "fidelity",
                        "highres",
                    )
                ),
                origin="model_order_confirmation",
            )
        )

    if not runs:
        raise RuntimeError(
            "No original high-resolution confirmation "
            "runs found in state.json."
        )

    return runs


def ensure_stability_run(
    args: argparse.Namespace,
    output_dir: Path,
    f_min: float,
    f_max: float,
    k: int,
    seed: int,
) -> RunResult:
    workflow_root = Path(
        args.workflow_root
    ).resolve()

    config_path = (
        output_dir
        / "configs"
        / f"k{k}.json"
    )

    summary_path = (
        output_dir
        / "runs"
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
                repr(f_min),
                "--f-max",
                repr(f_max),
                "--kmax",
                str(k),
            ],
            args.dry_run,
        )

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
                "[stability-invalid] "
                f"removing {summary_path}: {exc}",
                flush=True,
            )

            summary_path.unlink(
                missing_ok=True
            )

        else:
            summary_is_valid = True

            print(
                f"[reuse] stability K={k} "
                f"seed={seed}: {summary_path}",
                flush=True,
            )

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
            str(args.n_live),
            "--tol",
            str(args.tol),
            "--num-inner-steps",
            str(
                args.num_inner_steps
            ),
            "--num-delete-ratio",
            str(
                args.num_delete_ratio
            ),
            "--initial-num-steps",
            str(
                args.initial_num_steps
            ),
            "--refinement-num-steps",
            str(
                args.refinement_num_steps
            ),
            "--max-batches",
            str(
                args.max_batches
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
                args.window_id,
                f_min,
                f_max,
                k,
                seed,
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
            fidelity="stability",
            origin="stability_consensus",
        )

    logz, logz_err = extract_logz(
        summary_path
    )

    return RunResult(
        k=k,
        seed=seed,
        logz=logz,
        logz_err=logz_err,
        summary_path=str(
            summary_path
        ),
        fidelity="stability",
        origin="stability_consensus",
    )


def build_ranking_diagnostic(
    runs: Sequence[RunResult],
    ks: Sequence[int],
    selected_k: int,
) -> RankingDiagnostic:
    required_ks = set(
        ks
    )

    by_seed: dict[
        int,
        dict[int, float],
    ] = {}

    for run in runs:
        if run.k not in required_ks:
            continue

        by_seed.setdefault(
            run.seed,
            {},
        )

        by_seed[
            run.seed
        ][run.k] = run.logz

    complete_by_seed = {
        seed: values
        for seed, values in by_seed.items()
        if set(values) == required_ks
    }

    dropped = sorted(
        set(by_seed)
        - set(complete_by_seed)
    )

    if dropped:
        print(
            "[ranking] ignoring incomplete seeds: "
            + ",".join(
                map(
                    str,
                    dropped,
                )
            ),
            flush=True,
        )

    if not complete_by_seed:
        raise RuntimeError(
            "No seed has a complete model-order "
            "neighbourhood."
        )

    seed_best_k: dict[
        int,
        int,
    ] = {}

    seed_delta_logz: dict[
        int,
        dict[int, float],
    ] = {}

    for seed, values in sorted(
        complete_by_seed.items()
    ):
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
            k: values[k] - best_logz
            for k in sorted(
                required_ks
            )
        }

    median_delta_logz_by_k = {
        k: float(
            statistics.median(
                seed_delta_logz[
                    seed
                ][k]
                for seed in seed_delta_logz
            )
        )
        for k in sorted(
            required_ks
        )
    }

    ordered = sorted(
        median_delta_logz_by_k,
        key=lambda k: (
            median_delta_logz_by_k[k]
        ),
        reverse=True,
    )

    best_k = ordered[0]

    runner_up_k = (
        ordered[1]
        if len(ordered) > 1
        else None
    )

    if runner_up_k is None:
        margin = math.inf
    else:
        margin = (
            median_delta_logz_by_k[
                best_k
            ]
            - median_delta_logz_by_k[
                runner_up_k
            ]
        )

    best_k_counts: dict[
        int,
        int,
    ] = {
        k: 0
        for k in sorted(
            required_ks
        )
    }

    for seed_best in (
        seed_best_k.values()
    ):
        best_k_counts[
            seed_best
        ] += 1

    n_seeds = len(
        seed_best_k
    )

    selected_support = (
        best_k_counts.get(
            selected_k,
            0,
        )
    )

    return RankingDiagnostic(
        ks=sorted(
            required_ks
        ),
        seeds=sorted(
            seed_best_k
        ),
        seed_best_k=seed_best_k,
        seed_delta_logz=seed_delta_logz,
        median_delta_logz_by_k=(
            median_delta_logz_by_k
        ),
        best_k=best_k,
        runner_up_k=runner_up_k,
        margin_to_runner_up=float(
            margin
        ),
        best_k_counts=best_k_counts,
        selected_k_support=(
            selected_support
        ),
        selected_k_support_fraction=(
            selected_support
            / n_seeds
        ),
        n_seeds=n_seeds,
    )


def decide_consensus(
    selected_k: int,
    new_ranking: RankingDiagnostic,
    combined_ranking: RankingDiagnostic,
    min_support_fraction: float,
    min_margin: float,
) -> ConsensusDecision:
    new_support_ok = (
        new_ranking.selected_k_support_fraction
        >= min_support_fraction
    )

    combined_support_ok = (
        combined_ranking.selected_k_support_fraction
        >= min_support_fraction
    )

    new_selects_original = (
        new_ranking.best_k
        == selected_k
    )

    combined_selects_original = (
        combined_ranking.best_k
        == selected_k
    )

    margin_ok = (
        combined_ranking.margin_to_runner_up
        >= min_margin
    )

    if (
        new_selects_original
        and combined_selects_original
        and new_support_ok
        and combined_support_ok
        and margin_ok
    ):
        status = "stable_consensus"
        next_action = "accept_model_order"

        reason = (
            f"K={selected_k} reproduced by new seeds "
            f"({new_ranking.selected_k_support}/"
            f"{new_ranking.n_seeds}) and by the combined "
            f"sample "
            f"({combined_ranking.selected_k_support}/"
            f"{combined_ranking.n_seeds}); "
            f"combined median-delta margin="
            f"{combined_ranking.margin_to_runner_up:.3f}."
        )

    elif (
        new_selects_original
        and combined_selects_original
        and new_support_ok
        and combined_support_ok
    ):
        status = "stable_weak_margin"
        next_action = "add_stability_seeds"

        reason = (
            f"K={selected_k} has reproducible seed support, "
            f"but combined median-delta margin="
            f"{combined_ranking.margin_to_runner_up:.3f} "
            f"< {min_margin:.3f}."
        )

    elif (
        combined_ranking.best_k
        != selected_k
        and new_ranking.best_k
        == combined_ranking.best_k
        and (
            new_ranking.best_k_counts.get(
                new_ranking.best_k,
                0,
            )
            / new_ranking.n_seeds
            >= min_support_fraction
        )
    ):
        status = "model_order_shifted"
        next_action = "return_to_model_order_controller"

        reason = (
            f"Original K={selected_k} was not reproduced. "
            f"New and combined rankings prefer "
            f"K={combined_ranking.best_k}."
        )

    else:
        status = "seed_disagreement"
        next_action = "add_stability_seeds_or_refine_sampler"

        reason = (
            f"Stability seeds do not establish consensus: "
            f"new best K={new_ranking.best_k}, "
            f"combined best K={combined_ranking.best_k}, "
            f"original K={selected_k}."
        )

    return ConsensusDecision(
        status=status,
        selected_k=selected_k,
        new_seed_best_k=(
            new_ranking.best_k
        ),
        combined_best_k=(
            combined_ranking.best_k
        ),
        next_action=next_action,
        reason=reason,
    )


def write_ranking_csv(
    path: Path,
    window_id: str,
    f_min: float,
    f_max: float,
    selected_k: int,
    ranking: RankingDiagnostic,
) -> None:
    fields = [
        "window_id",
        "f_min",
        "f_max",
        "selected_k",
        "K",
        "n_seeds",
        "n_best",
        "support_fraction",
        "median_delta_logZ",
        "is_best",
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

        for k in ranking.ks:
            n_best = (
                ranking.best_k_counts.get(
                    k,
                    0,
                )
            )

            writer.writerow(
                {
                    "window_id": window_id,
                    "f_min": f_min,
                    "f_max": f_max,
                    "selected_k": selected_k,
                    "K": k,
                    "n_seeds": ranking.n_seeds,
                    "n_best": n_best,
                    "support_fraction": (
                        n_best
                        / ranking.n_seeds
                    ),
                    "median_delta_logZ": (
                        ranking
                        .median_delta_logz_by_k[k]
                    ),
                    "is_best": int(
                        k
                        == ranking.best_k
                    ),
                }
            )


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    model_state: dict[str, Any],
    selected_k: int,
    ks: Sequence[int],
    original_runs: Sequence[RunResult],
    new_runs: Sequence[RunResult],
    new_ranking: RankingDiagnostic,
    combined_ranking: RankingDiagnostic,
    decision: ConsensusDecision,
) -> None:
    f_min = float(
        model_state["f_min"]
    )

    f_max = float(
        model_state["f_max"]
    )

    state = {
        "window_id": args.window_id,
        "f_min": f_min,
        "f_max": f_max,
        "selected_k_from_model_order": (
            selected_k
        ),
        "neighbourhood": list(
            ks
        ),
        "policy": {
            "stability_seeds": (
                parse_seed_list(
                    args.seeds
                )
            ),
            "n_live": args.n_live,
            "tol": args.tol,
            "num_inner_steps": (
                args.num_inner_steps
            ),
            "num_delete_ratio": (
                args.num_delete_ratio
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
            "min_support_fraction": (
                args.min_support_fraction
            ),
            "min_median_delta_margin": (
                args.min_median_delta_margin
            ),
        },
        "decision": asdict(
            decision
        ),
        "new_seed_ranking": asdict(
            new_ranking
        ),
        "combined_ranking": asdict(
            combined_ranking
        ),
        "original_confirmation_runs": [
            asdict(run)
            for run in original_runs
        ],
        "new_stability_runs": [
            asdict(run)
            for run in new_runs
        ],
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

    consensus_path = (
        output_dir
        / "consensus.csv"
    )

    fields = [
        "window_id",
        "f_min",
        "f_max",
        "selected_k",
        "new_seed_best_k",
        "new_selected_support",
        "new_n_seeds",
        "combined_best_k",
        "combined_selected_support",
        "combined_n_seeds",
        "combined_margin",
        "status",
        "next_action",
        "reason",
    ]

    with consensus_path.open(
        "w",
        newline="",
    ) as handle:
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
                "f_min": f_min,
                "f_max": f_max,
                "selected_k": (
                    selected_k
                ),
                "new_seed_best_k": (
                    new_ranking.best_k
                ),
                "new_selected_support": (
                    new_ranking
                    .selected_k_support
                ),
                "new_n_seeds": (
                    new_ranking.n_seeds
                ),
                "combined_best_k": (
                    combined_ranking.best_k
                ),
                "combined_selected_support": (
                    combined_ranking
                    .selected_k_support
                ),
                "combined_n_seeds": (
                    combined_ranking.n_seeds
                ),
                "combined_margin": (
                    combined_ranking
                    .margin_to_runner_up
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

    write_ranking_csv(
        output_dir
        / "new_seed_evidence.csv",
        args.window_id,
        f_min,
        f_max,
        selected_k,
        new_ranking,
    )

    write_ranking_csv(
        output_dir
        / "combined_evidence.csv",
        args.window_id,
        f_min,
        f_max,
        selected_k,
        combined_ranking,
    )


def main() -> int:
    args = (
        build_parser()
        .parse_args()
    )

    stability_seeds = (
        parse_seed_list(
            args.seeds
        )
    )

    model_state = (
        load_model_order_state(
            args
        )
    )

    selected_k = int(
        model_state[
            "decision"
        ][
            "selected_k"
        ]
    )

    f_min = float(
        model_state[
            "f_min"
        ]
    )

    f_max = float(
        model_state[
            "f_max"
        ]
    )

    safety_k_max = int(
        model_state.get(
            "policy",
            {},
        ).get(
            "safety_k_max",
            selected_k + 1,
        )
    )

    if selected_k <= 1:
        raise RuntimeError(
            "Stability workflow requires "
            "an interior selected K > 1."
        )

    if selected_k >= safety_k_max:
        raise RuntimeError(
            f"Selected K={selected_k} is at the "
            f"safety boundary K={safety_k_max}; "
            "return to model-order search first."
        )

    ks = [
        selected_k - 1,
        selected_k,
        selected_k + 1,
    ]

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

    original_runs = (
        original_confirmation_runs(
            model_state,
            ks,
        )
    )

    original_seeds = {
        run.seed
        for run in original_runs
    }

    overlap = (
        original_seeds
        & set(stability_seeds)
    )

    if overlap:
        raise ValueError(
            "Stability seeds must be new independent "
            f"seeds. Overlap found: "
            f"{sorted(overlap)}"
        )

    new_runs: list[
        RunResult
    ] = []

    for k in ks:
        for seed in stability_seeds:
            run = ensure_stability_run(
                args,
                output_dir,
                f_min,
                f_max,
                k,
                seed,
            )

            new_runs.append(
                run
            )

    if args.dry_run:
        return 0

    new_ranking = (
        build_ranking_diagnostic(
            new_runs,
            ks,
            selected_k,
        )
    )

    combined_runs = [
        *original_runs,
        *new_runs,
    ]

    combined_ranking = (
        build_ranking_diagnostic(
            combined_runs,
            ks,
            selected_k,
        )
    )

    decision = (
        decide_consensus(
            selected_k,
            new_ranking,
            combined_ranking,
            args.min_support_fraction,
            args.min_median_delta_margin,
        )
    )

    write_outputs(
        output_dir,
        args,
        model_state,
        selected_k,
        ks,
        original_runs,
        new_runs,
        new_ranking,
        combined_ranking,
        decision,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
