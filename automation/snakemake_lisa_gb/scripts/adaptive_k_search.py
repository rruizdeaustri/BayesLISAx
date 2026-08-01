#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

@dataclass
class KSummary:
    k: int
    seeds: list[int]
    logz_values: list[float]
    logz_err_values: list[float]

    median_logz: float
    mean_logz: float
    std_logz: float
    mad_logz: float
    min_logz: float
    max_logz: float

    seed_deviations: list[float]
    n_consistent_seeds: int
    max_seed_deviation: float
    has_catastrophic_outlier: bool
    stability_status: str
    stable: bool
    

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive fixed-K nested-sampling search")
    p.add_argument("--window-id", required=True)
    p.add_argument("--f-min", type=float, required=True)
    p.add_argument("--f-max", type=float, required=True)
    p.add_argument("--base-json", required=True)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--workflow-root", default=".")
    p.add_argument("--output-root", default="results/adaptive_k")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--problem-factory", default="jax_samplers.problems.lisa_gb_transdim_problem:make")

    p.add_argument("--initial-k-max", type=int, default=4)
    p.add_argument("--safety-k-max", type=int, default=20)
    p.add_argument("--confirm-lower-points", type=int, default=2)
    p.add_argument("--min-peak-drop-logz", type=float, default=10.0)
    p.add_argument("--spread-factor", type=float, default=2.0)

    p.add_argument(
        "--seed-consistency-logz",
        type=float,
        default=100.0,
        help="Maximum distance from the median for a seed to be considered consistent.",
    )
    p.add_argument(
        "--min-consistent-seeds",
        type=int,
        default=2,
        help="Minimum number of mutually consistent seeds.",
    )
    p.add_argument(
        "--catastrophic-outlier-logz",
        type=float,
        default=500.0,
        help="Deviation from the median regarded as a catastrophic seed failure.",
    )
    
    p.add_argument("--initial-seeds", default="0,11,22")
    p.add_argument("--extra-seeds", default="33,44,55")
    p.add_argument("--max-seeds-per-k", type=int, default=5)

    p.add_argument("--n-live", type=int, default=500)
    p.add_argument("--tol", type=float, default=2.0)
    p.add_argument("--num-inner-steps", type=int, default=64)
    p.add_argument("--num-delete-ratio", type=float, default=0.1)
    p.add_argument("--initial-num-steps", type=int, default=16)
    p.add_argument("--refinement-num-steps", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_seed_list(raw: str) -> list[int]:
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"Invalid seed list: {raw}")
    return seeds


def run_command(command: Sequence[str], dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def nested_get(data: dict[str, Any], candidates: Iterable[Sequence[str]]) -> Any:
    for path in candidates:
        value: Any = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            return value
    return None


def extract_logz(summary_path: Path) -> tuple[float, float]:
    data = json.loads(summary_path.read_text())
    logz = nested_get(data, [("logZ",), ("logz",), ("results", "logZ"), ("results", "logz"),
                             ("summary", "logZ"), ("summary", "logz"), ("metrics", "logZ"),
                             ("metrics", "logz")])
    err = nested_get(data, [("logZ_err",), ("logz_err",), ("results", "logZ_err"),
                            ("results", "logz_err"), ("summary", "logZ_err"),
                            ("summary", "logz_err"), ("metrics", "logZ_err"),
                            ("metrics", "logz_err")])
    if logz is None:
        raise KeyError(f"Could not find logZ in {summary_path}; adapt extract_logz() to the JSON schema")
    return float(logz), float("nan") if err is None else float(err)


def mad(values: Sequence[float]) -> float:
    med = statistics.median(values)
    return float(statistics.median(abs(x - med) for x in values))


def summarize_k(
    k: int,
    results: dict[int, tuple[float, float]],
    seed_consistency_logz: float,
    min_consistent_seeds: int,
    catastrophic_outlier_logz: float,
) -> KSummary:
    seeds = sorted(results)
    logz = [results[s][0] for s in seeds]
    errs = [results[s][1] for s in seeds]

    median_logz = float(statistics.median(logz))
    deviations = [abs(value - median_logz) for value in logz]

    n_consistent = sum(
        deviation <= seed_consistency_logz
        for deviation in deviations
    )

    max_deviation = max(deviations)
    has_catastrophic_outlier = (
        max_deviation > catastrophic_outlier_logz
    )

    if n_consistent == len(logz):
        stability_status = "all_seeds_consistent"
        stable = True
    elif n_consistent >= min_consistent_seeds:
        stability_status = "majority_consistent_with_outlier"
        stable = True
    else:
        stability_status = "unstable"
        stable = False

    return KSummary(
        k=k,
        seeds=seeds,
        logz_values=logz,
        logz_err_values=errs,
        median_logz=median_logz,
        mean_logz=float(statistics.fmean(logz)),
        std_logz=float(statistics.stdev(logz)) if len(logz) > 1 else 0.0,
        mad_logz=mad(logz),
        min_logz=float(min(logz)),
        max_logz=float(max(logz)),
        seed_deviations=deviations,
        n_consistent_seeds=n_consistent,
        max_seed_deviation=float(max_deviation),
        has_catastrophic_outlier=has_catastrophic_outlier,
        stability_status=stability_status,
        stable=stable,
    )

def write_metadata(path: Path, args: argparse.Namespace, k: int, seed: int) -> None:
    data = json.loads(path.read_text())
    data["metadata"] = {
        "window_id": args.window_id,
        "f_min": args.f_min,
        "f_max": args.f_max,
        "K": k,
        "seed": seed,
        "num_delete_ratio": args.num_delete_ratio,
        "phase": "adaptive_scan",
    }
    path.write_text(json.dumps(data, indent=2))


def ensure_run(args: argparse.Namespace, k: int, seed: int, window_dir: Path) -> tuple[float, float]:
    config_path = window_dir / "configs" / f"k{k}.json"
    summary_path = window_dir / "scan" / f"k{k}" / f"seed{seed}" / "summary.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    workflow_root = Path(args.workflow_root).resolve()
    make_config = workflow_root / "scripts" / "make_band_config.py"
    benchmark = workflow_root / "scripts" / "run_benchmark.py"

    if args.force or not config_path.exists():
        run_command([args.python, str(make_config), "--base-json", str(Path(args.base_json).resolve()),
                     "--out-json", str(config_path), "--f-min", repr(args.f_min), "--f-max",
                     repr(args.f_max), "--kmax", str(k)], args.dry_run)

    if args.force or not summary_path.exists():
        run_command([args.python, str(benchmark), "--python", args.python, "--repo-root",
                     str(Path(args.repo_root).resolve()), "--problem-factory", args.problem_factory,
                     "--config-json", str(config_path), "--algo", "ns", "--seed", str(seed),
                     "--n-live", str(args.n_live), "--tol", str(args.tol), "--num-inner-steps",
                     str(args.num_inner_steps), "--num-delete-ratio", str(args.num_delete_ratio),
                     "--initial-num-steps", str(args.initial_num_steps), "--refinement-num-steps",
                     str(args.refinement_num_steps), "--max-batches", str(args.max_batches),
                     "--skip-plots", "--out-json", str(summary_path)], args.dry_run)
        if not args.dry_run:
            write_metadata(summary_path, args, k, seed)

    if args.dry_run:
        return float("nan"), float("nan")
    return extract_logz(summary_path)


def find_confirmed_peak(summaries: dict[int, KSummary], args: argparse.Namespace) -> int | None:
    if len(summaries) < 3:
        return None
    tested = sorted(summaries)
    best_k = max(tested, key=lambda kk: summaries[kk].median_logz)
    if best_k in (tested[0], tested[-1]):
        return None
    later = [k for k in tested if k > best_k]
    if len(later) < args.confirm_lower_points or not summaries[best_k].stable:
        return None
    peak = summaries[best_k]
    for k in later[:args.confirm_lower_points]:
        point = summaries[k]
        required_drop = max(args.min_peak_drop_logz,
                            args.spread_factor * max(peak.mad_logz, point.mad_logz))
        if peak.median_logz - point.median_logz <= required_drop:
            return None
    return best_k


def write_outputs(window_dir: Path, args: argparse.Namespace, summaries: dict[int, KSummary],
                  status: str, next_action: str, selected_k: int | None, next_k: int | None) -> None:
    best_k = max(summaries, key=lambda kk: summaries[kk].median_logz) if summaries else None
    state = {
        "window_id": args.window_id,
        "f_min": args.f_min,
        "f_max": args.f_max,
        "tested_k": sorted(summaries),
        "selected_k": selected_k,
        "current_best_k": best_k,
        "best_is_boundary": bool(summaries) and best_k == max(summaries),
        "status": status,
        "next_action": next_action,
        "next_k": next_k,
        "k_summaries": {str(k): asdict(v) for k, v in sorted(summaries.items())},
    }
    (window_dir / "state.json").write_text(json.dumps(state, indent=2))

    with (window_dir / "k_summary.csv").open("w", newline="") as handle:

        fields = [
            "window_id",
            "f_min",
            "f_max",
            "K",
            "n_seeds",
            "seeds",
            "median_logZ",
            "mean_logZ",
            "std_logZ",
            "mad_logZ",
            "min_logZ",
            "max_logZ",
            "n_consistent_seeds",
            "max_seed_deviation",
            "has_catastrophic_outlier",
            "stability_status",
            "stable",
        ]
        
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for k, s in sorted(summaries.items()):

            writer.writerow(
                {
                    "window_id": args.window_id,
                    "f_min": args.f_min,
                    "f_max": args.f_max,
                    "K": k,
                    "n_seeds": len(s.seeds),
                    "seeds": " ".join(map(str, s.seeds)),
                    "median_logZ": s.median_logz,
                    "mean_logZ": s.mean_logz,
                    "std_logZ": s.std_logz,
                    "mad_logZ": s.mad_logz,
                    "min_logZ": s.min_logz,
                    "max_logZ": s.max_logz,
                    "n_consistent_seeds": s.n_consistent_seeds,
                    "max_seed_deviation": s.max_seed_deviation,
                    "has_catastrophic_outlier": s.has_catastrophic_outlier,
                    "stability_status": s.stability_status,
                    "stable": s.stable,
                }
            )
            

def main() -> int:
    args = parse_args()
    if args.initial_k_max < 2 or args.safety_k_max <= args.initial_k_max:
        raise ValueError("Require initial_k_max >= 2 and safety_k_max > initial_k_max")

    initial_seeds = parse_seed_list(args.initial_seeds)
    extra_seeds = [s for s in parse_seed_list(args.extra_seeds) if s not in initial_seeds]
    window_dir = Path(args.output_root).resolve() / args.window_id
    window_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[int, KSummary] = {}
    current_k_max = args.initial_k_max

    for k in range(1, args.safety_k_max + 1):
        if k > current_k_max:
            break

        seed_results = {seed: ensure_run(args, k, seed, window_dir) for seed in initial_seeds}
        if args.dry_run:
            continue

        summary = summarize_k(
            k,
            seed_results,
            args.seed_consistency_logz,
            args.min_consistent_seeds,
            args.catastrophic_outlier_logz,
        )
        
        if (
                summary.stability_status == "unstable"
                or summary.has_catastrophic_outlier
        ):
            
            for seed in extra_seeds:
                if len(seed_results) >= args.max_seeds_per_k:
                    break
                seed_results[seed] = ensure_run(args, k, seed, window_dir)

                summary = summarize_k(
                    k,
                    seed_results,
                    args.seed_consistency_logz,
                    args.min_consistent_seeds,
                    args.catastrophic_outlier_logz,
                )

                if (
                        summary.stability_status == "all_seeds_consistent"
                        and not summary.has_catastrophic_outlier
):
                    break

        summaries[k] = summary

        peak = find_confirmed_peak(summaries, args)
        if peak is not None:
            write_outputs(window_dir, args, summaries, "complete", "highres_confirm", peak, None)
            print(f"[adaptive-k] confirmed K={peak}")
            return 0

        if k == current_k_max and current_k_max < args.safety_k_max:
            current_k_max += 1
        write_outputs(window_dir, args, summaries, "searching", "run_next_k", None, current_k_max)

    if args.dry_run:
        return 0

    best_k = max(summaries, key=lambda kk: summaries[kk].median_logz)
    write_outputs(window_dir, args, summaries, "model_order_unresolved",
                  "human_review_or_raise_safety_cap", None, None)
    print(f"[adaptive-k] unresolved before K={args.safety_k_max}; best median-evidence K={best_k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
