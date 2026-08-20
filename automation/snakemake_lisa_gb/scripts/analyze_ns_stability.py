#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze seed-to-seed nested-sampling stability "
            "using existing model-order and stability runs."
        )
    )

    parser.add_argument(
        "--model-order-root",
        default="results/model_order",
    )

    parser.add_argument(
        "--stability-root",
        default="results/stability_consensus",
    )

    parser.add_argument(
        "--output-root",
        default="results/ns_diagnostics",
    )

    parser.add_argument(
        "--bands",
        default="",
        help=(
            "Optional comma-separated band list. "
            "Example: gb_0004,gb_0009"
        ),
    )

    return parser


def nested_get(
    data: dict[str, Any],
    candidates: Iterable[tuple[str, ...]],
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


def safe_float(value: Any) -> float:
    if value is None:
        return math.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def find_recursive_key(
    obj: Any,
    names: set[str],
) -> Any:
    """
    Find the first occurrence of one of `names`
    anywhere in a nested dict/list structure.
    """

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in names:
                return value

        for value in obj.values():
            found = find_recursive_key(
                value,
                names,
            )

            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = find_recursive_key(
                value,
                names,
            )

            if found is not None:
                return found

    return None


def infer_identity(
    path: Path,
    data: dict[str, Any],
) -> tuple[str | None, int | None, int | None, str]:
    metadata = (
        data.get("metadata", {})
        if isinstance(
            data.get("metadata"),
            dict,
        )
        else {}
    )

    text = str(path)

    window_id = metadata.get(
        "window_id"
    )

    if window_id is None:
        match = re.search(
            r"(gb_\d+|manual_[A-Za-z0-9_]+)",
            text,
        )

        if match:
            window_id = match.group(1)

    k = metadata.get(
        "K"
    )

    if k is None:
        match = re.search(
            r"(?:^|/)[kK](\d+)(?:/|$)",
            text,
        )

        if match:
            k = int(
                match.group(1)
            )

    seed = metadata.get(
        "seed"
    )

    if seed is None:
        match = re.search(
            r"(?:^|/)seed(-?\d+)(?:/|$)",
            text,
        )

        if match:
            seed = int(
                match.group(1)
            )

    if "stability_consensus" in text:
        origin = "stability_consensus"

    elif "/highres/" in text:
        origin = "model_order_highres"

    elif "/search/" in text:
        origin = "model_order_search"

    elif "static_ns" in text:
        origin = "static_ns"

    elif "adaptive_k" in text:
        origin = "adaptive_k"

    else:
        origin = str(
            metadata.get(
                "origin",
                metadata.get(
                    "fidelity",
                    "unknown",
                ),
            )
        )

    return (
        None
        if window_id is None
        else str(window_id),
        safe_int(k),
        safe_int(seed),
        origin,
    )


def extract_log_tail_metrics(
    summary_path: Path,
) -> dict[str, float]:
    """
    Parse the final NSS progress line from the run.log
    located next to summary.json.

    Example:

      NSS dead points: 209300pts [15:43:42,  3.04pts/s,
      iter=2090, gap=-0.918, logZ=..., logZ_live=..., dead=...]

    The last matching progress line is retained.
    """

    run_log = (
        summary_path.parent
        / "run.log"
    )

    result = {
        "final_dead_points": math.nan,
        "final_iter": math.nan,
        "final_gap": math.nan,
        "final_pts_per_s": math.nan,
    }

    if not run_log.exists():
        return result

    pattern = re.compile(
        r"NSS dead points:\s+(\d+)pts.*?"
        r"([0-9.]+)pts/s,\s*"
        r"iter=(\d+),\s*"
        r"gap=([-+0-9.eE]+)"
    )

    last_match = None

    try:
        with run_log.open(
            "r",
            errors="replace",
        ) as handle:
            for line in handle:
                match = pattern.search(
                    line
                )

                if match:
                    last_match = match

    except OSError:
        return result

    if last_match is None:
        return result

    result[
        "final_dead_points"
    ] = float(
        last_match.group(1)
    )

    result[
        "final_pts_per_s"
    ] = float(
        last_match.group(2)
    )

    result[
        "final_iter"
    ] = float(
        last_match.group(3)
    )

    result[
        "final_gap"
    ] = float(
        last_match.group(4)
    )

    return result


def extract_metrics(
    path: Path,
) -> dict[str, Any]:
    data = json.loads(
        path.read_text()
    )

    (
        window_id,
        k,
        seed,
        origin,
    ) = infer_identity(
        path,
        data,
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
            ("logZ_std",),
            ("logz_std",),
            ("results", "logZ_err"),
            ("results", "logz_err"),
            ("results", "logZ_std"),
            ("results", "logz_std"),
            ("summary", "logZ_err"),
            ("summary", "logz_err"),
            ("summary", "logZ_std"),
            ("summary", "logz_std"),
        ],
    )

    runtime = find_recursive_key(
        data,
        {
            "runtime",
            "runtime_s",
            "runtime_seconds",
            "elapsed",
            "elapsed_seconds",
            "walltime",
            "wall_time",
            "wall_time_s",
        },
    )

    n_eval = find_recursive_key(
        data,
        {
            "n_eval",
            "n_evals",
            "ncall",
            "n_calls",
            "num_likelihood_evaluations",
            "n_likelihood_evaluations",
            "likelihood_evaluations",
        },
    )

    n_iter = find_recursive_key(
        data,
        {
            "n_iter",
            "n_iters",
            "n_iterations",
            "iterations",
            "num_iterations",
        },
    )

    n_batches = find_recursive_key(
        data,
        {
            "n_batches",
            "num_batches",
            "batches",
        },
    )

    max_logl = find_recursive_key(
        data,
        {
            "max_logl",
            "max_logL",
            "max_loglike",
            "max_log_likelihood",
            "maximum_log_likelihood",
            "logl_max",
            "best_logL",
            "best_loglike",
            "best_log_likelihood",
        },
    )

    acceptance = find_recursive_key(
        data,
        {
            "acceptance",
            "acceptance_rate",
            "accept_rate",
            "mean_acceptance",
        },
    )

    ess = find_recursive_key(
        data,
        {
            "ESS",
            "ess",
            "effective_sample_size",
            "n_eff",
        },
    )

    converged = find_recursive_key(
        data,
        {
            "converged",
            "success",
            "is_converged",
        },
    )

    termination = find_recursive_key(
        data,
        {
            "termination",
            "termination_reason",
            "stop_reason",
            "status",
            "message",
        },
    )

    metadata = (
        data.get("metadata", {})
        if isinstance(
            data.get("metadata"),
            dict,
        )
        else {}
    )

    log_metrics = (
        extract_log_tail_metrics(
            path
        )
    )

    runtime_value = safe_float(
        runtime
    )

    dead_points = log_metrics[
        "final_dead_points"
    ]

    if (
        math.isfinite(dead_points)
        and dead_points > 0.0
        and math.isfinite(
            runtime_value
        )
        and runtime_value > 0.0
    ):
        average_dead_points_per_s = (
            dead_points
            / runtime_value
        )

        seconds_per_dead_point = (
            runtime_value
            / dead_points
        )

    else:
        average_dead_points_per_s = (
            math.nan
        )

        seconds_per_dead_point = (
            math.nan
        )

    return {
        "window_id": window_id,
        "K": k,
        "seed": seed,
        "origin": origin,
        "summary_path": str(
            path
        ),
        "f_min": safe_float(
            metadata.get(
                "f_min"
            )
        ),
        "f_max": safe_float(
            metadata.get(
                "f_max"
            )
        ),
        "logZ": safe_float(
            logz
        ),
        "logZ_err": safe_float(
            logz_err
        ),
        "runtime": (
            runtime_value
        ),
        "runtime_hours": (
            runtime_value / 3600.0
            if math.isfinite(
                runtime_value
            )
            else math.nan
        ),
        "n_eval": safe_float(
            n_eval
        ),
        "n_iter": safe_float(
            n_iter
        ),
        "n_batches": safe_float(
            n_batches
        ),
        "max_logL": safe_float(
            max_logl
        ),
        "ess": safe_float(
            ess
        ),
        "acceptance": safe_float(
            acceptance
        ),
        "converged": (
            ""
            if converged is None
            else str(
                converged
            )
        ),
        "termination": (
            ""
            if termination is None
            else str(
                termination
            )
        ),
        "final_dead_points": (
            log_metrics[
                "final_dead_points"
            ]
        ),
        "final_iter": (
            log_metrics[
                "final_iter"
            ]
        ),
        "final_gap": (
            log_metrics[
                "final_gap"
            ]
        ),
        "final_pts_per_s": (
            log_metrics[
                "final_pts_per_s"
            ]
        ),
        "average_dead_points_per_s": (
            average_dead_points_per_s
        ),
        "seconds_per_dead_point": (
            seconds_per_dead_point
        ),
    }


def collect_summaries(
    roots: list[Path],
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    seen_paths: set[
        Path
    ] = set()

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob(
            "summary.json"
        ):
            resolved = (
                path.resolve()
            )

            if resolved in seen_paths:
                continue

            seen_paths.add(
                resolved
            )

            try:
                row = extract_metrics(
                    path
                )

            except Exception as exc:
                print(
                    f"[skip] "
                    f"{path}: "
                    f"{exc}"
                )
                continue

            if (
                row["window_id"]
                is None
                or row["K"]
                is None
                or row["seed"]
                is None
                or not math.isfinite(
                    row["logZ"]
                )
            ):
                continue

            rows.append(
                row
            )

    return rows


def select_preferred_runs(
    rows: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    Avoid duplicate copies of exactly the same
    (band, K, seed).

    Prefer stability runs over model-order highres,
    and highres over search.
    """

    priority = {
        "stability_consensus": 4,
        "model_order_highres": 3,
        "static_ns": 2,
        "adaptive_k": 2,
        "model_order_search": 1,
        "unknown": 0,
    }

    selected: dict[
        tuple[
            str,
            int,
            int,
        ],
        dict[str, Any],
    ] = {}

    for row in rows:
        key = (
            row[
                "window_id"
            ],
            row[
                "K"
            ],
            row[
                "seed"
            ],
        )

        old = selected.get(
            key
        )

        if old is None:
            selected[
                key
            ] = row
            continue

        old_priority = (
            priority.get(
                old[
                    "origin"
                ],
                0,
            )
        )

        new_priority = (
            priority.get(
                row[
                    "origin"
                ],
                0,
            )
        )

        if (
            new_priority
            > old_priority
        ):
            selected[
                key
            ] = row

    return list(
        selected.values()
    )


def build_seed_rankings(
    rows: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    grouped: dict[
        tuple[
            str,
            int,
        ],
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for row in rows:
        grouped[
            (
                row[
                    "window_id"
                ],
                row[
                    "seed"
                ],
            )
        ].append(
            row
        )

    output: list[
        dict[str, Any]
    ] = []

    for (
        window_id,
        seed,
    ), seed_rows in sorted(
        grouped.items()
    ):
        seed_rows = [
            row
            for row
            in seed_rows
            if math.isfinite(
                row[
                    "logZ"
                ]
            )
        ]

        if not seed_rows:
            continue

        best_row = max(
            seed_rows,
            key=lambda row: (
                row[
                    "logZ"
                ]
            ),
        )

        best_logz = (
            best_row[
                "logZ"
            ]
        )

        for row in sorted(
            seed_rows,
            key=lambda item: (
                item[
                    "K"
                ]
            ),
        ):
            output.append(
                {
                    **row,
                    "seed_best_K": (
                        best_row[
                            "K"
                        ]
                    ),
                    "delta_logZ": (
                        row[
                            "logZ"
                        ]
                        - best_logz
                    ),
                    "is_seed_best": int(
                        row[
                            "K"
                        ]
                        == best_row[
                            "K"
                        ]
                    ),
                }
            )

    return output


def build_band_summary(
    ranked_rows: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    by_band: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for row in ranked_rows:
        by_band[
            row[
                "window_id"
            ]
        ].append(
            row
        )

    summaries: list[
        dict[str, Any]
    ] = []

    for (
        window_id,
        rows,
    ) in sorted(
        by_band.items()
    ):
        seeds = sorted(
            {
                row[
                    "seed"
                ]
                for row
                in rows
            }
        )

        ks = sorted(
            {
                row[
                    "K"
                ]
                for row
                in rows
            }
        )

        best_k_by_seed: dict[
            int,
            int,
        ] = {}

        for seed in seeds:
            seed_rows = [
                row
                for row
                in rows
                if row[
                    "seed"
                ]
                == seed
            ]

            if seed_rows:
                best_k_by_seed[
                    seed
                ] = (
                    seed_rows[
                        0
                    ][
                        "seed_best_K"
                    ]
                )

        counts: dict[
            int,
            int,
        ] = defaultdict(
            int
        )

        for best_k in (
            best_k_by_seed.values()
        ):
            counts[
                best_k
            ] += 1

        majority_k = (
            max(
                counts,
                key=counts.get,
            )
            if counts
            else None
        )

        majority_support = (
            counts[
                majority_k
            ]
            if majority_k
            is not None
            else 0
        )

        summaries.append(
            {
                "window_id": (
                    window_id
                ),
                "n_seeds": len(
                    best_k_by_seed
                ),
                "seeds": (
                    " ".join(
                        map(
                            str,
                            sorted(
                                best_k_by_seed
                            ),
                        )
                    )
                ),
                "Ks_seen": (
                    " ".join(
                        map(
                            str,
                            ks,
                        )
                    )
                ),
                "majority_best_K": (
                    ""
                    if majority_k
                    is None
                    else majority_k
                ),
                "majority_support": (
                    majority_support
                ),
                "support_fraction": (
                    0.0
                    if not
                    best_k_by_seed
                    else (
                        majority_support
                        / len(
                            best_k_by_seed
                        )
                    )
                ),
                "seed_best_K": (
                    " ".join(
                        (
                            f"{seed}:"
                            f"{best_k_by_seed[seed]}"
                        )
                        for seed
                        in sorted(
                            best_k_by_seed
                        )
                    )
                ),
            }
        )

    return summaries


def write_csv(
    path: Path,
    rows: list[
        dict[str, Any]
    ],
) -> None:
    if not rows:
        path.write_text(
            ""
        )
        return

    fields = list(
        rows[
            0
        ].keys()
    )

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = (
            csv.DictWriter(
                handle,
                fieldnames=fields,
                extrasaction=(
                    "ignore"
                ),
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def print_band_report(
    ranked_rows: list[
        dict[str, Any]
    ],
) -> None:
    by_band: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for row in ranked_rows:
        by_band[
            row[
                "window_id"
            ]
        ].append(
            row
        )

    for (
        window_id,
        rows,
    ) in sorted(
        by_band.items()
    ):
        print()

        print(
            "=" * 100
        )

        print(
            window_id
        )

        print(
            "=" * 100
        )

        seeds = sorted(
            {
                row[
                    "seed"
                ]
                for row
                in rows
            }
        )

        for seed in seeds:
            seed_rows = sorted(
                [
                    row
                    for row
                    in rows
                    if row[
                        "seed"
                    ]
                    == seed
                ],
                key=lambda row: (
                    row[
                        "K"
                    ]
                ),
            )

            if not seed_rows:
                continue

            best_k = (
                seed_rows[
                    0
                ][
                    "seed_best_K"
                ]
            )

            print(
                f"\nseed {seed}: "
                f"best K={best_k}"
            )

            for row in seed_rows:
                pieces = [
                    (
                        f"K="
                        f"{row['K']:2d}"
                    ),
                    (
                        f"logZ="
                        f"{row['logZ']:.3f}"
                    ),
                    (
                        f"delta="
                        f"{row['delta_logZ']:.3f}"
                    ),
                ]

                if math.isfinite(
                    row[
                        "logZ_err"
                    ]
                ):
                    pieces.append(
                        (
                            f"err="
                            f"{row['logZ_err']:.3f}"
                        )
                    )

                if math.isfinite(
                    row[
                        "runtime_hours"
                    ]
                ):
                    pieces.append(
                        (
                            f"runtime="
                            f"{row['runtime_hours']:.2f}h"
                        )
                    )

                if math.isfinite(
                    row[
                        "final_dead_points"
                    ]
                ):
                    pieces.append(
                        (
                            f"dead="
                            f"{row['final_dead_points']:.0f}"
                        )
                    )

                if math.isfinite(
                    row[
                        "final_iter"
                    ]
                ):
                    pieces.append(
                        (
                            f"iter="
                            f"{row['final_iter']:.0f}"
                        )
                    )

                if math.isfinite(
                    row[
                        "average_dead_points_per_s"
                    ]
                ):
                    pieces.append(
                        (
                            f"avg_rate="
                            f"{row['average_dead_points_per_s']:.2f}"
                            "pts/s"
                        )
                    )

                if math.isfinite(
                    row[
                        "final_pts_per_s"
                    ]
                ):
                    pieces.append(
                        (
                            f"final_rate="
                            f"{row['final_pts_per_s']:.2f}"
                            "pts/s"
                        )
                    )

                if math.isfinite(
                    row[
                        "seconds_per_dead_point"
                    ]
                ):
                    pieces.append(
                        (
                            f"sec/pt="
                            f"{row['seconds_per_dead_point']:.4f}"
                        )
                    )

                if math.isfinite(
                    row[
                        "final_gap"
                    ]
                ):
                    pieces.append(
                        (
                            f"gap="
                            f"{row['final_gap']:.3f}"
                        )
                    )

                if math.isfinite(
                    row[
                        "max_logL"
                    ]
                ):
                    pieces.append(
                        (
                            f"best_logL="
                            f"{row['max_logL']:.3f}"
                        )
                    )

                if math.isfinite(
                    row[
                        "ess"
                    ]
                ):
                    pieces.append(
                        (
                            f"ESS="
                            f"{row['ess']:.0f}"
                        )
                    )

                print(
                    "  "
                    + "  ".join(
                        pieces
                    )
                )


def main() -> int:
    args = (
        build_parser()
        .parse_args()
    )

    output_root = (
        Path(
            args.output_root
        ).resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    requested_bands = {
        item.strip()
        for item
        in args.bands.split(
            ","
        )
        if item.strip()
    }

    rows = collect_summaries(
        [
            Path(
                args.model_order_root
            ).resolve(),
            Path(
                args.stability_root
            ).resolve(),
        ]
    )

    if requested_bands:
        rows = [
            row
            for row
            in rows
            if row[
                "window_id"
            ]
            in requested_bands
        ]

    print(
        f"Found "
        f"{len(rows)} "
        "valid summary files."
    )

    preferred_rows = (
        select_preferred_runs(
            rows
        )
    )

    print(
        f"Using "
        f"{len(preferred_rows)} "
        "unique "
        "(band, K, seed) runs."
    )

    ranked_rows = (
        build_seed_rankings(
            preferred_rows
        )
    )

    band_summary = (
        build_band_summary(
            ranked_rows
        )
    )

    write_csv(
        output_root
        / "runs.csv",
        preferred_rows,
    )

    write_csv(
        output_root
        / "seed_rankings.csv",
        ranked_rows,
    )

    write_csv(
        output_root
        / "band_summary.csv",
        band_summary,
    )

    print_band_report(
        ranked_rows
    )

    print()

    print(
        "Wrote:"
    )

    print(
        output_root
        / "runs.csv"
    )

    print(
        output_root
        / "seed_rankings.csv"
    )

    print(
        output_root
        / "band_summary.csv"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
