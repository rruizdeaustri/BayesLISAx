#!/usr/bin/env python3
"""Select consensus candidates and validate them against a Sangria catalogue.

This stage intentionally separates three concepts:

* ``sampling_quality``: reproducibility across independent posterior runs;
* ``statistically_selected``: Bayesian-FDR selection using posterior inclusion mass;
* ``truth_match_quality``: optional validation against the complete Sangria catalogue.

Catalogue information is never used to decide whether a candidate is statistically
selected. This keeps the same code usable for both Sangria validation and future
blind analyses.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from sangria_catalog import lisa_gb_approx_snr, read_catalogues


CATALOGUE_FIELDS = (
    "Frequency",
    "Amplitude",
    "FrequencyDerivative",
    "EclipticLatitude",
    "EclipticLongitude",
    "Inclination",
    "InitialPhase",
    "Polarization",
)


def _as_float(row: dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def read_clusters(path: str | Path, window_id: str | None = None) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if window_id is not None:
        rows = [row for row in rows if row.get("window_id") == window_id]
    return rows


def load_full_catalogue(h5_path: str, tobs: float) -> dict[str, np.ndarray]:
    """Return all available Sangria truth parameters in a common row layout."""
    chunks: dict[str, list[np.ndarray]] = {field: [] for field in CATALOGUE_FIELDS}
    source_types: list[np.ndarray] = []

    for catalogue in read_catalogues(h5_path):
        if "Frequency" not in catalogue.data:
            continue
        n = catalogue.size
        for field in CATALOGUE_FIELDS:
            values = catalogue.data.get(field)
            if values is None:
                chunks[field].append(np.full(n, np.nan, dtype=float))
            else:
                arr = np.asarray(values, dtype=float).reshape(-1)
                if arr.size != n:
                    raise ValueError(
                        f"Catalogue {catalogue.path!r} field {field!r} has "
                        f"length {arr.size}, expected {n}"
                    )
                chunks[field].append(arr)
        source_types.append(np.full(n, catalogue.source_type, dtype=object))

    if not chunks["Frequency"]:
        return {field: np.array([], dtype=float) for field in CATALOGUE_FIELDS} | {
            "source_type": np.array([], dtype=object),
            "approx_snr": np.array([], dtype=float),
        }

    out = {field: np.concatenate(parts) for field, parts in chunks.items()}
    out["source_type"] = np.concatenate(source_types)
    out["approx_snr"] = np.asarray(
        lisa_gb_approx_snr(
            out["Frequency"],
            out["Amplitude"],
            tobs,
            out["EclipticLatitude"],
            out["Inclination"],
        ),
        dtype=float,
    )
    return out


def apply_bayesian_fdr(
    rows: list[dict[str, object]],
    *,
    probability_key: str,
    q: float,
) -> None:
    """Select the largest ranked prefix whose expected false fraction is <= q."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("bfdr_q must lie in [0, 1]")

    order = sorted(
        range(len(rows)),
        key=lambda idx: float(rows[idx][probability_key]),
        reverse=True,
    )
    running_false = 0.0
    accepted_prefix = 0
    cumulative_bfdr: dict[int, float] = {}
    for rank, idx in enumerate(order, start=1):
        probability = float(np.clip(rows[idx][probability_key], 0.0, 1.0))
        running_false += 1.0 - probability
        value = running_false / rank
        cumulative_bfdr[idx] = value
        if value <= q:
            accepted_prefix = rank

    accepted = set(order[:accepted_prefix])
    for rank, idx in enumerate(order, start=1):
        probability = float(np.clip(rows[idx][probability_key], 0.0, 1.0))
        rows[idx]["selection_rank"] = rank
        rows[idx]["local_fdr"] = 1.0 - probability
        rows[idx]["cumulative_bfdr"] = cumulative_bfdr[idx]
        rows[idx]["statistically_selected"] = idx in accepted
        rows[idx]["statistical_status"] = "selected" if idx in accepted else "candidate"


def nearest_catalogue_match(
    frequency: float,
    catalogue: dict[str, np.ndarray],
    *,
    tobs: float,
    max_match_bins: float,
) -> dict[str, object]:
    frequencies = catalogue["Frequency"]
    if frequencies.size == 0 or not np.isfinite(frequency):
        return {"truth_match_quality": "unavailable", "catalogue_index": -1}

    idx = int(np.argmin(np.abs(frequencies - frequency)))
    delta_f = float(frequency - frequencies[idx])
    delta_bins = abs(delta_f) * tobs
    if delta_bins <= 1.0:
        quality = "within_one_bin"
    elif delta_bins <= max_match_bins:
        quality = "nearby"
    else:
        quality = "distant"

    result: dict[str, object] = {
        "catalogue_index": idx,
        "catalogue_source_type": str(catalogue["source_type"][idx]),
        "catalogue_frequency_hz": float(frequencies[idx]),
        "delta_f_hz": delta_f,
        "delta_f_bins": delta_bins,
        "truth_match_quality": quality,
        "catalogue_approx_snr": float(catalogue["approx_snr"][idx]),
    }
    for field in CATALOGUE_FIELDS:
        result[f"catalogue_{field}"] = float(catalogue[field][idx])
    return result


def build_candidates(
    clusters: list[dict[str, str]],
    *,
    catalogue: dict[str, np.ndarray] | None,
    tobs: float,
    bfdr_q: float,
    probability_field: str,
    max_match_bins: float,
) -> list[dict[str, object]]:
    if probability_field not in {"mean_inclusion", "median_inclusion"}:
        raise ValueError("probability_field must be mean_inclusion or median_inclusion")

    rows: list[dict[str, object]] = []
    for cluster in clusters:
        inclusion = float(np.clip(_as_float(cluster, probability_field, 0.0), 0.0, 1.0))
        row: dict[str, object] = dict(cluster)
        row["sampling_quality"] = cluster.get("quality", "unknown")
        row["posterior_inclusion_score"] = inclusion
        row["selection_probability_field"] = probability_field
        rows.append(row)

    apply_bayesian_fdr(
        rows,
        probability_key="posterior_inclusion_score",
        q=bfdr_q,
    )

    for row in rows:
        if catalogue is None:
            row.update({"truth_match_quality": "not_requested", "catalogue_index": -1})
        else:
            row.update(
                nearest_catalogue_match(
                    _as_float(row, "f0_median_hz"),
                    catalogue,
                    tobs=tobs,
                    max_match_bins=max_match_bins,
                )
            )
    return rows


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "window_id",
        "cluster_id",
        "sampling_quality",
        "statistical_status",
        "statistically_selected",
        "posterior_inclusion_score",
        "local_fdr",
        "cumulative_bfdr",
        "selection_rank",
        "f0_median_hz",
        "truth_match_quality",
        "catalogue_frequency_hz",
        "delta_f_hz",
        "delta_f_bins",
        "catalogue_approx_snr",
        "catalogue_source_type",
    ]
    all_fields = {key for row in rows for key in row}
    fields = [field for field in preferred if field in all_fields]
    fields.extend(sorted(all_fields - set(fields)))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: str | Path,
    rows: list[dict[str, object]],
    *,
    window_id: str,
    bfdr_q: float,
    probability_field: str,
) -> None:
    selected = [int(row["cluster_id"]) for row in rows if row["statistically_selected"]]
    payload: dict[str, object] = {
        "window_id": window_id,
        "n_candidates": len(rows),
        "bfdr_q": bfdr_q,
        "selection_probability_field": probability_field,
        "selected_cluster_ids": selected,
        "n_statistically_selected": len(selected),
        "sampling_quality_counts": {},
        "truth_match_quality_counts": {},
        "notes": {
            "sampling_quality": "Measures reproducibility across posterior seeds.",
            "statistical_selection": "Bayesian-FDR selection from posterior inclusion scores.",
            "truth_matching": "Validation only; never used for statistical selection.",
            "catalogue_approx_snr": "Planning diagnostic, not coherent likelihood SNR.",
        },
    }
    for row in rows:
        for key, output_key in (
            ("sampling_quality", "sampling_quality_counts"),
            ("truth_match_quality", "truth_match_quality_counts"),
        ):
            value = str(row.get(key, "unknown"))
            counts = payload[output_key]
            assert isinstance(counts, dict)
            counts[value] = counts.get(value, 0) + 1

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters-csv", required=True)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--tobs", type=float, required=True)
    parser.add_argument("--h5-path", default="")
    parser.add_argument("--bfdr-q", type=float, default=0.1)
    parser.add_argument(
        "--probability-field",
        choices=("mean_inclusion", "median_inclusion"),
        default="mean_inclusion",
    )
    parser.add_argument("--max-match-bins", type=float, default=5.0)
    parser.add_argument("--out-candidates", required=True)
    parser.add_argument("--out-summary", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.tobs <= 0.0:
        raise ValueError("--tobs must be positive")

    catalogue = load_full_catalogue(args.h5_path, args.tobs) if args.h5_path else None
    clusters = read_clusters(args.clusters_csv, window_id=args.window_id)
    rows = build_candidates(
        clusters,
        catalogue=catalogue,
        tobs=args.tobs,
        bfdr_q=args.bfdr_q,
        probability_field=args.probability_field,
        max_match_bins=args.max_match_bins,
    )
    write_csv(args.out_candidates, rows)
    write_summary(
        args.out_summary,
        rows,
        window_id=args.window_id,
        bfdr_q=args.bfdr_q,
        probability_field=args.probability_field,
    )


if __name__ == "__main__":
    main()
