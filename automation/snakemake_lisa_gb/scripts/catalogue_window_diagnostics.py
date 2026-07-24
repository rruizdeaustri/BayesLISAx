#!/usr/bin/env python3
"""Sangria catalogue diagnostics and recovered-source truth matching.

For each recovered frequency, the primary catalogue association is the
highest-approximate-SNR source within a configurable number of Fourier bins.
The nearest-frequency source is also recorded separately for diagnostics.

Truth information is diagnostic only and must not be used for candidate
selection.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from sangria_catalog import (
    concatenate_catalogues,
    lisa_gb_approx_snr,
    read_catalogues,
)


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_f0(value: object) -> list[float]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            return [float(text)]
        except ValueError:
            return []
    try:
        array = np.asarray(parsed, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return []
    return [float(item) for item in array if np.isfinite(item)]


def as_flat_array(value: object, dtype: Any = None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=dtype).reshape(-1)


def decode_source_type(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def load_recovered_frequencies(
    highres_csv: str,
) -> dict[str, list[dict[str, object]]]:
    recovered: dict[str, list[dict[str, object]]] = {}
    if not highres_csv or not Path(highres_csv).exists():
        return recovered
    with open(highres_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            window_id = row["window_id"]
            frequencies = parse_f0(
                row.get("all_best_f0") or row.get("best_f0", "")
            )
            for slot_index, frequency in enumerate(frequencies):
                recovered.setdefault(window_id, []).append(
                    {
                        "recovered_f0": frequency,
                        "recovered_seed": row.get("seed", ""),
                        "recovered_slot_index": slot_index,
                    }
                )
    return recovered


def catalogue_payload(
    catalogue: dict[str, object],
    index: int,
) -> dict[str, object]:
    payload: dict[str, object] = {"catalogue_index": int(index)}
    for key, value in catalogue.items():
        if key == "source_type":
            continue
        array = as_flat_array(value)
        if array is None or index >= len(array):
            continue
        item = array[index]
        if np.issubdtype(np.asarray(item).dtype, np.number):
            payload[f"catalogue_{key}"] = finite_float(item)
        else:
            payload[f"catalogue_{key}"] = str(item)
    source_type = catalogue.get("source_type")
    if source_type is not None:
        source_array = as_flat_array(source_type)
        if source_array is not None and index < len(source_array):
            payload["catalogue_source_type"] = decode_source_type(
                source_array[index]
            )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--highres-csv", default="")
    parser.add_argument("--tobs", type=float, required=True)
    parser.add_argument("--margin-hz", type=float, required=True)
    parser.add_argument("--match-radius-bins", type=float, default=5.0)
    parser.add_argument("--snr-thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-matches", default="")
    args = parser.parse_args()

    if args.tobs <= 0:
        raise ValueError("--tobs must be positive")
    if args.match_radius_bins <= 0:
        raise ValueError("--match-radius-bins must be positive")

    catalogue = concatenate_catalogues(read_catalogues(args.h5_path))
    frequency = as_flat_array(catalogue["Frequency"], dtype=float)
    amplitude = as_flat_array(catalogue["Amplitude"], dtype=float)
    latitude = as_flat_array(catalogue.get("EclipticLatitude"), dtype=float)
    inclination = as_flat_array(catalogue.get("Inclination"), dtype=float)
    approx_snr = as_flat_array(
        lisa_gb_approx_snr(
            frequency,
            amplitude,
            args.tobs,
            latitude,
            inclination,
        ),
        dtype=float,
    )

    source_type = catalogue.get("source_type")
    if source_type is None:
        source_type_array = np.array(["unknown"] * len(frequency), dtype=object)
    else:
        source_type_array = as_flat_array(source_type)

    n_sources = min(
        len(frequency),
        len(amplitude),
        len(approx_snr),
        len(source_type_array),
    )
    frequency = frequency[:n_sources]
    amplitude = amplitude[:n_sources]
    approx_snr = approx_snr[:n_sources]
    source_type_array = source_type_array[:n_sources]

    selected_windows = read_csv_rows(args.selected_csv)
    recovered = load_recovered_frequencies(args.highres_csv)

    fourier_bin_hz = 1.0 / args.tobs
    match_radius_hz = args.match_radius_bins * fourier_bin_hz

    summary_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []

    for window in selected_windows:
        window_id = window["window_id"]
        f_min = float(window["f_min"])
        f_max = float(window["f_max"])
        catalogue_f_min = f_min - args.margin_hz
        catalogue_f_max = f_max + args.margin_hz

        in_window = (
            (frequency >= catalogue_f_min)
            & (frequency <= catalogue_f_max)
        )
        window_indices = np.flatnonzero(in_window)
        order = (
            window_indices[np.argsort(approx_snr[window_indices])[::-1]]
            if window_indices.size
            else np.array([], dtype=int)
        )

        top_sources = [
            {
                "rank": rank + 1,
                "frequency": float(frequency[index]),
                "snr": float(approx_snr[index]),
                "source_type": decode_source_type(source_type_array[index]),
            }
            for rank, index in enumerate(order[: args.top_n])
        ]

        summary_row: dict[str, object] = {
            "window_id": window_id,
            "f_min": f_min,
            "f_max": f_max,
            "catalogue_f_min": catalogue_f_min,
            "catalogue_f_max": catalogue_f_max,
            "total_catalogue_sources": int(window_indices.size),
            "fourier_bin_hz": fourier_bin_hz,
            "match_radius_bins": args.match_radius_bins,
            "match_radius_hz": match_radius_hz,
            "top_sources_json": json.dumps(top_sources),
        }
        for threshold in args.snr_thresholds:
            summary_row[f"count_snr_gt_{threshold:g}"] = int(
                np.sum(approx_snr[window_indices] > threshold)
            )
        summary_rows.append(summary_row)

        for recovered_source in recovered.get(window_id, []):
            recovered_f0 = float(recovered_source["recovered_f0"])
            if not window_indices.size:
                continue

            offsets = np.abs(frequency[window_indices] - recovered_f0)
            nearest_index = int(window_indices[int(np.argmin(offsets))])
            candidate_indices = window_indices[offsets <= match_radius_hz]

            if candidate_indices.size:
                primary_index = int(
                    candidate_indices[int(np.argmax(approx_snr[candidate_indices]))]
                )
                match_status = "highest_snr_within_radius"
            else:
                primary_index = nearest_index
                match_status = "nearest_fallback_no_source_within_radius"

            primary_delta_f = recovered_f0 - frequency[primary_index]
            nearest_delta_f = recovered_f0 - frequency[nearest_index]

            primary_delta_f_bins = primary_delta_f / fourier_bin_hz
            abs_primary_delta_f_bins = abs(primary_delta_f_bins)

            if abs_primary_delta_f_bins <= 2.0:
                match_quality = "strong"
            elif abs_primary_delta_f_bins <= args.match_radius_bins:
                match_quality = "tentative"
            else:
                match_quality = "unmatched"
            
            row: dict[str, object] = {
                "window_id": window_id,
                "recovered_seed": recovered_source["recovered_seed"],
                "recovered_slot_index": recovered_source["recovered_slot_index"],
                "recovered_f0": recovered_f0,
                "match_status": match_status,
                "match_radius_bins": args.match_radius_bins,
                "match_radius_hz": match_radius_hz,
                "catalogue_frequency": float(frequency[primary_index]),
                "delta_f": float(primary_delta_f),
                "delta_f_bins": float(primary_delta_f_bins),
                "approx_snr": float(approx_snr[primary_index]),
                "source_type": decode_source_type(source_type_array[primary_index]),
                "primary_catalogue_index": primary_index,
                "primary_catalogue_frequency": float(frequency[primary_index]),
                "primary_delta_f_hz": float(primary_delta_f),
                "primary_delta_f_bins": float(primary_delta_f_bins),
                "primary_approx_snr": float(approx_snr[primary_index]),
                "primary_source_type": decode_source_type(source_type_array[primary_index]),
                "nearest_catalogue_index": nearest_index,
                "nearest_catalogue_frequency": float(frequency[nearest_index]),
                "nearest_delta_f_hz": float(nearest_delta_f),
                "nearest_delta_f_bins": float(nearest_delta_f / fourier_bin_hz),
                "nearest_approx_snr": float(approx_snr[nearest_index]),
                "nearest_source_type": decode_source_type(source_type_array[nearest_index]),
                "n_catalogue_sources_within_radius": int(candidate_indices.size),
                "match_quality": match_quality,
            }
            row.update(catalogue_payload(catalogue, primary_index))
            match_rows.append(row)

    summary_fields = (
        [
            "window_id",
            "f_min",
            "f_max",
            "catalogue_f_min",
            "catalogue_f_max",
            "total_catalogue_sources",
            "fourier_bin_hz",
            "match_radius_bins",
            "match_radius_hz",
        ]
        + [f"count_snr_gt_{threshold:g}" for threshold in args.snr_thresholds]
        + ["top_sources_json"]
    )

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    if args.out_matches:
        Path(args.out_matches).parent.mkdir(parents=True, exist_ok=True)
        base_match_fields = [
            "window_id",
            "recovered_seed",
            "recovered_slot_index",
            "recovered_f0",
            "match_status",
            "match_quality",
            "match_radius_bins",
            "match_radius_hz",
            "catalogue_frequency",
            "delta_f",
            "delta_f_bins",
            "approx_snr",
            "source_type",
            "primary_catalogue_index",
            "primary_catalogue_frequency",
            "primary_delta_f_hz",
            "primary_delta_f_bins",
            "primary_approx_snr",
            "primary_source_type",
            "nearest_catalogue_index",
            "nearest_catalogue_frequency",
            "nearest_delta_f_hz",
            "nearest_delta_f_bins",
            "nearest_approx_snr",
            "nearest_source_type",
            "n_catalogue_sources_within_radius",
        ]
        extra_fields = sorted(
            {
                key
                for row in match_rows
                for key in row
                if key not in base_match_fields
            }
        )
        match_fields = base_match_fields + extra_fields
        with open(args.out_matches, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=match_fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(match_rows)


if __name__ == "__main__":
    main()
