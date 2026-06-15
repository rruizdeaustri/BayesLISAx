#!/usr/bin/env python
"""Generate padded frequency windows from Sangria/LDC GB catalogues."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from sangria_catalog import concatenate_catalogues, lisa_gb_approx_snr, read_catalogues

FIELDNAMES = (
    "band_id",
    "core_f_min",
    "core_f_max",
    "analysis_f_min",
    "analysis_f_max",
    "n_catalog_sources",
    "n_bright_sources",
    "approx_max_snr",
    "approx_median_snr",
)


def generate_core_windows(f_min: float, f_max: float, width: float, overlap: float) -> list[tuple[float, float]]:
    if f_max <= f_min:
        raise ValueError("f_max must be larger than f_min")
    if width <= 0.0:
        raise ValueError("core window width must be positive")
    if overlap < 0.0:
        raise ValueError("overlap must be non-negative")
    if overlap >= width:
        raise ValueError("overlap must be smaller than core window width")

    step = width - overlap
    windows: list[tuple[float, float]] = []
    start = f_min
    while start < f_max:
        stop = min(start + width, f_max)
        windows.append((start, stop))
        if math.isclose(stop, f_max) or stop >= f_max:
            break
        start += step
    return windows


def build_window_rows(
    h5_path: str,
    f_min: float,
    f_max: float,
    tobs: float,
    core_width: float,
    guard_width: float,
    overlap: float,
    snr_threshold: float | None,
) -> list[dict[str, object]]:
    catalogues = read_catalogues(h5_path)
    data = concatenate_catalogues(catalogues)
    freq = data["Frequency"]
    amp = data["Amplitude"]
    snr = lisa_gb_approx_snr(freq, amp, tobs, data["EclipticLatitude"], data["Inclination"])

    rows: list[dict[str, object]] = []
    for index, (core_min, core_max) in enumerate(generate_core_windows(f_min, f_max, core_width, overlap)):
        analysis_min = max(f_min, core_min - guard_width)
        analysis_max = min(f_max, core_max + guard_width)
        in_band = (freq >= analysis_min) & (freq < analysis_max)
        band_snr = snr[in_band]
        if snr_threshold is None:
            bright = np.zeros_like(band_snr, dtype=bool)
        else:
            bright = band_snr >= snr_threshold
        rows.append(
            {
                "band_id": f"gb_{index:04d}",
                "core_f_min": f"{core_min:.12g}",
                "core_f_max": f"{core_max:.12g}",
                "analysis_f_min": f"{analysis_min:.12g}",
                "analysis_f_max": f"{analysis_max:.12g}",
                "n_catalog_sources": int(np.count_nonzero(in_band)),
                "n_bright_sources": int(np.count_nonzero(bright)),
                "approx_max_snr": f"{float(np.max(band_snr)):.12g}" if band_snr.size else "0",
                "approx_median_snr": f"{float(np.median(band_snr)):.12g}" if band_snr.size else "0",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", help="Input Sangria HDF5 file")
    parser.add_argument("--f-min", type=float, required=True, help="Global minimum frequency [Hz]")
    parser.add_argument("--f-max", type=float, required=True, help="Global maximum frequency [Hz]")
    parser.add_argument("--tobs", type=float, required=True, help="Observation time [s]")
    parser.add_argument("--core-width", type=float, required=True, help="Core window width [Hz]")
    parser.add_argument("--guard-width", type=float, required=True, help="Padding added on each side [Hz]")
    parser.add_argument("--overlap", type=float, default=0.0, help="Core-window overlap [Hz]")
    parser.add_argument("--snr-threshold", type=float, default=None, help="Optional bright-source SNR threshold")
    parser.add_argument("--out-csv", required=True, help="Output window CSV")
    args = parser.parse_args()

    rows = build_window_rows(
        args.h5_path,
        args.f_min,
        args.f_max,
        args.tobs,
        args.core_width,
        args.guard_width,
        args.overlap,
        args.snr_threshold,
    )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Sangria frequency windows to {args.out_csv}")


if __name__ == "__main__":
    main()
