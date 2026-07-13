#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np

from sangria_catalog import (
    read_catalogues,
    concatenate_catalogues,
    lisa_gb_approx_snr,
)


def read_selected(path: str):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_f0(x):
    if not x:
        return []

    try:
        v = ast.literal_eval(x)
    except Exception:
        try:
            return [float(x)]
        except Exception:
            return []

    arr = np.asarray(v, dtype=float).reshape(-1)
    return [float(y) for y in arr]


def as_flat_array(x, dtype=None):
    if x is None:
        return None
    arr = np.asarray(x, dtype=dtype)
    return arr.reshape(-1)


def decode_source_type(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5-path", required=True)
    p.add_argument("--selected-csv", required=True)
    p.add_argument("--highres-csv", default="")
    p.add_argument("--tobs", type=float, required=True)
    p.add_argument("--margin-hz", type=float, required=True)
    p.add_argument("--snr-thresholds", nargs="+", type=float, required=True)
    p.add_argument("--top-n", type=int, default=40)
    p.add_argument("--out-summary", required=True)
    p.add_argument("--out-matches", default="")
    a = p.parse_args()

    cat = concatenate_catalogues(read_catalogues(a.h5_path))

    f = as_flat_array(cat["Frequency"], dtype=float)
    amp = as_flat_array(cat["Amplitude"], dtype=float)

    lat = as_flat_array(cat.get("EclipticLatitude"), dtype=float)
    inc = as_flat_array(cat.get("Inclination"), dtype=float)

    snr = as_flat_array(
        lisa_gb_approx_snr(f, amp, a.tobs, lat, inc),
        dtype=float,
    )

    source_type = cat.get("source_type")
    if source_type is None:
        source_type = np.array(["unknown"] * len(f), dtype=object)
    else:
        source_type = as_flat_array(source_type)

    # Be defensive in case one of the catalogue fields has an unexpected shape.
    n = min(len(f), len(amp), len(snr), len(source_type))
    f = f[:n]
    amp = amp[:n]
    snr = snr[:n]
    source_type = source_type[:n]

    sel = read_selected(a.selected_csv)

    recovered = {}
    if a.out_matches and a.highres_csv and Path(a.highres_csv).exists():
        with open(a.highres_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                recovered.setdefault(r["window_id"], []).extend(
                    parse_f0(r.get("all_best_f0") or r.get("best_f0", ""))
                )

    summary = []
    matches = []

    for w in sel:
        wid = w["window_id"]
        f_min = float(w["f_min"])
        f_max = float(w["f_max"])
        lo = f_min - a.margin_hz
        hi = f_max + a.margin_hz

        m = (f >= lo) & (f <= hi)
        idx = np.flatnonzero(m)

        if idx.size:
            order = idx[np.argsort(snr[idx])[::-1]]
        else:
            order = np.array([], dtype=int)

        top_sources = [
            {
                "rank": i + 1,
                "frequency": float(f[j]),
                "snr": float(snr[j]),
                "source_type": decode_source_type(source_type[j]),
            }
            for i, j in enumerate(order[: a.top_n])
        ]

        row = {
            "window_id": wid,
            "f_min": w["f_min"],
            "f_max": w["f_max"],
            "catalogue_f_min": lo,
            "catalogue_f_max": hi,
            "total_catalogue_sources": int(idx.size),
            "top_sources_json": json.dumps(top_sources),
        }

        for th in a.snr_thresholds:
            row[f"count_snr_gt_{th:g}"] = int(np.sum(snr[idx] > th))

        summary.append(row)

        for rf0 in recovered.get(wid, []):
            if not idx.size:
                continue

            j = idx[np.argmin(np.abs(f[idx] - rf0))]
            matches.append(
                {
                    "window_id": wid,
                    "recovered_f0": rf0,
                    "catalogue_frequency": float(f[j]),
                    "delta_f": float(rf0 - f[j]),
                    "approx_snr": float(snr[j]),
                    "source_type": decode_source_type(source_type[j]),
                }
            )

    fields = (
        [
            "window_id",
            "f_min",
            "f_max",
            "catalogue_f_min",
            "catalogue_f_max",
            "total_catalogue_sources",
        ]
        + [f"count_snr_gt_{th:g}" for th in a.snr_thresholds]
        + ["top_sources_json"]
    )

    Path(a.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_summary, "w", newline="") as f1:
        writer = csv.DictWriter(f1, fields)
        writer.writeheader()
        writer.writerows(summary)

    if a.out_matches:
        Path(a.out_matches).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_matches, "w", newline="") as f2:
            writer = csv.DictWriter(
                f2,
                [
                    "window_id",
                    "recovered_f0",
                    "catalogue_frequency",
                    "delta_f",
                    "approx_snr",
                    "source_type",
                ],
            )
            writer.writeheader()
            writer.writerows(matches)


if __name__ == "__main__":
    main()
