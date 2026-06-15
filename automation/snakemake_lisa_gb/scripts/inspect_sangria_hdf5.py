#!/usr/bin/env python
"""Inspect Sangria/LDC HDF5 Galactic Binary source catalogues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from sangria_catalog import (
    CATALOGUE_PATHS,
    available_catalogue_paths,
    list_top_level_groups,
    read_catalogue,
    validate_expected_columns,
)


def _range(values: np.ndarray) -> list[float | None]:
    if values.size == 0:
        return [None, None]
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite.size == 0:
        return [None, None]
    return [float(np.min(finite)), float(np.max(finite))]


def build_summary(h5_path: str) -> dict[str, object]:
    with h5py.File(h5_path, "r") as handle:
        present = available_catalogue_paths(handle)
        summary: dict[str, object] = {
            "h5_path": str(h5_path),
            "top_level_groups": list_top_level_groups(handle),
            "configured_catalogue_paths": CATALOGUE_PATHS,
            "available_catalogue_paths": present,
            "catalogues": {},
        }
        catalogues: dict[str, object] = {}
        for source_type, path in present.items():
            cat = read_catalogue(handle, source_type, path)
            freq = cat.data.get("Frequency", np.array([], dtype=float))
            amp = cat.data.get("Amplitude", np.array([], dtype=float))
            catalogues[source_type] = {
                "path": path,
                "n_entries": cat.size,
                "columns": list(cat.columns),
                "missing_expected_columns": validate_expected_columns(cat),
                "frequency_min_max_hz": _range(freq),
                "amplitude_min_max": _range(amp),
            }
        summary["catalogues"] = catalogues
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", help="Input Sangria HDF5 file")
    parser.add_argument("--out-json", required=True, help="Path for JSON summary")
    args = parser.parse_args()

    summary = build_summary(args.h5_path)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"HDF5 file: {summary['h5_path']}")
    print("Top-level groups/datasets: " + ", ".join(summary["top_level_groups"]))
    print("Available source catalogue paths:")
    for source_type, path in summary["available_catalogue_paths"].items():
        print(f"  {source_type}: {path}")
    for source_type, cat in summary["catalogues"].items():
        print(f"\n[{source_type}] {cat['path']}")
        print(f"  entries: {cat['n_entries']}")
        print("  columns: " + ", ".join(cat["columns"]))
        print(f"  frequency min/max [Hz]: {cat['frequency_min_max_hz']}")
        print(f"  amplitude min/max: {cat['amplitude_min_max']}")
        if cat["missing_expected_columns"]:
            print("  missing expected columns: " + ", ".join(cat["missing_expected_columns"]))


if __name__ == "__main__":
    main()
