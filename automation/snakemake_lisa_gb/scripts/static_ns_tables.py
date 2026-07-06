#!/usr/bin/env python3
"""Utilities for static-NS Snakemake run summaries and aggregate tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import extract_logz_info  # noqa: E402


def _read_summary(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "kmax" in data and data["kmax"] not in (None, ""):
        data["kmax"] = int(data["kmax"])
    data["seed"] = int(data["seed"])
    data["logZ"] = float(data["logZ"])
    return data


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(args: argparse.Namespace) -> None:
    text = Path(args.log).read_text(encoding="utf-8", errors="ignore")
    logz, logz_std = extract_logz_info(text)
    row = {
        "stage": args.stage,
        "band_id": args.band_id,
        "kmax": int(args.kmax) if args.kmax is not None else None,
        "seed": int(args.seed),
        "f_min": float(args.f_min),
        "f_max": float(args.f_max),
        "logZ": logz,
        "logZ_std": logz_std,
        "log_path": args.log,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate_static(args: argparse.Namespace) -> None:
    runs = [_read_summary(p) for p in args.summaries]
    runs.sort(key=lambda r: (r["band_id"], r["kmax"], r["seed"]))
    all_fields = ["stage", "band_id", "kmax", "seed", "f_min", "f_max", "logZ", "logZ_std", "log_path"]
    _write_csv(args.all_runs, runs, all_fields)

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in runs:
        grouped[(row["band_id"], row["kmax"])].append(row)

    per_rows = []
    for (band_id, kmax), rows in sorted(grouped.items()):
        vals = [r["logZ"] for r in rows]
        best = max(rows, key=lambda r: r["logZ"])
        per_rows.append({
            "band_id": band_id,
            "kmax": kmax,
            "f_min": rows[0]["f_min"],
            "f_max": rows[0]["f_max"],
            "n_seeds": len(rows),
            "best_logZ": best["logZ"],
            "best_logZ_seed": best["seed"],
            "mean_logZ": mean(vals),
            "std_logZ": pstdev(vals) if len(vals) > 1 else 0.0,
            "min_logZ": min(vals),
            "max_logZ": max(vals),
            "delta_logZ_from_previous_K": "",
        })

    previous_by_band: dict[str, dict] = {}
    for row in per_rows:
        prev = previous_by_band.get(row["band_id"])
        if prev is not None:
            row["delta_logZ_from_previous_K"] = row["best_logZ"] - prev["best_logZ"]
        previous_by_band[row["band_id"]] = row

    per_fields = ["band_id", "kmax", "f_min", "f_max", "n_seeds", "best_logZ", "best_logZ_seed", "mean_logZ", "std_logZ", "min_logZ", "max_logZ", "delta_logZ_from_previous_K"]
    _write_csv(args.per_window, per_rows, per_fields)

    by_band: dict[str, list[dict]] = defaultdict(list)
    for row in per_rows:
        by_band[row["band_id"]].append(row)
    selected = []
    for band_id, rows in sorted(by_band.items()):
        finite = [r for r in rows if math.isfinite(float(r["best_logZ"]))]
        candidates = finite or rows
        best = max(candidates, key=lambda r: r["best_logZ"])
        threshold = best["best_logZ"] - float(args.delta_logz)
        credible = [r for r in candidates if r["best_logZ"] >= threshold]
        choice = min(credible, key=lambda r: r["kmax"])
        selected.append({"band_id": band_id, "selected_kmax": choice["kmax"], "best_logZ": choice["best_logZ"], "best_logZ_seed": choice["best_logZ_seed"], "f_min": choice["f_min"], "f_max": choice["f_max"]})
    _write_csv(args.selected_k, selected, ["band_id", "selected_kmax", "best_logZ", "best_logZ_seed", "f_min", "f_max"])


def aggregate_highres(args: argparse.Namespace) -> None:
    selected = {r["band_id"]: r for r in csv.DictReader(open(args.selected_k, encoding="utf-8"))}
    rows = []
    for path in args.summaries:
        row = _read_summary(path)
        sel = selected.get(row["band_id"], {})
        row["selected_kmax"] = sel.get("selected_kmax", row.get("kmax", ""))
        rows.append(row)
    rows.sort(key=lambda r: (r["band_id"], r["seed"]))
    _write_csv(args.out_csv, rows, ["stage", "band_id", "selected_kmax", "seed", "f_min", "f_max", "logZ", "logZ_std", "log_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("write-summary")
    p.add_argument("--log", required=True); p.add_argument("--out-json", required=True); p.add_argument("--band-id", required=True)
    p.add_argument("--kmax"); p.add_argument("--seed", required=True); p.add_argument("--f-min", required=True); p.add_argument("--f-max", required=True)
    p.add_argument("--stage", choices=["scan", "highres"], required=True); p.set_defaults(func=write_summary)
    p = sub.add_parser("aggregate-static")
    p.add_argument("--summaries", nargs="+", required=True); p.add_argument("--all-runs", required=True); p.add_argument("--per-window", required=True); p.add_argument("--selected-k", required=True); p.add_argument("--delta-logz", type=float, default=1.0); p.set_defaults(func=aggregate_static)
    p = sub.add_parser("aggregate-highres")
    p.add_argument("--selected-k", required=True); p.add_argument("--summaries", nargs="+", required=True); p.add_argument("--out-csv", required=True); p.set_defaults(func=aggregate_highres)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
