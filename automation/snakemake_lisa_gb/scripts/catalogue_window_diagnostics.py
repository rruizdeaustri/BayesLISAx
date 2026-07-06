#!/usr/bin/env python3
"""Create lightweight catalogue/window diagnostics for static-NS outputs."""
from __future__ import annotations
import argparse, csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--skip-nearest-matches", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.runs_csv, encoding="utf-8")))
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["band_id", "f_min", "f_max", "diagnostic", "nearest_match_status"]
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()
        seen = set()
        for row in rows:
            band = row.get("band_id", "")
            if band in seen: continue
            seen.add(band)
            writer.writerow({"band_id": band, "f_min": row.get("f_min", ""), "f_max": row.get("f_max", ""), "diagnostic": "scan_selected_runs" if args.skip_nearest_matches else "highres_selected_runs", "nearest_match_status": "skipped" if args.skip_nearest_matches else "not_configured"})

if __name__ == "__main__":
    main()
