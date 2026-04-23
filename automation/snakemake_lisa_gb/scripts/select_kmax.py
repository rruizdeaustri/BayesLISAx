#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def classify_jeffreys(delta_logz: float) -> str:
    # Jeffreys-style interpretation in log-evidence differences
    if delta_logz < 1.0:
        return "inconclusive"
    if delta_logz < 2.5:
        return "weak"
    if delta_logz < 5.0:
        return "strong"
    return "decisive"


def combined_sigma(a, b):
    if a is None or b is None:
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return math.sqrt(a * a + b * b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--delta-logz",
        type=float,
        default=1.0,
        help="Conservative ambiguity window: models within this many logZ units of best are treated as credible alternatives.",
    )
    parser.add_argument(
        "--fallback",
        choices=["simpler", "best"],
        default="simpler",
        help="If evidence is not decisive, choose the simpler model or keep the best-logZ model.",
    )
    args = parser.parse_args()

    rows = []
    with open(args.summary_csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                "kmax": int(row["kmax"]),
                "logz": float(row["logz"]),
                "logz_std": None,
            }
            if "logz_std" in row and row["logz_std"] not in ("", None):
                try:
                    item["logz_std"] = float(row["logz_std"])
                except ValueError:
                    item["logz_std"] = None
            rows.append(item)

    if not rows:
        raise RuntimeError("No rows found in summary CSV")

    ranked = sorted(rows, key=lambda x: x["logz"], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    # Conservative ambiguity window around the best model
    near_best = [r for r in ranked if r["logz"] >= best["logz"] - args.delta_logz]

    delta_logz = None
    delta_logz_sigma = None
    evidence_strength = "only_candidate"
    status = "selected"

    if second is not None:
        delta_logz = best["logz"] - second["logz"]
        evidence_strength = classify_jeffreys(delta_logz)
        delta_logz_sigma = combined_sigma(best.get("logz_std"), second.get("logz_std"))

        if evidence_strength == "inconclusive":
            status = "unresolved"
        elif evidence_strength == "weak":
            status = "ambiguous"
        else:
            status = "selected"

    # Preferred model = highest logZ
    preferred = best

    # Reported model = possibly conservative fallback
    if status in {"unresolved", "ambiguous"} and args.fallback == "simpler":
        selected = min(near_best, key=lambda x: x["kmax"])
    else:
        selected = preferred

    out = {
        "selected_kmax": selected["kmax"],
        "preferred_kmax": preferred["kmax"],
        "status": status,  # selected / ambiguous / unresolved / only_candidate
        "evidence_strength": evidence_strength,  # inconclusive / weak / strong / decisive
        "best_logz": best["logz"],
        "best_logz_std": best.get("logz_std"),
        "second_kmax": None if second is None else second["kmax"],
        "second_logz": None if second is None else second["logz"],
        "second_logz_std": None if second is None else second.get("logz_std"),
        "delta_logz_best_second": delta_logz,
        "delta_logz_sigma": delta_logz_sigma,
        "ambiguity_window": args.delta_logz,
        "credible_alternatives": [r["kmax"] for r in near_best],
        "candidates": ranked,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
