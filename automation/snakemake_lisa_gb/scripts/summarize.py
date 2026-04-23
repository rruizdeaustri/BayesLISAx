#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


# Prefer the final NS-report line, e.g.
# [NS report] logZ=-4700.034±0.377  D_KL=...
FINAL_REPORT_PATTERN = re.compile(
    r"^\[NS report\]\s+logZ\s*=\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"±"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.MULTILINE,
)

# Fallback: final report line without uncertainty
FINAL_REPORT_NO_STD_PATTERN = re.compile(
    r"^\[NS report\]\s+logZ\s*=\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.MULTILINE,
)

# Generic fallbacks if the final report line is absent.
FALLBACK_PATTERNS = [
    re.compile(r"log_evidence\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"),
    re.compile(r"logZ\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"),
]


def extract_logz_info(text: str) -> tuple[float, float | None]:
    # 1) Prefer the LAST [NS report] logZ=...±... occurrence
    matches = FINAL_REPORT_PATTERN.findall(text)
    if matches:
        logz, logz_std = matches[-1]
        return float(logz), float(logz_std)

    # 2) If report exists without std, still prefer that
    matches = FINAL_REPORT_NO_STD_PATTERN.findall(text)
    if matches:
        return float(matches[-1]), None

    # 3) Fallback: take the LAST generic match, not the first
    for pattern in FALLBACK_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return float(matches[-1]), None

    return float("-inf"), None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--kmaxs", nargs="+", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    if len(args.logs) != len(args.kmaxs):
        raise ValueError("--logs and --kmaxs must have same length")

    rows = []
    for log_path, kmax in zip(args.logs, args.kmaxs):
        text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
        logz, logz_std = extract_logz_info(text)
        rows.append(
            {
                "kmax": int(kmax),
                "logz": logz,
                "logz_std": logz_std if logz_std is not None else "",
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kmax", "logz", "logz_std"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
