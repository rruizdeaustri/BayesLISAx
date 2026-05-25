#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="+", required=True)
p.add_argument("--out-csv", required=True)
a = p.parse_args()

rows = [json.loads(Path(x).read_text(encoding="utf-8")) for x in a.inputs]
keys = sorted({k for r in rows for k in r.keys()})

Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        rr = {k: (json.dumps(v) if isinstance(v, (list, dict, tuple)) else v) for k, v in r.items()}
        w.writerow(rr)
