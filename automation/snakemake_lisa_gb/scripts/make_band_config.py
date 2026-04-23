#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--f-min", type=float, required=True)
    parser.add_argument("--f-max", type=float, required=True)
    parser.add_argument("--kmax", type=int, required=True)
    args = parser.parse_args()

    with open(args.base_json, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    cfg.setdefault("band", {})
    cfg["band"]["f_min"] = args.f_min
    cfg["band"]["f_max"] = args.f_max

    cfg.setdefault("model", {})
    cfg["model"]["Kmax"] = args.kmax

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


if __name__ == "__main__":
    main()
