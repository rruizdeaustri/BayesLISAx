#!/usr/bin/env python3
"""Legacy benchmark entry point retained for compatibility with older workflows."""
from __future__ import annotations
import argparse, subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to benchmark/run")
    args = parser.parse_args()
    if not args.command:
        parser.error("provide a command to run")
    raise SystemExit(subprocess.call(args.command))

if __name__ == "__main__":
    main()
