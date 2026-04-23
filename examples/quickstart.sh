#!/usr/bin/env bash
set -euo pipefail


pip install -e .[extras]


# NS
python -m sbi_samplers.cli --algo ns --n-live 400 --tol 3


# SMC+NUTS
python -m sbi_samplers.cli --algo smc-nuts --n-particles 1500 --nuts-step-size 1e-3


# RJ-MH
python -m sbi_samplers.cli --algo rj --rj-steps 30000
