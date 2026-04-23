#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./setup_local.sh /path/to/jax_samplers
#   PYTHON_BIN=python3.12 ./setup_local.sh /path/to/jax_samplers
#   EXPECT_PYTHON=3.12 ./setup_local.sh /path/to/jax_samplers
# If repo path is omitted, assumes this script lives in <repo>/automation/snakemake_lisa_gb.

REPO_ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECT_PYTHON="${EXPECT_PYTHON:-}"

echo "[setup] using interpreter: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

"${PYTHON_BIN}" -m venv .venv

VENV_PY_MM="$(
  .venv/bin/python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
echo "[setup] .venv python version: ${VENV_PY_MM}"

if [[ -n "${EXPECT_PYTHON}" && "${VENV_PY_MM}" != "${EXPECT_PYTHON}" ]]; then
  echo "[setup][error] expected .venv python ${EXPECT_PYTHON}, got ${VENV_PY_MM}" >&2
  exit 1
fi

if [[ "${PYTHON_BIN}" == *"3.12"* && "${VENV_PY_MM}" != "3.12" ]]; then
  echo "[setup][error] requested PYTHON_BIN='${PYTHON_BIN}' but .venv is ${VENV_PY_MM}" >&2
  echo "[setup][error] check your PATH/alias and retry with an absolute interpreter path." >&2
  exit 1
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "snakemake>=7.32.4,<10"
.venv/bin/python -m pip install -e "${REPO_ROOT}"

cat <<MSG
Environment ready.

Next:
  source .venv/bin/activate
  python --version
  python -m pip --version
  snakemake -n --cores 1
MSG
