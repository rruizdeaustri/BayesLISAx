# LISA GB transdim campaign (Snakemake)

This is an **external orchestration skeleton** for running `jax_samplers` in two phases:

1. low-resolution NS sweep over `Kmax` (model selection),
2. high-resolution NS run for the selected `Kmax`.

By default, phase-1 scan runs with `--skip-plots` to avoid spending time on corner plots during model selection. Phase-2 keeps plotting enabled (with `--no-show` by default for batch jobs).

It is designed to work with the existing command line interface:

```bash
python -m jax_samplers.cli \
  --algo ns \
  --problem factory \
  --problem-factory jax_samplers.problems.lisa_gb_transdim_problem:make \
  --n-live 1000 --tol 1 --num-inner-steps 64
```

## Quick start

1. Edit `config.yaml`:
   - `repo_root` to your local `jax_samplers` checkout.
   - `base_json` to your base JSON (for example `examples/json/lisa_gb_transdim.json`).
2. Edit `bands.csv` with one band for validation.
3. (Optional but recommended) bootstrap a local environment and install this repository package in editable mode:

```bash
./setup_local.sh /path/to/jax_samplers
source .venv/bin/activate
```

If you want strict fail-fast version checking:

```bash
PYTHON_BIN=python3.12 EXPECT_PYTHON=3.12 ./setup_local.sh /path/to/jax_samplers
source .venv/bin/activate
```

4. Dry-run:

```bash
snakemake -n --cores 1
```

5. Execute locally:

```bash
snakemake --cores 1 --jobs 4
```

6. Execute on Slurm cluster (example):

```bash
snakemake --cores 1 --jobs 100 \
  --cluster "sbatch --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=08:00:00"
```

## Outputs

- `results/scan/<band_id>/k<kmax>/config.json`: generated per-run config.
- `logs/scan/*.log`: phase-1 logs.
- `results/scan/<band_id>/summary.csv`: model sweep summary.
- `results/scan/<band_id>/selection.json`: selected `Kmax`.
- `logs/final/*.log`: phase-2 logs.
- `results/final/<band_id>/done.txt`: completion marker.
- `.venv/`: optional local environment with `snakemake` and editable install of this repo package.

## Notes

- The workflow sets `JAX_SAMPLERS_CONFIG` per run so each job uses the generated JSON config.
- `scripts/summarize.py` includes regex placeholders; adapt patterns to your actual log format.
- Plot behavior is configurable in `config.yaml` via `scan.skip_plots`, `final.skip_plots`, and `final.no_show`.

## Troubleshooting installation

If `pip` reports errors like `No matching distribution found for snakemake>=8` while you believe you are on Python 3.12, verify that `pip` is bound to the same interpreter:

```bash
python --version
python -m pip --version
```

Also verify the venv interpreter directly:

```bash
.venv/bin/python --version
.venv/bin/python -m pip --version
```

Then install with:

```bash
python -m pip install "snakemake>=7.32.4,<10"
```

(The campaign currently targets Snakemake `>=7.32.4,<10` for broader environment compatibility.)

## Static-NS production workflow notes

This branch treats the Snakemake campaign as a static nested-sampling production
workflow. Legacy helper scripts for benchmark execution and Sangria window/truth
exports are retained for compatibility, but the default `rule all` now builds the
static-NS aggregate tables.

Prefer config-file overlays for dry-runs instead of fragile nested command-line
overrides such as `--config static_ns.k_values=...`:

```bash
snakemake -n --cores 1 --configfile config.dryrun.yaml
```

When `static_ns.highres.enabled: false`, the workflow stops at scan aggregation
and catalogue diagnostics are generated from scan-selected runs with nearest
catalogue matching skipped. When it is `true`, high-resolution selected runs are
aggregated separately and nearest-match diagnostics target those high-resolution
outputs.
