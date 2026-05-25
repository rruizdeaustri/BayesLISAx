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
- `results/bench/**/summary.json`: per-run benchmark summaries.
- `results/bench/summary_table.csv`: aggregated benchmark table.
- `.venv/`: optional local environment with `snakemake` and editable install of this repo package.

## Notes

- The workflow sets `JAX_SAMPLERS_CONFIG` per run so each job uses the generated JSON config.
- `scripts/summarize.py` includes regex placeholders; adapt patterns to your actual log format.
- Plot behavior is configurable in `config.yaml` via `scan.skip_plots`, `final.skip_plots`, and `final.no_show`.

## Benchmark scans (multi-algorithm)

Benchmark mode is optional and preserves the existing two-phase NS workflow by default.

1. Set `benchmark.enabled: true` in `config.yaml`.
2. Configure scan dimensions in `benchmark.*` lists (`algos`, `seeds`, `n_live`, `tol`, `num_inner_steps`, `ggns_step_size`, `ggns_num_inner_steps`, `initial_num_steps`, `refinement_num_steps`, `max_batches`).
3. Run:

```bash
snakemake --cores 1 benchmark_all
```

Each run writes a compact JSON with runtime, return code, NS diagnostics (when present), and status labels (`pass`, `crash`, `low_ESS`, `bad_logZ`, `local_mode_suspected`). GGNS modes are treated as experimental comparison modes.

### Example command sets

Tiny smoke scan:

```bash
snakemake --cores 1 benchmark_all \
  --config benchmark.enabled=true benchmark.algos='["ns","ggns"]' \
  benchmark.seeds='[0]' benchmark.n_live='[64]' benchmark.num_inner_steps='[8]'
```

Pilot scan:

```bash
snakemake --cores 1 benchmark_all \
  --config benchmark.enabled=true benchmark.algos='["ns","dynamic_nss","ggns"]' \
  benchmark.seeds='[0,1,2]' benchmark.n_live='[256]' benchmark.num_inner_steps='[16,32]'
```

Production NS scan:

```bash
snakemake --cores 1 benchmark_all \
  --config benchmark.enabled=true benchmark.algos='["ns"]' \
  benchmark.seeds='[0,1,2,3,4]' benchmark.n_live='[1000]' benchmark.num_inner_steps='[64]' benchmark.tol='[1.0]'
```

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
