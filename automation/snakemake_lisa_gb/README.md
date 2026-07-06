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
2. Select a preset with `benchmark.preset` (`smoke`, `pilot`, `production`).
3. Optionally override any preset dimension by setting explicit `benchmark.*` arrays (`algos`, `seeds`, `n_live`, `tol`, `num_inner_steps`, `ggns_step_size`, `ggns_num_inner_steps`, `initial_num_steps`, `refinement_num_steps`, `max_batches`).
3. Run:

```bash
snakemake --cores 1 benchmark_all
```

Each run writes a compact JSON with runtime, return code, NS diagnostics (when present), and status labels (`pass`, `crash`, `low_ESS`, `bad_logZ`, `local_mode_suspected`). For `smoke` preset runs, summaries include `workflow_smoke_only` to make clear they are **workflow checks only** and not scientific validation. GGNS and dynamic GGNS are treated as experimental comparison modes.

### Preset intent

- `smoke`: tiny/fast workflow-only validation; do **not** interpret physically (local mode recovery is expected at low resolution).
- `pilot`: moderate settings for rough algorithm/configuration comparisons.
- `production`: trusted static NS benchmark settings for scientific validation.

### Example command sets

Tiny smoke scan:

```bash
snakemake --cores 1 benchmark_all \
  --config benchmark.enabled=true benchmark.preset=smoke
```

Pilot scan:

```bash
snakemake --cores 1 benchmark_all \
  --config benchmark.enabled=true benchmark.preset=pilot
```

Production NS scan:

```bash
snakemake --cores 1 benchmark_all \
  --config benchmark.enabled=true benchmark.preset=production
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

## Sangria catalogue inspection and window planning

This workflow also includes optional tools for planning a future full LISA Galactic Binary campaign from LDC/Sangria HDF5 catalogues. The segmentation idea is inspired by the Radler/GBMCMC practice of splitting the Galactic Binary foreground into frequency regions before running detailed inference. These rules are **diagnostic planning aids**: final per-window inference still uses the BayesLISAx nested-sampling workflow described above.

The manually maintained `bands.csv` remains the source of truth for the existing campaign rules. Generated Sangria windows are written to a separate CSV by default and are not consumed by `rule all` unless you explicitly choose to use them in a later workflow step.

### Inspect an HDF5 catalogue

Configure the HDF5 file path in `config.yaml` under `sangria.h5_path`, then run:

```bash
snakemake --cores 1 inspect_sangria
```

The `inspect_sangria` rule scans the expected Sangria source catalogue paths when present:

- `sky/dgb/cat`
- `sky/igb/cat`
- `sky/vgb/cat`

It prints the top-level HDF5 groups/datasets, available catalogue paths, entry counts, catalogue columns, and the minimum/maximum `Frequency` and `Amplitude` values. It also writes the same information to `sangria.inspection_json` for reproducibility.

### Generate frequency windows

After setting `sangria.f_min`, `sangria.f_max`, `sangria.tobs`, `sangria.core_width`, `sangria.guard_width`, `sangria.overlap`, and optionally `sangria.snr_threshold`, run:

```bash
snakemake --cores 1 make_sangria_windows
```

Each generated row contains a **core window** and a padded **analysis window**:

- The core window (`core_f_min`, `core_f_max`) is the nominal non-padded frequency segment used to tile the requested global frequency range.
- The analysis window (`analysis_f_min`, `analysis_f_max`) expands the core by `guard_width` on both sides, clipped to the requested global range. This padding is intended to catch leakage or boundary effects near the edge of a core segment.
- `overlap` controls overlap between adjacent core windows; it must be smaller than `core_width`.

The generated CSV reports catalogue source counts in each padded analysis window, an optional bright-source count above `sangria.snr_threshold`, and approximate maximum/median SNR values.

The approximate SNR calculation is intentionally simple and follows a sky-averaged monochromatic LISA planning estimate inspired by LDCio helper code. These counts and SNRs are only for diagnostics and window planning; they are not likelihood terms, priors, sampler settings, or production BayesLISAx evidence calculations.

## Static nested-sampling Sangria observed-data workflow

The default `Snakefile` now automates the validated **static nested-sampling only** workflow for observed Sangria Galactic Binary windows. The production rules intentionally invoke only:

```bash
--algo ns
```

They do not include `dynamic_nss`, `ggns`, `dynamic_ggns`, or `ns_hamiltonian` in the production target.

### Configuration

Edit `config.yaml`:

```yaml
static_ns:
  algo: ns
  k_values: [1, 2, 3, 4, 5]
  seeds: [0, 11, 22]
  n_live: 500
  num_delete_ratio: 0.1
  num_inner_steps: 64
  tol: 2

highres:
  enabled: true
  seeds: [0]
  n_live: 1000
  num_delete_ratio: 0.1
  num_inner_steps: 96
  tol: 1

catalogue:
  margin_hz: 5.0e-6
  snr_thresholds: [0.5, 1, 2, 3, 4, 5, 7, 10]
  top_n: 40
```

Set `sangria.h5_path` to enable catalogue diagnostics. If it is empty, the inference tables still build and the catalogue CSVs are written with headers only.

### Example dry run / smoke check

Use a one-row `bands.csv`, then run a Snakemake dry run with a tiny scan:

```bash
snakemake -n --cores 1 --config workflow_mode=dry_run
```

Execute the same tiny workflow locally:

```bash
snakemake --cores 1 --jobs 1 --config workflow_mode=dry_run
```

### Full static-NS run

```bash
snakemake --cores 1 --jobs 100
```

or on a Slurm cluster:

```bash
snakemake --cores 1 --jobs 100 \
  --cluster "sbatch --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=08:00:00"
```

### Static workflow outputs

The campaign writes structured JSON per run and aggregate CSV tables under `results/static_ns/`:

- `all_runs.csv`: every `(window, K, seed)` static-NS scan with `logZ`, `logZ_err`, `best_logL`, `ESS`, recovered `f0` summaries, and runtime.
- `per_window_k_summary.csv`: one row per `(window, K)` using the maximum `logZ` over seeds and `delta_logZ_from_previous_K` for neighbouring-K evidence scans.
- `selected_k.csv`: selected `K_best` per window, defined as the `K` with the largest seed-maximized evidence.
- `selected_highres_runs.csv`: high-resolution reruns of the selected `K_best` for configured `highres.seeds`.
- `catalogue_summary.csv`: catalogue source counts in an enlarged frequency interval, thresholded approximate-SNR counts, and the top-N sources serialized as JSON.
- `nearest_catalogue_matches.csv`: nearest catalogue source in frequency for recovered high-resolution `f0` values.

This reproduces the manual centered-band decision logic by comparing seed-maximized evidence across fixed K values, while retaining per-seed rows so crowded-window multimodality is visible.
