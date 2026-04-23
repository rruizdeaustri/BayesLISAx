# jax_samplers

A modular JAX-based Bayesian sampling toolkit with multiple inference backends,
including nested sampling, SMC variants, NumPyro NUTS, and reversible-jump samplers.

## Capabilities

The package exposes a unified sampler registry and CLI that can run different
algorithms against fixed-dimensional and transdimensional problems.

### Available samplers

Registered samplers include:

- `ns` (BlackJAX nested sampling)
- `smc-nuts`
- `smc-hmc`
- `es` / `elliptical-slice`
- `pt-rw`
- `rj`
- `rj-es`
- `numpyro-nuts`
- `jaxns` (lazy-loaded)

See sampler registry wiring in `src/jax_samplers/registry.py`.

### Problem modes supported by CLI

- `gmm2`
- `fixed`
- `transdim`
- `factory` (custom `module:callable` factory)

For LISA workflows, `factory` mode is commonly used with:

- `jax_samplers.problems.lisa_onegb_problem:make`
- `jax_samplers.problems.lisa_gb_transdim_problem:make`

### LISA transdim workflow support

The LISA transdim backend is designed for narrow-band Galactic Binary inference.
It supports gate-based effective source counting (`K_eff`) and marginalization
controls in JSON configuration files.

## Installation

### Requirements

- Python >= 3.9
- JAX stack + sampler dependencies (installed via project dependencies)

### Install from repository

```bash
git clone <your-fork-or-repo-url>
cd jax_samplers
python -m pip install -U pip
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e .[extras]
```

## CLI usage

The console script installed is `jax-samplers`, equivalent to
`python -m jax_samplers.cli`.

### Minimal command template

```bash
python -m jax_samplers.cli \
  --algo ns \
  --problem factory \
  --problem-factory jax_samplers.problems.lisa_gb_transdim_problem:make \
  --n-live 1000 \
  --tol 1 \
  --num-inner-steps 64
```

### Config file handling

Config can be provided either by:

- `--config /path/to/config.json`, or
- environment variable `JAX_SAMPLERS_CONFIG`.

`LISA_CFG` is also supported as a fallback env variable.

### Useful runtime flags

- `--disable-jit` for debugging
- `--dtype float32|float64`
- `--seed <int>`
- `--save <path>` / `--no-show`
- `--skip-plots` (run inference and skip all plotting/postprocessing)

## Running examples

The repository contains runnable examples and JSON configurations in:

- `examples/`
- `examples/json/`

Example run scripts include:

- `examples/gmm2_ns.sh`
- `examples/fermi_ns.sh`
- `examples/quickstart.sh`

## Automation (campaign orchestration)

For two-phase LISA GB campaign orchestration (scan `Kmax` then final run), see:

- `automation/snakemake_lisa_gb/README.md`

This is designed as an external orchestration layer around the existing CLI.

## Development notes

- Source code lives in `src/jax_samplers/`
- CLI entrypoint: `src/jax_samplers/cli.py`
- Registry: `src/jax_samplers/registry.py`
