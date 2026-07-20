# Consensus candidate statistical selection

This optional stage consumes the existing per-window `clusters.csv` produced by
`within_window_consensus.py`. It deliberately keeps three concepts separate:

1. **Sampling quality** (`robust`, `plausible`, `confused`) measures reproducibility
   across independent nested-sampling seeds.
2. **Statistical selection** applies Bayesian false-discovery-rate (BFDR) control to
   the posterior inclusion mass already computed by the consensus stage.
3. **Sangria truth matching** validates selected and unselected candidates against
   the complete injected catalogue. Truth information is never used in the BFDR
   decision.

## Configuration

The extension works with defaults, but the following block can be added to
`config.yaml`:

```yaml
candidate_selection:
  enabled: true
  bfdr_q: 0.10
  probability_field: mean_inclusion
  max_match_bins: 5.0
```

- `bfdr_q` is the target expected false-discovery fraction among selected modes.
- `probability_field` can be `mean_inclusion` or `median_inclusion`.
  `mean_inclusion` is recommended for strongly multimodal bands because a genuine
  mode found by only one seed retains non-zero ensemble posterior mass.
- `max_match_bins` only controls the validation label `nearby`; it does not affect
  statistical selection.

## Run one window

```bash
snakemake -s Snakefile.candidate_selection --cores 1 -p \
  results/static_ns/consensus/manual_00184/candidate_selection_summary.json
```

## Run all configured windows

```bash
snakemake -s Snakefile.candidate_selection --cores 1 -p \
  candidate_selection_all
```

## Outputs

For each window:

- `candidates.csv` contains the original consensus quantities plus:
  - `sampling_quality`
  - `posterior_inclusion_score`
  - `local_fdr`
  - `cumulative_bfdr`
  - `selection_rank`
  - `statistically_selected`
  - full nearest-catalogue parameters
  - frequency separation in Hz and Fourier bins
  - `truth_match_quality`
- `candidate_selection_summary.json` records the BFDR configuration, selected
  cluster IDs, and sampling/truth-match counts.

The `catalogue_approx_snr` column is the existing lightweight planning estimate.
It is explicitly not the coherent SNR from the production likelihood and is not
used to accept or reject candidates.

## Interpretation

A seed-specific mode can be statistically selected when its ensemble posterior
inclusion is high enough, even if its sampling label is `confused`. Conversely, a
mode can be reproducible across seeds but fail BFDR selection if its posterior
inclusion is weak. This is intentional: seed recurrence diagnoses mode coverage,
whereas BFDR controls the expected false-source fraction.

The optional conditional-likelihood stage below now provides a fixed-configuration
leave-one-candidate-out `Delta log L` using the production likelihood. A future
extension may add a profiled conditional statistic that reoptimizes remaining
candidates after each removal.

## Conditional-likelihood validation

After `candidates.csv` has been produced, the optional conditional stage evaluates
candidate necessity with the same high-resolution posterior bundles and the
BayesLISAx A/E likelihood used by the sampler:

```bash
snakemake -s Snakefile.conditional_likelihood --cores 1 -p \
  conditional_likelihood_all
```

For each window it writes:

- `results/static_ns/conditional/{window_id}/candidate_significance.csv`
- `results/static_ns/conditional/{window_id}/final_catalogue.csv`
- `results/static_ns/conditional/{window_id}/summary.json`

By default, the stage validates **all distinct clusters** present in
`candidates.csv`. The BFDR field `statistically_selected`, sampling quality, and
posterior-inclusion values are preserved as diagnostics only; they are not default
filters. Optional prefilters may be configured under `conditional_likelihood`:

- `minimum_mean_inclusion`
- `minimum_seed_support`
- `allowed_sampling_quality`

Selection uses only consensus candidates, posterior summaries, and likelihood
re-evaluations. Catalogue frequencies, catalogue SNR, and other truth fields may
be present in `candidates.csv` as validation diagnostics, but they are ignored by
the conditional selection code.

The reported statistic is a **fixed-configuration conditional likelihood**:
`delta_logl_fixed = logL_full - logL_without_i` and
`rho_cond_fixed = sqrt(max(0, 2 * delta_logl_fixed))`. Remaining candidates are
not reoptimized after one candidate is removed. Placeholder columns
`delta_logl_profiled` and `rho_cond_profiled` are reserved for a later profiled
conditional statistic.

Assumptions documented in `summary.json` include: posterior bundles contain
physical decoded rows `[f0, fdot, iota, psi, lam, beta, p]`; all seven physical
parameters for one representative source come from one posterior component/draw;
representatives for different clusters may originate from different posterior
draws or seeds, so the assembled full vector is a synthetic joint configuration;
amplitude and initial phase are nuisance parameters handled by the production
likelihood when `marg_Aphi=true`; inactive source slots are sent to the likelihood
with a near-zero gate probability; if the number of candidates exceeds the
likelihood `Kmax`, the stage fails clearly unless optional prefilters reduce the
set; `model.order_f0=true` is not supported because the inverse ordered-frequency
transform is not unique.
