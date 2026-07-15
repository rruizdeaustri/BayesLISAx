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

A later stage should add coherent conditional SNR and leave-one-candidate-out
`Delta log L`/`Delta log Z` using the production likelihood. Those quantities
require reconstructing complete candidate parameter vectors rather than using
catalogue frequencies alone.
