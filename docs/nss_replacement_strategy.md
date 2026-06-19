# BlackJAX NSS replacement strategy

BayesLISAx keeps the BlackJAX global NSS live-point replacement strategy as the default. Existing configurations do not need any changes.

To opt into an experimental BlackJAX branch that exposes cluster-aware NSS replacement, add a sampler block to your JSON config:

```json
{
  "sampler": {
    "replacement_strategy": "cluster_aware"
  }
}
```

or pass the equivalent CLI flag:

```bash
jax-samplers --algo ns --replacement-strategy cluster_aware ...
```

Allowed values are:

- `"global"` or `"default"`: preserve the current BlackJAX default replacement strategy.
- `"cluster_aware"`: use `blackjax.ns.nss.cluster_aware_update_with_mcmc_take_last` when available.

If `"cluster_aware"` is selected with a BlackJAX install that does not expose the experimental update function, BayesLISAx raises a clear error instead of silently falling back.
