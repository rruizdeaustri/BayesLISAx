from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import importlib
import inspect

import jax
import jax.numpy as jnp
import jax.random as jr

from .types import PRNGKey
from .utils import chunked_from_single

LogFn = Callable[[jnp.ndarray], jnp.ndarray]
PriorSamplerFn = Callable[[PRNGKey, int], jnp.ndarray]


@dataclass
class GenericProblem:
    """
    A model-agnostic fixed-dimension Problem.

    Provide:
      - dim: parameter dimension (int)
      - logprior(theta): scalar log prior (accepts (dim,) or (n,dim))
      - loglikelihood(theta): scalar log likelihood (accepts (dim,) or (n,dim))
      - sample_prior(key, n): (n, dim) prior draws  (OPTIONAL; required by some samplers)

    All callables must be JAX-friendly (use jnp / jax.scipy).
    """
    dim: int
    logprior_fn: LogFn
    loglikelihood_fn: LogFn
    sample_prior_fn: Optional[PriorSamplerFn] = None
    max_eval_batch: int | None = None

    # ---- helpers (now static methods) ----
    @staticmethod
    def _batched_from_single(fn: LogFn) -> LogFn:
        """Wrap (dim,) -> scalar so it also accepts (n,dim) via vmap."""
        def wrapped(x):
            return fn(x) if x.ndim == 1 else jax.vmap(fn)(x)
        return wrapped

    @staticmethod
    def _ensure_sample_prior_batchable(sample_prior_fn: PriorSamplerFn) -> PriorSamplerFn:
        """
        Ensure signature (key, n) -> (n, dim). If the provided fn only accepts (key),
        split the key and vmap it.
        """
        sig = inspect.signature(sample_prior_fn)
        if len(sig.parameters) >= 2:
            return sample_prior_fn  # already (key, n)
        def batched(key: PRNGKey, n: int):
            keys = jr.split(key, n)
            return jax.vmap(sample_prior_fn)(keys)  # (n, dim)
        return batched

    def __post_init__(self):
        # Start from originals
        orig_lp = self.logprior_fn
        orig_ll = self.loglikelihood_fn

        # Auto-batched log* ((dim,) or (n, dim))
        batched_lp = self._batched_from_single(orig_lp)
        batched_ll = self._batched_from_single(orig_ll)

        # Make prior sampler batchable if provided
        if self.sample_prior_fn is not None:
            self.sample_prior_fn = self._ensure_sample_prior_batchable(self.sample_prior_fn)

        # Optional chunking cap
        if self.max_eval_batch is not None and self.max_eval_batch > 0:
            self.logprior_fn      = chunked_from_single(lambda th: batched_lp(th), self.max_eval_batch)
            self.loglikelihood_fn = chunked_from_single(lambda th: batched_ll(th), self.max_eval_batch)
        else:
            self.logprior_fn      = batched_lp
            self.loglikelihood_fn = batched_ll

    # --- Problem interface expected by samplers ---
    @property
    def dim_(self) -> int:
        return self.dim

    def logprior(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self.logprior_fn(theta)

    def loglikelihood(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self.loglikelihood_fn(theta)

    def sample_prior(self, key: PRNGKey, n: int) -> jnp.ndarray:
        if self.sample_prior_fn is None:
            raise NotImplementedError("sample_prior is not available for this problem.")
        xs = self.sample_prior_fn(key, n)
        xs = jnp.asarray(xs)
        assert xs.shape == (n, self.dim), f"sample_prior returned {xs.shape}, expected {(n, self.dim)}"
        return xs

    # --- Convenience loader from 'module:function' entry points ---
    @staticmethod
    def _load_callable(entrypoint: str):
        mod, _, name = entrypoint.partition(":")
        if not mod or not name:
            raise ValueError(f"Invalid entry point '{entrypoint}'. Use 'pkg.module:function'.")
        return getattr(importlib.import_module(mod), name)

    @classmethod
    def from_entrypoints(
        cls,
        dim: int,
        logprior_ep: str,
        loglikelihood_ep: str,
        sample_prior_ep: Optional[str] = None,
    ) -> "GenericProblem":
        logprior = cls._load_callable(logprior_ep)
        loglikelihood = cls._load_callable(loglikelihood_ep)
        sample_prior = None if sample_prior_ep is None else cls._load_callable(sample_prior_ep)
        return cls(
            dim=dim,
            logprior_fn=logprior,
            loglikelihood_fn=loglikelihood,
            sample_prior_fn=sample_prior,
        )

