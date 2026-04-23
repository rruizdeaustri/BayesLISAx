# jax_samplers/samplers/jaxns_unified.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import tensorflow_probability.substrates.jax as tfp
from jaxns.framework.prior import Prior
from jaxns.framework.model import Model
from jaxns import NestedSampler, resample

# Robust Categorical import for trans-dim
try:
    from jaxns.framework.special_priors import Categorical as JAXNSCategorical
    _JAXNS_CATEGORICAL_SRC = "framework.special_priors"
except Exception:
    try:
        from jaxns.prior_transforms import Categorical as JAXNSCategororical  # fallback name in some versions
        JAXNSCategorical = JAXNSCategororical
        _JAXNS_CATEGORICAL_SRC = "prior_transforms"
    except Exception:
        JAXNSCategorical = None
        _JAXNS_CATEGORICAL_SRC = "tfp-fallback"

from jax_samplers.registry import register_sampler

tfd = tfp.distributions


# ----------------- Public config & result -----------------

@dataclass
class JAXNSConfig:
    # shared
    num_live_points: int = 600
    s: int = 10
    max_samples: Optional[float] = None
    gradient_guided: bool = False
    parameter_estimation: bool = True
    enable_x64: bool = True
    # box fallback for fixed-dim when prior_bounds are not provided
    box_low: float = -10.0
    box_high: float = 10.0
    verbose: bool = False

@dataclass
class SamplerResult:
    samples: np.ndarray
    weights: Optional[np.ndarray] = None
    info: Optional[dict] = None


# ----------------- Sampler -----------------

@register_sampler("jaxns")
class JAXNSSamplerUnified:
    """
    Unified JAXNS sampler:

      Fixed-dim problems:
        - Requires .dim (int), .loglikelihood(theta), .logprior(theta)
        - Optional .prior_bounds: list[(low, high)] of len dim
        - Optional .sample_prior(key, n) for dim inference fallback

      Trans-dim (product space):
        - Problem has K_min, K_max, dim_per_atom, loglikelihood_full(...)
        - Returns samples with leading K column encoded separately.
    """
    def __init__(self, problem, cfg: JAXNSConfig):
        self.problem = problem
        self.cfg = cfg

        # Decide mode
        self._is_transdim = hasattr(problem, "K_min") and hasattr(problem, "K_max")

        if self._is_transdim:
            fam = getattr(problem, "family", problem)
            for name in ("K_min", "K_max", "dim_per_atom", "loglikelihood_full"):
                if not hasattr(fam, name):
                    raise SystemExit(f"jaxns (trans-dim): family is missing '{name}'.")
            if JAXNSCategorical is None:
                raise SystemExit("jaxns (trans-dim): Categorical prior not found in jaxns. "
                                 "Upgrade jaxns or adjust imports.")
            self.family = fam
            self.K_min = int(fam.K_min)
            self.K_max = int(fam.K_max)
            self.dim_per_atom = int(fam.dim_per_atom)
            self.D_full = self.K_max * self.dim_per_atom
            self.model = self._build_model_transdim()
        else:
            # Fixed-dim path: resolve dim once
            if hasattr(problem, "dim") and isinstance(problem.dim, int):
                self.dim = int(problem.dim)
            elif hasattr(problem, "sample_prior"):
                th = problem.sample_prior(jr.PRNGKey(0), 1)  # (1, D)
                self.dim = int(th.shape[-1])
            else:
                raise SystemExit("jaxns (fixed): need problem.dim or problem.sample_prior to infer dimension.")
            self.model = self._build_model_fixeddim()

        # Construct sampler; you can JIT the call operator for speed
        self.ns = NestedSampler(
            model=self.model,
            num_live_points=cfg.num_live_points,
            s=cfg.s,
            max_samples=cfg.max_samples,
            gradient_guided=cfg.gradient_guided,
            parameter_estimation=cfg.parameter_estimation,
            verbose=True
        )
        self.ns_jit = jax.jit(self.ns)

    # ---------- fixed-dim model ----------
    def _build_model_fixeddim(self) -> Model:
        problem = self.problem
        dim = self.dim

        # Prefer per-dimension prior bounds if provided by the problem
        pb = getattr(problem, "prior_bounds", None)

        if pb is not None and len(pb) == dim:
            low  = jnp.array([lo for lo, hi in pb])
            high = jnp.array([hi for lo, hi in pb])
        else:
            low  = jnp.full((dim,), self.cfg.box_low)
            high = jnp.full((dim,), self.cfg.box_high)


        def prior_model():
            thetas = []
            for i in range(dim):
                thetas.append(
                    (yield Prior(tfd.Uniform(low=low[i], high=high[i]), name=f"theta_{i}"))
                )

            return tuple(thetas)

        def log_likelihood(*thetas):
            theta = jnp.stack(thetas)  # (D,)
            # Match NS target: posterior ∝ exp(loglikelihood)
            return problem.loglikelihood(theta)

        model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
        model.sanity_check(jr.PRNGKey(0), S=16)
        return model

    # ---------- trans-dim (product-space) model ----------
    def _build_model_transdim(self) -> Model:
        fam = self.family
        K_min, K_max = self.K_min, self.K_max
        D_full = self.D_full
        low, high = float(self.cfg.box_low), float(self.cfg.box_high)

        def prior_model():
            # Uniform categorical over K_min..K_max
            K0 = yield JAXNSCategorical(
                parametrisation="cdf",
                logits=jnp.zeros((K_max - K_min + 1,), dtype=jnp.float32),
                name="K0",
            )
            K = K0 + K_min
            # Padded parameter vector
            theta_full = yield Prior(
                tfd.Sample(tfd.Uniform(low=low, high=high), sample_shape=(D_full,)),
                name="theta_full"
            )
            return K, theta_full

        def log_likelihood(K, theta_full):
            # Prefer batched API if family exposes it
            try:
                return fam.loglikelihood_full(theta_full[None, :], jnp.array([K], jnp.int32))[0]
            except TypeError:
                # Fallback to concatenated convention if provided by problem
                theta_packed = jnp.concatenate([jnp.asarray([K], theta_full.dtype), theta_full], 0)
                try:
                    return fam.loglikelihood_full(theta_packed)
                except TypeError:
                    return self.problem.loglikelihood(theta_packed)

        model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
        model.sanity_check(jr.PRNGKey(0), S=16)
        return model

    # ---------- API ----------
    def init(self, key: jnp.ndarray) -> "JAXNSSamplerUnified":
        self._key = key
        return self

    def run(self, key: jnp.ndarray) -> SamplerResult:
        term, state = self.ns_jit(key)
        results = self.ns.to_results(term, state)

        key, sub = jr.split(key)
        S = int(max(1, results.ESS))
        posterior = resample(
            key=sub,
            samples=results.samples,
            log_weights=results.log_dp_mean,
            S=S,
            replace=True
        )

        if not self._is_transdim:
            xs: List[np.ndarray] = [np.array(posterior[f"theta_{i}"]) for i in range(self.dim)]
            samples = np.stack(xs, axis=1)
            return SamplerResult(samples=samples, weights=None, info={"ESS": float(results.ESS)})
        else:
            K0 = np.array(posterior["K0"], dtype=np.int64)     # {0..K_max-K_min}
            K  = K0 + self.K_min
            theta_full = np.array(posterior["theta_full"])     # (S, D_full)
            Kf = K.astype(np.float64)[:, None]
            samples = np.concatenate([Kf, theta_full], axis=1) # (S, 1 + D_full)
            return SamplerResult(samples=samples, weights=None, info={"ESS": float(results.ESS)})

        
