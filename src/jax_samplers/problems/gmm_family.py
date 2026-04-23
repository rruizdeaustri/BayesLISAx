from dataclasses import dataclass
import jax, jax.numpy as jnp, jax.random as jr, jax.scipy as jsp
from ..core.model_family import ModelFamily, PRNGKey

@dataclass
class GaussianMixtureFamily(ModelFamily):
    max_K: int = 5; n: int = 400
    lower: float = -10.; upper: float = 10.
    sigma: float = 0.5
    dim_per_atom: int = 1
    K_min: int = 1
    components_exchangeable: bool = True
    key: PRNGKey = jr.PRNGKey(0)

    repulsion_lambda: float = 0.0   # set >0 to enable (e.g. 0.05–0.2)
    repulsion_scale:  float = 0.6   # ~ expected component width
    
    def __post_init__(self):
        self.K_max = int(self.max_K)
        self.Ks    = tuple(range(self.K_min, self.K_max+1))
        # synth data + per-comp sigmas
        w = jnp.array([0.45,0.55]); mu = jnp.array([-2.,2.]); sd = jnp.array([0.4,0.6])
        self.key, k1, k2 = jr.split(self.key, 3)
        z = jr.categorical(k1, jnp.log(w), shape=(self.n,))
        self.y = jr.normal(k2, (self.n,)) * sd[z] + mu[z]
        pad = jnp.full((self.K_max - sd.size,), self.sigma) if self.K_max > sd.size else jnp.array([])
        #self.sds = (jnp.concatenate([sd[:self.K_max], pad]) if pad.size else sd[:self.K_max])
        pad = jnp.full((self.K_max - sd.size,), self.sigma) if self.K_max > sd.size else jnp.array([])
        self.sds = jnp.concatenate([sd[:self.K_max], pad]) if pad.size else sd[:self.K_max]

        self.logprior_jit = jax.jit(self.logprior_full)
        self.loglikelihood_jit = jax.jit(self.loglikelihood_full)
        
    def sample_K(self, key: PRNGKey, n: int) -> jnp.ndarray:
        p = jnp.full((len(self.Ks),), 1/len(self.Ks))
        return jr.choice(key, jnp.array(self.Ks), shape=(n,), p=p).astype(jnp.int32)

    #def sample_prior(self, key: PRNGKey, K: int, n: int) -> jnp.ndarray:
    #    key, sub1, sub2 = jr.split(key, 3)
    #    Kf = jnp.full((n,), K, dtype=jnp.float32)
    #    mus = jr.uniform(sub2, (n, self.max_K), minval=self.lower, maxval=self.upper)
    #    theta = jnp.concatenate([Kf[:, None], mus], axis=1)  # (n, 1+Kmax)
    #    return theta

    def sample_prior(self, key: PRNGKey, n: int) -> jnp.ndarray:
        key, kK, kTheta = jr.split(key, 3)
        Ks = self.sample_K(kK, n)  # shape (n,)
        Kf = Ks.astype(jnp.float32)[:, None]
        mus = jr.uniform(kTheta, (n, self.max_K), minval=self.lower, maxval=self.upper)
        check = jnp.concatenate([Kf, mus], axis=1)
        return jnp.concatenate([Kf, mus], axis=1)  # (n, 1+Kmax)
    
    
    #def sample_prior(self, key: PRNGKey, K: int, n: int) -> jnp.ndarray:
    #    return jr.uniform(key, (n, K*self.dim_per_atom), minval=self.lower, maxval=self.upper)

    def sample_prior_given_K_works(self, key: PRNGKey, K: int, n: int) -> jnp.ndarray:
        """
        Required by TransdimensionalProductSpace for fixed-K samples
        """
        Kf = jnp.full((n,), float(K))[:, None]
        mus = jr.uniform(key, (n, self.max_K), minval=self.lower, maxval=self.upper)
        check = jnp.concatenate([Kf, mus], axis=1)
        return jnp.concatenate([Kf, mus], axis=1)


    def sample_prior_given_K(self, key: PRNGKey, K: int, n: int) -> jnp.ndarray:
        # Return ONLY mus, shape (n, self.max_K)
        return jr.uniform(key, (n, self.max_K), minval=self.lower, maxval=self.upper)
  
    def logpmf_K(self, K: int) -> float:
        in_support = (K>=self.K_min) & (K<=self.K_max)
        return jnp.where(in_support, -jnp.log(len(self.Ks)), -jnp.inf)

    def logprior_full_works(self, theta_full: jnp.ndarray) -> jnp.ndarray:
        Kf      = theta_full[0]
        K       = Kf.astype(jnp.int32)
        mu_full = theta_full[1:]  # shape (K_max,)

        lp_K = jnp.where((1 <= K) & (K <= self.K_max), -jnp.log(self.K_max), -jnp.inf)
        mask = jnp.arange(mu_full.shape[0]) < K

        in_bounds = jnp.all((mu_full >= self.lower) & (mu_full <= self.upper) | (~mask))
        lp_mu = jnp.where(in_bounds, -K * jnp.log(self.upper - self.lower), -jnp.inf)

        return lp_K + lp_mu


    def logprior_full_works(self, theta_full: jnp.ndarray) -> jnp.ndarray:
        Kf      = theta_full[0]
        K       = Kf.astype(jnp.int32)
        mu_raw  = theta_full[1:]
        K_max   = self.K_max

        mu_full = jnp.zeros(K_max)
        mu_full = mu_full.at[:mu_raw.shape[0]].set(mu_raw[:K_max])

        mask = jnp.arange(K_max) < K
        in_bounds = (mu_full >= self.lower) & (mu_full <= self.upper)
        ok = jnp.all(jnp.logical_or(in_bounds, ~mask))

        # Uniform(-10,10) on active mus only
        lp_mu = jnp.sum(jnp.where(mask, -jnp.log(self.upper - self.lower), 0.0))
        return jnp.where(ok, lp_mu, -jnp.inf)

    def logprior_full_works_final(self, theta_full):
        Kf = theta_full[0]
        K  = Kf.astype(jnp.int32)
        mu_raw = theta_full[1:]
        K_max = self.K_max

        # pad/truncate to (K_max,)
        mu_full = jnp.zeros(K_max).at[:mu_raw.shape[0]].set(mu_raw[:K_max])

        mask = jnp.arange(K_max) < K
        in_bounds = (mu_full >= self.lower) & (mu_full <= self.upper)
        ok = jnp.all(jnp.logical_or(in_bounds, ~mask))

        # uniform prior over active mus only
        lp_mu = jnp.sum(jnp.where(mask, -jnp.log(self.upper - self.lower), 0.0))
        return jnp.where(ok, lp_mu, -jnp.inf)
    
    def logprior_full(self, theta_full: jnp.ndarray) -> jnp.ndarray:
        Kf      = theta_full[0]
        K       = Kf.astype(jnp.int32)
        mu_raw  = theta_full[1:]
        K_max   = self.K_max

        # pad/truncate to (K_max,)
        mu_full = jnp.zeros(K_max).at[:mu_raw.shape[0]].set(mu_raw[:K_max])

        mask = jnp.arange(K_max) < K  # active components (length K_max)
        in_bounds = (mu_full >= self.lower) & (mu_full <= self.upper)
        ok = jnp.all(jnp.logical_or(in_bounds, ~mask))

        # Uniform(-10,10) prior on ACTIVE means only
        lp_mu = jnp.sum(jnp.where(mask, -jnp.log(self.upper - self.lower), 0.0))

        # --- Repulsion prior between ACTIVE means (soft “don’t overlap”) ---
        #   penalty = λ * sum_{i<j} exp(- (Δμ)^2 / (2 ρ^2))
        # We subtract it from the log-prior.
        if self.repulsion_lambda > 0:
            mu_act = mu_full[mask]               # shape (K,)
            repulse = 0.0
            if mu_act.size > 1:
                diffs = mu_act[:, None] - mu_act[None, :]     # (K,K)
                tri   = jnp.triu(jnp.ones_like(diffs, dtype=bool), k=1)
                d2    = (diffs * diffs)[tri]                  # (K*(K-1)/2,)
                repulse = self.repulsion_lambda * jnp.sum(
                jnp.exp(- d2 / (2.0 * (self.repulsion_scale ** 2)))
                )
            lp_mu = lp_mu - repulse
        # -------------------------------------------------------------------

        return jnp.where(ok, lp_mu, -jnp.inf)
    
    def loglikelihood_full(self, theta_full: jnp.ndarray) -> jnp.ndarray:
        Kf = theta_full[0]
        K = Kf.astype(jnp.int32)

        mu_raw = theta_full[1:]
        K_max = self.K_max

        # Safely pad/truncate to shape (K_max,)
        mu_raw = mu_raw[:K_max]
        mu_full = jnp.zeros(K_max)
        mu_full = mu_full.at[:mu_raw.shape[0]].set(mu_raw)

        # Boolean mask of which components are active
        arange_K = jnp.arange(K_max)
        mask = arange_K < K  # shape (K_max,)

        # Compute logpdfs for all K_max components → shape (N, K_max)
        def comp_logpdf(i):
            return jsp.stats.norm.logpdf(self.y, mu_full[i], self.sds[i])  # shape (N,)

        lps_full = jnp.stack([comp_logpdf(i) for i in range(K_max)], axis=1)  # (N, K_max)

        # Apply mask to zero out inactive components
        lps_full = jnp.where(mask[None, :], lps_full, -jnp.inf)

        # Log equal weights for active components
        logw = jnp.where(mask, -jnp.log(K.astype(jnp.float32)), -jnp.inf)
        lps_full = lps_full + logw[None, :]  # (N, K_max)

        # Final marginal likelihood: sum over components, then over data
        loglike_per_point = jax.scipy.special.logsumexp(lps_full, axis=1)  # shape (N,)
        #print('loglike', loglike_per_point.shape,  jnp.sum(loglike_per_point).shape)
        return jnp.sum(loglike_per_point)  # scalar

    
