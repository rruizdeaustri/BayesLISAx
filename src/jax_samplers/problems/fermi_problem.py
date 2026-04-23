# jax_samplers/problems/fermi_problem.py
import jax.numpy as jnp
import jax.random as jr
from jax_samplers.core.problem import Problem, Array, Bounds
# import your existing implementations:
from jax_samplers.problems.fermi_ps import forward_model, poisson_loglik
from jax_samplers.problems.fermi_fixed import PROB, K_FIXED, LOG_FMIN, LOG_FMAX  # or pass via args

class FermiProblem(Problem):
    def __init__(self, patch=PROB.patch, K=K_FIXED, logf_min=LOG_FMIN, logf_max=LOG_FMAX,
                 bg_mu=None, bg_sigma=None):
        self.patch   = patch
        self.K       = int(K)
        self.dim     = 3*self.K + 2
        self.LOG_FMIN = float(logf_min)
        self.LOG_FMAX = float(logf_max)
        self.bg_mu    = jnp.asarray(bg_mu if bg_mu is not None else PROB.cfg.bg_scale_mu)
        self.bg_sigma = jnp.asarray(bg_sigma if bg_sigma is not None else PROB.cfg.bg_scale_sigma)

        H, W = self.patch.iso.shape
        self.prior_bounds: Bounds = (
            [(0.0, W-1.0)]*self.K +         # x
            [(0.0, H-1.0)]*self.K +         # y
            [(self.LOG_FMIN, self.LOG_FMAX)]*self.K +  # logF
            [(-jnp.inf, jnp.inf), (-jnp.inf, jnp.inf)] # bg0,bg1 (use Gaussian below)
        )

    @staticmethod
    def _split(theta: Array, K: int):
        theta = theta.reshape(-1)
        x    = theta[:K]
        y    = theta[K:2*K]
        logF = theta[2*K:3*K]
        bg   = theta[3*K:3*K+2]
        pos_xy = jnp.stack([x, y], axis=-1)   # (K,2) in (x,y)
        flux   = jnp.exp(logF)
        return pos_xy, flux, bg

    def loglikelihood(self, theta: Array) -> Array:
        pos, flx, bg = self._split(theta, self.K)
        lam = forward_model(self.patch.iso, self.patch.iem, self.patch.Fker, pos, flx, bg)
        # keep include_constant=True if you want exact logL; False for nested-sampling deltas
        return poisson_loglik(lam, self.patch.counts, include_constant=True)

    def logprior(self, theta: Array) -> Array:
        # Uniform in bounds for (x,y,logF) + Gaussian for bg
        pos, flx, bg = self._split(theta, self.K)
        H, W = self.patch.iso.shape

        in_x  = (pos[:,0] >= 0.0) & (pos[:,0] < W)
        in_y  = (pos[:,1] >= 0.0) & (pos[:,1] < H)
        in_lF = (jnp.log(flx) >= self.LOG_FMIN) & (jnp.log(flx) <= self.LOG_FMAX)
        in_support = jnp.all(in_x & in_y) & jnp.all(in_lF)

        lp_pos  = -self.K * jnp.log(float(H*W))
        lp_lF   = -self.K * jnp.log(self.LOG_FMAX - self.LOG_FMIN)
        # bg ~ N(mu, sigma) independent
        z = (bg - self.bg_mu) / self.bg_sigma
        lp_bg = -0.5*jnp.sum(z*z) - jnp.sum(jnp.log(self.bg_sigma*jnp.sqrt(2*jnp.pi)))
        lp = lp_pos + lp_lF + lp_bg
        return jnp.where(in_support, lp, -jnp.inf)

    def sample_prior(self, key: Array, n: int) -> Array:
        H, W = self.patch.iso.shape
        k1,k2,k3,k4 = jr.split(key, 4)
        x = jr.uniform(k1, (n,self.K), minval=0.0, maxval=float(W-1))
        y = jr.uniform(k2, (n,self.K), minval=0.0, maxval=float(H-1))
        logF = jr.uniform(k3, (n,self.K), minval=self.LOG_FMIN, maxval=self.LOG_FMAX)
        bg   = jr.normal(k4, (n,2)) * self.bg_sigma + self.bg_mu
        return jnp.concatenate([x, y, logF, bg], axis=1)  # (n, 3K+2)

def make(args):
    # optional factory for CLI: load your patch from args.* if needed
    return FermiProblem()
