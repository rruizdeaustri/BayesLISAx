# problems/fermi_jaxns.py
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp
from jaxns.framework.prior import Prior
from jaxns.framework.model import Model
tfd = tfp.distributions

def build_jaxns_model(problem):
    ps        = problem.patch
    H, W      = ps.iso.shape
    K         = problem.K_FIXED
    LOG_FMIN  = jnp.asarray(problem.LOG_FMIN, jnp.float32)
    LOG_FMAX  = jnp.asarray(problem.LOG_FMAX, jnp.float32)
    mu_bg     = jnp.asarray(problem.cfg.bg_scale_mu, jnp.float32)
    sig_bg    = jnp.asarray(problem.cfg.bg_scale_sigma, jnp.float32)

    def prior_model():
        x    = yield Prior(tfd.Sample(tfd.Uniform(0.0, float(W-1)), sample_shape=(K,)), name="x")
        y    = yield Prior(tfd.Sample(tfd.Uniform(0.0, float(H-1)), sample_shape=(K,)), name="y")
        logF = yield Prior(tfd.Sample(tfd.Uniform(LOG_FMIN, LOG_FMAX), sample_shape=(K,)), name="logF")
        bg   = yield Prior(tfd.MultivariateNormalDiag(loc=mu_bg, scale_diag=sig_bg), name="bg")
        return x, y, logF, bg

    def log_likelihood(x, y, logF, bg):
        pos_xy = jnp.stack([x, y], axis=-1)
        flx    = jnp.exp(logF)
        lam    = problem.forward_model(ps.iso, ps.iem, ps.Fker, pos_xy, flx, bg)
        return problem.poisson_loglik(lam, ps.counts, include_constant=True)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    model.sanity_check(jr.PRNGKey(0), S=16)
    return model
