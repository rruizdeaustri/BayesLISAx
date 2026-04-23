import time
import jax
import jax.random as jr
import jax.numpy as jnp
from jax_samplers.problems.lisa_gb_transdim_problem import make

problem = make()
key = jr.PRNGKey(0)

x = problem.sample_prior(key, 1)[0]

t0 = time.time()
lp = problem.logprior(x)
jax.block_until_ready(lp)
print("logprior =", lp, " time =", time.time() - t0)

t0 = time.time()
ll = problem.loglikelihood(x)
jax.block_until_ready(ll)
print("loglike =", ll, " time =", time.time() - t0)
