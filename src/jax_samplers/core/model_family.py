# src/jax_samplers/core/model_family.py
from abc import ABC, abstractmethod
from typing import Tuple
import jax.numpy as jnp
PRNGKey = jnp.ndarray

class ModelFamily(ABC):
    Ks: Tuple[int, ...]
    K_min: int
    K_max: int
    dim_per_atom: int
    components_exchangeable: bool = True

    @abstractmethod
    def sample_K(self, key: PRNGKey, n: int) -> jnp.ndarray: ...
    @abstractmethod
    def sample_prior(self, key: PRNGKey, K: int, n: int) -> jnp.ndarray: ...
    @abstractmethod
    def logpmf_K(self, K: int) -> float: ...
    @abstractmethod
    def logprior_full(self, theta_full: jnp.ndarray, Kf: jnp.ndarray) -> jnp.ndarray: ...
    @abstractmethod
    def loglikelihood_full(self, theta_full: jnp.ndarray, Kf: jnp.ndarray) -> jnp.ndarray: ...
