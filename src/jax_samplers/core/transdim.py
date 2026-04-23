# src/jax_samplers/core/transdim.py
from __future__ import annotations
"""Interfaces for trans-dimensional model families.

This module defines a *protocol* that any trans-dimensional family must implement
so that it can be adapted into a fixed-dimension :class:`Problem` using the
product-space adapter (see ``adapters/transdim_product_space.py``).

The idea: you have a discrete model index ``K`` (e.g., number of components,
polynomial order, basis size, etc.). For each ``K`` the parameter vector lives
in a space of dimension ``dim_of(K)``. The samplers in this package operate on
fixed-dimension spaces, so we provide an adapter that embeds all these models
into a single product space of dimension ``1 + max_K dim_of(K)``. The first
coordinate stores ``K`` (as a float; cast to ``int`` inside the densities), and
remaining coordinates store the active parameters padded with unused entries.

To be model-agnostic, the adapter only relies on the :class:`ModelFamily`
protocol below.
"""

from typing import Protocol, Sequence, runtime_checkable

import jax.numpy as jnp

from .types import PRNGKey


@runtime_checkable
class ModelFamily(Protocol):
    """Protocol for a *family* of models with varying dimensionality.

    Implementations can represent anything where a discrete index ``K`` selects
    a parameter space and corresponding densities/priors/likelihoods. Examples:

    - Gaussian mixtures with unknown number of components
    - Polynomial regression with unknown order
    - Basis expansions / spline models with unknown number of bases
    - State-space models with unknown latent dimension

    Methods operate in the *natural* parameter space for the chosen model
    ``K``. The adapter will take care of padding/embedding these parameters
    into a fixed-dimension product space for use by generic samplers.
    """

    # -------------------------- Model set -------------------------- #
    @property
    def Ks(self) -> Sequence[int]:
        """The allowed model indices (e.g. ``range(1, K_max+1)``).

        This sequence must be non-empty. All other methods should support any
        ``K`` contained in this sequence.
        """
        ...

    def dim_of(self, k: int) -> int:
        """Return the parameter dimension for model index ``k``.

        Parameters
        ----------
        k : int
            A valid model index, i.e. an element of ``self.Ks``.
        """
        ...

    # -------------------------- Priors ----------------------------- #
    def logpmf_K(self, k: int) -> jnp.ndarray:
        """Log prior probability *mass* of the model index ``K``.

        Returns a scalar log-probability corresponding to ``log p(K=k)``.
        """
        ...

    def logprior_k(self, k: int, theta_k: jnp.ndarray) -> jnp.ndarray:
        """Log prior density over parameters for model ``k``.

        Parameters
        ----------
        k : int
            Model index.
        theta_k : jnp.ndarray
            Parameter vector of shape ``(dim_of(k),)`` for model ``k``.
        """
        ...

    # ------------------------ Likelihood --------------------------- #
    def loglikelihood_k(self, k: int, theta_k: jnp.ndarray) -> jnp.ndarray:
        """Log likelihood under model ``k`` evaluated at ``theta_k``.

        Must return a scalar (JAX array). Any required data should be captured
        via closures when the family implementation is constructed.
        """
        ...

    # ---------------------- Prior sampling ------------------------- #
    def sample_prior_k(self, key: PRNGKey, k: int, n: int) -> jnp.ndarray:
        """Draw ``n`` iid samples from the prior of model ``k``.

        Returns an array of shape ``(n, dim_of(k))`` suitable to be embedded
        by the product-space adapter.
        """
        ...


__all__ = ["ModelFamily"]
