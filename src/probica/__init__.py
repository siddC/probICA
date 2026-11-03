"""probica: probabilistic and Bayesian ICA.

ICA as explicit probabilistic models (sources with stated priors, observation
noise, posterior uncertainty and source count by ARD) and as constrained or
nonparametric models (nonnegative sources, kernel dependence contrasts).

Every estimator shares one API: ``X`` is (n_samples, n_features), missing entries are NaN or
``mask=False``, ``score`` is the mean log-likelihood per observed entry, and ``sample`` draws
from the fitted model. See https://probica.readthedocs.io for the model pages.
"""

from . import datasets, metrics
from ._base import BaseFactorization

__version__ = "0.1.0.dev0"

__all__ = ["BaseFactorization", "datasets", "metrics", "__version__"]
