probICA
=======

Probabilistic and Bayesian ICA: independent factor analysis, variational Bayesian ICA with automatic source-count selection, positive mean-field, nonnegative and kernel ICA, with a scikit-learn API.

.. note::

   Pre-alpha. The shared engine, the API contract and the test harness are in place; the
   first models land in v0.1.

Existing Python ICA packages are point-estimate contrast methods with no way to choose
the number of sources. probICA writes ICA as explicit probabilistic models (source
priors, observation noise, posterior uncertainty, source count by ARD) and as
constrained or nonparametric models (nonnegative sources, kernel dependence measures).
FastICA, Picard, Infomax, AMICA and robust ICA stay in their own packages, and binary
ICA is NBMF, which lives in nbmf-mm.

The fitted solution is an orthonormal frame after whitening (a point on the Stiefel manifold).

.. toctree::
   :maxdepth: 2

   install
   models
   api_contract
   correctness
   auto_examples/index
   api
