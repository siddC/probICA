Models
======

Every model states its likelihood, fits it with an objective that provably improves at each
iteration, handles missing entries and reports a held-out log-likelihood. Each model gets its
own page here when it is released.

.. list-table::
   :header-rows: 1
   :widths: 18 30 22 22 8

   * - Class
     - Model
     - Inference
     - Source papers
     - Release
   * - ``IndependentFactorAnalysis``
     - x = As + e; each source a mixture of Gaussians
     - EM (exact for small k, variational for large k)
     - Attias 1999
     - v0.1
   * - ``BayesianICA``
     - IFA with ARD on mixing columns and conjugate priors throughout
     - Variational Bayes
     - Choudrey & Roberts 2003; Lawrence & Bishop 2000; Miskin & MacKay 2001
     - v0.1
   * - ``PositiveMeanFieldICA``
     - Positive source priors, optionally positive mixing
     - Mean field (naive, linear response, adaptive TAP); EM
     - Højen-Sørensen, Winther & Hansen 2002
     - v0.2
   * - ``NonnegativeICA``
     - Noise-free; nonnegative, well-grounded sources
     - Whitening, then rotation minimizing negative-output energy
     - Plumbley 2003; Plumbley & Oja 2004
     - v0.2
   * - ``KernelICA``
     - Noise-free; KCCA or KGV dependence contrast
     - Stiefel-manifold descent; incomplete Cholesky
     - Bach & Jordan 2002
     - v0.3
   * - Mixture of BayesianICA models
     - Mixture of variational Bayesian ICA models
     - Variational Bayes
     - Choudrey & Roberts 2003
     - v0.4
