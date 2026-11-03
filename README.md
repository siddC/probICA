# probICA

[![CI](https://github.com/siddC/probICA/actions/workflows/ci.yml/badge.svg)](https://github.com/siddC/probICA/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/probica/badge/?version=latest)](https://probica.readthedocs.io)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
![Python 3.12–3.14](https://img.shields.io/badge/python-3.12%E2%80%933.14-blue.svg)

Probabilistic and Bayesian ICA: independent factor analysis, variational Bayesian ICA with automatic source-count selection, positive mean-field, nonnegative and kernel ICA, with a scikit-learn API.

> **Status: pre-alpha.** The shared engine, the API contract and the test harness for the
> correctness standard are in place. The first models land in v0.1; see the
> [roadmap](ROADMAP.md).

## Why probICA

Existing Python ICA packages are point-estimate contrast methods with no way to choose
the number of sources. probICA writes ICA as explicit probabilistic models (source
priors, observation noise, posterior uncertainty, source count by ARD) and as
constrained or nonparametric models (nonnegative sources, kernel dependence measures).
FastICA, Picard, Infomax, AMICA and robust ICA stay in their own packages, and binary
ICA is NBMF, which lives in nbmf-mm.

The fitted solution is an orthonormal frame after whitening (a point on the Stiefel manifold).

## Models

| Class | Model | Inference | Source papers | Release |
| --- | --- | --- | --- | --- |
| `IndependentFactorAnalysis` | x = As + e; each source a mixture of Gaussians | EM (exact for small k, variational for large k) | Attias 1999 | v0.1 |
| `BayesianICA` | IFA with ARD on mixing columns and conjugate priors throughout | Variational Bayes | Choudrey & Roberts 2003; Lawrence & Bishop 2000; Miskin & MacKay 2001 | v0.1 |
| `PositiveMeanFieldICA` | Positive source priors, optionally positive mixing | Mean field (naive, linear response, adaptive TAP); EM | Højen-Sørensen, Winther & Hansen 2002 | v0.2 |
| `NonnegativeICA` | Noise-free; nonnegative, well-grounded sources | Whitening, then rotation minimizing negative-output energy | Plumbley 2003; Plumbley & Oja 2004 | v0.2 |
| `KernelICA` | Noise-free; KCCA or KGV dependence contrast | Stiefel-manifold descent; incomplete Cholesky | Bach & Jordan 2002 | v0.3 |
| Mixture of BayesianICA models | Mixture of variational Bayesian ICA models | Variational Bayes | Choudrey & Roberts 2003 | v0.4 |

Full citations are on each model's documentation page.

## One API for every model

```python
model = SomeModel(n_components=8, random_state=0)
Z = model.fit_transform(X, mask=observed)   # X: (n_samples, n_features); Z: (n_samples, k)
model.components_                           # (n_components_, n_features)
model.score(X_test, mask=heldout)           # mean held-out log-likelihood per observed entry
model.sample(100, random_state=1)           # synthetic data from the fitted model
```

| Member | Contract |
| --- | --- |
| `fit(X, y=None, mask=None)` | Fit the model; returns self |
| `transform(X, mask=None)` | Latent values for new samples, components held fixed |
| `fit_transform(X, mask=None)` | Fit, then return the latent values of the training samples |
| `inverse_transform(Z)` | Mean of the observation model on the data scale |
| `score(X, mask=None)` | Mean log-likelihood per observed entry (posterior predictive for Bayesian models) |
| `score_samples(X, mask=None)` | Log-likelihood per sample |
| `sample(n_samples, random_state=None)` | Synthetic data from the fitted model |
| `components_`, `n_components_` | Fitted components; effective number after any ARD pruning |
| `objective_`, `n_iter_`, `converged_` | Minimized objective per iteration, iterations used, convergence flag |
| `loglik_` | Final training log-likelihood |

Missing entries are NaN in `X` or `False` in `mask`; every likelihood, update and score sums
over observed entries only. float64 by default; float32 input stays float32.

## Correctness

No estimator ships without passing six layers of tests:

1. **Monotone objective**: MM and EM objectives never increase, ELBOs never decrease.
2. **Limiting cases**: each model reduces to its documented special case.
3. **Reproduction**: each source paper's experiments are rerun and must match the published
   numbers to the precision the paper reports.
4. **Recovery**: factors and rank are recovered from data simulated by the model, over 10 seeds.
5. **Stationarity**: KKT conditions hold at convergence.
6. **Conformance**: scikit-learn's estimator checks, pickling, float32 and float64.

Models are implemented from their papers only (no author code is read or translated).
Maintained public packages may be run as black-box cross-checks on shared inputs.

## Install

probICA is not on PyPI yet. Until the first release:

```bash
pip install "git+https://github.com/siddC/probICA"
```

Optional extras: `[jax]` for the JAX backend and `[mcmc]` (NumPyro) for the models that
sample. Python 3.12 to 3.14.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Releases follow [RELEASING.md](RELEASING.md).

## Citation

If you use probICA, please cite it; [CITATION.cff](CITATION.cff) has the details. A JOSS
paper will follow v1.0.

## License

BSD 3-Clause; see [LICENSE](LICENSE).
