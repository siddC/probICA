"""Metrics for comparing fitted factors with each other or with a known truth.

Factorizations are identifiable only up to a permutation of their components (and, depending
on the model, a scaling or sign), so recovery is measured after matching components.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["amari_index", "match_components"]


def _similarity(A, B, metric):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(
            "A and B must be 2-D with the same number of columns, got "
            f"{np.shape(A)} and {np.shape(B)}."
        )
    if metric == "correlation":
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)
    elif metric != "cosine":
        raise ValueError(f"metric must be 'correlation' or 'cosine', got {metric!r}.")
    na = np.linalg.norm(A, axis=1, keepdims=True)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = (A @ B.T) / (na * nb.T)
    return np.nan_to_num(S, nan=0.0)


def match_components(A, B, *, metric="correlation"):
    """Match the rows of ``A`` to the rows of ``B`` by the Hungarian algorithm.

    Each row is a component (for example a row of ``components_``). The assignment maximizes
    the total absolute similarity, so a component whose sign flipped still matches.

    Parameters
    ----------
    A : array-like of shape (k_a, n_features)
        First set of components.
    B : array-like of shape (k_b, n_features)
        Second set of components.
    metric : {"correlation", "cosine"}, default="correlation"
        Similarity between two components.

    Returns
    -------
    rows : ndarray of shape (min(k_a, k_b),)
        Indices into ``A``.
    cols : ndarray of shape (min(k_a, k_b),)
        Indices into ``B``; ``A[rows[i]]`` is matched to ``B[cols[i]]``.
    similarity : ndarray of shape (min(k_a, k_b),)
        Signed similarity of each matched pair.

    Examples
    --------
    >>> import numpy as np
    >>> from probica.metrics import match_components
    >>> A = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]])
    >>> rows, cols, sim = match_components(A, A[::-1])
    >>> cols.tolist(), np.round(sim, 6).tolist()
    ([1, 0], [1.0, 1.0])
    """
    S = _similarity(A, B, metric)
    rows, cols = linear_sum_assignment(-np.abs(S))
    return rows, cols, S[rows, cols]


def amari_index(W, A):
    r"""Amari index of an unmixing matrix against the true mixing matrix.

    With P = W A, the index is zero exactly when P is a scaled permutation, that is when
    every source is recovered up to order and scale (Amari, Cichocki & Yang 1996). This
    version is normalized to [0, 1]:

    .. math::

        \frac{1}{2k(k-1)} \Big[\sum_i \Big(\sum_j \frac{|p_{ij}|}{\max_l |p_{il}|} - 1\Big)
        + \sum_j \Big(\sum_i \frac{|p_{ij}|}{\max_l |p_{lj}|} - 1\Big)\Big]

    Parameters
    ----------
    W : array-like of shape (k, n_features)
        Estimated unmixing matrix (``components_``).
    A : array-like of shape (n_features, k)
        True mixing matrix.

    Returns
    -------
    index : float
        0 for perfect separation, 1 for the worst case.

    Examples
    --------
    >>> import numpy as np
    >>> from probica.metrics import amari_index
    >>> A = np.array([[2.0, 1.0], [1.0, 3.0]])
    >>> W = np.linalg.inv(A)[::-1] * np.array([[5.0], [-2.0]])
    >>> round(amari_index(W, A), 12)
    0.0
    """
    W = np.asarray(W, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    if W.ndim != 2 or A.ndim != 2 or W.shape[1] != A.shape[0] or W.shape[0] != A.shape[1]:
        raise ValueError(
            f"W must be (k, n_features) and A (n_features, k), got {W.shape} and {A.shape}."
        )
    P = np.abs(W @ A)
    k = P.shape[0]
    if k < 2:
        return 0.0
    rows = (P / P.max(axis=1, keepdims=True)).sum(axis=1) - 1.0
    cols = (P / P.max(axis=0, keepdims=True)).sum(axis=0) - 1.0
    return float((rows.sum() + cols.sum()) / (2.0 * k * (k - 1)))
