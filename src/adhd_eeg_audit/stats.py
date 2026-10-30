"""Effect sizes, bootstrap CIs, permutation tests, multiplicity control."""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import gammaln


def hedges_j(df: float) -> float:
    """Exact small-sample correction J(df) = Γ(df/2) / (sqrt(df/2) Γ((df-1)/2))."""
    return float(np.exp(gammaln(df / 2) - 0.5 * np.log(df / 2) - gammaln((df - 1) / 2)))


def hedges_g(a, b, axis: int = -1) -> np.ndarray:
    """Hedges' g for mean(a) - mean(b) with pooled SD (vectorised along axis)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = a.shape[axis], b.shape[axis]
    s2 = ((n1 - 1) * a.var(axis=axis, ddof=1) + (n2 - 1) * b.var(axis=axis, ddof=1)) / (n1 + n2 - 2)
    d = (a.mean(axis=axis) - b.mean(axis=axis)) / np.sqrt(s2)
    return d * hedges_j(n1 + n2 - 2)


def auc(a, b) -> float:
    """P(A > B) + 0.5 P(A = B): the ROC AUC of the raw score separating a from b."""
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))


def bootstrap_ci(stat, a, b, n_resamples: int = 10_000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """BCa bootstrap CI, resampling each group independently (stratified)."""
    res = stats.bootstrap((np.asarray(a, float), np.asarray(b, float)), stat,
                          n_resamples=n_resamples, method="BCa", vectorized=True,
                          confidence_level=level, random_state=np.random.default_rng(seed))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def perm_test(a, b, n_resamples: int = 10_000, seed: int = 0) -> float:
    """Two-sided permutation p-value for the difference in means."""
    res = stats.permutation_test((np.asarray(a, float), np.asarray(b, float)),
                                 lambda x, y, axis: x.mean(axis) - y.mean(axis),
                                 permutation_type="independent", vectorized=True,
                                 n_resamples=n_resamples, alternative="two-sided",
                                 random_state=np.random.default_rng(seed))
    return float(res.pvalue)


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def mde_two_sample(n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable |g| (normal approximation) for a two-sided test."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(1 / n1 + 1 / n2))


def icc1(x: np.ndarray, g: np.ndarray) -> float:
    """ICC(1), one-way random effects, unbalanced groups (n0 correction)."""
    subj = np.unique(g)
    k = len(subj)
    n_i = np.array([np.sum(g == s) for s in subj])
    N = n_i.sum()
    means = np.array([x[g == s].mean() for s in subj])
    grand = x.mean()
    ssb = np.sum(n_i * (means - grand) ** 2)
    ssw = sum(np.sum((x[g == s] - m) ** 2) for s, m in zip(subj, means))
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - np.sum(n_i ** 2) / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))
