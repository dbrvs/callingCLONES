#!/usr/bin/env python
"""Core stats for clonotype expansion/contraction calling.

Downsamples two abundance columns (t1, t2) for the same clonotypes to a
common sample size via multinomial resampling, then tests each clonotype's
t2 count against the null rate implied by its t1 proportion, using either
an exact binomial null (rho=0) or a beta-binomial null (rho>0) to allow
for extra-binomial (overdispersion) variance.
"""

import numpy as np
import scipy.stats as st


def downsample_counts(counts, ss, rng=None):
    """Multinomial-resample a vector of counts down (or up) to total ss."""
    rng = rng if rng is not None else np.random
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Column sums to zero; cannot downsample.")
    if ss <= 0:
        raise ValueError("Sample size must be positive.")
    p = counts / total
    return rng.multinomial(int(ss), p)


def call_clonotypes(c0_ds, c1_ds, ss, rho=0.0, alpha=0.05):
    """Test each clonotype's downsampled t2 count against the null rate
    implied by its downsampled t1 proportion.

    rho <= 0 uses an exact binomial null: k1 ~ Binomial(ss, p0).
    rho > 0  uses a beta-binomial null with intraclass correlation rho,
    which widens the null to allow for extra-binomial variance.

    Returns (pvals, log2fc, call) where call is one of
    'expansion' / 'contraction' / 'ns'.
    """
    c0_ds = np.asarray(c0_ds, dtype=float)
    c1_ds = np.asarray(c1_ds, dtype=float)
    k = c1_ds

    p0 = c0_ds / ss

    if rho <= 0:
        dist = st.binom(ss, p0)
    else:
        rho = min(rho, 0.999)
        p0_clip = np.clip(p0, 1e-12, 1 - 1e-12)
        a = p0_clip * (1 / rho - 1)
        b = (1 - p0_clip) * (1 / rho - 1)
        dist = st.betabinom(ss, a, b)

    mean = dist.mean()
    upper = dist.sf(k - 1)  # P(K >= k)
    lower = dist.cdf(k)     # P(K <= k)
    pvals = 2 * np.minimum(np.where(k >= mean, upper, lower), 0.5)
    pvals = np.clip(pvals, np.finfo(float).tiny, 1.0)

    p1 = c1_ds / ss
    fc = np.log2(p1 / p0)

    call = np.where(pvals >= alpha, "ns", np.where(fc > 0, "expansion", "contraction"))

    return pvals, fc, call


def benjamini_hochberg_threshold(pvals, q=0.05):
    """Benjamini-Hochberg p-value cutoff controlling the false discovery
    rate at q, given a set of p-values from independent-ish tests.

    Sorts p-values ascending and finds the largest rank k such that
    p_(k) <= (k/n)*q; every p-value at or below p_(k) is "significant" at
    this FDR. Returns None if no p-value satisfies the criterion (nothing
    passes at this q), since there is then no valid cutoff to report.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return None
    sorted_p = np.sort(pvals)
    ranks = np.arange(1, n + 1)
    below = sorted_p <= (ranks / n) * q
    if not below.any():
        return None
    k = np.max(np.nonzero(below)[0]) + 1
    return float(sorted_p[k - 1])
