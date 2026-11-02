"""Paired significance tests on per-question outcomes.

  paired_bootstrap  mean difference (b - a) with a 95% percentile CI, resampling QUESTIONS (the independent unit);
                    per-question values are first averaged over repeats.
  mcnemar           exact two-sided McNemar test on binary correctness, using the majority vote over repeats as
                    each question's outcome per system.
  mean_std          mean and sample std over repeats (the run-to-run variability).
"""

import numpy as np
from scipy.stats import binomtest


def mean_std(values):
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.size > 0
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled mean difference crosses zero
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"mean_diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "p_boot": float(min(p, 1.0)),
            "n": int(d.size)}


def mcnemar(a_bin, b_bin) -> dict:
    a, b = np.asarray(a_bin, bool), np.asarray(b_bin, bool)
    only_a = int((a & ~b).sum())          # a right, b wrong
    only_b = int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
    return {"a_only": only_a, "b_only": only_b, "p_mcnemar": p}
