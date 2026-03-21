import numpy as np
def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    # Benjamini-Hochberg FDR correction.
    pvals = np.asarray(pvals)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out