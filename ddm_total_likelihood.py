import numpy as np


def ddm_total_likelihood(t, mu, sigma, theta, K=25):
    """
    Total first-passage time density for a two-bound DDM, marginalized over choice:
        f(t | mu, sigma, theta) = f(t, +1) + f(t, -1)

    Bounds at +theta and -theta, starting at 0. (Navarro & Fuss, 2009)

    Parameters
    ----------
    t     : float or array – reaction time(s), must be > 0
    mu    : float          – drift rate
    sigma : float          – diffusion coefficient (> 0)
    theta : float          – bound height (> 0), bounds at ±theta
    K     : int            – number of series terms (default 25)

    Returns
    -------
    density : array – total density f(t,+1) + f(t,-1) for each RT
    """
    t = np.asarray(t, dtype=float)

    a = 2.0 * theta
    w = 0.5
    t_s = (sigma**2 / a**2) * t
    scale = sigma**2 / a**2

    pos = t_s > 0
    ts_p = t_s[pos] if t_s.ndim > 0 else np.atleast_1d(t_s)

    small = ts_p < 1.0 / (2.0 * np.pi)
    large = ~small

    ftt = np.zeros_like(ts_p)

    if np.any(small):
        ts_sm = ts_p[small]
        s = np.zeros_like(ts_sm)
        for k in range(-K, K + 1):
            d = w + 2.0 * k
            s += d * np.exp(-d**2 / (2.0 * ts_sm))
        ftt[small] = s / np.sqrt(2.0 * np.pi * ts_sm**3)

    if np.any(large):
        ts_lg = ts_p[large]
        s = np.zeros_like(ts_lg)
        for k in range(1, K + 1):
            s += k * np.sin(k * np.pi * w) * np.exp(-k**2 * np.pi**2 * ts_lg / 2.0)
        ftt[large] = np.pi * s

    # f0 is the zero-drift first-passage density (same for both bounds by symmetry)
    f0 = scale * ftt

    # drift factors for upper (+1) and lower (-1) bounds
    exp_common = np.exp(-mu**2 * t[pos] / (2.0 * sigma**2))
    f_upper = f0 * np.exp(mu * theta / sigma**2) * exp_common
    f_lower = f0 * np.exp(-mu * theta / sigma**2) * exp_common

    result = np.zeros_like(t)
    result[pos] = np.maximum(f_upper + f_lower, 0.0)
    return result
