# -*- coding: utf-8 -*-
"""
Isolated VBMC first-fit section from tutorial_draft (1).py.
Use # %% cells to run/debug in VS Code or Jupyter.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(x, **kwargs):
        return x

# %%
# DDM simulator (vectorized, dt=1e-4 to minimise discretisation bias)

def run_ddm_simulations(
    mu,
    sigma,
    theta,
    n_trials,
    dt=1e-4,
    show_progress=True,
):
    """
    Vectorised Euler–Maruyama DDM simulation.
    Returns (rts, choices) arrays of length n_trials.
    """
    rts = np.zeros(n_trials)
    choices = np.zeros(n_trials)
    dv = np.zeros(n_trials)
    active = np.ones(n_trials, dtype=bool)
    t = 0.0

    while np.any(active):
        n_active = int(np.sum(active))
        dv[active] += mu * dt + sigma * np.random.normal(0, dt**0.5, size=n_active)
        t += dt

        hit_upper = active & (dv >= theta)
        hit_lower = active & (dv <= -theta)

        rts[hit_upper] = t
        choices[hit_upper] = 1
        rts[hit_lower] = t
        choices[hit_lower] = -1

        active[hit_upper | hit_lower] = False

    return rts, choices

# %%
# Simulate ground-truth data (used for VBMC fitting)

mu = 200
sigma = 100
theta = 40
N_trials = 50_000
RTs_ground_truth, choices_ground_truth = run_ddm_simulations(mu, sigma, theta, N_trials)

bin_width = 10e-3
bins = np.arange(0, 1, bin_width)
plt.hist(RTs_ground_truth, bins=bins, density=True, histtype='step')
plt.xlabel('RT')
plt.ylabel('Density')
plt.title('Ground-truth RTD')

# %%
# Likelihood (Navarro & Fuss, 2009)

def ddm_total_likelihood(t, mu, sigma, theta, K=10):
    """
    Total first-passage time density for a two-bound DDM, marginalized over choice:
        f(t | mu, sigma, theta) = f(t, +1) + f(t, -1)

    Bounds at +theta and -theta, starting at 0.
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

    f0 = scale * ftt

    exp_common = np.exp(-mu**2 * t[pos] / (2.0 * sigma**2))
    f_upper = f0 * np.exp(mu * theta / sigma**2) * exp_common
    f_lower = f0 * np.exp(-mu * theta / sigma**2) * exp_common

    result = np.zeros_like(t)
    result[pos] = np.maximum(f_upper + f_lower, 0.0)
    return result

# %%
# Install/import VBMC (script-friendly)
try:
    from pyvbmc import VBMC
except ModuleNotFoundError:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvbmc"])
    from pyvbmc import VBMC

# %%
# Log-likelihood and priors

def loglike_function(params):
    mu_p, sigma_p, theta_p = params
    liks = ddm_total_likelihood(RTs_ground_truth, mu_p, sigma_p, theta_p)
    return np.sum(np.log(np.maximum(liks, 1e-300)))


mu_bounds = [150, 250]
sigma_bounds = [50, 150]
theta_bounds = [20, 60]

mu_plausible_bounds = [195, 205]
sigma_plausible_bounds = [95, 105]
theta_plausible_bounds = [35, 45]


def log_prior(params):
    mu_p, sigma_p, theta_p = params

    log_prior_mu = np.log(1 / (mu_bounds[1] - mu_bounds[0]))
    log_prior_sigma = np.log(1 / (sigma_bounds[1] - sigma_bounds[0]))
    log_prior_theta = np.log(1 / (theta_bounds[1] - theta_bounds[0]))

    return log_prior_mu + log_prior_sigma + log_prior_theta


def prior_plus_likelihood_fn(params):
    return log_prior(params) + loglike_function(params)

# %%
# VBMC setup and first fit

lb = np.array([[mu_bounds[0], sigma_bounds[0], theta_bounds[0]]], dtype=np.float64)
ub = np.array([[mu_bounds[1], sigma_bounds[1], theta_bounds[1]]], dtype=np.float64)
plb = np.array(
    [[mu_plausible_bounds[0], sigma_plausible_bounds[0], theta_plausible_bounds[0]]],
    dtype=np.float64,
)
pub = np.array(
    [[mu_plausible_bounds[1], sigma_plausible_bounds[1], theta_plausible_bounds[1]]],
    dtype=np.float64,
)

np.random.seed(42)
mu_0 = np.random.uniform(mu_plausible_bounds[0], mu_plausible_bounds[1])
sigma_0 = np.random.uniform(sigma_plausible_bounds[0], sigma_plausible_bounds[1])
theta_0 = np.random.uniform(theta_plausible_bounds[0], theta_plausible_bounds[1])

x0 = np.array([[mu_0, sigma_0, theta_0]])

vbmc = VBMC(prior_plus_likelihood_fn, x0, lb, ub, plb, pub, options={"display": "on"})
vp, results = vbmc.optimize()

# %%
# Posterior samples and summary

try:
    import corner
    has_corner = True
except ModuleNotFoundError:
    has_corner = False

vp_samples = vp.sample(int(1e5))[0]
param_labels = ['mu', 'sigma', 'theta']
true_arr = [mu, sigma, theta]

if has_corner:
    corner.corner(
        vp_samples,
        labels=param_labels,
        show_titles=True,
        quantiles=[0.025, 0.5, 0.975],
        truths=true_arr,
        title_fmt=".2f",
    )

mu_fit = np.mean(vp_samples[:, 0])
sigma_fit = np.mean(vp_samples[:, 1])
theta_fit = np.mean(vp_samples[:, 2])

mu_pct_diff = abs((mu_fit - mu) / mu) * 100
sigma_pct_diff = abs((sigma_fit - sigma) / sigma) * 100
theta_pct_diff = abs((theta_fit - theta) / theta) * 100

print(f"{'Parameter':<15} | {'Ground Truth':<15} | {'VBMC Fit':<15} | {'Abs % Change':<15}")
print("-" * 70)
print(f"{'mu':<15} | {mu:<15.2f} | {mu_fit:<15.2f} | {mu_pct_diff:<15.2f}")
print(f"{'sigma':<15} | {sigma:<15.2f} | {sigma_fit:<15.2f} | {sigma_pct_diff:<15.2f}")
print(f"{'theta':<15} | {theta:<15.2f} | {theta_fit:<15.2f} | {theta_pct_diff:<15.2f}")

# %%
# Compare RTDs: ground truth vs fitted

N_trials_fit = 50_000
RTs_fit, choices_fit = run_ddm_simulations(mu_fit, sigma_fit, theta_fit, N_trials_fit)

bins = np.arange(0, 1, 0.01)
plt.figure()
plt.hist(RTs_ground_truth, histtype='step', density=True, bins=bins, label='Ground Truth', color='b')
plt.hist(RTs_fit, histtype='step', density=True, bins=bins, label='Fitted', color='r')
plt.xlabel('RT')
plt.ylabel('Density')
plt.legend()
plt.title('RTD: Ground Truth vs VBMC Fit')
plt.savefig('rtd_comparison.png', dpi=150)
print("Saved rtd_comparison.png")