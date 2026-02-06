import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def ddm_likelihood(t, choice, mu, sigma, theta, K=25):
    """
    Joint density f(t, choice | mu, sigma, theta) for a two-bound DDM.
    Bounds at +theta and -theta, starting at 0. (Navarro & Fuss, 2009)

    Parameters: t (RT), choice (+1 or -1), mu (drift), sigma (diffusion), theta (bound).
    Returns: density value(s).
    """
    t = np.asarray(t, dtype=float)
    choice = np.asarray(choice, dtype=float)

    a = 2.0 * theta
    w = 0.5
    t_s = (sigma**2 / a**2) * t
    w_eff = np.where(choice == 1, 1.0 - w, w)

    ftt = np.zeros_like(t)
    pos = t_s > 0
    for wv in np.unique(w_eff):
        mask = (w_eff == wv) & pos
        ts_m = t_s[mask] if t_s.ndim > 0 else np.atleast_1d(t_s)

        small = ts_m < 1.0 / (2.0 * np.pi)
        large = ~small

        s = np.zeros_like(ts_m)
        if np.any(small):
            ts_sm = ts_m[small]
            ss = np.zeros_like(ts_sm)
            for k in range(-K, K + 1):
                d = wv + 2.0 * k
                ss += d * np.exp(-d**2 / (2.0 * ts_sm))
            s[small] = ss / np.sqrt(2.0 * np.pi * ts_sm**3)

        if np.any(large):
            ts_lg = ts_m[large]
            sl = np.zeros_like(ts_lg)
            for k in range(1, K + 1):
                sl += k * np.sin(k * np.pi * wv) * np.exp(-k**2 * np.pi**2 * ts_lg / 2.0)
            s[large] = np.pi * sl

        if ftt.ndim == 0:
            ftt = (sigma**2 / a**2) * s[0]
        else:
            ftt[mask] = (sigma**2 / a**2) * s

    drift_factor = np.exp(choice * mu * theta / sigma**2 - mu**2 * t / (2.0 * sigma**2))
    return np.maximum(ftt * drift_factor, 0.0)


def ddm_log_likelihood(rts, choices, mu, sigma, theta, K=25):
    """Total log-likelihood over a set of (RT, choice) observations."""
    densities = ddm_likelihood(rts, choices, mu, sigma, theta, K=K)
    densities = np.maximum(densities, 1e-300)
    return np.sum(np.log(densities))


# ── DDM Simulation ───────────────────────────────────────────────────────────
def simulate_ddm(mu, sigma, theta, dt=1e-4):
    """Simulate one DDM trial. Returns (RT, choice)."""
    t = 0.0
    DV = 0.0
    while True:
        DV += mu * dt + sigma * np.sqrt(dt) * np.random.randn()
        t += dt
        if DV >= theta:
            return t, 1
        elif DV <= -theta:
            return t, -1


def run_ddm_simulations(mu, sigma, theta, n_trials, dt=1e-4):
    rts = np.empty(n_trials)
    choices = np.empty(n_trials)
    for i in tqdm(range(n_trials)):
        rts[i], choices[i] = simulate_ddm(mu, sigma, theta, dt=dt)
    return rts, choices


# ── Main: simulate + validate ────────────────────────────────────────────────
if __name__ == "__main__":
    mu_true = 1.0
    sigma_true = 1.5
    theta_true = 1.0
    n_trials = 50_000

    print(f"Simulating {n_trials} DDM trials  (mu={mu_true}, sigma={sigma_true}, theta={theta_true}) ...")
    rts, choices = run_ddm_simulations(mu_true, sigma_true, theta_true, n_trials)

    n_upper = int(np.sum(choices == 1))
    n_lower = int(np.sum(choices == -1))
    print(f"  Upper-bound hits: {n_upper}  ({100*n_upper/n_trials:.1f}%)")
    print(f"  Lower-bound hits: {n_lower}  ({100*n_lower/n_trials:.1f}%)")
    print(f"  Mean RT: {np.mean(rts):.4f} s")

    # --- Log-likelihood at the true parameters ---
    ll_true = ddm_log_likelihood(rts, choices, mu_true, sigma_true, theta_true)
    print(f"\nLog-likelihood at TRUE params: {ll_true:.2f}")

    # --- Log-likelihood at wrong parameters (should be lower) ---
    ll_wrong = ddm_log_likelihood(rts, choices, mu_true + 1.0, sigma_true, theta_true)
    print(f"Log-likelihood at WRONG mu:    {ll_wrong:.2f}")
    assert ll_true > ll_wrong, "True params should have higher LL than wrong params!"
    print("✓  True params beat wrong params in log-likelihood.\n")

    # --- Visual validation ---
    t_grid = np.linspace(0.001, np.percentile(rts, 99), 500)

    # Plot 1: Total RT density (sum over both choices) vs histogram of all RTs
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    total_lik = ddm_likelihood(t_grid, 1, mu_true, sigma_true, theta_true) + \
                ddm_likelihood(t_grid, -1, mu_true, sigma_true, theta_true)
    axes[0].hist(rts, bins=100, density=True, alpha=0.5, label="Simulated RTs")
    axes[0].plot(t_grid, total_lik, "r-", lw=2, label="Analytical (sum of choices)")
    axes[0].set_title("All RTs: f(t,+1) + f(t,-1)")
    axes[0].set_xlabel("RT (s)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Plot 2 & 3: Per-choice densities
    for ax, c, label in zip(axes[1:], [1, -1], ["Upper (+θ)", "Lower (−θ)"]):
        rts_c = rts[choices == c]
        frac = len(rts_c) / n_trials

        ax.hist(rts_c, bins=80, density=True, alpha=0.5, label="Simulation")
        pdf_vals = ddm_likelihood(t_grid, c, mu_true, sigma_true, theta_true)
        ax.plot(t_grid, pdf_vals / frac, "r-", lw=2, label="Analytical (conditional)")

        ax.set_title(f"Choice = {label}")
        ax.set_xlabel("RT (s)")
        ax.set_ylabel("Density")
        ax.legend()

    plt.suptitle("DDM Likelihood Validation: Analytical vs Simulated", fontsize=13)
    plt.tight_layout()
    plt.savefig("ddm_likelihood_validation.png", dpi=150)
    plt.show()
    print("Saved plot to ddm_likelihood_validation.png")
