# %%
"""Compare DDM trajectories for two drift values."""

import numpy as np
import matplotlib.pyplot as plt


def epanechnikov_kde(samples, grid, bandwidth):
    if samples.size == 0:
        return np.zeros_like(grid)
    u = (grid[:, None] - samples[None, :]) / bandwidth
    weights = 0.75 * (1 - u**2)
    weights[np.abs(u) > 1] = 0
    return weights.sum(axis=1) / (samples.size * bandwidth)


def simulate_ddm_batch(mu, sigma, theta, dt, T, n_trials, seed, n_keep=300):
    rng = np.random.default_rng(seed)
    n_steps = int(np.round(T / dt)) + 1
    t = np.arange(n_steps) * dt

    x = np.zeros(n_trials, dtype=float)
    hit = np.full(n_trials, -1, dtype=int)

    keep_idx = np.arange(min(n_keep, n_trials))
    trajectories = np.zeros((keep_idx.size, n_steps), dtype=float)

    for i in range(1, n_steps):
        active = hit < 0
        if np.any(active):
            x[active] += (
                mu * dt + sigma * np.sqrt(dt) * rng.normal(size=active.sum())
            )
            new_hits = active & (np.abs(x) >= theta)
            hit[new_hits] = i

        if keep_idx.size:
            trajectories[:, i] = x[keep_idx]

    rts = hit[hit >= 0] * dt
    hit_keep = hit[keep_idx]
    return t, trajectories, rts, hit_keep


def plot_trajectories(ax, t, trajectories, hit_keep, theta, title):
    for x, hit_idx in zip(trajectories, hit_keep):
        x_plot = x.copy()
        if hit_idx >= 0 and hit_idx + 1 < x_plot.size:
            x_plot[hit_idx + 1 :] = np.nan
        ax.plot(t, x_plot, color="#1f77b4", alpha=0.05, linewidth=1)

    ax.axhline(+theta, linestyle="--", linewidth=1, color="#333333")
    ax.axhline(-theta, linestyle="--", linewidth=1, color="#333333")

    ax.set_title(title)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.2 * theta, 1.4 * theta)

    ax.set_yticks([-theta, 0.0, +theta])
    ax.set_yticklabels(["-theta", "0", "+theta"])

    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("DV")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_density(ax, T, rts, title):
    t_density = np.arange(0.01, T + 1e-9, 0.01)
    if rts.size < 2:
        density = np.zeros_like(t_density)
    else:
        sigma = np.std(rts)
        bandwidth = 2.34 * sigma * (rts.size ** (-1 / 5))
        bandwidth = max(bandwidth, 1e-3)
        density = epanechnikov_kde(rts, t_density, bandwidth)
        if density.max() > 0:
            density = density / density.max()

    ax.plot(t_density, density, color="#1f77b4", linewidth=2)
    ax.fill_between(t_density, 0, density, color="#1f77b4", alpha=0.2)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_ylabel("density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    dt = 0.001
    T = 1.0
    theta = 50.0
    sigma = 100.0
    n_trials = int(1e4)
    n_plot = 300

    mu_small = 1.0
    mu_large = 500.0

    t, traj_small, rt_small, hit_small = simulate_ddm_batch(
        mu_small, sigma, theta, dt, T, n_trials, seed=1, n_keep=n_plot
    )
    _, traj_large, rt_large, hit_large = simulate_ddm_batch(
        mu_large, sigma, theta, dt, T, n_trials, seed=2, n_keep=n_plot
    )

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 6), sharex="col", gridspec_kw={"height_ratios": [1, 3]}
    )
    plot_density(axes[0, 0], T, rt_small, f"RTD (mu = {mu_small:.0f})")
    plot_density(axes[0, 1], T, rt_large, f"RTD (mu = {mu_large:.0f})")
    plot_trajectories(
        axes[1, 0], t, traj_small, hit_small, theta, f"mu = {mu_small:.0f}"
    )
    plot_trajectories(
        axes[1, 1], t, traj_large, hit_large, theta, f"mu = {mu_large:.0f}"
    )

    fig.suptitle("DDM trajectories: small vs large drift", y=1.02)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# %%
