"""Compare DDM trajectories for two parameter sets."""

import numpy as np
import matplotlib.pyplot as plt

__all__ = ["compare_ddm_params"]


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


def plot_trajectories(ax, t, trajectories, hit_keep, theta, rts, title):
    for x, hit_idx in zip(trajectories, hit_keep):
        x_plot = x.copy()
        if hit_idx >= 0 and hit_idx + 1 < x_plot.size:
            x_plot[hit_idx + 1 :] = np.nan
        ax.plot(t, x_plot, color="#1f77b4", alpha=0.05, linewidth=1)

    ax.axhline(+theta, linestyle="--", linewidth=1, color="#333333")
    ax.axhline(-theta, linestyle="--", linewidth=1, color="#333333")

    ax.set_title(title, fontsize=28)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.2 * theta, 1.45 * theta)

    ax.set_yticks([-theta, 0.0, +theta])
    ax.set_yticklabels(["-θ", "0", "+θ"], fontsize=20)

    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("time (s)", fontsize=20)
    ax.set_ylabel("DV", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    t_density = np.arange(0.01, t[-1] + 1e-9, 0.01)
    if rts.size >= 2:
        sigma = np.std(rts)
        bandwidth = 2.34 * sigma * (rts.size ** (-1 / 5))
        bandwidth = max(bandwidth, 1e-3)
        density = epanechnikov_kde(rts, t_density, bandwidth)
        if density.max() > 0:
            density = density / density.max()
            height = 0.4 * theta
            baseline = theta
            scaled = baseline + height * density
            ax.plot(t_density, scaled, color="#1f77b4", linewidth=2)
            ax.fill_between(t_density, baseline, scaled, color="#1f77b4", alpha=0.2)


def compare_ddm_params(
    params_a,
    params_b,
    *,
    dt=0.001,
    T=1.0,
    n_trials=int(1e4),
    n_plot=300,
    seed_a=1,
    seed_b=2,
):
    required = ("mu", "sigma", "theta")
    for name, params in (("params_a", params_a), ("params_b", params_b)):
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"{name} missing keys: {', '.join(missing)}")

    t_a, traj_a, rt_a, hit_a = simulate_ddm_batch(
        params_a["mu"],
        params_a["sigma"],
        params_a["theta"],
        dt,
        T,
        n_trials,
        seed=seed_a,
        n_keep=n_plot,
    )
    t_b, traj_b, rt_b, hit_b = simulate_ddm_batch(
        params_b["mu"],
        params_b["sigma"],
        params_b["theta"],
        dt,
        T,
        n_trials,
        seed=seed_b,
        n_keep=n_plot,
    )

    if t_b.size != t_a.size or not np.allclose(t_b, t_a):
        raise ValueError("Time grids do not match between parameter sets.")

    diff_keys = [
        key for key in required if not np.isclose(params_a[key], params_b[key])
    ]
    common_keys = [key for key in required if key not in diff_keys]
    symbol_map = {"mu": "μ", "sigma": "σ", "theta": "θ"}
    diff_label = (
        ", ".join(symbol_map.get(key, key) for key in diff_keys)
        if diff_keys
        else "none"
    )

    def format_values(keys, params):
        return ", ".join(
            f"{symbol_map.get(key, key)}={params[key]:g}" for key in keys
        )

    title_a = format_values(diff_keys, params_a) if diff_keys else "set 1"
    title_b = format_values(diff_keys, params_b) if diff_keys else "set 2"
    common_label = format_values(common_keys, params_a) if common_keys else ""

    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
    plot_trajectories(
        axes[0], t_a, traj_a, hit_a, params_a["theta"], rt_a, title_a
    )
    plot_trajectories(
        axes[1], t_a, traj_b, hit_b, params_b["theta"], rt_b, title_b
    )

    if common_label:
        fig.suptitle(
            f"DDM comparison (varying: {diff_label}; common: {common_label})",
            y=1.02,
            fontsize=32,
        )
    else:
        fig.suptitle(f"DDM comparison (varying: {diff_label})", y=1.02, fontsize=32)
    fig.tight_layout()
    plt.show()

    return fig, axes


