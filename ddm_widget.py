"""Minimal DDM widget with drift/noise/bound sliders."""

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipywidgets import VBox, HBox, Layout
from IPython.display import display, clear_output

DEFAULTS = dict(
    mu=0.0,        # drift (s^-1)
    sigma=104.0,   # noise (s^-1/2)
    theta=50.0,    # bound
)


def ddm_widget(dt=0.001, T=1.0, defaults=None):
    defaults = {**DEFAULTS, **(defaults or {})}

    def simulate_ddm(mu, sigma, theta, seed=None):
        rng = np.random.default_rng(seed)
        n = int(np.round(T / dt)) + 1
        t = np.arange(n) * dt

        x = np.zeros(n, dtype=float)
        hit = None

        for i in range(1, n):
            x[i] = x[i - 1] + mu * dt + sigma * np.sqrt(dt) * rng.normal()
            if hit is None and abs(x[i]) >= theta:
                hit = i
                x[i:] = x[i]
                break

        return t, x, hit

    mu_slider = widgets.FloatSlider(
        value=defaults["mu"], min=-441.0, max=441.0, step=1.0,
        description='drift μ', continuous_update=False,
        layout=Layout(width='300px'),
    )

    sigma_slider = widgets.FloatSlider(
        value=defaults["sigma"], min=103.0, max=104.0, step=1.0,
        description='noise σ', continuous_update=False,
        layout=Layout(width='300px'),
    )

    theta_slider = widgets.FloatSlider(
        value=defaults["theta"], min=40.0, max=60.0, step=1.0,
        description='bound θ', continuous_update=False,
        layout=Layout(width='300px'),
    )

    new_trial_btn = widgets.Button(
        description='New trial (reset)', button_style='primary',
        layout=Layout(width='160px'),
    )

    k_slider = widgets.FloatSlider(
        value=0.0, min=0.0, max=T, step=dt,
        description='time (s)', continuous_update=True,
        layout=Layout(width='460px'),
    )

    out = widgets.Output()

    state = {"t": None, "x": None, "hit": None, "seed": 0, "suspend": False}

    def regen():
        state["seed"] += 1
        t, x, hit = simulate_ddm(
            mu_slider.value,
            sigma_slider.value,
            theta_slider.value,
            seed=state["seed"],
        )
        state["t"], state["x"], state["hit"] = t, x, hit

    def draw(t_sec):
        with out:
            clear_output(wait=True)

            t = state["t"]
            x = state["x"]
            theta = theta_slider.value
            hit = state["hit"]

            k = int(np.round(t_sec / dt))
            k = max(0, min(k, len(t) - 1))

            k_eff = min(k, hit) if (hit is not None) else k

            fig, ax = plt.subplots(figsize=(6.6, 2.8))

            ax.axhline(+theta, linestyle='--', linewidth=1)
            ax.axhline(-theta, linestyle='--', linewidth=1)

            ax.plot(t[:k_eff + 1], x[:k_eff + 1], linewidth=2)
            ax.scatter([t[k_eff]], [x[k_eff]], s=26)

            ax.text(0.99, 0.92, f"t = {t[k_eff]:.3f} s",
                    transform=ax.transAxes, fontsize=13, va='top', ha='right')

            ax.set_xlim(0, T)
            ax.set_ylim(-1.2 * theta, 1.2 * theta)

            ax.set_yticks([-theta, 0.0, +theta])
            ax.set_yticklabels(["-θ", "0", "+θ"])

            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_xlabel("time (s)")
            ax.set_ylabel("DV")

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            fig.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.24)

            plt.show()

    def safe_reset_all(_=None):
        state["suspend"] = True
        try:
            regen()
            k_slider.value = 0.0
        finally:
            state["suspend"] = False
        draw(0.0)

    def on_param_change(_):
        if state["suspend"]:
            return
        state["suspend"] = True
        try:
            regen()
            k_slider.value = 0.0
        finally:
            state["suspend"] = False
        draw(0.0)

    def on_time_change(change):
        if state["suspend"]:
            return
        draw(change["new"])

    mu_slider.observe(on_param_change, names='value')
    sigma_slider.observe(on_param_change, names='value')
    theta_slider.observe(on_param_change, names='value')
    k_slider.observe(on_time_change, names='value')
    new_trial_btn.on_click(safe_reset_all)

    regen()
    draw(0.0)

    controls = HBox([mu_slider, sigma_slider, theta_slider, new_trial_btn])
    time_controls = HBox([k_slider])
    ui = VBox([controls, time_controls, out])
    display(ui)

    return {
        "ui": ui,
        "out": out,
        "state": state,
        "sliders": {
            "mu": mu_slider,
            "sigma": sigma_slider,
            "theta": theta_slider,
            "time": k_slider,
        },
        "button": new_trial_btn,
    }


if __name__ == "__main__":
    ddm_widget()
