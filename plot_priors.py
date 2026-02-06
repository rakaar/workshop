import numpy as np
import matplotlib.pyplot as plt


def plot_prior_examples():
    """Plot a 1x3 figure illustrating uniform and normal priors."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # --- Panel 1: Uniform(10, 20) - Well-connected bus stop ---
    x1 = np.linspace(5, 25, 500)
    y1 = np.where((x1 >= 10) & (x1 <= 20), 1 / (20 - 10), 0)
    axes[0].fill_between(x1, y1, alpha=0.3, color='#2176AE')
    axes[0].plot(x1, y1, color='#2176AE', lw=2)
    axes[0].set_xlabel('Waiting time (min)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Well-connected stop\nUniform(10, 20)', fontsize=11)
    axes[0].set_ylim(bottom=0)

    # --- Panel 2: Uniform(0, 60) - Deserted city ---
    x2 = np.linspace(-5, 65, 500)
    y2 = np.where((x2 >= 0) & (x2 <= 60), 1 / 60, 0)
    axes[1].fill_between(x2, y2, alpha=0.3, color='#D7263D')
    axes[1].plot(x2, y2, color='#D7263D', lw=2)
    axes[1].set_xlabel('Waiting time (min)', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Deserted city\nUniform(0, 60)', fontsize=11)
    axes[1].set_ylim(bottom=0)

    # --- Panel 3: Normal height distribution ---
    mu_cm = 170
    sigma_cm = 7
    x3 = np.linspace(140, 220, 500)
    y3 = (1 / (sigma_cm * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x3 - mu_cm) / sigma_cm) ** 2)
    axes[2].fill_between(x3, y3, alpha=0.3, color='#57A773')
    axes[2].plot(x3, y3, color='#57A773', lw=2)
    seven_ft_cm = 7 * 30.48  # ~213.4 cm
    axes[2].axvline(seven_ft_cm, color='k', linestyle=':', lw=1.5, label=f'7 ft ({seven_ft_cm:.0f} cm)')
    axes[2].legend(fontsize=9, frameon=False)
    axes[2].set_xlabel('Height (cm)', fontsize=11)
    axes[2].set_ylabel('Density', fontsize=11)
    axes[2].set_title('Height prior\nNormal(170, 7)', fontsize=11)
    axes[2].set_ylim(bottom=0)

    # --- Publication-grade styling ---
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()


def plot_ddm_priors():
    """Plot a 1x3 figure showing uniform priors for DDM parameters mu, sigma, theta."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    params = [
        {'label': r'$\mu$', 'lo': 190, 'hi': 210, 'pad': 5, 'color': '#2176AE'},
        {'label': r'$\sigma$', 'lo': 90, 'hi': 110, 'pad': 5, 'color': '#D7263D'},
        {'label': r'$\theta$', 'lo': 35, 'hi': 45, 'pad': 5, 'color': '#57A773'},
    ]

    for ax, p in zip(axes, params):
        lo, hi, pad = p['lo'], p['hi'], p['pad']
        x = np.linspace(lo - pad, hi + pad, 500)
        y = np.where((x >= lo) & (x <= hi), 1 / (hi - lo), 0)
        ax.fill_between(x, y, alpha=0.3, color=p['color'])
        ax.plot(x, y, color=p['color'], lw=2)
        ax.set_xlabel(p['label'], fontsize=13)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{p["label"]} ~ Uniform({lo}, {hi})', fontsize=11)
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()
