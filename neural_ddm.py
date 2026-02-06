# %% [markdown]
# # Part 4 – Neural Equivalent of the DDM
#
# We interpret the DDM's "evidence accumulation" as something implemented by
# two competing pools of spiking neurons:
#
# - **Pool R** (supports "Right" / upper bound)
# - **Pool L** (supports "Left" / lower bound)
#
# Decision variable: $e(t) = N_R(t) - N_L(t)$
#
# A decision is made when $e(t)$ hits $+\theta$ or $-\theta$.

# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% [markdown]
# ## 1. Single Poisson Neuron
#
# A Poisson neuron fires spikes at a constant rate $r$ (spikes/s).
# The inter-spike intervals (ISIs) are exponentially distributed
# with mean $1/r$.

# %%
def simulate_poisson_neuron(rate, duration):
    """
    Simulate a single Poisson neuron.

    Parameters
    ----------
    rate : float
        Firing rate in spikes/second (Hz).
    duration : float
        Total simulation time in seconds.

    Returns
    -------
    spike_times : np.ndarray
        Array of spike times (in seconds).
    """
    # Generate ISIs from an exponential distribution
    spike_times = []
    t = 0.0
    while t < duration:
        isi = np.random.exponential(1.0 / rate)
        t += isi
        if t < duration:
            spike_times.append(t)
    return np.array(spike_times)


def plot_single_neuron(rate=50.0, duration=10.0):
    """Simulate one Poisson neuron, show its raster and ISI histogram."""
    spike_times = simulate_poisson_neuron(rate, duration)
    isis = np.diff(spike_times)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    # Raster / spike train
    axes[0].eventplot(spike_times, lineoffsets=0, linelengths=0.5, color='k')
    axes[0].set_xlim(0, min(duration, 2.0))  # show first 2 s
    axes[0].set_yticks([])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title(f"Spike train (rate = {rate} Hz, first 2 s)")

    # ISI histogram vs exponential pdf
    axes[1].hist(isis, bins=50, density=True, alpha=0.6, label="ISI histogram")
    t_grid = np.linspace(0, np.percentile(isis, 99), 200)
    axes[1].plot(t_grid, rate * np.exp(-rate * t_grid), 'r-', lw=2,
                 label=f"Exp(λ={rate})")
    axes[1].set_xlabel("ISI (s)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Inter-Spike Interval Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


plot_single_neuron(rate=50.0, duration=10.0)

# %% [markdown]
# ## 2. Simulate a Pool of N Poisson Neurons
#
# Each neuron in the pool fires independently at rate $r$.
# We collect all spike times from the pool.

# %%
def simulate_pool(n_neurons, rate, duration):
    """
    Simulate a pool of N independent Poisson neurons.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the pool.
    rate : float
        Firing rate of each neuron (Hz).
    duration : float
        Simulation duration (s).

    Returns
    -------
    spike_times : np.ndarray
        Sorted array of all spike times from the pool.
    """
    all_spikes = []
    for _ in range(n_neurons):
        spikes = simulate_poisson_neuron(rate, duration)
        all_spikes.append(spikes)
    all_spikes = np.concatenate(all_spikes)
    all_spikes.sort()
    return all_spikes


def plot_pool(n_neurons=20, rate=50.0, duration=2.0):
    """Visualise the pool: raster plot of all neurons."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(n_neurons):
        spikes = simulate_poisson_neuron(rate, duration)
        ax.eventplot(spikes, lineoffsets=i, linelengths=0.6, color='k')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron #")
    ax.set_title(f"Raster plot – Pool of {n_neurons} neurons @ {rate} Hz each")
    plt.tight_layout()
    plt.show()


plot_pool(n_neurons=20, rate=50.0, duration=2.0)

# %% [markdown]
# ## 3. Evidence Accumulation from Two Competing Pools
#
# - Pool R fires at rate $R$ (spikes/s per neuron, $N_R$ neurons)
# - Pool L fires at rate $L$ (spikes/s per neuron, $N_L$ neurons)
#
# Every spike from Pool R increments the decision variable by +1.
# Every spike from Pool L decrements it by −1.
#
# The trial ends when the accumulated evidence $e(t)$ hits $+\theta$ or $-\theta$.

# %%
def simulate_neural_trial(n_neurons_R, rate_R, n_neurons_L, rate_L, theta, dt=1e-4, max_time=10.0):
    """
    Simulate one trial of the neural race model.

    Parameters
    ----------
    n_neurons_R : int   – number of neurons in Pool R
    rate_R      : float – firing rate per neuron in Pool R (Hz)
    n_neurons_L : int   – number of neurons in Pool L
    rate_L      : float – firing rate per neuron in Pool L (Hz)
    theta       : int   – decision threshold (in spike-count units)
    dt          : float – time step (s)
    max_time    : float – maximum trial duration (s)

    Returns
    -------
    RT     : float – reaction time (s)
    choice : int   – +1 (upper/right) or -1 (lower/left)
    e_trace: np.ndarray – trajectory of the decision variable
    t_trace: np.ndarray – corresponding time points
    """
    # Total pool firing rates
    pool_rate_R = n_neurons_R * rate_R  # total spikes/s from Pool R
    pool_rate_L = n_neurons_L * rate_L  # total spikes/s from Pool L

    e = 0       # decision variable
    t = 0.0
    e_trace = [0]
    t_trace = [0.0]

    while t < max_time:
        # Expected number of spikes in this dt from each pool
        n_spikes_R = np.random.poisson(pool_rate_R * dt)
        n_spikes_L = np.random.poisson(pool_rate_L * dt)

        e += n_spikes_R - n_spikes_L
        t += dt

        e_trace.append(e)
        t_trace.append(t)

        if e >= theta:
            return t, +1, np.array(e_trace), np.array(t_trace)
        elif e <= -theta:
            return t, -1, np.array(e_trace), np.array(t_trace)

    # If max_time reached without decision, return current state
    choice = +1 if e >= 0 else -1
    return t, choice, np.array(e_trace), np.array(t_trace)


# %% [markdown]
# ### Visualise a few example trajectories

# %%
def plot_example_trajectories(n_trials=5, n_neurons=20, rate_R=52.0, rate_L=50.0, theta=50):
    """Plot a few example evidence-accumulation trajectories."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(n_trials):
        RT, choice, e_trace, t_trace = simulate_neural_trial(
            n_neurons_R=n_neurons, rate_R=rate_R,
            n_neurons_L=n_neurons, rate_L=rate_L,
            theta=theta
        )
        color = 'steelblue' if choice == 1 else 'coral'
        ax.plot(t_trace, e_trace, color=color, alpha=0.7, lw=0.8)

    ax.axhline(+theta, color='green', ls='--', lw=2, label=f'+θ = {theta} (Right)')
    ax.axhline(-theta, color='red', ls='--', lw=2, label=f'−θ = {-theta} (Left)')
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Evidence  e(t) = N_R − N_L")
    ax.set_title("Neural Evidence Accumulation – Example Trajectories")
    ax.legend()
    plt.tight_layout()
    plt.show()


plot_example_trajectories(n_trials=8, n_neurons=10, rate_R=55.0, rate_L=45.0, theta=30)

# %% [markdown]
# ## 4. RT Distribution from the Neural Model
#
# Run many trials and collect RTs and choices.

# %%
def run_neural_simulations(n_neurons_R, rate_R, n_neurons_L, rate_L, theta, n_trials, dt=1e-4):
    """
    Run many neural-model trials and return RTs and choices.
    """
    rts = np.empty(n_trials)
    choices = np.empty(n_trials)
    for i in tqdm(range(n_trials), desc="Neural model"):
        rt, ch, _, _ = simulate_neural_trial(
            n_neurons_R, rate_R, n_neurons_L, rate_L, theta, dt=dt
        )
        rts[i] = rt
        choices[i] = ch
    return rts, choices


# --- Parameters ---
# Chosen so that all RTs stay below ~1 s:
#   μ = N*(R-L) = 10*(55-45) = 100 spikes/s  (strong drift)
#   σ = √(N*(R+L)) = √(10*100) ≈ 31.6
#   Expected RT ≈ θ/μ = 30/100 = 0.3 s; tail well under 1 s
N_NEURONS = 10       # neurons per pool
RATE_R    = 55.0     # Hz per neuron in Pool R
RATE_L    = 45.0     # Hz per neuron in Pool L
THETA     = 30       # threshold in spike-count units
N_TRIALS  = 5000

neural_rts, neural_choices = run_neural_simulations(
    N_NEURONS, RATE_R, N_NEURONS, RATE_L, THETA, N_TRIALS
)

# %%
def plot_neural_rt_distribution(rts, choices, theta):
    """Plot RT distributions for the neural model, split by choice."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # All RTs
    axes[0].hist(rts, bins=80, density=True, alpha=0.6, color='gray')
    axes[0].set_title("All RTs (Neural Model)")
    axes[0].set_xlabel("RT (s)")
    axes[0].set_ylabel("Density")

    # Upper bound hits
    rts_upper = rts[choices == 1]
    axes[1].hist(rts_upper, bins=60, density=True, alpha=0.6, color='steelblue')
    axes[1].set_title(f"Right choices ({len(rts_upper)}/{len(rts)})")
    axes[1].set_xlabel("RT (s)")

    # Lower bound hits
    rts_lower = rts[choices == -1]
    axes[2].hist(rts_lower, bins=60, density=True, alpha=0.6, color='coral')
    axes[2].set_title(f"Left choices ({len(rts_lower)}/{len(rts)})")
    axes[2].set_xlabel("RT (s)")

    plt.suptitle(f"Neural Model RT Distributions  (θ = {theta})", fontsize=13)
    plt.tight_layout()
    plt.show()

    print(f"Mean RT: {np.mean(rts):.4f} s")
    print(f"P(Right): {np.mean(choices == 1):.3f}")


plot_neural_rt_distribution(neural_rts, neural_choices, THETA)

# %% [markdown]
# ## 5. DDM Equivalence
#
# The neural model maps onto a standard DDM with:
#
# $$\mu = R_{\text{total}} - L_{\text{total}}$$
# $$\sigma = \sqrt{R_{\text{total}} + L_{\text{total}}}$$
#
# where $R_{\text{total}} = N \cdot r_R$ and $L_{\text{total}} = N \cdot r_L$
# are the total pool firing rates (spikes/s).
#
# **Why?** In a small time window $dt$:
# - Pool R produces $\sim \text{Poisson}(R_{\text{total}} \cdot dt)$ spikes
# - Pool L produces $\sim \text{Poisson}(L_{\text{total}} \cdot dt)$ spikes
# - The increment $\Delta e = n_R - n_L$ has:
#   - $E[\Delta e] = (R_{\text{total}} - L_{\text{total}}) \cdot dt = \mu \cdot dt$
#   - $\text{Var}[\Delta e] = (R_{\text{total}} + L_{\text{total}}) \cdot dt = \sigma^2 \cdot dt$
#
# This is exactly the DDM update: $\Delta e = \mu \cdot dt + \sigma \sqrt{dt} \cdot \mathcal{N}(0,1)$

# %%
# Compute equivalent DDM parameters
R_total = N_NEURONS * RATE_R   # total Pool R rate
L_total = N_NEURONS * RATE_L   # total Pool L rate

mu_equiv    = R_total - L_total                # drift
sigma_equiv = np.sqrt(R_total + L_total)       # noise
theta_equiv = THETA                            # same threshold

print("=== Neural → DDM Parameter Mapping ===")
print(f"  R_total = {N_NEURONS} × {RATE_R} = {R_total} spikes/s")
print(f"  L_total = {N_NEURONS} × {RATE_L} = {L_total} spikes/s")
print(f"  μ  = R_total − L_total       = {mu_equiv:.1f}")
print(f"  σ  = √(R_total + L_total)    = {sigma_equiv:.2f}")
print(f"  θ  = {theta_equiv}")

# %% [markdown]
# ## 6. Overlay: Neural Model vs Equivalent DDM
#
# We simulate the equivalent DDM and overlay its RT distribution
# on top of the neural model's RT distribution. They should match!

# %%
def simulate_ddm(mu, sigma, theta, dt=1e-4):
    """Simulate one DDM trial. Returns (RT, choice)."""
    t = 0.0
    DV = 0.0
    while True:
        DV += mu * dt + sigma * np.sqrt(dt) * np.random.randn()
        t += dt
        if DV >= theta:
            return t, +1
        elif DV <= -theta:
            return t, -1


def run_ddm_simulations(mu, sigma, theta, n_trials, dt=1e-4):
    """Run many DDM trials."""
    rts = np.empty(n_trials)
    choices = np.empty(n_trials)
    for i in tqdm(range(n_trials), desc="DDM"):
        rts[i], choices[i] = simulate_ddm(mu, sigma, theta, dt=dt)
    return rts, choices


# Simulate the equivalent DDM
ddm_rts, ddm_choices = run_ddm_simulations(mu_equiv, sigma_equiv, theta_equiv, N_TRIALS)

# %%
def plot_overlay(neural_rts, neural_choices, ddm_rts, ddm_choices, mu, sigma, theta):
    """Overlay neural and DDM RT distributions to show equivalence."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Determine common bin edges
    all_rts = np.concatenate([neural_rts, ddm_rts])
    bins = np.linspace(0, np.percentile(all_rts, 99), 80)

    # --- All RTs ---
    axes[0].hist(neural_rts, bins=bins, density=True, alpha=0.5,
                 color='steelblue', label='Neural model')
    axes[0].hist(ddm_rts, bins=bins, density=True, alpha=0.5,
                 color='coral', label='DDM equivalent')
    axes[0].set_title("All RTs")
    axes[0].set_xlabel("RT (s)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # --- Right choices ---
    n_rts_upper = neural_rts[neural_choices == 1]
    d_rts_upper = ddm_rts[ddm_choices == 1]
    axes[1].hist(n_rts_upper, bins=bins, density=True, alpha=0.5,
                 color='steelblue', label='Neural')
    axes[1].hist(d_rts_upper, bins=bins, density=True, alpha=0.5,
                 color='coral', label='DDM')
    axes[1].set_title("Right Choices (+θ)")
    axes[1].set_xlabel("RT (s)")
    axes[1].legend()

    # --- Left choices ---
    n_rts_lower = neural_rts[neural_choices == -1]
    d_rts_lower = ddm_rts[ddm_choices == -1]
    axes[2].hist(n_rts_lower, bins=bins, density=True, alpha=0.5,
                 color='steelblue', label='Neural')
    axes[2].hist(d_rts_lower, bins=bins, density=True, alpha=0.5,
                 color='coral', label='DDM')
    axes[2].set_title("Left Choices (−θ)")
    axes[2].set_xlabel("RT (s)")
    axes[2].legend()

    plt.suptitle(
        f"Neural Model vs DDM Equivalence\n"
        f"μ = {mu:.1f},  σ = {sigma:.2f},  θ = {theta}",
        fontsize=13
    )
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("=== Summary ===")
    print(f"  Neural  – Mean RT: {np.mean(neural_rts):.4f} s,  P(Right): {np.mean(neural_choices==1):.3f}")
    print(f"  DDM     – Mean RT: {np.mean(ddm_rts):.4f} s,  P(Right): {np.mean(ddm_choices==1):.3f}")


plot_overlay(neural_rts, neural_choices, ddm_rts, ddm_choices,
             mu_equiv, sigma_equiv, theta_equiv)
