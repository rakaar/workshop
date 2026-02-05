# %%
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

# %%

def simulate_ddm(mu, sigma, theta, dt=1e-3):
  """
  Given a mu, sigma, theta, simulate a trial of DDM
  return RT, choice
  RT: time taken to hit the bound
  choice: +1 if upper bonud is it, -1 if lower bound is hit
  """
  # TODO 1: Initialize time and DV at zero
  t = 0
  DV = 0

  # TODO 2: use a while True loop where u keep accumulating evidence and incrementing
  while True:
    DV = DV + mu * dt + sigma * np.random.normal(0, dt**0.5)
    t += dt

    # add a if condition to check if DV exceeds one of the threshold, DV > theta or DV < -theta
    if DV > theta:
      RT = t
      choice = 1
      break
    elif DV < -theta:
      RT = t
      choice = -1
      break

  return RT, choice




def run_ddm_simulations(
    mu,
    sigma,
    theta,
    n_trials,
    dt=1e-3,
    show_progress=True,
):
    rts = np.empty(n_trials)
    choices = np.empty(n_trials)

    progress = tqdm(range(n_trials), disable=not show_progress)
    for i in progress:
        rts[i], choices[i] = simulate_ddm(mu, sigma, theta, dt=dt)

    return rts, choices


mu = 200
sigma = 100
theta = 40
N_trials = 50_000
RTs_ground_truth, choices_ground_truth = run_ddm_simulations(mu, sigma,theta, N_trials)
# %%
# TODO: search over grid space 3D , where we simulate 10K trials and compare with the 
# RTD of the original data ( plt.hist(RTs_ground_truth,bins=bins, density=True, histtype='step')) 
# just by plotting  and seing visually
# note that the RTDs should be checked visually by plotting
# also, plz write beginner friendly code, this is for content in a tutorial with many epxerimental scientists

# %%
# Define a simple 3D grid for (mu, sigma, theta). Keep it small for a tutorial.
mu_values = [150, 200, 250]
sigma_values = [80, 100, 120]
theta_values = [30, 40, 50]

# We will use the same bins for all histograms so comparisons are fair.
max_rt = np.percentile(RTs_ground_truth, 99)
bins = np.linspace(0, max_rt, 40)

# Run a small grid search: simulate 10K trials for each combination
# and visually compare the RT distributions.
n_trials_grid = 10_000

param_grid = [
    (mu_candidate, sigma_candidate, theta_candidate)
    for mu_candidate in mu_values
    for sigma_candidate in sigma_values
    for theta_candidate in theta_values
]

total_params = len(param_grid)
# fig, axes = plt.subplots(3, 9, figsize=(18, 6), sharex=True, sharey=True)
# axes = axes.flatten()

# with tqdm(total=total_params, desc="Grid search") as pbar:
#     for idx, (mu_candidate, sigma_candidate, theta_candidate) in enumerate(
#         param_grid,
#         start=1,
#     ):
#         rts_sim, _ = run_ddm_simulations(
#             mu_candidate,
#             sigma_candidate,
#             theta_candidate,
#             n_trials_grid,
#             show_progress=False,
#         )
#         pbar.set_postfix_str(
#             f"{idx}/{total_params} mu={mu_candidate} sigma={sigma_candidate} theta={theta_candidate}"
#         )
#         pbar.update(1)

#         ax = axes[idx - 1]
#         ax.hist(
#             RTs_ground_truth,
#             bins=bins,
#             density=True,
#             histtype="step",
#             label="Ground truth",
#         )
#         ax.hist(
#             rts_sim,
#             bins=bins,
#             density=True,
#             histtype="step",
#             label="Simulation",
#         )
#         if (
#             mu_candidate == mu
#             and sigma_candidate == sigma
#             and theta_candidate == theta
#         ):
#             for spine in ax.spines.values():
#                 spine.set_edgecolor("green")
#                 spine.set_linewidth(2)
#         ax.set_title(
#             f"mu={mu_candidate}\nsigma={sigma_candidate}\ntheta={theta_candidate}",
#             fontsize=8,
#         )

# for ax in axes:
#     ax.label_outer()

# axes[0].legend(fontsize=7, loc="upper right")
# fig.suptitle("RTD grid comparison (3 x 9)")
# fig.tight_layout()
# plt.show()

# %%
# TODO
# Ok, we found a set of parameters that match the RTD. But here we search over a very coarse and tiny parameter space. In reality, the parameter space is much finer and larger, like $ \mu $ range would be from 0 to 400 in steps of 0.1.

# We need some kind of a method way, the search over parameter space is much more cleverly rather than trying brute force over entire possible parameter space.

# But to do that, we first need to define some metric which we try to minimize. And we need an intelligent algorithm that searches intelligently to minimize that error. For example, one such pair we could try is 
# - **Metric**: Mean Square Error(MSE) between simulated and Ground truth Reaction Time Distribution
# - **Algorithm**: Gradient Descent algorithm, an algorithm that is used to train Neural Networks.

# %%
# Step 1: define a simple metric (MSE between RT histograms)
def rtd_mse(mu, sigma, theta, n_trials=2_000):
    rts_sim, _ = run_ddm_simulations(
        mu,
        sigma,
        theta,
        n_trials,
        show_progress=False,
    )
    hist_sim, _ = np.histogram(rts_sim, bins=bins, density=True)
    hist_truth, _ = np.histogram(RTs_ground_truth, bins=bins, density=True)
    return np.mean((hist_sim - hist_truth) ** 2)


# Step 2: gradient descent with finite-difference gradients
# We start close to the ground truth so it finishes fast in a tutorial.
mu_cur = 180
sigma_cur = 110
theta_cur = 35

delta_mu = 10
delta_sigma = 5
delta_theta = 2

lr_mu = 5
lr_sigma = 2
lr_theta = 1

# We will store the history of (mu, sigma, theta, mse) at each step
history = []

for step in range(20):
    mse_center = rtd_mse(mu_cur, sigma_cur, theta_cur)
    history.append((mu_cur, sigma_cur, theta_cur, mse_center))

    print(
        f"Step {step}: mu={mu_cur:.2f}, sigma={sigma_cur:.2f}, theta={theta_cur:.2f}, mse={mse_center:.6f}"
    )

    if mse_center < 0.1:
        print("MSE below 0.1 — stopping early!")
        break

    grad_mu = (
        rtd_mse(mu_cur + delta_mu, sigma_cur, theta_cur)
        - rtd_mse(mu_cur - delta_mu, sigma_cur, theta_cur)
    ) / (2 * delta_mu)
    grad_sigma = (
        rtd_mse(mu_cur, sigma_cur + delta_sigma, theta_cur)
        - rtd_mse(mu_cur, sigma_cur - delta_sigma, theta_cur)
    ) / (2 * delta_sigma)
    grad_theta = (
        rtd_mse(mu_cur, sigma_cur, theta_cur + delta_theta)
        - rtd_mse(mu_cur, sigma_cur, theta_cur - delta_theta)
    ) / (2 * delta_theta)

    mu_cur = max(mu_cur - lr_mu * grad_mu, 0)
    sigma_cur = max(sigma_cur - lr_sigma * grad_sigma, 1)
    theta_cur = max(theta_cur - lr_theta * grad_theta, 1)

print(f"Final: mu={mu_cur:.2f}, sigma={sigma_cur:.2f}, theta={theta_cur:.2f}")

# %%
# Pick 3 snapshots: beginning, middle, end — sorted by decreasing MSE
snap_begin = history[0]
snap_end = history[-1]
snap_mid = history[len(history) // 2]

# Make sure they are in decreasing MSE order
snapshots = sorted(
    [snap_begin, snap_mid, snap_end],
    key=lambda x: x[3],
    reverse=True,
)
labels = ["Beginning", "Middle", "End"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, label, (m, s, th, mse_val) in zip(axes, labels, snapshots):
    rts_snap, _ = run_ddm_simulations(m, s, th, 5_000, show_progress=False)
    ax.hist(
        RTs_ground_truth,
        bins=bins,
        density=True,
        histtype="step",
        label="Ground truth",
    )
    ax.hist(
        rts_snap,
        bins=bins,
        density=True,
        histtype="step",
        label="Simulation",
    )
    ax.set_title(
        f"{label}\nmu={m:.1f}, sigma={s:.1f}, theta={th:.1f}\nMSE={mse_val:.4f}",
        fontsize=9,
    )
    ax.set_xlabel("Reaction time (s)")
    ax.legend(fontsize=7)

axes[0].set_ylabel("Density")
fig.suptitle("Gradient descent progression: RTD comparison", fontsize=12)
fig.tight_layout()
plt.show()
# %%