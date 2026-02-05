# @title
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    DV = DV + mu * dt + sigma * np.sqrt(dt) * np.random.randn()
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
    dt=1e-3
):
    rts = np.empty(n_trials)
    choices = np.empty(n_trials)

    for i in tqdm(range(n_trials)):
        rts[i], choices[i] = simulate_ddm(mu, sigma, theta, dt=dt)

    return rts, choices


def build_ddm_dataframe(mu_sigma_dict, theta, n_trials, dt=1e-3):
    """Return a DataFrame with columns: difficulty, rt, choice."""
    rows = []
    for difficulty, (mu, sigma) in mu_sigma_dict.items():
        rts, choices = run_ddm_simulations(mu, sigma, theta, n_trials, dt=dt)
        rows.extend(
            {
                "difficulty": difficulty,
                "rt": float(rt),
                "choice": int(choice),
            }
            for rt, choice in zip(rts, choices)
        )
    return pd.DataFrame(rows)



theta = 40  # Parameter related to the agent! Not stimulus
