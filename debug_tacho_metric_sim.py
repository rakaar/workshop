
# %%
import matplotlib.pyplot as plot
import numpy as np
from tqdm import tqdm
import pandas as pd

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
    dt=1e-3
):
    rts = np.empty(n_trials)
    choices = np.empty(n_trials)

    for i in tqdm(range(n_trials)):
        rts[i], choices[i] = simulate_ddm(mu, sigma, theta, dt=dt)

    return rts, choices


mu = 220
sigma = 103
theta = 40
N_trials = 50_000
RTs, choices = run_ddm_simulations(mu, sigma,theta, N_trials)
# %%

################################################################33####################
################# \FUTURE: Put these function in utils to hide them####################
#####################################################################################
def get_mu_sigma():
    ABL = 40
    ILD_range = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]

    lam = 0.089
    T0 = 0.28 * 1e-3
    qe = 1.0
    chi = 17.37

    mu_sigma_dict = {}  # key: ILD, value: (mu, sigma)

    # Common factor
    base = (2.0 * (qe ** 2) / T0) * (10.0 ** (lam * ABL / 20.0))

    for ild in ILD_range:
        x = lam * ild / chi
        mu = (2.0 * qe / T0) * (10.0 ** (lam * ABL / 20.0)) * np.sinh(x)
        sigma = np.sqrt(base * np.cosh(x))
        mu_sigma_dict[ild] = (float(mu), float(sigma))

    return mu_sigma_dict


mu_sigma_dict = get_mu_sigma()


theta = 40 # Parameter related to the Agent! Not stimulus
N_trials = 10_000
rows = []
for difficulty, (mu, sigma) in mu_sigma_dict.items():
  print(f"Simulating Difficulty: {difficulty}, mu: {mu :.2f}, sigma: {sigma :.2f}")
  rts, choices = run_ddm_simulations(mu, sigma, theta, N_trials)
  ABL = 40
  right_db = ABL + difficulty/2
  left_db = ABL - difficulty/2
  NDT = 0
  rows.extend(
      {
          "right_db": right_db,
          "left_db": left_db,
          "RT": rt + NDT,
          "choice": choice,
      }
      for rt, choice in zip(rts, choices)
  )

df_simulation = pd.DataFrame(rows)

# %%
import matplotlib.pyplot as plt

df_simulation['abs_difficulty'] = abs(df_simulation['right_db'] - df_simulation['left_db'])

def plot_tachometric_function(df, abs_difficulties, bin_width):
  """
  Given a set of absolute difficulties and bin width,
  plot Accuracy vs RT
  """
  # TODO 1: filter out rows related to absolute difficulty
  df_abs_difficulty = df[df['abs_difficulty'].isin(abs_difficulties)]

  # TODO 2: define bins of RTs. The goal is to calculate accuracy in each bin of RT
  bins = np.arange(0, 1, bin_width)

  # TODO 3: Iterate through each bin and append accuracy in each bin
  accuracy_vector = []
  for bin_left_wall in bins[:-1]:
    # TODO 4:  filter trials that fall in the bin
    df_bin = df_abs_difficulty[(df_abs_difficulty['RT'] >= bin_left_wall) & (df_abs_difficulty['RT'] < bin_left_wall + bin_width)]

    # TODO 5: calc num of trials
    N_trials = len(df_bin)

    if N_trials == 0:
      accuracy_vector.append(np.nan)
      continue

    # TODO 6: calc num of corrects
    N_correct = 0
    for right_db, left_db, choice in zip(df_bin['right_db'].values, df_bin['left_db'].values, df_bin['choice'].values):
      if (right_db - left_db > 0) and (choice == 1):
        N_correct += 1
      elif (right_db - left_db < 0) and (choice == -1):
        N_correct += 1

    # TODO 7: append in the Accuracy vector
    accuracy_vector.append((N_correct + 1e-10)/ (N_trials + 1e-10))

  # plot accuracy vs time bins
  bin_centers = bins[:-1] + bin_width/2
  accuracy_vector = np.array(accuracy_vector, dtype=float)
  valid = ~np.isnan(accuracy_vector)
  plt.plot(bin_centers[valid], accuracy_vector[valid])
  plt.xlim(0, 0.6)
  plt.ylabel('Accuracy')
  plt.xlabel('RT')
  plt.title('Tachometric Function')



bin_width = 20 * 1e-3 # 50ms
easy_trial_difficulties = [8, 16]
hard_trial_difficulties = [1, 2, 4]

plot_tachometric_function(df_simulation, easy_trial_difficulties, bin_width)
plot_tachometric_function(df_simulation, hard_trial_difficulties, bin_width)
plt.title('Tachometric Function (Simulated Data)');
plt.xlim(0, 0.4)
# %%
