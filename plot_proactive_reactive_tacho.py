import numpy as np
import matplotlib.pyplot as plt
from psiam_tied_utils import (
    all_RTs_fit_fn, up_RTs_fit_fn, down_RTs_fit_fn
)


def plot_tachometric(V_A, theta_A, theta_E, Z_E=0, ABL=40,
                     rate_lambda=0.089, T_0=0.0001858,
                     t_stim=0, t_A_aff=0, t_E_aff=0, t_motor=0,
                     K_max=12,
                     easy_ILDs=None, hard_ILDs=None,
                     t_max=0.6, n_pts=300):
    """
    Plot the tachometric function (accuracy vs RT) for the PSIAM-tied model.

    Parameters
    ----------
    V_A : float
        Proactive accumulator drift rate.
    theta_A : float
        Proactive accumulator threshold.
    theta_E : float
        Reactive (evidence) accumulator threshold.
    Z_E : float
        Reactive accumulator starting point (default 0).
    ABL : float
        Average binaural level in dB (default 40).
    rate_lambda : float
        Cochlear mapping rate parameter (default 0.089).
    T_0 : float
        Cochlear mapping time constant (default 0.0001858).
    t_stim : float
        Stimulus onset delay (default 0).
    t_A_aff : float
        Proactive afferent delay (default 0).
    t_E_aff : float
        Reactive afferent delay (default 0).
    t_motor : float
        Motor delay (default 0).
    K_max : int
        Max terms in series expansion (default 12).
    easy_ILDs : list
        ILD values for easy trials (default [8, 16]).
    hard_ILDs : list
        ILD values for hard trials (default [1, 2, 4]).
    t_max : float
        Maximum RT for x-axis (default 0.6).
    n_pts : int
        Number of time points (default 300).
    """
    if easy_ILDs is None:
        easy_ILDs = [8, 16]
    if hard_ILDs is None:
        hard_ILDs = [1, 2, 4]

    t_pts = np.linspace(1e-4, t_max, n_pts)

    def _accuracy_curve(ILD):
        args = (V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E,
                t_stim, t_A_aff, t_E_aff, t_motor, K_max)
        P_all = all_RTs_fit_fn(t_pts, *args)
        P_up = up_RTs_fit_fn(t_pts, *args)
        return np.where(P_all > 1e-20, P_up / P_all, np.nan)

    easy_acc = np.nanmean([_accuracy_curve(ild) for ild in easy_ILDs], axis=0)
    hard_acc = np.nanmean([_accuracy_curve(ild) for ild in hard_ILDs], axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(t_pts, easy_acc, label=f'Easy (|ILD| = {", ".join(map(str, easy_ILDs))})')
    plt.plot(t_pts, hard_acc, label=f'Hard (|ILD| = {", ".join(map(str, hard_ILDs))})')
    plt.xlabel('RT')
    plt.ylabel('Accuracy')
    plt.title('Tachometric Function (Analytical)')
    plt.xlim(0, t_max)
    plt.ylim(0.4, 1.05)
    plt.legend()
    plt.show()
