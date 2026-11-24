"""Secondary null: pseudo-session engagement labels (Findling et al. 2025-style).

For a target session with T trials:
  1. draw a pseudo stimulus sequence from the IBL generative process
     (first 90 trials 50/50; then 20/80 blocks with lengths from an exponential
     with mean 60 truncated to [20, 100]); absolute contrasts are resampled from
     the target session's own contrast values;
  2. simulate choices from the mouse's fitted GLM-HMM (design P), building the
     history inputs (pc, wsls) and the Bayes prior from the simulated sequence;
  3. infer states with the same model and threshold the posterior exactly as
     for the real labels.
The resulting labels have the statistics of real labels but no link to the
session's neurons.
"""
from __future__ import annotations

import numpy as np

from src import config as C
from src import glmhmm as G


def ibl_block_sides(T: int, rng, n_unbiased=C.UNBIASED_TRIALS) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_left per trial, stimulus side +1 right / -1 left)."""
    pL = np.full(T, 0.5)
    t, blk = n_unbiased, rng.choice([0.2, 0.8])
    while t < T:
        while True:
            L = int(round(rng.exponential(60)))
            if 20 <= L <= 100:
                break
        pL[t:t + L] = blk
        blk, t = round(1.0 - blk, 1), t + L
    side = np.where(rng.random(T) < pL, -1.0, 1.0)
    return pL, side


def pseudo_session_labels(p: G.Params, T: int, abs_contrasts: np.ndarray, stim_scale: dict, rng,
                          kind: str = "smooth", thr=C.STATE_POSTERIOR_THRESHOLD,
                          engaged: tuple[int, ...] = (0,)) -> np.ndarray:
    pL, side = ibl_block_sides(T, rng)
    c = rng.choice(abs_contrasts, T)
    stim = (side * c - stim_scale["stim_mean"]) / stim_scale["stim_sd"]
    prior = G.bayes_prior(side)
    prior = np.log(prior / (1 - prior)) / np.log(4.0)
    K = p.K
    X = np.zeros((T, 5))
    y = np.zeros(T, np.int64)
    z = np.zeros(T, int)
    pc = ws = 0.0
    for t in range(T):
        z[t] = rng.choice(K, p=p.pi) if t == 0 else rng.choice(K, p=p.A[z[t - 1]])
        X[t] = [stim[t], pc, ws, prior[t], 1.0]
        pr = 1 / (1 + np.exp(-X[t] @ p.W[z[t]]))
        y[t] = rng.random() < pr
        ch = 1.0 if y[t] else -1.0
        correct = ch == side[t]          # at 0% contrast IBL still rewards the hidden side
        pc, ws = ch, ch * (1.0 if correct else -1.0)
    s = G.Seqs(X, y, np.array([0]), np.array([T]))
    gamma, alpha, pred, _ = G.posteriors(p, s)
    post = {"smooth": gamma, "filt": alpha, "pred": pred}[kind][:, list(engaged)].sum(1)
    return np.where(post > thr, 1, np.where(post < 1 - thr, 0, -1))
