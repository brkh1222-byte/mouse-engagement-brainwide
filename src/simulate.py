"""Synthetic sessions for testing the decoding pipeline (no real neural data).

Each simulated session has
  * an engagement sequence from a sticky 2-state Markov chain whose
    disengagement probability rises late in the session (satiety),
  * a block-prior signal and a movement (motion-energy) signal; movement is
    higher when disengaged, so it is a genuine confound,
  * a slow drift in firing shared by all neurons (recording drift),
  * neurons in four regions:
      ENG  rates depend on engagement (true effect)
      MOV  rates depend on movement only (confound; should vanish in Model B)
      PRI  rates depend on the block prior only (confound; should vanish in Model B)
      NUL  rates depend on slow drift only (should not be significant under the
           session-permutation null)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import covariates as CV


def simulate_session(rng, T=600, n_units=15, eng_gain=0.35, mov_gain=0.5, pri_gain=0.5,
                     drift_gain=0.6, base_rate=3.0):
    # engagement: sticky chain, disengagement more likely late
    z = np.zeros(T, int)
    z[0] = 1
    for t in range(1, T):
        frac = t / T
        p_leave = 0.02 + 0.04 * frac          # engaged -> disengaged
        p_back = 0.08 - 0.05 * frac           # disengaged -> engaged
        z[t] = (rng.random() > p_leave) if z[t - 1] == 1 else (rng.random() < p_back)
    # block prior (IBL-like blocks after 90 trials), movement and drift
    side = np.zeros(T)
    t, blk = 90, rng.choice([-1, 1])
    while t < T:
        L = int(np.clip(rng.exponential(60), 20, 100))
        side[t:t + L] = blk
        blk, t = -blk, t + L
    prior = np.clip(np.convolve(side, np.ones(10) / 10, "full")[:T], -1, 1)
    motion = rng.normal(0, 1, T) + 0.8 * (1 - z)
    wheel = np.abs(rng.normal(0, 0.05, T))
    drift = np.cumsum(rng.normal(0, 1, T))
    drift = (drift - drift.mean()) / (drift.std() + 1e-9)
    trend = np.linspace(-1, 1, T)

    def units(signal, gain):
        w = rng.normal(0, 1, n_units)
        lam = base_rate * np.exp(gain * np.outer(signal, w) / np.sqrt(1 + gain)
                                 + 0.3 * drift_gain * np.outer(drift + trend, rng.normal(0, 1, n_units)))
        return rng.poisson(lam)

    regions = {"ENG": units(z - z.mean(), eng_gain * 3),
               "MOV": units(motion, mov_gain),
               "PRI": units(prior, pri_gain * 2),
               "NUL": units(np.zeros(T), 0.0)}
    trials = pd.DataFrame({"rewardVolume": np.where(rng.random(T) < 0.8, 1.5, 0.0),
                           "probabilityLeft": np.where(side > 0, 0.2, np.where(side < 0, 0.8, 0.5))})
    cov = CV.assemble(trials, prior, wheel, motion)
    # labels as a GLM-HMM would deliver: some trials unlabelled
    post = np.where(z == 1, rng.beta(8, 1.5, T), rng.beta(1.5, 8, T))
    lab = np.where(post > 0.8, 1, np.where(post < 0.2, 0, -1))
    return {"z_true": z, "labels": lab, "regions": regions, "cov": cov}
