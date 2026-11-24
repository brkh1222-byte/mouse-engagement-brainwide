"""T2.2: per-trial nuisance covariates for the pre-stimulus window.

Primary covariate set (Model B), all computed in the same window as the spike
counts (default -0.4 to -0.1 s before stimulus onset) unless stated:
  prior        block prior: log-odds of the Bayes-observer p(stim right), as used
               as a GLM-HMM input (src.glmhmm.bayes_prior), scaled to [-1, 1]
  wheel_speed  mean |wheel velocity| (rad/s) in the window
  motion       mean left-camera whisker-pad ROI motion energy in the window
               (z-scored within session)
  trial_spline natural-ish cubic B-spline basis of trial index / n_trials (4 columns)
  cum_reward   cumulative reward volume before the trial / total (satiety proxy)
Robustness only: true block side (probabilityLeft), lick count, pupil diameter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

PRIMARY = ["prior", "wheel_speed", "motion", "ts0", "ts1", "ts2", "ts3", "cum_reward"]


# --------------------------------------------------------------------------- pure functions
def window_mean(times: np.ndarray, values: np.ndarray, align: np.ndarray, window) -> np.ndarray:
    """Mean of a regularly or irregularly sampled signal over [t+w0, t+w1) for each t."""
    out = np.full(len(align), np.nan)
    ok = np.isfinite(align)
    lo = np.searchsorted(times, align[ok] + window[0])
    hi = np.searchsorted(times, align[ok] + window[1])
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values))])
    cnt = np.concatenate([[0], np.cumsum(np.isfinite(values))])
    n = cnt[hi] - cnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = np.where(n > 0, (csum[hi] - csum[lo]) / n, np.nan)
    return out


def event_counts(event_times: np.ndarray, align: np.ndarray, window) -> np.ndarray:
    out = np.full(len(align), np.nan)
    ok = np.isfinite(align)
    out[ok] = (np.searchsorted(event_times, align[ok] + window[1])
               - np.searchsorted(event_times, align[ok] + window[0]))
    return out


def wheel_speed(wheel_t: np.ndarray, wheel_pos: np.ndarray, align, window, fs=1000.0):
    """Interpolate wheel position at fs Hz, differentiate, average |velocity| in window."""
    t = np.arange(wheel_t[0], wheel_t[-1], 1 / fs)
    pos = np.interp(t, wheel_t, wheel_pos)
    vel = np.abs(np.gradient(pos, 1 / fs))
    return window_mean(t, vel, align, window)


def trial_spline(n_trials: int, n_basis: int = 4) -> np.ndarray:
    from sklearn.preprocessing import SplineTransformer
    x = (np.arange(n_trials) / max(n_trials - 1, 1))[:, None]
    st = SplineTransformer(n_knots=n_basis - 1, degree=3, include_bias=False)
    return st.fit_transform(x)[:, :n_basis]


def cumulative_reward(reward_volume: np.ndarray) -> np.ndarray:
    r = np.nan_to_num(reward_volume)
    c = np.concatenate([[0.0], np.cumsum(r)[:-1]])
    return c / max(c[-1] + r[-1], 1e-9)


def assemble(trials: pd.DataFrame, prior_logodds: np.ndarray, wheel_spd: np.ndarray,
             motion: np.ndarray | None) -> pd.DataFrame:
    n = len(trials)
    ts = trial_spline(n)
    df = pd.DataFrame({"prior": prior_logodds, "wheel_speed": wheel_spd,
                       "motion": np.nan if motion is None else (motion - np.nanmean(motion)) / np.nanstd(motion),
                       "ts0": ts[:, 0], "ts1": ts[:, 1], "ts2": ts[:, 2], "ts3": ts[:, 3],
                       "cum_reward": cumulative_reward(trials["rewardVolume"].to_numpy())})
    df["block_side"] = np.sign(0.5 - trials["probabilityLeft"].to_numpy())   # robustness only
    return df


# --------------------------------------------------------------------------- ONE-backed loading
def video_ok(times: np.ndarray, me: np.ndarray, max_nan_frac: float = 0.1) -> bool:
    """Minimal pre-registered video check: lengths match and <10% missing."""
    return len(times) == len(me) and np.mean(~np.isfinite(me)) < max_nan_frac


def session_covariates(one, eid: str, trials: pd.DataFrame, prior_logodds: np.ndarray,
                       window=C.PRESTIM_WINDOW) -> pd.DataFrame:
    align = trials["stimOn_times"].to_numpy()
    wheel = one.load_object(eid, "wheel", collection="alf")
    wspd = wheel_speed(wheel["timestamps"], wheel["position"], align, window)
    motion = None
    try:
        me = one.load_dataset(eid, "leftCamera.ROIMotionEnergy.npy", collection="alf")
        ct = one.load_dataset(eid, "_ibl_leftCamera.times.npy", collection="alf")
        if video_ok(ct, me):
            motion = window_mean(ct, me.astype(float), align, window)
    except Exception:
        motion = None
    return assemble(trials, prior_logodds, wspd, motion)
