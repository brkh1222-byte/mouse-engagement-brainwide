"""T1.3: Bernoulli GLM-HMM of IBL choices (Ashwood et al. 2022 design). Behaviour only.

Model
-----
Each trial t of a session is in one hidden state z_t in {1..K}. States persist
according to a Markov transition matrix A (K x K) with initial distribution pi.
In state k the mouse chooses "right" with probability sigmoid(w_k . x_t), where
x_t holds the trial's inputs. Each state is therefore its own little choice
strategy; "engaged" is the state that weights the stimulus most.

Inputs (columns of X)
  stim   signed contrast (right - left), z-scored over the whole dataset (Ashwood)
  pc     previous valid choice (+1 right / -1 left; 0 if none yet in the session)
  wsls   win-stay/lose-switch = pc * previous outcome (+1 rewarded / -1 not)
  prior  [design "P" only] block prior: log-odds of p(stim right) from a Bayes
         observer of past stimulus sides, scaled to [-1, 1] (see `bayes_prior`)
  bias   constant 1

Designs
  "A90"  Ashwood replication: first-90 (50/50) trials of biased/ephys sessions,
         inputs stim, pc, wsls, bias. Sessions with >= 10 no-go trials in those
         90 trials are dropped (as in Ashwood's code).
  "P"    Primary: all trials of biased/ephys sessions, with the block-prior input.

No-go trials (choice == 0) carry no choice; they are masked out of the
likelihood (their emission likelihood is 1), exactly as in Ashwood.

Fitting: MAP-EM. Gaussian prior N(0, sigma^2) on weights, Dirichlet(alpha) on
each transition row. E-step: scaled forward-backward per session (numba).
M-step: closed form for A and pi; Newton's method for each state's weighted
logistic regression.

Protocol (fixed in advance; see preregistration/prereg.md)
  1. Global fit (all mice pooled), K = 1..5, 20 initialisations each.
  2. Session-level 5-fold CV of the global model (3 inits per fold, trained on
     train sessions only).
  3. Per-mouse MAP fits initialised from the global fit of the same K
     (and, for CV, from the global fit of the same fold, so no test data leak).
  4. K chosen by the 1-SE rule on per-mouse held-out log-likelihood.
  5. Final per-mouse fits on all sessions -> trial posteriors.

Usage
  python -m src.glmhmm build  --design P
  python -m src.glmhmm global --design P [--k 3]
  python -m src.glmhmm cv     --design P
  python -m src.glmhmm mouse  --design P
  python -m src.glmhmm select --design P
  python -m src.glmhmm label  --design P --k K
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from src import config as C

OUT = C.RESULTS / "glmhmm"
BLOCK_SESSION_TYPES = ("biasedChoiceWorld", "ephysChoiceWorld")
LOG2 = np.log(2.0)


# =========================================================================== data
def bayes_prior(stim_side: np.ndarray, n_unbiased: int = C.UNBIASED_TRIALS,
                hazard: float = C.BLOCK_HAZARD, p_block: float = 0.8) -> np.ndarray:
    """Predictive p(stimulus on the right) for each trial, before it is shown.

    A Bayes-optimal observer that knows the IBL generative structure, in a
    constant-hazard approximation: the first `n_unbiased` trials are 50/50;
    afterwards a hidden block is 'right-likely' (p=0.8) or 'left-likely'
    (p=0.2), equally likely at first, and switches with probability `hazard`
    per trial. The observer updates on every past stimulus side
    (+1 right, -1 left, 0 = unknown/skip). Returns p_right in [0.2, 0.8].
    """
    T = len(stim_side)
    p_right = np.full(T, 0.5)
    b = 0.5  # belief that the block is right-likely
    for t in range(T):
        if t >= n_unbiased:
            # predictive for trial t (after the transition into t)
            p_right[t] = b * p_block + (1 - b) * (1 - p_block)
        s = stim_side[t]
        if t >= n_unbiased and s != 0:
            lr = p_block if s > 0 else 1 - p_block
            ll = 1 - p_block if s > 0 else p_block
            b = b * lr / (b * lr + (1 - b) * ll)
        if t >= n_unbiased - 1:
            b = b * (1 - hazard) + (1 - b) * hazard
    return p_right


def session_design(df: pd.DataFrame, with_prior: bool) -> tuple[np.ndarray, np.ndarray]:
    """Unnormalised design matrix and choices for one session (rows in trial order).

    Returns X (T x D) and y (T,) with y=1 right, 0 left, -1 no-go (masked).
    IBL choice coding: -1 = CCW = correct for right stimulus -> 'right'.
    """
    cl = np.nan_to_num(df["contrastLeft"].to_numpy(float))
    cr = np.nan_to_num(df["contrastRight"].to_numpy(float))
    stim = cr - cl
    ch = df["choice"].to_numpy()
    y = np.where(ch == -1, 1, np.where(ch == 1, 0, -1)).astype(np.int64)
    success = np.where(df["feedbackType"].to_numpy() > 0, 1.0, -1.0)
    T = len(y)
    pc = np.zeros(T)
    wsls = np.zeros(T)
    last_c, last_s = 0.0, 0.0
    for t in range(T):
        pc[t], wsls[t] = last_c, last_c * last_s
        if y[t] >= 0:
            last_c = 1.0 if y[t] == 1 else -1.0
            last_s = success[t]
    cols = [stim, pc, wsls]
    if with_prior:
        # stimulus side: from which contrast column is non-NaN (defined even at 0% contrast)
        side = np.where(df["contrastRight"].notna().to_numpy(), 1.0,
                        np.where(df["contrastLeft"].notna().to_numpy(), -1.0, 0.0))
        p = bayes_prior(side)
        cols.append(np.log(p / (1 - p)) / np.log(4.0))   # log-odds scaled to [-1, 1]
    cols.append(np.ones(T))
    return np.column_stack(cols), y


def session_prior_logodds(df: pd.DataFrame) -> np.ndarray:
    """Block-prior covariate for one session (same definition as the GLM-HMM input)."""
    side = np.where(df["contrastRight"].notna().to_numpy(), 1.0,
                    np.where(df["contrastLeft"].notna().to_numpy(), -1.0, 0.0))
    p = bayes_prior(side)
    return np.log(p / (1 - p)) / np.log(4.0)


def input_names(design: str) -> list[str]:
    return ["stim", "pc", "wsls"] + (["prior"] if design == "P" else []) + ["bias"]


def build_dataset(design: str) -> pd.DataFrame:
    """Concatenate all BWM mice's biased/ephys sessions into one long table."""
    sessions = pd.read_csv(C.IBL_DATA / "bwm_sessions.csv")
    bwm_eids = set(sessions["eid"])
    agg = pd.read_parquet(C.IBL_AGG / "bwm_trials_2026_Q2.pqt")
    rows = []
    for subject, sg in sessions.groupby("subject"):
        f = C.IBL_SUBJECT_TRIALS / subject / "_ibl_subjectTrials.table.pqt"
        if f.exists():
            st = pd.read_parquet(f)
            st = st[st["task_protocol"].isin(BLOCK_SESSION_TYPES)]
        else:
            st = pd.DataFrame(columns=["session"])
        # BWM recording sessions always come from the BWM aggregate, so that trial
        # indices match the table used for the neural analysis exactly.
        if len(st):
            st = st[~st["session"].astype(str).isin(bwm_eids)].copy()
            st["session"] = st["session"].astype(str)
        parts = [st] if len(st) else []
        for e in sg["eid"]:
            g = agg[agg["eid"] == e].copy()
            g["session"] = e
            g["task_protocol"] = "ephysChoiceWorld"
            g["session_start_time"] = sg.loc[sg["eid"] == e, "start_time"].iloc[0]
            parts.append(g)
        st = pd.concat(parts, ignore_index=False)
        st["_row"] = st.groupby("session").cumcount()   # original within-session order
        st["session_start_time"] = pd.to_datetime(st["session_start_time"], format="mixed")
        order = st.groupby("session")["session_start_time"].first().sort_values().index
        for i, eid in enumerate(order):
            g = st[st["session"] == eid].sort_values("_row")
            g = g.assign(trial=np.arange(len(g)))
            if design == "A90":
                g = g.iloc[: C.UNBIASED_TRIALS]
                g = g[g["probabilityLeft"] == 0.5]
                if (g["choice"] == 0).sum() >= 10 or len(g) < 10:
                    continue
            if len(g) < 10:
                continue
            X, y = session_design(g, with_prior=(design == "P"))
            d = pd.DataFrame(X, columns=input_names(design))
            d["y"] = y
            d["subject"] = subject
            d["eid"] = str(eid)
            d["trial"] = g["trial"].to_numpy()
            d["session_order"] = i
            d["is_bwm"] = str(eid) in bwm_eids
            d["probabilityLeft"] = g["probabilityLeft"].to_numpy()
            rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    # z-score stim over all trials (Ashwood); keep the scale for reproducibility
    mu, sd = df["stim"].mean(), df["stim"].std()
    df["stim"] = (df["stim"] - mu) / sd
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / design).mkdir(exist_ok=True)
    json.dump({"stim_mean": mu, "stim_sd": sd}, open(OUT / design / "stim_scale.json", "w"))
    # deterministic session-level CV folds, assigned within each mouse
    rng = np.random.default_rng(C.SEED)
    fold = {}
    for subject, g in df.groupby("subject"):
        eids = g["eid"].unique()
        perm = rng.permutation(len(eids))
        for j, e in zip(perm, eids):
            fold[e] = j % C.GLMHMM_N_FOLDS
    df["fold"] = df["eid"].map(fold)
    df.to_parquet(C.IBL_DATA / f"glmhmm_{design}.pqt")
    return df


# =========================================================================== core numerics
@njit(cache=True)
def _forward_backward(logB, log_pi, A, starts, ends):
    """Scaled forward-backward over independent sequences.

    logB: T x K emission log-likelihoods. Returns smoothed gamma (T x K),
    filtered alpha (T x K), predictive (T x K, p(z_t | y_<t)), summed xi (K x K)
    and total log-likelihood.
    """
    T, K = logB.shape
    gamma = np.zeros((T, K))
    alpha = np.zeros((T, K))
    pred = np.zeros((T, K))
    xi = np.zeros((K, K))
    ll = 0.0
    pi = np.exp(log_pi)
    beta = np.zeros((T, K))
    c = np.zeros(T)
    for s in range(len(starts)):
        a, b = starts[s], ends[s]
        # forward
        for t in range(a, b):
            m = logB[t].max()
            e = np.exp(logB[t] - m)
            if t == a:
                p = pi.copy()
            else:
                p = alpha[t - 1] @ A
            pred[t] = p
            v = p * e
            z = v.sum()
            alpha[t] = v / z
            c[t] = z
            ll += np.log(z) + m
        # backward
        beta[b - 1] = 1.0
        for t in range(b - 2, a - 1, -1):
            m = logB[t + 1].max()
            e = np.exp(logB[t + 1] - m)
            v = A @ (e * beta[t + 1])
            beta[t] = v / c[t + 1]
            # xi
            for i in range(K):
                for j in range(K):
                    xi[i, j] += alpha[t, i] * A[i, j] * e[j] * beta[t + 1, j] / c[t + 1]
        for t in range(a, b):
            g = alpha[t] * beta[t]
            gamma[t] = g / g.sum()
    return gamma, alpha, pred, xi, ll


def _emission_loglik(X, y, W):
    """T x K log p(y_t | z_t = k); masked (y<0) rows contribute 0."""
    eta = X @ W.T
    ll = np.where(y[:, None] == 1, -np.logaddexp(0, -eta), -np.logaddexp(0, eta))
    ll[y < 0] = 0.0
    return ll


@njit(cache=True)
def _obj_grad_hess(X, y, wts, w, lam):
    """Weighted logistic log posterior, its gradient and (negative) Hessian, in one pass."""
    T, D = X.shape
    f = 0.0
    for d in range(D):
        f -= 0.5 * lam * w[d] * w[d]
    g = -lam * w
    H = lam * np.eye(D)
    for t in range(T):
        eta = 0.0
        for d in range(D):
            eta += X[t, d] * w[d]
        # log p(y) = y*eta - log(1+exp(eta)), computed stably
        sp = eta + np.log1p(np.exp(-eta)) if eta > 0 else np.log1p(np.exp(eta))
        f += wts[t] * (y[t] * eta - sp)
        p = 1.0 / (1.0 + np.exp(-eta))
        r = wts[t] * (y[t] - p)
        v = wts[t] * p * (1.0 - p)
        for d in range(D):
            g[d] += r * X[t, d]
            xv = v * X[t, d]
            for e in range(D):
                H[d, e] += xv * X[t, e]
    return f, g, H


def _newton_logistic(X, y, wts, w0, sigma, n_iter=25, tol=1e-8):
    """MAP weighted logistic regression by damped Newton (step halving guarantees
    the concave objective never decreases, so the M-step cannot diverge)."""
    w = w0.copy()
    lam = 1.0 / sigma ** 2
    f, g, H = _obj_grad_hess(X, y, wts, w, lam)
    for _ in range(n_iter):
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            w_new = w + t * step
            f_new, g_new, H_new = _obj_grad_hess(X, y, wts, w_new, lam)
            if f_new >= f - 1e-12 * abs(f) or t < 1e-6:
                break
            t *= 0.5
        if f_new < f - 1e-12 * abs(f):      # no improving step found: stay
            break
        done = np.max(np.abs(w_new - w)) < tol
        w, f, g, H = w_new, f_new, g_new, H_new
        if done:
            break
    return w


@dataclass
class Params:
    W: np.ndarray          # K x D
    A: np.ndarray          # K x K
    pi: np.ndarray         # K

    @property
    def K(self):
        return self.W.shape[0]

    def to_dict(self):
        return {"W": self.W.tolist(), "A": self.A.tolist(), "pi": self.pi.tolist()}

    @staticmethod
    def from_dict(d):
        return Params(np.array(d["W"]), np.array(d["A"]), np.array(d["pi"]))


@dataclass
class Seqs:
    X: np.ndarray
    y: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    n_obs: int = field(init=False)

    def __post_init__(self):
        self.n_obs = int((self.y >= 0).sum())

    @staticmethod
    def from_df(df: pd.DataFrame, cols: list[str]) -> "Seqs":
        # rows must be grouped by session (contiguous)
        key = df["eid"].to_numpy()
        brk = np.flatnonzero(key[1:] != key[:-1]) + 1
        starts = np.concatenate([[0], brk]).astype(np.int64)
        ends = np.concatenate([brk, [len(df)]]).astype(np.int64)
        return Seqs(df[cols].to_numpy(float), df["y"].to_numpy(np.int64), starts, ends)


def log_prior(p: Params, sigma, alpha) -> float:
    lp = -0.5 * np.sum(p.W ** 2) / sigma ** 2
    lp += (alpha - 1) * np.sum(np.log(np.maximum(p.A, 1e-300)))
    return float(lp)


def loglik(p: Params, s: Seqs) -> float:
    logB = _emission_loglik(s.X, s.y, p.W)
    return float(_forward_backward(logB, np.log(p.pi), p.A, s.starts, s.ends)[4])


def posteriors(p: Params, s: Seqs):
    logB = _emission_loglik(s.X, s.y, p.W)
    gamma, alpha, pred, _, ll = _forward_backward(logB, np.log(p.pi), p.A, s.starts, s.ends)
    return gamma, alpha, pred, ll


def fit_em(s: Seqs, init: Params, sigma=C.GLMHMM_PRIOR_SIGMA, alpha=C.GLMHMM_DIRICHLET_ALPHA,
           max_iter=1000, tol=1e-7, verbose=False, newton_steps=5):
    """MAP-EM. Stops when the relative change of the log posterior < tol."""
    p = Params(init.W.copy(), init.A.copy(), init.pi.copy())
    K = p.K
    obs = s.y >= 0
    Xo = np.ascontiguousarray(s.X[obs])
    yo = s.y[obs].astype(float)
    hist = []
    prev = -np.inf
    for it in range(max_iter):
        logB = _emission_loglik(s.X, s.y, p.W)
        gamma, _, _, xi, ll = _forward_backward(logB, np.log(p.pi), p.A, s.starts, s.ends)
        lpost = ll + log_prior(p, sigma, alpha)
        hist.append(lpost)
        if verbose and it % 10 == 0:
            print(f"    it {it:3d}  log post {lpost:.3f}", flush=True)
        if np.isfinite(prev) and abs(lpost - prev) < tol * abs(prev):
            break
        if lpost < prev - 1e-6 * abs(prev):
            print(f"    [warn] log posterior decreased at it {it}: {prev:.4f} -> {lpost:.4f}")
        prev = lpost
        # M-step
        if K > 1:
            A = xi + (alpha - 1)
            p.A = A / A.sum(1, keepdims=True)
            pi = gamma[s.starts].sum(0) + 1e-3
            p.pi = pi / pi.sum()
        go = gamma[obs]
        for k in range(K):
            # generalised EM: a few warm-started Newton steps per M-step are enough
            p.W[k] = _newton_logistic(Xo, yo, go[:, k], p.W[k], sigma, n_iter=newton_steps)
    return p, hist


def init_params(s: Seqs, K: int, rng, w_glm=None, noise=0.2, sticky=C.GLMHMM_TRANSITION_STICKY_INIT):
    """Ashwood-style init: GLM weights + Gaussian noise per state; sticky A."""
    D = s.X.shape[1]
    if w_glm is None:
        obs = s.y >= 0
        w_glm = _newton_logistic(s.X[obs], s.y[obs].astype(float), np.ones(obs.sum()),
                                 np.zeros(D), C.GLMHMM_PRIOR_SIGMA)
    W = w_glm[None, :] + noise * rng.standard_normal((K, D))
    if K == 1:
        W = w_glm[None, :].copy()
    A = sticky * np.eye(K) + (1 - sticky) / max(K - 1, 1) * (1 - np.eye(K)) if K > 1 else np.ones((1, 1))
    A = A + 0.05 * rng.random((K, K)) * (K > 1)
    A = A / A.sum(1, keepdims=True)
    return Params(W, A, np.ones(K) / K), w_glm


def best_of_inits(s: Seqs, K: int, n_inits: int, seed: int, verbose=False,
                  short_iter: int = 100, n_finish: int = 3):
    """Two-stage multi-start: every init runs `short_iter` EM iterations; the
    best `n_finish` are then run to convergence. Returns the best final fit."""
    rng = np.random.default_rng(seed)
    if K == 1:
        init, _ = init_params(s, 1, rng)
        p, h = fit_em(s, init)
        return p, h[-1], [h[-1]]
    w_glm, short = None, []
    for i in range(n_inits):
        init, w_glm = init_params(s, K, rng, w_glm)
        p, h = fit_em(s, init, max_iter=short_iter)
        short.append((h[-1], p))
        if verbose:
            print(f"    init {i}: log post after {len(h)} its {h[-1]:.2f}", flush=True)
    short.sort(key=lambda x: -x[0])
    best, best_lp, finals = None, -np.inf, []
    for lp0, p0 in short[:n_finish]:
        p, h = fit_em(s, p0)
        finals.append(h[-1])
        if h[-1] > best_lp:
            best, best_lp = p, h[-1]
    return best, best_lp, [x[0] for x in short] + finals


def bits_per_trial(ll: float, s: Seqs) -> float:
    """Held-out LL relative to a coin flip, in bits per observed trial."""
    return (ll + s.n_obs * LOG2) / (s.n_obs * LOG2)


def sort_states(p: Params, names: list[str]) -> Params:
    """Canonical order: engaged (largest stim weight) first, then by bias weight
    (most leftward-bias first)."""
    si, bi = names.index("stim"), names.index("bias")
    eng = int(np.argmax(p.W[:, si]))
    rest = [k for k in np.argsort(p.W[:, bi]) if k != eng]
    order = [eng] + rest
    return Params(p.W[order], p.A[np.ix_(order, order)], p.pi[order])


def split_by_stim_gap(w: np.ndarray) -> list[int]:
    """Indices of 'engaged' states given each state's stimulus sensitivity.

    States are sorted by sensitivity and split at the largest gap in
    log-sensitivity; the high group is engaged. States with sensitivity <= 0
    are always disengaged. Parameter-free, so the same rule serves the IBL and
    Steinmetz models and every K. With one state, that state is engaged.
    """
    w = np.asarray(w, float)
    pos = np.flatnonzero(w > 0)
    if len(pos) <= 1:
        return [int(k) for k in pos]
    order = pos[np.argsort(-w[pos])]
    gaps = np.diff(-np.log(w[order]))          # positive log-ratios between neighbours
    cut = int(np.argmax(gaps))
    return sorted(int(k) for k in order[: cut + 1])


def engaged_states(p: Params, names: list[str]) -> list[int]:
    """Engaged states of the global fit (see `split_by_stim_gap`). Per-mouse states
    are aligned to the global states, so the same indices are used for every mouse."""
    return split_by_stim_gap(p.W[:, names.index("stim")])


def align_to(p: Params, ref: Params) -> Params:
    """Permute states of p to best match ref (Hungarian on weight distance)."""
    from scipy.optimize import linear_sum_assignment
    cost = ((p.W[:, None, :] - ref.W[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    order = r[np.argsort(c)]
    return Params(p.W[order], p.A[np.ix_(order, order)], p.pi[order])


# =========================================================================== protocol steps
def load(design: str) -> pd.DataFrame:
    return pd.read_parquet(C.IBL_DATA / f"glmhmm_{design}.pqt")


def step_global(design: str, ks=C.GLMHMM_K_RANGE, n_inits=C.GLMHMM_N_INITS_GLOBAL):
    df = load(design)
    names = input_names(design)
    s = Seqs.from_df(df, names)
    d = OUT / design
    for K in ks:
        t0 = time.time()
        p, lp, all_lp = best_of_inits(s, K, n_inits, C.SEED + K)
        p = sort_states(p, names)
        rec = {"K": K, "log_post": lp, "all_inits_log_post": all_lp, "n_obs": s.n_obs,
               "n_trials": len(df), "n_sessions": len(s.starts), "inputs": names, **p.to_dict(),
               "seconds": time.time() - t0}
        json.dump(rec, open(d / f"global_K{K}.json", "w"), indent=1)
        n_close = int(np.sum(np.array(all_lp) > lp - 1.0))
        print(f"[global {design}] K={K} log post {lp:.2f}  inits within 1 nat of best: "
              f"{n_close}/{len(all_lp)}  {time.time()-t0:.0f}s", flush=True)


def step_cv(design: str, ks=C.GLMHMM_K_RANGE, n_inits=3):
    """Global-model CV over held-out sessions; saves fold fits for per-mouse init."""
    df = load(design)
    names = input_names(design)
    d = OUT / design
    rows = []
    for f in range(C.GLMHMM_N_FOLDS):
        tr, te = df[df["fold"] != f], df[df["fold"] == f]
        s_tr, s_te = Seqs.from_df(tr, names), Seqs.from_df(te, names)
        for K in ks:
            p, lp, _ = best_of_inits(s_tr, K, n_inits, C.SEED + 100 * f + K)
            p = sort_states(p, names)
            ll_te = loglik(p, s_te)
            rows.append({"fold": f, "K": K, "test_ll": ll_te, "test_n": s_te.n_obs,
                         "test_bits": bits_per_trial(ll_te, s_te)})
            json.dump(p.to_dict(), open(d / f"cvfold{f}_K{K}.json", "w"))
            print(f"[cv {design}] fold {f} K={K} test bits/trial {rows[-1]['test_bits']:.4f}",
                  flush=True)
    pd.DataFrame(rows).to_csv(d / "cv_global.csv", index=False)


def _fit_mouse(args):
    design, subject, g, names, ks = args
    d = OUT / design
    out_rows, params = [], {}
    s_all = Seqs.from_df(g, names)
    for K in ks:
        gp = Params.from_dict(json.load(open(d / f"global_K{K}.json")))
        p, h = fit_em(s_all, gp)
        p = align_to(p, gp)
        params[K] = p.to_dict()
        for f in sorted(g["fold"].unique()):
            tr, te = g[g["fold"] != f], g[g["fold"] == f]
            if tr["eid"].nunique() < 1 or te["eid"].nunique() < 1:
                continue
            s_tr, s_te = Seqs.from_df(tr, names), Seqs.from_df(te, names)
            fp = Params.from_dict(json.load(open(d / f"cvfold{f}_K{K}.json")))
            pf, _ = fit_em(s_tr, fp)
            ll = loglik(pf, s_te)
            out_rows.append({"subject": subject, "K": K, "fold": f, "test_ll": ll,
                             "test_ll_global": loglik(fp, s_te), "test_n": s_te.n_obs})
    return subject, out_rows, params


def step_mouse(design: str, ks=C.GLMHMM_K_RANGE):
    df = load(design)
    names = input_names(design)
    d = OUT / design
    rows, allp = [], {}
    for subject, g in df.groupby("subject", sort=True):
        t0 = time.time()
        _, r, p = _fit_mouse((design, subject, g, names, ks))
        rows += r
        allp[subject] = p
        print(f"[mouse {design}] {subject}: {g['eid'].nunique()} sessions, "
              f"{len(g)} trials, {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(d / "cv_mouse.csv", index=False)
    json.dump(allp, open(d / "mouse_params.json", "w"))


def _one_se_rule(piv: pd.DataFrame) -> tuple[int, int]:
    """(best K by mean, smallest K within one paired SE of the best)."""
    means = piv.mean()
    kbest = int(means.idxmax())
    for K in sorted(piv.columns):
        diff = piv[kbest] - piv[K]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        if means[kbest] - means[K] <= se:
            return kbest, int(K)
    return kbest, kbest


def step_select(design: str) -> dict:
    """Model selection on per-mouse held-out bits/trial (mice with >= 2 sessions).

    Two parameter families are scored on exactly the same held-out sessions:
      "mouse"  per-mouse MAP fits initialised from the fold's global fit
      "global" the fold's global (pooled) fit itself
    The family with the higher best mean is used for labelling. Within it, K is
    the smallest K whose mean is within one standard error (of the paired
    per-mouse difference) of the best K.
    """
    d = OUT / design
    cv = pd.read_csv(d / "cv_mouse.csv")
    piv = {}
    for fam, col in (("mouse", "test_ll"), ("global", "test_ll_global")):
        per = cv.groupby(["subject", "K"]).agg(ll=(col, "sum"), n=("test_n", "sum")).reset_index()
        per["bits"] = (per["ll"] + per["n"] * LOG2) / (per["n"] * LOG2)
        piv[fam] = per.pivot(index="subject", columns="K", values="bits").dropna()
    fam = max(piv, key=lambda f: piv[f].mean().max())
    kbest, chosen = _one_se_rule(piv[fam])
    summ = pd.DataFrame({"mean_bits": piv["mouse"].mean(),
                         "sem_bits": piv["mouse"].std(ddof=1) / np.sqrt(len(piv["mouse"])),
                         "mean_bits_global_params": piv["global"].mean(),
                         "sem_bits_global_params": piv["global"].std(ddof=1) / np.sqrt(len(piv["global"])),
                         "frac_mice_permouse_better": (piv["mouse"] > piv["global"]).mean()})
    summ.index.name = "K"
    summ.to_csv(d / "k_selection.csv")
    res = {"family": fam, "K_best_mean": kbest, "K_chosen_1se": chosen, "n_mice": len(piv[fam]),
           "K_chosen_1se_mouse_family": _one_se_rule(piv["mouse"])[1],
           "K_chosen_1se_global_family": _one_se_rule(piv["global"])[1]}
    json.dump(res, open(d / "k_selection.json", "w"))
    print(summ, "\n", res)
    return res


def step_label(design: str, K: int, params: str = "mouse"):
    """Per-mouse posteriors for every trial. p(engaged) = summed posterior of the
    engaged states (`engaged_states` on the global fit)."""
    df = load(design)
    names = input_names(design)
    d = OUT / design
    mp = json.load(open(d / "mouse_params.json"))
    gp = Params.from_dict(json.load(open(d / f"global_K{K}.json")))
    eng = engaged_states(gp, names)
    json.dump({"K": K, "engaged_states": eng, "stim_weights": gp.W[:, names.index("stim")].tolist()},
              open(d / f"engaged_states_K{K}.json", "w"))
    out = []
    for subject, g in df.groupby("subject", sort=True):
        p = gp if params == "global" else Params.from_dict(mp[subject][str(K)])
        s = Seqs.from_df(g, names)
        gamma, alpha, pred, _ = posteriors(p, s)
        o = g[["subject", "eid", "trial", "session_order", "is_bwm", "y", "probabilityLeft"]].copy()
        for k in range(K):
            o[f"p_smooth_{k}"] = gamma[:, k]
            o[f"p_filt_{k}"] = alpha[:, k]
            o[f"p_pred_{k}"] = pred[:, k]
        out.append(o)
    out = pd.concat(out, ignore_index=True)
    thr = C.STATE_POSTERIOR_THRESHOLD
    for kind in ("smooth", "filt", "pred"):
        pe = out[[f"p_{kind}_{k}" for k in eng]].sum(1)
        out[f"p_engaged_{kind}"] = pe
        out[f"engaged_{kind}"] = np.where(pe > thr, 1, np.where(pe < 1 - thr, 0, -1))
    out["params_used"] = params
    out.to_parquet(d / f"states_K{K}.pqt")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "global", "cv", "mouse", "select", "label"])
    ap.add_argument("--design", default="P", choices=["P", "A90"])
    ap.add_argument("--k", type=int, nargs="*")
    ap.add_argument("--params", default="mouse", choices=["mouse", "global"])
    a = ap.parse_args(argv)
    ks = tuple(a.k) if a.k else C.GLMHMM_K_RANGE
    if a.step == "build":
        df = build_dataset(a.design)
        print(f"{a.design}: {len(df)} trials, {df['eid'].nunique()} sessions, "
              f"{df['subject'].nunique()} mice, BWM sessions {df.loc[df.is_bwm, 'eid'].nunique()}")
    elif a.step == "global":
        step_global(a.design, ks)
    elif a.step == "cv":
        step_cv(a.design, ks)
    elif a.step == "mouse":
        step_mouse(a.design, ks)
    elif a.step == "select":
        step_select(a.design)
    elif a.step == "label":
        step_label(a.design, ks[0], a.params)


if __name__ == "__main__":
    main()
