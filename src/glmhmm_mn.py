"""Multinomial GLM-HMM for 3-outcome tasks (Steinmetz et al. 2019). Behaviour only.

Outcome y in {0: left choice, 1: right choice, 2: no-go}; no-go is the
reference class. In state k:  p(y=c) = softmax_c(W[k, c] . x), W[k, 2] = 0.
Same MAP-EM scheme, priors, multi-start and forward-backward as src.glmhmm.
Inputs: contrast left, contrast right, previous choice (+1 right, -1 left,
0 after a no-go or at session start), win-stay/lose-switch (previous choice x
previous outcome), bias.
Stimulus sensitivity of a state = W[k, left, cL] + W[k, right, cR].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from src import config as C
from src.glmhmm import _forward_backward

NAMES = ["cL", "cR", "pc", "wsls", "bias"]
LOG3 = np.log(3.0)


@dataclass
class ParamsMN:
    W: np.ndarray   # K x 2 x D
    A: np.ndarray
    pi: np.ndarray

    @property
    def K(self):
        return self.W.shape[0]

    def to_dict(self):
        return {"W": self.W.tolist(), "A": self.A.tolist(), "pi": self.pi.tolist()}

    @staticmethod
    def from_dict(d):
        return ParamsMN(np.array(d["W"]), np.array(d["A"]), np.array(d["pi"]))


def design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """X (T x 5), y (T,) for one session. `choice`: -1 left, +1 right, 0 no-go
    (already mapped by the caller); `feedback`: +1 / -1."""
    ch = df["choice"].to_numpy()
    fb = df["feedback"].to_numpy()
    T = len(df)
    pc = np.zeros(T)
    ws = np.zeros(T)
    pc[1:] = ch[:-1]
    ws[1:] = ch[:-1] * np.sign(fb[:-1])
    X = np.column_stack([df["cL"].to_numpy(float), df["cR"].to_numpy(float), pc, ws, np.ones(T)])
    y = np.where(ch == -1, 0, np.where(ch == 1, 1, 2)).astype(np.int64)
    return X, y


def _logp(X, Wk):
    eta = np.column_stack([X @ Wk[0], X @ Wk[1], np.zeros(len(X))])
    return eta - logsumexp(eta, axis=1, keepdims=True)


def emission_loglik(X, y, W):
    return np.column_stack([_logp(X, W[k])[np.arange(len(y)), y] for k in range(W.shape[0])])


def _mstep_state(X, Y1h, g, w0, sigma):
    D = X.shape[1]

    def f(w):
        Wk = w.reshape(2, D)
        lp = _logp(X, Wk)
        val = -(g[:, None] * Y1h * lp).sum() + 0.5 * (w ** 2).sum() / sigma ** 2
        R = g[:, None] * (Y1h[:, :2] - np.exp(lp[:, :2]))
        grad = -(R.T @ X).ravel() + w / sigma ** 2
        return val, grad
    r = minimize(f, w0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 50})
    return r.x.reshape(2, D)


def fit_em(X, y, starts, ends, init: ParamsMN, sigma=C.GLMHMM_PRIOR_SIGMA,
           alpha=C.GLMHMM_DIRICHLET_ALPHA, max_iter=500, tol=1e-7):
    p = ParamsMN(init.W.copy(), init.A.copy(), init.pi.copy())
    Y1h = np.eye(3)[y]
    hist, prev = [], -np.inf
    for _ in range(max_iter):
        logB = emission_loglik(X, y, p.W)
        gamma, _, _, xi, ll = _forward_backward(logB, np.log(p.pi), p.A, starts, ends)
        lpost = ll - 0.5 * (p.W ** 2).sum() / sigma ** 2 + (alpha - 1) * np.log(np.maximum(p.A, 1e-300)).sum()
        hist.append(lpost)
        if np.isfinite(prev) and abs(lpost - prev) < tol * abs(prev):
            break
        prev = lpost
        if p.K > 1:
            A = xi + (alpha - 1)
            p.A = A / A.sum(1, keepdims=True)
            pi = gamma[starts].sum(0) + 1e-3
            p.pi = pi / pi.sum()
        for k in range(p.K):
            p.W[k] = _mstep_state(X, Y1h, gamma[:, k], p.W[k], sigma)
    return p, hist


def loglik(p: ParamsMN, X, y, starts, ends) -> float:
    return float(_forward_backward(emission_loglik(X, y, p.W), np.log(p.pi), p.A, starts, ends)[4])


def posteriors(p: ParamsMN, X, y, starts, ends):
    g, a, pr, _, ll = _forward_backward(emission_loglik(X, y, p.W), np.log(p.pi), p.A, starts, ends)
    return g, a, pr, ll


def best_of_inits(X, y, starts, ends, K, n_inits, seed, noise=0.5,
                  sticky=C.GLMHMM_TRANSITION_STICKY_INIT):
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    Y1h = np.eye(3)[y]
    w_glm = _mstep_state(X, Y1h, np.ones(len(y)), np.zeros((2, D)), C.GLMHMM_PRIOR_SIGMA)
    best, best_lp, lps = None, -np.inf, []
    for _ in range(n_inits if K > 1 else 1):
        W = np.repeat(w_glm[None], K, 0) + (noise * rng.standard_normal((K, 2, D)) if K > 1 else 0)
        A = np.ones((1, 1)) if K == 1 else sticky * np.eye(K) + (1 - sticky) / (K - 1) * (1 - np.eye(K))
        p, h = fit_em(X, y, starts, ends, ParamsMN(W, A, np.ones(K) / K))
        lps.append(h[-1])
        if h[-1] > best_lp:
            best, best_lp = p, h[-1]
    return best, best_lp, lps


def stim_sensitivity(p: ParamsMN) -> np.ndarray:
    return p.W[:, 0, NAMES.index("cL")] + p.W[:, 1, NAMES.index("cR")]


def sort_states(p: ParamsMN) -> ParamsMN:
    order = np.argsort(-stim_sensitivity(p))
    return ParamsMN(p.W[order], p.A[np.ix_(order, order)], p.pi[order])


def engaged_states(p: ParamsMN, frac: float = 0.5) -> list[int]:
    s = stim_sensitivity(p)
    return [int(k) for k in np.flatnonzero(s >= frac * s.max())]


def seq_bounds(eids: np.ndarray):
    brk = np.flatnonzero(eids[1:] != eids[:-1]) + 1
    return (np.concatenate([[0], brk]).astype(np.int64),
            np.concatenate([brk, [len(eids)]]).astype(np.int64))
