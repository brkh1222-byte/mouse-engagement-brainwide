"""Multinomial GLM-HMM recovery on simulated 3-outcome data."""
import numpy as np

from src import glmhmm_mn as M


def test_recovers_engaged_and_disengaged_states():
    rng = np.random.default_rng(0)
    W = np.array([[[5.0, -1.0, 0.0, 0.0, 0.5], [-1.0, 5.0, 0.0, 0.0, 0.5]],     # engaged
                  [[1.0, 0.0, 0.0, 0.0, -2.0], [0.0, 1.0, 0.0, 0.0, -2.0]]])    # disengaged (no-go)
    A = np.array([[0.97, 0.03], [0.06, 0.94]])
    Xs, ys, zs, starts, ends, n = [], [], [], [], [], 0
    for s in range(30):
        T = 300
        cL, cR = rng.choice([0, .25, .5, 1], T), rng.choice([0, .25, .5, 1], T)
        z = np.zeros(T, int)
        y = np.zeros(T, int)
        X = np.column_stack([cL, cR, np.zeros(T), np.zeros(T), np.ones(T)])
        for t in range(T):
            z[t] = 0 if t == 0 else rng.choice(2, p=A[z[t - 1]])
            y[t] = rng.choice(3, p=np.exp(M._logp(X[t:t + 1], W[z[t]]))[0])
        Xs.append(X); ys.append(y); zs.append(z); starts.append(n); n += T; ends.append(n)
    X, y, z = np.vstack(Xs), np.concatenate(ys), np.concatenate(zs)
    starts, ends = np.array(starts), np.array(ends)
    p, lp, _ = M.best_of_inits(X, y, starts, ends, 2, 3, 0)
    p = M.sort_states(p)
    assert M.engaged_states(p) == [0]
    assert abs(p.W[0, 0, 0] - 5) < 1.2 and abs(p.W[1, 0, 4] + 2) < 0.6
    g = M.posteriors(p, X, y, starts, ends)[0]
    assert np.mean(g.argmax(1) == z) > 0.85
    p1, _, _ = M.best_of_inits(X, y, starts, ends, 1, 1, 0)
    assert M.loglik(p, X, y, starts, ends) > M.loglik(p1, X, y, starts, ends)
