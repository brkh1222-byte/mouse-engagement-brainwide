"""GLM-HMM correctness checks on simulated data (no real data needed)."""
import numpy as np
import pandas as pd

from src import glmhmm as G


def simulate(W, A, pi, n_sess=40, T=500, seed=0):
    rng = np.random.default_rng(seed)
    K, D = W.shape
    Xs, ys, zs, eids = [], [], [], []
    for s in range(n_sess):
        stim = rng.choice([-1, -.25, -.125, -.0625, 0, .0625, .125, .25, 1], T) / 0.5
        z = np.zeros(T, int)
        y = np.zeros(T, int)
        X = np.zeros((T, D))
        pc, prev_s = 0.0, 0.0
        for t in range(T):
            z[t] = rng.choice(K, p=pi) if t == 0 else rng.choice(K, p=A[z[t - 1]])
            X[t] = [stim[t], pc, pc * prev_s, 1.0][:D]
            p = 1 / (1 + np.exp(-X[t] @ W[z[t]]))
            y[t] = rng.random() < p
            pc = 1.0 if y[t] else -1.0
            prev_s = 1.0 if (y[t] == (stim[t] > 0)) or stim[t] == 0 else -1.0
        Xs.append(X); ys.append(y); zs.append(z); eids += [f"s{s}"] * T
    df = pd.DataFrame(np.vstack(Xs), columns=["stim", "pc", "wsls", "bias"])
    df["y"] = np.concatenate(ys)
    df["eid"] = eids
    return df, np.concatenate(zs)


def test_recovers_three_state_model():
    W = np.array([[4.0, 0.2, 0.1, 0.0],     # engaged
                  [0.5, 0.5, 0.0, -2.0],    # biased left
                  [0.5, 0.5, 0.0, 2.0]])    # biased right
    A = np.array([[0.96, 0.02, 0.02], [0.05, 0.93, 0.02], [0.05, 0.02, 0.93]])
    pi = np.array([0.6, 0.2, 0.2])
    df, z = simulate(W, A, pi)
    df.loc[df.sample(frac=0.02, random_state=1).index, "y"] = -1     # some no-go trials
    s = G.Seqs.from_df(df, ["stim", "pc", "wsls", "bias"])
    p, lp, all_lp = G.best_of_inits(s, 3, n_inits=4, seed=1)
    p = G.sort_states(p, ["stim", "pc", "wsls", "bias"])
    assert np.abs(p.W - W).max() < 0.6, p.W
    assert np.abs(np.diag(p.A) - np.diag(A)).max() < 0.03
    gamma, alpha, pred, _ = G.posteriors(p, s)
    acc = np.mean(gamma.argmax(1) == z)
    assert acc > 0.85, acc
    # monotone EM: refit from the solution and check the log posterior never drops
    _, hist = G.fit_em(s, p, max_iter=20, tol=0)
    assert np.all(np.diff(hist) > -1e-6 * abs(hist[0]))
    # K selection by held-out LL prefers K=3 over K=1
    tr, te = df[df.eid.isin([f"s{i}" for i in range(30)])], df[~df.eid.isin([f"s{i}" for i in range(30)])]
    s_tr, s_te = (G.Seqs.from_df(d, ["stim", "pc", "wsls", "bias"]) for d in (tr, te))
    b = {}
    for K in (1, 3):
        pk, _, _ = G.best_of_inits(s_tr, K, 2, 0)
        b[K] = G.bits_per_trial(G.loglik(pk, s_te), s_te)
    assert b[3] > b[1]


def test_bayes_prior_tracks_blocks():
    rng = np.random.default_rng(0)
    side = np.concatenate([rng.choice([-1, 1], 90),
                           np.where(rng.random(200) < 0.8, 1, -1),
                           np.where(rng.random(200) < 0.8, -1, 1)])
    p = G.bayes_prior(side)
    assert np.all(p[:91] == 0.5)
    assert p[250:290].mean() > 0.7 and p[450:490].mean() < 0.3
    assert p.min() >= 0.2 - 1e-12 and p.max() <= 0.8 + 1e-12


def test_session_design_coding():
    df = pd.DataFrame({"contrastLeft": [1.0, np.nan, np.nan, 0.0],
                       "contrastRight": [np.nan, 0.25, 0.0, np.nan],
                       "choice": [1, -1, 0, 1],            # CW=left, CCW=right, 0=no-go
                       "feedbackType": [1, 1, -1, 1]})
    X, y = G.session_design(df, with_prior=False)
    assert list(y) == [0, 1, -1, 0]
    assert list(X[:, 0]) == [-1.0, 0.25, 0.0, 0.0]
    assert list(X[:, 1]) == [0, -1, 1, 1]                   # previous valid choice
    assert list(X[:, 2]) == [0, -1, 1, 1]                   # wsls = pc * prev outcome


def test_split_by_stim_gap():
    assert G.split_by_stim_gap([7.9, 0.8, 7.0, 1.2]) == [0, 2]
    assert G.split_by_stim_gap([7.4, 1.5, 1.4]) == [0]
    assert G.split_by_stim_gap([21.5, 11.8, 10.9, 6.9, 2.9]) == [0, 1, 2, 3]
    assert G.split_by_stim_gap([5.0, -0.3]) == [0]
    assert G.split_by_stim_gap([3.0]) == [0]
