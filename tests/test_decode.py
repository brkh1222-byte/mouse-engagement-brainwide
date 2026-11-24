"""Decoding pipeline checks on simulated sessions only (pre-registration blindness)."""
import numpy as np
import pandas as pd

from src import covariates as CV
from src import decode as D
from src import spikes as SP
from src.simulate import simulate_session


def test_contiguous_folds_cover_once():
    f = D.contiguous_folds(103, 5)
    allidx = np.concatenate(f)
    assert np.array_equal(allidx, np.arange(103))
    assert all(np.all(np.diff(x) == 1) for x in f)


def test_window_counts_matches_bruteforce():
    rng = np.random.default_rng(0)
    st = np.sort(rng.uniform(0, 100, 5000))
    sc = rng.choice([3, 7, 11], 5000)
    align = np.array([10.0, 50.0, np.nan, 99.0])
    out = SP.window_counts(st, sc, np.array([3, 7, 11]), align, (-0.4, -0.1))
    for i, t in enumerate(align):
        if np.isnan(t):
            assert np.all(np.isnan(out[i]))
            continue
        for j, c in enumerate([3, 7, 11]):
            assert out[i, j] == np.sum((st >= t - 0.4) & (st < t - 0.1) & (sc == c))


def test_covariate_helpers():
    t = np.arange(0, 10, 0.01)
    v = np.where(t > 5, 2.0, 0.0)
    m = CV.window_mean(t, v, np.array([3.0, 8.0]), (-0.4, -0.1))
    assert np.allclose(m, [0.0, 2.0])
    ws = CV.wheel_speed(t, 0.5 * t, np.array([3.0]), (-0.4, -0.1))
    assert np.allclose(ws, 0.5, atol=1e-6)
    ts = CV.trial_spline(500)
    assert ts.shape == (500, 4) and np.isfinite(ts).all()
    cr = CV.cumulative_reward(np.array([1.5, 0, 1.5, 1.5]))
    assert np.allclose(cr, [0, 1 / 3, 1 / 3, 2 / 3])


def test_model_A_vs_B_separates_true_effect_from_confounds():
    rng = np.random.default_rng(1)
    res = {r: {"A": [], "B": []} for r in ("ENG", "MOV", "PRI")}
    for _ in range(3):
        s = simulate_session(rng)
        Cv = s["cov"][CV.PRIMARY].to_numpy()
        for r in res:
            for m in ("A", "B"):
                res[r][m].append(D.decode_auc(s["regions"][r], s["labels"], Cv, m))
    mean = {r: {m: np.mean(v) for m, v in d.items()} for r, d in res.items()}
    assert mean["ENG"]["A"] > 0.65 and mean["ENG"]["B"] > 0.65, mean
    assert mean["MOV"]["A"] > 0.6, mean               # confound visible without controls
    assert mean["MOV"]["B"] < mean["MOV"]["A"] - 0.05, mean   # ... and removed by Model B


def test_session_permutation_null_calibrated_under_drift():
    """NUL region (drift only) must not be significant more often than alpha,
    while ENG is detected. Small numbers to keep the test fast."""
    rng = np.random.default_rng(2)
    sess = [simulate_session(rng, T=500, n_units=12) for _ in range(14)]
    donors = {f"s{i}": s["labels"] for i, s in enumerate(sess)}
    subj = {f"s{i}": f"m{i}" for i in range(len(sess))}
    p_nul, p_eng = [], []
    for i, s in enumerate(sess[:8]):
        T_use, nulls = D.session_permutation_donors(500, donors, f"m{i}", subj, 13, rng, min_donors=10)
        Cv = s["cov"][CV.PRIMARY].to_numpy()
        p_nul.append(D.decode_with_null(s["regions"]["NUL"], s["labels"], Cv, nulls, "B", T_use).p)
        p_eng.append(D.decode_with_null(s["regions"]["ENG"], s["labels"], Cv, nulls, "B", T_use).p)
    assert np.mean(np.array(p_eng) <= 1 / 14 + 1e-9) >= 0.75, p_eng   # ENG at the floor in most
    assert np.mean(np.array(p_nul) <= 0.05) <= 0.25, p_nul           # NUL not systematically significant


def test_aggregation():
    rej, q = D.bh_fdr(np.array([0.001, 0.01, 0.02, 0.5]))
    assert rej[:3].all() and not rej[3]
    assert D.fisher_combine([0.01, 0.02]) < 0.01
    assert 0 < D.stouffer_combine([0.5, 0.5]) <= 0.51


def test_pseudo_session_labels():
    from src import glmhmm as G
    from src import nulls as N
    rng = np.random.default_rng(3)
    pL, side = N.ibl_block_sides(700, rng)
    assert np.all(pL[:90] == 0.5) and set(np.unique(pL[90:])) <= {0.2, 0.8}
    W = np.array([[4.0, 0.1, 0.1, 0.5, 0.0], [0.3, 0.3, 0.0, 0.5, -1.5], [0.3, 0.3, 0.0, 0.5, 1.5]])
    A = np.array([[0.96, 0.02, 0.02], [0.05, 0.93, 0.02], [0.05, 0.02, 0.93]])
    p = G.Params(W, A, np.array([0.6, 0.2, 0.2]))
    lab = N.pseudo_session_labels(p, 600, np.array([1, .25, .125, .0625, 0]),
                                  {"stim_mean": 0.0, "stim_sd": 0.5}, rng)
    assert len(lab) == 600 and set(np.unique(lab)) <= {-1, 0, 1}
    assert 0.3 < np.mean(lab == 1) < 0.95 and np.mean(lab == 0) > 0.03


def test_end_to_end_simulated_run_and_aggregation():
    """Whole path: session loop -> donors -> decoding -> region table -> H2/H3 helpers."""
    from src import aggregate as AG
    from src.run_decoding import run_simulated
    df = run_simulated(n_sessions=8, n_null=12)
    assert set(df["region"]) == {"ENG", "MOV", "PRI", "NUL"} and set(df["model"]) == {"A", "B"}
    rt = AG.region_table(df, n_boot=200)
    b = rt[rt.model == "B"].set_index("region")
    assert b.loc["ENG", "sig_fisher"] and not b.loc["NUL", "sig_fisher"]
    assert b.loc["ENG", "median_effect"] > 0.1 and abs(b.loc["NUL", "median_effect"]) < 0.08
    a = rt[rt.model == "A"].set_index("region")
    assert a.loc["MOV", "median_effect"] > b.loc["MOV", "median_effect"]
    r = AG.h3_test(b["median_effect"], b["median_effect"] + 0.001, n_perm=200, n_boot=200, min_regions=4)
    assert r["testable"] and r["rho"] > 0.99
    assert not AG.h3_test(b["median_effect"], b["median_effect"])["testable"]   # < 8 shared regions


def test_blindness_guard_blocks_real_run():
    from src import run_decoding as R
    import pytest
    if R.prereg_committed():
        pytest.skip("pre-registration already committed")
    with pytest.raises(SystemExit):
        R.run_real(0, 1, 1)
