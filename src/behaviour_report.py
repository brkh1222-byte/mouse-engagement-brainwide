"""Behaviour-only report: frozen label file, session inclusion counts, figures.

Usage: python -m src.behaviour_report --k K [--design P]
Writes
  results/glmhmm_states.parquet     BWM trials with engagement posteriors and labels
  results/inclusion_sessions.csv    per-session label counts and inclusion flag
  results/behaviour_summary.json    key numbers for the paper and pre-registration
  results/figures/fig1_*.png/pdf    behaviour figures
"""
from __future__ import annotations

import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src import glmhmm as G

# validated categorical slots (dataviz default palette, light surface); fixed order
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED = "#1f1f1e", "#6b6a63"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.6})


def state_names(gp: G.Params, names, eng):
    bi = names.index("bias")
    out = []
    for k in range(gp.K):
        side = "right" if gp.W[k, bi] > 0 else "left"
        out.append(f"engaged ({side}-leaning)" if k in eng and len(eng) > 1 else
                   "engaged" if k in eng else f"disengaged ({side}-biased)")
    return out


def inclusion_table(states: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    sat = trials.groupby("eid").cumcount()
    t = trials.assign(trial=sat)[["eid", "trial", "stimOn_times", "saturation_stim_minus04_minus01"]]
    s = states.merge(t, on=["eid", "trial"], how="left", validate="1:1")
    s["usable"] = np.isfinite(s["stimOn_times"]) & ~s["saturation_stim_minus04_minus01"].astype(bool)
    rows = []
    for (subj, eid), g in s.groupby(["subject", "eid"]):
        u = g[g["usable"]]
        r = {"subject": subj, "eid": eid, "n_trials": len(g), "n_usable": len(u)}
        for kind in ("smooth", "filt"):
            r[f"n_eng_{kind}"] = int((u[f"engaged_{kind}"] == 1).sum())
            r[f"n_dis_{kind}"] = int((u[f"engaged_{kind}"] == 0).sum())
        for thr in (0.7, 0.9):
            r[f"n_eng_thr{thr}"] = int((u["p_engaged_smooth"] > thr).sum())
            r[f"n_dis_thr{thr}"] = int((u["p_engaged_smooth"] < 1 - thr).sum())
        rows.append(r)
    df = pd.DataFrame(rows)
    m = C.MIN_LABELLED_PER_CLASS
    df["include_behaviour"] = ((df["n_trials"] >= C.MIN_TRIALS_SESSION)
                               & (df["n_eng_smooth"] >= m) & (df["n_dis_smooth"] >= m))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--design", default="P")
    a = ap.parse_args(argv)
    K, design = a.k, a.design
    d = G.OUT / design
    names = G.input_names(design)
    gp = G.Params.from_dict(json.load(open(d / f"global_K{K}.json")))
    eng = G.engaged_states(gp, names)
    snames = state_names(gp, names, eng)
    st = pd.read_parquet(d / f"states_K{K}.pqt")
    post = st[[f"p_smooth_{k}" for k in range(K)]].to_numpy()
    st["state"] = post.argmax(1)
    bwm = st[st["is_bwm"]].copy()
    keep = ["subject", "eid", "trial", "y", "probabilityLeft", "state", "params_used",
            "p_engaged_smooth", "p_engaged_filt", "p_engaged_pred",
            "engaged_smooth", "engaged_filt", "engaged_pred"]
    C.RESULTS.mkdir(exist_ok=True)
    out = C.RESULTS / "glmhmm_states.parquet"
    bwm[keep + ["is_bwm"]].sort_values(["eid", "trial"]).reset_index(drop=True).to_parquet(out)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()

    trials = pd.read_parquet(C.IBL_AGG / "bwm_trials_2026_Q2.pqt")
    inc = inclusion_table(bwm, trials)
    inc.to_csv(C.RESULTS / "inclusion_sessions.csv", index=False)
    ok = inc[inc["include_behaviour"]]

    occ_all = np.bincount(st["state"], minlength=K) / len(st)
    occ_bwm = np.bincount(bwm["state"], minlength=K) / len(bwm)
    dwell = 1 / (1 - np.diag(gp.A))
    m = C.MIN_LABELLED_PER_CLASS
    summ = {
        "design": design, "K": K, "inputs": names, "engaged_states": eng, "state_names": snames,
        "params_used": str(st["params_used"].iloc[0]),
        "global_weights": np.round(gp.W, 3).tolist(), "expected_dwell_trials_global": np.round(dwell, 1).tolist(),
        "n_trials_fit": int(len(st)), "n_sessions_fit": int(st["eid"].nunique()), "n_mice": int(st["subject"].nunique()),
        "state_occupancy_all": np.round(occ_all, 4).tolist(), "state_occupancy_bwm": np.round(occ_bwm, 4).tolist(),
        "bwm_label_fractions": {k: round(float((bwm["engaged_smooth"] == v).mean()), 4)
                                for k, v in (("engaged", 1), ("disengaged", 0), ("unlabelled", -1))},
        "bwm_sessions": int(len(inc)), "sessions_included": int(len(ok)), "mice_included": int(ok["subject"].nunique()),
        "included_median_n_dis": float(ok["n_dis_smooth"].median()) if len(ok) else None,
        "included_median_n_eng": float(ok["n_eng_smooth"].median()) if len(ok) else None,
        "sessions_included_filtered_labels": int(((inc.n_trials >= 400) & (inc.n_eng_filt >= m) & (inc.n_dis_filt >= m)).sum()),
        "sessions_included_thr0.7": int(((inc.n_trials >= 400) & (inc["n_eng_thr0.7"] >= m) & (inc["n_dis_thr0.7"] >= m)).sum()),
        "sessions_included_thr0.9": int(((inc.n_trials >= 400) & (inc["n_eng_thr0.9"] >= m) & (inc["n_dis_thr0.9"] >= m)).sum()),
        "label_file": "results/glmhmm_states.parquet", "label_file_sha256": sha,
    }
    # disengagement vs block prior and time in session (behaviour only)
    b = bwm[bwm["engaged_smooth"] >= 0]
    summ["frac_disengaged_by_block"] = {str(k): round(float(v), 4) for k, v in
                                        (1 - b.groupby("probabilityLeft")["engaged_smooth"].mean()).items()}
    q = b.assign(q=pd.cut(b["trial"] / b.groupby("eid")["trial"].transform("max"), [0, .25, .5, .75, 1.0001],
                          labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True))
    summ["frac_disengaged_by_session_quarter"] = {str(k): round(float(v), 4) for k, v in
                                                  (1 - q.groupby("q", observed=True)["engaged_smooth"].mean()).items()}
    json.dump(summ, open(C.RESULTS / "behaviour_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))

    # ------------------------------------------------------------------ figures
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.4))
    # a: model selection
    sel = pd.read_csv(d / "k_selection.csv")
    a0 = ax[0, 0]
    a0.errorbar(sel["K"], sel["mean_bits"], yerr=sel["sem_bits"], color=PAL[0], lw=2, marker="o", ms=5, capsize=2)
    a0.plot(sel["K"], sel["mean_bits_global_params"], color=MUTED, lw=1.5, marker="s", ms=4)
    a0.annotate("per-mouse fits", (sel["K"].iloc[-1], sel["mean_bits"].iloc[-1]), xytext=(-4, 8),
                textcoords="offset points", ha="right", color=INK, fontsize=8)
    a0.annotate("global parameters", (sel["K"].iloc[-1], sel["mean_bits_global_params"].iloc[-1]), xytext=(-4, -12),
                textcoords="offset points", ha="right", color=INK, fontsize=8)
    a0.axvline(K, color=MUTED, lw=0.8, ls=":")
    a0.set(xlabel="number of states K", ylabel="held-out bits per trial (vs coin flip)", title="a  Model selection",
           xticks=sel["K"])
    # b: weights
    a1 = ax[0, 1]
    x = np.arange(len(names))
    for k in range(K):
        a1.plot(x, gp.W[k], color=PAL[k], lw=2, marker="o", ms=5)
        a1.annotate(snames[k], (x[0], gp.W[k, 0]), xytext=(6, 2), textcoords="offset points", fontsize=7.5, color=INK)
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set(xticks=x, xticklabels=names, ylabel="GLM weight", title="b  State weights (global fit)")
    # c: psychometric curves per state (empirical, BWM trials, argmax state)
    a2 = ax[0, 2]
    full = pd.read_parquet(C.IBL_DATA / f"glmhmm_{design}.pqt")[["eid", "trial", "stim"]]
    sc = json.load(open(d / "stim_scale.json"))
    j = bwm.merge(full, on=["eid", "trial"])
    j["contrast"] = (j["stim"] * sc["stim_sd"] + sc["stim_mean"]).round(4) * 100
    j = j[j["y"] >= 0]
    for k in range(K):
        g = j[j["state"] == k].groupby("contrast")["y"].mean()
        a2.plot(g.index, g.values, color=PAL[k], lw=2, marker="o", ms=4)
    a2.set(xlabel="signed contrast (%)", ylabel="P(choose right)", title="c  Choice by state", ylim=(0, 1))
    # d: example session
    a3 = ax[1, 0]
    cand = inc[inc["include_behaviour"]].assign(bal=lambda t: (t.n_dis_smooth / t.n_usable - 0.25).abs()).sort_values("bal")
    ex = cand["eid"].iloc[0] if len(cand) else inc["eid"].iloc[0]
    g = bwm[bwm["eid"] == ex].sort_values("trial")
    a3.fill_between(g["trial"], 0, 1, where=g["probabilityLeft"] == 0.8, color="#e6e5df", lw=0, step="mid")
    a3.plot(g["trial"], g["p_engaged_smooth"], color=PAL[0], lw=1.5)
    a3.axhline(0.8, color=MUTED, lw=0.8, ls=":"); a3.axhline(0.2, color=MUTED, lw=0.8, ls=":")
    a3.set(xlabel="trial", ylabel="P(engaged)", title="d  Example session (grey: left-likely blocks)", ylim=(0, 1))
    a3.grid(False)
    # e: disengaged trials per session
    a4 = ax[1, 1]
    v = np.clip(inc["n_dis_smooth"], 0, 400)
    a4.hist(v, bins=np.arange(0, 410, 10), color=PAL[0], edgecolor="white", lw=0.5)
    a4.axvline(m, color=INK, lw=1)
    a4.annotate(f"inclusion: ≥ {m}", (m, a4.get_ylim()[1] * 0.92), xytext=(5, 0), textcoords="offset points", fontsize=8)
    a4.set(xlabel="labelled disengaged trials per session (clipped at 400)", ylabel="sessions",
           title=f"e  {len(ok)} of {len(inc)} sessions included")
    # f: disengagement over session time
    a5 = ax[1, 2]
    b2 = b.assign(pos=b["trial"] / b.groupby("eid")["trial"].transform("max"))
    bins = np.linspace(0, 1, 11)
    cur = 1 - b2.groupby(pd.cut(b2["pos"], bins, include_lowest=True), observed=True)["engaged_smooth"].mean()
    a5.plot((bins[:-1] + bins[1:]) / 2, cur.values, color=PAL[0], lw=2, marker="o", ms=4)
    a5.set(xlabel="position in session (fraction of trials)", ylabel="fraction of labelled trials disengaged",
           title="f  Disengagement rises within sessions", ylim=(0, None))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig1_behaviour.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
