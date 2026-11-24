"""T4.1 (behaviour part): Steinmetz et al. 2019 trials tables from DANDI 000017.

Only the `intervals/trials` table of each NWB file is read, by HTTP range
requests (h5py + fsspec); spikes are not touched. Output:
data/steinmetz/trials.parquet and data/steinmetz/trials_columns.json
(column descriptions copied from the NWB files).

Usage: python -m src.steinmetz        (download trials tables)
       python -m src.steinmetz fit    (GLM-HMM, labels, inclusion counts)
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src import config as C

DANDI_API = "https://api.dandiarchive.org/api"
COLS = ["start_time", "stop_time", "visual_stimulus_time", "go_cue", "response_time", "feedback_time",
        "feedback_type", "response_choice", "visual_stimulus_left_contrast",
        "visual_stimulus_right_contrast", "included", "rep_num"]


def list_assets() -> list[dict]:
    import requests
    r = requests.get(f"{DANDI_API}/dandisets/000017/versions/draft/assets/?page_size=100", timeout=60)
    r.raise_for_status()
    return sorted(r.json()["results"], key=lambda a: a["path"])


def read_trials(asset: dict, retries: int = 4) -> tuple[pd.DataFrame, dict]:
    import fsspec
    import h5py
    out = C.STEINMETZ_DATA / "trials" / (asset["path"].split("/")[-1].replace(".nwb", ".parquet"))
    desc_f = out.with_suffix(".json")
    if out.exists() and desc_f.exists():
        return pd.read_parquet(out), json.load(open(desc_f))
    url = f"{DANDI_API}/assets/{asset['asset_id']}/download/"
    for attempt in range(retries):
        try:
            fs = fsspec.filesystem("http")
            with fs.open(url, "rb", block_size=1 << 20) as fh, h5py.File(fh, "r") as f:
                tr = f["intervals/trials"]
                df = pd.DataFrame({c: tr[c][:] for c in COLS if c in tr})
                desc = {c: str(tr[c].attrs.get("description", "")) for c in COLS if c in tr}
            break
        except Exception as e:
            print(f"  [retry {attempt+1}] {asset['path']}: {e}", file=sys.stderr, flush=True)
            if attempt == retries - 1:
                raise
    sub, ses = asset["path"].split("/")[-1].replace(".nwb", "").split("_")
    df["subject"] = sub.replace("sub-", "")
    df["session"] = ses.replace("ses-", "")
    df["eid"] = df["subject"] + "_" + df["session"]
    df["trial"] = np.arange(len(df))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    json.dump(desc, open(desc_f, "w"), indent=1)
    print(f"  {asset['path']}: {len(df)} trials", flush=True)
    return df, desc


# --------------------------------------------------------------------------- GLM-HMM (behaviour only)
OUT = C.RESULTS / "glmhmm" / "steinmetz"


def load_design():
    """All sessions concatenated: X, y, eid, subject. Choice coding in the NWB files:
    response_choice +1 = left, -1 = right, 0 = no-go; we map to -1 left / +1 right / 0 no-go."""
    from src import glmhmm_mn as M
    df = pd.read_parquet(C.STEINMETZ_DATA / "trials.parquet").sort_values(["eid", "trial"])
    df["choice"] = -df["response_choice"].astype(int)
    df["feedback"] = df["feedback_type"]
    df["cL"] = df["visual_stimulus_left_contrast"]
    df["cR"] = df["visual_stimulus_right_contrast"]
    Xs, ys = zip(*[M.design(g) for _, g in df.groupby("eid", sort=False)])
    return df.reset_index(drop=True), np.vstack(Xs), np.concatenate(ys)


def fit(ks=(1, 2, 3, 4, 5), n_inits=20, n_folds=5):
    """Pooled fits, session-level CV (folds assigned within mouse), 1-SE rule across mice."""
    from src import glmhmm_mn as M
    from src.glmhmm import _one_se_rule
    df, X, y = load_design()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    fold = {}
    for _, g in df.groupby("subject"):
        e = g["eid"].unique()
        for j, x in zip(rng.permutation(len(e)), e):
            fold[x] = j % n_folds
    f = df["eid"].map(fold).to_numpy()
    eid = df["eid"].to_numpy()
    rows = []
    for K in ks:
        for k in range(n_folds):
            tr, te = f != k, f == k
            p, _, _ = M.best_of_inits(X[tr], y[tr], *M.seq_bounds(eid[tr]), K, max(n_inits // 4, 3), C.SEED + 10 * k + K)
            for subj in np.unique(df["subject"][te]):
                m = te & (df["subject"] == subj).to_numpy()
                ll = M.loglik(p, X[m], y[m], *M.seq_bounds(eid[m]))
                rows.append({"K": K, "fold": k, "subject": subj, "test_ll": ll, "test_n": int(m.sum())})
        p, lp, lps = M.best_of_inits(X, y, *M.seq_bounds(eid), K, n_inits, C.SEED + K)
        p = M.sort_states(p)
        json.dump({"K": K, "log_post": lp, "all_inits_log_post": lps, "inputs": M.NAMES,
                   "stim_sensitivity": M.stim_sensitivity(p).tolist(), **p.to_dict()},
                  open(OUT / f"global_K{K}.json", "w"), indent=1)
        print(f"[steinmetz] K={K} log post {lp:.1f}; inits within 1 nat: "
              f"{int(np.sum(np.array(lps) > lp - 1))}/{len(lps)}", flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(OUT / "cv.csv", index=False)
    per = cv.groupby(["subject", "K"]).agg(ll=("test_ll", "sum"), n=("test_n", "sum")).reset_index()
    per["bits"] = (per["ll"] + per["n"] * M.LOG3) / (per["n"] * np.log(2))      # vs uniform over 3 outcomes
    piv = per.pivot(index="subject", columns="K", values="bits").dropna()
    kbest, chosen = _one_se_rule(piv)
    pd.DataFrame({"mean_bits": piv.mean(), "sem_bits": piv.std(ddof=1) / np.sqrt(len(piv))}).to_csv(OUT / "k_selection.csv")
    json.dump({"K_best_mean": kbest, "K_chosen_1se": chosen, "n_mice": len(piv)}, open(OUT / "k_selection.json", "w"))
    print(piv.mean().round(4).to_dict(), "best", kbest, "chosen", chosen)
    # labels
    p = M.ParamsMN.from_dict(json.load(open(OUT / f"global_K{chosen}.json")))
    eng = M.engaged_states(p)
    g, a, pr, _ = M.posteriors(p, X, y, *M.seq_bounds(eid))
    out = df[["subject", "eid", "trial", "choice", "feedback", "cL", "cR", "included"]].copy()
    out["state"] = g.argmax(1)
    for kind, arr in (("smooth", g), ("filt", a), ("pred", pr)):
        pe = arr[:, eng].sum(1) if chosen > 1 else np.ones(len(out))
        out[f"p_engaged_{kind}"] = pe
        thr = C.STATE_POSTERIOR_THRESHOLD
        out[f"engaged_{kind}"] = np.where(pe > thr, 1, np.where(pe < 1 - thr, 0, -1))
    out.to_parquet(C.RESULTS / "steinmetz_states.parquet")
    per_s = out.groupby(["subject", "eid"]).agg(n_trials=("trial", "size"),
                                                n_eng=("engaged_smooth", lambda v: int((v == 1).sum())),
                                                n_dis=("engaged_smooth", lambda v: int((v == 0).sum())),
                                                n_excluded_by_steinmetz=("included", lambda v: int((~v.astype(bool)).sum()))).reset_index()
    m = C.MIN_LABELLED_PER_CLASS
    per_s["include_behaviour"] = (per_s["n_eng"] >= m) & (per_s["n_dis"] >= m)
    per_s.to_csv(C.RESULTS / "steinmetz_inclusion_sessions.csv", index=False)
    # agreement with Steinmetz's own disengagement flag (behaviour only)
    lab = out[out["engaged_smooth"] >= 0]
    summ = {"K": chosen, "K_best_mean": kbest, "engaged_states": eng, "inputs": M.NAMES,
            "weights": np.round(p.W, 2).tolist(), "stim_sensitivity": np.round(M.stim_sensitivity(p), 2).tolist(),
            "expected_dwell": np.round(1 / (1 - np.diag(p.A)), 1).tolist() if chosen > 1 else None,
            "occupancy": np.round(np.bincount(out["state"], minlength=chosen) / len(out), 3).tolist(),
            "n_sessions": int(per_s.shape[0]), "n_mice": int(out["subject"].nunique()), "n_trials": int(len(out)),
            "sessions_included": int(per_s["include_behaviour"].sum()),
            "mice_included": int(per_s.loc[per_s["include_behaviour"], "subject"].nunique()),
            "label_fractions": {k: round(float((out["engaged_smooth"] == v).mean()), 3)
                                for k, v in (("engaged", 1), ("disengaged", 0), ("unlabelled", -1))},
            "frac_disengaged_given_steinmetz_included": round(float((lab.loc[lab["included"].astype(bool), "engaged_smooth"] == 0).mean()), 3),
            "frac_disengaged_given_steinmetz_excluded": round(float((lab.loc[~lab["included"].astype(bool), "engaged_smooth"] == 0).mean()), 3)
            if (~lab["included"].astype(bool)).any() else None,
            "nogo_rate_by_label": {str(k): round(float(v), 3) for k, v in
                                   lab.assign(nogo=lab["choice"] == 0).groupby("engaged_smooth")["nogo"].mean().items()}}
    json.dump(summ, open(C.RESULTS / "steinmetz_behaviour_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


def main():
    assets = list_assets()
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(read_trials, assets))
    df = pd.concat([r[0] for r in res], ignore_index=True)
    C.STEINMETZ_DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.STEINMETZ_DATA / "trials.parquet")
    json.dump(res[0][1], open(C.STEINMETZ_DATA / "trials_columns.json", "w"), indent=1)
    print(f"Steinmetz: {df['eid'].nunique()} sessions, {df['subject'].nunique()} mice, {len(df)} trials")


if __name__ == "__main__":
    fit() if len(sys.argv) > 1 and sys.argv[1] == "fit" else main()
