"""T3.1/T3.2: resumable, streaming decoding run (one session at a time).

For each included session:
  1. features: download spikes -> pre-stimulus counts -> cache -> delete raw (src.spikes)
  2. covariates: wheel + left-camera motion energy -> cache (src.covariates)
  3. decode every region with >= 10 good units, Models A and B, against the
     session-permutation null (200 donors) -> results/decoding/<eid>.parquet
A session whose result file exists is skipped, so the run can be stopped and
restarted at any time. Memory stays small (one session in RAM).

BLINDNESS GUARD: the real-data run refuses to start unless
preregistration/prereg.md is committed in git. `--simulate` runs the same
code path on synthetic sessions and needs no guard.

Run (after the pre-registration is committed), at most 2 workers:
  caffeinate -i nice -n 10 .venv/bin/python -m src.run_decoding --worker 0 --n-workers 2 &
  caffeinate -i nice -n 10 .venv/bin/python -m src.run_decoding --worker 1 --n-workers 2 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src import covariates as CV
from src import decode as D

OUT = C.RESULTS / "decoding"


def prereg_committed() -> bool:
    r = subprocess.run(["git", "log", "--oneline", "--", "preregistration/prereg.md"],
                       cwd=C.ROOT, capture_output=True, text=True)
    clean = subprocess.run(["git", "status", "--porcelain", "--", "preregistration/prereg.md"],
                           cwd=C.ROOT, capture_output=True, text=True)
    return bool(r.stdout.strip()) and not clean.stdout.strip()


def session_seed(eid: str) -> int:
    return C.SEED + zlib.crc32(eid.encode())


def decode_session(eid: str, subject: str, regions: dict[str, np.ndarray], labels: np.ndarray,
                   cov: pd.DataFrame, usable: np.ndarray, donor_labels: dict[str, np.ndarray],
                   donor_subject: dict[str, str], n_null=C.N_NULL, models=("A", "B"),
                   min_donors=100) -> pd.DataFrame:
    """Decode all regions of one session. `usable` marks trials that may be used
    (finite stimulus time, no saturation, finite covariates); other trials are
    set to unlabelled for the real labels and for every null sequence."""
    rng = np.random.default_rng(session_seed(eid))
    T_use, nulls = D.session_permutation_donors(len(labels), donor_labels, subject, donor_subject,
                                                n_null, rng, min_donors=min_donors)
    z = np.where(usable, labels, -1)[:T_use]
    nulls = [np.where(usable[:T_use], n, -1) for n in nulls]
    Cv = cov[CV.PRIMARY].to_numpy()[:T_use]
    rows = []
    for region, Y in regions.items():
        Y = Y[:T_use]
        for m in models:
            t0 = time.time()
            real = D.decode_auc(Y, z, Cv, m)
            null = np.array([D.decode_auc(Y, zn, Cv, m) for zn in nulls])
            res = D.UnitResult(real, null, T_use)
            rows.append({"eid": eid, "subject": subject, "region": region, "model": m,
                         "n_units": Y.shape[1], "T_used": T_use,
                         "n_eng": int((z == 1).sum()), "n_dis": int((z == 0).sum()),
                         "auc": real, "null_median": float(np.nanmedian(null)),
                         "effect": res.effect, "p": res.p, "n_null_valid": int(np.isfinite(null).sum()),
                         "null": json.dumps(np.round(null, 4).tolist()), "seconds": time.time() - t0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- real-data run
def run_real(worker: int, n_workers: int, limit: int | None):
    if not prereg_committed():
        sys.exit("REFUSING TO RUN: preregistration/prereg.md is not committed (or has uncommitted "
                 "changes). Neural decoding before registration would break blindness.")
    from src import spikes as SP
    from src import glmhmm as G
    incl = pd.read_csv(C.RESULTS / "inclusion_sessions.csv")
    incl = incl[incl["include_behaviour"]].sort_values("eid").reset_index(drop=True)
    labels_all = pd.read_parquet(C.RESULTS / "glmhmm_states.parquet")
    labels_all = labels_all[labels_all["is_bwm"]]
    by_eid = {e: g.sort_values("trial") for e, g in labels_all.groupby("eid")}
    donor_labels = {e: by_eid[e]["engaged_smooth"].to_numpy() for e in incl["eid"]}
    donor_subject = dict(zip(incl["eid"], incl["subject"]))
    trials_all = pd.read_parquet(C.IBL_AGG / "bwm_trials_2026_Q2.pqt")
    one = SP.get_one()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [r for i, r in incl.iterrows() if i % n_workers == worker]
    for r in todo[:limit]:
        eid = r["eid"]
        out = OUT / f"{eid}.parquet"
        skip = OUT / f"{eid}.skip.json"
        if out.exists() or skip.exists():
            continue
        try:
            trials = trials_all[trials_all["eid"] == eid].reset_index(drop=True)
            pids = [i["id"] for i in one.alyx.rest("insertions", "list", session=eid)]
            SP.session_features(one, eid, trials, pids, cleanup=False)
            prior = G.session_prior_logodds(trials)
            cov = CV.session_covariates(one, eid, trials, prior)
            SP.purge_raw()
            if cov["motion"].isna().all():
                json.dump({"reason": "no usable left-camera motion energy"}, open(skip, "w"))
                continue
            counts, beryl = SP.load_features(eid)
            usable = (np.isfinite(trials["stimOn_times"].to_numpy())
                      & ~trials["saturation_stim_minus04_minus01"].to_numpy(bool)
                      & np.isfinite(cov[CV.PRIMARY].to_numpy()).all(1)
                      & np.isfinite(counts).all(1))
            regions = SP.region_matrices(np.nan_to_num(counts), beryl)
            lab = donor_labels[eid]
            if (lab[usable] == 1).sum() < C.MIN_LABELLED_PER_CLASS or \
               (lab[usable] == 0).sum() < C.MIN_LABELLED_PER_CLASS:
                json.dump({"reason": "fewer than 30 usable labelled trials per class"}, open(skip, "w"))
                continue
            df = decode_session(eid, r["subject"], regions, lab, cov, usable,
                                donor_labels, donor_subject)
            df.to_parquet(out)
            print(f"[done] {eid}: {df['region'].nunique()} regions", flush=True)
        except Exception as e:  # keep the run alive; failures are retried on the next start
            print(f"[error] {eid}: {type(e).__name__}: {e}", flush=True)
            SP.purge_raw()


# --------------------------------------------------------------------------- simulation run
def run_simulated(n_sessions=12, n_null=20, out: Path | None = None) -> pd.DataFrame:
    from src.simulate import simulate_session
    rng = np.random.default_rng(C.SEED)
    sess = {f"sim{i:02d}": simulate_session(rng, T=int(rng.integers(450, 700))) for i in range(n_sessions)}
    donor_labels = {e: s["labels"] for e, s in sess.items()}
    donor_subject = {e: f"mouse{i}" for i, e in enumerate(sess)}
    frames = []
    for e, s in sess.items():
        usable = np.ones(len(s["labels"]), bool)
        frames.append(decode_session(e, donor_subject[e], s["regions"], s["labels"], s["cov"], usable,
                                     donor_labels, donor_subject, n_null=n_null, min_donors=6))
    df = pd.concat(frames, ignore_index=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.simulate:
        from src.aggregate import region_table
        df = run_simulated(out=C.RESULTS / "simulated" / "decoding_sim.parquet")
        print(region_table(df).to_string())
    else:
        run_real(a.worker, a.n_workers, a.limit)


if __name__ == "__main__":
    main()
