"""T1.2: list Brain-Wide Map (BWM) sessions and download behaviour tables.

Behaviour only. Nothing here touches spikes.

Sources (all open, CC BY 4.0):
  * OpenAlyx REST (session metadata). Uses the legacy shared public account
    if no personal account is configured; IBL say it will be retired, so we
    fall back to the BWM aggregate table, which carries `eid` only.
  * Public AWS bucket `ibl-brain-wide-map-public`, read anonymously over HTTPS:
      - aggregates/2026_Q2_IBL_et_al_BWM/trials.pqt   (all BWM trials, 459 sessions)
      - aggregates/Subjects/<lab>/<subject>/_ibl_subjectTrials.table.*.pqt
        (each mouse's full training history; used to fit the GLM-HMM on
         more data than the 1-2 recording sessions per mouse)

Usage:  python -m src.ibl_sessions [--no-subjects]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src import config as C

S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


# --------------------------------------------------------------------------- bucket helpers
def s3_list(prefix: str) -> list[tuple[str, int]]:
    """List (key, size) under a prefix of the public bucket (anonymous, paginated)."""
    out, token = [], None
    while True:
        url = f"{C.IBL_BUCKET_URL}/?list-type=2&prefix={urllib.parse.quote(prefix)}"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", S3_NS):
            out.append((c.find("s3:Key", S3_NS).text, int(c.find("s3:Size", S3_NS).text)))
        if root.findtext("s3:IsTruncated", namespaces=S3_NS) == "true":
            token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)
        else:
            return out


def s3_get(key: str, dest: Path, size: int | None = None, retries: int = 5,
           timeout: float = 60) -> Path:
    """Download one object (with timeout and retries) unless already present at the right size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"{C.IBL_BUCKET_URL}/{urllib.parse.quote(key)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if size is None or tmp.stat().st_size == size:
                tmp.rename(dest)
                return dest
        except Exception as e:  # network hiccup: retry with backoff
            print(f"  [retry {attempt+1}/{retries}] {key}: {e}", file=sys.stderr, flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {key}")


def latest_revision(keys: list[str], stem: str) -> str | None:
    """Pick the newest revision of a dataset. ONE revisions live in '#YYYY-MM-DD#' folders;
    a key without a revision folder is the original (oldest) version."""
    cands = [k for k in keys if f"/{stem}." in k]
    if not cands:
        return None

    def rev(k: str) -> str:
        m = re.search(r"#(\d{4}-\d{2}-\d{2}[^#]*)#", k)
        return m.group(1) if m else "0000"
    return max(cands, key=rev)


# --------------------------------------------------------------------------- sessions
def bwm_trials() -> pd.DataFrame:
    dest = C.IBL_AGG / "bwm_trials_2026_Q2.pqt"
    s3_get(f"{C.BWM_AGG_PREFIX}/trials.pqt", dest)
    return pd.read_parquet(dest)


def bwm_session_metadata() -> pd.DataFrame:
    """Session table: eid, subject, lab, start_time, task_protocol, number."""
    try:
        from one.api import ONE
        try:
            one = ONE(base_url="https://openalyx.internationalbrainlab.org", silent=True,
                      cache_dir=str(C.DATA / "one_cache"))
        except ValueError:  # no personal account configured: legacy shared account
            one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab",
                      password="international", silent=True, cache_dir=str(C.DATA / "one_cache"))
        sess = one.alyx.rest("sessions", "list", tag="2024_Q2_IBL_et_al_BWM")
        df = pd.DataFrame(sess)[["id", "subject", "lab", "start_time", "number", "task_protocol"]]
        return df.rename(columns={"id": "eid"})
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[warn] OpenAlyx unavailable ({e}); falling back to eids from aggregate only",
              file=sys.stderr)
        return pd.DataFrame({"eid": bwm_trials()["eid"].unique()})


def build_session_table() -> pd.DataFrame:
    trials = bwm_trials()
    meta = bwm_session_metadata()
    per = trials.groupby("eid").agg(n_trials=("choice", "size"),
                                    n_nogo=("choice", lambda c: int((c == 0).sum())),
                                    n_bwm_include=("bwm_include", "sum")).reset_index()
    df = meta.merge(per, on="eid", how="outer", indicator=True)
    bad = df["_merge"] != "both"
    if bad.any():
        print(f"[warn] {bad.sum()} sessions not in both metadata and trials table", file=sys.stderr)
    df = df.drop(columns="_merge").sort_values(["subject", "start_time"]).reset_index(drop=True)
    C.IBL_DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.IBL_DATA / "bwm_sessions.csv", index=False)
    # per-session trials tables (T1.2 deliverable)
    C.IBL_TRIALS.mkdir(parents=True, exist_ok=True)
    for eid, g in trials.groupby("eid"):
        p = C.IBL_TRIALS / f"{eid}.pqt"
        if not p.exists():
            g.reset_index(drop=True).to_parquet(p)
    return df


# --------------------------------------------------------------------------- subject histories
def _download_one_subject(lab: str, subject: str) -> dict:
    sizes = dict(s3_list(f"aggregates/Subjects/{lab}/{subject}/"))
    keys = list(sizes)
    row = {"lab": lab, "subject": subject}
    for stem in ("_ibl_subjectTrials.table", "_ibl_subjectTraining.table",
                 "_ibl_subjectSessions.table"):
        k = latest_revision(keys, stem)
        row[stem] = k
        if k is not None:
            s3_get(k, C.IBL_SUBJECT_TRIALS / subject / f"{stem}.pqt", sizes.get(k))
    print(f"  {lab}/{subject}: {'ok' if row['_ibl_subjectTrials.table'] else 'MISSING'}", flush=True)
    return row


def download_subject_tables(sessions: pd.DataFrame, n_threads: int = 4) -> pd.DataFrame:
    """Download each BWM mouse's subjectTrials / subjectTraining / subjectSessions tables
    (I/O bound, so a few threads)."""
    pairs = sessions[["lab", "subject"]].drop_duplicates().itertuples(index=False)
    with ThreadPoolExecutor(n_threads) as ex:
        rows = list(ex.map(lambda ls: _download_one_subject(*ls), pairs))
    df = pd.DataFrame(rows)
    df.to_csv(C.IBL_SUBJECT_TRIALS / "manifest.csv", index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-subjects", action="store_true")
    a = ap.parse_args(argv)
    s = build_session_table()
    print(f"BWM sessions: {len(s)}  mice: {s['subject'].nunique()}  labs: {s['lab'].nunique()}  "
          f"trials: {int(s['n_trials'].sum())}")
    if not a.no_subjects:
        m = download_subject_tables(s)
        print(f"subject tables: {m['_ibl_subjectTrials.table'].notna().sum()}/{len(m)} mice")


if __name__ == "__main__":
    main()
