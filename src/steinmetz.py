"""T4.1 (behaviour part): Steinmetz et al. 2019 trials tables from DANDI 000017.

Only the `intervals/trials` table of each NWB file is read, by HTTP range
requests (h5py + fsspec); spikes are not touched. Output:
data/steinmetz/trials.parquet and data/steinmetz/trials_columns.json
(column descriptions copied from the NWB files).

Usage: python -m src.steinmetz
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
    main()
