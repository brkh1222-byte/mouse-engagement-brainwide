"""T2.1: pre-stimulus spike counts per good unit, grouped by Beryl region.

Streaming design (disk budget <= 40 GB): for each session,
  1. download spikes.times / spikes.clusters / clusters tables for each probe
     into data/raw_scratch/<eid>/ (ONE cache),
  2. compute a trials x units count matrix for each window,
  3. save only the counts + unit table to data/features/<eid>.npz,
  4. delete the raw files.
The step is resumable: a session whose feature file exists is skipped.

Good units: IBL 'label' == 1 (all three single-unit QC metrics pass), as in
the Brain-Wide Map papers. Region: Allen atlas id remapped to Beryl; units in
'void'/'root' are dropped.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C


# --------------------------------------------------------------------------- pure functions (tested)
def window_counts(spike_times: np.ndarray, spike_clusters: np.ndarray, cluster_ids: np.ndarray,
                  align_times: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Spike counts (n_trials x n_clusters) in [t + w0, t + w1) for each align time t.

    spike_times must be sorted. Trials with NaN align time get NaN rows.
    """
    order = {c: i for i, c in enumerate(cluster_ids)}
    col = np.array([order.get(c, -1) for c in spike_clusters], dtype=np.int64) \
        if len(cluster_ids) < 5000 else _map_ids(spike_clusters, cluster_ids)
    n_t, n_c = len(align_times), len(cluster_ids)
    out = np.full((n_t, n_c), np.nan)
    ok = np.isfinite(align_times)
    lo = np.searchsorted(spike_times, align_times[ok] + window[0], side="left")
    hi = np.searchsorted(spike_times, align_times[ok] + window[1], side="left")
    rows = np.flatnonzero(ok)
    for r, a, b in zip(rows, lo, hi):
        c = col[a:b]
        c = c[c >= 0]
        out[r] = np.bincount(c, minlength=n_c)
    return out


def _map_ids(spike_clusters, cluster_ids):
    lut = np.full(int(max(spike_clusters.max(), cluster_ids.max())) + 1, -1, dtype=np.int64)
    lut[cluster_ids] = np.arange(len(cluster_ids))
    return lut[spike_clusters]


def beryl_acronyms(atlas_ids: np.ndarray) -> np.ndarray:
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    beryl_ids = br.remap(np.asarray(atlas_ids), source_map="Allen", target_map="Beryl")
    return br.id2acronym(beryl_ids)


def good_units(clusters: pd.DataFrame) -> pd.DataFrame:
    """Filter a clusters table (needs 'label' and 'atlas_id') to good units in real regions."""
    c = clusters[clusters["label"] == 1].copy()
    c["beryl"] = beryl_acronyms(c["atlas_id"].to_numpy())
    return c[~c["beryl"].isin(["void", "root"])]


# --------------------------------------------------------------------------- ONE-backed session processing
def get_one():
    from one.api import ONE
    kw = dict(base_url="https://openalyx.internationalbrainlab.org", silent=True,
              cache_dir=str(C.RAW_SCRATCH))
    try:
        return ONE(**kw)
    except ValueError:  # legacy shared public account (to be retired by IBL)
        return ONE(username="intbrainlab", password="international", **kw)


def load_probe(one, pid: str):
    """Spikes + clusters for one insertion, via brainbox's SpikeSortingLoader."""
    from brainbox.io.one import SpikeSortingLoader
    from iblatlas.atlas import AllenAtlas
    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=AllenAtlas())
    spikes, clusters, channels = ssl.load_spike_sorting(dataset_types=["spikes.times",
                                                                       "spikes.clusters"])
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    cl = pd.DataFrame({k: clusters[k] for k in ("cluster_id", "label", "atlas_id")})
    return spikes["times"], spikes["clusters"], cl


def session_features(one, eid: str, trials: pd.DataFrame, pids: list[str],
                     windows=(C.PRESTIM_WINDOW, C.PRESTIM_WINDOW_ALT), cleanup=True) -> Path:
    """Compute and cache pre-stimulus counts for all probes of one session."""
    out = C.FEATURE_CACHE / f"{eid}.npz"
    if out.exists():
        return out
    C.FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    blocks, units = {w: [] for w in windows}, []
    for pid in pids:
        st, sc, cl = load_probe(one, pid)
        gu = good_units(cl)
        for w in windows:
            blocks[w].append(window_counts(st, sc, gu["cluster_id"].to_numpy(),
                                           trials["stimOn_times"].to_numpy(), w))
        gu = gu.assign(pid=pid)
        units.append(gu[["pid", "cluster_id", "atlas_id", "beryl"]])
        del st, sc
    units = pd.concat(units, ignore_index=True)
    np.savez_compressed(out, **{f"counts_{w[0]}_{w[1]}": np.hstack(blocks[w]).astype(np.float32)
                                for w in windows},
                        unit_pid=units["pid"].to_numpy(str), unit_cluster=units["cluster_id"].to_numpy(),
                        unit_beryl=units["beryl"].to_numpy(str), n_trials=len(trials))
    if cleanup:
        purge_raw()
    return out


def purge_raw():
    """Delete raw downloads (keep the ONE cache tables so lookups stay fast)."""
    for p in Path(C.RAW_SCRATCH).glob("*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def load_features(eid: str, window=C.PRESTIM_WINDOW):
    z = np.load(C.FEATURE_CACHE / f"{eid}.npz", allow_pickle=False)
    return z[f"counts_{window[0]}_{window[1]}"], z["unit_beryl"]


def region_matrices(counts: np.ndarray, beryl: np.ndarray, min_units=C.MIN_GOOD_UNITS_REGION):
    """{region: trials x units} for regions with >= min_units good units (pooled over probes)."""
    return {r: counts[:, beryl == r] for r in np.unique(beryl) if (beryl == r).sum() >= min_units}
