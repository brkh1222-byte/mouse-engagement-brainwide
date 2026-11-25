"""T3.1: decode engagement state from pre-stimulus population activity.

One "unit of analysis" = one Beryl region in one session (units pooled over
probes). For that unit we have
    Y  : trials x neurons pre-stimulus spike counts (from src.spikes)
    z  : trials engagement label (1 engaged, 0 disengaged, -1 unlabelled)
    Cv : trials x covariates (from src.covariates)

Decoder (pre-registered)
  * features: sqrt(count + 3/8) (Anscombe; stabilises Poisson variance),
    z-scored with training-fold statistics.
  * Model A ("raw"): features as they are.
  * Model B ("controlled", primary): each feature is first residualised on
    the covariates by ordinary least squares fitted on the training fold only.
  * L2 logistic regression, class_weight='balanced'. Outer CV: 5 contiguous
    folds of labelled trials (respects slow structure); inner CV: 3 contiguous
    folds to pick C from a fixed grid, maximising inner ROC-AUC.
  * Statistic: ROC-AUC of pooled out-of-fold decision values.

Null (pre-registered)
  * Primary, session permutation: the real neural data are decoded against
    engagement label sequences of *other* BWM sessions from *other* mice,
    resampled to the target session's length by relative position in the
    session. Keeps each sequence's real temporal structure (including the
    rise of disengagement at the end of sessions), breaks any true link to
    this session's neurons.
  * Secondary, pseudo-sessions: labels inferred from choices simulated from
    the mouse's own GLM-HMM on IBL-generated pseudo stimulus sequences
    (Findling et al. 2025-style), see `pseudo_session_labels`.
  * p = (1 + #{null >= real}) / (1 + n_null). Corrected effect = real - median(null).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import config as C


# --------------------------------------------------------------------------- folds
def contiguous_folds(n: int, k: int) -> list[np.ndarray]:
    """Split indices 0..n-1 into k contiguous blocks (as equal as possible)."""
    edges = np.linspace(0, n, k + 1).round().astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k)]


def _both_classes(y):
    return len(np.unique(y)) == 2


# --------------------------------------------------------------------------- preprocessing
def transform_counts(Y: np.ndarray) -> np.ndarray:
    return np.sqrt(Y + 0.375)


def residualise(F_tr, F_te, Cv_tr, Cv_te):
    """Remove the linear covariate component (fit on train only) from features."""
    Ctr = np.column_stack([np.ones(len(Cv_tr)), Cv_tr])
    Cte = np.column_stack([np.ones(len(Cv_te)), Cv_te])
    beta, *_ = np.linalg.lstsq(Ctr, F_tr, rcond=None)
    return F_tr - Ctr @ beta, F_te - Cte @ beta


def standardise(F_tr, F_te):
    mu, sd = F_tr.mean(0), F_tr.std(0)
    sd[sd < 1e-8] = 1.0
    return (F_tr - mu) / sd, (F_te - mu) / sd


def _prep(F_tr, F_te, Cv_tr, Cv_te, model):
    if model == "B":
        # standardise covariates with train stats too (numerical conditioning only)
        Cv_tr, Cv_te = standardise(Cv_tr, Cv_te)
        F_tr, F_te = residualise(F_tr, F_te, Cv_tr, Cv_te)
    return standardise(F_tr, F_te)


def _fit_predict(F_tr, y_tr, F_te, Cval):
    clf = LogisticRegression(C=Cval, class_weight="balanced", max_iter=2000)
    clf.fit(F_tr, y_tr)
    return clf.decision_function(F_te)


# --------------------------------------------------------------------------- decoding
def decode_auc(Y: np.ndarray, z: np.ndarray, Cv: np.ndarray | None, model: str = "B",
               n_outer=C.N_OUTER_FOLDS, n_inner=C.N_INNER_FOLDS, c_grid=C.C_GRID) -> float:
    """Nested-CV ROC-AUC for one region x session. Unlabelled trials (z<0) are dropped
    (trial order kept, so folds stay contiguous in time). Returns NaN if a fold
    lacks a class."""
    keep = z >= 0
    F = transform_counts(Y[keep].astype(float))
    y = z[keep].astype(int)
    Cv = None if Cv is None else Cv[keep]
    if model == "B" and Cv is None:
        raise ValueError("model B needs covariates")
    scores = np.full(len(y), np.nan)
    for te in contiguous_folds(len(y), n_outer):
        tr = np.setdiff1d(np.arange(len(y)), te)
        if not _both_classes(y[tr]):
            return np.nan
        # inner CV for C
        inner = contiguous_folds(len(tr), n_inner)
        best_c, best_auc = c_grid[0], -np.inf
        for Cval in c_grid:
            dec, lab = [], []
            ok = True
            for ite in inner:
                itr = np.setdiff1d(np.arange(len(tr)), ite)
                a, b = tr[itr], tr[ite]
                if not _both_classes(y[a]):
                    ok = False
                    break
                Fa, Fb = _prep(F[a], F[b], None if Cv is None else Cv[a],
                               None if Cv is None else Cv[b], model)
                dec.append(_fit_predict(Fa, y[a], Fb, Cval))
                lab.append(y[b])
            if not ok:
                continue
            dec, lab = np.concatenate(dec), np.concatenate(lab)
            auc = roc_auc_score(lab, dec) if _both_classes(lab) else -np.inf
            if auc > best_auc:
                best_c, best_auc = Cval, auc
        Ftr, Fte = _prep(F[tr], F[te], None if Cv is None else Cv[tr],
                         None if Cv is None else Cv[te], model)
        scores[te] = _fit_predict(Ftr, y[tr], Fte, best_c)
    if not _both_classes(y):
        return np.nan
    return float(roc_auc_score(y, scores))


# --------------------------------------------------------------------------- nulls
def warp_labels(donor: np.ndarray, T: int) -> np.ndarray:
    """Resample a donor label sequence to length T by relative position in the
    session (nearest neighbour). Start maps to start and end to end, so
    within-session trends (e.g. disengagement at the end of a session) stay
    aligned with the target session's own time course."""
    idx = np.round(np.linspace(0, len(donor) - 1, T)).astype(int)
    return donor[idx]


def session_permutation_donors(target_len: int, donor_labels: dict[str, np.ndarray],
                               target_subject: str, donor_subject: dict[str, str],
                               n_null: int, rng, min_donors: int = 100,
                               max_ratio: float = 2.0):
    """Label sequences for the session-permutation null.

    Donors are sessions of other mice whose length is within a factor
    `max_ratio` of the target's (so state dwell times are stretched by at most
    that factor); each is warped to the target length (`warp_labels`). If fewer
    than `min_donors` qualify, the length restriction is dropped. Sampling is
    without replacement when enough donors exist.
    Returns (target_len, list of label arrays of length target_len).
    """
    other = [e for e in donor_labels if donor_subject[e] != target_subject]
    if not other:
        raise ValueError("no donors")
    elig = [e for e in other
            if 1 / max_ratio <= len(donor_labels[e]) / target_len <= max_ratio]
    if len(elig) < min_donors:
        elig = other
    pick = rng.choice(len(elig), n_null, replace=len(elig) < n_null)
    return target_len, [warp_labels(donor_labels[elig[i]], target_len) for i in pick]


@dataclass
class UnitResult:
    auc: float
    null: np.ndarray
    T_used: int

    @property
    def p(self) -> float:
        n = np.isfinite(self.null)
        return float((1 + np.sum(self.null[n] >= self.auc)) / (1 + n.sum()))

    @property
    def effect(self) -> float:
        return float(self.auc - np.nanmedian(self.null))


def decode_with_null(Y, z, Cv, null_labels: list[np.ndarray], model="B", T_use=None) -> UnitResult:
    """Real decoding + decoding against each null label sequence (same trials)."""
    T = len(z) if T_use is None else T_use
    Y, z = Y[:T], z[:T]
    Cv = None if Cv is None else Cv[:T]
    real = decode_auc(Y, z, Cv, model)
    null = np.array([decode_auc(Y, zn[:T], Cv, model) for zn in null_labels])
    return UnitResult(real, null, T)


# --------------------------------------------------------------------------- aggregation
def fisher_combine(pvals: np.ndarray) -> float:
    from scipy.stats import combine_pvalues
    pvals = np.asarray(pvals, float)
    pvals = pvals[np.isfinite(pvals)]
    return float(combine_pvalues(pvals, method="fisher").pvalue) if len(pvals) else np.nan


def stouffer_combine(pvals: np.ndarray) -> float:
    from scipy.stats import combine_pvalues
    pvals = np.asarray(pvals, float)
    pvals = pvals[np.isfinite(pvals)]
    return float(combine_pvalues(pvals, method="stouffer").pvalue) if len(pvals) else np.nan


def bh_fdr(pvals: np.ndarray, q: float = C.FDR_Q) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg. Returns (reject mask, adjusted q-values)."""
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(pvals, float)
    rej, qv, _, _ = multipletests(p, alpha=q, method="fdr_bh")
    return rej, qv
