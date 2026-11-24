"""T3.2: region-level aggregation of region x session decoding results."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src import decode as D

GROUP_H = ["MRN", "SCm", "PAG", "MOs", "ACAd"]
GROUP_S = ["VISp", "AUDp", "SSp-bfd", "SSp-ul", "SSp-ll", "SSp-m", "SSp-n"]


def region_table(df: pd.DataFrame, min_sessions: int = 2, n_boot: int = 2000,
                 seed: int = C.SEED) -> pd.DataFrame:
    """One row per region x model: Fisher and Stouffer combined p, BH q (within model),
    median effect with bootstrap 95% CI over sessions."""
    rng = np.random.default_rng(seed)
    rows = []
    for (model, region), g in df.dropna(subset=["auc"]).groupby(["model", "region"]):
        if g["eid"].nunique() < min_sessions:
            continue
        eff = g["effect"].to_numpy()
        boot = np.median(rng.choice(eff, (n_boot, len(eff))), axis=1)
        rows.append({"model": model, "region": region, "n_sessions": len(g),
                     "n_mice": g["subject"].nunique(), "median_effect": float(np.median(eff)),
                     "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
                     "median_auc": float(g["auc"].median()),
                     "p_fisher": D.fisher_combine(g["p"]), "p_stouffer": D.stouffer_combine(g["p"])})
    out = pd.DataFrame(rows)
    for model, idx in out.groupby("model").groups.items():
        for m in ("fisher", "stouffer"):
            rej, q = D.bh_fdr(out.loc[idx, f"p_{m}"].to_numpy())
            out.loc[idx, f"q_{m}"] = q
            out.loc[idx, f"sig_{m}"] = rej
    return out.sort_values(["model", "p_fisher"]).reset_index(drop=True)


def h2_test(df: pd.DataFrame, model: str = "B") -> dict:
    """Mixed model: effect ~ group + (1|mouse) + (1|session); one-sided p for group H > S.
    Falls back to a Welch t-test on region medians if the model does not converge."""
    import statsmodels.formula.api as smf
    from scipy import stats
    d = df[(df["model"] == model) & df["region"].isin(GROUP_H + GROUP_S)].dropna(subset=["effect"]).copy()
    d["is_H"] = d["region"].isin(GROUP_H).astype(int)
    try:
        m = smf.mixedlm("effect ~ is_H", d, groups=d["subject"], re_formula="1",
                        vc_formula={"session": "0 + C(eid)"}).fit(reml=True)
        if not m.converged:
            raise RuntimeError("not converged")
        z = m.params["is_H"] / m.bse["is_H"]
        return {"method": "mixedlm", "coef": float(m.params["is_H"]), "se": float(m.bse["is_H"]),
                "p_one_sided": float(stats.norm.sf(z)), "n_rows": len(d)}
    except Exception as e:
        med = d.groupby(["region", "is_H"])["effect"].median().reset_index()
        t = stats.ttest_ind(med.loc[med.is_H == 1, "effect"], med.loc[med.is_H == 0, "effect"],
                            equal_var=False, alternative="greater")
        return {"method": f"welch fallback ({e})", "t": float(t.statistic),
                "p_one_sided": float(t.pvalue), "n_rows": len(d)}


def h3_test(eff_a: pd.Series, eff_b: pd.Series, n_perm: int = 10000, n_boot: int = 10000,
            min_regions: int = 8, seed: int = C.SEED) -> dict:
    """Spearman rho between two region-effect series over shared regions; one-sided
    region-label permutation p; bootstrap CI."""
    from scipy.stats import spearmanr
    shared = eff_a.index.intersection(eff_b.index)
    if len(shared) < min_regions:
        return {"testable": False, "n_shared": len(shared)}
    a, b = eff_a[shared].to_numpy(), eff_b[shared].to_numpy()
    rng = np.random.default_rng(seed)
    rho = spearmanr(a, b).statistic
    perm = np.array([spearmanr(a, rng.permutation(b)).statistic for _ in range(n_perm)])
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    boot = np.array([spearmanr(a[i], b[i]).statistic for i in idx])
    return {"testable": True, "n_shared": len(shared), "rho": float(rho),
            "p_one_sided": float((1 + np.sum(perm >= rho)) / (1 + n_perm)),
            "ci_lo": float(np.nanpercentile(boot, 2.5)), "ci_hi": float(np.nanpercentile(boot, 97.5))}
