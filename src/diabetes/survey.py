"""Design-based prevalence estimates for NHANES.

NHANES is a stratified, clustered sample with unequal selection probabilities, so a
plain sample mean is not a US-population estimate. ``weighted_prevalence`` uses the
MEC exam weights and Taylor-series linearisation over the masked variance units
(strata and PSUs), which is the method NCHS recommends.
"""
import numpy as np
import pandas as pd


def weighted_prevalence(y, weight, stratum, psu, z=1.96):
    """Return (estimate, lower, upper) of a weighted proportion with a design-based CI."""
    d = pd.DataFrame({"y": np.asarray(y, float), "w": np.asarray(weight, float),
                      "h": np.asarray(stratum), "j": np.asarray(psu)})
    d = d[d["w"] > 0]
    total = d["w"].sum()
    p = float((d["w"] * d["y"]).sum() / total)
    d["z"] = d["w"] * (d["y"] - p) / total
    psu_totals = d.groupby(["h", "j"])["z"].sum()
    var = 0.0
    for _, zj in psu_totals.groupby(level="h"):
        n = len(zj)
        if n > 1:  # a stratum with a single PSU contributes no variance information
            var += n / (n - 1) * float(((zj - zj.mean()) ** 2).sum())
    se = var ** 0.5
    return p, max(0.0, p - z * se), min(1.0, p + z * se)
