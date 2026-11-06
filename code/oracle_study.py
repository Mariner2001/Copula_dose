"""
oracle_study.py -- population-level (true-margin) analysis of the role of the copula.

Truth: Scenario 1 of Li et al. (2022) margins, Gaussian copula (latent probit) with
correlation alpha.  For a utility matrix (w1, w2) we compare
  * the oracle rule (true copula),
  * the independence rule (true margins, copula ignored),
  * dependence-robust rules (max-min over all copulas / over PQD copulas),
  * the best single fixed dose,
evaluating all rules under the true model.  Doses are optimised on a 201-point grid
and refined by golden-section search, so regrets are exact up to ~1e-10.  For the selected
utilities, Monte Carlo standard errors are obtained by resampling patients.

Outputs: results/oracle_selected.csv, results/oracle_heatmap.csv,
         results/oracle_symmetric.csv, results/oracle_meta.json
"""
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.optimize import minimize_scalar
from core import (bvn_cdf, sample_x, intercept_slope, BETA_E, BETA_T, kendall_tau)

rng = np.random.default_rng(20261017)
N = 20_000
X = sample_x(N, rng)
aE, sE = intercept_slope(X, BETA_E)
aT, sT = intercept_slope(X, BETA_T)
DG = np.linspace(-1.0, 1.0, 201)
STEP = DG[1] - DG[0]

etaE_g = aE[:, None] + sE[:, None] * DG[None, :]
etaT_g = aT[:, None] + sT[:, None] * DG[None, :]
pE_g, pT_g = ndtr(etaE_g), ndtr(etaT_g)
ind_g = pE_g * pT_g
M_g = np.minimum(pE_g, pT_g)
W_g = np.maximum(pE_g + pT_g - 1.0, 0.0)
_cache = {}


def p11_grid(alpha):
    if alpha not in _cache:
        _cache[alpha] = bvn_cdf(etaE_g, etaT_g, alpha)
    return _cache[alpha]


def p11_at(d, dep):
    """p11 for every patient at patient-specific doses d (shape (N,))."""
    eE, eT = aE + sE * d, aT + sT * d
    kind = dep[0]
    if kind == "gauss":
        return eE, eT, bvn_cdf(eE, eT, dep[1])
    pE, pT = ndtr(eE), ndtr(eT)
    if kind == "indep":
        return eE, eT, pE * pT
    if kind == "M":
        return eE, eT, np.minimum(pE, pT)
    if kind == "W":
        return eE, eT, np.maximum(pE + pT - 1.0, 0.0)
    raise ValueError(kind)


def U_at(d, dep, w1, w2):
    eE, eT, p11 = p11_at(d, dep)
    return w1 + (1 - w1) * ndtr(eE) - w1 * ndtr(eT) + (w1 + w2 - 1) * p11


def U_grid(dep, w1, w2):
    k = w1 + w2 - 1
    base = w1 + (1 - w1) * pE_g - w1 * pT_g
    if k == 0:
        return base
    p11 = {"gauss": lambda: p11_grid(dep[1]), "indep": lambda: ind_g,
           "M": lambda: M_g, "W": lambda: W_g}[dep[0]]()
    return base + k * p11


def golden(f, lo, hi, iters=48):
    g = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(iters):
        c = hi - g * (hi - lo)
        d = lo + g * (hi - lo)
        m = f(c) >= f(d)
        hi = np.where(m, d, hi)
        lo = np.where(m, lo, c)
    return 0.5 * (lo + hi)


def argmax_dose(dep, w1, w2):
    Ug = U_grid(dep, w1, w2)
    j = np.argmax(Ug, axis=1)
    d0 = DG[j]
    lo, hi = np.clip(d0 - STEP, -1, 1), np.clip(d0 + STEP, -1, 1)
    return golden(lambda d: U_at(d, dep, w1, w2), lo, hi)


def best_fixed(dep, w1, w2):
    Ug = U_grid(dep, w1, w2).mean(axis=0)
    j = int(np.argmax(Ug))
    lo, hi = max(-1.0, DG[j] - STEP), min(1.0, DG[j] + STEP)
    res = minimize_scalar(lambda d: -U_at(np.full(N, d), dep, w1, w2).mean(),
                          bounds=(lo, hi), method="bounded", options=dict(xatol=1e-10))
    cand = [(res.x, -res.fun)] + [(e, U_at(np.full(N, e), dep, w1, w2).mean()) for e in (lo, hi)]
    return max(cand, key=lambda t: t[1])


def boot_se_ratio(num, den, B=500, seed=2026):
    """Monte Carlo standard error, by resampling patients, of 100 * mean(num) / mean(den)."""
    if not np.any(num):
        return 0.0
    rb = np.random.default_rng(seed)
    est = []
    for _ in range(B):
        i = rb.integers(0, num.size, num.size)
        est.append(100 * num[i].mean() / den[i].mean())
    return float(np.std(est, ddof=1))


def analyse(w1, w2, alpha, robust=True, quad_check=False):
    k = w1 + w2 - 1
    truth = ("gauss", alpha)
    d_t = argmax_dose(truth, w1, w2)
    V_or = U_at(d_t, truth, w1, w2)
    d_i = argmax_dose(("indep",), w1, w2)
    V_in = U_at(d_i, truth, w1, w2)
    d_fix, V_fix = best_fixed(truth, w1, w2)
    row = dict(w1=w1, w2=w2, kappa=k, alpha=alpha, tau=kendall_tau("gauss", alpha),
               V_oracle=V_or.mean(), V_fixed=V_fix, d_fixed=d_fix,
               gain=V_or.mean() - V_fix, regret_ind=(V_or - V_in).mean(),
               regret_ind_max=(V_or - V_in).max(),
               share_dchange_01=np.mean(np.abs(d_i - d_t) >= 0.1),
               mean_abs_dchange=np.mean(np.abs(d_i - d_t)),
               share_corner_oracle=np.mean((d_t < -1 + 1e-6) | (d_t > 1 - 1e-6)),
               uniform_bound=abs(k) * kendall_tau("gauss", alpha) / 4)
    Dg = p11_grid(alpha) - ind_g
    row["mean_osc_bound"] = abs(k) * np.mean(Dg.max(axis=1) - Dg.min(axis=1))
    if robust:
        if k < 0:
            dep_f = dep_p = ("M",)
        elif k > 0:
            dep_f, dep_p = ("W",), ("indep",)
        else:
            dep_f = dep_p = ("indep",)
        d_f = argmax_dose(dep_f, w1, w2)
        d_p = argmax_dose(dep_p, w1, w2)
        R_f = V_or - U_at(d_f, truth, w1, w2)
        R_p = V_or - U_at(d_p, truth, w1, w2)
        row["regret_frechet_robust"] = R_f.mean()
        row["regret_pqd_robust"] = R_p.mean()
        # Monte Carlo standard errors (patient bootstrap) of the regrets as % of the gain
        g = V_or - U_at(np.full(N, d_fix), truth, w1, w2)
        row["se_rel_regret_ind"] = boot_se_ratio(V_or - V_in, g)
        row["se_rel_regret_pqd"] = boot_se_ratio(R_p, g)
        row["se_rel_regret_frechet"] = boot_se_ratio(R_f, g)
    if quad_check and k != 0:
        # second-order prediction kappa^2 D'(d0)^2 / (2h) at interior optima of G = U_ind
        eps = 1e-4
        interior = (d_i > -1 + 1e-3) & (d_i < 1 - 1e-3)
        di = np.clip(d_i, -1 + 2e-4, 1 - 2e-4)

        def D(d):
            eE, eT = aE + sE * d, aT + sT * d
            return bvn_cdf(eE, eT, alpha) - ndtr(eE) * ndtr(eT)
        G = lambda d: U_at(d, ("indep",), w1, w2)
        Dp = (D(di + eps) - D(di - eps)) / (2 * eps)
        h = -(G(di + eps) - 2 * G(di) + G(di - eps)) / eps ** 2
        pred = np.where(interior & (h > 0), k * k * Dp * Dp / (2 * np.maximum(h, 1e-12)), 0.0)
        row["regret_quad_pred"] = pred.mean()
        row["share_interior_ind"] = interior.mean()
        # where does the approximation fall short? patients with both optima interior vs the rest
        both = interior & (d_t > -1 + 1e-3) & (d_t < 1 - 1e-3)
        r = V_or - V_in
        row["regret_share_switch"] = float(r[~both].sum() / r.sum()) if r.sum() > 0 else np.nan
        row["quad_ratio_both_interior"] = (float(pred[both].sum() / r[both].sum())
                                           if r[both].sum() > 0 else np.nan)
    return row


if __name__ == "__main__":
    meta = {}
    # conditional binary correlation in the training design (d ~ U(-1,1))
    dtr = rng.uniform(-1, 1, N)
    eE, eT = aE + sE * dtr, aT + sT * dtr
    pE, pT = np.clip(ndtr(eE), 1e-12, 1 - 1e-12), np.clip(ndtr(eT), 1e-12, 1 - 1e-12)
    p11 = bvn_cdf(eE, eT, 0.8)
    phi = (p11 - pE * pT) / np.sqrt(pE * (1 - pE) * pT * (1 - pT))
    cells = np.stack([pE - p11, pT - p11, 1 - pE - pT + p11, p11])
    ok = np.all(cells > 1e-9, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        OR = (p11 * (1 - pE - pT + p11) / ((pE - p11) * (pT - p11)))[ok]
    meta["cond_phi_mean"] = float(phi.mean())
    meta["cond_phi_q"] = [float(q) for q in np.quantile(phi, [0.1, 0.5, 0.9])]
    meta["cond_OR_q"] = [float(q) for q in np.quantile(OR, [0.1, 0.5, 0.9])]
    meta["N_pop"] = N

    # (a) selected utilities, alpha = 0.8 ------------------------------------
    sel = [("TFE", 0.0, 0.0), ("k-0.6", 0.2, 0.2), ("U1", 0.3, 0.5), ("U3", 0.5, 0.3),
           ("U2", 0.5, 0.5), ("APP", 0.6, 0.4), ("k+0.6", 0.8, 0.8), ("NFT", 1.0, 1.0)]
    rows = []
    for lab, w1, w2 in sel:
        r = analyse(w1, w2, 0.8, robust=True, quad_check=True)
        r["label"] = lab
        rows.append(r)
        print(lab, {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()})
    pd.DataFrame(rows).to_csv("../results/oracle_selected.csv", index=False)

    # (b) heat map over (w1, w2), alpha = 0.8 --------------------------------
    grid = np.round(np.linspace(0, 1, 21), 4)
    hm = []
    for w1 in grid:
        for w2 in grid:
            hm.append(analyse(float(w1), float(w2), 0.8, robust=False))
    pd.DataFrame(hm).to_csv("../results/oracle_heatmap.csv", index=False)
    print("heatmap done")

    # (c) symmetric utilities w1 = w2 = w, several alphas ---------------------
    sym = []
    for alpha in (0.3, 0.5, 0.8, 0.95):
        for w in np.round(np.linspace(0, 1, 41), 4):
            if abs(2 * w - 1) < 1e-12:
                continue
            sym.append(analyse(float(w), float(w), alpha, robust=False, quad_check=True))
        print("alpha", alpha, "done")
    pd.DataFrame(sym).to_csv("../results/oracle_symmetric.csv", index=False)
    with open("../results/oracle_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
