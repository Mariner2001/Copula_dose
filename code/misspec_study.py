"""
misspec_study.py -- copula-family misspecification and dose-dependent dependence
(population level, true margins of Scenario 1 of Li et al., 2022).

A misspecified family K_theta is represented by its pseudo-true parameter, i.e. the
maximiser of the expected log-likelihood under the training design (d ~ U(-1,1),
constrained normal covariates).  This is the large-sample limit of fitting that
family by maximum likelihood (with correctly specified margins).

Rules are evaluated on the same 20,000 patients as in oracle_study.py; Monte Carlo
standard errors are obtained by resampling patients.

Outputs: results/misspec_family.csv, results/misspec_conditional.csv,
         results/misspec_meta.json
"""
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.optimize import minimize_scalar, brentq
from core import (bvn_cdf, K_family, PAR, kendall_tau, kendall_tau_numeric, sample_x,
                  intercept_slope, BETA_E, BETA_T)

rng = np.random.default_rng(777)

# design sample for pseudo-true parameters ----------------------------------
ND = 20_000
Xd = sample_x(ND, rng)
dd = rng.uniform(-1, 1, ND)
aEd, sEd = intercept_slope(Xd, BETA_E)
aTd, sTd = intercept_slope(Xd, BETA_T)
CLIP = 1e-9
pEd = np.clip(ndtr(aEd + sEd * dd), CLIP, 1 - CLIP)
pTd = np.clip(ndtr(aTd + sTd * dd), CLIP, 1 - CLIP)

# evaluation population: the same 20,000 patients as in oracle_study.py -----
NP = 20_000
Xp = sample_x(NP, np.random.default_rng(20261017))
aE, sE = intercept_slope(Xp, BETA_E)
aT, sT = intercept_slope(Xp, BETA_T)
DG = np.linspace(-1, 1, 201)
STEP = DG[1] - DG[0]


def margins(d):
    if np.ndim(d) == 2:
        eE = aE[:, None] + sE[:, None] * d
        eT = aT[:, None] + sT[:, None] * d
    else:
        eE, eT = aE + sE * d, aT + sT * d
    return np.clip(ndtr(eE), CLIP, 1 - CLIP), np.clip(ndtr(eT), CLIP, 1 - CLIP)


# dependence specifications: ("fam", name, theta) or ("cond", g0, g1)
def p11_spec(spec, pE, pT, d):
    if spec[0] == "fam":
        return K_family(spec[1], pE, pT, spec[2])
    if spec[0] == "cond":
        a = np.tanh(spec[1] + spec[2] * d)
        from scipy.special import ndtri
        return bvn_cdf(ndtri(pE), ndtri(pT), a)
    raise ValueError(spec)


def U(spec, d, w1, w2):
    pE, pT = margins(d)
    return w1 + (1 - w1) * pE - w1 * pT + (w1 + w2 - 1) * p11_spec(spec, pE, pT, d)


def golden(f, lo, hi, iters=48):
    g = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(iters):
        c = hi - g * (hi - lo)
        e = lo + g * (hi - lo)
        m = f(c) >= f(e)
        hi = np.where(m, e, hi)
        lo = np.where(m, lo, c)
    return 0.5 * (lo + hi)


def argmax_dose(spec, w1, w2):
    Dg = np.broadcast_to(DG, (NP, DG.size))
    j = np.argmax(U(spec, Dg, w1, w2), axis=1)
    d0 = DG[j]
    return golden(lambda d: U(spec, d, w1, w2), np.clip(d0 - STEP, -1, 1),
                  np.clip(d0 + STEP, -1, 1))


def best_fixed(spec, w1, w2):
    """Best single dose and its mean expected utility."""
    Dg = np.broadcast_to(DG, (NP, DG.size))
    m = U(spec, Dg, w1, w2).mean(axis=0)
    j = int(np.argmax(m))
    lo, hi = max(-1, DG[j] - STEP), min(1, DG[j] + STEP)
    r = minimize_scalar(lambda x: -U(spec, np.full(NP, x), w1, w2).mean(),
                        bounds=(lo, hi), method="bounded", options=dict(xatol=1e-10))
    return (float(r.x), -r.fun) if -r.fun >= m[j] else (float(DG[j]), m[j])


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


def true_cells(spec):
    p11 = p11_spec(spec, pEd, pTd, dd)
    return 1 - pEd - pTd + p11, pEd - p11, pTd - p11, p11


def pseudo_true(fam, truth_cells):
    """argmax_theta E[sum_c p_c log q_c(theta)] for family fam."""
    c00, c10, c01, c11 = truth_cells
    fwd, _ = PAR[fam]
    rng_t = {"gauss": (-3.8, 3.8), "plackett": (-6, 8), "frank": (-30, 40),
             "fgm": (-6, 6), "clayton": (-6, np.log(30)), "sclayton": (-6, np.log(30)),
             "gumbel": (-8, np.log(30)), "sgumbel": (-8, np.log(30))}[fam]

    def nell(t):
        q11 = K_family(fam, pEd, pTd, fwd(t))
        q10, q01 = pEd - q11, pTd - q11
        q00 = 1 - pEd - pTd + q11
        lg = lambda x: np.log(np.maximum(x, 1e-300))
        return -np.mean(c00 * lg(q00) + c10 * lg(q10) + c01 * lg(q01) + c11 * lg(q11))
    # coarse scan then refine (likelihood can be flat near boundaries)
    ts = np.linspace(*rng_t, 81)
    vals = [nell(t) for t in ts]
    i = int(np.argmin(vals))
    lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
    r = minimize_scalar(nell, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-7))
    return float(fwd(r.x)), float(r.fun)


def solve_tau(fam, tau, bracket):
    return brentq(lambda th: kendall_tau(fam, th) - tau, *bracket, xtol=1e-6)


UTILS = [("TFE", 0.0, 0.0), ("k-0.6", 0.2, 0.2), ("U1", 0.3, 0.5), ("U3", 0.5, 0.3),
         ("k+0.6", 0.8, 0.8), ("NFT", 1.0, 1.0)]

if __name__ == "__main__":
    meta = {}
    tau0 = kendall_tau("gauss", 0.8)
    th_clay = 2 * tau0 / (1 - tau0)
    th_plack = solve_tau("plackett", tau0, (1.5, 200.0))
    th_frank = solve_tau("frank", tau0, (0.5, 60.0))
    meta["tau0"] = tau0
    meta["matched_params"] = dict(clayton=th_clay, plackett=th_plack, frank=th_frank)
    truths = {
        "Gaussian": ("fam", "gauss", 0.8),
        "Plackett": ("fam", "plackett", th_plack),
        "Clayton": ("fam", "clayton", th_clay),
        "Survival Clayton": ("fam", "sclayton", th_clay),
    }
    fitted = ["indep", "fgm", "gauss", "plackett", "frank", "clayton", "sclayton",
              "gumbel", "sgumbel"]
    rows = []
    for tname, tspec in truths.items():
        cells = true_cells(tspec)
        oracle = {}
        for lab, w1, w2 in UTILS:
            d_or = argmax_dose(tspec, w1, w2)
            V_or = U(tspec, d_or, w1, w2)
            d_fix, _ = best_fixed(tspec, w1, w2)
            oracle[lab] = (V_or, U(tspec, np.full(NP, d_fix), w1, w2))
        for fam in fitted:
            if fam == "indep":
                th, nell = 1.0, None
                spec = ("fam", "indep", 0.0)
            else:
                th, nell = pseudo_true(fam, cells)
                spec = ("fam", fam, th)
            tau_f = 0.0 if fam == "indep" else float(kendall_tau(fam, th))
            for lab, w1, w2 in UTILS:
                d_f = argmax_dose(spec, w1, w2)
                V_f = U(tspec, d_f, w1, w2)
                V_or, V_fix = oracle[lab]
                r_, g_ = V_or - V_f, V_or - V_fix
                rows.append(dict(truth=tname, fitted=fam, theta=th, tau_fitted=tau_f,
                                 utility=lab, w1=w1, w2=w2, kappa=w1 + w2 - 1,
                                 regret=r_.mean(), gain=g_.mean(),
                                 rel_regret=r_.mean() / g_.mean(),
                                 se_rel_regret=boot_se_ratio(r_, g_)))
            print(tname, fam, round(th, 4), round(tau_f, 4), flush=True)
    pd.DataFrame(rows).to_csv("../results/misspec_family.csv", index=False)

    # dose-dependent latent correlation ------------------------------------
    crow = []
    for sname, g in {"increasing": (0.75, 0.75), "decreasing": (0.75, -0.75)}.items():
        tspec = ("cond", g[0], g[1])
        cells = true_cells(tspec)
        abar, _ = pseudo_true("gauss", cells)
        a_lo, a_hi = np.tanh(g[0] - abs(g[1])), np.tanh(g[0] + abs(g[1]))
        meta[f"cond_{sname}"] = dict(g0=g[0], g1=g[1], alpha_range=[float(a_lo), float(a_hi)],
                                     tau_range=[float(kendall_tau("gauss", a_lo)),
                                                float(kendall_tau("gauss", a_hi))],
                                     alpha_bar=abar)
        for lab, w1, w2 in UTILS:
            d_or = argmax_dose(tspec, w1, w2)
            V_or = U(tspec, d_or, w1, w2)
            d_fix, _ = best_fixed(tspec, w1, w2)
            V_fix = U(tspec, np.full(NP, d_fix), w1, w2)
            for fname, spec in {"indep": ("fam", "indep", 0.0),
                                "gauss_const": ("fam", "gauss", abar)}.items():
                d_f = argmax_dose(spec, w1, w2)
                V_f = U(tspec, d_f, w1, w2)
                k = w1 + w2 - 1
                r_, g_ = V_or - V_f, V_or - V_fix
                crow.append(dict(scenario=sname, fitted=fname, utility=lab, kappa=k,
                                 regret=r_.mean(), gain=g_.mean(),
                                 rel_regret=r_.mean() / g_.mean(),
                                 se_rel_regret=boot_se_ratio(r_, g_),
                                 share_dchange_01=float(np.mean(np.abs(d_f - d_or) >= 0.1)),
                                 bound=abs(k) * (kendall_tau("gauss", a_hi)
                                                 - min(0.0, kendall_tau("gauss", a_lo))) / 4))
        print("conditional", sname, "done", flush=True)
    pd.DataFrame(crow).to_csv("../results/misspec_conditional.csv", index=False)
    with open("../results/misspec_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
