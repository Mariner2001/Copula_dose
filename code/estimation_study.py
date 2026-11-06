"""
estimation_study.py -- finite-sample study.

Part A (Scenario 1 of Li et al. 2022, Gaussian copula alpha = 0.8, N = 200):
   margins: correctly specified probit (12 coefficients each) or random forest
            (500 fully grown trees, sqrt(p) features per split; the analogue of M2/M3)
   copula : independence | Gaussian with alpha estimated by IFM
            (RF: alpha from in-sample predictions [naive] or out-of-bag predictions)
            | Gaussian with the true alpha (to isolate the cost of estimating alpha)
   reference rules: oracle (true model) and estimated fixed dose (multinomial
   logistic regression of the 4-category outcome on dose, as M12 in Li et al.)

Part B (dose-dependent latent correlation, probit margins):
   truth alpha(d) = tanh(0.75 + 0.75 d) (and constant alpha = 0.8 for test size);
   rules with independence, constant alpha, alpha(d) = tanh(g0 + g1 d);
   likelihood-ratio test of g1 = 0.

usage: python3 estimation_study.py A <R> <n_jobs>   |   python3 estimation_study.py B <R> <n_jobs>
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from core import (bvn_cdf, sample_x, intercept_slope, BETA_E, BETA_T, design,
                  simulate_training, fit_probit, ifm_alpha, ifm_alpha_dose, ifm_loglik)

UTILS = [("TFE", 0.0, 0.0), ("k-0.6", 0.2, 0.2), ("U1", 0.3, 0.5), ("U3", 0.5, 0.3),
         ("U2", 0.5, 0.5), ("k+0.6", 0.8, 0.8), ("NFT", 1.0, 1.0)]
DG = np.linspace(-1, 1, 41)
NVAL = 1000
CLIP_U = 1e-6

# fixed validation population ------------------------------------------------
_vrng = np.random.default_rng(424242)
XV = sample_x(NVAL, _vrng)
aEv, sEv = intercept_slope(XV, BETA_E)
aTv, sTv = intercept_slope(XV, BETA_T)
etaEv = aEv[:, None] + sEv[:, None] * DG
etaTv = aTv[:, None] + sTv[:, None] * DG
pEv, pTv = ndtr(etaEv), ndtr(etaTv)


def true_p11(alpha_fun=None, alpha=0.8):
    if alpha_fun is None:
        return bvn_cdf(etaEv, etaTv, alpha)
    return bvn_cdf(etaEv, etaTv, alpha_fun(DG)[None, :])


def utilities(pE, pT, p11, w1, w2):
    return w1 + (1 - w1) * pE - w1 * pT + (w1 + w2 - 1) * p11


def value(U_true, U_hat):
    j = np.argmax(U_hat, axis=1)
    return U_true[np.arange(U_true.shape[0]), j].mean()


def fixed_dose_values(X, d, E, T, U_true_by_util):
    """M12-like rule: multinomial logistic regression of (E,T) category on dose."""
    cat = E + 2 * T        # 0:(0,0) 1:(1,0) 2:(0,1) 3:(1,1)
    lr = LogisticRegression(C=1e6, max_iter=2000)
    lr.fit(d[:, None], cat)
    P = np.zeros((DG.size, 4))
    P[:, lr.classes_] = lr.predict_proba(DG[:, None])
    out = {}
    for lab, w1, w2 in UTILS:
        u = w1 * P[:, 0] + P[:, 1] + w2 * P[:, 3]
        j = int(np.argmax(u))
        out[lab] = U_true_by_util[lab][:, j].mean()
    return out


def rule_values(pE_hat, pT_hat, p11_hat, U_true_by_util):
    return {lab: value(U_true_by_util[lab], utilities(pE_hat, pT_hat, p11_hat, w1, w2))
            for lab, w1, w2 in UTILS}


# ------------------------------------------------------------------------------
# Part A
# ------------------------------------------------------------------------------
P11_TRUE_A = true_p11(alpha=0.8)
UT_A = {lab: utilities(pEv, pTv, P11_TRUE_A, w1, w2) for lab, w1, w2 in UTILS}
V_OR_A = {lab: UT_A[lab].max(axis=1).mean() for lab, _, _ in UTILS}


def one_rep_A(seed, n=200):
    rng = np.random.default_rng(seed)
    X, d, E, T = simulate_training(n, rng, alpha=0.8)
    rec = {"seed": seed}
    vals = {}
    vals["FD"] = fixed_dose_values(X, d, E, T, UT_A)

    # probit margins
    Z = design(X, d)
    bE, bT = fit_probit(Z, E), fit_probit(Z, T)
    k = X.shape[1]
    aE_h = bE[0] + XV @ bE[1:1 + k]
    sE_h = bE[1 + k] + XV @ bE[2 + k:]
    aT_h = bT[0] + XV @ bT[1:1 + k]
    sT_h = bT[1 + k] + XV @ bT[2 + k:]
    eE = aE_h[:, None] + sE_h[:, None] * DG
    eT = aT_h[:, None] + sT_h[:, None] * DG
    pE_h, pT_h = ndtr(eE), ndtr(eT)
    a_pr, _ = ifm_alpha(ndtr(Z @ bE), ndtr(Z @ bT), E, T)
    rec["alpha_probit"] = a_pr
    vals["PR-IND"] = rule_values(pE_h, pT_h, pE_h * pT_h, UT_A)
    vals["PR-GC"] = rule_values(pE_h, pT_h, bvn_cdf(eE, eT, a_pr), UT_A)
    vals["PR-GC*"] = rule_values(pE_h, pT_h, bvn_cdf(eE, eT, 0.8), UT_A)

    # random forest margins
    F = np.column_stack([X, d])
    FV = np.column_stack([np.repeat(XV, DG.size, axis=0), np.tile(DG, NVAL)])
    preds, insample, oob = [], [], []
    for y in (E, T):
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                    oob_score=True, random_state=int(seed % 2**31), n_jobs=1)
        rf.fit(F, y)
        preds.append(rf.predict_proba(FV)[:, 1].reshape(NVAL, DG.size))
        insample.append(rf.predict_proba(F)[:, 1])
        oob.append(rf.oob_decision_function_[:, 1])
    pE_r = np.clip(preds[0], CLIP_U, 1 - CLIP_U)
    pT_r = np.clip(preds[1], CLIP_U, 1 - CLIP_U)
    zE, zT = ndtri(pE_r), ndtri(pT_r)
    oobE = np.where(np.isnan(oob[0]), insample[0], oob[0])
    oobT = np.where(np.isnan(oob[1]), insample[1], oob[1])
    a_naive, _ = ifm_alpha(insample[0], insample[1], E, T, clip=0.01)
    a_oob, _ = ifm_alpha(oobE, oobT, E, T, clip=0.01)
    rec["alpha_rf_naive"] = a_naive
    rec["alpha_rf_oob"] = a_oob
    vals["RF-IND"] = rule_values(pE_r, pT_r, pE_r * pT_r, UT_A)
    vals["RF-GC-naive"] = rule_values(pE_r, pT_r, bvn_cdf(zE, zT, a_naive), UT_A)
    vals["RF-GC-oob"] = rule_values(pE_r, pT_r, bvn_cdf(zE, zT, a_oob), UT_A)
    vals["RF-GC*"] = rule_values(pE_r, pT_r, bvn_cdf(zE, zT, 0.8), UT_A)

    for m, dct in vals.items():
        for lab, v in dct.items():
            rec[f"V|{m}|{lab}"] = v
    return rec


# ------------------------------------------------------------------------------
# Part B
# ------------------------------------------------------------------------------
ALPHA_INC = lambda d: np.tanh(0.75 + 0.75 * d)


def one_rep_B(seed, n, truth):
    rng = np.random.default_rng(seed)
    afun = ALPHA_INC if truth == "increasing" else None
    X, d, E, T = simulate_training(n, rng, alpha=0.8, alpha_fun=afun)
    UT = UT_B[truth]
    rec = {"seed": seed, "n": n, "truth": truth}
    vals = {"FD": fixed_dose_values(X, d, E, T, UT)}
    Z = design(X, d)
    bE, bT = fit_probit(Z, E), fit_probit(Z, T)
    k = X.shape[1]
    eE = (bE[0] + XV @ bE[1:1 + k])[:, None] + (bE[1 + k] + XV @ bE[2 + k:])[:, None] * DG
    eT = (bT[0] + XV @ bT[1:1 + k])[:, None] + (bT[1 + k] + XV @ bT[2 + k:])[:, None] * DG
    pE_h, pT_h = ndtr(eE), ndtr(eT)
    fE, fT = ndtr(Z @ bE), ndtr(Z @ bT)
    a_c, ll_c = ifm_alpha(fE, fT, E, T)
    g, ll_d = ifm_alpha_dose(fE, fT, E, T, d, start=(np.arctanh(np.clip(a_c, -0.95, 0.95)), 0.0))
    rec.update(alpha_const=a_c, g0=g[0], g1=g[1], LR=2 * (ll_d - ll_c))
    vals["PR-IND"] = rule_values(pE_h, pT_h, pE_h * pT_h, UT)
    vals["PR-GC"] = rule_values(pE_h, pT_h, bvn_cdf(eE, eT, a_c), UT)
    a_grid = np.clip(np.tanh(g[0] + g[1] * DG), -0.995, 0.995)[None, :]
    vals["PR-GCd"] = rule_values(pE_h, pT_h, bvn_cdf(eE, eT, a_grid), UT)
    for m, dct in vals.items():
        for lab, v in dct.items():
            rec[f"V|{m}|{lab}"] = v
    return rec


UT_B, V_OR_B = {}, {}
for _truth, _af in (("constant", None), ("increasing", ALPHA_INC)):
    _p11 = true_p11(alpha_fun=_af, alpha=0.8)
    UT_B[_truth] = {lab: utilities(pEv, pTv, _p11, w1, w2) for lab, w1, w2 in UTILS}
    V_OR_B[_truth] = {lab: UT_B[_truth][lab].max(axis=1).mean() for lab, _, _ in UTILS}


if __name__ == "__main__":
    part, R, nj = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    t0 = time.time()
    if part == "A":
        recs = Parallel(n_jobs=nj, verbose=0)(delayed(one_rep_A)(1000 + r) for r in range(R))
        df = pd.DataFrame(recs)
        df.to_csv("../results/estimation_A.csv", index=False)
        with open("../results/estimation_A_oracle.json", "w") as f:
            json.dump(V_OR_A, f, indent=2)
    else:
        jobs = [(5000 + r, n, truth) for truth in ("constant", "increasing")
                for n in (200, 500) for r in range(R)]
        recs = Parallel(n_jobs=nj, verbose=0)(delayed(one_rep_B)(*j) for j in jobs)
        pd.DataFrame(recs).to_csv("../results/estimation_B.csv", index=False)
        with open("../results/estimation_B_oracle.json", "w") as f:
            json.dump(V_OR_B, f, indent=2)
    print("done in", round(time.time() - t0, 1), "s")
