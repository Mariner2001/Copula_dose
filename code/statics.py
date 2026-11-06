"""
statics.py -- comparative statics of the optimal dose in the latent correlation alpha
(Gaussian copula, Scenario 1 margins).  Verifies

   d d*/d alpha = - kappa * phi2(zE*, zT*; alpha) * Q'(d*) / (2 (1 - alpha^2) h),
   Q(d) = zE(d)^2 - 2 alpha zE(d) zT(d) + zT(d)^2,   h = -U''(d*),

and stores curves d*(alpha), d_c(alpha) = argmin_d Q(d) for illustration.
Outputs: results/statics.json, results/statics_curves.npz
"""
import json
import numpy as np
from scipy.special import ndtr
from core import bvn_cdf, bvn_pdf, sample_x, intercept_slope, BETA_E, BETA_T

rng = np.random.default_rng(99)
N = 4000
X = sample_x(N, rng)
aE, sE = intercept_slope(X, BETA_E)
aT, sT = intercept_slope(X, BETA_T)
DG = np.linspace(-1, 1, 401)
STEP = DG[1] - DG[0]


def U(d, alpha, w1, w2, idx):
    zE, zT = aE[idx] + sE[idx] * d, aT[idx] + sT[idx] * d
    return w1 + (1 - w1) * ndtr(zE) - w1 * ndtr(zT) + (w1 + w2 - 1) * bvn_cdf(zE, zT, alpha)


def dstar(alpha, w1, w2, idx):
    Ug = U(DG[None, :], alpha, w1, w2, idx[:, None])
    j = np.argmax(Ug, axis=1)
    lo, hi = np.clip(DG[j] - STEP, -1, 1), np.clip(DG[j] + STEP, -1, 1)
    g = (np.sqrt(5) - 1) / 2
    for _ in range(60):
        c, e = hi - g * (hi - lo), lo + g * (hi - lo)
        m = U(c, alpha, w1, w2, idx) >= U(e, alpha, w1, w2, idx)
        hi, lo = np.where(m, e, hi), np.where(m, lo, c)
    return 0.5 * (lo + hi)


def dc(alpha, idx):
    num = (aE[idx] - alpha * aT[idx]) * sE[idx] + (aT[idx] - alpha * aE[idx]) * sT[idx]
    den = sE[idx] ** 2 - 2 * alpha * sE[idx] * sT[idx] + sT[idx] ** 2
    return -num / den


def formula(alpha, w1, w2, idx, d):
    k = w1 + w2 - 1
    zE, zT = aE[idx] + sE[idx] * d, aT[idx] + sT[idx] * d
    Qp = 2 * ((zE - alpha * zT) * sE[idx] + (zT - alpha * zE) * sT[idx])
    eps = 1e-4
    h = -(U(d + eps, alpha, w1, w2, idx) - 2 * U(d, alpha, w1, w2, idx)
          + U(d - eps, alpha, w1, w2, idx)) / eps ** 2
    with np.errstate(divide="ignore", invalid="ignore"):   # h = 0 on flat (degenerate) optima
        return -k * bvn_pdf(zE, zT, alpha) * Qp / (2 * (1 - alpha ** 2) * h), h


out = {}
allidx = np.arange(N)
for lab, w1, w2 in (("TFE", 0.0, 0.0), ("k-0.6", 0.2, 0.2), ("k+0.6", 0.8, 0.8), ("NFT", 1.0, 1.0)):
    a0 = 0.8
    d0 = dstar(a0, w1, w2, allidx)
    interior = (d0 > -0.97) & (d0 < 0.97)
    idx = allidx[interior]
    dA = (dstar(a0 + 1e-4, w1, w2, idx) - dstar(a0 - 1e-4, w1, w2, idx)) / 2e-4
    fA, h = formula(a0, w1, w2, idx, d0[interior])
    ok = h > 1e-3                      # non-degenerate interior optima
    if ok.sum() == 0:
        out[lab] = dict(n_interior=int(interior.sum()), n_nondegenerate=0)
        continue
    rel = np.abs(dA[ok] - fA[ok]) / np.maximum(np.abs(dA[ok]), 1e-3)
    # sign rule: kappa > 0 attracts towards d_c, kappa < 0 repels
    toward = np.sign(fA[ok]) == np.sign(dc(a0, idx[ok]) - d0[interior][ok])
    out[lab] = dict(n_interior=int(interior.sum()), n_nondegenerate=int(ok.sum()),
                    share_interior=float(interior.mean()),
                    median_rel_err=float(np.median(rel)), q95_rel_err=float(np.quantile(rel, 0.95)),
                    max_abs_err=float(np.max(np.abs(dA[ok] - fA[ok]))),
                    share_moving_toward_dc=float(np.mean(toward)))

# curves for three illustrative patients: interior optimum under TFE and NFT at alpha=0.8
dm, dp = dstar(0.8, 0.2, 0.2, allidx), dstar(0.8, 0.8, 0.8, allidx)
_, hm = formula(0.8, 0.2, 0.2, allidx, dm)
_, hp = formula(0.8, 0.8, 0.8, allidx, dp)
cand = allidx[(np.abs(dm) < 0.6) & (np.abs(dp) < 0.6) & (hm > 0.05) & (hp > 0.05)]
order = cand[np.argsort(dc(0.8, cand) - dm[cand])]
pick = order[[int(0.1 * len(order)), len(order) // 2, int(0.9 * len(order))]]
alphas = np.round(np.arange(-0.9, 0.951, 0.01), 4)
curves = dict(alphas=alphas, pick=pick)
for lab, w1, w2 in (("k-0.6", 0.2, 0.2), ("k+0.6", 0.8, 0.8)):
    curves[lab] = np.array([dstar(a, w1, w2, pick) for a in alphas])
curves["dc"] = np.array([dc(a, pick) for a in alphas])
curves["coef"] = np.column_stack([aE[pick], sE[pick], aT[pick], sT[pick]])
np.savez("../results/statics_curves.npz", **{k.replace("+", "p").replace("-", "m"): v for k, v in curves.items()})
out["n_candidates_both_interior"] = int(len(cand))
out["picked"] = [dict(aE=float(aE[i]), sE=float(sE[i]), aT=float(aT[i]), sT=float(sT[i])) for i in pick]
with open("../results/statics.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
