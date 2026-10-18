"""
extra_checks.py -- (i) monotonicity of scikit-learn monotone forests (bound propagation),
(ii) conditional and pooled association measures in Scenario 1 (alpha = 0.8).
Writes results/extra_checks.json
"""
import json
import numpy as np
from scipy.special import ndtr
from sklearn.ensemble import RandomForestClassifier
from core import bvn_cdf, sample_x, intercept_slope, simulate_training, BETA_E, BETA_T

out = {}
# (i) ---------------------------------------------------------------------------
rng = np.random.default_rng(31)
XV = sample_x(500, rng)
DG = np.linspace(-1, 1, 41)
FV = np.column_stack([np.repeat(XV, DG.size, axis=0), np.tile(DG, len(XV))])
viol, curves = 0, 0
for r in range(30):
    X, d, E, T = simulate_training(200, np.random.default_rng(9000 + r), alpha=0.8)
    F = np.column_stack([X, d])
    for y in (E, T):
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=1, max_features="sqrt",
                                    monotonic_cst=[0, 0, 0, 0, 0, 1], random_state=r, n_jobs=2)
        rf.fit(F, y)
        P = rf.predict_proba(FV)[:, 1].reshape(len(XV), DG.size)
        viol += int(np.sum(np.diff(P, axis=1) < -1e-12))
        curves += len(XV)
out["sklearn_monotone_rf"] = dict(fits=60, curves=curves, decreasing_steps=viol)

# (ii) --------------------------------------------------------------------------
rng = np.random.default_rng(32)
X = sample_x(200_000, rng)
d = rng.uniform(-1, 1, len(X))
aE, sE = intercept_slope(X, BETA_E)
aT, sT = intercept_slope(X, BETA_T)
eE, eT = aE + sE * d, aT + sT * d
pE, pT = np.clip(ndtr(eE), 1e-12, 1 - 1e-12), np.clip(ndtr(eT), 1e-12, 1 - 1e-12)
p11 = bvn_cdf(eE, eT, 0.8)
p10, p01 = pE - p11, pT - p11
p00 = 1 - pE - pT + p11
phi = (p11 - pE * pT) / np.sqrt(pE * (1 - pE) * pT * (1 - pT))
ok = (p10 > 1e-9) & (p01 > 1e-9) & (p00 > 1e-9) & (p11 > 1e-9)
OR = p11[ok] * p00[ok] / (p10[ok] * p01[ok])
# pooled (marginal over d and x) association
PE, PT, P11 = pE.mean(), pT.mean(), p11.mean()
out["scenario1_conditional"] = dict(
    phi_median=float(np.median(phi)), phi_q10=float(np.quantile(phi, 0.1)),
    phi_q90=float(np.quantile(phi, 0.9)),
    OR_median=float(np.median(OR)), OR_q10=float(np.quantile(OR, 0.1)),
    OR_q90=float(np.quantile(OR, 0.9)))
out["scenario1_pooled"] = dict(pE=float(PE), pT=float(PT), p11=float(P11),
                               phi=float((P11 - PE * PT) / np.sqrt(PE * (1 - PE) * PT * (1 - PT))))
with open("../results/extra_checks.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
