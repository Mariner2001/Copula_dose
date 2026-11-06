"""
checks.py -- numerical verification of the formulas discussed in the paper.

Writes results/checks.json.
"""
import json
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import ndtr, ndtri
from core import (bvn_cdf, K_family, kendall_tau, kendall_tau_numeric, sample_x,
                  simulate_training, intercept_slope, BETA_E, BETA_T, frechet_bounds)

rng = np.random.default_rng(20261017)
out = {}

# 1. bivariate normal cdf against scipy -------------------------------------
h = rng.normal(0, 1.5, 400)
k = rng.normal(0, 1.5, 400)
r = rng.uniform(-0.95, 0.95, 400)
h[:20] = 0.0                      # exact zeros
k[20:40] = 1e-9
ref = np.array([multivariate_normal(mean=[0, 0], cov=[[1, ri], [ri, 1]]).cdf([hi, ki])
                for hi, ki, ri in zip(h, k, r)])
out["bvn_max_abs_err_vs_scipy"] = float(np.max(np.abs(bvn_cdf(h, k, r) - ref)))

# 2. joint probabilities under the latent construction ---------------------
#    E = 1{U_E > 1-pE}, T = 1{U_T > 1-pT}, (U_E, U_T) ~ C (failure orientation).
#    Correct: p11 = pE + pT - 1 + C(1-pE, 1-pT).   Li et al., App. A: C - pE - pT + 1.
pE, pT, th = 0.65, 0.30, 3.0
n = 2_000_000
# Clayton(th) sample by conditional inversion (C is NOT radially symmetric)
u = rng.uniform(size=n)
w = rng.uniform(size=n)
v = ((w ** (-th / (1 + th)) - 1.0) * u ** (-th) + 1.0) ** (-1.0 / th)
E = (u > 1 - pE)
T = (v > 1 - pT)
p11_mc = float(np.mean(E & T))
C = K_family("clayton", 1 - pE, 1 - pT, th)
out["latent_check"] = dict(
    pE=pE, pT=pT, clayton_theta=th,
    p11_monte_carlo=p11_mc,
    p11_correct=float(pE + pT - 1 + C),
    p11_li_et_al_appendixA=float(C - pE - pT + 1),
    p00_monte_carlo=float(np.mean(~E & ~T)), p00_formula=float(C),
    FE_values=sorted(set(np.round(np.where(E, 1.0, 1 - pE), 6).tolist())),
)

# 3. GP joint probabilities with distinct intercepts a != b -----------------
fE, fT, a, b, al = 0.2, -0.4, -0.7, 0.9, 0.5
P_E0 = ndtr(fE + a)                  # Pr(E = 0)
P_T0 = ndtr(fT + b)                  # Pr(T = 0)
p00 = float(bvn_cdf(fE + a, fT + b, al))
p01_true = P_E0 - p00                # Pr(E=0, T=1)
p11_true = 1 - P_E0 - P_T0 + p00
out["gp_check"] = dict(
    p01_correct=float(p01_true), p01_as_printed=float(ndtr(fE + b) - p00),
    p11_correct=float(p11_true), p11_as_printed=float(p00 - ndtr(fE + b) - ndtr(fT + b) + 1),
)

# 4. Kendall's tau: numeric routine vs closed forms --------------------------
out["tau_check"] = {
    "gauss_0.8": [float(kendall_tau("gauss", 0.8)), float(kendall_tau_numeric("gauss", 0.8))],
    "clayton_2": [float(kendall_tau("clayton", 2.0)), float(kendall_tau_numeric("clayton", 2.0))],
    "frank_5": [float(kendall_tau("frank", 5.0)), float(kendall_tau_numeric("frank", 5.0))],
}

# 5. Plackett copula <=> constant odds ratio ---------------------------------
uu = rng.uniform(0.05, 0.95, 1000)
vv = rng.uniform(0.05, 0.95, 1000)
thp = 7.3
p11 = K_family("plackett", uu, vv, thp)
OR = p11 * (1 - uu - vv + p11) / ((uu - p11) * (vv - p11))
out["plackett_OR_range"] = [float(OR.min()), float(OR.max())]

# 6. Scenario 1 descriptives -------------------------------------------------
X, rate = sample_x(400_000, rng, return_rate=True)
out["scenario1_acceptance_rate"] = float(rate)
Xt, d, E, T = simulate_training(400_000, rng, alpha=0.8)
out["scenario1_rates"] = dict(E=float(E.mean()), T=float(T.mean()), ET=float((E * T).mean()))
# 2x2 phi correlation in the training distribution
out["scenario1_phi_ET"] = float(np.corrcoef(E, T)[0, 1])

# 7. Figure 1 of Li et al.: is cor(E,T) = 0.8 attainable? ---------------------
gE, gT = np.meshgrid(np.linspace(0.12, 0.88, 300), np.linspace(0.03, 0.65, 300))
W, M = frechet_bounds(gE, gT)
phi_max = (M - gE * gT) / np.sqrt(gE * (1 - gE) * gT * (1 - gT))
out["fig1_share_region_phi0.8_feasible"] = float(np.mean(phi_max >= 0.8))
phi_min = (W - gE * gT) / np.sqrt(gE * (1 - gE) * gT * (1 - gT))
out["fig1_share_region_phi-0.8_feasible"] = float(np.mean(phi_min <= -0.8))
out["phi_max_examples"] = {
    "pE=0.6,pT=0.3": float((0.3 - 0.18) / np.sqrt(0.6 * 0.4 * 0.3 * 0.7)),
    "pE=0.8,pT=0.2": float((0.2 - 0.16) / np.sqrt(0.8 * 0.2 * 0.2 * 0.8)),
}
# bound on the utility change of Fig 1(F),(G): |kappa| * tau/4 with kappa=-0.2, alpha=0.8
out["fig1_FG_max_utility_change"] = float(0.2 * kendall_tau("gauss", 0.8) / 4)

with open("../results/checks.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
