"""
verify_props.py -- numerical checks of Propositions 1(d), 2 and 6 of the article.
  * sup |Delta_alpha| over (pE, pT) equals tau_alpha/4 and is attained at (1/2, 1/2);
  * per patient, regret <= kappa {D(d_C) - D(d_C')} <= |kappa| osc D (independence rule);
  * ordinal outcomes: E[U] = A(margins) + sum_kl kappa_kl C(F_E(k), F_T(l)) on random tables.
"""
import numpy as np
from scipy.special import ndtr, ndtri
from core import bvn_cdf, kendall_tau, sample_x, intercept_slope, BETA_E, BETA_T

g = np.linspace(0.005, 0.995, 397)
U, V = np.meshgrid(g, g)
for a in (0.3, 0.8, -0.8, 0.95):
    D = bvn_cdf(ndtri(U), ndtri(V), a) - U * V
    i = np.unravel_index(np.argmax(np.abs(D)), D.shape)
    print(f"alpha={a:5}: max|Delta|={np.abs(D).max():.6f} at ({U[i]:.3f},{V[i]:.3f}); "
          f"tau/4={abs(kendall_tau('gauss', a)) / 4:.6f}")

rng = np.random.default_rng(5)
X = sample_x(3000, rng)
aE, sE = intercept_slope(X, BETA_E)
aT, sT = intercept_slope(X, BETA_T)
d = np.linspace(-1, 1, 2001)
eE, eT = aE[:, None] + sE[:, None] * d, aT[:, None] + sT[:, None] * d
pE, pT = ndtr(eE), ndtr(eT)
p11 = bvn_cdf(eE, eT, 0.8)
Dl = p11 - pE * pT
rows = np.arange(len(X))
violations = 0
for w1, w2 in ((0, 0), (0.3, 0.5), (0.8, 0.8), (1, 1), (0.1, 0.9), (0.9, 0.6)):
    k = w1 + w2 - 1
    Ut = w1 + (1 - w1) * pE - w1 * pT + k * p11
    Ui = w1 + (1 - w1) * pE - w1 * pT + k * pE * pT
    jt, ji = Ut.argmax(1), Ui.argmax(1)
    r = Ut.max(1) - Ut[rows, ji]
    b1 = k * (Dl[rows, jt] - Dl[rows, ji])
    b2 = abs(k) * (Dl.max(1) - Dl.min(1))
    violations += int(np.sum(r > b1 + 1e-12) + np.sum(b1 > b2 + 1e-12))
    print(f"(w1,w2)=({w1},{w2}): max regret {r.max():.5f}, max bound {b2.max():.5f}")
print("violations of Proposition 2:", violations)


# ---------------------------------------------------------------------------
# Proposition 6 (ordinal outcomes): E[U] = A(margins) + sum_{k<K, l<L} kappa_kl C(F_E(k), F_T(l))
# ---------------------------------------------------------------------------
from core import K_family

rng = np.random.default_rng(11)
worst = 0.0
for rep in range(200):
    K, L = rng.integers(2, 6), rng.integers(2, 6)
    Umat = rng.normal(size=(K, L))
    pe = rng.dirichlet(np.ones(K))
    pt = rng.dirichlet(np.ones(L))
    FE, FT = np.cumsum(pe), np.cumsum(pt)
    FE[-1] = FT[-1] = 1.0
    fam, th = [("gauss", rng.uniform(-0.9, 0.9)), ("clayton", rng.uniform(0.2, 5)),
               ("frank", rng.uniform(-8, 8)), ("plackett", np.exp(rng.uniform(-3, 3)))][rep % 4]
    H = np.zeros((K + 1, L + 1))
    for k in range(1, K + 1):
        for l in range(1, L + 1):
            u, v = min(FE[k - 1], 1 - 1e-15), min(FT[l - 1], 1 - 1e-15)
            H[k, l] = FE[k - 1] if l == L else (FT[l - 1] if k == K else float(K_family(fam, u, v, th)))
    P = H[1:, 1:] - H[:-1, 1:] - H[1:, :-1] + H[:-1, :-1]
    direct = np.sum(Umat * P)
    kap = Umat[:-1, :-1] - Umat[1:, :-1] - Umat[:-1, 1:] + Umat[1:, 1:]
    A = np.sum(Umat[-1, :] * pt) - np.sum((Umat[1:, -1] - Umat[:-1, -1]) * FE[:-1])
    formula = A + np.sum(kap * H[1:K, 1:L])
    worst = max(worst, abs(direct - formula))
print("Proposition 6 identity, max abs discrepancy over 200 random tables:", worst)
