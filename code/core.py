"""
core.py -- shared building blocks for the study

  * bivariate normal cdf/pdf (vectorised, Owen's T representation)
  * copula families written as the copula K of the latent "success" variables,
    so that p11 = P(E=1, T=1) = K(pE, pT).  In the notation of Li et al. (2022),
    whose copula C acts on the "failure" orientation (p00 = C(1-pE, 1-pT)),
    K is the survival copula of C:  K(u, v) = u + v - 1 + C(1-u, 1-v).
  * Scenario 1 of Li et al. (2022): covariates, linear predictors, sampling
  * utilities, probit fitting and IFM estimation of the copula parameter
"""
import numpy as np
from scipy.special import owens_t, ndtr, ndtri, log_ndtr
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import quad

# ----------------------------------------------------------------------------
# Bivariate normal
# ----------------------------------------------------------------------------

def bvn_cdf(h, k, rho):
    """P(Z1 <= h, Z2 <= k) for a standard bivariate normal with correlation rho.

    Owen (1956): Phi2 = 0.5 Phi(h) + 0.5 Phi(k) - T(h, a_h) - T(k, a_k) - c,
    a_h = (k - rho h)/(h sqrt(1-rho^2)), a_k = (h - rho k)/(k sqrt(1-rho^2)),
    c = 1/2 if h k < 0 and 0 otherwise.  Exact zeros are jittered (the formula
    has a removable singularity there).
    """
    h, k, rho = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float),
                                    np.asarray(rho, float))
    h = np.where(h == 0.0, 1e-13, h)
    k = np.where(k == 0.0, 1e-13, k)
    s = np.sqrt(1.0 - rho * rho)
    ah = (k - rho * h) / (h * s)
    ak = (h - rho * k) / (k * s)
    c = np.where(h * k < 0.0, 0.5, 0.0)
    out = 0.5 * ndtr(h) + 0.5 * ndtr(k) - owens_t(h, ah) - owens_t(k, ak) - c
    return np.clip(out, 0.0, 1.0)


def bvn_pdf(h, k, rho):
    q = (h * h - 2.0 * rho * h * k + k * k) / (1.0 - rho * rho)
    return np.exp(-0.5 * q) / (2.0 * np.pi * np.sqrt(1.0 - rho * rho))


# ----------------------------------------------------------------------------
# Copula families (K = copula of the success variables; p11 = K(pE, pT))
# ----------------------------------------------------------------------------

def _clayton(u, v, th):
    """Clayton copula (theta > 0), evaluated on the log scale to avoid overflow."""
    a = -th * np.log(u)
    b = -th * np.log(v)
    m = np.maximum(a, b)
    L = m + np.log(np.exp(a - m) + np.exp(b - m) - np.exp(-m))
    return np.exp(-L / th)


def _gumbel(u, v, th):
    return np.exp(-((-np.log(u)) ** th + (-np.log(v)) ** th) ** (1.0 / th))


def K_family(fam, u, v, th):
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    if fam == "indep":
        return u * v
    if fam == "gauss":
        return bvn_cdf(ndtri(u), ndtri(v), th)
    if fam == "plackett":
        if abs(th - 1.0) < 1e-10:
            return u * v
        s = 1.0 + (th - 1.0) * (u + v)
        r = np.sqrt(np.maximum(s * s - 4.0 * th * (th - 1.0) * u * v, 0.0))
        # rationalised form avoids cancellation when s >= 0
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(s >= 0, 2.0 * th * u * v / (s + r), (s - r) / (2.0 * (th - 1.0)))
        return out
    if fam == "frank":
        if abs(th) < 1e-10:
            return u * v
        arg = np.expm1(-th * u) * np.expm1(-th * v) / np.expm1(-th)
        if th > 0:
            # when arg is close to -1 (large theta) use
            # C = -(1/th) [log(A + B - AB - c) - log(1 - c)], A = e^{-th u}, B = e^{-th v}, c = e^{-th}
            la, lb, lc = -th * u, -th * v, -th
            m = np.maximum(la, lb)
            S = np.exp(la - m) + np.exp(lb - m) - np.exp(la + lb - m) - np.exp(lc - m)
            with np.errstate(divide="ignore", invalid="ignore"):
                alt = -(m + np.log(S) - np.log1p(-np.exp(lc))) / th
                std = -np.log1p(np.maximum(arg, -1.0)) / th
            return np.where(arg > -0.5, std, alt)
        return -np.log1p(arg) / th
    if fam == "fgm":
        return u * v * (1.0 + th * (1.0 - u) * (1.0 - v))
    if fam == "clayton":
        return _clayton(u, v, th)
    if fam == "sclayton":            # survival Clayton
        return u + v - 1.0 + _clayton(1.0 - u, 1.0 - v, th)
    if fam == "gumbel":
        return _gumbel(u, v, th)
    if fam == "sgumbel":             # survival Gumbel
        return u + v - 1.0 + _gumbel(1.0 - u, 1.0 - v, th)
    raise ValueError(fam)


# parameter <-> unconstrained transforms used in optimisation
PAR = {
    "gauss":    (lambda t: np.tanh(t),           lambda th: np.arctanh(th)),
    "plackett": (lambda t: np.exp(t),            lambda th: np.log(th)),
    "frank":    (lambda t: t,                    lambda th: th),
    "fgm":      (lambda t: np.tanh(t),           lambda th: np.arctanh(th)),
    "clayton":  (lambda t: np.exp(t),            lambda th: np.log(th)),
    "sclayton": (lambda t: np.exp(t),            lambda th: np.log(th)),
    "gumbel":   (lambda t: 1.0 + np.exp(t),      lambda th: np.log(th - 1.0)),
    "sgumbel":  (lambda t: 1.0 + np.exp(t),      lambda th: np.log(th - 1.0)),
}

FAMILY_LABEL = {
    "indep": "Independence", "gauss": "Gaussian", "plackett": "Plackett",
    "frank": "Frank", "fgm": "FGM", "clayton": "Clayton",
    "sclayton": "Survival Clayton", "gumbel": "Gumbel",
    "sgumbel": "Survival Gumbel",
}


def debye1(x):
    if abs(x) < 1e-8:
        return 1.0
    val, _ = quad(lambda t: t / np.expm1(t) if t != 0 else 1.0, 0.0, x)
    return val / x


def kendall_tau(fam, th):
    """Kendall's tau of the copula (closed form where available)."""
    if fam == "indep":
        return 0.0
    if fam == "gauss":
        return 2.0 / np.pi * np.arcsin(th)
    if fam in ("clayton", "sclayton"):
        return th / (th + 2.0)
    if fam in ("gumbel", "sgumbel"):
        return 1.0 - 1.0 / th
    if fam == "fgm":
        return 2.0 * th / 9.0
    if fam == "frank":
        return 1.0 - 4.0 / th * (1.0 - debye1(th))
    return kendall_tau_numeric(fam, th)


def kendall_tau_numeric(fam, th, m=1200):
    """tau = 1 - 4 * int int dC/du dC/dv du dv (midpoint rule, central differences)."""
    g = (np.arange(m) + 0.5) / m
    U, V = np.meshgrid(g, g, indexing="ij")
    eps = 1e-6
    du = (K_family(fam, np.minimum(U + eps, 1 - 1e-12), V, th)
          - K_family(fam, np.maximum(U - eps, 1e-12), V, th)) / (2 * eps)
    dv = (K_family(fam, U, np.minimum(V + eps, 1 - 1e-12), th)
          - K_family(fam, U, np.maximum(V - eps, 1e-12), th)) / (2 * eps)
    return 1.0 - 4.0 * np.mean(du * dv)


def frechet_bounds(pE, pT):
    return np.maximum(pE + pT - 1.0, 0.0), np.minimum(pE, pT)


# ----------------------------------------------------------------------------
# Scenario 1 of Li et al. (2022)
# ----------------------------------------------------------------------------

BETA_E = dict(b0=0.0, bx=np.array([0.49, -1.11, 0.77, 1.51, 0.0]), bd=1.0,
              bdx=np.array([0.23, 0.61, 0.0, 1.69, 0.5]))
BETA_T = dict(b0=-1.386, bx=np.array([0.0, 1.14, -0.33, 0.0, 0.0]), bd=1.0,
              bdx=np.array([0.03, 0.6, 0.0, -0.42, 1.04]))


def _load_coefficients():
    """Read data/scenario1_coefficients.csv (if present) so that the data-generating
    model is documented in, and taken from, the repository's data folder."""
    import csv
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                        "scenario1_coefficients.csv")
    if not os.path.exists(path):
        return
    coef = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            coef[(row["outcome"], row["term"])] = float(row["coefficient"])
    for out, beta in (("E", BETA_E), ("T", BETA_T)):
        beta["b0"] = coef[(out, "intercept")]
        beta["bx"] = np.array([coef[(out, f"x{j}")] for j in range(1, 6)])
        beta["bd"] = coef[(out, "d")]
        beta["bdx"] = np.array([coef[(out, f"d:x{j}")] for j in range(1, 6)])


_load_coefficients()


def sample_x(n, rng, return_rate=False):
    """N(0, I_5) covariates subject to positive dose slopes for E and T."""
    keep, drawn, acc = [], 0, 0
    while acc < n:
        X = rng.standard_normal((2 * n + 200, 5))
        ok = (BETA_E["bd"] + X @ BETA_E["bdx"] > 0) & (BETA_T["bd"] + X @ BETA_T["bdx"] > 0)
        keep.append(X[ok])
        drawn += len(X)
        acc += ok.sum()
    X = np.vstack(keep)[:n]
    if return_rate:
        return X, acc / drawn
    return X


def intercept_slope(X, beta):
    """Linear predictor eta(d) = a + s d (probit scale)."""
    return beta["b0"] + X @ beta["bx"], beta["bd"] + X @ beta["bdx"]


def eta_grid(X, beta, dgrid):
    a, s = intercept_slope(X, beta)
    return a[:, None] + s[:, None] * dgrid[None, :]


def design(X, d):
    """Probit design used by the correctly specified parametric margins."""
    d = np.asarray(d, float)
    return np.column_stack([np.ones(len(X)), X, d, X * d[:, None]])


def simulate_training(n, rng, alpha=0.8, alpha_fun=None):
    """(X, d, E, T) with Gaussian-copula (latent probit) dependence.

    alpha_fun, if given, maps dose -> latent correlation (conditional copula)."""
    X = sample_x(n, rng)
    d = rng.uniform(-1.0, 1.0, n)
    aE, sE = intercept_slope(X, BETA_E)
    aT, sT = intercept_slope(X, BETA_T)
    etaE, etaT = aE + sE * d, aT + sT * d
    r = alpha_fun(d) if alpha_fun is not None else np.full(n, alpha)
    z1 = rng.standard_normal(n)
    z2 = r * z1 + np.sqrt(1.0 - r * r) * rng.standard_normal(n)
    E = (z1 <= etaE).astype(int)       # P(E=1) = Phi(etaE)
    T = (z2 <= etaT).astype(int)       # P(E=1,T=1) = Phi2(etaE, etaT; r)
    return X, d, E, T


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def kappa(w1, w2):
    return w1 + w2 - 1.0


def utility(pE, pT, p11, w1, w2):
    """Expected utility for the normalised matrix (U00, U10, U01, U11) = (w1, 1, 0, w2)."""
    return w1 + (1.0 - w1) * pE - w1 * pT + (w1 + w2 - 1.0) * p11


# ----------------------------------------------------------------------------
# Estimation helpers
# ----------------------------------------------------------------------------

def fit_probit(Z, y, lam=1e-3):
    """Probit MLE with a tiny ridge on non-intercept coefficients (stability)."""
    q = 2.0 * y - 1.0
    pen = np.full(Z.shape[1], lam)
    pen[0] = 0.0

    def f(b):
        eta = Z @ b
        ll = log_ndtr(q * eta)
        lam_ = q * np.exp(-0.5 * eta * eta - 0.5 * np.log(2 * np.pi) - ll)
        val = -ll.sum() + 0.5 * np.sum(pen * b * b)
        grad = -(Z.T @ lam_) + pen * b
        return val, grad

    b0 = np.zeros(Z.shape[1])
    res = minimize(f, b0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    return res.x


def _cell_probs(pE, pT, p11):
    p10 = pE - p11
    p01 = pT - p11
    p00 = 1.0 - pE - pT + p11
    return p00, p10, p01, p11


def ifm_loglik(alpha, pE, pT, E, T):
    zE, zT = ndtri(pE), ndtri(pT)
    p11 = bvn_cdf(zE, zT, alpha)
    p00, p10, p01, p11 = _cell_probs(pE, pT, p11)
    P = np.where(E == 1, np.where(T == 1, p11, p10), np.where(T == 1, p01, p00))
    return np.sum(np.log(np.maximum(P, 1e-300)))


def ifm_alpha(pE, pT, E, T, clip=0.005):
    """Second (copula) stage of IFM for a Gaussian copula with constant alpha."""
    pE = np.clip(pE, clip, 1 - clip)
    pT = np.clip(pT, clip, 1 - clip)
    res = minimize_scalar(lambda a: -ifm_loglik(a, pE, pT, E, T),
                          bounds=(-0.99, 0.99), method="bounded",
                          options=dict(xatol=1e-5))
    return res.x, -res.fun


def ifm_alpha_dose(pE, pT, E, T, d, clip=0.005, start=(0.5, 0.0)):
    """IFM with a dose-dependent latent correlation alpha(d) = tanh(g0 + g1 d)."""
    pE = np.clip(pE, clip, 1 - clip)
    pT = np.clip(pT, clip, 1 - clip)

    def nll(g):
        a = np.clip(np.tanh(g[0] + g[1] * d), -0.995, 0.995)
        return -ifm_loglik(a, pE, pT, E, T)

    best = None
    for s0 in (start, (0.0, 0.0), (1.0, 0.5)):
        res = minimize(nll, np.array(s0, float), method="Nelder-Mead",
                       options=dict(xatol=1e-5, fatol=1e-7, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, -best.fun
