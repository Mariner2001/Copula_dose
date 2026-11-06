"""
application.py -- what the published counts of the liver SBRT application imply
about the efficacy-toxicity dependence (Section 5 of Li et al., 2022).

Reported: n = 182; toxicity in 48; efficacy failures (progression < 1 y) 11;
E = 1 for 171 (123 progression-free + 39 deaths + 9 transplants without progression);
46 patients with both E = 1 and T = 1.  Hence the 2x2 table
            T=1   T=0
    E=1      46   125
    E=0       2     9
"""
import json
import numpy as np
from scipy.special import ndtri
from scipy.optimize import brentq, minimize
from scipy.stats import chi2
from scipy.stats.contingency import odds_ratio
from core import bvn_cdf, kendall_tau

import os
import csv
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sbrt_counts_li2022.csv")
_cnt = {}
with open(_path, newline="") as _fh:
    for _row in csv.DictReader(_fh):
        _cnt[(int(_row["E"]), int(_row["T"]))] = int(_row["count"])
n11, n10, n01, n00 = _cnt[(1, 1)], _cnt[(1, 0)], _cnt[(0, 1)], _cnt[(0, 0)]
n = n11 + n10 + n01 + n00
pE, pT, p11 = (n11 + n10) / n, (n11 + n01) / n, n11 / n
W, M = max(pE + pT - 1, 0.0), min(pE, pT)
out = dict(n=n, pE=pE, pT=pT, p11=p11, p11_indep=pE * pT, frechet=[W, M],
           frechet_width=M - W)

res = odds_ratio(np.array([[n11, n10], [n01, n00]]), kind="sample")
out["OR_sample"] = float(res.statistic)
resc = odds_ratio(np.array([[n11, n10], [n01, n00]]), kind="conditional")
ci = resc.confidence_interval(0.95)
out["OR_conditional_mle"] = float(resc.statistic)
out["OR_exact_CI"] = [float(ci.low), float(ci.high)]


def loglik(par):
    a, b, al = par
    qE, qT = 1 / (1 + np.exp(-a)), 1 / (1 + np.exp(-b))
    q11 = bvn_cdf(ndtri(qE), ndtri(qT), np.tanh(al))
    q10, q01 = qE - q11, qT - q11
    q00 = 1 - qE - qT + q11
    q = np.maximum(np.array([q11, q10, q01, q00]), 1e-300)
    return float(np.dot([n11, n10, n01, n00], np.log(q)))


zE, zT = ndtri(pE), ndtri(pT)
alpha_hat = brentq(lambda a: bvn_cdf(zE, zT, a) - p11, -0.99, 0.99)
lmax = loglik([np.log(pE / (1 - pE)), np.log(pT / (1 - pT)), np.arctanh(alpha_hat)])


def profile(al):
    r = minimize(lambda ab: -loglik([ab[0], ab[1], np.arctanh(al)]),
                 x0=[np.log(pE / (1 - pE)), np.log(pT / (1 - pT))], method="Nelder-Mead",
                 options=dict(xatol=1e-8, fatol=1e-10))
    return -r.fun


crit = chi2.ppf(0.95, 1) / 2
f = lambda al: lmax - profile(al) - crit
lo = brentq(f, -0.95, alpha_hat - 1e-6)
try:
    hi = brentq(f, alpha_hat + 1e-6, 0.995)
except ValueError:
    hi = 0.995
out["tetrachoric_alpha_hat"] = alpha_hat
out["tetrachoric_profile_CI"] = [lo, hi]
out["tau_hat"] = kendall_tau("gauss", alpha_hat)
out["max_utility_effect_by_kappa"] = {str(k): abs(k) * (M - W) for k in (-1, -0.2, 0, 0.2, 1)}
out["share_E1_death_or_transplant"] = (39 + 9) / 171
with open("../results/application.json", "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
