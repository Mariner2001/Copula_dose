"""
make_numbers.py -- writes tables/numbers.tex (macros used in the text) and the
numeric tables tables/tab_*.tex from the result files, so that every number in
the article comes directly from the code.
"""
import json
import numpy as np
import pandas as pd
from core import kendall_tau

RES, PAP = "../results/", "../tables/"
M = {}


def fmt(x, nd):
    s = f"{x:.{nd}f}"
    return s.replace("-", "\\ensuremath{-}", 1) if s.startswith("-") else s


def mac(name, value):
    M[name] = value


# ---------------------------------------------------------------- checks ----
ck = json.load(open(RES + "checks.json"))
mac("nAccRate", fmt(100 * ck["scenario1_acceptance_rate"], 1))
mac("nExclRate", fmt(100 * (1 - ck["scenario1_acceptance_rate"]), 1))
mac("nRateE", fmt(100 * ck["scenario1_rates"]["E"], 1))
mac("nRateT", fmt(100 * ck["scenario1_rates"]["T"], 1))
mac("nFigFeas", fmt(100 * ck["fig1_share_region_phi0.8_feasible"], 0))
mac("nFigInfeas", fmt(100 * (1 - ck["fig1_share_region_phi0.8_feasible"]), 0))
mac("nFigBound", fmt(ck["fig1_FG_max_utility_change"], 3))
# largest difference between the surfaces with alpha = 0.8 and alpha = -0.8 (kappa = -0.2)
mac("nFigBoundFG", fmt(0.2 * (kendall_tau("gauss", 0.8) - kendall_tau("gauss", -0.8)) / 4, 3))
mac("nTauEight", fmt(kendall_tau("gauss", 0.8), 3))
lc = ck["latent_check"]
mac("nMCpEE", fmt(lc["p11_monte_carlo"], 4))
mac("nCorrpEE", fmt(lc["p11_correct"], 4))
mac("nPrintpEE", fmt(lc["p11_li_et_al_appendixA"], 4))
gp = ck["gp_check"]
mac("nGPpzoCorr", fmt(gp["p01_correct"], 3))
mac("nGPpzoPrint", fmt(gp["p01_as_printed"], 3))
mac("nGPpooCorr", fmt(gp["p11_correct"], 3))
mac("nGPpooPrint", fmt(gp["p11_as_printed"], 3))

ex = json.load(open(RES + "extra_checks.json"))
mac("nPhiPooled", fmt(ex["scenario1_pooled"]["phi"], 3))
mac("nPhiMed", fmt(ex["scenario1_conditional"]["phi_median"], 2))
mac("nORmed", fmt(ex["scenario1_conditional"]["OR_median"], 0))
mac("nORlo", fmt(ex["scenario1_conditional"]["OR_q10"], 0))
orhi = ex["scenario1_conditional"]["OR_q90"]
e = int(np.floor(np.log10(orhi)))
mac("nORhi", f"${orhi / 10 ** e:.1f}\\times10^{{{e}}}$")
mac("nMonoCurves", f"{ex['sklearn_monotone_rf']['curves']:,}".replace(",", "{,}"))
assert ex["sklearn_monotone_rf"]["decreasing_steps"] == 0

# ---------------------------------------------------------------- oracle ----
sel = pd.read_csv(RES + "oracle_selected.csv").set_index("label")
mac("nNpop", "20{,}000")
key = {"TFE": "TFE", "k-0.6": "KMS", "U1": "UOne", "U3": "UThree", "U2": "UTwo",
       "APP": "App", "k+0.6": "KPS", "NFT": "NFT"}
for lab, k in key.items():
    r = sel.loc[lab]
    mac(f"nG{k}", fmt(100 * r.gain, 2))
    mac(f"nR{k}", fmt(100 * r.regret_ind, 3))
    mac(f"nRel{k}", fmt(100 * r.regret_ind / r.gain, 2 if r.regret_ind / r.gain < 0.01 else 1))
    mac(f"nDch{k}", fmt(100 * r.share_dchange_01, 1))
    mac(f"nRobP{k}", fmt(100 * r.regret_pqd_robust / r.gain, 1))
    mac(f"nRobF{k}", fmt(100 * r.regret_frechet_robust / r.gain, 1))
mac("nDchUOne", fmt(np.ceil(100 * max(sel.loc["U1"].share_dchange_01, sel.loc["U3"].share_dchange_01)), 0))
mac("nQuadUOne", fmt(100 * sel.loc["U1"].regret_quad_pred, 3))
mac("nCornerNFT", fmt(100 * (1 - sel.loc["NFT"].share_interior_ind), 0))
sym = pd.read_csv(RES + "oracle_symmetric.csv")
small = sym[np.abs(sym.kappa) <= 0.2001]
ratio = small.regret_quad_pred / small.regret_ind
mac("nQuadMin", fmt(np.floor(100 * ratio.min()), 0))
mac("nQuadMax", fmt(np.ceil(100 * ratio.max()), 0))
mac("nQuadUOnePct", fmt(100 * sel.loc["U1"].regret_quad_pred / sel.loc["U1"].regret_ind, 0))
mac("nQuadTFEPct", fmt(100 * sel.loc["TFE"].regret_quad_pred / sel.loc["TFE"].regret_ind, 0))
mac("nQuadUThreePct", fmt(100 * sel.loc["U3"].regret_quad_pred / sel.loc["U3"].regret_ind, 0))
for lab_, k_ in (("U1", "UOne"), ("U3", "UThree")):
    mac(f"nQuadBoth{k_}", fmt(100 * sel.loc[lab_].quad_ratio_both_interior, 0))
    mac(f"nSwitch{k_}", fmt(100 * sel.loc[lab_].regret_share_switch, 0))
# Monte Carlo standard errors (patient bootstrap), percentage points of the gain
se_cols = ["se_rel_regret_ind", "se_rel_regret_pqd", "se_rel_regret_frechet"]
mac("nSEOracleMax", fmt(np.ceil(100 * sel[se_cols].max().max()) / 100, 2))
mac("nSEOracleTFE", fmt(sel.loc["TFE"].se_rel_regret_ind, 2))
r95 = sym[(sym.alpha == 0.95) & (np.isclose(sym.kappa, -1))].iloc[0]
mac("nRelTFEhigh", fmt(100 * r95.regret_ind / r95.gain, 1))

rows = []
names = {"TFE": "TFE", "k-0.6": r"$\kappa=-0.6$", "U1": "U1", "U3": "U3", "U2": "U2",
         "APP": "Application", "k+0.6": r"$\kappa=+0.6$", "NFT": "NFT"}
for lab in ["TFE", "k-0.6", "U1", "U3", "U2", "APP", "k+0.6", "NFT"]:
    r = sel.loc[lab]
    kap = r.kappa
    q = "--" if (np.isnan(r.regret_quad_pred) or kap == 0 or r.share_interior_ind < 0.1) \
        else fmt(100 * r.regret_quad_pred, 3)
    rows.append(" & ".join([
        names[lab], f"({r.w1:g}, {r.w2:g})", fmt(kap, 1).replace("0.0", "0"),
        fmt(100 * r.gain, 2), fmt(100 * r.regret_ind, 3),
        fmt(100 * r.regret_ind / r.gain, 2), q, fmt(100 * r.share_dchange_01, 1),
        fmt(100 * r.uniform_bound, 2), fmt(100 * r.regret_pqd_robust / r.gain, 2),
        fmt(100 * r.regret_frechet_robust / r.gain, 2)]) + r"\\")
tab = r"""\small
\begin{tabular}{@{}lcrrrrrrrrr@{}}
\toprule
& & & & & & & & & \multicolumn{2}{c}{robust $R/G$ (\%)}\\
\cmidrule(l){10-11}
Utility & $(\omega_1,\omega_2)$ & $\kappa$ & $G$ & $R_\Pi$ & $R_\Pi/G$ (\%) & 2nd order & $\Delta d$ (\%) & bound & PQD & all\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
"""
open(PAP + "tab_oracle.tex", "w").write(tab)

# ------------------------------------------------------------- families ----
fam = pd.read_csv(RES + "misspec_family.csv")
fam["rel"] = 100 * fam.rel_regret
truths = ["Gaussian", "Plackett", "Clayton", "Survival Clayton"]
fits = ["indep", "fgm", "gauss", "plackett", "frank", "clayton", "sclayton", "gumbel", "sgumbel"]
lab = {"indep": "Independence", "fgm": "FGM", "gauss": "Gaussian", "plackett": "Plackett",
       "frank": "Frank", "clayton": "Clayton", "sclayton": "Survival Clayton",
       "gumbel": "Gumbel", "sgumbel": "Survival Gumbel"}
truthkey = {"Gaussian": "gauss", "Plackett": "plackett", "Clayton": "clayton",
            "Survival Clayton": "sclayton"}
rows = []
for f_ in fits:
    tau = fam[(fam.truth == "Gaussian") & (fam.fitted == f_)].tau_fitted.iloc[0]
    cells = [lab[f_], fmt(tau, 2)]
    for t in truths:
        for u in ("TFE", "NFT"):
            v = fam[(fam.truth == t) & (fam.fitted == f_) & (fam.utility == u)].rel.iloc[0]
            cells.append(r"\textit{0}" if truthkey[t] == f_ else fmt(abs(v) if abs(v) < 5e-3 else v, 2))
    rows.append(" & ".join(cells) + r"\\")
tab = r"""\small
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
& & \multicolumn{8}{c}{True copula}\\
\cmidrule(l){3-10}
& & \multicolumn{2}{c}{Gaussian} & \multicolumn{2}{c}{Plackett} & \multicolumn{2}{c}{Clayton} & \multicolumn{2}{c}{Survival Clayton}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}
Working family & $\tau^*$ & TFE & NFT & TFE & NFT & TFE & NFT & TFE & NFT\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
"""
open(PAP + "tab_family.tex", "w").write(tab)
flex = ["gauss", "plackett", "frank", "clayton", "sclayton", "gumbel", "sgumbel"]
g_tfe = fam[(fam.truth == "Gaussian") & (fam.utility == "TFE")].set_index("fitted").rel
mac("nFamIndTFE", fmt(g_tfe["indep"], 1))
mac("nFamWorstFlexTFE", fmt(g_tfe[flex].max(), 1))
fgm_tfe = fam[(fam.fitted == "fgm") & (fam.utility == "TFE")].rel
mac("nFamFGMTFE", fmt(fgm_tfe.min(), 1))
mac("nFamFGMTFEmax", fmt(fgm_tfe.max(), 1))
mac("nFamOrientTFE", fmt(fam[(fam.truth == "Survival Clayton") & (fam.fitted == "clayton")
                             & (fam.utility == "TFE")].rel.iloc[0], 1))
u1 = fam[(fam.utility == "U1") & (fam.fitted.isin(flex))].rel.max()
mac("nFamUOneMax", fmt(np.ceil(100 * u1 + 1e-9) / 100, 2))
shown = fam[fam.utility.isin(["TFE", "NFT"])]
mac("nSEFamMax", fmt(np.ceil(100 * shown.se_rel_regret.max()) / 100, 2))

# ----------------------------------------------------------- conditional ----
cond = pd.read_csv(RES + "misspec_conditional.csv")
cond["rel"] = 100 * cond.rel_regret
cm = json.load(open(RES + "misspec_meta.json"))
mac("nCondAbarInc", fmt(cm["cond_increasing"]["alpha_bar"], 2))
mac("nCondAbarDec", fmt(cm["cond_decreasing"]["alpha_bar"], 2))
gv = lambda s, f_, u: cond[(cond.scenario == s) & (cond.fitted == f_) & (cond.utility == u)].rel.iloc[0]
mac("nCondIncIndTFE", fmt(gv("increasing", "indep", "TFE"), 1))
mac("nCondIncConstTFE", fmt(gv("increasing", "gauss_const", "TFE"), 1))
mac("nCondDecIndNFT", fmt(gv("decreasing", "indep", "NFT"), 1))
mac("nCondDecConstNFT", fmt(gv("decreasing", "gauss_const", "NFT"), 1))
mac("nSECondMax", fmt(np.ceil(100 * cond.se_rel_regret.max()) / 100, 2))
U = ["TFE", "k-0.6", "U1", "U3", "k+0.6", "NFT"]
rows = []
for s, sl in (("increasing", r"$\alpha(d)$ increasing"), ("decreasing", r"$\alpha(d)$ decreasing")):
    gains = [cond[(cond.scenario == s) & (cond.utility == u)].gain.iloc[0] for u in U]
    rows.append(r"\multicolumn{7}{@{}l}{\textit{" + sl + r"}}\\")
    for f_, fl in (("indep", "Independence"), ("gauss_const", r"Gaussian, constant $\bar\alpha$")):
        rows.append(" & ".join(["\\quad " + fl] + [fmt(gv(s, f_, u), 2) for u in U]) + r"\\")
    rows.append(" & ".join(["\\quad Gain $G$ (points)"] + [fmt(100 * g, 2) for g in gains]) + r"\\")
tab = r"""\small
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Working copula & TFE & $\kappa=-0.6$ & U1 & U3 & $\kappa=+0.6$ & NFT\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
"""
open(PAP + "tab_cond.tex", "w").write(tab)

# --------------------------------------------------------------- statics ----
st = json.load(open(RES + "statics.json"))
meds = [st[k]["median_rel_err"] for k in ("TFE", "k-0.6", "k+0.6")]
mac("nStatMedErr", fmt(np.ceil(1000 * 100 * max(meds)) / 1000, 2))
mac("nStatMaxAbs", fmt(max(st[k]["max_abs_err"] for k in ("TFE", "k-0.6", "k+0.6")), 3))
mac("nStatN", "4{,}000")
assert st["k+0.6"]["share_moving_toward_dc"] == 1.0 and st["TFE"]["share_moving_toward_dc"] == 0.0 \
    and st["k-0.6"]["share_moving_toward_dc"] == 0.0

fi = json.load(open(RES + "figure_info.json"))["patient"]
mac("nPatDtrue", fmt(fi["d_true"], 2))
mac("nPatDind", fmt(fi["d_ind"], 2))
mac("nPatRegret", fmt(fi["regret_pts"], 2))

# ----------------------------------------------------------- application ----
ap = json.load(open(RES + "application.json"))
mac("nAppOR", fmt(ap["OR_sample"], 2))
mac("nAppORlo", fmt(ap["OR_exact_CI"][0], 2))
mac("nAppORhi", fmt(ap["OR_exact_CI"][1], 1))
mac("nAppAlpha", fmt(ap["tetrachoric_alpha_hat"], 2))
mac("nAppAlphaLo", fmt(ap["tetrachoric_profile_CI"][0], 2))
mac("nAppAlphaHi", fmt(ap["tetrachoric_profile_CI"][1], 2))
mac("nAppWidthProb", fmt(ap["frechet_width"], 3))
mac("nAppWidthPts", fmt(100 * ap["frechet_width"], 1))
mac("nAppDeath", fmt(100 * ap["share_E1_death_or_transplant"], 0))
tab = r"""\small
\begin{tabular}{@{}lr@{}}
\toprule
Quantity & Value\\
\midrule
Patients analysed; $E=1$; $T=1$; $E=T=1$ & 182; 171; 48; 46\\
$\hat p_E$; $\hat p_T$ & """ + fmt(ap["pE"], 3) + "; " + fmt(ap["pT"], 3) + r"""\\
$\hat p_{11}$ (observed); $\hat p_E\hat p_T$ (independence) & """ + fmt(ap["p11"], 3) + "; " + fmt(ap["p11_indep"], 3) + r"""\\
Fr\'echet interval for $p_{11}$ given $(\hat p_E,\hat p_T)$ & $[""" + fmt(ap["frechet"][0], 3) + ", " + fmt(ap["frechet"][1], 3) + r"""]$\\
Odds ratio (exact 95\% CI) & """ + fmt(ap["OR_sample"], 2) + " (" + fmt(ap["OR_exact_CI"][0], 2) + ", " + fmt(ap["OR_exact_CI"][1], 1) + r""")\\
Tetrachoric correlation $\hat\alpha$ (profile 95\% CI) & """ + fmt(ap["tetrachoric_alpha_hat"], 2) + " (" + fmt(ap["tetrachoric_profile_CI"][0], 2) + ", " + fmt(ap["tetrachoric_profile_CI"][1], 2) + r""")\\
Implied Kendall's $\tau$ of the latent Gaussian copula & """ + fmt(ap["tau_hat"], 2) + r"""\\
Largest possible effect of the copula on $\bar U$, $|\kappa|(M-W)$, in points: & \\
\quad $\kappa=0$ (utility used by Li et al.); $|\kappa|=0.2$; $|\kappa|=1$ & 0; """ + fmt(100 * ap["max_utility_effect_by_kappa"]["0.2"], 1) + "; " + fmt(100 * ap["max_utility_effect_by_kappa"]["1"], 1) + r"""\\
Deaths or transplants among patients with $E=1$ & """ + fmt(100 * ap["share_E1_death_or_transplant"], 0) + r"""\%\\
\bottomrule
\end{tabular}
"""
open(PAP + "tab_app.tex", "w").write(tab)

# ----------------------------------------------------------- estimation ----
import os
if os.path.exists(RES + "estimation_A.csv"):
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_numbers_est.py")).read())

with open(PAP + "numbers.tex", "w") as fh:
    fh.write("% generated by code/make_numbers.py -- do not edit by hand\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros written")
