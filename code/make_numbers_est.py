# executed from make_numbers.py (shares its namespace: M, mac, fmt, RES, PAP, np, pd, json)
UT = ["TFE", "k-0.6", "U1", "U3", "U2", "k+0.6", "NFT"]
UTL = ["TFE", r"$\kappa=-0.6$", "U1", "U3", "U2", r"$\kappa=+0.6$", "NFT"]

# ---------------------------------------------------------------- Part A ----
A = pd.read_csv(RES + "estimation_A.csv")
orA = json.load(open(RES + "estimation_A_oracle.json"))
R_A = len(A)
mac("nRepA", f"{R_A:,}".replace(",", "{,}"))
mac("nNval", "1{,}000")


def ip(df, m, u, orc):
    return 100 * np.mean((df[f"V|{m}|{u}"] - df[f"V|FD|{u}"]) / (orc[u] - df[f"V|FD|{u}"]))


def pdiff(df, m1, m0, u):
    x = 100 * (df[f"V|{m1}|{u}"] - df[f"V|{m0}|{u}"])
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


methods = [("PR-IND", "Probit, independence"), ("PR-GC", r"Probit, Gaussian $\hat\alpha$ (IFM)"),
           ("PR-GC*", r"Probit, Gaussian, true $\alpha$"), ("RF-IND", "Forest, independence"),
           ("RF-GC-naive", r"Forest, $\hat\alpha$ in-sample"), ("RF-GC-oob", r"Forest, $\hat\alpha$ out-of-bag"),
           ("RF-GC*", r"Forest, true $\alpha$")]
rows = [r"\multicolumn{8}{@{}l}{\textit{(a) Improvement over the estimated fixed dose (\%IP)}}\\"]
for m, ml in methods:
    rows.append(" & ".join([ml] + [fmt(ip(A, m, u, orA), 1) for u in UT]) + r"\\")
rows.append(r"\midrule")
rows.append(r"\multicolumn{8}{@{}l}{\textit{(b) Paired difference with the independence rule (utility points)}}\\")
diffs = [("PR-GC", "PR-IND", r"Probit: $\hat\alpha$ $-$ independence"),
         ("PR-GC*", "PR-IND", r"Probit: true $\alpha$ $-$ independence"),
         ("RF-GC-naive", "RF-IND", r"Forest: in-sample $\hat\alpha$ $-$ indep."),
         ("RF-GC-oob", "RF-IND", r"Forest: out-of-bag $\hat\alpha$ $-$ indep."),
         ("RF-GC*", "RF-IND", r"Forest: true $\alpha$ $-$ independence")]
maxse = 0.0
for m1, m0, lab_ in diffs:
    cells = [lab_]
    for u in UT:
        mu, se = pdiff(A, m1, m0, u)
        maxse = max(maxse, se)
        cells.append(fmt(mu, 3))
    rows.append(" & ".join(cells) + r"\\")
tab = r"""\small
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Rule & """ + " & ".join(UTL) + r"""\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
"""
open(PAP + "tab_estA.tex", "w").write(tab)
mac("nMaxSEA", fmt(np.ceil(1000 * maxse) / 1000, 3))

for col, nm in (("alpha_probit", "Pr"), ("alpha_rf_naive", "Naive"), ("alpha_rf_oob", "Oob")):
    mac(f"nAlpha{nm}Mean", fmt(A[col].mean(), 2))
    mac(f"nAlpha{nm}SD", fmt(A[col].std(ddof=1), 2))
    mac(f"nAlpha{nm}Med", fmt(A[col].median(), 2))
    mac(f"nAlpha{nm}Top", fmt(100 * np.mean(A[col] >= 0.985), 0))
for m, nm in (("RF-IND", "RfInd"), ("PR-IND", "PrInd"), ("PR-GC", "PrGc"), ("RF-GC-oob", "RfOob"),
              ("RF-GC-naive", "RfNaive")):
    for u, un in (("U1", "UOne"), ("U3", "UThree"), ("TFE", "TFE"), ("NFT", "NFT"), ("k+0.6", "KPS"),
                  ("k-0.6", "KMS")):
        mac(f"nIP{nm}{un}", fmt(ip(A, m, u, orA), 1))
for (m1, m0, nm) in (("PR-GC", "PR-IND", "Pr"), ("PR-GC*", "PR-IND", "PrStar"),
                     ("RF-GC-naive", "RF-IND", "RfNaive"), ("RF-GC-oob", "RF-IND", "RfOob"),
                     ("RF-GC*", "RF-IND", "RfStar")):
    for u, un in (("U1", "UOne"), ("U3", "UThree"), ("TFE", "TFE"), ("NFT", "NFT"), ("k+0.6", "KPS"),
                  ("k-0.6", "KMS")):
        mu, se = pdiff(A, m1, m0, u)
        mac(f"nD{nm}{un}", fmt(mu, 3))
        mac(f"nDse{nm}{un}", fmt(se, 3))

# ---------------------------------------------------------------- Part B ----
B = pd.read_csv(RES + "estimation_B.csv")
orB = json.load(open(RES + "estimation_B_oracle.json"))
R_B = int(B.groupby(["truth", "n"]).size().min())
mac("nRepB", f"{R_B:,}".replace(",", "{,}"))
rows = []
maxse = 0.0
for truth, tl in (("constant", r"constant $\alpha=0.8$"), ("increasing", r"$\alpha(d)=\tanh(0.75+0.75d)$")):
    for n in (200, 500):
        g = B[(B.truth == truth) & (B.n == n)]
        rej = 100 * np.mean(g.LR > 3.841459)
        cells = [tl if n == 200 else "", str(n), fmt(rej, 1),
                 fmt(g.alpha_const.mean(), 2) + " (" + fmt(g.alpha_const.std(ddof=1), 2) + ")"]
        for u in ("TFE", "U1", "NFT"):
            mu, se = pdiff(g, "PR-GC", "PR-IND", u)
            maxse = max(maxse, se)
            cells.append(fmt(mu, 3))
        for u in ("TFE", "NFT"):
            mu, se = pdiff(g, "PR-GCd", "PR-GC", u)
            maxse = max(maxse, se)
            cells.append(fmt(mu, 3))
        rows.append(" & ".join(cells) + r"\\")
        key = ("C" if truth == "constant" else "I") + ("S" if n == 200 else "L")
        mac(f"nRej{key}", fmt(rej, 1))
        mac(f"nAlphaB{key}", fmt(g.alpha_const.mean(), 2))
        mac(f"nAlphaBsd{key}", fmt(g.alpha_const.std(ddof=1), 2))
        for u, un in (("TFE", "TFE"), ("U1", "UOne"), ("NFT", "NFT"), ("k+0.6", "KPS")):
            mac(f"nDB{key}{un}", fmt(pdiff(g, "PR-GC", "PR-IND", u)[0], 2))
            mac(f"nDdB{key}{un}", fmt(pdiff(g, "PR-GCd", "PR-GC", u)[0], 3))
            mac(f"nIPB{key}Ind{un}", fmt(ip(g, "PR-IND", u, orB[truth]), 0))
            mac(f"nIPB{key}Gc{un}", fmt(ip(g, "PR-GC", u, orB[truth]), 0))
tab = r"""\small
\begin{tabular}{@{}llrrrrrrr@{}}
\toprule
& & LR test & & \multicolumn{3}{c}{$\Delta_1$} & \multicolumn{2}{c}{$\Delta_2$}\\
\cmidrule(lr){5-7}\cmidrule(l){8-9}
Truth & $N$ & reject (\%) & $\hat\alpha$: mean (SD) & TFE & U1 & NFT & TFE & NFT\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
"""
open(PAP + "tab_estB.tex", "w").write(tab)
mac("nMaxSEB", fmt(np.ceil(1000 * maxse) / 1000, 3))
