"""
figures.py -- all figures of the paper (PDF, vector).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.special import ndtr
from core import bvn_cdf, K_family, kendall_tau

OUT = "../figures/"
META = {"CreationDate": None}          # no time stamp, so that the PDF files are reproducible
RES = "../results/"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]          # ordinal blue ramp (validated)
CMAP = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#9ec5f4", "#5598e7",
                                                     "#256abf", "#104281", "#0d366b"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": BASE, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def panel_label(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=INK, va="bottom")


# --------------------------------------------------------------------------------
# Figure 1: anatomy of the copula effect for one patient
# --------------------------------------------------------------------------------
def fig_patient():
    cur = np.load(RES + "statics_curves.npz")
    coef = cur["coef"]
    d = np.linspace(-1, 1, 801)
    best = None
    for c in coef:
        aE, sE, aT, sT = c
        zE, zT = aE + sE * d, aT + sT * d
        pE, pT = ndtr(zE), ndtr(zT)
        Ut = pE - bvn_cdf(zE, zT, 0.8)
        Ui = pE - pE * pT
        shift = abs(d[np.argmax(Ut)] - d[np.argmax(Ui)])
        if best is None or shift > best[0]:
            best = (shift, c)
    aE, sE, aT, sT = best[1]
    zE, zT = aE + sE * d, aT + sT * d
    pE, pT = ndtr(zE), ndtr(zT)
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.15))
    ax = axs[0]
    ax.plot(d, pE, color=BLUE)
    ax.plot(d, pT, color=ORANGE)
    iE, iT = np.searchsorted(d, -0.55), np.searchsorted(d, 0.55)
    ax.text(d[iE], pE[iE] + 0.05, "$p_E(d)$", color=INK, ha="center", va="bottom")
    ax.text(d[iT] + 0.06, pT[iT] - 0.02, "$p_T(d)$", color=INK, ha="left", va="top")
    ax.set_xlabel("dose $d$")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("Marginal dose-response curves", loc="left")
    panel_label(ax, "A")
    ax = axs[1]
    for a, col in zip((0.3, 0.8, 0.95), (SEQ[0], SEQ[2], SEQ[3])):
        D = bvn_cdf(zE, zT, a) - pE * pT
        ax.plot(d, D, color=col, label=rf"$\alpha={a}$")
        ax.axhline(kendall_tau("gauss", a) / 4, color=col, lw=0.8, ls=(0, (3, 2)))
    ax.legend(loc="center left", fontsize=6.5, bbox_to_anchor=(0.0, 0.55))
    ax.set_xlabel("dose $d$")
    ax.set_ylabel(r"$\hat C_\alpha(p_E,p_T)-p_Ep_T$")
    ax.set_title(r"Dependence excess $\Delta_\alpha(d)$", loc="left")
    ax.set_ylim(0, 0.215)
    panel_label(ax, "B")
    ax = axs[2]
    Ut = pE - bvn_cdf(zE, zT, 0.8)
    Ui = pE - pE * pT
    jt, ji = np.argmax(Ut), np.argmax(Ui)
    ax.plot(d, Ut, color=BLUE, label=r"truth ($\alpha=0.8$)")
    ax.plot(d, Ui, color=ORANGE, ls=(0, (4, 2)), label="independence")
    ax.plot(d[jt], Ut[jt], "o", ms=5, color=BLUE, mec="white", mew=1.0)
    ax.plot(d[ji], Ut[ji], "s", ms=5, color=ORANGE, mec="white", mew=1.0)
    ax.plot(d[ji], Ui[ji], "s", ms=4, mfc="white", mec=ORANGE, mew=1.0)
    ax.annotate("", xy=(d[ji], Ut[ji]), xytext=(d[jt], Ut[jt]),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.set_xlabel("dose $d$")
    ax.set_ylabel(r"$\bar U(d)=\Pr(E=1,T=0)$")
    ax.set_title(r"Toxicity-free efficacy ($\kappa=-1$)", loc="left")
    ax.legend(loc="lower center")
    panel_label(ax, "C")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT + "fig_patient.pdf", metadata=META)
    plt.close(fig)
    return dict(aE=aE, sE=sE, aT=aT, sT=sT, d_true=float(d[jt]), d_ind=float(d[ji]),
                regret_pts=float(100 * (Ut[jt] - Ut[ji])))


# --------------------------------------------------------------------------------
# Figure 3: regret of ignoring dependence over the utility simplex
# --------------------------------------------------------------------------------
def fig_heatmap():
    h = pd.read_csv(RES + "oracle_heatmap.csv")
    g = np.sort(h.w1.unique())
    A = np.full((g.size, g.size), np.nan)
    Rr = np.full((g.size, g.size), np.nan)
    for _, r in h.iterrows():
        i, j = np.searchsorted(g, r.w2), np.searchsorted(g, r.w1)
        A[i, j] = 100 * r.regret_ind
        Rr[i, j] = 100 * r.regret_ind / r.gain if r.gain > 0.01 else np.nan
    step = g[1] - g[0]
    ext = (g[0] - step / 2, g[-1] + step / 2, g[0] - step / 2, g[-1] + step / 2)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.95))
    marks = [("U1", 0.3, 0.5), ("U2", 0.5, 0.5), ("U3", 0.5, 0.3), ("App.", 0.6, 0.4),
             ("TFE", 0.0, 0.0), ("NFT", 1.0, 1.0)]
    for ax, M, title, norm, lab in (
            (axs[0], np.where(A > 0, A, np.nan), "Regret of the independence rule\n(utility points, log scale)",
             LogNorm(vmin=1e-3, vmax=np.nanmax(A)), "points"),
            (axs[1], Rr, "Regret relative to the personalization gain\n(%; hatched: gain < 1 point)",
             None, "%")):
        ax.set_facecolor("#f0efec")
        im = ax.imshow(M, origin="lower", extent=ext, cmap=CMAP, norm=norm,
                       vmin=None if norm else 0, vmax=None if norm else np.nanmax(M),
                       interpolation="nearest", aspect="equal")
        if ax is axs[1]:
            mask = np.isnan(M)
            ax.contourf(g, g, mask.astype(float), levels=[0.5, 1.5], colors="none",
                        hatches=["////"], extend="neither")
        ax.plot([0, 1], [1, 0], color=INK, lw=0.8, ls=(0, (2, 2)))
        ax.text(0.83, 0.21, r"$\kappa=0$", rotation=-45, color=INK, fontsize=7,
                ha="center", va="center", bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
        for nm, x, y in marks:
            ax.plot(x, y, "o", ms=4.5, mfc="white", mec=INK, mew=1.0, clip_on=False)
            ha = "left" if x < 0.9 else "right"
            dx = 0.035 if x < 0.9 else -0.035
            ax.text(x + dx, y + 0.03, nm, fontsize=6.5, color=INK, ha=ha, va="bottom")
        ax.set_xlabel(r"$\omega_1=U_{00}$ (neither)")
        ax.set_ylabel(r"$\omega_2=U_{11}$ (both)")
        ax.set_title(title, loc="left")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_edgecolor(BASE)
    panel_label(axs[0], "A")
    panel_label(axs[1], "B")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT + "fig_heatmap.pdf", metadata=META)
    plt.close(fig)


# --------------------------------------------------------------------------------
# Figure 4: quadratic scaling of the regret in kappa
# --------------------------------------------------------------------------------
def fig_scaling():
    y = pd.read_csv(RES + "oracle_symmetric.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, sgn, title in ((axs[0], -1, r"$\kappa<0$ ($\omega_1=\omega_2<1/2$)"),
                           (axs[1], 1, r"$\kappa>0$ ($\omega_1=\omega_2>1/2$)")):
        for a, col in zip((0.3, 0.5, 0.8, 0.95), SEQ):
            z = y[(y.alpha == a) & (np.sign(y.kappa) == sgn)].sort_values("kappa")
            k = np.abs(z.kappa.values)
            ax.plot(k, 100 * z.regret_ind.values, "-o", color=col, ms=2.6,
                    label=rf"$\alpha={a}$")
        z = y[(y.alpha == 0.8) & (np.sign(y.kappa) == sgn)].sort_values("kappa")
        k = np.abs(z.kappa.values)
        ax.plot(k, 100 * k * kendall_tau("gauss", 0.8) / 4, color=INK2, lw=0.9,
                ls=(0, (5, 2)), label=r"bound $|\kappa|\tau/4$ ($\alpha=0.8$)")
        kk = np.array([0.05, 0.5])
        ref = 0.4 * 100 * z.regret_ind.values[np.argmin(np.abs(k - 0.05))] * (kk / 0.05) ** 2
        ax.plot(kk, ref, color=MUTED, lw=0.9, ls=":", label="slope 2 (reference)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$|\kappa|$")
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("mean regret (utility points)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    panel_label(axs[0], "A")
    panel_label(axs[1], "B")
    fig.tight_layout(w_pad=1.0, rect=(0, 0.08, 1, 1))
    fig.savefig(OUT + "fig_scaling.pdf", metadata=META)
    plt.close(fig)


# --------------------------------------------------------------------------------
# Figure 5: comparative statics (attraction / repulsion)
# --------------------------------------------------------------------------------
def fig_statics():
    cur = np.load(RES + "statics_curves.npz")
    al = cur["alphas"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    cols = (BLUE, ORANGE, AQUA)
    for ax, key, title in ((axs[0], "km0.6", r"$\kappa=-0.6$: repulsion from $d_c$"),
                           (axs[1], "kp0.6", r"$\kappa=+0.6$: attraction to $d_c$")):
        for p in range(3):
            ax.plot(al, cur[key][:, p], color=cols[p], label=f"patient {p + 1}: $d^*(\\alpha)$")
            ax.plot(al, cur["dc"][:, p], color=cols[p], lw=0.9, ls=(0, (3, 2)))
        ax.axvline(0, color=MUTED, lw=0.7)
        ax.axvline(0.8, color=MUTED, lw=0.7, ls=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r"latent correlation $\alpha$")
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("dose")
    axs[0].legend(loc="lower left", fontsize=6.5)
    panel_label(axs[0], "A")
    panel_label(axs[1], "B")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT + "fig_statics.pdf", metadata=META)
    plt.close(fig)


# --------------------------------------------------------------------------------
# Figure 2: odds ratio implied by a constant-parameter Gaussian copula
# --------------------------------------------------------------------------------
def fig_oddsratio():
    from scipy.special import ndtri
    cur = np.load(RES + "statics_curves.npz")
    f = pd.read_csv(RES + "misspec_family.csv")
    th = float(f[(f.truth == "Gaussian") & (f.fitted == "plackett")].theta.iloc[0])
    g = np.linspace(0.02, 0.98, 241)
    U_, V_ = np.meshgrid(g, g)
    p11 = bvn_cdf(ndtri(U_), ndtri(V_), 0.8)
    OR = p11 * (1 - U_ - V_ + p11) / ((U_ - p11) * (V_ - p11))
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    lv = np.arange(1.1, 3.01, 0.1)
    cs = ax.contourf(U_, V_, np.log10(OR), levels=lv, cmap=CMAP, extend="both", antialiased=True)
    cs.set_edgecolor("face")
    cl = ax.contour(U_, V_, np.log10(OR), levels=[np.log10(th), 2], colors=[INK, INK2],
                    linewidths=[1.2, 0.6], linestyles=["--", "-"])
    ax.clabel(cl, fmt={np.log10(th): rf"$\theta^*={th:.1f}$", 2: "OR = 100"},
              fontsize=6.5, inline=True)
    d = np.linspace(-1, 1, 201)
    for p, col in zip(range(3), (BLUE, ORANGE, AQUA)):
        aE, sE, aT, sT = cur["coef"][p]
        pE, pT = ndtr(aE + sE * d), ndtr(aT + sT * d)
        keep = (pE > 0.01) & (pE < 0.99) & (pT > 0.01) & (pT < 0.99)
        ax.plot(pE[keep], pT[keep], color="white", lw=3.0)
        ax.plot(pE[keep], pT[keep], color=col, lw=1.6, label=f"patient {p + 1}")
        k = np.where(keep)[0]
        ax.annotate("", xy=(pE[k[-1]], pT[k[-1]]), xytext=(pE[k[-5]], pT[k[-5]]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
    cb = fig.colorbar(cs, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(r"$\log_{10}$ odds ratio", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_edgecolor(BASE)
    ax.set_xlabel(r"$p_E$")
    ax.set_ylabel(r"$p_T$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(r"Odds ratio implied by a Gaussian copula ($\alpha=0.8$)", loc="left", fontsize=7.5)
    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.9, frameon=True, edgecolor="none")
    fig.tight_layout()
    fig.savefig(OUT + "fig_oddsratio.pdf", metadata=META)
    plt.close(fig)


# --------------------------------------------------------------------------------
# Figure 6: sampling distribution of alpha-hat
# --------------------------------------------------------------------------------
def fig_alphahat():
    df = pd.read_csv(RES + "estimation_A.csv")
    data = [df.alpha_probit.values, df.alpha_rf_naive.values, df.alpha_rf_oob.values]
    labs = ["probit margins\n(in-sample, IFM)", "random forest\n(in-sample)",
            "random forest\n(out-of-bag)"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                    medianprops=dict(color=INK, lw=1.2),
                    boxprops=dict(facecolor="#e1e0d9", edgecolor=INK2, lw=0.8),
                    whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2, lw=0.8),
                    flierprops=dict(marker="o", ms=2, mfc=MUTED, mec="none", alpha=0.6))
    ax.set_xticks([1, 2, 3], labs)
    ax.axhline(0.8, color=BLUE, lw=1.2)
    ax.text(3.45, 0.8, r"true $\alpha$", color=BLUE, fontsize=6.5, va="bottom", ha="right")
    for i, x in enumerate(data):
        ax.text(i + 1, -0.62, f"{100 * np.mean(x >= 0.985):.0f}% at\nupper bound", ha="center",
                va="center", fontsize=6.5, color=INK2)
    ax.set_ylabel(r"$\hat\alpha$")
    ax.set_ylim(-1.0, 1.02)
    ax.set_title(r"Estimates of $\alpha$ (Scenario 1, $N=200$)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT + "fig_alphahat.pdf", metadata=META)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["patient", "heatmap", "scaling", "statics", "oddsratio", "alphahat"]
    info = {}
    if "patient" in which:
        info["patient"] = fig_patient()
    if "heatmap" in which:
        fig_heatmap()
    if "scaling" in which:
        fig_scaling()
    if "statics" in which:
        fig_statics()
    if "oddsratio" in which:
        fig_oddsratio()
    if "alphahat" in which:
        fig_alphahat()
    if info:
        with open(RES + "figure_info.json", "w") as fh:
            json.dump(info, fh, indent=2, default=float)
        print(info)
