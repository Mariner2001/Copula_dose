# Simulation outputs (data dictionary)

Utilities are on the 0–1 scale in these files (the article reports them multiplied by 100).
Utility labels: `TFE` = (ω1, ω2) = (0, 0); `k-0.6` = (0.2, 0.2); `U1` = (0.3, 0.5); `U3` = (0.5, 0.3);
`U2` = (0.5, 0.5); `APP` = (0.6, 0.4); `k+0.6` = (0.8, 0.8); `NFT` = (1, 1); κ = ω1 + ω2 − 1.

## Population-level analyses (true margins) — `oracle_study.py`

`oracle_selected.csv` (selected utilities, α = 0.8), `oracle_heatmap.csv` (21 × 21 grid of (ω1, ω2),
α = 0.8) and `oracle_symmetric.csv` (ω1 = ω2, α ∈ {0.3, 0.5, 0.8, 0.95}):

| Column | Description |
|---|---|
| `w1`, `w2`, `kappa` | utility matrix and utility interaction |
| `alpha`, `tau` | latent correlation of the true Gaussian copula and its Kendall's tau |
| `V_oracle` | mean expected utility of the oracle rule (true margins and copula) |
| `V_fixed`, `d_fixed` | mean expected utility of the best single dose, and that dose |
| `gain` | personalization gain, `V_oracle − V_fixed` |
| `regret_ind`, `regret_ind_max` | mean and maximum regret of the independence rule |
| `share_dchange_01`, `mean_abs_dchange` | share of patients whose dose changes by ≥ 0.1; mean absolute change |
| `share_corner_oracle` | share of oracle optima at d = ±1 |
| `uniform_bound`, `mean_osc_bound` | bound \|κ\|τ/4 and mean of the per-patient bound \|κ\| osc Δ |
| `regret_frechet_robust`, `regret_pqd_robust` | mean regret of the max–min doses over all copulas and over PQD copulas |
| `regret_quad_pred`, `share_interior_ind` | mean second-order approximation of the regret; share of interior optima of the independence rule |
| `regret_share_switch`, `quad_ratio_both_interior` | share of the regret due to patients whose oracle or independence optimum is a corner; ratio of the second-order approximation to the regret among patients with both optima interior |
| `se_rel_regret_ind`, `se_rel_regret_pqd`, `se_rel_regret_frechet` | Monte Carlo standard errors (500 bootstrap resamples of the patients) of the three regrets as percentages of the gain (`oracle_selected.csv` only) |
| `label` | utility label (`oracle_selected.csv` only) |

`oracle_meta.json`: summaries of the conditional association implied by α = 0.8.

## Copula misspecification — `misspec_study.py`

`misspec_family.csv`: `truth` (true copula, Kendall's tau 0.59), `fitted` (working family: indep, fgm,
gauss, plackett, frank, clayton, sclayton = survival Clayton, gumbel, sgumbel = survival Gumbel), `theta`
(pseudo-true parameter), `tau_fitted`, `utility`, `w1`, `w2`, `kappa`, `regret`, `gain`, `rel_regret`
(= regret / gain), `se_rel_regret` (Monte Carlo standard error of 100 × `rel_regret`, from 500 bootstrap
resamples of the patients). The rules are evaluated on the same 20,000 patients as in `oracle_study.py`.

`misspec_conditional.csv`: `scenario` (`increasing`: α(d) = tanh(0.75 + 0.75d); `decreasing`:
α(d) = tanh(0.75 − 0.75d)), `fitted` (`indep` or `gauss_const`), `utility`, `kappa`, `regret`, `gain`,
`rel_regret`, `se_rel_regret`, `share_dchange_01`, `bound`.

`misspec_meta.json`: matched parameters and pseudo-true constant correlations.

## Finite-sample studies — `estimation_study.py`

`estimation_A.csv` (500 replications, N = 200): `seed`; `alpha_probit` (IFM estimate with probit
margins); `alpha_rf_naive` and `alpha_rf_oob` (estimates from in-sample and out-of-bag random-forest
predictions); `V|method|utility` = mean true expected utility, on the validation set of 1,000 patients,
of the rule `method` for `utility`. Methods: `FD` (estimated fixed dose), `PR-IND`, `PR-GC`, `PR-GC*`
(probit margins with independence, estimated α, true α), `RF-IND`, `RF-GC-naive`, `RF-GC-oob`, `RF-GC*`
(random-forest margins with independence, in-sample α̂, out-of-bag α̂, true α).
`estimation_A_oracle.json`: oracle expected utility on the validation set, by utility.

`estimation_B.csv` (500 replications per truth and N ∈ {200, 500}): `seed`, `n`, `truth` (`constant`
α = 0.8 or `increasing` α(d)), `alpha_const` (constant IFM estimate), `g0`, `g1` (IFM estimates in
α(d) = tanh(g0 + g1 d)), `LR` (likelihood-ratio statistic for g1 = 0), and `V|method|utility` for `FD`,
`PR-IND`, `PR-GC` (constant α̂) and `PR-GCd` (dose-dependent α̂). `estimation_B_oracle.json`: oracle values.

## Other outputs

| File | Content |
|---|---|
| `statics.json` | check of formula (8): discrepancies with numerical derivatives, direction of the shift |
| `statics_curves.npz` | `alphas`; optimal-dose curves `km0.6`, `kp0.6`; centre doses `dc`; `coef` = (aE, sE, aT, sT) of the three illustrative patients |
| `application.json` | odds ratio, tetrachoric correlation and Fréchet interval for the published 2×2 table |
| `checks.json` | checks of the copula formulas (Remarks 1–2) |
| `extra_checks.json` | monotone forests; conditional and pooled associations in the simulation model |
| `figure_info.json` | values quoted in the caption of Figure 1 |
