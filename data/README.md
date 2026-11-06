# Input data

Both files reproduce information published in Li P, Taylor JMG, Boonstra PS, Lawrence TS, Schipper MJ.
*Utility based approach in individualized optimal dose selection using machine learning methods.*
Statistics in Medicine 2022;41(16):2957–2977 (open access under CC BY).

## `scenario1_coefficients.csv`

Coefficients of the data-generating model of Scenario 1 (Li et al., 2022, Section 4.1), read by
`code/core.py`.

| Column | Description |
|---|---|
| `outcome` | `E` (efficacy) or `T` (toxicity) |
| `term` | `intercept`, covariates `x1`–`x5`, dose `d`, dose–covariate interactions `d:x1`–`d:x5` |
| `coefficient` | coefficient on the probit scale |

Model: Pr(E = 1 | d, x) = Φ(β0E + x'βxE + d βdE + d x'βdxE), and likewise for T; covariates are
independent N(0, 1), drawn subject to positive dose slopes for both outcomes, i.e.
1 + 0.23x1 + 0.61x2 + 1.69x4 + 0.5x5 > 0 and 1 + 0.03x1 + 0.6x2 − 0.42x4 + 1.04x5 > 0; dose
d ~ U(−1, 1); efficacy and toxicity are linked by a Gaussian copula with latent correlation 0.8.

## `sbrt_counts_li2022.csv`

2×2 table of efficacy and toxicity in the liver cancer SBRT study (Li et al., 2022, Section 5), read by
`code/application.py`.

| Column | Description |
|---|---|
| `E` | 1 = no tumour progression within one year (efficacy), 0 = progression |
| `T` | 1 = increase of at least two points in the Child–Pugh score within six months (toxicity), 0 = otherwise |
| `count` | number of patients |

The table is derived from the counts reported in the article: 182 analysed patients, 48 with toxicity,
11 with progression within one year (E = 0) and 46 with both efficacy and toxicity. Hence
n(E=1, T=1) = 46, n(E=0, T=1) = 48 − 46 = 2, n(E=1, T=0) = 171 − 46 = 125 and n(E=0, T=0) = 11 − 2 = 9.
As in the original study, the 39 deaths and 9 liver transplants without prior progression are counted as
E = 1.
