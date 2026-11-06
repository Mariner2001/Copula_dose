# When does the dependence between efficacy and toxicity matter in utility-based individualized dose selection? — code and data

This repository contains the code, the input data and all simulation outputs behind the article

> [Authors]. *When does the dependence between efficacy and toxicity matter in utility-based individualized dose selection?* [Journal], [year]. DOI: [to be added].

Every number, table and figure in the article and its Supplementary Material can be regenerated with `run_all.sh`. No individual
patient data are used: the only empirical input is a 2×2 table of summary counts published by
Li et al. (2022).

## Contents

| Path | Content |
|---|---|
| `code/` | Python scripts (see the table below) |
| `data/` | Input data: coefficients of the simulation model and the published summary counts (see `data/README.md`) |
| `results/` | Simulation outputs used in the article (see `results/README.md` for a data dictionary) |
| `figures/` | Figures 1–2 and Supplementary Figures S1–S4 (PDF) |
| `tables/` | LaTeX tables and `numbers.tex`, which defines every number quoted in the text |
| `run_all.sh` | Runs the complete pipeline |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT licence (code); data and outputs are released under CC BY 4.0 |
| `CITATION.cff` | Citation metadata |

## Reproducing the article

```bash
pip install -r requirements.txt
NJOBS=2 ./run_all.sh          # 30–45 minutes on 2 CPU cores
```

All random numbers are drawn from fixed seeds, so a clean run reproduces the files in `results/`, `tables/` and `figures/` byte for byte (tested with the versions listed below).

| Script | Produces | Used in the article |
|---|---|---|
| `core.py` | shared functions: bivariate normal distribution (Owen's T), copula families, simulation model, utilities, probit and IFM estimation | everywhere |
| `checks.py` | numerical checks of the copula formulas and of the corrections to Li et al. | Section 2.2 |
| `oracle_study.py` | population-level regret of ignoring dependence, with Monte Carlo standard errors | Table 2, Figure 2, Supplementary Figure S3 |
| `misspec_study.py` | misspecified copula families; dose-dependent dependence (same 20,000 patients as Table 2) | Table 3, Supplementary Table S1 and Figure S2 |
| `statics.py` | direction of the shift of the optimal dose | Supplementary Proposition S2; Figure 1, Supplementary Figures S1–S2 |
| `extra_checks.py` | monotone random forests; associations in the simulation model | Section 3.4, Supplementary Section S4 |
| `estimation_study.py A` | finite-sample study with probit and random-forest margins | Table 4, Supplementary Figure S4 |
| `estimation_study.py B` | dose-dependent dependence with estimated margins | Supplementary Table S2 |
| `application.py` | analysis of the published 2×2 table of the liver study | Section 6, Supplementary Table S3 |
| `verify_props.py` | numerical verification of Propositions 1(c), 2(a) and S1 | Supplementary Section S7 |
| `figures.py` | all figures (readable in black and white) | Figures 1–2, Supplementary Figures S1–S4 |
| `make_numbers.py` | `tables/numbers.tex` and `tables/tab_*.tex` | Tables 1–4, Supplementary Tables S1–S3 and all numbers in the text |

## Software

Tested with Python 3.13, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, scikit-learn 1.9.1,
matplotlib 3.11.2 and joblib 1.6.0. Minimum versions are given in `requirements.txt`
(scikit-learn ≥ 1.4 is needed for monotone forests, scipy ≥ 1.10 for the exact odds-ratio interval).

## Licence and citation

Code: MIT licence (`LICENSE`). Data and outputs: Creative Commons Attribution 4.0 (CC BY 4.0).
The two input files reproduce information published in Li P, Taylor JMG, Boonstra PS, Lawrence TS,
Schipper MJ. Utility based approach in individualized optimal dose selection using machine learning
methods. *Statistics in Medicine* 2022;41(16):2957–2977 (open access, CC BY); please cite that article
together with ours when using them.
