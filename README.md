# When does the copula matter in utility-based individualized dose selection? — code and data

This repository contains the code, the input data and all simulation outputs behind the article

> [Authors]. *When does the copula matter in utility-based individualized dose selection?* [Journal], [year]. DOI: [to be added].

Every number, table and figure in the article can be regenerated with `run_all.sh`. No individual
patient data are used: the only empirical input is a 2×2 table of summary counts published by
Li et al. (2022).

## Contents

| Path | Content |
|---|---|
| `code/` | Python scripts (see the table below) |
| `data/` | Input data: coefficients of the simulation model and the published summary counts (see `data/README.md`) |
| `results/` | Simulation outputs used in the article (see `results/README.md` for a data dictionary) |
| `figures/` | Figures 1–6 of the article (PDF) |
| `tables/` | LaTeX tables and `numbers.tex`, which defines every number quoted in the text |
| `run_all.sh` | Runs the complete pipeline |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT licence (code); data and outputs are released under CC BY 4.0 |
| `CITATION.cff` | Citation metadata |

## Reproducing the article

```bash
pip install -r requirements.txt
NJOBS=2 ./run_all.sh          # about 30 minutes on 2 CPU cores
```

All random numbers are drawn from fixed seeds, so a clean run reproduces the files in `results/`, `tables/` and `figures/` byte for byte (tested with the versions listed below).

| Script | Produces | Used in the article |
|---|---|---|
| `core.py` | shared functions: bivariate normal distribution (Owen's T), copula families, simulation model, utilities, probit and IFM estimation | everywhere |
| `checks.py` | numerical checks of the copula formulas | Remarks 1–2 |
| `oracle_study.py` | population-level regret of ignoring dependence, with Monte Carlo standard errors | Table 1, Figures 3–4 |
| `misspec_study.py` | misspecified copula families; dose-dependent dependence (same 20,000 patients as Table 1) | Tables 2–3, Figure 2 |
| `statics.py` | comparative statics of the optimal dose | Proposition 4, Figures 1, 2 and 5 |
| `extra_checks.py` | monotone random forests; associations in the simulation model | Remarks 2–3 |
| `estimation_study.py A` | finite-sample study with probit and random-forest margins | Table 4, Figure 6 |
| `estimation_study.py B` | dose-dependent dependence with estimated margins | Table 5 |
| `application.py` | analysis of the published 2×2 table of the liver study | Table 6 |
| `verify_props.py` | numerical verification of Propositions 1, 2 and 6 | Appendix B |
| `figures.py` | all figures | Figures 1–6 |
| `make_numbers.py` | `tables/numbers.tex` and `tables/tab_*.tex` | all tables and numbers in the text |

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
