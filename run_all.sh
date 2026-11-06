#!/usr/bin/env bash
# Reproduces every number, table and figure of the article and its Supplementary Material
# (30-45 min on 2 CPU cores).
# Usage: NJOBS=2 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")/code"
NJOBS=${NJOBS:-2}
python3 checks.py           > /dev/null      # Section 2.2
python3 oracle_study.py     > /dev/null      # Table 2, Figure 2, Figure S3
python3 misspec_study.py    > /dev/null      # Table 3, Table S1 (and theta* in Figure S2)
python3 statics.py          > /dev/null      # Proposition S2; curves for Figures 1, S1 and S2
python3 extra_checks.py     > /dev/null      # Section 3.4, Section S4
python3 estimation_study.py A 500 "$NJOBS"   # Table 4, Figure S4
python3 estimation_study.py B 500 "$NJOBS"   # Table S2
python3 application.py      > /dev/null      # Section 6, Table S3
python3 verify_props.py                      # numerical checks of Propositions 1(c), 2(a) and S1
python3 figures.py          > /dev/null      # figures/*.pdf
python3 make_numbers.py                      # tables/numbers.tex and tables/tab_*.tex
echo "done"
