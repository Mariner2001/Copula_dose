#!/usr/bin/env bash
# Reproduces every number, table and figure of the article (about 30 min on 2 CPU cores).
# Usage: NJOBS=2 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")/code"
NJOBS=${NJOBS:-2}
python3 checks.py           > /dev/null      # Remarks 1-2
python3 oracle_study.py     > /dev/null      # Table 1, Figures 3-4
python3 misspec_study.py    > /dev/null      # Tables 2-3 (and theta* in Figure 2)
python3 statics.py          > /dev/null      # Proposition 4; curves for Figures 1, 2 and 5
python3 extra_checks.py     > /dev/null      # Remarks 2-3
python3 estimation_study.py A 500 "$NJOBS"   # Table 4, Figure 6
python3 estimation_study.py B 500 "$NJOBS"   # Table 5
python3 application.py      > /dev/null      # Table 6
python3 verify_props.py                      # numerical checks of Propositions 1, 2 and 6
python3 figures.py          > /dev/null      # figures/*.pdf
python3 make_numbers.py                      # tables/numbers.tex and tables/tab_*.tex
echo "done"
