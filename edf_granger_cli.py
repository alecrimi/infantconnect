"""Command-line Granger-causality connectome from EDF, saved as CSV.

Uses effconn.compute_granger (pairwise F-test on OLS/VAR residuals):
  * values are -log10(p); >= 1.3 means p < 0.05; 0 = no test / failed fit
  * default lag = 5 samples (as in the viewer)
  * CSV layout follows effconn.py: row i, column j = influence of channel j
    (source) on channel i (target); add --source-rows to transpose.

Examples:
  python edf_granger_cli.py rec.edf -o out/
  python edf_granger_cli.py rec.edf --lag 8 --edge-list --min-strength 1.3
See edf_effconn_cli.py for all options. Needs effconn.py next to the scripts.
"""
import sys

from edf_effconn_cli import main

if __name__ == "__main__":
    sys.exit(main(fixed_method="granger"))
