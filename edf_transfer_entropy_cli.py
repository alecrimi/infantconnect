"""Command-line transfer-entropy connectome from EDF, saved as CSV.

Uses effconn.compute_transfer_entropy (binned plug-in estimator, natural log):
  * values are in nats, >= 0
  * defaults: --bins 4 (quantile bins per channel), --lag 1 sample (as in the viewer)
  * CSV layout follows effconn.py: row i, column j = TE from channel j (source)
    to channel i (target); add --source-rows to transpose.

Examples:
  python edf_transfer_entropy_cli.py rec.edf -o out/
  python edf_transfer_entropy_cli.py rec.edf --bins 6 --lag 2 --edge-list
See edf_effconn_cli.py for all options. Needs effconn.py next to the scripts.
"""
import sys

from edf_effconn_cli import main

if __name__ == "__main__":
    sys.exit(main(fixed_method="transfer_entropy"))
