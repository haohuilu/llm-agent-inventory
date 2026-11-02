# Builds run_mac.py from the notebook so the experiment can run in a plain terminal (no Jupyter kernel involved).
# usage: python3 nb2py.py
import json, re
nb = json.load(open("UTD24_LLM_inventory_calibration_mac.ipynb"))
cells = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
out = ['import matplotlib; matplotlib.use("Agg")      # figures go to files, no window',
       'display = print                                # notebook display() -> print',
       'import sys, os']
for k, s in enumerate(cells):
    if k == 0:   # configuration cell: allow overrides from the command line
        s += ('\nif len(sys.argv) > 1: DATASET = sys.argv[1]            # nhs | pbs | freshretail\n'
              'if len(sys.argv) > 2: RUN = sys.argv[2]                # open | openai | anthropic | all\n'
              'STOP_AFTER_SECTION3 = len(sys.argv) > 3 and sys.argv[3] == "stop3"\n')
    s = s.replace('PBS_IN_THOUSANDS = True', 'PBS_IN_THOUSANDS = os.environ.get("PBS_UNITS", "k") != "raw"')
    if 'this run done; the next cell gathers' in s:
        s += '\nif STOP_AFTER_SECTION3: print("stopping after section 3 as requested"); sys.exit(0)\n'
    # drop bare trailing expressions that only render a table in a notebook (harmless, but noisy as script)
    out.append(f"\n# ===== cell {k} =====\n" + s)
open("run_mac.py", "w").write("\n".join(out) + "\n")
print("run_mac.py written from", len(cells), "cells")
