"""Parse EFFMASS.OUT (task 25, src/effmass.f90): one block per state, each
holding the 3x3 matrix of eigenvalue derivatives w.r.t. k (Cartesian) and
its trace.
"""

import numpy as np


def parse_effective_mass(effmass_out_path):
    """Return a list of dicts, one per state:
    {"state": int, "eigenvalue": float, "tensor": (3,3) array, "trace": float}
    """
    lines = [line.rstrip("\n") for line in open(effmass_out_path)]
    results = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("State, eigenvalue"):
            parts = lines[i].split(":", 1)[1].split()
            state, eigenvalue = int(parts[0]), float(parts[1])
            # next non-blank line is the "matrix of eigenvalue derivatives"
            # header, then exactly 3 rows, then a "trace :" line
            j = i + 1
            while not lines[j].strip().startswith("matrix"):
                j += 1
            rows = [
                [float(x) for x in lines[j + 1 + r].split()] for r in range(3)
            ]
            trace_line = lines[j + 4]
            trace = float(trace_line.split(":", 1)[1])
            results.append(
                {
                    "state": state,
                    "eigenvalue": eigenvalue,
                    "tensor": np.array(rows),
                    "trace": trace,
                }
            )
            i = j + 5
        else:
            i += 1
    if not results:
        raise ValueError(f"no state blocks found in {effmass_out_path}")
    return results
