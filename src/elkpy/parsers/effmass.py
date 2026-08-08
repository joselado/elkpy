"""Parse EFFMASS.OUT (task 25, src/effmass.f90): one block per state,
holding two distinct 3x3 matrices in this order -- the raw matrix of
eigenvalue derivatives w.r.t. k ("d", the un-inverted curvature), and then
the actual effective mass tensor ("em = inverse of d", via `call
r3minv(d,em)`), each with its own trace. `tensor`/`trace` below are the
physical effective-mass tensor (the second block), not the raw derivatives.
"""

import numpy as np


def parse_effective_mass(effmass_out_path):
    """Return a list of dicts, one per state:
    {"state": int, "eigenvalue": float,
     "tensor": (3,3) array, "trace": float,               # effective mass (inverted)
     "derivative_tensor": (3,3) array, "derivative_trace": float}  # raw d^2E/dk^2
    """
    lines = [line.rstrip("\n") for line in open(effmass_out_path)]
    results = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("State, eigenvalue"):
            parts = lines[i].split(":", 1)[1].split()
            state, eigenvalue = int(parts[0]), float(parts[1])

            j = i + 1
            while not lines[j].strip().startswith("matrix of eigenvalue derivatives"):
                j += 1
            derivative_rows = [[float(x) for x in lines[j + 1 + r].split()] for r in range(3)]
            derivative_trace = float(lines[j + 4].split(":", 1)[1])

            k = j + 5
            while not lines[k].strip().startswith("effective mass tensor"):
                k += 1
            mass_rows = [[float(x) for x in lines[k + 1 + r].split()] for r in range(3)]
            mass_trace = float(lines[k + 4].split(":", 1)[1])

            results.append(
                {
                    "state": state,
                    "eigenvalue": eigenvalue,
                    "tensor": np.array(mass_rows),
                    "trace": mass_trace,
                    "derivative_tensor": np.array(derivative_rows),
                    "derivative_trace": derivative_trace,
                }
            )
            i = k + 5
        else:
            i += 1
    if not results:
        raise ValueError(f"no state blocks found in {effmass_out_path}")
    return results
