"""Parse TOTENERGY.OUT: one total energy value (Hartree) per SCF iteration,
last line is the final (converged, if converged) value.
"""


def parse_final_energy(totenergy_out_path):
    with open(totenergy_out_path) as fh:
        lines = [line.strip() for line in fh if line.strip()]
    if not lines:
        raise ValueError(f"{totenergy_out_path} is empty")
    return float(lines[-1])
