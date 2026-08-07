"""Parse INFO.OUT for SCF convergence status.

Verified against a real Elk 11.0.2 run (Si, examples/basic/Si): INFO.OUT
contains the literal line "Convergence targets achieved" on success, or
"Reached self-consistent loops maximum" (src/gndstate.f90) if maxscl is hit
without converging. Prefer this over trying to parse the loop-by-loop energy
table, which is more likely to drift across Elk versions (see
docs/design.md #7).
"""

CONVERGED_MARKER = "Convergence targets achieved"
NOT_CONVERGED_MARKER = "Reached self-consistent loops maximum"


def parse_convergence(info_out_path):
    """Return True if converged, False if maxscl was hit, None if neither
    marker is present (e.g. INFO.OUT is from a non-ground-state task)."""
    text = open(info_out_path).read()
    if CONVERGED_MARKER in text:
        return True
    if NOT_CONVERGED_MARKER in text:
        return False
    return None
