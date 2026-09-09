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


def parse_charges(info_out_path, index=-1):
    """Parse the "Charges :" block of INFO.OUT (src/writechg.f90).

    ``writechg`` is called once per SCF iteration, so INFO.OUT holds one
    block per loop; `index` selects which (default -1, the converged one).

    Returns a dict:

    - "core", "valence", "interstitial", "muffin_tin_total",
      "total_calculated", "total", "error": floats, in electrons.
    - "excess": present only when ``chgexs`` is non-zero, which is the only
      case writechg emits the line.
    - "muffin_tin" (natmtot,), "core_leakage" (natmtot,): the per-atom
      muffin-tin charge and core leakage, in Elk's own ``ias`` order --
      species outer, atom inner (src/init0.f90's idxas loop). That is the
      same atom ordering STATE.OUT uses, so the two index together.
    - "species" (natmtot,): the element symbol of each ias.

    Which of these numbers a density reader may check itself against is
    decided by ``rhonorm`` (src/rhonorm.f90, called from src/rhomag.f90:24,
    on by default). It adds a uniform constant to ``rhoir`` and to the l=0
    channel of ``rhomt`` so the total charge comes out right, then updates
    ``chgmt``/``chgmttot`` and sets ``chgir = chgtot - chgmttot``. It does
    NOT update ``chgcalc``. So, all measured on tests/fixtures/h_sc:

    - "muffin_tin" is the clean check. It is a pure radial integral inside
      the sphere (src/charge.f90:30) with the characteristic function
      nowhere in it, and it is post-rhonorm, so it describes the ``rhomt``
      that STATE.OUT actually holds.
    - "total_calculated" and "error" are PRE-rhonorm. The fixture's 7.4e-4
      is the discrepancy rhonorm corrected, not a floor under a
      reintegration check -- the density in STATE.OUT integrates to the
      total, and "muffin_tin_total" + "interstitial" is exactly "total".
    - "interstitial" is not comparable to a sharp-boundary sum over
      ``rhoir``. Before rhonorm it is ``rhoir`` weighted by the SMOOTH
      characteristic function (a Fourier-truncated step, src/gencfun.f90);
      after it, it is whatever closes the total. On the fixture a sharp
      in-or-out sum gives 0.38466 against the printed 0.38742, 0.7% apart.
    """
    blocks = []
    current = None
    for raw in open(info_out_path):
        line = raw.rstrip("\n")
        if line.startswith("Charges :"):
            current = []
            blocks.append(current)
            continue
        if current is None:
            continue
        if not line.strip():
            current = None
            continue
        current.append(line)
    if not blocks:
        raise ValueError(f"no 'Charges :' block found in {info_out_path}")
    return _parse_charge_block(blocks[index])


def _parse_charge_block(lines):
    scalars = {
        " core": "core",
        " valence": "valence",
        " interstitial": "interstitial",
        " total in muffin-tins": "muffin_tin_total",
        " excess": "excess",
        " total calculated charge": "total_calculated",
        " total charge": "total",
        " error": "error",
    }
    result = {"muffin_tin": [], "core_leakage": [], "species": []}
    symbol = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("species :"):
            # "  species : 1 (H)" -- the symbol is spsymb(is), the element
            symbol = stripped.split("(", 1)[1].rstrip(")")
            continue
        if stripped.startswith("atom "):
            # "   atom 1  : <chgmt> ( <chgcrlk> )"
            value = stripped.split(":", 1)[1]
            charge, leakage = value.split("(", 1)
            result["muffin_tin"].append(float(charge))
            result["core_leakage"].append(float(leakage.rstrip().rstrip(")")))
            result["species"].append(symbol)
            continue
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = scalars.get(label.rstrip())
        if key is not None:
            result[key] = float(value)
    return result
