"""Parser for TENSMOM.OUT (task 400, src/writetm.f90 -> src/writetm3.f90).

The DFT+U density matrix of an (atom, l) shell, n_{m sigma, m' sigma'}, is
a 2(2l+1) x 2(2l+1) Hermitian object with no obvious physical reading. The
coupled-tensor-moment decomposition (Bultmark, Cricchio, Granas and
Nordstrom, PRB 80, 035121 (2009); van der Laan and Thole, J. Phys.:
Condens. Matter 7, 9947 (1995)) re-expands it in irreducible spherical
tensors w^{kpr}_t built by coupling an orbital rank k (0..2l) to a spin
rank p (0 or 1), giving a total rank r in |k-p| .. k+p and 2r+1 components
t = -r..r. The physical content is then read off the ranks rather than
the matrix:

- (k,p,r) = (0,0,0) is the shell occupation,
- p = 1, k = 0 is the spin moment,
- k = 1, p = 0 is the orbital moment,
- k = 2 terms are quadrupolar (charge/spin) multipoles, and
- k = 1, p = 1 mixed terms are the on-site spin-orbit-like moments.

``writetm3`` also reports, per moment, the muffin-tin Hartree + exchange
energy that moment contributes, ``0.5*Tr[Gamma V]``, so the decomposition
doubles as an energy decomposition of the +U term.

Three output conventions exist, selected by the input variable
``tm3type``: 0 (default) real components corresponding to Hermitian Gamma
matrices, 1 the complex van der Laan convention, 2 real tesseral. Type 1
prints two numbers per component (``'("   t = ",I2," : ",2F14.8)'``), the
other two print one -- this parser handles both and reports the value as
complex only for type 1.
"""

import numpy as np


def _lines(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def parse_tensor_moments(tensmom_out_path):
    """Parse TENSMOM.OUT.

    File shape, per (atom, l) and then per (k,p,r) moment::

        write(50,'("Species : ",I4," (",A,"), atom : ",I4)') is,spsymb,ia
        write(50,'(" l = ",I1)') l
          write(50,'("  k = ",I1,", p = ",I1,", r = ",I1)') k,p,r
            write(50,'("   t = ",I2," : ",F14.8)') t,value
          write(50,'("  magnitude : ",F14.8)') t1
          write(50,'("  Hartree + exchange energy : ",F14.8)') ehx

    Returns a list of dicts, one per moment::

        {"species": int, "symbol": str, "atom": int, "l": int,
         "k": int, "p": int, "r": int,
         "t": (2r+1,) int array, "value": (2r+1,) float or complex array,
         "magnitude": float, "energy": float}

    ``energy`` is the moment's contribution to the muffin-tin Hartree +
    exchange energy, in Hartree.
    """
    lines = _lines(tensmom_out_path)
    results = []
    context = {}
    current = None
    for line in lines:
        s = line.strip()
        if s.startswith("Species :"):
            head, tail = s.split(",", 1)
            species_part = head.split(":", 1)[1]
            context = {
                "species": int(species_part.split("(", 1)[0]),
                "symbol": species_part.split("(", 1)[1].rsplit(")", 1)[0].strip(),
                "atom": int(tail.split(":", 1)[1]),
            }
        elif s.startswith("l ="):
            context["l"] = int(s.split("=", 1)[1])
        elif s.startswith("k ="):
            kpr = {}
            for part in s.split(","):
                name, _, value = part.partition("=")
                kpr[name.strip()] = int(value)
            current = dict(context, **kpr, t=[], value=[])
            results.append(current)
        elif s.startswith("t ="):
            if current is None:
                raise ValueError(f"{tensmom_out_path}: 't =' line outside a (k,p,r) block")
            index, _, value = s.partition(":")
            current["t"].append(int(index.split("=", 1)[1]))
            numbers = [float(x) for x in value.split()]
            current["value"].append(
                complex(numbers[0], numbers[1]) if len(numbers) == 2 else numbers[0]
            )
        elif s.startswith("magnitude :"):
            current["magnitude"] = float(s.split(":", 1)[1])
        elif s.startswith("Hartree + exchange energy :"):
            current["energy"] = float(s.split(":", 1)[1])
    if not results:
        raise ValueError(f"no tensor-moment blocks found in {tensmom_out_path}")
    for entry in results:
        entry["t"] = np.array(entry["t"], dtype=int)
        entry["value"] = np.array(entry["value"])
    return results
