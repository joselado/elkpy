"""Integration tests for Calculation.get_z2_invariant_3d() (the full 3D
strong/weak Z2 classification (nu0; nu1, nu2, nu3), Fu, Kane & Mele, PRL 98,
106803 (2007) -- see docs/design.md #21 and docs/physics.tex Part X), against
a real compiled elk binary.

The physics test is a "dimerized diamond lattice" built from cesium (Cs) --
the exact minimal lattice model Fu & Kane use to introduce the (nu0;nu1,nu2,nu3)
classification in the first place (Fu & Kane, PRB 76, 045302 (2007),
arXiv:cond-mat/0611341, their Eq. 4 and Sec. IV.3, confirmed directly against
the arXiv HTML source): the diamond structure (space group Fd-3m, same
2-atom-per-cell structure as this project's own Si tests), with the second
basis atom displaced along the cubic body diagonal [111] by a small amount
delta -- from the ideal (0.25,0.25,0.25) to (0.25-delta,0.25-delta,0.25-delta)
-- shortening exactly one of the four tetrahedral bonds per atom (the "111"
bond) while lengthening the other three. This is a TRIGONAL distortion
(reduces the space group to R-3m, #166, symmorphic), unlike a [001]
TETRAGONAL strain of the same diamond lattice (space group I4_1/amd, #141,
NONSYMMORPHIC -- see this module's "A discarded/corrected candidate" note
below).

Cesium (a single 6s valence electron, [Xe]6s^1) was chosen because its
low-energy physics is naturally close to the single-s-orbital-per-site
picture FKM's own tight-binding Hamiltonian assumes -- not because cesium
diamond is a real crystal phase (it is not; the ~16 Bohr lattice constant
used here is chosen only to keep muffin-tin spheres comfortably
non-overlapping, per this project's standing rule to consult Fable for
material/structure choices, CLAUDE.md's "Development practices" section).
Real intrinsic spin-orbit coupling from a single, energetically isolated 6s
band is expected to be extremely weak (the s-orbital itself carries no
orbital angular momentum; any effective SOC comes from a small admixture of
higher orbitals), so `soc_scale` (docs/design.md #12) enhances it here to a
numerically convenient magnitude -- the same "changes only the gap size, not
which phase the system is in" reasoning as this project's graphene Z2 test
(test_calculation_z2.py), not a claim that real cesium is topological.

FKM's own stated sign convention (Fu & Kane, Sec. IV.3, quoted directly from
the arXiv HTML source): "When the 111 distorted bond is stronger than the
other three bonds, so that the system is dimerized, the system is a strong
topological insulator [(nu0;nu1,nu2,nu3)=(1;111)]. When the 111 bond is
weaker than the other three, so that the system is layered, it is a weak
topological insulator [(0;111)]." A shorter bond means a shorter distance,
which is delta > 0 in this module's parametrization (moving the second atom
CLOSER to the first along [111]) -- the sign used below.

Getting the SAME (nu1,nu2,nu3)=(1,1,1) FKM report is not itself expected:
(nu1,nu2,nu3) are basis-dependent (docs/design.md #21) and FKM's own
Hamiltonian is written in a different primitive-lattice-vector convention
than this module's Structure -- what IS asserted, and IS basis-independent,
is nu0=1 (agreeing identically across all three axis choices, an algebraic
guarantee -- see wilson.combine_3d_invariants()) and nu1=nu2=nu3 (guaranteed
here by this structure's own residual 3-fold rotation about [111], which
cyclically permutes the three primitive reciprocal directions -- both atomic
positions are fixed points of that permutation, since atom 1 is at the
origin and atom 2's three fractional coordinates are all equal).

A discarded/corrected candidate: strained alpha-Sn (see git history of this
module and docs/design.md #21) was tried first with a [001] TETRAGONAL
lattice strain (matching a specific epitaxial-growth geometry from Huang &
Liu, PRB 95, 201101(R) (2017) -- not the same distortion as FKM's own [111]
model above). That probe found the gap pinned to ~1e-6 eV at the strained
Brillouin-zone boundary, identically across four strains tried -- initially
misdiagnosed in this project's history as proof the material isn't gapped
at all. Consulting Fable (per this project's standing "ask Fable about
material/structure questions" rule) corrected this: [001] strain reduces the
space group to I4_1/amd (#141), which is NONSYMMORPHIC and (per Hirschmann,
Leonhardt, Kilic, Fabini & Schnyder, Phys. Rev. Materials 5, 054202 (2021),
arXiv:2102.04134, Table II) enforces exact band sticking along zone-boundary
lines at ANY strain magnitude -- but Watanabe, Po, Zaletel & Vishwanath, PRL
117, 096404 (2016), arXiv:1603.05646, show SG 141 still admits genuine band
insulators at fillings that are multiples of 4, so this sticking does not by
itself forbid a gap at alpha-Sn's actual filling -- the ~1e-6 eV pinned
value most likely reflects measuring a splitting INSIDE one such
symmetry-stuck quartet (or in the nearly-flat semicore manifold) rather than
the true valence-conduction gap. That probe should therefore be recorded as
inconclusive, not as a disproof -- [111] distortion (this module's Cs model,
and FKM's own toy Hamiltonian) sidesteps the question entirely by reducing
to the SYMMORPHIC R-3m instead, which has no such enforced sticking.

A real-material attempt, bulk Bi2Se3 (rhombohedral, R-3m, sourced from a
real deposited structure -- COD entry 9011965 -- and independently
bond-length-verified, 3.075/2.851 Angstrom Bi-Se distances, matching the
well-known physical picture), was also tried and is recorded here for
honesty rather than silently dropped: ground state converged with a robust,
correctly-sized gap (0.258 eV at Gamma, the Brillouin-zone minimum sampled
along a Gamma-Z-F-L-Gamma path), but get_z2_invariant_3d(1, 78, nkx=8, nt=5)
gave nu0=0 on all six planes -- the WRONG answer relative to the
well-established literature result (1;000) (Zhang, Liu, Qi, Dai, Fang &
Zhang, Nature Physics 5, 438 (2009)). A narrower band window (61 to 78,
isolating the upper valence complex from ~3600 near-machine-precision
internal degeneracies found in the deeper semicore manifold on the same
mesh) gave the SAME nu0=0 on the one plane it was tested on, ruling out
semicore-window contamination as the cause. Whether this is a genuine
mesh-convergence problem (nkx=8/nt=5 too coarse for this system's more
intricate, all-electron band structure) or something else was not resolved
-- doing so would need a materially denser mesh, at a cost of many times the
~45 minutes a single six-plane sweep already took, so this was left as an
open, explicitly documented question rather than chased further; see
docs/design.md #21. No assertion about Bi2Se3 is made in this test file.

nkx=12, nt=7 (a 12x12 loop mesh per plane) is used below, matching the
parameters this module's result was verified with. Skipped by default (six
full mesh runs); set ELKPY_RUN_SLOW_TESTS=1 to run. Also skipped if the elk
binary hasn't been built, same as test_calculation_si.py.
"""

import os
import re

import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = [
    pytest.mark.skipif(
        not config.default_elk_binary().is_file(),
        reason="elk binary not built; see docs/design.md #8",
    ),
    pytest.mark.skipif(
        os.environ.get("ELKPY_RUN_SLOW_TESTS") != "1",
        reason="six full Z2 mesh runs; set ELKPY_RUN_SLOW_TESTS=1 to run",
    ),
]

# Bohr. Diamond structure (same avec convention as test_calculation_si.py's
# SI_AVEC), a=16.0 Bohr -- chosen only to keep Cs's (RMT=2.8 Bohr) muffin-tin
# spheres comfortably non-overlapping (nearest-neighbour distance a*sqrt(3)/4
# = 6.93 Bohr), not a real cesium crystal phase.
A = 16.0
DIAMOND_AVEC = [(0.0, A / 2, A / 2), (A / 2, 0.0, A / 2), (A / 2, A / 2, 0.0)]
# delta > 0 shortens the [111] bond (atom 2 moved toward atom 1) -- FKM's
# "dimerized" branch, the strong-topological-insulator sign (see module
# docstring).
DELTA = 0.03
CS_DIMERIZED_SPECIES = {
    "Cs": [(0.0, 0.0, 0.0), (0.25 - DELTA, 0.25 - DELTA, 0.25 - DELTA)]
}


@pytest.fixture
def cs_dimerized_calculation(tmp_path):
    calc = Structure(DIAMOND_AVEC, CS_DIMERIZED_SPECIES).get_calculation(
        tmp_path / "cs_dimerized", xc="PW", ngridk=(4, 4, 4), rgkmax=7.0,
        spinorb=True, soc_scale={"Cs": 3000.0},
    )
    calc.get_energy()
    first_block = (calc.workdir / "EIGVAL.OUT").read_text().split("k-point")[1]
    state_line = re.compile(r"^\s*\d+\s+(\S+)\s+(\S+)\s*$")
    occ = [float(m.group(2)) for m in map(state_line.match, first_block.splitlines()) if m]
    ist1 = sum(o > 0.5 for o in occ)
    return calc, ist1


def test_z2_3d_dimerized_cs_diamond_is_a_strong_topological_insulator(cs_dimerized_calculation):
    """Fu & Kane, PRB 76, 045302 (2007), Sec. IV.3: their own diamond-lattice
    model, dimerized (shortened [111] bond, delta>0) is a strong topological
    insulator, nu0=1. (nu1,nu2,nu3) are basis-dependent (docs/design.md #21)
    so are not asserted against FKM's own (1;1,1,1) -- only that they come
    out mutually equal, guaranteed by this structure's residual 3-fold
    rotation about [111] (see module docstring)."""
    calc, ist1 = cs_dimerized_calculation
    result = calc.get_z2_invariant_3d(1, ist1, nkx=12, nt=7)
    assert result["nu0"] == 1
    assert len(set(result["nu0_by_axis"])) == 1
    assert len(set(result["nu"])) == 1
