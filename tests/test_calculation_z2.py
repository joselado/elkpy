"""Integration tests for Calculation.get_z2_invariant() (Z2 topological
invariant via Wannier-charge-center pumping -- no new Fortran, reuses task
9000's mesh export, see docs/design.md #20), against a real compiled elk
binary.

The physics test is monolayer graphene with intrinsic spin-orbit coupling
scaled up (soc_scale, docs/design.md #12): Kane & Mele's original
prediction (PRL 95, 146802 (2005)) is that planar graphene is a quantum
spin Hall insulator (Z2=1) for *any* nonzero intrinsic SOC strength -- real
unscaled carbon SOC is far too weak (~1 microeV-scale) to resolve on a
practical k-mesh, hence the artificial enhancement (changes only the size
of the resulting gap, not which topological phase the system is in).

soc_scale=3000, not the more modest 100x initially tried: a first attempt
at 100x passed check_gap() (a real ~15 meV gap at K) but gave the WRONG
answer (Z2=0), traced to mesh aliasing, not a bug -- the Dirac-point
anticrossing is only ~1e-3 wide in fractional coordinates at that scale,
far narrower than any practical mesh spacing, so check_gap() passing (it
only checks the sampled points, not the gap width in between) gave false
confidence. See docs/design.md #20 for the diagnostic (an
eigenstate_session() overlap-singular-value scan through K) that both
caught this and confirmed 3000x opens a mesh-resolvable ~1.4 eV gap.

A second, independent physics test is freestanding monolayer bismuth
("bismuthene"): a buckled honeycomb lattice (2 atoms/cell, vertically
offset -- same structural motif as buckled silicene/germanene, space group
P-3m1), predicted to be a QSH insulator by Murakami, PRL 97, 236805 (2006)
(arXiv:cond-mat/0607001) via Bi's own large ATOMIC spin-orbit coupling --
no soc_scale enhancement needed here, unlike graphene's artificially tiny
intrinsic SOC. Structure (a=4.34 Angstrom, buckling=1.73 Angstrom) and gap (0.555 eV
without SOC / 0.500 eV with SOC, DFT) from Cheng, Liu, Tan, Zhang, Wei,
Lv, Shi & Tang, "Thermoelectric Properties of a Monolayer Bismuth", J.
Phys. Chem. C 118, 904 (2014), confirmed independently in Freitas,
Rivelino, de Brito Mota, de Castilho, Kakanakova-Georgieva & Gueorguiev,
J. Phys. Chem. C 119, 23599 (2015), Table 1 -- a genuinely different
band-inversion mechanism
from graphene's Dirac-point picture: the gap minimum sits at Gamma (an
s-p-orbital band inversion, HgTe/CdTe-style), not at K, confirmed directly
by scanning get_eigenstates() across Gamma-K-M before trusting any mesh.

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.
"""

import re

import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}

# Bohr, monolayer graphene (a=2.46 Angstrom, 20 Bohr vacuum -- same vacuum
# convention as the hBN/WSe2 slabs elsewhere in this test suite).
_BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
A = 2.46 * _BOHR_PER_ANGSTROM
VACUUM = 20.0
GRAPHENE_AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3**0.5 / 2, 0.0), (0.0, 0.0, VACUUM)]
GRAPHENE_SPECIES = {"C": [(0.0, 0.0, 0.5), (1 / 3, 2 / 3, 0.5)]}

K = (1 / 3, 1 / 3, 0)
GAMMA = (0, 0, 0)

# Bohr, freestanding buckled-honeycomb monolayer Bi ("bismuthene", 2 atoms/
# cell, P-3m1): a=4.34 Angstrom, buckling=1.73 Angstrom (Cheng, Liu, Tan,
# Zhang, Wei, Lv, Shi & Tang, J. Phys. Chem. C 118, 904 (2014); confirmed
# in Freitas, Rivelino, de Brito Mota, de Castilho, Kakanakova-Georgieva &
# Gueorguiev, J. Phys. Chem. C 119, 23599 (2015), Table 1). 28 Bohr vacuum
# (a bit more than the planar graphene/hBN slabs elsewhere in this suite,
# since the buckling itself already consumes ~3.3 Bohr of the cell along z).
BI_A = 4.34 * _BOHR_PER_ANGSTROM
BI_BUCKLING = 1.73 * _BOHR_PER_ANGSTROM
BI_VACUUM = 28.0
BI_AVEC = [(BI_A, 0.0, 0.0), (-BI_A / 2, BI_A * 3**0.5 / 2, 0.0), (0.0, 0.0, BI_VACUUM)]
_bi_half_delta = (BI_BUCKLING / 2) / BI_VACUUM
BI_SPECIES = {"Bi": [(0.0, 0.0, 0.5 + _bi_half_delta), (1 / 3, 2 / 3, 0.5 - _bi_half_delta)]}


@pytest.fixture
def si_calculation(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    return s.get_calculation(tmp_path / "si", xc="PW", ngridk=(4, 4, 4))


@pytest.fixture
def graphene_soc_calculation(tmp_path):
    calc = Structure(GRAPHENE_AVEC, GRAPHENE_SPECIES).get_calculation(
        tmp_path / "graphene_soc",
        xc="PW",
        ngridk=(6, 6, 1),
        rgkmax=7.0,
        spinorb=True,
        soc_scale={"C": 3000.0},
    )
    calc.get_energy()
    # occupied-band count from EIGVAL.OUT's own occupation numbers, not an
    # assumed electron count -- same pitfall/fix as
    # test_calculation_spin.py's wse2_calculation fixture (spinorb forces
    # nspinor=2, so Elk's own occmax is 1.0 per second-variational state).
    first_block = (calc.workdir / "EIGVAL.OUT").read_text().split("k-point")[1]
    state_line = re.compile(r"^\s*\d+\s+(\S+)\s+(\S+)\s*$")
    occ = [float(m.group(2)) for m in map(state_line.match, first_block.splitlines()) if m]
    ist1 = sum(o > 0.5 for o in occ)
    return calc, ist1


@pytest.fixture
def bi_calculation(tmp_path):
    calc = Structure(BI_AVEC, BI_SPECIES).get_calculation(
        tmp_path / "bi", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0, spinorb=True
    )
    calc.get_energy()
    first_block = (calc.workdir / "EIGVAL.OUT").read_text().split("k-point")[1]
    state_line = re.compile(r"^\s*\d+\s+(\S+)\s+(\S+)\s*$")
    occ = [float(m.group(2)) for m in map(state_line.match, first_block.splitlines()) if m]
    ist1 = sum(o > 0.5 for o in occ)
    return calc, ist1


def test_trivial_bulk_si_smoke_test(si_calculation):
    """Bulk Si has no spin-orbit coupling here (spinorb=False) and isn't a
    genuinely 2D system -- this just exercises the plumbing (mesh
    construction, gap check, WCC/Z2 arithmetic) end-to-end on a small,
    fast system, not a claim about Si's own topology (a single k3=0 slice
    of a 3D crystal isn't the physically meaningful 3D Z2/weak-index
    construction -- see docs/design.md #20)."""
    result = si_calculation.get_z2_invariant(1, 4, nkx=9, nt=5)
    assert result["z2"] in (0, 1)
    assert result["wannier_centers"].shape == (5, 4)
    assert result["pump"] == pytest.approx([0.0, 0.125, 0.25, 0.375, 0.5])


def test_soc_enhanced_graphene_gap_is_resolvable(graphene_soc_calculation):
    """Sanity check before trusting the Z2 result: the occupied pi band
    must stay clearly gapped from pi* at K (the Brillouin-zone minimum of
    that gap, and the point where the Kane-Mele SOC term actually acts).
    This alone is NOT sufficient to trust get_z2_invariant()'s mesh (a
    ~15 meV gap at soc_scale=100 also passes this, yet that mesh aliases
    away the topological signal -- see this module's docstring); it only
    confirms the scaled SOC term (docs/design.md #12) is acting on the
    right states, not that nkx/nt below are dense enough to resolve it."""
    calc, ist1 = graphene_soc_calculation
    with calc.eigenstate_session() as session:
        energies = session.get_eigenstates(K).energies
    gap = energies[ist1] - energies[ist1 - 1]
    assert gap > 0.02  # Hartree (~544 meV); ~1.4 eV (~0.05 Ha) expected at soc_scale=3000


def test_z2_enhanced_soc_graphene_is_quantum_spin_hall(graphene_soc_calculation):
    """Kane & Mele, PRL 95, 146802 (2005): planar graphene with intrinsic
    spin-orbit coupling is a quantum spin Hall insulator (Z2=1) -- the
    founding prediction of the whole field. ist0/ist1 spans every occupied
    valence band (sigma manifold included), relying on Z2's additivity mod
    2 across independently-gapped band groups (docs/design.md #20) rather
    than hand-isolating just the pi/pi* complex."""
    calc, ist1 = graphene_soc_calculation
    result = calc.get_z2_invariant(1, ist1, nkx=24, nt=13)
    assert result["z2"] == 1


def test_bi_gap_minimum_is_at_gamma_not_k(bi_calculation):
    """Bismuthene's band inversion is an s-p-orbital effect at Gamma
    (HgTe/CdTe-style), unlike graphene's Dirac-point-at-K mechanism -- a
    genuinely different topological gap-opening mechanism from the
    graphene test above, not just a bigger version of the same physics.
    Confirms the assumption get_z2_invariant()'s default loop_direction/
    pump_direction=(1,2) mesh (which always includes Gamma at index (0,0),
    unlike K which needs nkx/nky_full to be multiples of 3 to land on it)
    is adequate here without special mesh alignment."""
    calc, ist1 = bi_calculation
    with calc.eigenstate_session() as session:
        gap_gamma = session.get_eigenstates(GAMMA).energies[ist1] - session.get_eigenstates(GAMMA).energies[ist1 - 1]
        gap_k = session.get_eigenstates(K).energies[ist1] - session.get_eigenstates(K).energies[ist1 - 1]
    assert gap_gamma < gap_k
    assert gap_gamma > 0.02  # Hartree (~544 meV); ~0.6 eV expected


def test_z2_buckled_bismuthene_is_quantum_spin_hall(bi_calculation):
    """Murakami, PRL 97, 236805 (2006): freestanding buckled-honeycomb
    monolayer Bi is a quantum spin Hall insulator (Z2=1), driven by Bi's
    own large atomic spin-orbit coupling -- no soc_scale enhancement
    needed, unlike graphene. Same additivity argument as the graphene test
    justifies using the full occupied valence manifold."""
    calc, ist1 = bi_calculation
    result = calc.get_z2_invariant(1, ist1, nkx=24, nt=13)
    assert result["z2"] == 1
