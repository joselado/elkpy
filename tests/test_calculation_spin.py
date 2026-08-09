"""Integration tests for Calculation.get_spin_operator()/
EigenstateSession.spin_operator() (built purely from evecsv -- no new
Fortran, see parsers/spin.py and docs/design.md #17), against a real
compiled elk binary.

The physics test is monolayer WSe2 with spin-orbit coupling: broken
inversion symmetry (monolayer TMDs lack the inversion center bulk 2H
stacking restores) plus strong SOC locks the valence-band-top spin to the
valley index, Sz(K) = -Sz(K') (Xiao, Liu, Feng,
Xu & Yao, "Coupled Spin and Valley Physics in Monolayers of MoS2 and Other
Group-VI Dichalcogenides," Phys. Rev. Lett. 108, 196802 (2012),
arXiv:1112.3144) -- the same K/K' time-reversal-partner argument already
used for Berry curvature (test_calculation_berry.py) and the quantum metric
(test_calculation_quantum_geometry.py), here applied to spin instead of
orbital/geometric quantities.

A second, independent check closes a gap the WSe2 test alone cannot: nothing
about Hermiticity, the su(2) algebra, or the K/K' antisymmetry can tell
whether evecsv's row block 1 (rows 0:nstfv) is actually Elk's own "spin-up"
-- a global relabelling of the two blocks flips the sign of every one of
those checks identically, so none of them can catch it (see
parsers.spin.compute_spin_operator's use of Elk's own row convention,
confirmed by reading eveqnsv.f90, but not otherwise runtime-tested until the
checks below). Collinear ferromagnetic Fe (spinpol=True, no SOC) closes this
two ways: eveqnsv.f90's own block-diagonalization branch (taken whenever
spinorb and noncollinear magnetism are both off) makes the first nstfv
eigenvectors exactly pure-up and the last nstfv exactly pure-down by
construction -- an exact (not approximate) structural check; and upstream
bandstr.f90 task 23 ("spin character of band") is an entirely separate
Fortran code path (gendmatk/wfmtsv, not evecsv arithmetic at all) that
prints its own column headers "spin-up and spin-down characters" for
ispn=1,2 -- agreement with THAT pins the absolute up/down labelling, not
just self-consistency of this feature's own arithmetic.

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.
"""

import re

import numpy as np
import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}

# Bohr, monolayer 2H-WSe2 (a=3.32 Angstrom, thickness=3.34 Angstrom, 12
# Angstrom vacuum on each side -- ase.build.mx2(formula="WSe2", kind="2H",
# a=3.32, thickness=3.34, vacuum=12) converted to Bohr, same convention as
# the hBN slab elsewhere in this test suite).
A = 6.273890733757557
C = 51.66511224726855
WSE2_AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3**0.5 / 2, 0.0), (0.0, 0.0, C)]
WSE2_SPECIES = {
    "W": [(0.0, 0.0, 0.5)],
    "Se": [(2 / 3, 1 / 3, 0.5610826627651793), (2 / 3, 1 / 3, 0.43891733723482085)],
}

K = (1 / 3, 1 / 3, 0)
KPRIME = (-1 / 3, -1 / 3, 0)

FE_AVEC = [(-2.71, 2.71, 2.71), (2.71, -2.71, 2.71), (2.71, 2.71, -2.71)]
FE_SPECIES = {"Fe": [((0.0, 0.0, 0.0), (0.0, 0.0, 4.0))]}  # bfcmt seeds the FM state


@pytest.fixture
def si_calculation(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    return s.get_calculation(tmp_path / "si", xc="PW", ngridk=(4, 4, 4))


@pytest.fixture
def fe_calculation(tmp_path):
    calc = Structure(FE_AVEC, FE_SPECIES).get_calculation(
        tmp_path / "fe", xc="PW", spinpol=True, ngridk=(2, 2, 2)
    )
    calc.get_energy()
    return calc


@pytest.fixture
def wse2_calculation(tmp_path):
    calc = Structure(WSE2_AVEC, WSE2_SPECIES).get_calculation(
        tmp_path / "wse2", xc="PW", ngridk=(3, 3, 1), rgkmax=7.0, spinorb=True
    )
    calc.get_energy()
    # Occupied-band count from EIGVAL.OUT's own occupation numbers, not an
    # assumed electron count -- same pitfall/fix as
    # test_calculation_atom_projection.py's hbn_calculation fixture, but
    # with the threshold halved: spinorb forces spinpol (nspinor=2), so
    # Elk's own occmax is 1.0 per second-variational state here, not 2.0
    # (each state already carries a definite, generally mixed, spin
    # character -- there is no separate up/down state to double-count).
    first_block = (calc.workdir / "EIGVAL.OUT").read_text().split("k-point")[1]
    state_line = re.compile(r"^\s*\d+\s+(\S+)\s+(\S+)\s*$")
    occ = [float(m.group(2)) for m in map(state_line.match, first_block.splitlines()) if m]
    ist1 = sum(o > 0.5 for o in occ)
    return calc, ist1


def test_spin_operator_requires_spin_polarization(si_calculation):
    """Bulk Si here is a plain spinpol=False, spinorb=False calculation
    (nspinor=1): there is no spin-up/spin-down block in evecsv to build
    sx/sy/sz from, so this must fail loudly rather than silently returning
    a meaningless or zero operator."""
    with pytest.raises(ValueError, match="spin-polarized"):
        si_calculation.get_spin_operator((0.1, 0.2, 0.05), ist0=1, ist1=1)


def test_matrices_are_hermitian_at_generic_k(wse2_calculation):
    """sx, sy, sz are Hermitian by construction (each is a sum of Gram-like
    products of evecsv's up/down blocks -- see compute_spin_operator's
    docstring) -- checkable independent of any external reference data."""
    calc, ist1 = wse2_calculation
    ops = calc.get_spin_operator((0.1, 0.2, 0.0), ist0=max(1, ist1 - 3), ist1=ist1 + 3)
    for name in ("sx", "sy", "sz"):
        m = getattr(ops, name)
        assert m == pytest.approx(m.conj().T, abs=1e-8), f"{name} not Hermitian"


def test_valence_band_spin_valley_locking(wse2_calculation):
    """Monolayer WSe2's broken inversion symmetry plus strong spin-orbit
    coupling locks the valence-band-top spin to the valley: <Sz> at the
    K and K' points (time-reversal partners, K'=-K mod a reciprocal
    lattice vector) must be equal in magnitude and opposite in sign (Xiao
    et al., PRL 108, 196802 (2012)) -- the RELATIVE sign is that published
    prediction, a sharp sign-of-the-effect check, not a plausibility band,
    the same spirit as this test suite's other K/K' checks (Berry curvature
    antisymmetry, quantum-metric parity, atom-projection N/B character).
    The ABSOLUTE sign pinned below (Sz(K) specifically negative) is not
    part of that prediction -- it's a regression pin, fixed by this
    structure's own conventions (chalcogen z-ordering, lattice-vector
    handedness) and by evecsv's up/down row-block assignment, independently
    verified against Elk's own bandstr.f90 spin-character output in
    test_spin_block_convention_matches_bandstr_spin_character below."""
    calc, ist1 = wse2_calculation
    with calc.eigenstate_session() as session:
        sz_k = session.spin_operator(K, ist1, ist1).sz[0, 0].real
        sz_kprime = session.spin_operator(KPRIME, ist1, ist1).sz[0, 0].real

    # Strong SOC in WSe2 (splitting ~400+ meV) nearly fully polarizes the
    # valence-band-top spin -- expect close to the maximal +-0.5, not just
    # any nonzero value.
    assert sz_k == pytest.approx(-0.5, abs=0.05)
    assert sz_kprime == pytest.approx(0.5, abs=0.05)
    assert sz_k == pytest.approx(-sz_kprime, abs=1e-3)


def test_pure_spin_states_are_exact_on_collinear_fe(fe_calculation):
    """Collinear ferromagnetic Fe (spinpol=True, spinorb=False, no
    noncollinear magnetism) takes eveqnsv.f90's block-diagonalization
    branch (line ~372: `else` of `if (ncmag.or.spinorb.or.(.not.spinpol))`),
    which zeros the off-diagonal spin blocks and diagonalizes the two
    remaining blocks separately: columns 1..nstfv of evecsv are then EXACTLY
    pure spin-up (identically zero in the spin-down row block) and columns
    nstfv+1..nstsv EXACTLY pure spin-down, by construction -- not merely to
    the ~1e-3 genolpq real-space-truncation floor documented elsewhere
    (docs/design.md #14), since no real-space expansion is involved in this
    feature at all (docs/physics.tex Part VI). Band 1 and band nstfv+1 are
    therefore guaranteed pure states at ANY k-point, with no search needed."""
    k0 = (0.1, 0.2, 0.05)
    with fe_calculation.eigenstate_session() as session:
        nstfv = session.get_eigenstates(k0).evecsv.shape[0] // 2
        up = session.spin_operator(k0, 1, 1)
        down = session.spin_operator(k0, nstfv + 1, nstfv + 1)

    assert up.sz[0, 0].real == pytest.approx(0.5, abs=1e-9)
    assert up.sx[0, 0].real == pytest.approx(0.0, abs=1e-9)
    assert up.sy[0, 0].real == pytest.approx(0.0, abs=1e-9)
    assert down.sz[0, 0].real == pytest.approx(-0.5, abs=1e-9)
    assert down.sx[0, 0].real == pytest.approx(0.0, abs=1e-9)
    assert down.sy[0, 0].real == pytest.approx(0.0, abs=1e-9)


def test_spin_block_convention_matches_bandstr_spin_character(fe_calculation):
    """Closes the one thing the exact-Sz check above cannot: WHICH physical
    spin (up vs. down) evecsv's row block 1 actually is. A global
    relabelling of the two blocks would flip every sign this feature
    produces identically -- Hermiticity, the su(2) algebra, and even the
    exact +-0.5 values above are all invariant under that relabelling (see
    this file's module docstring) -- so only an independent Fortran code
    path that does NOT go through evecsv-block arithmetic at all can pin
    it. upstream bandstr.f90 task 23 ("spin character of band", via
    gendmatk/wfmtsv, the same machinery test_calculation_atom_projection.py
    cross-checks task 21 against) prints exactly that: its own log message
    says "distance, eigenvalue, spin-up and spin-down characters" for
    columns 3/4, i.e. Elk's own code labels ispn=1 "spin-up" -- the same
    convention parsers.spin.compute_spin_operator assumes (row block 0,
    ispn=1) but had not, before this test, independently verified.

    A 2-point plot1d path whose first vertex is k0 gives literally the same
    k-point through both code paths (same argument as
    test_calculation_atom_projection.py's bandstr cross-check: Elk's own
    plotpt1d.f90 sets dp(1)=dv(1)=0, vpl(:,1)=vvl(:,1))."""
    k0 = (0.1, 0.2, 0.05)
    k1 = (0.15, 0.2, 0.05)
    with fe_calculation.eigenstate_session() as session:
        nstfv = session.get_eigenstates(k0).evecsv.shape[0] // 2
        up = session.spin_operator(k0, 1, 1)
        down = session.spin_operator(k0, nstfv + 1, nstfv + 1)

    subdir = fe_calculation.run_tasks(
        [23], blocks={"plot1d": [(2, 2), k0, k1]}, label="spin_character_check"
    )
    blocks = (subdir / "BAND_S01_A0001.OUT").read_text().split("\n\n")
    # each block is one band's data across the whole path; first line is k0
    up_char_up_band, up_char_down_band = (
        float(blocks[0].splitlines()[0].split()[2]),
        float(blocks[0].splitlines()[0].split()[3]),
    )
    down_char_up_band, down_char_down_band = (
        float(blocks[nstfv].splitlines()[0].split()[2]),
        float(blocks[nstfv].splitlines()[0].split()[3]),
    )

    assert up.sz[0, 0].real > 0  # this feature's own labelling for band 1
    assert up_char_up_band > 0.9 and up_char_down_band < 0.1  # bandstr agrees
    assert down.sz[0, 0].real < 0  # this feature's own labelling for band nstfv+1
    assert down_char_down_band > 0.9 and down_char_up_band < 0.1  # bandstr agrees
