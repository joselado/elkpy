"""Integration tests for Calculation.get_orbital_projection()/
EigenstateSession.orbital_projection() (task 9002's ORBITAL query, backed by
elkpy_orbitalproj in src/elkpy_eigenstates.f90 -- patches/0005-orbital-projection.patch),
against a real compiled elk binary.

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.
"""

import re

import numpy as np
import pytest

from elkpy import config
from elkpy.session import ORBITAL_LABELS
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}

A, VACUUM = 4.743210000, 20.0  # Bohr, same hBN slab as notebooks/05_berry_curvature.ipynb
HBN_AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3**0.5 / 2, 0.0), (0.0, 0.0, VACUUM)]
HBN_SPECIES = {"B": [(1 / 3, 2 / 3, 0.5)], "N": [(2 / 3, 1 / 3, 0.5)]}


@pytest.fixture
def si_calculation(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    return s.get_calculation(tmp_path / "si", xc="PW", ngridk=(4, 4, 4))


@pytest.fixture
def hbn_calculation(tmp_path):
    calc = Structure(HBN_AVEC, HBN_SPECIES).get_calculation(
        tmp_path / "hbn", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0
    )
    calc.get_energy()
    # occupied-band count from EIGVAL.OUT's own occupation numbers, not an
    # assumed electron count -- same pitfall/fix as
    # test_calculation_atom_projection.py's hbn_calculation fixture.
    first_block = (calc.workdir / "EIGVAL.OUT").read_text().split("k-point")[1]
    state_line = re.compile(r"^\s*\d+\s+(\S+)\s+(\S+)\s*$")
    occ = [float(m.group(2)) for m in map(state_line.match, first_block.splitlines()) if m]
    ist0, ist1 = 1, sum(o > 1.0 for o in occ)
    return calc, ist0, ist1


def test_orbital_labels_order():
    assert ORBITAL_LABELS == ("s", "p", "d", "f")


def test_matrices_are_hermitian_and_positive_semidefinite(si_calculation):
    """Each (atom, l) P_{alpha,l} is a weighted Gram matrix (wr2cmt is a
    positive quadrature weight, see elkpy_orbitalproj's docstring) --
    Hermitian and PSD by construction, independent of any external reference
    data."""
    orb = si_calculation.get_orbital_projection((0.1, 0.2, 0.05), ist0=1, ist1=4)
    natmtot, nl, nst, _ = orb.matrices.shape
    assert natmtot == 2  # two Si atoms
    assert nl == 4
    for a in range(natmtot):
        for l in range(nl):
            m = orb.matrices[a, l]
            assert m == pytest.approx(m.conj().T, abs=1e-8)
            eigvals = np.linalg.eigvalsh(m)
            assert (eigvals > -1e-8).all(), f"atom {a} l={l} not PSD: {eigvals}"


def test_orbital_sum_falls_short_of_atom_total(si_calculation):
    """Summing P_{alpha,l} over l=0..3 (s,p,d,f) falls short of
    get_atom_projection()'s P_alpha (which sums over l=0..lmaxo, 6 by
    default) by exactly the l=4,5,6 (g,h,i) weight -- so the total minus
    the s+p+d+f sum must be Hermitian PSD (that weight can't be negative),
    the same "partial coverage, never an overshoot" shape as
    test_calculation_atom_projection.py's own atom-vs-identity check."""
    k = (0.1, 0.2, 0.05)
    orb = si_calculation.get_orbital_projection(k, ist0=1, ist1=4)
    proj = si_calculation.get_atom_projection(k, ist0=1, ist1=4)
    for a in range(orb.matrices.shape[0]):
        spdf_sum = orb.matrices[a].sum(axis=0)
        remainder = proj.matrices[a] - spdf_sum
        assert remainder == pytest.approx(remainder.conj().T, abs=1e-8)
        eigvals = np.linalg.eigvalsh(remainder)
        assert (eigvals > -1e-6).all(), f"atom {a}: s+p+d+f overshoots the atom total: {eigvals}"


def test_diagonal_matches_bandstr_l_resolved_character(si_calculation):
    """Cross-check against an entirely independent Fortran code path:
    task 21's own atom/l-resolved band-character output (bandstr.f90,
    upstream, unmodified -- BAND_Sss_Aaaaa.OUT's columns 4 onward, one per
    l), which calls the same gendmatk/wfmtsv machinery elkpy_orbitalproj's
    docstring describes reusing, but via a completely separate call site
    (Elk's own path diagonalisation, not elkpy_orbitalproj's fresh one) and
    a completely separate reduction (an explicit per-l loop, not this
    feature's masked zgemm). Unlike test_calculation_atom_projection.py's
    matching test, this needs no lmaxdb override: task 21's own default,
    lmaxdb=3, is EXACTLY s,p,d,f -- the same four channels
    get_orbital_projection() returns -- so this is an exact comparison at
    lmaxdb's own default, not merely a close one after raising it.

    Same 2-point plot1d path trick as the atom-projection cross-check
    (first point is exactly the first vertex): band 1 only, since
    bandstr.f90 does not gap-check for degeneracy."""
    k0 = (0.1, 0.2, 0.05)
    k1 = (0.15, 0.2, 0.05)
    subdir = si_calculation.run_tasks([21], blocks={"plot1d": [(2, 2), k0, k1]}, label="lcheck")
    lines = {}
    for symbol_index, fname in [(0, "BAND_S01_A0001.OUT"), (1, "BAND_S01_A0002.OUT")]:
        first_line = (subdir / fname).read_text().splitlines()[0]
        cols = [float(c) for c in first_line.split()]
        # columns: distance, energy, total(sum l=0..3), then l=0,1,2,3 (s,p,d,f)
        lines[symbol_index] = cols[3:7]

    orb = si_calculation.get_orbital_projection(k0, ist0=1, ist1=1)
    for atom_index, bandstr_weights in lines.items():
        for l, expected in enumerate(bandstr_weights):
            assert orb.matrices[atom_index, l][0, 0].real == pytest.approx(expected, abs=1e-5)


def test_spin_polarized_matches_bandstr_l_resolved_character(tmp_path):
    """The zgemm reduction accumulates over nspinor spin channels
    (elkpy_orbitalproj's `do ispn=1,nspinor` loop, inside the l loop) --
    untested by the unpolarized si_calculation fixture above, where
    nspinor=1 and that loop never runs twice. Rather than only checking
    Hermitian/PSD (which any consistent normalization satisfies, even an
    undercounting one), cross-check against bandstr.f90 task 21's own
    `bc(l,ias,ist,ik)` -- which sums over `ispn` the exact same way -- on a
    spin-polarized run, closing the gap the atom-projection test suite
    leaves open (it only checks Hermitian/PSD for nspinor=2, no external
    reference)."""
    calc = Structure(SI_AVEC, SI_SPECIES).get_calculation(
        tmp_path / "si_spinpol", xc="PW", ngridk=(2, 2, 2), spinpol=True
    )
    k0 = (0.1, 0.2, 0.05)
    k1 = (0.15, 0.2, 0.05)
    subdir = calc.run_tasks([21], blocks={"plot1d": [(2, 2), k0, k1]}, label="lcheck_spinpol")
    first_line = (subdir / "BAND_S01_A0001.OUT").read_text().splitlines()[0]
    cols = [float(c) for c in first_line.split()]
    bandstr_weights = cols[3:7]

    orb = calc.get_orbital_projection(k0, ist0=1, ist1=1)
    for l, expected in enumerate(bandstr_weights):
        assert orb.matrices[0, l][0, 0].real == pytest.approx(expected, abs=1e-5)


def test_hbn_valence_top_is_p_deep_band_is_s_on_the_same_atom(hbn_calculation):
    """Monolayer h-BN, on nitrogen, at K = (1/3, 1/3, 0): the occupied pi
    (valence-top) band is a N-2p_z state -- l=1 (p) should hold essentially
    all of N's muffin-tin weight there, s/d/f small by comparison -- while
    a much deeper (bonding sigma-type) valence band is N-2s dominated
    instead, l=0 now the largest. This is a sign-of-the-effect claim about
    WHICH l dominates on WHICH band of the SAME atom (the dominant channel
    flips between the two bands), not a plausibility band, the same spirit
    as test_calculation_atom_projection.py's N/B valence/conduction check.
    Measured: top band N (s,p,d,f) approx (0, 0.522, 0, 0); deep band N
    approx (0.534, 0, 0.0005, 0.0004)."""
    calc, ist0, ist1 = hbn_calculation
    n_index = calc.structure.atom_index("N")
    K = (1 / 3, 1 / 3, 0)

    orb_top = calc.get_orbital_projection(K, ist0=ist1, ist1=ist1)
    # matrices[n_index, :, 0, 0] already selects the (0, 0) diagonal entry
    # for each of the 4 l channels -- a length-4 vector, not a matrix to
    # further np.diag() (that would build a spurious 4x4 diagonal matrix).
    s_top, p_top, d_top, f_top = np.real(orb_top.matrices[n_index, :, 0, 0])
    assert p_top > s_top and p_top > d_top and p_top > f_top, (s_top, p_top, d_top, f_top)
    assert p_top > 0.5 * (s_top + p_top + d_top + f_top)  # p alone carries most of N's weight

    orb_deep = calc.get_orbital_projection(K, ist0=1, ist1=1)
    s_deep, p_deep, d_deep, f_deep = np.real(orb_deep.matrices[n_index, :, 0, 0])
    assert s_deep > p_deep and s_deep > d_deep and s_deep > f_deep, (
        s_deep, p_deep, d_deep, f_deep,
    )
    assert s_deep > 0.5 * (s_deep + p_deep + d_deep + f_deep)
    assert p_top > s_top and s_deep > p_deep  # the dominant channel flips between the two bands
