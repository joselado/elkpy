"""Integration tests for Calculation.get_atom_projection()/
EigenstateSession.atom_projection() (task 9002's PROJECTION query, backed by
elkpy_atomproj in src/elkpy_eigenstates.f90 -- patches/0004-atom-projection.patch),
against a real compiled elk binary.

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
    # test_calculation_quantum_geometry.py's hbn_calculation fixture.
    first_block = (calc.workdir / "EIGVAL.OUT").read_text().split("k-point")[1]
    state_line = re.compile(r"^\s*\d+\s+(\S+)\s+(\S+)\s*$")
    occ = [float(m.group(2)) for m in map(state_line.match, first_block.splitlines()) if m]
    ist0, ist1 = 1, sum(o > 1.0 for o in occ)
    return calc, ist0, ist1


def test_matrices_are_hermitian_and_positive_semidefinite(si_calculation):
    """Each atom's P_alpha is a weighted Gram matrix (wr2cmt is a positive
    quadrature weight, see elkpy_atomproj's docstring) -- Hermitian and PSD
    by construction, independent of any external reference data."""
    proj = si_calculation.get_atom_projection((0.1, 0.2, 0.05), ist0=1, ist1=4)
    natmtot, nst, _ = proj.matrices.shape
    assert natmtot == 2  # two Si atoms
    for a in range(natmtot):
        m = proj.matrices[a]
        assert m == pytest.approx(m.conj().T, abs=1e-8)
        eigvals = np.linalg.eigvalsh(m)
        assert (eigvals > -1e-8).all(), f"atom {a} not PSD: {eigvals}"


def test_atom_weights_sum_below_identity(si_calculation):
    """Sigma_alpha P_alpha + P_interstitial = <psi_i|psi_j> = identity: the
    muffin-tin-plus-interstitial VOLUME partition is exact (no volume is
    double-counted or omitted -- see elkpy_atomproj's docstring), though the
    wavefunction expansion inside each muffin tin is still only as accurate
    as wfmtsv's own angular-momentum cutoff (lmaxo) and radial quadrature
    (wr2cmt on the lradstp-coarsened mesh) -- the same cutoffs dos.f90/
    bandstr.f90 already accept for their own atom-projected output, not the
    separate genolpq real-space-expansion truncation get_overlap() carries
    (docs/design.md #14). So the identity minus the sum over every atom in
    the cell must be Hermitian PSD up to that ordinary basis-truncation
    error (the interstitial region's own weight can't be negative), and
    each atom's total weight (diagonal, summed) should be a substantial,
    physically reasonable fraction of the cell for a tetrahedral solid like
    Si, not near zero."""
    proj = si_calculation.get_atom_projection((0.1, 0.2, 0.05), ist0=1, ist1=4)
    total = proj.matrices.sum(axis=0)
    nst = total.shape[0]
    remainder = np.eye(nst) - total
    assert remainder == pytest.approx(remainder.conj().T, abs=1e-8)
    eigvals = np.linalg.eigvalsh(remainder)
    assert (eigvals > -1e-6).all(), f"atoms overshoot the identity: {eigvals}"
    diag = np.real(np.diag(total))
    assert (diag > 0.3).all() and (diag < 1.0).all(), diag


def test_si_atoms_are_symmetry_equivalent_at_gamma(si_calculation):
    """Diamond Si's two atoms are related by inversion through the bond
    midpoint; inversion sends k -> -k, and Gamma is its own image, so the
    two atoms' weight on the single non-degenerate band 1 must be equal at
    Gamma (bands 2-4 are degenerate there, same caveat
    test_calculation_eigenstates.py's mesh/fresh cross-check already notes,
    so this check deliberately uses band 1 alone)."""
    proj = si_calculation.get_atom_projection((0.0, 0.0, 0.0), ist0=1, ist1=1)
    w0 = proj.matrices[0][0, 0].real
    w1 = proj.matrices[1][0, 0].real
    assert w0 == pytest.approx(w1, rel=1e-6)


def test_diagonal_matches_bandstr_total_atomic_character(si_calculation):
    """Cross-check against an entirely independent Fortran code path: task
    21's own atom-projected band-character output (bandstr.f90, upstream,
    unmodified -- BAND_Sss_Aaaaa.OUT's "total atomic character" column,
    'sm=sum(bc(0:lmaxdb,...))'), which calls the very same gendmatk/wfmtsv
    machinery elkpy_atomproj's docstring describes reusing, but via a
    completely separate call site (Elk's own mesh/path diagonalisation,
    not elkpy_atomproj's fresh on-the-fly one) and a completely separate
    reduction (an explicit per-l loop, not this feature's zgemm). Agreement
    here catches things the Hermitian/PSD and sum-below-identity checks
    above can't: those are satisfied by ANY consistent normalization, even
    one that silently undercounts weight (e.g. a packing/stride bug in the
    zgemm reduction) -- only an external reference computing the same
    number a different way can catch that.

    task 21's default lmaxdb=3 truncates l earlier than wfmtsv's own
    lmaxo=6 the ground state actually used, so lmaxdb is raised to 6 here
    to make this an exact comparison, not merely a close one. The path's
    first point is exactly its first vertex (Elk's own plotpt1d.f90,
    dp(1)=dv(1)=0, vpl(:,1)=vvl(:,1)), so a 2-vertex, 2-point plot1d path
    starting at the same k as the PROJECTION query below gives literally
    the same k-point through both code paths -- band 1 only, since
    bandstr.f90 does not gap-check for degeneracy."""
    k0 = (0.1, 0.2, 0.05)
    k1 = (0.15, 0.2, 0.05)
    subdir = si_calculation.run_tasks(
        [21], blocks={"plot1d": [(2, 2), k0, k1], "lmaxdb": [6]}, label="bandcheck"
    )
    lines = {}
    for symbol_index, fname in [(0, "BAND_S01_A0001.OUT"), (1, "BAND_S01_A0002.OUT")]:
        first_line = (subdir / fname).read_text().splitlines()[0]
        lines[symbol_index] = float(first_line.split()[2])  # column 3: total atomic character

    proj = si_calculation.get_atom_projection(k0, ist0=1, ist1=1)
    for atom_index, bandstr_weight in lines.items():
        assert proj.matrices[atom_index][0, 0].real == pytest.approx(bandstr_weight, abs=1e-5)


def test_spin_polarized_matrices_are_hermitian_psd(tmp_path):
    """The zgemm reduction accumulates over nspinor spin channels
    (elkpy_atomproj's `do ispn=1,nspinor` loop) -- untested by the
    unpolarized si_calculation fixture above, where nspinor=1 and that
    loop never runs twice. A spin-polarized (nspinor=2) run is enough to
    exercise the accumulation across both channels; SOC (soc_scale,
    patches/0001) is a separate, still-untested axis, not covered here."""
    calc = Structure(SI_AVEC, SI_SPECIES).get_calculation(
        tmp_path / "si_spinpol", xc="PW", ngridk=(2, 2, 2), spinpol=True
    )
    proj = calc.get_atom_projection((0.1, 0.2, 0.05), ist0=1, ist1=4)
    for a in range(proj.matrices.shape[0]):
        m = proj.matrices[a]
        assert m == pytest.approx(m.conj().T, abs=1e-8)
        assert (np.linalg.eigvalsh(m) > -1e-8).all()
    remainder = np.eye(proj.matrices.shape[1]) - proj.matrices.sum(axis=0)
    assert (np.linalg.eigvalsh(remainder) > -1e-6).all()


def test_hbn_valence_is_nitrogen_conduction_is_boron_at_k(hbn_calculation):
    """Monolayer h-BN: the occupied pi (valence-top) band is dominated by
    the more electronegative N's 2pz orbital, the unoccupied pi* (conduction-
    bottom) band by B's -- a sharp, sign-of-the-effect prediction (not a
    plausibility band), same spirit as the K/K' Berry-curvature-antisymmetry
    check in test_calculation_berry.py and the quantum-metric K/K' parity
    check in test_calculation_quantum_geometry.py. K = (1/3, 1/3, 0)."""
    calc, ist0, ist1 = hbn_calculation
    n_index = calc.structure.atom_index("N")
    b_index = calc.structure.atom_index("B")

    proj = calc.get_atom_projection((1 / 3, 1 / 3, 0), ist0=ist1, ist1=ist1 + 1)
    valence_row, conduction_row = 0, 1  # ist1, ist1+1 in that order

    n_valence = proj.matrices[n_index][valence_row, valence_row].real
    b_valence = proj.matrices[b_index][valence_row, valence_row].real
    n_conduction = proj.matrices[n_index][conduction_row, conduction_row].real
    b_conduction = proj.matrices[b_index][conduction_row, conduction_row].real

    assert n_valence > b_valence, (n_valence, b_valence)
    assert b_conduction > n_conduction, (b_conduction, n_conduction)


def test_kpoints_batch_matches_per_point_calls(si_calculation):
    """kpoints=[...] opens ONE session and reuses it across every point
    (see _session_query_path()'s docstring) -- pin that this is purely a
    plumbing change, not a numerical one, by checking it reproduces the
    same matrices as calling get_atom_projection(k, ...) once per point
    (each of which opens/closes its own session)."""
    kpoints = [(0.1, 0.2, 0.05), (0.0, 0.0, 0.0), (0.3, 0.1, 0.0)]
    batch = si_calculation.get_atom_projection(kpoints=kpoints, ist0=1, ist1=1)
    assert len(batch) == len(kpoints)
    for point, k in zip(batch, kpoints):
        assert point["k"] == pytest.approx(k)
        single = si_calculation.get_atom_projection(k, ist0=1, ist1=1)
        assert point["matrices"] == pytest.approx(single.matrices, abs=1e-8)


def test_kpath_keyword_matches_explicit_kpoints(si_calculation):
    """get_atom_projection(kpath=...) (ASE symbolic path) resolves through
    the same _kpath_to_points() every other kpath= consumer uses (see
    get_berry_curvature_path()'s equivalent test) and attaches a
    "distance" entry per point."""
    pytest.importorskip("ase")
    from_kpath = si_calculation.get_atom_projection(
        ist0=1, ist1=1, kpath="GXW", npoints=6, label="kpath_check"
    )
    explicit_kpoints, distances = si_calculation._kpath_to_points("GXW", 6)
    from_kpoints = si_calculation.get_atom_projection(
        kpoints=list(explicit_kpoints), ist0=1, ist1=1, label="kpath_check_explicit"
    )

    assert len(from_kpath) == len(explicit_kpoints)
    for got, expected_k, expected_d, ref in zip(
        from_kpath, explicit_kpoints, distances, from_kpoints
    ):
        assert got["k"] == pytest.approx(tuple(expected_k))
        assert got["distance"] == pytest.approx(expected_d)
        assert got["matrices"] == pytest.approx(ref["matrices"])


def test_kpath_and_kpoints_are_mutually_exclusive(si_calculation):
    with pytest.raises(ValueError):
        si_calculation.get_atom_projection(
            kpoints=[(0.0, 0.0, 0.0)], kpath="GXW", ist0=1, ist1=1
        )
