"""Integration test for Calculation.eigenstate_session()/get_eigenstates()/
get_overlap() (interactive eigenstate/overlap query session, task 9002),
backed by the elkpy Fortran extension in
patches/0003-eigenstate-session.patch.

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.
"""

import numpy as np
import pytest

from elkpy import config, spec
from elkpy.parsers import berry
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}


@pytest.fixture
def si_calculation(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    return s.get_calculation(tmp_path / "si", xc="PW", ngridk=(4, 4, 4))


def test_evecsv_is_orthonormal(si_calculation):
    """evecsv is built from an already-orthonormalized first-variational
    basis (docs/design.md #14) -- evecsv^H @ evecsv must be the identity to
    ordinary numerical precision, independent of any reference data."""
    state = si_calculation.get_eigenstates((0.1, 0.2, 0.05))
    nstsv = state.evecsv.shape[0]
    gram = state.evecsv.conj().T @ state.evecsv
    assert gram == pytest.approx(np.eye(nstsv), abs=1e-8)


def test_overlap_with_itself_is_identity(si_calculation):
    """<psi_a(k)|psi_b(k)> for the same k on both sides must be the
    identity matrix -- a sanity check requiring no external reference.
    Tolerance is looser than machine precision: genwfsvp/genolpq's
    muffin-tin-plus-interstitial real-space expansion has its own inherent
    truncation error (angular momentum cutoff, interstitial G-vector
    cutoff) at ordinary rgkmax, observed here at the ~1e-3 level even for a
    single non-degenerate band at a generic k-point -- this checks the
    result is close to the identity, not bit-exact."""
    with si_calculation.eigenstate_session() as session:
        m = session.overlap((0.1, 0.2, 0.05), (0.1, 0.2, 0.05), ist0=1, ist1=4)
    assert m == pytest.approx(np.eye(4), abs=2e-3)


def test_session_survives_many_queries(si_calculation):
    """The whole point of eigenstate_session(): repeated queries through one
    session return self-consistent results without needing to restart the
    process between them."""
    kpoints = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.1, 0.0), (0.1, 0.1, 0.1)]
    with si_calculation.eigenstate_session() as session:
        first_pass = [session.get_eigenstates(k).energies[:4] for k in kpoints]
        second_pass = [session.get_eigenstates(k).energies[:4] for k in kpoints]
    for a, b in zip(first_pass, second_pass):
        assert a == pytest.approx(b, abs=1e-10)


def test_overlap_matches_mesh_based_berry_export(si_calculation):
    """Cross-check between two independent Fortran code paths: task 9002's
    fresh-diagonalisation OVERLAP query vs. task 9000's periodic-mesh export
    (src/elkpy_berry.f90's elkpy_berrycurv), on literally the same pair of
    k-points -- Gamma and its ngridk=(2,2,2) mesh neighbour in direction 1,
    e_1=(0.5,0,0). Both derive M(a,b)=<psi_a|psi_b> via genolpq, one from a
    fresh on-the-fly diagonalisation, the other from a periodic-mesh
    eigenvector lookup -- agreement confirms both compute the same physical
    quantity (same spirit as
    test_calculation_berry.py::test_path_and_mesh_conventions_agree). Grid
    index (0,0,0) corresponding to Gamma is the same assumption that
    existing test already relies on.

    Deliberately checks only band 1 (a single, non-degenerate state), not
    the full occupied window: bands 2-4 are triply degenerate at Gamma (see
    test_calculation_berry.py::test_gap_check_rejects_ungapped_window), so
    the two *independent* diagonalisations here (mesh-cached vs. fresh) are
    free to -- and empirically do -- pick different, equally valid unitary
    bases within that degenerate subspace; comparing raw matrix elements of
    a degenerate window between two independent diagonalisations is not
    meaningful (only gauge-invariant quantities built from the whole
    window, e.g. the FHS flux in test_calculation_berry.py, are). Band 1
    alone has no such ambiguity.
    """
    mesh_calc = si_calculation.structure.get_calculation(
        si_calculation.workdir.parent / "si_mesh", xc="PW", ngridk=(2, 2, 2)
    )
    subdir = mesh_calc.run_tasks(
        [spec.TASKS["berry_curvature"]],
        blocks={"elkpy_berry": [(1, 2, 1, 4)], "reducek": [0]},
    )
    parsed = berry.parse_berry_overlaps(subdir / spec.OUTPUT_FILES["berry"])
    mesh_overlap = parsed["overlaps"][(0, 0, 0)][0][0, 0]  # band 1 only, Gamma -> Gamma+e1

    fresh_overlap = mesh_calc.get_overlap((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), ist0=1, ist1=1)

    assert fresh_overlap[0, 0] == pytest.approx(mesh_overlap, abs=1e-3)
