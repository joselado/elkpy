"""Integration test for Calculation.get_berry_curvature() (Wilson-loop /
Fukui-Hatsugai-Suzuki Berry curvature, task 9000), backed by the elkpy
Fortran extension in patches/0002-berry-curvature-wilson-loop.patch.

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.
"""

import pytest

from elkpy import config
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


def test_si_valence_manifold_chern_number_is_zero(si_calculation):
    """Bulk Si is a trivial (non-topological) insulator: the Chern number of
    its 4 valence bands must be exactly 0 on every k3 slice -- a nonzero
    integer here would mean a bug (wrong sign/convention), not real physics
    (see docs/design.md #13). Also checks the FHS admissibility diagnostic
    (max_flux) stays well under pi, i.e. this mesh is fine enough to trust."""
    result = si_calculation.get_berry_curvature(1, 4, directions=(1, 2))
    for chern in result["chern_number"]:
        assert chern == pytest.approx(0.0, abs=1e-6)
    assert result["max_flux"] < 0.5


def test_gap_check_rejects_ungapped_window(si_calculation):
    """Bands 2-4 are triply degenerate at Gamma (Si's p-like valence band
    maximum, no SOC). A window ending at band 3 splits that degeneracy --
    the gap to band 4 is ~1e-8 Ha there, far below the default tolerance --
    and should raise rather than silently return a meaningless result."""
    with pytest.raises(ValueError, match="gapped"):
        si_calculation.get_berry_curvature(1, 3, directions=(1, 2))


def test_path_curvature_converges_with_dk(si_calculation):
    """get_berry_curvature_path() (task 9001, small Wilson loop at an
    arbitrary point via fresh on-the-fly diagonalisation -- no periodic mesh,
    no reducek requirement, unlike get_berry_curvature()) exercised at an
    arbitrary k-point that isn't part of the ground state's own ngridk=(4,4,4)
    mesh at all -- confirming the path mode's fresh diagonalisation is
    independent of the ground state's sampling. There's no zero-curvature
    prediction to check at a generic point (only the *integrated* Chern
    number of a trivial insulator must vanish, not the local curvature
    everywhere), so the correctness signal here is dk-convergence: two
    different loop sizes at the same point should agree to within their own
    truncation error, not diverge."""
    kpoint = (0.05, 0.11, 0.03)
    coarse = si_calculation.get_berry_curvature_path([kpoint], 1, 4, directions=(1, 2), dk=0.02)[0]
    fine = si_calculation.get_berry_curvature_path([kpoint], 1, 4, directions=(1, 2), dk=0.01)[0]
    assert coarse["curvature"] == pytest.approx(fine["curvature"], abs=5e-3)


def test_path_and_mesh_conventions_agree(si_calculation):
    """Cross-check between the two Fortran code paths (task 9000's periodic
    mesh vs task 9001's fresh on-the-fly diagonalisation), on literally the
    same four k-points: for ngridk=(2,2,2), the mesh plaquette anchored at
    Gamma visits corners Gamma, Gamma+e1, Gamma+e1+e2, Gamma+e2 (e_d =
    1/ngridk(d) = 0.5) in that cyclic order. Centering a path loop at
    k0=(0.25,0.25,0) with dk=0.25 visits corner1=k0-dx-dy=Gamma,
    corner2=k0+dx-dy=Gamma+e1, corner3=k0+dx+dy=Gamma+e1+e2,
    corner4=k0-dx+dy=Gamma+e2 -- the identical four points in the identical
    order. Both derive M(a,b) = <psi_a|psi_b> from genolpq, one via a
    periodic-mesh eigenvector lookup, the other via fresh diagonalisation --
    agreement here confirms both Fortran code paths compute the same
    physical quantity, not just that each is internally self-consistent --
    exact bit-for-bit agreement isn't expected (one reads a converged
    eigenvector from a mesh diagonalisation, the other solves fresh), so the
    tolerance is loose enough to pass on ordinary numerical noise but tight
    enough that a real convention bug (wrong sign, transposed indices, wrong
    corner order) -- which would flip the sign or change the value by an
    order of magnitude -- would still fail it."""
    ngridk = (2, 2, 2)
    mesh = si_calculation.get_berry_curvature(1, 4, directions=(1, 2), ngridk=ngridk)
    path = si_calculation.get_berry_curvature_path([(0.25, 0.25, 0.0)], 1, 4, directions=(1, 2), dk=0.25)
    assert path[0]["flux"] == pytest.approx(mesh["flux"][0, 0, 0], abs=1e-4)


def test_path_kpath_keyword_matches_explicit_kpoints(si_calculation):
    """get_berry_curvature_path(kpath=...) (ASE symbolic path, same
    convention as get_bands()/get_phonon_dispersion()) must be equivalent to
    resolving the same points by hand and passing kpoints= directly -- this
    is purely a convenience wrapper around _kpath_to_points(), so this pins
    that it doesn't silently reorder/mis-discretize points, not any new
    physics. Also checks that each returned point carries a "distance" entry
    (absent when kpoints= is used directly, see the other tests above)."""
    pytest.importorskip("ase")
    from_kpath = si_calculation.get_berry_curvature_path(
        ist0=1, ist1=4, directions=(1, 2), dk=0.02, kpath="GXW", npoints=6, label="kpath_check"
    )
    explicit_kpoints, distances = si_calculation._kpath_to_points("GXW", 6)
    from_kpoints = si_calculation.get_berry_curvature_path(
        list(explicit_kpoints), 1, 4, directions=(1, 2), dk=0.02, label="kpath_check_explicit"
    )
    assert len(from_kpath) == len(explicit_kpoints)
    for got, expected_k, expected_d, ref in zip(from_kpath, explicit_kpoints, distances, from_kpoints):
        assert got["k"] == pytest.approx(expected_k)
        assert got["distance"] == pytest.approx(expected_d)
        assert got["flux"] == pytest.approx(ref["flux"])
        assert "distance" not in ref


def test_path_kpath_and_kpoints_are_mutually_exclusive(si_calculation):
    with pytest.raises(ValueError, match="exactly one"):
        si_calculation.get_berry_curvature_path(
            kpoints=[(0.0, 0.0, 0.0)], kpath="GXW", ist0=1, ist1=4
        )
    with pytest.raises(ValueError, match="exactly one"):
        si_calculation.get_berry_curvature_path(ist0=1, ist1=4)
