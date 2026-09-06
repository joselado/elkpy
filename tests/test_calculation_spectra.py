"""Integration tests for elkpy.tasks.spectra, running the real Elk binary.

Skipped if the binary hasn't been built (see docs/design.md #8). Same
self-skip pattern as tests/test_calculation_si.py.

FIXTURE CHOICE. Most of this bucket is exercised on face-centred-cubic
ALUMINIUM (one atom per cell -- cheaper than bulk Si) rather than on Si,
because half of it is meaningless for an insulator: a Fermi surface, a
Fermi-surface nesting function and a smeared Fermi-level delta are all
degenerate when no band crosses E_F. Al is the textbook nearly-free-electron
metal, so its Fermi surface is a genuine, non-trivial object. A small bcc Fe
fixture covers the magnetic-only tasks (magnetisation, B_xc, spin-resolved
partial DOS), which need a spin-polarised ground state.

The checks that carry real weight, rather than merely asserting shapes:

* ``IDOS + sum(PDOS) == TDOS`` to ~1e-9. This is an EXACT identity, not a
  tolerance: src/dos.f90 subtracts each muffin-tin channel from the running
  total as it computes it, so the file called IDOS.OUT holds the
  interstitial remainder. It pins the partial-DOS block ordering, the
  spin-sign convention and the meaning of IDOS all at once.
* ``total character == sum over l`` in the band-character file, to the F12.6
  rounding limit. Mistaking bandstr.f90's third column for the l=0 channel
  would pass every shape check while inflating every reported s weight by
  the whole muffin-tin total.
* ``N(q) <= N(0)`` for the nesting function. Cauchy-Schwarz on
  N(q) = sum_k n_k(E_F) n_{k+q}(E_F): the maximum is at q = 0, always.
* FERMISURF.OUT and FERMISURF.bxsf must agree POINTWISE even though the two
  Fortran writers run their loops in OPPOSITE orders (i1 fastest in the
  first, i3 fastest in the second). Run on an ASYMMETRIC grid so that a
  permuted reshape is a shape error rather than a silent no-op.
* ``L = S = J = 0`` for non-magnetic Al, and ``0 <= ELF <= 1``.
"""

import numpy as np
import pytest

from elkpy import config
from elkpy.calculation import Calculation
from elkpy.parsers import plots
from elkpy.tasks.spectra import SpectraTasks
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)


# Calculation now inherits SpectraTasks (see elkpy/tasks/__init__.py), so the
# alias IS the wired class -- and a ``(SpectraTasks, Calculation)`` subclass
# would now be an inconsistent MRO rather than a redundant one.
SpectraCalculation = Calculation


# fcc Al, a = 4.05 Angstrom = 7.6532 Bohr, primitive vectors a/2 * (110) etc.
AL = 3.8266
AL_AVEC = [(AL, AL, 0.0), (AL, 0.0, AL), (0.0, AL, AL)]

# bcc Fe, a = 5.42 Bohr, primitive vectors a/2 * (-1 1 1) etc.
FE_AVEC = [(-2.71, 2.71, 2.71), (2.71, -2.71, 2.71), (2.71, 2.71, -2.71)]

LINE = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
PLANE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
# deliberately asymmetric so an (n1, n2) / (n2, n1) mix-up is a shape error
GRID2D = (10, 12)


@pytest.fixture(scope="module")
def al(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("al")
    s = Structure(AL_AVEC, {"Al": [(0.0, 0.0, 0.0)]})
    return SpectraCalculation(
        s, workdir / "al", xc="PW", ngridk=(4, 4, 4), rgkmax=6.0,
        extra_blocks={"maxscl": [30]},
    )


@pytest.fixture(scope="module")
def fe(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("fe")
    s = Structure(FE_AVEC, {"Fe": [((0.0, 0.0, 0.0), (0.0, 0.0, 4.0))]})
    # no maxscl cap: a magnetic ground state that has not converged raises
    # and would take every Fe test below down with it
    return SpectraCalculation(
        s, workdir / "fe", xc="PW", spinpol=True, ngridk=(4, 4, 4), rgkmax=6.0,
    )


# ------------------------------------------------------------------
# the plotting triples: 1D and 2D members
# ------------------------------------------------------------------


def test_density_1d(al):
    result = al.get_density_1d(line=LINE, npoints=40)
    assert result["distance"].shape == (40,)
    assert result["values"].shape == (40,)
    assert (result["values"] > 0).all(), "a charge density is positive everywhere"
    # the line starts on the nucleus, so the density is largest there
    assert result["values"].argmax() == 0
    assert result["vertices"] == pytest.approx([0.0, result["distance"][-1]], rel=1e-3)


def test_density_2d_grid_order(al):
    """The image reshapes as (n2, n1); an asymmetric grid makes a transposed
    reshape a hard error (src/plotpt2d.f90 runs i1 innermost)."""
    result = al.get_density_2d(plane=PLANE, grid=GRID2D)
    assert result["grid"] == GRID2D
    assert result["values"].shape == (GRID2D[0] * GRID2D[1],)
    assert result["values_grid"].shape == (GRID2D[1], GRID2D[0])
    assert (result["values"] > 0).all()


def test_potential_components_differ(al):
    """Task 41 writes VCL1D.OUT and VXC1D.OUT in one run; `component` only
    selects which is parsed."""
    coulomb = al.get_potential_1d(component="coulomb", line=LINE, npoints=30)
    xc = al.get_potential_1d(component="xc", line=LINE, npoints=30)
    # v_C carries the nuclear -Z/r divergence; v_xc is a few Hartree at most
    assert coulomb["values"].min() < -100.0
    assert -50.0 < xc["values"].min() < 0.0
    assert xc["values"].max() < 0.0, "the LDA v_xc is negative everywhere"


def test_potential_2d(al):
    result = al.get_potential_2d(component="xc", plane=PLANE, grid=GRID2D)
    assert result["values_grid"].shape == (GRID2D[1], GRID2D[0])


def test_elf_is_bounded_to_the_unit_interval(al):
    """f_ELF = 1/(1 + (D/D0)^2) is in [0, 1] by construction."""
    result = al.get_elf_1d(line=LINE, npoints=30)
    assert (result["values"] >= 0.0).all()
    assert (result["values"] <= 1.0).all()


def test_elf_2d(al):
    result = al.get_elf_2d(plane=PLANE, grid=GRID2D)
    assert result["values_grid"].shape == (GRID2D[1], GRID2D[0])
    assert (result["values"] >= 0.0).all() and (result["values"] <= 1.0).all()


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_wavefunction_modulus_is_non_negative(al, dim):
    result = al.get_wavefunction(
        ik=1, ist=1, dim=dim, line=LINE, npoints=20, plane=PLANE,
        grid=GRID2D if dim == 2 else (6, 6, 6),
    )
    assert (result["values"] >= -1e-10).all(), "|psi|^2 cannot be negative"


def test_stm_differs_from_a_single_state_plot(al):
    """Task 162 replaces the occupations with a smeared delta at E_F, so its
    image is a different function of position from any one state's."""
    stm = al.get_stm(plane=PLANE, grid=GRID2D)
    wf = al.get_wavefunction(ik=1, ist=1, dim=2, plane=PLANE, grid=GRID2D)
    assert stm["values_grid"].shape == (GRID2D[1], GRID2D[0])
    assert not np.allclose(stm["values"], wf["values"])


def test_electric_field_is_a_three_component_field(al):
    result = al.get_electric_field(dim=1, line=LINE, npoints=30)
    assert result["values"].shape == (30, 3)
    # E = -grad v_C diverges as Z/r^2 at the nucleus
    assert np.abs(result["values"]).max() > 1.0


def test_paramagnetic_current_vanishes_without_a_field(al):
    """j_p is odd under time reversal, so a time-reversal-symmetric ground
    state with no applied vector potential carries none.

    CAVEAT: src/genjpr.f90 also returns exactly zero when ``iscl < 1``
    (no wavefunctions yet), so this assertion can pass for the wrong
    reason. It is here for the shape and for a gross-regression bound, not
    as a sharp physics check."""
    result = al.get_paramagnetic_current(dim=1, line=LINE, npoints=20)
    assert result["values"].shape == (20, 3)
    assert np.isfinite(result["values"]).all()
    assert np.abs(result["values"]).max() < 1e-3


def test_core_wavefunctions_are_normalised(al):
    """Elk stores u(r) = r R(r), so int |u|^2 dr = 1 with no r^2 Jacobian."""
    cores = al.get_core_wavefunctions()
    r, u = cores[("Al", 1)]
    assert u.shape[0] >= 1
    for row in u:
        assert np.trapezoid(row**2, r) == pytest.approx(1.0, abs=5e-3)


# ------------------------------------------------------------------
# band character and partial DOS
# ------------------------------------------------------------------


def test_band_character_total_is_the_sum_over_l(al):
    """bandstr.f90 writes ``sm = sum(bc(0:lmaxdb,...))`` as the third column,
    BEFORE the per-l ones -- an exact identity to the F12.6 rounding limit."""
    result = al.get_band_character(kind="l", vertices=LINE, npoints=20)
    atom = result[("Al", 1)]
    assert atom["energies"].shape[1] == 20
    assert atom["lmaxdb"] == 3  # Elk's default: s, p, d, f
    np.testing.assert_allclose(
        atom["total"], atom["l"].sum(axis=2), rtol=0, atol=2e-6
    )
    # a muffin-tin weight is a fraction of one state
    assert (atom["total"] >= -1e-9).all() and (atom["total"] <= 1.0 + 1e-9).all()


def test_band_character_energies_match_the_plain_band_structure(al):
    """BAND.OUT (task 20) and BAND_Sss_Aaaaa.OUT (task 21) write the same
    ``evalsv - efermi`` from two separate branches of bandstr.f90."""
    distances, energies = al.get_bands(vertices=LINE, npoints=20)
    result = al.get_band_character(kind="l", vertices=LINE, npoints=20)
    atom = result[("Al", 1)]
    np.testing.assert_allclose(atom["distances"], distances, rtol=1e-8)
    np.testing.assert_allclose(atom["energies"], energies, atol=1e-8)


def test_band_character_lm_channel_count(al):
    result = al.get_band_character(kind="lm", vertices=LINE, npoints=10)
    atom = result[("Al", 1)]
    assert atom["characters"].shape[2] == 16  # (lmaxdb + 1)^2 with lmaxdb = 3
    assert atom["lmaxdb"] == 3


def test_partial_dos_sums_to_the_total(al):
    """THE identity: src/dos.f90 subtracts every muffin-tin channel from the
    running total, so IDOS.OUT is the interstitial remainder and
    IDOS + sum(PDOS) reconstructs TDOS exactly."""
    result = al.get_partial_dos(nwplot=200, wplot=(-0.5, 0.5))
    assert result["nspin"] == 1
    assert result["channels"] == "l"
    partial = result["partial"][("Al", 1)]
    assert partial.shape[0] == 1
    assert partial.shape[1] == 4  # s, p, d, f with dosmsum
    reconstructed = result["interstitial"][0] + partial[0].sum(axis=0)
    scale = np.abs(result["total"][0]).max()
    np.testing.assert_allclose(
        reconstructed, result["total"][0], atol=1e-6 * max(scale, 1.0)
    )


# ------------------------------------------------------------------
# Fermi surfaces and nesting -- a metal is essential here
# ------------------------------------------------------------------


def test_fermi_surface_bands_form(al):
    result = al.get_fermi_surface(kind="bands", grid=(4, 4, 4))
    surface = result["total"]
    assert surface["per_band"] is True
    assert surface["grid"] == (4, 4, 4)
    assert surface["nstates"] >= 1, "aluminium is a metal: some band crosses E_F"
    for i in range(surface["nstates"]):
        column = surface["values"][:, i]
        assert column.min() < 0.0 < column.max(), (
            "a band Elk selected as crossing E_F must change sign on the grid"
        )


def test_fermi_surface_product_form(al):
    """Task 100's scalar is prod_n (e_n - E_F) over the crossing bands, so
    its ZERO isosurface is the Fermi surface."""
    result = al.get_fermi_surface(kind="product", grid=(4, 4, 4))
    surface = result["total"]
    assert surface["per_band"] is False
    assert surface["values"].shape == (64,)
    assert surface["values"].min() < 0.0 < surface["values"].max()


def test_fermi_surface_delta_form_is_non_negative(al):
    """Task 103 sums a smeared delta over ALL bands, so it cannot go
    negative for Elk's default Fermi-Dirac smearing (stype = 3)."""
    result = al.get_fermi_surface(kind="delta", grid=(4, 4, 4))
    values = result["total"]["values"]
    assert values.shape == (64,)
    assert (values >= 0.0).all()
    assert values.max() > 0.0


def test_fermi_surface_and_bxsf_agree_on_an_asymmetric_grid(al):
    """The sharpest available cross-check of BOTH axis orders at once:
    src/fermisurf.f90 writes with i1 fastest and src/fermisurfbxsf.f90 with
    i3 fastest, from two independent loops over the same eigenvalues. An
    ASYMMETRIC grid is essential -- on a cubic-symmetric mesh a permuted
    reshape gives numerically identical arrays and the check is vacuous.

    Both tasks index their values through the same ``ivkik`` map, so
    symmetry reduction cannot make them disagree; should ``genppts`` object
    to a non-cubic mesh on a cubic lattice, pass
    ``extra_blocks={"reducek": [0]}`` on the fixture."""
    assert all(abs(x) < 1e-12 for x in al.vkloff), (
        "task 102 forces vkloff = 0, so the two runs share a mesh only when "
        "the calculation's own offset is zero"
    )
    grid = (4, 5, 6)
    surface = al.get_fermi_surface(kind="bands", grid=grid)["total"]
    bxsf = al.get_fermi_surface_bxsf(grid=grid)["total"]

    assert bxsf["grid"] == (grid[0] + 1, grid[1] + 1, grid[2] + 1)
    assert len(bxsf["band_indices"]) == surface["nstates"]

    for i, _ist in enumerate(bxsf["band_indices"]):
        # (N,) with i1 fastest -> (n3, n2, n1) -> (n1, n2, n3)
        from_fermisurf = np.transpose(
            plots.reshape_plot3d(surface["values"][:, i], surface["grid"]), (2, 1, 0)
        )
        from_bxsf = bxsf["energies"][i][: grid[0], : grid[1], : grid[2]]
        assert from_fermisurf.shape == from_bxsf.shape
        np.testing.assert_allclose(from_fermisurf, from_bxsf, atol=1e-6)


def test_bxsf_repeats_the_first_plane(al):
    """The band grid is ngridk+1 per direction; the extra plane is the
    periodic image of the first, which is what a periodic isosurface
    needs."""
    bxsf = al.get_fermi_surface_bxsf(grid=(4, 4, 4))["total"]
    energies = bxsf["energies"]
    for axis in range(1, 4):
        first = np.take(energies, 0, axis=axis)
        last = np.take(energies, -1, axis=axis)
        np.testing.assert_allclose(first, last, atol=1e-10)


def test_nesting_function_is_maximal_at_zero_momentum_transfer(al):
    """N(q) = sum_k n_k(E_F) n_{k+q}(E_F) is a self-correlation of a
    non-negative function, so Cauchy-Schwarz puts its maximum at q = 0 --
    a genuine bound, not a tolerance."""
    result = al.get_nesting(ngridk=(4, 4, 4))
    assert result["grid"] == (4, 4, 4)
    assert (result["values"] >= 0.0).all()
    assert result["total"] > 0.0, "a metal has a nonzero nesting integral"
    assert result["values"].argmax() == 0
    assert result["values"][0] == pytest.approx(result["values"].max())


# ------------------------------------------------------------------
# expectation values and atomic data
# ------------------------------------------------------------------


def test_lsj_vanishes_for_a_non_magnetic_crystal(al):
    """Without spin polarisation or spin-orbit coupling, time reversal
    forces every on-site L, S and J to zero."""
    entries = al.get_lsj()
    assert len(entries) == 1
    entry = entries[0]
    assert (entry["species"], entry["symbol"], entry["atom"]) == (1, "Al", 1)
    for key in ("L", "S", "J"):
        assert np.abs(entry[key]).max() < 1e-8
    np.testing.assert_allclose(entry["J"], entry["L"] + entry["S"], atol=1e-12)


def test_lsj_states(al):
    entries = al.get_lsj_states(kstlist=[(1, 1), (1, 2)])
    assert len(entries) == 2  # one atom, two (k, state) pairs
    assert [e["ist"] for e in entries] == [1, 2]
    for entry in entries:
        np.testing.assert_allclose(entry["J"], entry["L"] + entry["S"], atol=1e-12)


def test_atomic_eigenvalues_show_the_free_atom_spin_orbit_pair(al):
    """ksp is |kappa| = j + 1/2, so an l > 0 shell appears twice: k = l
    (j = l - 1/2) below k = l + 1 (j = l + 1/2)."""
    species = al.get_atomic_eigenvalues()
    assert len(species) == 1
    states = species[0]["states"]
    assert species[0]["symbol"] == "Al"
    assert states[0]["n"] == 1 and states[0]["l"] == 0
    assert states[0]["energy"] < -50.0, "the Al 1s level is tens of Hartree deep"
    p_states = [s for s in states if s["l"] == 1 and s["n"] == 2]
    assert len(p_states) == 2
    lower, upper = sorted(p_states, key=lambda s: s["k"])
    assert (lower["k"], upper["k"]) == (1, 2)
    assert lower["energy"] < upper["energy"], "j = 1/2 lies below j = 3/2"


def test_smearing_functions(al):
    """The kernel every Fermi-level quantity in this bucket is built from:
    the delta integrates to 1 and the step rises monotonically from 0 to 1."""
    result = al.get_smearing_functions(nwplot=400)
    assert np.trapezoid(result["delta"], result["energy"]) == pytest.approx(
        1.0, abs=1e-3
    )
    assert result["theta"][0] < 1e-3
    assert result["theta"][-1] > 1.0 - 1e-3
    assert (np.diff(result["theta"]) >= -1e-12).all()


def test_elnes_at_zero_momentum_transfer(al):
    """Al's shallow core levels sit at about -2.6 Ha (2p) and -4.0 Ha (2s)
    -- see test_atomic_eigenvalues_show_the_free_atom_spin_orbit_pair -- so
    the loss window has to reach several Hartree to contain an edge at all.
    A window around zero returns a spectrum that is identically zero and
    makes the non-negativity assertion vacuous."""
    energies, cross_section = al.get_elnes(
        q=(0.0, 0.0, 0.0), wplot=(0.0, 6.0), nwplot=100
    )
    assert energies.shape == (100,)
    assert cross_section.shape == (100,)
    assert (cross_section >= -1e-12).all(), "a cross-section cannot be negative"
    assert cross_section.max() > 0.0, "the window must contain a real edge"


def test_electron_momentum_density(al):
    """rho(p) is non-negative and, being the momentum distribution of a
    nearly-free-electron metal, is largest at p = 0 (the start of a line
    through the origin)."""
    # hkmax and npoints kept small: the 1D form integrates over the two
    # perpendicular directions, so its cost is (2 max_i nh_i ngridk_i)^2
    # interpolations PER plotted point
    result = al.get_electron_momentum_density(
        dim=1, line=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], npoints=10, hkmax=4.0
    )
    assert result["values"].shape == (10,)
    assert (result["values"] >= -1e-10).all()
    assert result["values"].argmax() == 0


# ------------------------------------------------------------------
# magnetic vector fields -- need a spin-polarised ground state
# ------------------------------------------------------------------


def test_magnetisation_and_bxc_are_collinear_and_locked_in_lsda(fe):
    """Within a LOCAL spin-density functional B_xc(r) is a function of the
    local n and m alone, so it is strictly (anti)parallel to m at every
    point: for a collinear run both fields have only a z component, and the
    sign of m_z B_xc,z is the SAME everywhere m is appreciable.

    Which sign that is depends on whether Elk's ``bxcmt`` is the field
    entering the Kohn-Sham equation as +B.sigma or -B.sigma, a convention
    this test deliberately does not assert -- the convention-free content
    is the pointwise locking, which a non-local or mis-assembled field
    would break."""
    magnetisation = fe.get_magnetisation(dim=1, line=LINE, npoints=40)
    field = fe.get_bxc(dim=1, line=LINE, npoints=40)
    m = magnetisation["values"]
    b = field["values"]
    assert m.shape == (40, 3) and b.shape == (40, 3)
    # collinear run: Elk stores one component, written as (0, 0, m_z)
    np.testing.assert_allclose(m[:, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(b[:, :2], 0.0, atol=1e-12)
    strong = np.abs(m[:, 2]) > 0.1 * np.abs(m[:, 2]).max()
    assert strong.any(), "bcc Fe must carry a magnetisation somewhere"
    product = m[strong, 2] * b[strong, 2]
    assert (product > 0).all() or (product < 0).all()


def test_bxc_divergence_is_a_scalar_field(fe):
    result = fe.get_bxc_divergence(dim=1, line=LINE, npoints=30)
    assert result["values"].shape == (30,)
    assert np.abs(result["values"]).max() > 0.0


def test_magnetisation_2d_grid_order(fe):
    result = fe.get_magnetisation(dim=2, plane=PLANE, grid=GRID2D)
    assert result["values"].shape == (GRID2D[0] * GRID2D[1], 3)
    assert result["values_grid"].shape == (3, GRID2D[1], GRID2D[0])


def test_partial_dos_splits_the_two_spin_channels(fe):
    """With nsd = 2 the blocks run spin-slowest and Elk negates the
    spin-down one (sps(2) = -1); the parser flips it back, so both channels
    are non-negative and the identity against TDOS still holds per spin."""
    result = fe.get_partial_dos(nwplot=200, wplot=(-0.5, 0.5))
    assert result["nspin"] == 2
    partial = result["partial"][("Fe", 1)]
    assert partial.shape[:2] == (2, 4)
    # brzint's tetrahedron-style interpolation can undershoot slightly
    assert (partial >= -1e-6).all()
    for ispn in range(2):
        reconstructed = result["interstitial"][ispn] + partial[ispn].sum(axis=0)
        scale = np.abs(result["total"][ispn]).max()
        np.testing.assert_allclose(
            reconstructed, result["total"][ispn], atol=1e-6 * max(scale, 1.0)
        )


def test_fermi_surface_splits_into_up_and_down_for_collinear_magnetism(fe):
    """src/fermisurf.f90 writes FERMISURF_UP.OUT and FERMISURF_DN.OUT (and
    forces reducek = 0) whenever ndmag == 1."""
    assert fe._ndmag() == 1
    result = fe.get_fermi_surface(kind="bands", grid=(4, 4, 4))
    assert set(result["paths"]) == {"up", "dn"}
    assert "total" not in result
    for key in ("up", "dn"):
        assert result[key]["grid"] == (4, 4, 4)
        assert result[key]["nstates"] >= 1
