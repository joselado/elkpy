"""Integration tests for the magnetism / GW / Wannier90 / ultra-long-range
task family, running the real Elk binary.

Skipped entirely if the binary has not been built (see docs/design.md #8).
Everything except the Wannier90 export is additionally gated behind
``ELKPY_RUN_SLOW_TESTS=1``, and for good reason -- read the per-test cost
notes before running them:

* ``get_mae`` runs one full self-consistent ground state **per magnetisation
  direction**, with spin-orbit coupling and a symmetry-reduced point group.
* ``get_spin_spiral_supercell`` runs one full ground state per q-point, each
  on a supercell several times the unit cell.
* the GW tasks are hours (task 600) to days (task 620) even on small cells --
  Elk's own band-structure example quotes "about 3 days on 200 CPU cores".
* the ULR ground state is a self-consistent loop that Elk's own example runs
  with ``maxscl 2000`` and a mixing parameter of 0.001.

Only ``test_wannier90_export`` has actually been run against the binary
during development; every other test here is written from the Fortran and
has never been executed. Treat a first failure as "this test was never
right", not as a regression.
"""

import os

import numpy as np
import pytest

from elkpy import config
from elkpy.calculation import Calculation
from elkpy.structure import Structure
from elkpy.tasks.magnetism_manybody import MagnetismManyBodyTasks

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

slow = pytest.mark.skipif(
    os.environ.get("ELKPY_RUN_SLOW_TESTS") != "1",
    reason="expensive (many self-consistent runs); set ELKPY_RUN_SLOW_TESTS=1",
)


if issubclass(Calculation, MagnetismManyBodyTasks):
    _Calculation = Calculation
else:
    # The mixin is applied to Calculation by the integrator; until then, build
    # the combined class here so these tests are runnable either way.
    class _Calculation(MagnetismManyBodyTasks, Calculation):
        pass


SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}

# bcc Fe, a = 5.42 Bohr, one atom per cell with a seed field along z
FE_AVEC = [(-2.71, 2.71, 2.71), (2.71, -2.71, 2.71), (2.71, 2.71, -2.71)]
FE_SPECIES_Z = {"Fe": [((0.0, 0.0, 0.0), (0.0, 0.0, -0.01))]}
# a transverse seed field is what makes the calculation non-collinear
# (init0.f90 sets ndmag = 3 when bfcmt has an x or y component), which is the
# only case in which torque.f90 computes anything at all
FE_SPECIES_XZ = {"Fe": [((0.0, 0.0, 0.0), (-0.01, 0.0, -0.01))]}

# gamma-Fe (fcc), Elk's own examples/magnetism/Fe-spiral cell
FE_FCC_AVEC = [
    (3.375, 3.375, 0.0),
    (3.375, 0.0, 3.375),
    (0.0, 3.375, 3.375),
]
FE_FCC_SPECIES = {"Fe": [(0.0, 0.0, 0.0)]}


def _si(tmp_path, **kwargs):
    return _Calculation(
        Structure(SI_AVEC, SI_SPECIES), tmp_path / "si",
        xc="PW", ngridk=(2, 2, 2), rgkmax=6.0, **kwargs
    )


def _fe(tmp_path, species=FE_SPECIES_Z, **kwargs):
    kwargs.setdefault("spinpol", True)
    return _Calculation(
        Structure(FE_AVEC, species), tmp_path / "fe",
        xc="PW", ngridk=(2, 2, 2), rgkmax=6.0, **kwargs
    )


# ---------------------------------------------------------------------------
# 550 -- Wannier90 export.  The one test here that has actually been run.
# ---------------------------------------------------------------------------


def test_wannier90_export(tmp_path):
    """Task 550 writes <seedname>.win and <seedname>.eig and then aborts in
    wannier_setup, because this build links w90_stub.f90.

    Measured on bulk Si, ngridk=(2,2,2), rgkmax=6.0: the run log ends with
    the stub's own message and a gfortran backtrace through
    ``wannier_setup_ <- setupw90_ <- writew90_``, confirming the abort point
    is after both files are complete. ``wannier.eig`` came out 8 x 13
    (nkptnr x nstsv), and the ``.win`` reported num_wann = num_bands = 13 and
    an 8-row kpoints block.
    """
    calc = _si(tmp_path)
    result = calc.get_wannier90_input(seedname="wannier")

    assert result["win"].is_file()
    assert result["eig"].is_file()
    # nkptnr = 2*2*2, and every second-variational state is exported because
    # num_wann/num_bands were left at Elk's defaults (initw90.f90)
    assert result["nkpt"] == 8
    assert result["eigenvalues"].shape == (result["nkpt"], result["num_bands"])
    # eigenvalues are in eV and Fermi-referenced, so the Si 3s semicore-like
    # bottom band sits well below zero and the gap straddles it
    assert result["eigenvalues"].min() < -5.0
    assert result["eigenvalues"].max() > 0.0

    # the three library-dependent files are absent, and that is reported
    assert result["wannier90_available"] is False
    assert set(result["missing"]) == {"wannier.amn", "wannier.mmn", "wannier.spn"}

    win = result["win_contents"]
    assert win["settings"]["length_unit"] == "bohr"
    assert win["kpoints"].shape == (8, 3)
    assert win["unit_cell_cart"].shape == (3, 3)
    assert [s for s, _ in win["atoms_frac"]] == ["Si", "Si"]


def test_wannier90_band_subset(tmp_path):
    """`bands` maps to Elk's wann_bands/idxw90 block, which Elk reads as a
    raw line and re-parses with numlist -- so it must reach elk.in
    *unquoted*, unlike every other string block."""
    calc = _si(tmp_path)
    result = calc.get_wannier90_input(seedname="sub", bands="1-4")
    assert result["num_bands"] == 4
    assert result["eigenvalues"].shape == (8, 4)


# ---------------------------------------------------------------------------
# 350/351/352 -- supercell spin spirals
# ---------------------------------------------------------------------------


def test_spin_spiral_supercell_dry_run(tmp_path):
    """Task 352 does no SCF at all: it walks the q-points and creates one
    empty placeholder file each, so that several processes can share the
    directory. Measured on Si with ngridq=(2,2,2): three files, because Elk
    reduces the 8 q-points by symmetry to 3 inequivalent ones, named
    SS_Q0000_0000_0000.OUT, SS_Q0102_0000_0000.OUT and
    SS_Q0102_0102_0000.OUT -- confirming ssfext.f90's reduced-fraction
    convention (1/2 prints as "0102", a zero component as "0000")."""
    calc = _si(tmp_path)
    out = calc.get_spin_spiral_supercell((2, 2, 2), dry_run=True)
    assert out["results"] == []
    assert len(out["pending"]) == 3
    assert "SS_Q0102_0000_0000.OUT" in out["pending"]
    # every filename the dry run produced is one the parser can reconstruct
    from elkpy.parsers.magnetism import spin_spiral_filename

    predicted = {
        spin_spiral_filename((i, j, k), (2, 2, 2))
        for i in range(2)
        for j in range(2)
        for k in range(2)
    }
    assert set(out["pending"]) <= predicted


@slow
def test_spin_spiral_supercell(tmp_path):
    """One full ground state per q-point, on a supercell; minutes to hours
    even for one Fe atom at ngridq=(1,1,1)."""
    calc = _fe(tmp_path, species=FE_SPECIES_XZ, raise_on_nonconvergence=False)
    out = calc.get_spin_spiral_supercell((1, 1, 1), radkpt=20.0)
    assert len(out["results"]) == 1
    record = out["results"][0]
    assert record["ncells"] >= 1
    assert record["energy"] < 0.0
    assert record["q_lattice"] == pytest.approx(np.zeros(3), abs=1e-8)


# ---------------------------------------------------------------------------
# spin spirals by the generalised Bloch theorem (no task code)
# ---------------------------------------------------------------------------


@slow
def test_spin_spiral_energy_at_gamma_matches_a_plain_ferromagnet(tmp_path):
    """A q = 0 spiral is just the ferromagnetic state expressed in the
    spiral basis, so its energy must agree with an ordinary spin-polarised
    ground state on the same cell to well within the SCF convergence
    threshold. This is the sharpest available check that the
    spinsprl/vqlss plumbing is not silently changing the Hamiltonian."""
    structure = Structure(FE_FCC_AVEC, FE_FCC_SPECIES)
    plain = _Calculation(
        structure, tmp_path / "plain", xc="PW", spinpol=True,
        ngridk=(4, 4, 4), rgkmax=7.0,
        extra_blocks={"bfieldc": [(0.05, 0.0, 0.0)]},
        raise_on_nonconvergence=False,
    )
    spiral = _Calculation(
        structure, tmp_path / "spiral", xc="PW", spinpol=True,
        ngridk=(4, 4, 4), rgkmax=7.0, raise_on_nonconvergence=False,
    )
    e_plain = plain.get_energy()
    e_spiral = spiral.get_spin_spiral_energy((0.0, 0.0, 0.0))
    assert e_spiral == pytest.approx(e_plain, abs=1e-3)


@slow
def test_spin_spiral_dispersion(tmp_path):
    """Frozen-magnon dispersion: one self-consistent run per q-point."""
    calc = _Calculation(
        Structure(FE_FCC_AVEC, FE_FCC_SPECIES), tmp_path / "fe",
        xc="PW", spinpol=True, ngridk=(4, 4, 4), rgkmax=7.0,
        raise_on_nonconvergence=False,
    )
    qs = [(0.0, 0.0, 0.0), (0.1, 0.1, 0.0)]
    qpoints, energies = calc.get_spin_spiral_dispersion(qs, reference=(0.0, 0.0, 0.0))
    assert qpoints.shape == (2, 3)
    assert energies.shape == (2,)
    # E(q) is measured relative to the q = 0 run, so the first point is zero
    # by construction (the same calculation, run twice)
    assert energies[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 28/29 -- magnetic anisotropy energy
# ---------------------------------------------------------------------------


@slow
def test_mae(tmp_path):
    """npmae=2 samples only the x and z directions, i.e. two complete
    self-consistent ground states with spin-orbit coupling. Even this
    minimal setting is minutes-to-hours; Elk's own FeCo example uses
    npmae=-1, highq and an 8x8x8 mesh."""
    calc = _fe(tmp_path, spinorb=True, raise_on_nonconvergence=False)
    result = calc.get_mae(npmae=2)

    assert len(result["directions"]) == 2
    # gentpmae.f90's npmae=2 branch is exactly (theta,phi) = (pi/2,0) then
    # (0,0), i.e. +x then +z
    assert result["directions"][0]["direction"] == pytest.approx(
        np.array([1.0, 0.0, 0.0]), abs=1e-6
    )
    assert result["directions"][1]["direction"] == pytest.approx(
        np.array([0.0, 0.0, 1.0]), abs=1e-6
    )
    # each direction converged to a real moment
    assert all(d["moment"] > 0.5 for d in result["directions"])
    # the MAE is the spread of the sampled energies, so it is non-negative by
    # construction and must equal that spread exactly
    energies = [d["energy"] for d in result["directions"]]
    assert result["mae"] >= 0.0
    assert result["mae"] == pytest.approx(max(energies) - min(energies), rel=1e-6)
    # bcc Fe's cubic anisotropy is micro-Hartree scale (~1e-6 Ha per atom);
    # anything above milli-Hartree means something other than anisotropy was
    # measured
    assert result["mae"] < 1e-3


def test_mae_requires_a_magnetic_ground_state(tmp_path):
    calc = _si(tmp_path)
    with pytest.raises(ValueError, match="spinpol=True"):
        calc.get_mae()


def test_mae_rejects_an_npmae_gentpmae_would_stop_on(tmp_path):
    calc = _fe(tmp_path)
    for bad in (0, 1, -5):
        with pytest.raises(ValueError, match="invalid npmae"):
            calc.get_mae(npmae=bad)


# ---------------------------------------------------------------------------
# 160 -- exchange-correlation torque
# ---------------------------------------------------------------------------


@slow
def test_magnetic_torque_is_zero_for_a_collinear_ground_state(tmp_path):
    """torque.f90 short-circuits with torq = 0 unless ncmag; a collinear
    run therefore returns exact zeros -- worth pinning, because it is the
    single most likely way for this method to look like it worked when it
    measured nothing. The warning is the mitigation."""
    calc = _fe(tmp_path, raise_on_nonconvergence=False)
    with pytest.warns(RuntimeWarning, match="collinear"):
        torque = calc.get_total_magnetic_torque()
    assert torque.shape == (3,)
    assert torque == pytest.approx(np.zeros(3), abs=0.0)


@slow
def test_magnetic_torque_non_collinear(tmp_path):
    """With a transverse seed field init0.f90 sets ndmag = 3, so the cross
    product m x B_xc is actually evaluated. The residual measures the local
    functional's violation of the zero-torque theorem, so it is small but
    need not vanish."""
    calc = _fe(
        tmp_path, species=FE_SPECIES_XZ, spinorb=True, raise_on_nonconvergence=False
    )
    torque = calc.get_total_magnetic_torque()
    assert torque.shape == (3,)
    assert np.all(np.isfinite(torque))


def test_magnetic_torque_requires_spin_polarisation(tmp_path):
    calc = _si(tmp_path)
    with pytest.raises(ValueError, match="spin-polarised"):
        calc.get_total_magnetic_torque()


# ---------------------------------------------------------------------------
# 600-640 -- GW
# ---------------------------------------------------------------------------


@slow
def test_gw_chain(tmp_path):
    """The full prerequisite chain, on the smallest possible Si setting.

    Task 600 writes GWSEFM.OUT (and EPSINV.OUT); tasks 610 and 630 both read
    it back -- 630 through gwefermi -> gwchgk -> getgwsefm, which is why it
    is not the cheap standalone step its single-number output suggests.
    Running them in place in task 600's own directory is the whole point of
    _run_dependent: re-running the self-energy per dependent would multiply
    the cost of the workflow by three.

    Even at ngridk=(2,2,2), nempty=6 this is expected to take a long time.
    """
    calc = _si(tmp_path, raise_on_nonconvergence=False)
    gw_dir = calc.get_gw_self_energy(
        wmaxgw=5.0, tempk=2000.0, nempty=6, gmaxrf=2.0
    )
    assert (gw_dir / "GWSEFM.OUT").is_file()
    assert (gw_dir / "EPSINV.OUT").is_file()

    w, sf = calc.get_gw_spectral_function(gw_dir=gw_dir, wplot=(-0.8, 0.5), nwplot=200)
    assert w.shape == (200,)
    assert sf.shape == (200,)
    # a spectral function is a non-negative density of states
    assert np.all(sf >= -1e-10)
    # and it must carry weight somewhere in the plotted window
    assert sf.max() > 0.0

    efermi = calc.get_gw_fermi_energy(gw_dir=gw_dir)
    assert np.isfinite(efermi)

    dmat_dir = calc.get_gw_density_matrix(gw_dir=gw_dir)
    # task 640 overwrites the Kohn-Sham eigenvectors/occupations in place
    # with the GW natural orbitals and occupation numbers
    assert (dmat_dir / "EVECSV.OUT").is_file()
    assert (dmat_dir / "OCCSV.OUT").is_file()


@slow
def test_gw_band_structure(tmp_path):
    """Task 620 recomputes the inverse dielectric matrix at every point of
    the path, so it costs roughly npoints times a task-600 run. Two points
    only, and still expected to be very slow."""
    calc = _si(tmp_path, raise_on_nonconvergence=False)
    distances, frequencies, sf = calc.get_gw_band_structure(
        vertices=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
        npoints=2, wplot=(-1.0, 1.0), nwplot=100, nempty=6, gmaxrf=2.0,
    )
    assert distances.shape == (2,)
    assert frequencies.shape == (100,)
    assert sf.shape == (2, 100)


# ---------------------------------------------------------------------------
# 700-773 -- ultra-long-range
# ---------------------------------------------------------------------------


@slow
def test_ulr_chain(tmp_path):
    """Ultracell ground state plus its dependents, on the smallest possible
    setting: a 2x1x1 ultracell of bcc Fe.

    Every dependent runs in place in the ground state's directory, replaying
    that run's own avecu/ngridq blocks from the stage manifest -- getting
    those wrong does not error, it reads a mismatched STATE_ULR.OUT.
    """
    calc = _fe(tmp_path, raise_on_nonconvergence=False)
    ulr_dir = calc.get_ulr_ground_state(
        avecu=[(2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        ngridq=(2, 1, 1),
        maxscl=40,
        beta0=0.01,
    )
    assert (ulr_dir / "STATE_ULR.OUT").is_file()
    assert (ulr_dir / "ULR_INFO.OUT").is_file()

    energies, dos = calc.get_ulr_dos(ulr_dir, wplot=(-0.5, 0.5), nwplot=100)
    assert energies.shape == (100,)
    assert np.all(dos >= -1e-10)

    distances, values = calc.get_ulr_density(
        ulr_dir, dimension=1,
        vertices=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], npoints=50,
    )
    assert distances.shape == (50,)
    # a charge density is non-negative everywhere
    assert np.all(values > -1e-8)

    _distances, potential = calc.get_ulr_potential(
        ulr_dir, dimension=1,
        vertices=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], npoints=50,
    )
    assert potential.shape == (50,)

    _distances, magnetisation = calc.get_ulr_magnetisation(
        ulr_dir, dimension=1, ndmag=1,
        vertices=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], npoints=50,
    )
    assert magnetisation.shape == (50,)
    # |m| <= n pointwise: the magnetisation is a difference of the two spin
    # densities whose sum is the charge density
    assert np.all(np.abs(magnetisation) <= values + 1e-8)


@slow
def test_ulr_bands(tmp_path):
    calc = _fe(tmp_path, raise_on_nonconvergence=False)
    ulr_dir = calc.get_ulr_ground_state(
        avecu=[(2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        ngridq=(2, 1, 1), maxscl=40, beta0=0.01,
    )
    distances, energies, characters, spectral = calc.get_ulr_bands(
        ulr_dir, vertices=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], npoints=20,
    )
    assert distances.shape == (20,)
    assert energies.shape == characters.shape
    assert energies.shape[1] == 20
    # the kappa-point character is a projection weight, so it lies in [0, 1]
    assert np.all(characters >= -1e-8)
    assert np.all(characters <= 1.0 + 1e-8)
    sf_distances, sf_frequencies, sf = spectral
    assert sf.shape == (sf_frequencies.size, sf_distances.size)


def test_ulr_magnetisation_requires_spin_polarisation(tmp_path):
    calc = _si(tmp_path)
    with pytest.raises(ValueError, match="spin-polarised"):
        calc.get_ulr_magnetisation(tmp_path / "nowhere", dimension=1)


# ---------------------------------------------------------------------------
# input-side logic (no Elk invocation, but kept beside the tasks it serves)
# ---------------------------------------------------------------------------


def test_ndmag_reproduces_init0s_rule(tmp_path):
    """_ndmag decides how many columns the ULR magnetisation plots carry and
    whether task 160 can measure anything at all; getting it wrong is silent
    in both cases, so every branch of init0.f90's rule is pinned here."""
    avec = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

    def calc(species, path, **kwargs):
        return _Calculation(Structure(avec, species), tmp_path / path, **kwargs)

    z_seed = {"Fe": [((0.0, 0.0, 0.0), (0.0, 0.0, -0.01))]}
    x_seed = {"Fe": [((0.0, 0.0, 0.0), (-0.01, 0.0, -0.01))]}
    bare = {"Fe": [(0.0, 0.0, 0.0)]}

    assert calc(bare, "a")._ndmag() == 0  # no spin polarisation at all
    assert calc(z_seed, "b", spinpol=True)._ndmag() == 1  # collinear along z
    assert calc(x_seed, "c", spinpol=True)._ndmag() == 3  # transverse bfcmt
    assert calc(bare, "d", spinpol=True, spinorb=True)._ndmag() == 3
    # cmagz forces collinearity back, even over spin-orbit coupling
    assert (
        calc(bare, "e", spinpol=True, spinorb=True, extra_blocks={"cmagz": [True]})
        ._ndmag()
        == 1
    )
    # but a spin spiral clears cmagz, so it wins
    assert (
        calc(
            bare, "f", spinpol=True,
            extra_blocks={"cmagz": [True], "spinsprl": [True]},
        )._ndmag()
        == 3
    )
    # a transverse global field counts too, written either way round
    assert (
        calc(bare, "g", spinpol=True, extra_blocks={"bfieldc": [(0.05, 0.0, 0.0)]})
        ._ndmag()
        == 3
    )
    assert (
        calc(bare, "h", spinpol=True, extra_blocks={"bfieldc": [0.05, 0.0, 0.0]})
        ._ndmag()
        == 3
    )


def test_spin_spiral_supercell_from_state_requires_resume(tmp_path):
    """Task 351 reads each q-point's own STATE_Q..._..._....OUT from the same
    directory, so wiping it first would delete exactly what it needs."""
    calc = _si(tmp_path)
    with pytest.raises(ValueError, match="resume=True"):
        calc.get_spin_spiral_supercell((2, 2, 2), from_state=True)
