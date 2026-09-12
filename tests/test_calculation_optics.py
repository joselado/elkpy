"""Integration tests for the optics/TDDFT/BSE task family
(src/elkpy/tasks/optics.py), running the real Elk binary on bulk Si.

Skipped if the binary hasn't been built (see docs/design.md #8), same
pattern as tests/test_calculation_si.py.

The four tests that are NOT slow-gated cover tasks 120+125, 130, 135 and
450+455+456; that exact chain was measured at 32 s on 2x2x2 Si after a
13 s ground state, so they are affordable. Everything gated behind
ELKPY_RUN_SLOW_TESTS is a genuine response calculation (a BSE
diagonalisation, a chi0 accumulation over the zone, or a real-time
propagation) whose cost is not bounded by anything cheap.
"""

import os

import numpy as np
import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

slow = pytest.mark.skipif(
    not os.environ.get("ELKPY_RUN_SLOW_TESTS"),
    reason="set ELKPY_RUN_SLOW_TESTS=1 to run the response/real-time calculations",
)

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}
SI_VOLUME = abs(np.linalg.det(np.array(SI_AVEC)))


@pytest.fixture(scope="module")
def si_optics(tmp_path_factory):
    """One ground state shared by every test in this module -- the manifest
    cache makes a second Calculation on the same directory free, and each
    get_* still runs in its own wiped subdirectory."""
    workdir = tmp_path_factory.mktemp("si_optics")
    s = Structure(SI_AVEC, SI_SPECIES)
    return s.get_calculation(
        workdir / "si", xc="PW", rgkmax=6.0, ngridk=(2, 2, 2),
        extra_blocks={"nempty": [8]},
    )


# --------------------------------------------------------------------------
# task 125 -- second-harmonic generation
# --------------------------------------------------------------------------

def test_get_shg(si_optics):
    from elkpy.parsers.nonlinopt import energies_from_wplot

    result = si_optics.get_shg(
        components=((1, 2, 3), (1, 1, 1)), wmax=1.0, nwplot=200, swidth=0.005
    )
    assert result["energies"].shape == (200,)
    # src/nonlinopt.f90 builds its own grid from wplot(2) alone, starting
    # at zero -- reproduced independently in Python
    assert result["energies"] == pytest.approx(
        energies_from_wplot(1.0, 200), rel=1e-8, abs=1e-12
    )
    for key in ("chi", "chi_ii", "eta_ii", "sigma_ii"):
        assert set(result[key]) == {(1, 2, 3), (1, 1, 1)}
        assert result[key][(1, 2, 3)].shape == (200,)

    # src/nonlinopt.f90 forms CHI_2WWW as chi_II + eta_II + i/2w sigma_II
    # of the other three files it writes, so re-summing them must
    # reproduce the total exactly. This is the check with teeth: it fails
    # if any of the four OUTPUT_FILE_TEMPLATES entries is mapped to the
    # wrong quantity, which a shape/finiteness check cannot see.
    for component in ((1, 2, 3), (1, 1, 1)):
        total = (result["chi_ii"][component] + result["eta_ii"][component]
                 + result["sigma_ii"][component])
        assert total == pytest.approx(result["chi"][component], rel=1e-6, abs=1e-14)

    # diamond Si is CENTROSYMMETRIC, so chi^(2) vanishes identically by
    # symmetry (chi -> -chi under inversion). This is a null test: it
    # constrains nothing about the magnitude scale, only that the code
    # path returns the symmetry-required zero rather than something large.
    assert np.abs(result["chi"][(1, 2, 3)]).max() < 1e-3


# --------------------------------------------------------------------------
# task 130 -- < i,k+q | e^{iq.r} | j,k >
# --------------------------------------------------------------------------

def test_get_expiqr(si_optics):
    result = si_optics.get_expiqr(vecql=(0.5, 0.0, 0.0))
    assert result["vecql"] == pytest.approx([0.5, 0.0, 0.0])
    assert len(result["kpoints"]) == 1
    matrix = result["kpoints"][0]["matrix"]
    assert matrix.shape[0] == matrix.shape[1]

    # src/writeexpmat.f90 scales the muffin-tin phase factor by the cell
    # volume (expmt = omega*expmt), so completeness bounds each row by
    # Omega^2 -- approached from below because the state sum is truncated
    # at nstsv. A missing or doubled Omega would break this.
    row_sums = (np.abs(matrix) ** 2).sum(axis=1)
    assert np.all(row_sums <= SI_VOLUME ** 2 * (1 + 1e-9))
    assert row_sums.max() > 0.1 * SI_VOLUME ** 2


def test_get_expiqr_rejects_incommensurate_q(si_optics):
    # src/tddftlr.f90 and src/genexpmat.f90 need ngridk*q integral; the
    # check is in Python so a lost run becomes an immediate error
    with pytest.raises(ValueError, match="incommensurate"):
        si_optics.get_expiqr(vecql=(0.3, 0.0, 0.0))


# --------------------------------------------------------------------------
# task 135 -- plane-wave wavefunctions
# --------------------------------------------------------------------------

def test_get_plane_wave_wavefunctions(si_optics):
    """sum_H |c_H|^2 is the norm of the state in the plane-wave
    representation: Bessel's inequality in an orthonormal basis, so bounded
    by 1 and approaching it as the cut-off grows.

    It holds only on a radial mesh fine enough to do the muffin-tin
    transform. genwfpw integrates j_l(|H+k| r) against u_l(r) on the COARSE
    mesh (every ``lradstp``-th point, Elk's default 4), which at Si's default
    cut-off overshoots 1 by 7.7e-5 -- and, because the error is the
    quadrature's and not the basis's, overshoots by MORE as the cut-off grows
    (3.2e-4 at gmaxvr 16, 1.7e-3 at 20). lradstp=1 is what makes the
    inequality a statement about the basis again.

    Getting the Fortran-order reshape wrong would scramble the
    (H, spinor, state) axes and break the bound for every state at once.
    """
    result = si_optics.get_plane_wave_wavefunctions(lradstp=1)
    nkpt, nhkmax, nspinor, nstsv = result["wfpw"].shape
    assert result["kpoints"].shape == (nkpt, 3)
    assert nspinor == 1                      # no spinpol, no spinorb

    norms = (np.abs(result["wfpw"]) ** 2).sum(axis=(1, 2))
    assert np.all(norms <= 1.0 + 1e-8)
    assert np.all(norms[:, :4] > 0.9)


def test_plane_wave_cutoff_is_gmaxvr_not_hkmax(si_optics):
    """init4.f90:24 overwrites hkmax with 0.5*gmaxvr for task 135, so the
    hkmax block is inert here and the method refuses it rather than
    returning the default cut-off. gmaxvr is the knob that works: 12 -> 16
    took the H-set from 982 to 2346 vectors when this was measured."""
    with pytest.raises(ValueError, match="init4"):
        si_optics.get_plane_wave_wavefunctions(hkmax=3.0)

    small = si_optics.get_plane_wave_wavefunctions(label="wfpw_g12")
    large = si_optics.get_plane_wave_wavefunctions(gmaxvr=16.0, label="wfpw_g16")
    assert large["wfpw"].shape[1] > small["wfpw"].shape[1]


def test_plane_wave_norm_overshoots_on_the_coarse_radial_mesh(si_optics):
    """The other half of the same measurement, asserted so the mechanism
    cannot be quietly mistaken for a transcription bug again: on Elk's
    default lradstp the sum EXCEEDS 1, and refining the radial mesh alone --
    same states, same cut-off, same H-set -- brings it back under."""
    coarse = si_optics.get_plane_wave_wavefunctions(label="wfpw_l4")
    fine = si_optics.get_plane_wave_wavefunctions(lradstp=1, label="wfpw_l1")
    assert coarse["wfpw"].shape == fine["wfpw"].shape

    def norm(result):
        return (np.abs(result["wfpw"]) ** 2).sum(axis=(1, 2)).max()

    assert norm(coarse) > 1.0 + 1e-5
    assert norm(fine) <= 1.0


# --------------------------------------------------------------------------
# tasks 450/455/456 -- the laser field
# --------------------------------------------------------------------------

def test_get_afield(si_optics):
    result = si_optics.get_afield(
        pulses=[(0.0, 0.0, 0.1, 0.5, 0.0, 0.0, 10.0, 5.0)],
        tstime=20.0, dtimes=0.1, wplot=(0.0, 1.0), nwplot=200, swidth=0.005,
    )
    # src/gentimes.f90: ntimes = nint(tstime/dtimes) + 1
    assert result["times"].shape == (201,)
    assert result["afield"].shape == (201, 3)
    assert result["times"][1] == pytest.approx(0.1)

    # the pulse is polarised along z, so the other two components are
    # identically zero -- the check that the Cartesian columns are not
    # transposed
    assert np.all(result["afield"][:, :2] == 0.0)
    # a Gaussian envelope centred at t0 = 10 with FWHM 5: the field peaks
    # near the middle of the window, not at either end
    assert 5.0 < result["times"][np.argmax(np.abs(result["afield"][:, 2]))] < 15.0

    # src/writeafpdt.f90 computes AFTED.OUT by spline-integrating the same
    # AFPDT.OUT it writes, so a trapezium integral of the returned power
    # density must reproduce the returned energy density. Two files, one
    # quantity: this catches a mis-mapped filename.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy < 2.0
    integral = trapezoid(result["power_density"], result["times"])
    assert integral == pytest.approx(result["energy_density"], rel=1e-6)

    # E(w) = -(i w / c) A(w) is likewise polarised along z alone
    assert result["efield_w"].shape == (200, 3)
    assert np.all(np.abs(result["efield_w"][:, :2]) == 0.0)


def test_get_afield_requires_a_field(si_optics):
    with pytest.raises(ValueError, match="no A-field specified"):
        si_optics.get_afield(tstime=20.0, dtimes=0.1)


# --------------------------------------------------------------------------
# tasks 180/185/186/187 -- the Bethe-Salpeter chain
# --------------------------------------------------------------------------

@slow
def test_get_bse_dielectric(si_optics):
    result = si_optics.get_bse_dielectric(
        components=((1, 1),), wmax=1.0, nwplot=200, nvbse=2, ncbse=3,
        gmaxrf=2.0, swidth=0.005,
    )
    assert result["energies"].shape == (200,)
    eps = result["epsilon"][(1, 1)]
    assert eps.shape == (200,)
    # eps_11 carries the free-space delta_ij, so its real part starts near
    # or above 1 and Im eps is non-negative absorption
    assert np.all(eps.imag > -1e-6)
    # nmbse = nvbse*ncbse*nkptnr (src/genidxbse.f90); with the Hermitian
    # (Tamm-Dancoff) block the eigenvalues are real and positive
    assert result["excitations"].shape[0] == 2 * 3 * 8
    assert np.all(np.abs(result["excitations"].imag) < 1e-12)


@slow
def test_get_bse_excitations_without_screening(si_optics):
    """With hdbse=.false. the direct (screened) term is dropped, so task
    180 leaves the chain entirely and the excitation energies collapse
    toward the bare transition energies -- the excitonic binding is what
    the screened term supplies."""
    bound = si_optics.get_bse_excitations(nvbse=1, ncbse=1, gmaxrf=2.0)
    bare = si_optics.get_bse_excitations(
        nvbse=1, ncbse=1, direct=False, exchange=False, label="bse_bare"
    )
    assert bound["energies"].shape == bare["energies"].shape
    assert np.all(bare["energies"].real > 0)
    # the attractive electron-hole interaction lowers the spectrum
    assert bound["energies"].real.min() < bare["energies"].real.min()


def test_bse_refuses_an_ncbse_genidxbse_would_stop_on():
    """The refusal is ncbse > nempty + 1, where nempty is what init1.f90:316
    builds -- nint(nempty0*natmtot) -- and NOT the `nempty` block's own value.

    This test used to assert that Elk's defaults raise. They do not:
    2-atom Si at nempty0=4 has 8 empty states against a default ncbse of 3,
    and genidxbse.f90:82 passes with room to spare. Refusing it cost a run
    elkpy could have made."""
    s = Structure(SI_AVEC, SI_SPECIES)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        c = s.get_calculation(tmp, ngridk=(2, 2, 2))
        c._check_bse_states(3)          # Elk's own defaults: allowed
        c._check_bse_states(9)          # exactly the bound, nempty + 1
        with pytest.raises(ValueError, match="not enough conduction states"):
            c._check_bse_states(10)


# --------------------------------------------------------------------------
# task 320 -- linear-response TDDFT
# --------------------------------------------------------------------------

@slow
def test_get_tddft_dielectric_optical_limit(si_optics):
    result = si_optics.get_tddft_dielectric(
        components=((1, 1),), fxc="RPA", wplot=(0.0, 1.0), nwplot=100,
        swidth=0.005, gmaxrf=2.0,
    )
    # src/tddftlr.f90 writes `do iw=2,nwrf`, so one row fewer than nwplot
    assert result["energies"].shape == (99,)
    for key in ("epsilon", "epsinv", "epsm"):
        assert result[key][(1, 1)].shape == (99,)
    # eps^-1 is the inverse of eps in the FULL G,G' matrix; the head of
    # the inverse is not 1/eps_head, which is exactly why EPSM (the 3x3
    # head of eps^-1, re-inverted) exists as a separate file
    assert not np.allclose(result["epsm"][(1, 1)], result["epsilon"][(1, 1)])
    # a non-magnetic cell has no Kerr rotation
    assert np.abs(result["kerr"]).max() < 1e-6


@slow
def test_get_tddft_dielectric_finite_q(si_optics):
    """A finite q makes the head of the response a scalar rather than a
    3x3 matrix, so src/tddftlr.f90 writes the UNINDEXED
    EPSILON_TDDFT.OUT and no macroscopic/magneto-optic files."""
    result = si_optics.get_tddft_dielectric(
        vecql=(0.5, 0.0, 0.0), fxc="RPA", wplot=(0.0, 1.0), nwplot=100,
        gmaxrf=2.0, label="tddftlr_q",
    )
    assert "epsm" not in result
    assert np.iscomplexobj(result["epsilon"])
    assert result["epsilon"].shape == (99,)


# --------------------------------------------------------------------------
# tasks 330/331 -- spin-polarised linear response
# --------------------------------------------------------------------------

def test_spin_response_requires_spinpol(si_optics):
    # src/tddftsplr.f90's first statement stops on .not.spinpol
    with pytest.raises(ValueError, match="spinpol"):
        si_optics.get_spin_response()


@slow
def test_get_spin_response_bcc_fe(tmp_path):
    """bcc Fe, a collinear ferromagnet, so the transverse channel exists
    (src/tddftsplr.f90 writes CHI_T.OUT only for `.not.ncmag`)."""
    a = 5.42
    s = Structure([(-a / 2, a / 2, a / 2), (a / 2, -a / 2, a / 2),
                   (a / 2, a / 2, -a / 2)], {"Fe": [(0.0, 0.0, 0.0)]})
    c = s.get_calculation(
        tmp_path / "fe", spinpol=True, ngridk=(4, 4, 4), rgkmax=7.0,
        extra_blocks={"bfieldc": [(0.0, 0.0, 0.01)], "nempty": [8]},
    )
    result = c.get_spin_response(
        vecql=(0.0, 0.0, 0.0), fxc="ALDA", wplot=(0.0, 0.05), nwplot=50,
        gmaxrf=2.0,
    )
    assert result["energies"].shape == (50,)
    assert len(result["chi"]) == 16 and len(result["chi0"]) == 16
    assert result["chi_transverse"] is not None
    assert result["chi0_transverse"] is not None
    # the interacting chi differs from the Kohn-Sham chi0 whenever the
    # kernel is not zero -- with fxc="RPA" they would agree in the
    # magnetisation block
    assert not np.allclose(result["chi"][(1, 1)], result["chi0"][(1, 1)])


# --------------------------------------------------------------------------
# tasks 450/460/480 -- real-time TDDFT
# --------------------------------------------------------------------------

def test_tddft_requires_unshifted_origin(si_optics):
    # src/tddft.f90 stops on its first statement unless tshift = .false.
    with pytest.raises(ValueError, match="tshift"):
        si_optics.get_tddft_evolution(
            pulses=[(0.0, 0.0, 0.1, 0.5, 0.0, 0.0, 10.0, 5.0)]
        )


@slow
def test_get_tddft_evolution(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    c = s.get_calculation(
        tmp_path / "si_td", rgkmax=6.0, ngridk=(2, 2, 2),
        extra_blocks={"nempty": [8], "tshift": [False]},
    )
    result = c.get_tddft_evolution(
        pulses=[(0.0, 0.0, 0.05, 0.5, 0.0, 0.0, 5.0, 3.0)],
        tstime=10.0, dtimes=0.1,
    )
    # src/writetddft.f90 writes one row per completed step; the loop runs
    # itimes = 1 .. ntimes-1
    assert result["times"].shape[0] == 100
    assert result["current"].shape == (100, 3)
    # the field is polarised along z, so only J_z is driven
    assert np.abs(result["current"][:, 2]).max() > np.abs(result["current"][:, :2]).max()
    # a non-magnetic ground state writes no MOMENT_TD.OUT
    assert result["moment"] is None


@slow
def test_get_tddft_rt_dielectric(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    c = s.get_calculation(
        tmp_path / "si_tdrt", rgkmax=6.0, ngridk=(2, 2, 2),
        extra_blocks={"nempty": [8], "tshift": [False]},
    )
    # task 481 assumes E(t) is a SINGLE delta function at t = 0, i.e. A(t)
    # is a step switched on at t = 0 and never switched off; a step that
    # ends inside the window is two opposite kicks and would be
    # mis-analysed with no error
    result = c.get_tddft_rt_dielectric(
        steps=[(0.0, 0.0, 0.01, 0.0, 100.0)], tstime=20.0, dtimes=0.1,
        wplot=(0.0, 1.0), nwplot=100, delta_kick=True,
    )
    # src/dielectric_tdrt.f90 loops i,j = 1..3 unconditionally, ignoring
    # optcomp -- all nine components always exist
    assert len(result["epsilon"]) == 9
    assert result["energies"].shape == (100,)
    assert result["current_w"].shape == (100, 3)
    # the kick is along z, so eps_11 and eps_22 are zeroed by
    # dielectric_tdrt's own |E_j| < 1e-8 guard while eps_33 is not
    assert np.abs(result["epsilon"][(3, 3)] - 1.0).max() > 1e-6
