"""Integration tests for the ground-state / geometry / mechanical-electric
response task family (elkpy.tasks.groundstate).

Skipped if the Elk binary hasn't been built (see docs/design.md #8). The
cheap tests run bulk Si at ngridk 2x2x2, rgkmax 5 -- the whole combined
sequence (tasks 0, 115, 110, 195, 430, 190, 440) took 13 s on the machine
these were written on, most of it the three SCF cycles task 440 needs.

The expensive ones -- Hartree-Fock (nonlocal exchange, O(nk^2)), the
piezoelectric and magnetoelectric tensors (a Berry-phase polarisation per
strain/field step, each itself a refined-mesh ground state), molecular
dynamics (one full SCF per force step), RDMFT (Hartree-Fock cost per outer
loop) and the DFT+U tensor moments -- are gated behind
ELKPY_RUN_SLOW_TESTS=1, the same convention tests/test_calculation_si_phonons.py
uses.
"""

import os

import numpy as np
import pytest

from elkpy import config
from elkpy.calculation import Calculation
from elkpy.structure import Structure
from elkpy.tasks.groundstate import GroundStateResponseTasks

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

slow = pytest.mark.skipif(
    os.environ.get("ELKPY_RUN_SLOW_TESTS") != "1",
    reason="slow (several SCF cycles per call); set ELKPY_RUN_SLOW_TESTS=1",
)

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}
HARTREE_PER_BOHR3_IN_GPA = 29421.02


class _Calculation(Calculation, GroundStateResponseTasks):
    """``Calculation`` with this task family mixed in.

    Once ``Calculation`` itself inherits ``GroundStateResponseTasks`` this
    subclass is a no-op (repeating a base that is already in the MRO
    leaves the linearisation unchanged), so the tests read the same before
    and after integration.
    """


@pytest.fixture
def si(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    return _Calculation(s, tmp_path / "si", xc="PW", ngridk=(2, 2, 2), rgkmax=5.0)


# ----------------------------------------------------------------------
# 430 / 440 -- strain and stress
# ----------------------------------------------------------------------
def test_strain_basis_of_a_cubic_crystal_is_one_isotropic_tensor(si):
    """genstrain symmetrises each candidate delta_ij over the crystal's own
    point group. For a cubic crystal every anisotropic candidate averages
    to a multiple of the identity, which the isotropic first tensor
    already spans, so exactly one strain direction survives."""
    tensors = si.get_strain_tensors()
    assert len(tensors) == 1
    avec = np.array(SI_AVEC)
    assert tensors[0] == pytest.approx(avec / np.linalg.norm(avec), abs=1e-6)
    assert np.linalg.norm(tensors[0]) == pytest.approx(1.0, abs=1e-6)


def test_get_stress(si):
    result = si.get_stress()
    assert result["stress"].shape == (1,)
    assert len(result["strain"]) == len(result["stress"])
    # a total-energy derivative per unit strain: order 10^-2 Hartree here,
    # nowhere near zero (this cell is not at the rgkmax=5 equilibrium) and
    # nowhere near the 10^2 that a units or sign-convention slip would give
    assert 1e-4 < abs(result["stress"][0]) < 1.0
    # the hydrostatic pressure follows from the isotropic component
    assert result["pressure"] is not None
    assert result["pressure"] == pytest.approx(
        -result["stress"][0] * np.linalg.norm(SI_AVEC) / (3 * abs(np.linalg.det(SI_AVEC))),
        rel=1e-9,
    )
    # a few GPa, not a few thousand -- catches a wrong volume or a missing
    # Frobenius norm. The VALUE at rgkmax=5 is dominated by Pulay stress
    # (the basis improves as the cell expands at fixed rgkmax), so this is
    # a magnitude bound, not a physical prediction.
    assert abs(result["pressure"]) * HARTREE_PER_BOHR3_IN_GPA < 100.0


def test_stress_step_size_is_honoured(si):
    """Halving deltast must not change the derivative by much -- it is a
    forward difference of an energy that is smooth in the strain."""
    coarse = si.get_stress(deltast=0.01, label="stress_coarse")
    fine = si.get_stress(deltast=0.005, label="stress_fine")
    assert fine["stress"][0] == pytest.approx(coarse["stress"][0], rel=0.2)


# ----------------------------------------------------------------------
# 115 -- electric field gradient
# ----------------------------------------------------------------------
def test_efg_vanishes_at_a_cubic_site(si):
    """The EFG is a traceless symmetric rank-2 tensor, which is forbidden
    at a site with cubic point symmetry. Both silicon sites in diamond are
    T_d, so every component must be zero -- an exact null test rather than
    a plausibility band."""
    atoms = si.get_efg()
    assert len(atoms) == 2
    for entry in atoms:
        assert entry["symbol"] == "Si"
        assert entry["tensor"] == pytest.approx(entry["tensor"].T)
        assert np.abs(entry["tensor"]).max() < 1e-6
        assert abs(entry["trace"]) < 1e-6
        assert np.abs(entry["eigenvalues"]).max() < 1e-6


def test_efg_refuses_an_inadequate_lmaxi(si):
    """Elk's default lmaxi=1 cannot represent an l=2 object and
    src/writeefg.f90 stops outright; elkpy raises before launching."""
    with pytest.raises(ValueError, match="lmaxi"):
        si.get_efg(lmaxi=1)


# ----------------------------------------------------------------------
# 110 -- Moessbauer parameters
# ----------------------------------------------------------------------
def test_mossbauer_contact_density(si):
    atoms = si.get_mossbauer()
    assert len(atoms) == 2
    for entry in atoms:
        density = entry["contact_density"]
        # rho(0) for Z=14 is of order 2000 Bohr^-3 -- a plausibility band,
        # not a prediction
        assert 500.0 < density["nuclear_center"] < 10000.0
        # the density falls monotonically outwards from the nucleus
        assert (
            density["nuclear_center"]
            > density["nuclear_surface"]
            > density["thomson_radius"]
        )
        # the nuclear radius is ~1e-4 Bohr, well inside the Thomson radius
        assert 0.0 < entry["nuclear_radius"] < entry["thomson_radius"]
        assert entry["nuclear_mesh_points"] < entry["thomson_mesh_points"]
    # both silicon sites are equivalent by symmetry
    assert atoms[0]["contact_density"]["nuclear_center"] == pytest.approx(
        atoms[1]["contact_density"]["nuclear_center"], rel=1e-6
    )
    # no hyperfine section for a non-magnetic run
    assert "contact_nuclear" not in atoms[0]


# ----------------------------------------------------------------------
# 195 / 196 -- structure factors
# ----------------------------------------------------------------------
def test_structure_factor_normalisation(si):
    """Elk prints omega*F with the imaginary part negated (the
    crystallographic convention), so the H = 0 coefficient is exactly the
    number of electrons in the cell: 2 x Z(Si) = 28. This is the check a
    wrong prefactor or a missing volume factor cannot survive."""
    result = si.get_structure_factors(hmaxvr=6.0)
    assert result["h"][0] == pytest.approx(0.0)
    assert result["F"][0].real == pytest.approx(28.0, abs=0.05)
    assert abs(result["F"][0].imag) < 1e-10
    # genhvec sorts by |H|
    assert np.all(np.diff(result["h"]) >= -1e-12)
    assert result["multiplicity"][0] == 1
    assert len(result["h"]) > 10


def test_structure_factor_diamond_extinction(si):
    """Diamond's two-atom basis makes reflections with h+k+l = 4n+2
    (in the conventional cubic setting) systematically absent. Whatever
    the setting, the intensity distribution must contain such exact zeros
    among reflections whose neighbours are strong."""
    result = si.get_structure_factors(hmaxvr=6.0)
    magnitudes = np.abs(result["F"])
    assert magnitudes.max() > 10.0
    assert np.any(magnitudes[1:] < 1e-6 * magnitudes.max())


def test_magnetic_structure_factors_refused_without_spin(si):
    with pytest.raises(ValueError, match="spin"):
        si.get_structure_factors(magnetic=True)


# ----------------------------------------------------------------------
# 190 -- geometry export
# ----------------------------------------------------------------------
def test_geometry_files_show_elks_own_frame(si):
    """No SCF is involved, but the output is not the input echoed back:
    with tshift=.true. Elk moves the origin onto an inversion centre, so
    Si's two atoms come back at +-(3/8,3/8,3/8) instead of (0,0,0) and
    (1/4,1/4,1/4). The lattice vectors are untouched."""
    result = si.get_geometry_files()
    xsf = result["xsf"]
    assert xsf["symbols"] == ["Si", "Si"]

    from elkpy.parsers.geomfile import BOHR_IN_ANGSTROM

    assert xsf["avec"] == pytest.approx(np.array(SI_AVEC) * BOHR_IN_ANGSTROM, rel=1e-6)

    fractional = xsf["positions"] @ np.linalg.inv(xsf["avec"])
    assert fractional[0] == pytest.approx(-fractional[1], abs=1e-8)
    separation = (fractional[1] - fractional[0]) % 1.0
    assert separation == pytest.approx([0.25, 0.25, 0.25], abs=1e-5)

    # the V_Sim file is in Bohr and in a rotated frame, but must describe
    # the same cell volume
    assert abs(np.linalg.det(result["ascii"]["avec"])) == pytest.approx(
        abs(np.linalg.det(np.array(SI_AVEC))), rel=1e-6
    )


# ----------------------------------------------------------------------
# 68 -- RAM-disk diagnostic
# ----------------------------------------------------------------------
def test_ramdisk_status(si):
    result = si.get_ramdisk_status()
    assert isinstance(result["initialised"], bool)
    if result["initialised"]:
        assert result["nfiles"] == len(result["files"])
        assert sum(f["bytes"] for f in result["files"]) == result["bytes"]


# ----------------------------------------------------------------------
# 500 -- writetest values
# ----------------------------------------------------------------------
def test_test_values_carry_upstream_tolerances(si):
    """writetest dumps the characteristic array of a task together with
    the tolerance Elk's own developers consider meaningful for it -- 5e-2
    for the stress (src/writestress.f90), 1e-3 for the EFG
    (src/writeefg.f90)."""
    values = si.get_test_values([115], blocks={"lmaxi": [2]})
    assert 115 in values
    entry = values[115]
    assert entry["description"].lower().startswith("electric field gradient")
    assert entry["tolerance"] == pytest.approx(1e-3)
    assert entry["values"].shape == (9,)
    assert np.abs(entry["values"]).max() < 1e-6  # cubic site, as above


# ----------------------------------------------------------------------
# slow: alternative ground states and the field-derivative tensors
# ----------------------------------------------------------------------
@slow
def test_hartree_fock_raises_the_gap(si):
    """Hartree-Fock has no correlation and no self-interaction error, so it
    overestimates the gap as badly as LDA underestimates it. Both bracket
    experiment; here the only assertion is the direction: the HF gap must
    exceed the Kohn-Sham one."""
    ks_distances, ks_bands = si.get_bands(
        [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], npoints=5
    )
    del ks_distances
    occupied = ks_bands[ks_bands.max(axis=1) < 0.0]
    empty = ks_bands[ks_bands.min(axis=1) > 0.0]
    ks_gap = empty.min() - occupied.max()

    result = si.get_hartree_fock(maxscl=6)
    assert np.isfinite(result["energy"])
    assert result["energy_history"].size >= 1
    assert result["band_gap"].size >= 1
    assert result["band_gap"][-1] > ks_gap


@slow
def test_piezoelectric_tensor_vanishes_in_a_centrosymmetric_crystal(si):
    """Diamond silicon has an inversion centre, and the piezoelectric
    tensor is odd under inversion, so every component is forbidden. A
    genuine null test -- the same run on a non-centrosymmetric crystal
    (zincblende, wurtzite) is where the effect appears."""
    entries = si.get_piezoelectric_tensor(nkspolar=2)
    assert len(entries) == 1
    for entry in entries:
        assert entry["length"] < 1e-3


@slow
def test_magnetoelectric_tensor_requires_spin_orbit(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    calc = _Calculation(s, tmp_path / "si_nosoc", ngridk=(2, 2, 2), rgkmax=5.0)
    with pytest.raises(ValueError, match="spinorb"):
        calc.get_magnetoelectric_tensor()


@slow
def test_molecular_dynamics_conserves_the_atom_count_and_restarts(si):
    """Two force steps of Born-Oppenheimer dynamics on the undisplaced
    cell. Silicon starts at its equilibrium positions with zero force by
    symmetry, so the trajectory should barely move -- what is tested is
    the bookkeeping (shapes, times, the restart handshake), not dynamics.
    """
    result = si.get_molecular_dynamics(tstime=4.0, dtimes=0.5, ntsforce=4)
    nsteps = result["time"].size
    assert nsteps >= 1
    assert result["energy"].shape == (nsteps,)
    assert result["force"].shape == (nsteps, 2, 3)
    assert result["displacement_cartesian"].shape == (nsteps, 2, 3)
    if "final_state" in result:  # atptstep skips the last step's ATDVC.OUT
        assert result["final_state"]["displacement"].shape == (2, 3)
    # the forces on the two atoms of a centrosymmetric cell cancel
    assert result["force"][0].sum(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    restarted = si.get_molecular_dynamics(tstime=8.0, dtimes=0.5, ntsforce=4, restart=True)
    assert restarted["time"].size > nsteps
    assert restarted["time"][:nsteps] == pytest.approx(result["time"])


@slow
def test_rdmft_runs(si):
    result = si.get_rdmft(rdmmaxscl=1, maxitn=1, maxitc=1)
    assert np.isfinite(result["energy"])
    assert result["energy_history"].size >= 1
    assert (result["directory"] / "RDM_INFO.OUT").exists()


@slow
def test_tensor_moments_need_dft_plus_u(si):
    with pytest.raises(ValueError, match="DFT\\+U"):
        si.get_tensor_moments()


@slow
def test_tensor_moments_of_a_dft_plus_u_run(tmp_path):
    """The (k,p,r) = (0,0,0) moment is the shell occupation, so it must be
    real, positive and no larger than the 2(2l+1) states of the shell."""
    s = Structure(SI_AVEC, SI_SPECIES)
    calc = _Calculation(
        s,
        tmp_path / "si_u",
        ngridk=(2, 2, 2),
        rgkmax=5.0,
        # dftu=1 (around mean field), inpdftu=1 (U and J given directly);
        # species 1, l=1, U=0.15 Ha, J=0.0 -- see docs/elk_manual.txt 5.42
        extra_blocks={"dft+u": [(1, 1), (1, 1, 0.15, 0.0)]},
    )
    moments = calc.get_tensor_moments()
    assert moments
    occupations = [m for m in moments if (m["k"], m["p"], m["r"]) == (0, 0, 0)]
    assert occupations
    for entry in occupations:
        assert entry["t"].tolist() == [0]
        value = float(np.real(entry["value"][0]))
        assert 0.0 <= value <= 2 * (2 * entry["l"] + 1) + 1e-6
