"""Tests for elkpy.tasks.phonons.PhononTasks.

Two halves, deliberately:

1. WIRING tests that need no Elk binary at all. They subclass Calculation
   with the mixin and stub out ``_run_resumed``, recording the task list,
   input blocks and k-mesh it was handed and dropping fixture files into the
   subdirectory. That is the only way to check the things a parser test
   structurally cannot: that the tasks are chained in the order the Fortran
   requires (task 120's PMAT before 121's dielectric before 208's Born
   charges before 205's dynamical matrices before 220's dispersion), that
   the guards fire, and that the right input blocks are written.

2. INTEGRATION tests against the real binary, self-skipping when it has not
   been built. The Born-charge one is real physics: diamond Si is homopolar
   and every atom sits on a site of T_d symmetry, so Z* must vanish on each
   atom individually -- a null that a wrong filename template, a
   misinterpreted file layout, or a lost ``chgcr + spzn`` term would all
   break, since the bare electronic term alone is +4 and the nuclear one
   -14. The rest of the family is far too expensive to run here (a single
   DFPT phonon run on this same cell takes 11-13 minutes) and is left
   format-derived.

The mixin is applied here rather than assumed on Calculation, so these
tests pass both before and after the integrator wires it in.
"""

import os

import numpy as np
import pytest

from elkpy import config
from elkpy.calculation import Calculation
from elkpy.structure import Structure
from elkpy.tasks.phonons import PHONON_TASKS, PhononTasks

from test_parsers_eliashberg import NB_MCMILLAN
from test_parsers_phonon import BEC_P1, BEC_P2, BEC_P3, GAMMAQ_OUT, PHONON_OUT


SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}


# What the integrator produced: Calculation itself inherits PhononTasks.
_Calc = Calculation


# ==========================================================================
# 1. wiring tests -- no binary needed
# ==========================================================================


class _StubLauncher:
    elk_binary = None


class _Recorder(_Calc):
    """Records every _run_resumed call instead of running Elk, and drops
    `fixtures` (name -> text) into the subdirectory so the parsing half of
    each method still executes."""

    def __init__(self, *args, **kwargs):
        self.fixtures = kwargs.pop("fixtures", {})
        super().__init__(*args, **kwargs)
        self.calls = []

    def _run_resumed(self, subdir_name, tasks, extra_blocks=None, ngridk=None, vkloff=None):
        self.calls.append(
            {
                "subdir": subdir_name,
                # NOTE the real _run_resumed prepends task 1 (ground-state
                # resume) to whatever it is given; `tasks` here is the list
                # the method asked for, which is what these tests pin.
                "tasks": list(tasks),
                "blocks": dict(extra_blocks or {}),
                "ngridk": ngridk,
            }
        )
        subdir = self.workdir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        for name, text in self.fixtures.items():
            (subdir / name).write_text(text, encoding="utf-8")
        return subdir


def _recorder(tmp_path, fixtures=None, **kwargs):
    structure = Structure(SI_AVEC, SI_SPECIES)
    return _Recorder(
        structure,
        tmp_path / "run",
        ngridk=kwargs.pop("ngridk", (4, 4, 4)),
        launcher=_StubLauncher(),
        fixtures=fixtures or {},
        **kwargs,
    )


def _bec_fixtures():
    """The real Si task-208 output, written for both atoms (they are
    symmetry-equivalent in diamond, so this is also physically right)."""
    files = {}
    for iatom in (1, 2):
        for ip, text in zip((1, 2, 3), (BEC_P1, BEC_P2, BEC_P3)):
            files[f"BEC_S01_A{iatom:03d}_P{ip}.OUT"] = text
    return files


def test_born_charges_wiring(tmp_path):
    calc = _recorder(tmp_path, fixtures=_bec_fixtures())
    result = calc.get_born_charges(deltaph=0.02, nkspolar=6, ngridk=(6, 6, 6))
    (call,) = calc.calls
    assert call["tasks"] == [PHONON_TASKS["born_charges"]]
    assert call["blocks"]["deltaph"] == [0.02]
    assert call["blocks"]["nkspolar"] == [6]
    assert call["ngridk"] == (6, 6, 6)
    assert set(result["charges"]) == {(1, 1), (1, 2)}
    assert result["symbols"][(1, 2)] == "Si"
    assert result["acoustic_sum"].shape == (3, 3)


def test_born_charges_dry_run_uses_task_209(tmp_path):
    calc = _recorder(tmp_path)
    subdir = calc.get_born_charges(dry_run=True)
    (call,) = calc.calls
    assert call["tasks"] == [PHONON_TASKS["born_charges_dryrun"]]
    assert subdir.is_dir()


def test_phonon_modes_wiring(tmp_path):
    calc = _recorder(tmp_path, fixtures={"PHONON.OUT": PHONON_OUT})
    qpoints = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    modes = calc.get_phonon_modes(qpoints=qpoints, ngridq=(2, 2, 2))
    (call,) = calc.calls
    # dynamical matrices (205) must come before the mode writer (230)
    assert call["tasks"] == [205, PHONON_TASKS["phonon_modes"]]
    assert call["blocks"]["phwrite"] == [2, (0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    assert call["blocks"]["ngridq"] == [(2, 2, 2)]
    assert len(modes) == 2


def test_loto_chain_order_and_blocks(tmp_path):
    fixtures = dict(_bec_fixtures())
    fixtures["PHDISP.OUT"] = "  0.0  0.001\n  1.0  0.002\n\n  0.0  0.003\n  1.0  0.004\n"
    fixtures["PHDLINES.OUT"] = "  0.0  0.0\n  0.0  0.01\n\n  1.0  0.0\n  1.0  0.01\n"
    calc = _recorder(tmp_path, fixtures=fixtures)
    result = calc.get_phonon_dispersion_loto(
        vertices=[(0, 0, 0), (0.5, 0, 0)], ngridq=(2, 2, 2), npoints=2
    )
    (call,) = calc.calls
    # the exact ordering Elk's own GaAs-LO-TO example prescribes: momentum
    # matrix elements, dielectric tensor, Born charges, dynamical matrices,
    # dispersion. readepsw0/readbec are called from initph, i.e. inside 220.
    assert call["tasks"] == [120, 121, PHONON_TASKS["born_charges"], 205, 220]
    assert call["blocks"]["tphnat"] == [True]
    assert len(call["blocks"]["optcomp"]) == 9
    assert call["blocks"]["optcomp"][0] == (1, 1)
    assert result["frequencies"].shape == (2, 2)
    assert "born_charges" in result


def test_loto_refuses_positive_lower_frequency(tmp_path):
    calc = _recorder(tmp_path)
    with pytest.raises(ValueError, match="omega = 0"):
        calc.get_phonon_dispersion_loto(vertices=[(0, 0, 0)], wplot=(0.1, 0.5))
    assert calc.calls == []


def test_electron_phonon_requires_commensurate_meshes(tmp_path):
    calc = _recorder(tmp_path, ngridk=(4, 4, 4))
    with pytest.raises(ValueError, match="commensurate"):
        calc.get_electron_phonon_coupling(ngridq=(3, 3, 3))
    # nothing must have run: the check is before _run_resumed
    assert calc.calls == []
    # an override that IS commensurate is accepted
    calc.fixtures = {"GAMMAQ.OUT": GAMMAQ_OUT, "LAMBDAQ.OUT": GAMMAQ_OUT}
    calc.get_electron_phonon_coupling(ngridq=(3, 3, 3), ngridk=(6, 6, 6))
    assert calc.calls[-1]["ngridk"] == (6, 6, 6)


def test_electron_phonon_wiring_and_task_selection(tmp_path):
    fixtures = {"GAMMAQ.OUT": GAMMAQ_OUT, "LAMBDAQ.OUT": GAMMAQ_OUT}
    calc = _recorder(tmp_path, fixtures=fixtures, ngridk=(4, 4, 4))
    result = calc.get_electron_phonon_coupling(ngridq=(2, 2, 2), swidth=0.005, stype=1)
    assert calc.calls[-1]["tasks"] == [205, PHONON_TASKS["ephcouple"]]
    assert calc.calls[-1]["blocks"]["swidth"] == [0.005]
    assert calc.calls[-1]["blocks"]["stype"] == [1]
    assert result["linewidths"].shape == (2, 3)
    assert result["couplings"].shape == (2, 3)
    # asking for the matrix elements switches 240 -> 241, which is what
    # writes EPHMAT.OUT for the Bogoliubov tasks
    calc.get_electron_phonon_coupling(ngridq=(2, 2, 2), write_matrix_elements=True)
    assert calc.calls[-1]["tasks"] == [205, PHONON_TASKS["ephcouple_write"]]


def test_superconductivity_chain_order(tmp_path):
    fixtures = {
        "GAMMAQ.OUT": GAMMAQ_OUT,
        "LAMBDAQ.OUT": GAMMAQ_OUT,
        "ALPHA2F.OUT": "  0.0  0.0\n  0.001  1.0\n",
        "MCMILLAN.OUT": NB_MCMILLAN.read_text(encoding="utf-8"),
        "ELIASHBERG_GAP_T.OUT": "  2.0  5.0e-4  2.0\n  4.0  1.0e-4  1.9\n",
        "ELIASHBERG_IA.OUT": "  1.0  2.0  3.0\n\n  1.0  2.0  3.0\n",
        "ELIASHBERG_GAP_RA.OUT": "  0.0  1.0  0.0\n\n",
        "ELIASHBERG_Z_RA.OUT": "  0.0  1.0  0.0\n\n",
        "PHDOS.OUT": "  0.0  0.0\n  0.001  10.0\n",
        "PHDISP.OUT": "  0.0  0.001\n  1.0  0.002\n",
        "PHDLINES.OUT": "  0.0  0.0\n  0.0  0.01\n",
        "PHLWIDTH.OUT": "  0.0  1.0e-6\n  1.0  2.0e-6\n",
    }
    calc = _recorder(tmp_path, fixtures=fixtures, ngridk=(4, 4, 4))
    result = calc.get_superconductivity(vertices=[(0, 0, 0), (0.5, 0, 0)], ngridq=(2, 2, 2))
    (call,) = calc.calls
    # one DFPT run feeding every downstream task, in dependency order:
    # 240 writes GAMMAQ.OUT, which 245 and 250 both read; 250 writes
    # ALPHA2F.OUT, which 260 reads.
    assert call["tasks"] == [
        205,
        210,
        220,
        PHONON_TASKS["ephcouple"],
        PHONON_TASKS["phonon_linewidths"],
        PHONON_TASKS["alpha2f"],
        PHONON_TASKS["eliashberg"],
    ]
    assert result["lambda"] == pytest.approx(1.053424300)
    assert result["tc"] == pytest.approx(12.44672586)
    assert len(result["temperatures"]) == 2
    assert result["phonon_dos"][1][-1] == pytest.approx(10.0)


def test_eliashberg_function_wiring(tmp_path):
    fixtures = {
        "GAMMAQ.OUT": GAMMAQ_OUT,
        "LAMBDAQ.OUT": GAMMAQ_OUT,
        "ALPHA2F.OUT": "  0.0  0.0\n  0.001  1.0\n",
        "MCMILLAN.OUT": NB_MCMILLAN.read_text(encoding="utf-8"),
    }
    calc = _recorder(tmp_path, fixtures=fixtures, ngridk=(4, 4, 4))
    result = calc.get_eliashberg_function(ngridq=(2, 2, 2), mustar=0.1, ngrkf=30)
    (call,) = calc.calls
    # 240 writes GAMMAQ.OUT, which 250 reads back through readgamma
    assert call["tasks"] == [205, PHONON_TASKS["ephcouple"], PHONON_TASKS["alpha2f"]]
    assert call["blocks"]["mustar"] == [0.1]
    assert call["blocks"]["wplot"][0] == (500, 30, 1)
    # the a2F grid is keyed apart from any dispersion grid in the same dict
    assert "a2f_frequencies" in result and "alpha2f" in result
    assert result["lambda"] == pytest.approx(1.053424300)
    assert result["couplings"].shape == (2, 3)


def test_eliashberg_gap_chain_and_blocks(tmp_path):
    fixtures = {
        "GAMMAQ.OUT": GAMMAQ_OUT,
        "LAMBDAQ.OUT": GAMMAQ_OUT,
        "ALPHA2F.OUT": "  0.0  0.0\n  0.001  1.0\n",
        "MCMILLAN.OUT": NB_MCMILLAN.read_text(encoding="utf-8"),
        "ELIASHBERG_GAP_T.OUT": "  2.0  5.0e-4  2.0\n",
        # ragged on purpose: nwf shrinks as the temperature rises
        "ELIASHBERG_IA.OUT": "  1.0 2.0 3.0\n  2.0 2.0 3.0\n\n  1.0 2.0 3.0\n",
        "ELIASHBERG_GAP_RA.OUT": "  0.0  1.0  0.0\n\n",
        "ELIASHBERG_Z_RA.OUT": "  0.0  1.0  0.0\n\n",
    }
    calc = _recorder(tmp_path, fixtures=fixtures, ngridk=(4, 4, 4))
    result = calc.get_eliashberg_gap(ngridq=(2, 2, 2), mustar=0.12, ntemp=20, ngrkf=20)
    (call,) = calc.calls
    assert call["tasks"] == [
        205,
        PHONON_TASKS["ephcouple"],
        PHONON_TASKS["alpha2f"],
        PHONON_TASKS["eliashberg"],
    ]
    assert call["blocks"]["mustar"] == [0.12]
    assert call["blocks"]["ntemp"] == [20]
    # the wplot block carries nwplot/ngrkf/nswplot on its first line
    assert call["blocks"]["wplot"][0] == (500, 20, 1)
    assert [len(b[0]) for b in result["imaginary_axis"]] == [2, 1]


def test_lmaxi_below_two_is_refused(tmp_path):
    calc = _recorder(tmp_path)
    with pytest.raises(ValueError, match="lmaxi must be >= 2"):
        calc.get_phonon_modes(ngridq=(2, 2, 2), lmaxi=1)
    assert calc.calls == []


def test_dfpt_refused_for_spin_polarised_cell_supercell_offered(tmp_path):
    """src/phonon.f90 hard-stops on spinpol; src/phononsc.f90 is the only
    magnetic route, and it preserves bfcmt0/mommtfix across displacements."""
    calc = _recorder(tmp_path, spinpol=True)
    with pytest.raises(ValueError, match="method='supercell'"):
        calc.get_phonon_modes(ngridq=(2, 2, 2))
    assert calc.calls == []
    calc.fixtures = {"PHONON.OUT": PHONON_OUT}
    calc.get_phonon_modes(ngridq=(2, 2, 2), method="supercell")
    assert calc.calls[-1]["tasks"][0] == PHONON_TASKS["phonons_supercell"]


def test_supercell_wiring(tmp_path):
    fixtures = {"PHDISP.OUT": "  0.0  0.001\n  1.0  0.002\n"}
    calc = _recorder(tmp_path, fixtures=fixtures)
    distances, frequencies = calc.get_phonons_supercell(
        vertices=[(0, 0, 0), (0.5, 0, 0)], ngridq=(2, 2, 2), radkpt=30.0, npoints=2
    )
    (call,) = calc.calls
    assert call["tasks"] == [PHONON_TASKS["phonons_supercell"], 220]
    # task 200 forces autokpt, so radkpt (not ngridk) sets the supercell mesh
    assert call["blocks"]["radkpt"] == [30.0]
    assert len(distances) == 2 and frequencies.shape[0] == 1
    subdir = calc.get_phonons_supercell(ngridq=(2, 2, 2), dry_run=True)
    assert calc.calls[-1]["tasks"] == [PHONON_TASKS["phonons_supercell_dryrun"]]
    assert subdir.is_dir()


def test_bogoliubov_chain_order(tmp_path):
    fixtures = {
        "TDOS_EPH.OUT": " -0.001  0.0\n  0.001  5.0\n",
        "FACEEH.OUT": "  0.001  0.69  0.0  0.0  0.0\n",
    }
    calc = _recorder(tmp_path, fixtures=fixtures, ngridk=(4, 4, 4))
    result = calc.get_electron_phonon_bogoliubov(ngridq=(2, 2, 2), maxscl=500)
    (call,) = calc.calls
    # 241 (not 240) is required: only it writes EPHMAT.OUT, which initeph
    # reads back for tasks 270/280
    assert call["tasks"] == [
        205,
        PHONON_TASKS["ephcouple_write"],
        PHONON_TASKS["gndsteph"],
        PHONON_TASKS["ephdos"],
    ]
    assert call["blocks"]["maxscl"] == [500]
    assert call["blocks"]["ephscf"] == [(8.0, 0.02)]
    assert result["dos"][1][-1] == pytest.approx(5.0)
    assert len(result["face_energies"]) == 1


def test_dynamical_born_charges_wiring(tmp_path):
    from test_parsers_phonon import _dynamical_bec_text

    fixtures = {
        f"BEC_S01_A{ia:03d}_P{ip}.OUT": _dynamical_bec_text()
        for ia in (1, 2)
        for ip in (1, 2, 3)
    }
    calc = _recorder(tmp_path, fixtures=fixtures)
    result = calc.get_born_charges_dynamical(nwplot=3, wplot=(0.0, 1.0), tstime=100.0)
    (call,) = calc.calls
    assert call["tasks"] == [PHONON_TASKS["born_charges_dynamical"]]
    assert call["blocks"]["tstime"] == [100.0]
    assert call["blocks"]["wplot"][1] == (0.0, 1.0)
    assert result["charges"][(1, 1)].shape == (3, 3, 3)
    np.testing.assert_allclose(result["frequencies"], [0.0, 0.5, 1.0])


# ==========================================================================
# 2. integration tests -- real binary
# ==========================================================================

requires_binary = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)
requires_slow = pytest.mark.skipif(
    os.environ.get("ELKPY_RUN_SLOW_TESTS") != "1",
    reason="phonon-family runs take minutes to hours; set ELKPY_RUN_SLOW_TESTS=1",
)


@pytest.fixture
def si_calculation(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    return _Calc(
        s,
        tmp_path / "si",
        xc="PW",
        ngridk=(2, 2, 2),
        rgkmax=6.0,
        vkloff=(0.0, 0.0, 0.0),
    )


@requires_binary
@requires_slow
def test_born_charges_vanish_in_diamond_silicon(si_calculation):
    """Z* = 0 for every atom of diamond Si.

    Homopolar bonding plus T_d site symmetry force it: the site symmetry
    makes Z* a scalar, and the acoustic sum rule with two symmetry-related
    atoms then forces that scalar to zero. This is a genuine null -- the
    electronic Berry-phase term alone is about +4 and the core-plus-nuclear
    term about -14, so getting 0 back means both are present and correctly
    combined, and it is blind to neither sign nor filename convention.

    Roughly 12 ground-state runs plus 36 single-iteration polarisation runs
    even at this minimal setting; a few minutes on a 2-atom cell with the
    reference BLAS build.
    """
    result = si_calculation.get_born_charges(deltaph=0.01, nkspolar=2)
    assert set(result["charges"]) == {(1, 1), (1, 2)}
    for key, z in result["charges"].items():
        assert z.shape == (3, 3)
        assert np.max(np.abs(z)) < 0.05, f"atom {key} has a non-vanishing Z*:\n{z}"
    # the acoustic sum rule is then trivially satisfied, but check it anyway:
    # it is the quantity that stays meaningful for a polar crystal
    assert np.max(np.abs(result["acoustic_sum"])) < 0.1


@requires_binary
@requires_slow
def test_phonon_modes_at_gamma(si_calculation):
    """Three acoustic modes at Gamma with zero frequency (the translational
    invariance Elk enforces through src/sumrule.f90), and orthonormal
    eigenvectors.

    ~11-13 minutes for the DFPT half even at ngridq=(2,2,2); this is the
    cheapest way to exercise the task-230 path end to end.
    """
    modes = si_calculation.get_phonon_modes(qpoints=[(0.0, 0.0, 0.0)], ngridq=(2, 2, 2))
    assert len(modes) == 1
    frequencies = modes[0]["frequencies"]
    assert len(frequencies) == 6  # 3 * natoms
    assert np.max(np.abs(frequencies[:3])) < 1e-5
    ev = modes[0]["eigenvectors"]
    np.testing.assert_allclose(ev.conj() @ ev.T, np.eye(6), atol=1e-6)
