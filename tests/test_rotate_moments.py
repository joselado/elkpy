"""Unit tests for turning the moments inside the SCF (docs/design.md #36,
patches/0027-rotate-moments.patch): the Python half, with no Elk binary.

What Elk itself does with the blocks -- the gradient, the refusals, the BFGS
step -- is exercised by tests/test_calculation_rotate_moments.py against a
real run; here only what elkpy writes, what it refuses up front, and how it
reads ELKPY_ROTMOM.OUT back.
"""

import numpy as np
import pytest

from elkpy import params
from elkpy.calculation import Calculation, _raise_on_elk_error
from elkpy.inputfile import InputFile
from elkpy.parsers import orientation
from elkpy.structure import Structure

FEPT_AVEC = [(5.147, 0.0, 0.0), (0.0, 5.147, 0.0), (0.0, 0.0, 7.017)]


def _fept(seed=(0.6645, 0.2418, 0.7071)):
    b = 0.5 * np.asarray(seed) / np.linalg.norm(seed)
    return Structure(FEPT_AVEC, {
        "Fe": [((0.0, 0.0, 0.0), tuple(b))],
        "Pt": [((0.5, 0.5, 0.5), (0.0, 0.0, 0.0))],
    })


def _elk_in(calc):
    f = InputFile()
    calc._add_base_blocks(f)
    return f.render()


class _FakeLauncher:
    """Enough of LocalLauncher for _basis_signature, which stats the binary."""

    def __init__(self, path):
        path.write_text("")
        self.elk_binary = path


def test_rotating_needs_spin_orbit_coupling(tmp_path):
    with pytest.raises(ValueError, match="spinorb=True"):
        Calculation(_fept(), tmp_path, spinpol=True, rotate_moments=True)


def test_block_is_written_and_the_seed_fields_die_away(tmp_path):
    calc = Calculation(_fept(), tmp_path, spinorb=True, rotate_moments=True)
    text = _elk_in(calc)
    assert "elkpy_rotmom\n.true.\n" in text
    # the seed field only chooses the start; Elk's default reducebf=1 would
    # hold the moments on it, and elkpy_rotmom.f90 refuses that
    assert "reducebf\n0.5" in text


def test_a_reducebf_the_user_chose_is_kept(tmp_path):
    calc = Calculation(_fept(), tmp_path, spinorb=True, rotate_moments=True,
                       extra_blocks={"reducebf": [0.8]})
    text = _elk_in(calc)
    assert "reducebf\n0.8" in text
    assert "reducebf\n0.5" not in text


def test_off_by_default_and_then_nothing_is_written(tmp_path):
    calc = Calculation(_fept(), tmp_path, spinorb=True)
    text = _elk_in(calc)
    assert "elkpy_rotmom" not in text
    assert "reducebf" not in text


def test_the_flag_is_part_of_the_ground_state_signature(tmp_path):
    launcher = _FakeLauncher(tmp_path / "elk")
    off = Calculation(_fept(), tmp_path / "a", spinorb=True, launcher=launcher)
    on = Calculation(_fept(), tmp_path / "b", spinorb=True, rotate_moments=True,
                     launcher=launcher)
    assert off._basis_signature()["rotate_moments"] is False
    assert on._basis_signature()["rotate_moments"] is True


def test_elk_refusal_reaches_the_caller_as_its_own_text(tmp_path):
    """Elk prints Error(...) and exits 0, so without this a refused input
    surfaced as 'Ground state did not converge'."""
    log = tmp_path / "elk.out"
    log.write_text(
        "\n Info(readinput): something\n\n"
        "Error(elkpy_rotmom): crystal symmetry 3 turns the spins\n"
        " The magnetic group found from the seed fields contains a rotation\n"
        "\nElk code stopped\n"
    )
    with pytest.raises(RuntimeError, match="crystal symmetry 3 turns the spins"):
        _raise_on_elk_error(log)
    clean = tmp_path / "clean.out"
    clean.write_text("Elk code stopped\n")
    _raise_on_elk_error(clean)          # nothing to report, no exception


@pytest.mark.parametrize("name,value,expected", [
    ("elkpy_rotmom", True, "elkpy_rotmom\n.true.\n"),
    ("elkpy_torque", True, "elkpy_torque\n.true.\n"),
    ("elkpy_rotmom_fixphase", True, "elkpy_rotmom_fixphase\n.true.\n"),
])
def test_logical_blocks_render(name, value, expected):
    f = InputFile()
    for key, lines in params.render_blocks({name: value}).items():
        f.add_block(key, lines)
    assert expected in f.render()


def test_step_parameters_are_four_positive_numbers():
    rendered = params.render_blocks({"elkpy_rotmom_pm": (1e-6, 0.2, 0.05, 1e-4)})
    assert list(rendered) == ["elkpy_rotmom_pm"]
    with pytest.raises(params.ParameterError):
        params.render_blocks({"elkpy_rotmom_pm": (1e-6, 0.2, 0.05)})


# ---------------------------------------------------------------------------
# ELKPY_ROTMOM.OUT
# ---------------------------------------------------------------------------

HEADER_ROTATING = """\
# elkpy orientation gradient, one line per self-consistent loop
# moments turned: yes
# parameters (tolerance Ha/rad, trust rad, first step rad, start below dv):  0.100000E-06  0.100000      0.500000E-01  0.100000E-03
# in-plane phase of a coplanar texture held: F
# columns: loop, dE/dw (3, Ha/rad), |dE/dw| acted on, external-field bound, dv,
#  step w (3, rad), total angle turned (rad), moment axis (3), plane normal (3),
#  texture class (1 collinear, 2 coplanar, 3 non-coplanar), step taken (0/1)
"""
ROWS = """\
    14 -0.1615000271E-04  0.4797748245E-04 -0.1225993485E-05  0.506376E-04  0.507328E-05  0.500000E-04  0.1600000E-01 -0.4750000E-01  0.1000000E-02  0.500000E-01 -0.664023 -0.241642 -0.707589  0.342868  0.795027 -0.500374  1  1
    15  0.2000000000E-07  0.1000000000E-07  0.0000000000E+00  0.223607E-07  0.253664E-05  0.200000E-06   0.00000       0.00000       0.00000      0.500000E-01 -0.690000 -0.250000 -0.679000  0.342868  0.795027 -0.500374  1  0
"""


def test_parse_a_rotating_run(tmp_path):
    path = tmp_path / "ELKPY_ROTMOM.OUT"
    path.write_text(HEADER_ROTATING + ROWS)
    data = orientation.parse_orientation(path)
    assert data["rotating"] is True
    parameters = data["parameters"]
    assert parameters.pop("hold_phase") is False
    assert parameters == pytest.approx(
        {"tolerance": 1e-7, "trust": 0.1, "first_step": 0.05, "start": 1e-4})
    assert list(data["loop"]) == [14, 15]
    assert data["gradient"].shape == (2, 3)
    assert data["gradient"][0, 1] == pytest.approx(4.797748245e-5)
    assert data["step"][0] == pytest.approx([0.016, -0.0475, 0.001])
    assert list(data["stepped"]) == [True, False]
    assert list(data["texture"]) == [1, 1]
    assert data["axis"][1] == pytest.approx([-0.69, -0.25, -0.679])
    assert data["angle"][1] == pytest.approx(0.05)


def test_parse_a_report_only_run(tmp_path):
    path = tmp_path / "ELKPY_ROTMOM.OUT"
    path.write_text(
        "# elkpy orientation gradient, one line per self-consistent loop\n"
        "# moments turned: no\n" + ROWS.splitlines()[1] + "\n")
    data = orientation.parse_orientation(path)
    assert data["rotating"] is False
    assert data["parameters"] == {}
    assert data["loop"].tolist() == [15]


def test_parse_a_file_with_no_loops_yet(tmp_path):
    path = tmp_path / "ELKPY_ROTMOM.OUT"
    path.write_text(HEADER_ROTATING)
    data = orientation.parse_orientation(path)
    assert data["gradient"].shape == (0, 3)
    assert data["stepped"].dtype == bool


# ---------------------------------------------------------------------------
# INFO.OUT moments, loop by loop
# ---------------------------------------------------------------------------

MOMENTS = """\
Moments :
 interstitial                :  -0.2683554981E-03 -0.7097416337E-03  0.1732903420E-03
 muffin-tins
  species : 1 (Ir)
   atom 1                    :   0.2707802371E-02  0.1572105837E-02 -0.1238925094E-02
  species : 2 (Mn)
   atom 1                    :    1.026193876       1.223324339      -2.367511651
   atom 2                    :    1.399315931      -2.182885896       1.193443338
   atom 3                    :   -2.430113363      0.9378299546       1.171574291
 total in muffin-tins        :  -0.1895753766E-02 -0.2015949751E-01 -0.3732947482E-02
 total moment                :  -0.2164109264E-02 -0.2086923915E-01 -0.3559657140E-02

"""


def test_moments_are_read_for_every_loop(tmp_path):
    from elkpy.parsers import info
    path = tmp_path / "INFO.OUT"
    second = MOMENTS.replace("1.026193876", "1.100000000")
    path.write_text("+ Loop number : 1\n\n" + MOMENTS + "+ Loop number : 2\n\n" + second)
    data = info.parse_moments(path)
    assert data["muffin_tin"].shape == (2, 4, 3)
    assert data["species"] == ["Ir", "Mn", "Mn", "Mn"]
    assert data["muffin_tin"][0, 1, 0] == pytest.approx(1.026193876)
    assert data["muffin_tin"][1, 1, 0] == pytest.approx(1.1)
    assert data["total"][1] == pytest.approx([-0.2164109264e-2, -0.2086923915e-1, -0.3559657140e-2])
    assert data["interstitial"].shape == (2, 3)
    # the three Mn of the T1 state, 120 degrees apart
    u = data["muffin_tin"][0, 1:] / np.linalg.norm(data["muffin_tin"][0, 1:], axis=1)[:, None]
    assert np.degrees(np.arccos(u[0] @ u[1])) == pytest.approx(120.0, abs=0.5)
