"""Integration tests for turning the moments inside the SCF (docs/design.md #36,
patches/0027-rotate-moments.patch), against a real Elk binary.

The refusals and the loop-1 null are cheap and always run. The FePt relaxation
onto its easy axis is the physics test and takes about 20 minutes on 3 cores,
so it runs only with ELKPY_RUN_SLOW_TESTS=1.

The null is the test of patch 0028 as much as of the gradient: without
spin-orbit coupling the gradient is zero for exact eigenstates of a Hamiltonian
that commutes with a global spin rotation, and upstream Elk's does not once the
muffin-tin field is non-spherical (its sigma.B block is Hermitian only for a
spherical field). Measured on this cell, upstream: 3.5e-9 Ha/rad at loop 1,
1.37e-6 from loop 10, no convergence in 60 loops. With 0028: 2.7e-17 at loop 1,
at most 1.3e-12 at any loop, converged in 23.
"""

import os

import numpy as np
import pytest

from elkpy import config
from elkpy.calculation import Calculation
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

slow = pytest.mark.skipif(
    os.environ.get("ELKPY_RUN_SLOW_TESTS") != "1",
    reason="about 20 minutes; set ELKPY_RUN_SLOW_TESTS=1",
)

FE_AVEC = [(-2.71, 2.71, 2.71), (2.71, -2.71, 2.71), (2.71, 2.71, -2.71)]
FEPT_AVEC = [(5.147, 0.0, 0.0), (0.0, 5.147, 0.0), (0.0, 0.0, 7.017)]
# 45 degrees from c and 20 from a: off every mirror of P4/mmm, so the
# magnetic group is {E, I} and every orientation is allowed
GENERIC = np.array([0.6645, 0.2418, 0.7071])


def _fept(seed):
    b = 0.5 * np.asarray(seed, dtype=float) / np.linalg.norm(seed)
    return Structure(FEPT_AVEC, {
        "Fe": [((0.0, 0.0, 0.0), tuple(b))],
        "Pt": [((0.5, 0.5, 0.5), (0.0, 0.0, 0.0))],
    })


def test_a_seed_on_a_symmetry_axis_is_refused(tmp_path):
    """Along c the magnetic group keeps the four-fold axis, whose spin
    rotation symrvf would use to undo any turn off it."""
    calc = Calculation(_fept((0.0, 0.0, 1.0)), tmp_path, spinorb=True,
                       rotate_moments=True, ngridk=(2, 2, 2), rgkmax=5.0)
    with pytest.raises(RuntimeError, match="turns the spins"):
        calc.ensure_ground_state()


def test_a_seed_field_that_never_dies_is_refused(tmp_path):
    calc = Calculation(_fept(GENERIC), tmp_path, spinorb=True, rotate_moments=True,
                       ngridk=(2, 2, 2), rgkmax=5.0, extra_blocks={"reducebf": [1.0]})
    with pytest.raises(RuntimeError, match="must die away"):
        calc.ensure_ground_state()


def test_gradient_without_spin_orbit_is_null_at_every_loop(tmp_path):
    s = Structure(FE_AVEC, {"Fe": [((0.0, 0.0, 0.0), (0.3, 0.5, 1.0))]})
    calc = Calculation(s, tmp_path, spinpol=True, ngridk=(6, 6, 6),
                       extra_blocks={"elkpy_torque": [True], "reducebf": [0.5],
                                     "epspot": [1e-7], "epsengy": [1e-7], "maxscl": [60]})
    data = calc.get_orientation_relaxation()
    # upstream Elk stalls here with dv at 1.3e-7: its own Hamiltonian turns
    # the moment (patch 0028)
    assert calc.converged
    assert data["rotating"] is False
    assert not data["stepped"].any()
    # the external field's own share is taken out, so what is left is the
    # exchange-correlation field against the output magnetisation
    assert np.linalg.norm(data["gradient"], axis=1).max() < 1e-10
    # a collinear texture is recognised, and its axis is the seed's
    assert (data["texture"] == 1).all()
    seed = np.array([0.3, 0.5, 1.0]) / np.linalg.norm([0.3, 0.5, 1.0])
    assert abs(abs(data["axis"][-1] @ seed) - 1.0) < 1e-6


@slow
def test_fept_turns_onto_its_easy_axis_in_one_ground_state(tmp_path):
    """45 degrees off c to within a tenth of a degree of it, converged, and at
    the energy of a plain run seeded 0.06 degrees off c on the same k-set.

    The reference has to be on the SAME magnetic group: seeded exactly along
    c, Elk's eight-operation run sits 0.73 meV above the same state computed on
    the {E, I} or the unreduced mesh (docs/design.md #36), which would be
    mistaken for the rotation missing the minimum.
    """
    common = dict(spinorb=True, ngridk=(8, 8, 6), rgkmax=7.0,
                  extra_blocks={"swidth": [0.005], "nempty": [8], "maxscl": [150],
                                "epspot": [1e-7], "epsengy": [1e-7]})
    turned = Calculation(_fept(GENERIC), tmp_path / "turned", rotate_moments=True,
                         **common)
    data = turned.get_orientation_relaxation()
    assert turned.converged
    axis = data["axis"]
    theta = np.degrees(np.arccos(np.abs(axis[:, 2])))
    assert theta[0] > 44.0
    assert theta[-1] < 0.2
    assert data["stepped"].sum() >= 10
    assert data["acting"][-1] < data["parameters"]["tolerance"]

    reference = dict(common)
    reference["extra_blocks"] = dict(common["extra_blocks"], reducebf=[0.5])
    near_c = Calculation(_fept((0.001, 0.0004, 1.0)), tmp_path / "near_c", **reference)
    assert abs(turned.get_energy() - near_c.get_energy()) < 1e-6
