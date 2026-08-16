"""Integration tests for the magneto-optic Kerr effect (tasks 120 + 122,
src/writepmat.f90 / src/moke.f90) on ferromagnetic fcc nickel.

The Kerr angle needs BOTH a net magnetization and spin-orbit coupling, so
bulk Si (this repo's usual cheap fixture) can't exercise it at all -- the
structure here is Elk's own MOKE example,
vendor/elk/examples/TDDFT-optics/Ni-MOKE, at a much coarser k-mesh: these
tests check sign/parity/structure of the response, not a converged
spectrum (the example itself uses 32x32x32).

Skipped if the binary hasn't been built, same as test_calculation_si.py.
"""

import numpy as np
import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

NI_AVEC = [(1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]
NI_SPECIES = {"Ni": [(0.0, 0.0, 0.0)]}


def _ni_calculation(workdir, bz):
    """fcc Ni magnetized along +z or -z by a small symmetry-breaking field
    (`bfieldc`, the same knob Elk's Ni-MOKE example uses)."""
    s = Structure(NI_AVEC, NI_SPECIES, scale=3.33)
    return s.get_calculation(
        workdir,
        xc="PW",
        spinpol=True,
        spinorb=True,
        ngridk=(4, 4, 4),
        extra_blocks={"bfieldc": [(0.0, 0.0, bz)]},
    )


def test_get_moke_requires_spin_orbit_coupling(tmp_path):
    """No SOC (or no magnetization) means sigma_xy vanishes identically --
    raise before spending a real Elk run on a guaranteed zero."""
    s = Structure(NI_AVEC, NI_SPECIES, scale=3.33)
    no_soc = s.get_calculation(tmp_path / "ni_nosoc", xc="PW", spinpol=True, ngridk=(2, 2, 2))
    with pytest.raises(ValueError):
        no_soc.get_moke()
    nonmagnetic = s.get_calculation(
        tmp_path / "ni_nonmag", xc="PW", spinorb=True, ngridk=(2, 2, 2)
    )
    with pytest.raises(ValueError):
        nonmagnetic.get_moke()


def test_get_moke_is_odd_under_magnetization_reversal(tmp_path):
    """The Kerr angle is time-reversal odd: it exists only because the
    magnetization breaks T, so reversing M must flip its sign at every
    photon energy while leaving |theta_K| alone.

    This is the sign-of-the-effect check for MOKE -- two independent
    ground states (bfieldc along +z and -z), each with its own momentum
    matrix elements and conductivity tensor, sharing no arithmetic. A
    magnitude check alone would pass just as happily on |sigma_xy|, which
    is even under T and therefore not a Kerr effect at all.
    """
    up = _ni_calculation(tmp_path / "ni_up", +0.01)
    down = _ni_calculation(tmp_path / "ni_down", -0.01)

    w_up, kerr_up = up.get_moke(wplot=(0.0, 0.5), nwplot=200)
    w_down, kerr_down = down.get_moke(wplot=(0.0, 0.5), nwplot=200)

    assert w_up.shape == kerr_up.shape == (200,)
    assert np.iscomplexobj(kerr_up)
    assert np.allclose(w_up, w_down)
    # moke.f90 builds the grid from wplot(1) (clipped at 0) upwards
    assert w_up[0] == 0.0 and w_up[-1] < 0.5
    # ... and returns exactly zero at omega = 0 (the 1/omega in the polar
    # Kerr denominator is guarded there, not evaluated)
    assert kerr_up[0] == 0.0

    scale = np.abs(kerr_up).max()
    # a real, resolvable response at this (very coarse) mesh
    assert scale > 1e-3
    assert np.abs(kerr_down).max() == pytest.approx(scale, rel=0.05)
    # odd in M, both in the rotation (real part) and the ellipticity
    assert np.abs(kerr_up + kerr_down).max() < 0.05 * scale
