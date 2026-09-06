r"""Phase 2 of the JAX port: the XC functional against Elk's own v_xc.

`tests/test_jax_xc.py` pins `src/elkjax/xc.py` against closed forms -- Dirac
exchange, the exact exchange spin scaling, the Gell-Mann-Brueckner
high-density limit -- and against `jax.grad`.  All of those are statements
about the FORM of the functional.  None of them says it is the functional Elk
actually used, which is what this file checks: Elk's own v_xc along a line
through bulk silicon, against v_xc evaluated here on Elk's own density along
the same line.

**The agreement is limited by a commutation, not by the transcription**, and
the mechanism is worth stating because it sets the tolerance.  v_xc is a
NONLINEAR function of the density, and neither of Elk's two representations of
a real-space function commutes with it: inside a muffin tin, Elk applies the
functional on an angular grid and keeps `lmaxo` spherical harmonics of the
result, while this file applies it to the `lmaxo`-truncated density; in the
interstitial, Elk applies it pointwise on the FFT grid and `plot1d` Fourier-
interpolates the result, while this file applies it to the interpolated
density.  Measured on bulk Si: 3e-5 median relative over 200 points, rising to
1.2e-2 in the shell where the two representations meet at the muffin-tin
boundary.

That is still a discriminating check.  Correlation is ~17-20% of v_xc along
this line -- asserted below, so the claim is measured rather than assumed --
so a median agreement of 3e-5 identifies the parameterisation to about four
significant figures, four orders of magnitude finer than the difference
between having this correlation functional and having none.

Skipped without the elk binary, and without jax.
"""

import importlib.util

import numpy as np
import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = [
    pytest.mark.skipif(not config.default_elk_binary().is_file(),
                       reason="elk binary not built; see docs/design.md #8"),
    pytest.mark.skipif(importlib.util.find_spec("jax") is None,
                       reason="jax not installed; pip install -e .[jax]"),
]

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
LINE = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]      # atom, interstitial, atom
NPOINTS = 200


@pytest.fixture(scope="module")
def line_profile(tmp_path_factory):
    """Elk's own density and v_xc along one line of a converged bulk Si.

    Two separate task runs on the same ground state; `plot1d` puts them on the
    identical point set, which the test asserts rather than assumes.
    """
    workdir = tmp_path_factory.mktemp("xc") / "si"
    calculation = Structure(
        avec=SI_AVEC,
        species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    ).get_calculation(workdir, xc="PW", ngridk=(2, 2, 2), rgkmax=7.0)
    calculation.ensure_ground_state()
    density = calculation.get_density_1d(line=LINE, npoints=NPOINTS)
    potential = calculation.get_potential_1d(
        component="xc", line=LINE, npoints=NPOINTS)
    return density, potential


def test_elks_own_potential_is_this_functional_of_elks_own_density(
        line_profile):
    density, potential = line_profile
    from elkjax import xc
    assert np.abs(density["distance"] - potential["distance"]).max() == 0.0
    rho = density["values"]
    assert (rho > 0).all()
    assert rho.max() > 1e2 and rho.min() < 1e-2, "no dynamic range on this line"
    _, _, vx, _, vc, _ = xc.pwca(0.5 * rho, 0.5 * rho)
    got = np.asarray(vx) + np.asarray(vc)
    reference = potential["values"]
    relative = np.abs(got - reference) / np.abs(reference)
    assert np.median(relative) < 1e-3, np.median(relative)
    assert relative.max() < 5e-2, relative.max()


def test_the_comparison_discriminates_the_correlation_functional(line_profile):
    """The premise of the tolerance above: correlation is a large fraction of
    v_xc here, so agreeing with Elk to 3e-5 is not something an
    exchange-only functional could do."""
    density, _ = line_profile
    from elkjax import xc
    rho = density["values"]
    _, _, vx, _, vc, _ = xc.pwca(0.5 * rho, 0.5 * rho)
    fraction = np.abs(np.asarray(vc)) / np.abs(np.asarray(vx) + np.asarray(vc))
    assert 0.05 < np.median(fraction) < 0.5, np.median(fraction)
