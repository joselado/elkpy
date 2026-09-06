r"""Phase 2 of the JAX port: the muffin-tin angular transform, and an anomaly.

A nonlinear functional cannot be applied in the spherical-harmonic basis, so
Elk evaluates it on an angular grid: `rbsht` maps the `lmmax` coefficients at
each radial point to `lmmax` values on that grid, the functional is applied
pointwise, and `rfsht` maps back.  `elkjax.grid.to_angular`/`from_angular`
transcribe the pair, with patch 0016 exporting the four matrices.

**Two results are solid.**  The transforms are mutual inverses to 2.7e-12, and
the exchange and correlation ENERGY densities built this way reproduce Elk's
own `exmt` and `ecmt` to 1.8e-15 and 2.8e-16 -- so both the transform and the
density it is applied to are right.

**One is not, and this file pins it rather than hiding it.**  The same
construction applied to the POTENTIAL misses Elk's `vxcmt` by 5.3e-3 on a
scale of 45 (1.2e-4 relative).  What has been established about it:

  * it is entirely in CORRELATION.  Elk's v_x equals (4/3) times its own eps_x
    exactly at every point, and eps_x itself is exact, so exchange is right.
  * it is NOT a density error.  eps_c agrees to 2.8e-16 at the same points, and
    eps_c and v_c have comparable sensitivity to rho.
  * it is NOT a function of rho.  Points at rho = 11.2 disagree while points
    at rho = 0.78 agree; the two sets overlap in density.  It tracks RADIUS,
    growing smoothly from below 1e-9 at r = 0.30 Bohr to 5.3e-3 at r = R_MT.
  * it is NOT the interstitial's story.  In the interstitial the identical
    transcription reproduces Elk's `vxcir` to 4.4e-16 over an overlapping
    density range (test_calculation_xc.py).
  * it is NOT mixing.  vsmt - vclmt - vxcmt is 1.7e-7 on a scale of 9e7, so
    the exported potentials are mutually consistent, and comparing against
    vsmt - vclmt instead gives the identical 5.3e-3.

Ruled out and unexplained.  The assertions below therefore pin the CURRENT
state: if a later change explains it, they fail, which is the point.

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


@pytest.fixture(scope="module")
def groundstate(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("mtxc") / "si"
    calculation = Structure(
        avec=SI_AVEC,
        species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    ).get_calculation(workdir, xc="PW", ngridk=(2, 2, 2), rgkmax=7.0)
    calculation.ensure_ground_state()
    with calculation.eigenstate_session() as session:
        return session.ground_state()


def test_the_angular_transforms_are_mutual_inverses(groundstate):
    """`rfsht(rbsht(f)) = f`, on Elk's own density rather than a random
    vector: the packing (lmmaxi per point inside `nrmti`, lmmaxo outside) is
    what a transcription gets wrong, and a random vector would exercise it
    just as well only if the packing were right to begin with."""
    from elkjax import grid
    for ias in range(int(groundstate["natmtot"])):
        packed = groundstate["rhomt"][ias]
        back = np.asarray(grid.from_angular(
            grid.to_angular(packed, groundstate, ias), groundstate, ias))
        assert np.abs(back - packed).max() / np.abs(packed).max() < 1e-14


def test_the_energy_densities_are_exact_in_the_muffin_tin(groundstate):
    """`exmt` and `ecmt` from the angular-grid density, element-wise."""
    from elkjax import grid, xc
    for ias in range(int(groundstate["natmtot"])):
        rho = np.asarray(grid.to_angular(groundstate["rhomt"][ias],
                                         groundstate, ias))
        ex, ec = xc.pwca(0.5 * rho, 0.5 * rho)[:2]
        for values, key in ((ex, "exmt"), (ec, "ecmt")):
            got = np.asarray(grid.from_angular(np.asarray(values),
                                               groundstate, ias))
            reference = groundstate[key][ias]
            assert np.abs(reference).max() > 1e-2
            assert np.abs(got - reference).max() < 1e-13


def test_the_muffin_tin_potential_does_not_match_and_the_gap_is_in_correlation(
        groundstate):
    """The anomaly, pinned with what has been ruled out.

    Exchange is exact: Elk's own v_x is (4/3) eps_x to roundoff.  Correlation
    is not, by 1.2e-4 relative -- while eps_c at the very same points is exact.
    """
    from elkjax import grid, xc
    ias = 0
    isp = int(groundstate["idxis"][ias]) - 1
    npmt = int(groundstate["npmt"][isp])
    rho = np.asarray(grid.to_angular(groundstate["rhomt"][ias],
                                     groundstate, ias))[:npmt]
    elk_vxc = np.asarray(grid.to_angular(groundstate["vxcmt"][ias],
                                         groundstate, ias))[:npmt]
    elk_ex = np.asarray(grid.to_angular(groundstate["exmt"][ias],
                                        groundstate, ias))[:npmt]
    _, _, vx, _, vc, _ = xc.pwca(0.5 * rho, 0.5 * rho)
    # exchange, exactly
    assert np.abs(np.asarray(vx) - (4.0 / 3.0) * elk_ex).max() < 1e-12
    # correlation, not
    elk_vc = elk_vxc - (4.0 / 3.0) * elk_ex
    residual = np.abs(np.asarray(vc) - elk_vc)
    assert residual.max() > 1e-4, "the anomaly is gone -- update this file"
    assert residual.max() / np.abs(elk_vxc).max() < 1e-3


def test_the_muffin_tin_gap_is_not_a_function_of_the_density(groundstate):
    """The observation that rules out the obvious explanations: points at
    rho = 11 disagree while points at rho = 0.78 agree, so the two sets
    overlap in density and the discrepancy tracks radius instead."""
    from elkjax import grid, xc
    ias = 0
    isp = int(groundstate["idxis"][ias]) - 1
    npmt = int(groundstate["npmt"][isp])
    rho = np.asarray(grid.to_angular(groundstate["rhomt"][ias],
                                     groundstate, ias))[:npmt]
    elk = np.asarray(grid.to_angular(groundstate["vxcmt"][ias],
                                     groundstate, ias))[:npmt]
    _, _, vx, _, vc, _ = xc.pwca(0.5 * rho, 0.5 * rho)
    bad = np.abs(np.asarray(vx) + np.asarray(vc) - elk) > 1e-9
    assert bad.any() and (~bad).any()
    assert rho[bad].max() > rho[~bad].min(), (
        "the disagreeing and agreeing points no longer overlap in density; "
        "if they now separate cleanly, the explanation may be a density "
        "threshold after all")
