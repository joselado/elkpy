r"""Phase 2, differentiated: dE/drho against the Kohn-Sham potential.

The port's justification is differentiability, and nothing in Phase 2 had been
differentiated except the functional itself -- every section here starts from
Elk's converged density and checks a value.  The defining identity of the
Kohn-Sham potential is available and costs no Elk run beyond the fixture:

    delta E_xc[rho] / delta rho(r)  =  v_xc(r).

**The XC half closes exactly**: AD through `xc.pwca` AND through the cell inner
product gives v_xc[rho] to 5.7e-17 median.  The residual against Elk's own
`vxcir` is 1.25e-5, and it is ENTIRELY `trimrfg` -- section 2a's low-pass --
which is asserted by reproducing it with `grid.trim` rather than argued.

**The electrostatic half does not close**, at 0.33 relative, and this file pins
that as an open question rather than hiding it.  The obvious bookkeeping --
half the Coulomb energy plus the Madelung term, whose functional derivative
should be v_cl by reciprocity -- moves the residual from 1.3 to 0.33 and no
further.  See `docs/jax_port_phase2.md` section 2k.

Finding this needed a fix first: `elkjax.integrate.cell_inner_product` called
`np.asarray` on its muffin-tin argument, so it could not be traced at all.  A
module that silently refuses to be differentiated is a defect in this port, not
a limitation, and there is a test for it below.

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
def silicon(tmp_path_factory):
    calculation = Structure(
        avec=SI_AVEC,
        species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    ).get_calculation(tmp_path_factory.mktemp("dedrho") / "si",
                      xc="PW", ngridk=(2, 2, 2), rgkmax=7.0)
    calculation.ensure_ground_state()
    with calculation.eigenstate_session() as session:
        return session.ground_state()


def _weight_and_mask(groundstate):
    """d(cell integral)/d(rhoir), and the genuinely interstitial points.

    The cell inner product weights the interstitial by `cfunir`, which is
    band-limited and therefore small-but-nonzero INSIDE the spheres -- so
    dividing the gradient by it there amplifies nothing into everything.  The
    mask is `cfunir > 0.5`, not `> 0`.
    """
    omega = float(groundstate["omega"])
    ngtot = int(groundstate["ngtot"])
    theta = np.asarray(groundstate["cfunir"])
    return (omega / ngtot) * theta, theta > 0.5


def _exchange_correlation_energy(groundstate):
    import jax.numpy as jnp
    from elkjax import integrate, xc
    rhomt = jnp.asarray(groundstate["rhomt"])
    exmt = jnp.asarray(groundstate["exmt"])
    ecmt = jnp.asarray(groundstate["ecmt"])

    def energy(rhoir):
        ex, ec = xc.pwca(0.5 * rhoir, 0.5 * rhoir)[:2]
        return (integrate.cell_inner_product(rhomt, rhoir, exmt, ex,
                                             groundstate)
                + integrate.cell_inner_product(rhomt, rhoir, ecmt, ec,
                                               groundstate))
    return energy


def test_the_cell_inner_product_is_traceable(silicon):
    """The defect this section found: it used to call `np.asarray`.

    Asserted directly rather than only through the identity below, because the
    failure was a `TracerArrayConversionError` -- an outright refusal, not a
    wrong number -- and a port justified by differentiability should notice
    that in a unit test rather than the first time someone differentiates.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import integrate
    rhomt = jnp.asarray(silicon["rhomt"])

    def total(rhoir):
        return integrate.cell_integral(rhomt, rhoir, silicon)

    gradient = jax.grad(total)(jnp.asarray(silicon["rhoir"]))
    assert np.isfinite(np.asarray(gradient)).all()
    # d(integral)/d(rhoir) IS the quadrature weight, exactly
    weight, _ = _weight_and_mask(silicon)
    assert np.abs(np.asarray(gradient) - weight).max() \
        / np.abs(weight).max() < 1e-14


def test_the_xc_functional_derivative_is_the_xc_potential(silicon):
    """dE_xc/drho = v_xc[rho], to machine precision.

    AD runs through `xc_pwca` and through the cell inner product's own
    quadrature, so this is not the same statement as section 2a's
    `jax.grad(eps_xc)` check -- that differentiated the functional at a point,
    this differentiates an integral of it over the cell.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import xc
    weight, live = _weight_and_mask(silicon)
    gradient = np.asarray(
        jax.grad(_exchange_correlation_energy(silicon))(
            jnp.asarray(silicon["rhoir"])))
    mine = gradient[live] / weight[live]

    rho = jnp.asarray(silicon["rhoir"])
    _, _, vx, _, vc, _ = xc.pwca(0.5 * rho, 0.5 * rho)
    reference = (np.asarray(vx) + np.asarray(vc))[live]
    assert np.abs(mine - reference).max() / np.abs(reference).max() < 1e-14


def test_the_gap_to_elks_vxcir_is_exactly_the_trim(silicon):
    """1.25e-5 against Elk's stored `vxcir`, and it is `trimrfg` and nothing
    else.

    Asserted by reproducing the number with `grid.trim` applied to this
    module's own potential -- so the claim is an equality, not an
    order-of-magnitude coincidence.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import grid, xc
    weight, live = _weight_and_mask(silicon)
    gradient = np.asarray(
        jax.grad(_exchange_correlation_energy(silicon))(
            jnp.asarray(silicon["rhoir"])))
    mine = gradient[live] / weight[live]

    rho = jnp.asarray(silicon["rhoir"])
    _, _, vx, _, vc, _ = xc.pwca(0.5 * rho, 0.5 * rho)
    untrimmed = jnp.asarray(vx) + jnp.asarray(vc)
    trimmed = np.asarray(grid.trim(untrimmed, silicon))[live]
    elk = np.asarray(silicon["vxcir"])[live]
    scale = np.abs(elk).max()

    against_elk = np.median(np.abs(mine - elk)) / scale
    against_trim = np.median(np.abs(mine - trimmed)) / scale
    assert against_elk > 1e-6, "the trim no longer shows; update this file"
    assert abs(against_elk - against_trim) / against_elk < 1e-6


def test_the_electrostatic_half_does_not_close_yet(silicon):
    """An OPEN result, pinned so it cannot be forgotten or silently fixed.

    Half the Coulomb energy plus the Madelung term should have functional
    derivative v_cl: the Hartree part is quadratic so the half is right, and
    the Madelung term supplies the other half of the nuclear reciprocity.
    Measured, adding it moves the residual from 1.3 to 0.33 -- the right
    direction, not far enough.

    The bounds below are deliberately two-sided.  If a later change closes
    this, the test fails and gets rewritten as a real check; if something
    breaks it further, that shows too.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import integrate, poisson
    weight, live = _weight_and_mask(silicon)
    rhomt = jnp.asarray(silicon["rhomt"])
    natmtot = int(silicon["natmtot"])
    y00 = 0.28209479177387814347

    def energy(rhoir, madelung):
        local = dict(silicon)
        local["rhoir"] = rhoir
        vclmt, vclir = poisson.coulomb_potential(local)
        packed = jnp.stack([poisson.pack(vclmt[i], silicon, i)
                            for i in range(natmtot)])
        total = 0.5 * integrate.cell_inner_product(rhomt, rhoir, packed,
                                                   vclir, silicon)
        if madelung:
            for ias in range(natmtot):
                isp = int(silicon["idxis"][ias]) - 1
                total = total + (float(silicon["spzn"][isp])
                                 * (vclmt[ias][0, 0]
                                    - float(silicon["vcln"][isp][0]))
                                 * y00 / 2.0)
        return total

    reference = np.asarray(silicon["vclir"])[live]
    scale = np.abs(reference).max()
    errors = []
    for madelung in (False, True):
        gradient = np.asarray(jax.grad(energy)(jnp.asarray(silicon["rhoir"]),
                                               madelung))
        errors.append(np.median(np.abs(gradient[live] / weight[live]
                                       - reference)) / scale)
    without, with_madelung = errors
    assert with_madelung < without, (
        "the Madelung term no longer improves the derivative")
    assert 0.05 < with_madelung < 1.0, (
        "the electrostatic functional derivative has moved; if it now closes, "
        "rewrite this as a real check -- see docs/jax_port_phase2.md 2k")
