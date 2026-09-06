r"""Phase 2 item 2f: `energy.f90`'s density-functional terms, from the JAX chain.

This is the first thing in Phase 2 that uses more than one of its own pieces at
once, which is the point.  The XC functional (2a/2b), the cell integral (2c) and
the Poisson solve (2e) were each checked against Elk alone; none of those checks
says the three are consistent WITH EACH OTHER -- that the potential built by one
is the one integrated by another, against the density they share.  Elk's own
decomposition says so term by term, and patch 0017 exports its thirteen scalars
at full precision so the comparison is not limited to INFO.OUT's print width.

Three terms are IMPORTED rather than reproduced, and the file says so rather
than hiding them inside a total: `evalsum` and `engyts` need the
second-variational step and a zone sum (Phase 3), and `engynn` is a property of
the lattice, not of the density.

**The headline is a prediction that turned out wrong, and its correction.**
Section 2d found that Elk symmetrises the muffin-tin v_xc and not its energy
densities, and predicted that `engyvxc` -- an integral against v_xc -- would
therefore disagree with Elk at ~1e-4 on a symmetric cell.  It agrees at 3e-16.
The reason is that symmetrisation is a GROUP AVERAGE, hence an orthogonal
projection, and rho is already in its range: <rho, S v> = <S rho, v> = <rho, v>
identically.  The 5.3e-3 pointwise leak lives entirely in the harmonics rho does
not have, so it is invisible to every integral against rho.  That is asserted
below on both structures, and it is why Elk can get away with the asymmetry.

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
HBN_A = 4.7455
HBN_AVEC = [(HBN_A, 0.0, 0.0),
            (-HBN_A / 2, HBN_A * np.sqrt(3) / 2, 0.0),
            (0.0, 0.0, 20.0)]

REPRODUCED = ("engyvcl", "engymad", "engyen", "engyhar", "engycl",
              "engyvxc", "engyx", "engyc")


def _ground_state(workdir, structure, ngridk, extra_blocks=None):
    calculation = structure.get_calculation(
        workdir, xc="PW", ngridk=ngridk, rgkmax=7.0, extra_blocks=extra_blocks)
    calculation.ensure_ground_state()
    with calculation.eigenstate_session() as session:
        return session.ground_state()


@pytest.fixture(scope="module")
def silicon(tmp_path_factory):
    return _ground_state(
        tmp_path_factory.mktemp("energy") / "si",
        Structure(avec=SI_AVEC,
                  species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}),
        (2, 2, 2))


@pytest.fixture(scope="module")
def hbn(tmp_path_factory):
    """Two species, and a slab: the vacuum makes the interstitial dominate the
    cell integral, where silicon's is a small correction to the spheres."""
    return _ground_state(
        tmp_path_factory.mktemp("energy") / "hbn",
        Structure(avec=HBN_AVEC,
                  species={"B": [(0.0, 0.0, 0.0)],
                           "N": [(1 / 3, 2 / 3, 0.0)]}),
        (2, 2, 1))


def _fixture(request, name):
    return request.getfixturevalue(name)


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_every_reproduced_term_matches_elk(request, name):
    """Term by term, against Elk's own converged scalars.

    Asserted individually rather than through the total, because the total is
    a sum of terms spanning three orders of magnitude with cancellation between
    them: `engykn` is +579 and `engyen` is -1219 on silicon, so an error of
    1e-3 in either would leave `engytot` looking fine at 1e-6 relative.
    """
    from elkjax import energy
    groundstate = _fixture(request, name)
    got = energy.terms(groundstate)
    for term in REPRODUCED + ("engykn", "engytot"):
        mine, reference = float(got[term]), float(groundstate[term])
        assert abs(mine - reference) / abs(reference) < 1e-13, term


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_the_poisson_potential_gives_the_same_energy_as_elks_own(request, name):
    """Substituting Elk's `vclmt`/`vclir` for section 2e's must change nothing.

    This is the integrated counterpart of 2e's pointwise check, and it is not
    redundant with it: a pointwise agreement at 1e-14 could still integrate to
    something different if the quadrature and the solve disagreed about the
    packing or the region split, which is exactly the kind of mismatch a
    single-module test cannot see.
    """
    from elkjax import energy
    groundstate = _fixture(request, name)
    built = energy.terms(groundstate, poisson_potential=True)
    imported = energy.terms(groundstate, poisson_potential=False)
    for term in REPRODUCED:
        assert abs(float(built[term]) - float(imported[term])) \
            / max(abs(float(imported[term])), 1e-12) < 1e-13, term


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_the_symmetrisation_leak_is_orthogonal_to_the_density(request, name):
    """Section 2d's correction, measured on both structures.

    The pointwise gap between Elk's symmetrised muffin-tin v_xc and the
    pointwise one is 1e-3.  Integrated against rho it is 1e-16 relative,
    because a group average is an orthogonal projection and rho is already in
    its range.  Both halves are asserted: without the first this would be a
    test that two identical things are identical.
    """
    import jax.numpy as jnp
    from elkjax import energy, integrate
    groundstate = _fixture(request, name)
    potentials = energy.kohn_sham_potentials(groundstate)
    mine = np.asarray(potentials["vxcmt"])
    elk = np.asarray(groundstate["vxcmt"])

    pointwise = 0.0
    for ias in range(int(groundstate["natmtot"])):
        n = int(groundstate["npmt"][int(groundstate["idxis"][ias]) - 1])
        pointwise = max(pointwise, np.abs(mine[ias, :n] - elk[ias, :n]).max())
    assert pointwise > 1e-4, (
        "no leak to be orthogonal to -- has `potxc` stopped symmetrising?")

    zero = jnp.zeros_like(jnp.asarray(groundstate["rhoir"]))
    overlap = integrate.cell_inner_product(
        groundstate["rhomt"], groundstate["rhoir"],
        jnp.asarray(mine - elk), zero, groundstate)
    scale = integrate.cell_inner_product(
        groundstate["rhomt"], groundstate["rhoir"],
        jnp.asarray(elk), zero, groundstate)
    assert abs(float(overlap) / float(scale)) < 1e-14


def test_the_madelung_term_needs_the_nuclear_subtraction(silicon):
    """A mutation test on the one term that is not an integral.

    `energy.f90` reads the l=0 potential at the FIRST radial point and
    subtracts `vcln` there -- the nucleus's own divergent self-term.  Both are
    of order 1e7 and the difference is of order 1e2, so dropping the
    subtraction leaves a finite, smooth number wrong by five orders of
    magnitude, and the total energy along with it.
    """
    from elkjax import energy, poisson
    vclmt, _ = poisson.coulomb_potential(silicon)
    good = float(energy.madelung(vclmt, silicon))
    bare = sum(float(silicon["spzn"][int(silicon["idxis"][ias]) - 1])
               * float(vclmt[ias][0, 0]) * 0.28209479177387814347 / 2.0
               for ias in range(int(silicon["natmtot"])))
    assert abs(good - float(silicon["engymad"])) / abs(good) < 1e-14
    assert abs(bare) > 1e4 * abs(good)
