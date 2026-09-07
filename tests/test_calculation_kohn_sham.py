r"""The Kohn-Sham potential, end to end from Elk's own density.

Sections 2a-2g each verified one piece against Elk, and 2f verified that the
pieces are consistent inside an INTEGRAL.  Nothing has checked the composition
pointwise, which is what an SCF iteration actually consumes:

    v_s = v_cl[rho]  +  S v_xc[rho]

with v_cl from the Weinert solve (2e), v_xc from the functional applied on the
angular grid (2a/2d), and S the symmetrisation Elk applies to the potential and
not to the energy densities (2g).  Every term on the right is built here; the
left is Elk's own.

Two references, and they are not the same check.  In the muffin tin Elk stores
`vclmt` and `vxcmt` separately, so `vsmt` is their sum by construction and
comparing against it adds nothing beyond 2e and 2g.  In the interstitial Elk
stores `vsir` itself -- formed inside `potks` after `trimrfg` has been applied
to `vxcir` and not to `vclir` -- so that IS an independent reference, and it is
the one that would catch trimming the wrong term.

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


def _ground_state(workdir, structure, ngridk):
    calculation = structure.get_calculation(workdir, xc="PW", ngridk=ngridk,
                                            rgkmax=7.0)
    calculation.ensure_ground_state()
    with calculation.eigenstate_session() as session:
        return session.ground_state()


@pytest.fixture(scope="module")
def silicon(tmp_path_factory):
    return _ground_state(
        tmp_path_factory.mktemp("ks") / "si",
        Structure(avec=SI_AVEC,
                  species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}),
        (2, 2, 2))


@pytest.fixture(scope="module")
def hbn(tmp_path_factory):
    return _ground_state(
        tmp_path_factory.mktemp("ks") / "hbn",
        Structure(avec=HBN_AVEC,
                  species={"B": [(0.0, 0.0, 0.0)],
                           "N": [(1 / 3, 2 / 3, 0.0)]}),
        (2, 2, 1))


def _fixture(request, name):
    return request.getfixturevalue(name)


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_the_muffin_tin_potential_composes(request, name):
    """v_cl + S v_xc against Elk's own `vclmt + vxcmt`, pointwise.

    Both halves come from a different module -- `elkjax.poisson` and
    `elkjax.energy` plus `elkjax.symmetry` -- driven from the same density, and
    neither has been asked before to land in the same array as the other.
    """
    import jax.numpy as jnp
    from elkjax import energy, poisson, symmetry
    groundstate = _fixture(request, name)
    natmtot = int(groundstate["natmtot"])

    coulomb, _ = poisson.coulomb_potential(groundstate)
    xc_packed = energy.kohn_sham_potentials(groundstate,
                                            symmetrise=True)["vxcmt"]
    for ias in range(natmtot):
        n = int(groundstate["npmt"][int(groundstate["idxis"][ias]) - 1])
        mine = (np.asarray(poisson.pack(coulomb[ias], groundstate, ias))[:n]
                + np.asarray(xc_packed[ias])[:n])
        reference = (np.asarray(groundstate["vclmt"][ias])[:n]
                     + np.asarray(groundstate["vxcmt"][ias])[:n])
        assert np.abs(mine - reference).max() \
            / np.abs(reference).max() < 1e-14
        assert jnp.isfinite(jnp.asarray(mine)).all()


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_the_interstitial_potential_matches_elks_own_vsir(request, name):
    """v_cl + trim(v_xc) against `vsir`, which Elk forms itself.

    This is the independent one: `potks` applies `trimrfg` to `vxcir` and NOT
    to `vclir` before adding them, so a chain that trimmed the sum, or neither
    term, or the Coulomb one, would still be smooth and would still integrate
    correctly against rho -- and would miss here.
    """
    from elkjax import energy, poisson
    groundstate = _fixture(request, name)
    _, coulomb = poisson.coulomb_potential(groundstate)
    exchange = energy.kohn_sham_potentials(groundstate)["vxcir"]
    mine = np.asarray(coulomb) + np.asarray(exchange)
    reference = np.asarray(groundstate["vsir"])
    assert np.abs(reference).max() > 1e-3
    assert np.abs(mine - reference).max() / np.abs(reference).max() < 1e-13


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_trimming_the_wrong_term_is_visible_only_here(request, name):
    """The mutation the `vsir` reference exists to catch.

    Applying `trimrfg` to the Coulomb potential as well leaves a smooth
    interstitial potential of the right magnitude -- it removes high-|G|
    content that is small -- and it is wrong.  Asserted to be BOTH wrong
    against `vsir` and small enough that a loose tolerance would have missed
    it, which is why the tolerance above is 1e-13.
    """
    from elkjax import energy, grid, poisson
    groundstate = _fixture(request, name)
    _, coulomb = poisson.coulomb_potential(groundstate)
    exchange = energy.kohn_sham_potentials(groundstate)["vxcir"]
    reference = np.asarray(groundstate["vsir"])
    mutant = np.asarray(grid.trim(coulomb, groundstate)) + np.asarray(exchange)
    error = np.abs(mutant - reference).max() / np.abs(reference).max()
    assert error > 1e-12, "the mutation no longer changes anything"
    assert error < 1e-2, (
        "the mutation is now large enough that any tolerance would catch it; "
        "the point of this test is that it is not")
