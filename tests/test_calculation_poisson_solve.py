r"""Phase 2 item 2c: Elk's Weinert Poisson solve, in JAX (`elkjax.poisson`).

The Coulomb potential is the one ingredient of the Kohn-Sham potential that no
export supplies as a function of the density -- `elkjax.xc` covers exchange and
correlation pointwise, but `vclmt`/`vclir` come out of `potcoul` and nowhere
else, so without this the SCF loop cannot close.

The endpoints are checked separately, and the muffin-tin one split by l, because
the two halves fail for different reasons: `vclir` is the G-space pseudocharge
solve and `vclmt`'s l=0 channel is dominated by the nucleus, which would hide
an error in every other harmonic if they were compared together.

Two checks owe Elk's own `vclmt` nothing, which matters because the rest of the
file is a comparison against it:

  * the MONOPOLE identity.  `zpotcoul` reads q_lm off the sphere-BOUNDARY value
    of the intra-sphere potential, after the nucleus has been added, so
    sqrt(4pi) q_00 must equal N_MT - Z with N_MT from an entirely separate code
    path (`elkjax.integrate`, `rfmtint`) and Z a known integer.  It comes out
    exact.
  * two MUTATION tests, each removing one thing Elk does and asserting the
    answer moves.  Both survive every structural check -- Hermiticity, sign,
    smoothness -- while being wrong, which is why they are pinned rather than
    argued.

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
        tmp_path_factory.mktemp("poisson") / "si",
        Structure(avec=SI_AVEC,
                  species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}),
        (2, 2, 2))


@pytest.fixture(scope="module")
def hbn(tmp_path_factory):
    """Two species with different meshes and different nuclear charges, which
    silicon's single species cannot exercise: every per-species array here
    (`wprmt`, `vcln`, the Bessel table) is indexed by `idxis`, and a
    transcription that indexes by atom instead is invisible on one species."""
    return _ground_state(
        tmp_path_factory.mktemp("poisson") / "hbn",
        Structure(avec=HBN_AVEC,
                  species={"B": [(0.0, 0.0, 0.0)],
                           "N": [(1 / 3, 2 / 3, 0.0)]}),
        (2, 2, 1))


def _fixture(request, name):
    return request.getfixturevalue(name)


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_the_harmonic_transform_round_trips(request, name):
    """`ztorfmt(rtozfmt(f)) = f`.  Necessary and nowhere near sufficient: the
    inverse here is built by inverting the same 2x2 blocks, so this cannot
    catch a wrong convention, only a wrong index.  What catches the convention
    is `vclir`, since the multipoles are contracted against `ylmg` in the
    complex basis."""
    from elkjax import poisson
    groundstate = _fixture(request, name)
    lmaxo = int(groundstate["lmaxo"])
    for ias in range(int(groundstate["natmtot"])):
        rho = poisson.dense(groundstate["rhomt"][ias], groundstate, ias)
        back = poisson.complex_to_real(
            poisson.real_to_complex(rho, lmaxo), lmaxo)
        assert np.abs(np.asarray(back) - np.asarray(rho)).max() < 1e-15


@pytest.mark.parametrize("name,charges", [("silicon", [14, 14]),
                                          ("hbn", [5, 7])])
def test_the_monopole_is_the_enclosed_charge(request, name, charges):
    r"""sqrt(4pi) q_00 = N_MT - Z, against a separate code path and an integer.

    This is the one check on the intra-sphere half that does not compare
    against Elk.  It also pins the ORDER of the chain: the nucleus is added
    before the multipoles are read, so if `add_nuclear` moved after
    `multipoles` this would come out as +N_MT with no minus sign anywhere.
    """
    import jax.numpy as jnp
    from elkjax import integrate, poisson
    groundstate = _fixture(request, name)
    lmaxo = int(groundstate["lmaxo"])
    potentials = []
    for ias in range(int(groundstate["natmtot"])):
        rho = poisson.real_to_complex(
            poisson.dense(groundstate["rhomt"][ias], groundstate, ias), lmaxo)
        potentials.append(poisson.add_nuclear(
            poisson.intra_sphere(rho, groundstate, ias), groundstate, ias))
    qlm = np.asarray(poisson.multipoles(jnp.stack(potentials), groundstate))
    for ias, z in enumerate(charges):
        electrons = float(integrate.muffin_tin_integral(
            groundstate["rhomt"][ias], groundstate, ias))
        monopole = np.sqrt(4.0 * np.pi) * qlm[ias, 0].real
        assert abs(electrons - monopole - z) < 1e-8


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_the_interstitial_potential_matches_elk(request, name):
    """`vclir`, which is where a wrong complex-harmonic convention shows up:
    the multipoles are contracted against `ylmg` here, and a self-consistent
    but wrong convention survives the intra-sphere solve untouched."""
    from elkjax import poisson
    groundstate = _fixture(request, name)
    _, vclir = poisson.coulomb_potential(groundstate)
    reference = np.asarray(groundstate["vclir"])
    scale = np.abs(reference).max()
    assert np.abs(np.asarray(vclir) - reference).max() / scale < 1e-12


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_the_muffin_tin_potential_matches_elk(request, name):
    """`vclmt`, split by l.

    The l=0 channel carries the nucleus and is of order 1e7, so comparing the
    whole array at once would let any error in the other 48 harmonics -- whose
    scale is 1e-1 -- pass at any relative tolerance worth stating.
    """
    from elkjax import poisson
    groundstate = _fixture(request, name)
    vclmt, _ = poisson.coulomb_potential(groundstate)
    for ias in range(int(groundstate["natmtot"])):
        reference = np.asarray(
            poisson.dense(groundstate["vclmt"][ias], groundstate, ias))
        got = np.asarray(vclmt[ias])
        difference = np.abs(got - reference)
        assert difference[:, 0].max() / np.abs(reference[:, 0]).max() < 1e-15
        assert difference[:, 1:].max() / np.abs(reference[:, 1:]).max() < 1e-12


def test_dropping_the_nucleus_moves_the_multipoles(silicon):
    """A mutation test on the step order.

    `potcoul` adds `vcln` BEFORE `zpotcoul` reads the boundary multipoles, so
    q_00 is the moment of the TOTAL charge.  Skipping it leaves a perfectly
    smooth, perfectly plausible intra-sphere potential and a q_00 of the wrong
    sign; nothing structural notices.
    """
    import jax.numpy as jnp
    from elkjax import poisson
    lmaxo = int(silicon["lmaxo"])
    with_nucleus, without = [], []
    for ias in range(int(silicon["natmtot"])):
        rho = poisson.real_to_complex(
            poisson.dense(silicon["rhomt"][ias], silicon, ias), lmaxo)
        bare = poisson.intra_sphere(rho, silicon, ias)
        without.append(bare)
        with_nucleus.append(poisson.add_nuclear(bare, silicon, ias))
    good = np.asarray(poisson.multipoles(jnp.stack(with_nucleus), silicon))
    bad = np.asarray(poisson.multipoles(jnp.stack(without), silicon))
    assert good[0, 0].real < 0.0 < bad[0, 0].real


def test_the_outer_region_uses_its_own_spline_weights(silicon):
    """A mutation test on the region split.

    For l > lmaxi the inner region does not store the harmonic at all, so Elk
    integrates from r_iro outward with the SUB-MESH's own weights.  Padding
    with zeros and integrating from the origin uses the wrong weights at the
    lower boundary; the result is smooth and of the right order and differs
    from Elk in the fifth digit.
    """
    from elkjax import poisson
    lmaxi, lmaxo = int(silicon["lmaxi"]), int(silicon["lmaxo"])
    assert lmaxo > lmaxi, "this fixture cannot exercise the split"
    isp = int(silicon["idxis"][0]) - 1
    nr = int(silicon["nrmt"][isp])
    r = np.asarray(silicon["rlmt"][isp][:nr])
    weights = np.asarray(silicon["wprmt"][isp].T[:nr])
    rho = poisson.real_to_complex(
        poisson.dense(silicon["rhomt"][0], silicon, 0), lmaxo)

    lm, l = 10, 3                       # an l > lmaxi channel with real weight
    correct = np.asarray(poisson.intra_sphere(rho, silicon, 0))[:, lm]
    padded = np.asarray(
        poisson._solve_channel(r, weights, rho[:, lm], l))
    scale = np.abs(correct).max()
    assert scale > 1e-8
    assert np.abs(padded - correct).max() / scale > 1e-6
