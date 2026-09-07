r"""`occupy.f90`: the zone-summed Fermi level and the occupation numbers.

Phase 1i built mu at ONE k-point, where the k-point weights cancel.  Study
section 8(b)'s rule has weights in it, and nothing had tested that half.  A
self-consistent field cannot start without it: the two half-steps of Phase 2
(density -> potential, potential -> eigenvectors -> density) both exist and
compose, and the occupations are what stands between them.

The reference here is Elk's own `efermi` and `occsv` evaluated on Elk's own
`evalsv` -- patch 0023 exports all three -- so the assembly is NOT in the path.
That is deliberate: with the Hamiltonian in the chain, a wrong bisection and a
wrong H are the same symptom.

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
AL_A = 7.6534 / 2
AL_AVEC = [(0.0, AL_A, AL_A), (AL_A, 0.0, AL_A), (AL_A, AL_A, 0.0)]


def _run(workdir, structure, ngridk):
    calculation = structure.get_calculation(workdir, xc="PW", ngridk=ngridk,
                                            rgkmax=7.0)
    calculation.ensure_ground_state()
    with calculation.eigenstate_session() as session:
        return session.density_k()


@pytest.fixture(scope="module")
def silicon(tmp_path_factory):
    """Elk's default reduced mesh: 3 k-points out of 8, so the weights are NOT
    uniform -- which is what makes the weighted sum testable at all."""
    return _run(tmp_path_factory.mktemp("occ") / "si",
                Structure(avec=SI_AVEC,
                          species={"Si": [(0.0, 0.0, 0.0),
                                          (0.25, 0.25, 0.25)]}),
                (2, 2, 2))


@pytest.fixture(scope="module")
def aluminium(tmp_path_factory):
    """fcc Al: one atom, and a real metal -- the Fermi level is fixed by a
    finite density of states rather than by roundoff inside a gap, which is
    what the derivative rule needs."""
    return _run(tmp_path_factory.mktemp("occ") / "al",
                Structure(avec=AL_AVEC, species={"Al": [(0.0, 0.0, 0.0)]}),
                (4, 4, 4))


@pytest.mark.parametrize("name", ["silicon", "aluminium"])
def test_the_fermi_level_and_occupations_match_elks_own(request, name):
    """`occupy` on Elk's own eigenvalues, against Elk's own answer.

    Measured, both are reproduced BITWISE on the two fixtures -- every one of
    the bisection's ~45 comparisons goes the same way, which is a stronger
    statement than the tolerance asserted here and the reason the tolerance can
    be the bisection's own bracket width rather than something comfortable.
    """
    from elkjax import occupations
    densityk = request.getfixturevalue(name)
    mu, occsv = occupations.occupy(np.asarray(densityk["evalsv"]), densityk)
    assert abs(float(mu) - float(densityk["efermi"])) < 1e-12
    assert np.abs(np.asarray(occsv)
                  - np.asarray(densityk["occsv"])).max() < 1e-14


@pytest.mark.parametrize("name", ["silicon", "aluminium"])
def test_the_count_the_bisection_solves_for_is_elks_chgval(request, name):
    """The constraint itself, not just its root.

    `chgval` is the VALENCE electron count and is not an integer in Elk (it
    carries its own tiny offset), so taking it from the export rather than from
    the species is load-bearing.
    """
    from elkjax import occupations
    densityk = request.getfixturevalue(name)
    got = occupations.inputs(densityk)
    _, occsv = occupations.occupy(np.asarray(densityk["evalsv"]), densityk)
    count = float(occupations.electron_count(occsv, got["wkpt"]))
    assert abs(count - got["chgval"]) < 1e-9


def test_the_k_point_weights_do_not_cancel(silicon):
    """The half of study 8(b)'s rule a single k-point cannot test.

    Elk reduces the 2x2x2 mesh to 3 k-points with unequal weights.  Replacing
    them with uniform ones -- the shape the Phase 1 single-k rule has -- moves
    the Fermi level.  Asserted to move, so that "the weights are wired in"
    cannot regress into "the weights happened to be equal".
    """
    from elkjax import occupations
    weights = np.asarray(silicon["wkpt"])
    assert weights.size < 8, "this fixture is not actually reduced"
    assert weights.std() > 1e-12, "this fixture's weights are uniform"

    got = occupations.inputs(silicon)
    evalsv = np.asarray(silicon["evalsv"])
    right = float(occupations.fermi_level(
        evalsv, weights, got["chgval"], got["swidth"], got["occmax"],
        got["e0min"], got["stype"]))
    uniform = float(occupations.fermi_level(
        evalsv, np.full(weights.size, 1.0 / weights.size), got["chgval"],
        got["swidth"], got["occmax"], got["e0min"], got["stype"]))
    assert abs(right - uniform) > 1e-4


def test_dmu_agrees_in_both_modes_and_with_finite_differences(aluminium):
    """d(mu)/d(evalsv) on a real metal: forward, reverse and central FD.

    Forward and reverse are the same number for a scalar-in scalar-out
    function, so their disagreement would be proof on its own and costs
    nothing.  The FD arbiter is legitimate here because mu is a smooth
    function of the spectrum away from a gap -- which is exactly what a metal
    supplies and what Si at Elk's default smearing does not.

    The direction is a general one, not a single entry: a sparse direction is
    the trap Phase 0b measured, where a wrong rule looks right.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import occupations
    got = occupations.inputs(aluminium)
    evalsv = jnp.asarray(np.asarray(aluminium["evalsv"]))

    def mu(values):
        return occupations.fermi_level(
            values, got["wkpt"], got["chgval"], got["swidth"], got["occmax"],
            got["e0min"], got["stype"])

    direction = jnp.asarray(
        np.random.default_rng(20260907).normal(size=evalsv.shape))
    forward = float(jax.jvp(mu, (evalsv,), (direction,))[1])
    reverse = float(jnp.sum(jax.grad(mu)(evalsv) * direction))
    step = 1e-6
    difference = float((mu(evalsv + step * direction)
                        - mu(evalsv - step * direction)) / (2 * step))

    assert abs(forward) > 1e-3, "the direction barely moves mu; not a test"
    assert abs(forward - reverse) < 1e-13 * abs(forward)
    assert abs(forward - difference) < 1e-6 * abs(forward)


def test_a_gapped_spectrum_refuses_rather_than_returning_a_ratio(silicon):
    """The denominator vanishing is physics, and it is refused.

    Elk's default `swidth` leaves Si's conduction band about seven widths
    above mu on this mesh, so the response is small but finite.  Narrow the
    smearing and it underflows to exactly zero: no state responds, mu may be
    placed anywhere in the gap, and d(mu) is genuinely 0/0.  The forward
    Fermi level is unaffected -- only the derivative is refused.
    """
    from elkjax import occupations
    got = occupations.inputs(silicon)
    evalsv = np.asarray(silicon["evalsv"])
    arguments = (evalsv, got["wkpt"], got["chgval"])

    mu, total = occupations.check_fermi_level_determined(
        *arguments, got["swidth"], got["occmax"], got["e0min"], got["stype"])
    assert total > 1e-8 and np.isfinite(mu)

    with pytest.raises(ValueError, match="does not determine the Fermi level"):
        occupations.check_fermi_level_determined(
            *arguments, 1e-4, got["occmax"], got["e0min"], got["stype"])
