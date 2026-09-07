"""`elkjax.scf`'s host-side pieces: no Elk run, no ground state.

The refusal and the packing are the parts that must hold before any of the
physics runs, and neither needs a reference.

Self-skips without jax.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="jax not installed; pip install -e .[jax]")


def test_a_spin_polarised_ground_state_is_refused():
    """`eveqnsv` is not transcribed, so `evalsv` is not `evalfv`.

    Elk's own marker for a scalar calculation is `nstsv == nstfv`; with spin
    polarisation or spin-orbit coupling the second-variational step doubles
    the state count and mixes the first-variational states.  Treating those
    eigenvalues as the spectrum would give a plausible, wrong Fermi level, so
    this raises rather than proceeding.
    """
    from elkjax import scf
    scf.check_scalar({"nstfv": 13, "nstsv": 13, "nspnfv": 1})
    with pytest.raises(ValueError, match="eveqnsv"):
        scf.check_scalar({"nstfv": 13, "nstsv": 26, "nspnfv": 1})
    with pytest.raises(ValueError, match="spin-spiral"):
        scf.check_scalar({"nstfv": 13, "nstsv": 13, "nspnfv": 2})


def test_the_potential_packing_round_trips():
    """`mixpack`: the two halves of the potential as one flat vector.

    The muffin-tin half is a stack of packed per-atom arrays and the
    interstitial half is a grid, so the only thing that can go wrong is the
    split point -- which is why this checks the halves and not just the norm.
    """
    from elkjax import scf
    rng = np.random.default_rng(3)
    vsmt = rng.normal(size=(3, 40))
    vsir = rng.normal(size=17)
    got_mt, got_ir = scf.unpack(scf.pack(vsmt, vsir), vsmt.shape, vsir.size)
    assert np.array_equal(np.asarray(got_mt), vsmt)
    assert np.array_equal(np.asarray(got_ir), vsir)


def test_the_entropy_term_is_zero_for_every_smearing_but_fermi_dirac():
    """`energy.f90:242` sets `engyts = 0` unless `stype == 3`.

    The study names this as the source of a force error one would otherwise
    chase for a week, so the branch is transcribed rather than assumed away --
    and asserted, because a term that is *supposed* to vanish is the one a
    transcription silently gets right for the wrong reason.
    """
    from elkjax import energy
    occsv = np.array([[2.0, 1.4, 0.6, 0.0]])
    wkpt = np.array([1.0])
    fermi = float(energy.entropy_term(occsv, wkpt, 0.01, 2.0, 3))
    assert fermi < -1e-6, "the Fermi-Dirac entropy term should be negative"
    for stype in (0, 1, 2):
        assert float(energy.entropy_term(occsv, wkpt, 0.01, 2.0, stype)) == 0.0


def test_the_entropy_term_survives_integer_occupations():
    """Full and empty states contribute nothing and their logarithms do not
    exist, so the argument must be made safe BEFORE the log -- a `jnp.where`
    evaluates both branches and a NaN in the discarded one still reaches the
    gradient.  A gapped insulator has nothing but integer occupations.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import energy
    occsv = jnp.asarray([[2.0, 2.0, 0.0, 0.0]])
    wkpt = jnp.asarray([1.0])
    value = energy.entropy_term(occsv, wkpt, 0.01, 2.0, 3)
    assert float(value) == 0.0
    slope = jax.grad(lambda o: energy.entropy_term(o, wkpt, 0.01, 2.0, 3))(occsv)
    assert jnp.isfinite(slope).all()


def test_the_eigenvalue_sum_carries_the_weights_and_the_core():
    """`evalsum`: a weighted zone sum plus an imported core scalar."""
    from elkjax import energy
    evalsv = np.array([[-1.0, 0.5], [-2.0, 0.25]])
    occsv = np.array([[2.0, 0.0], [2.0, 1.0]])
    wkpt = np.array([0.25, 0.75])
    got = float(energy.eigenvalue_sum(evalsv, occsv, wkpt, -10.0))
    expected = 0.25 * (2 * -1.0) + 0.75 * (2 * -2.0 + 1 * 0.25) - 10.0
    assert got == pytest.approx(expected, rel=1e-15)
