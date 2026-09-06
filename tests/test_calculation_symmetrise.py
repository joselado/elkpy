r"""`symrfmt`, the muffin-tin symmetrisation Elk applies to v_xc but not to e_xc.

Section 2d found that `potxc.f90:55-58` symmetrises `vxcmt` and `bxcmt` and not
`exmt`/`ecmt`, so applying the functional pointwise reproduces the energy
densities exactly and misses `vxcmt` by 1.2e-4 relative.  Section 2f then found
that this costs nothing in any integral against rho, S being an orthogonal
projection with rho in its range -- but an SCF iteration compares potentials
POINTWISE, so closing the loop needs the operator itself.  This file is that
operator, and the measurement that it closes the gap: 5.3e-3 to 6.4e-14.

**The operator is exported, not transcribed** (patch 0018), and that is the
better route rather than the cheaper one.  `rotrflm`'s Euler-angle and Wigner-D
construction has no consumer inside Elk but `symrfmt`, so a Python re-derivation
would have no independent check except agreement with the thing it replaces --
and Elk's atom bookkeeping would have to come with it: `ieqatom`, `tfeqat`, and
the INVERSE lattice rotation in the rotate-into-equivalent-atoms loop.  None of
that is transcribed here, so none of it can be got wrong here; what these tests
check is the operator's APPLICATION, which has no convention in it.

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


def _ground_state(workdir, structure, ngridk, extra_blocks=None):
    calculation = structure.get_calculation(
        workdir, xc="PW", ngridk=ngridk, rgkmax=7.0, extra_blocks=extra_blocks)
    calculation.ensure_ground_state()
    with calculation.eigenstate_session() as session:
        return session.ground_state()


@pytest.fixture(scope="module")
def silicon(tmp_path_factory):
    """Two equivalent atoms, so the operator mixes SITES as well as harmonics --
    which is what makes this fixture test the atom-permutation half at all.
    `symop[0, 1]` has entries of 0.5 here; on a cell where every atom is alone
    in its species the off-diagonal blocks vanish and this file would only be
    testing an angular rotation."""
    return _ground_state(
        tmp_path_factory.mktemp("symmetrise") / "si",
        Structure(avec=SI_AVEC,
                  species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}),
        (2, 2, 2))


@pytest.fixture(scope="module")
def hbn(tmp_path_factory):
    return _ground_state(
        tmp_path_factory.mktemp("symmetrise") / "hbn",
        Structure(avec=HBN_AVEC,
                  species={"B": [(0.0, 0.0, 0.0)],
                           "N": [(1 / 3, 2 / 3, 0.0)]}),
        (2, 2, 1))


@pytest.fixture(scope="module")
def unsymmetrised(tmp_path_factory):
    return _ground_state(
        tmp_path_factory.mktemp("symmetrise") / "si_nosym",
        Structure(avec=SI_AVEC,
                  species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}),
        (2, 2, 2), extra_blocks={"symtype": [0]})


def _fixture(request, name):
    return request.getfixturevalue(name)


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_the_operator_is_a_projection(request, name):
    """S^2 = S, on the exported matrices alone.

    A group average is idempotent, so this catches a missing operation or a
    double-counted one without needing a ground state, a functional, or Elk's
    own `vxcmt`.  It is the only check here that does not go through the
    comparison the rest of the file makes.
    """
    from elkjax import symmetry
    assert symmetry.is_projection(_fixture(request, name))


def test_idempotence_is_exact_on_a_cubic_lattice_and_not_on_a_hexagonal_one(
        silicon, hbn):
    """The contrast, pinned as a contrast rather than absorbed into a loose
    tolerance -- because it is a property of ELK's operator, not of this
    module.

    `symrfmt` builds each rotation through `roteuler`, which extracts Euler
    angles from the Cartesian `symlatc` by inverse trigonometry.  On a cubic
    lattice those entries are exactly 0 and +-1 and the angles are exact
    multiples of pi/2; on a hexagonal one they carry 1/2 and sqrt(3)/2 and they
    are not.  Measured: 1e-16 on silicon against 1.2e-11 on h-BN, five orders
    apart, with the h-BN residual growing from 2.5e-12 at l=1 to 1.2e-11 at
    l=5 as the Wigner-D order compounds the angle error.

    A single tolerance covering both would say the operator is good to 1e-10
    everywhere, which is wrong in one direction and uninformative in the other.
    """
    from elkjax import symmetry
    cubic = symmetry.idempotence_residual(silicon)
    hexagonal = symmetry.idempotence_residual(hbn)
    assert cubic < 1e-14
    assert 1e-14 < hexagonal < 1e-9
    assert hexagonal > 1e3 * cubic


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_the_density_is_already_symmetric(request, name):
    """S rho = rho -- section 2d's premise, asserted rather than assumed.

    The whole argument for why the pointwise v_xc differs from Elk's is that S
    is not the identity on v_xc[rho] even though rho is in its range.  If rho
    were NOT symmetric, the discrepancy would have a different cause and the
    orthogonality result of section 2f would not follow either.
    """
    import jax.numpy as jnp
    from elkjax import poisson, symmetry
    groundstate = _fixture(request, name)
    dense = jnp.stack([poisson.dense(groundstate["rhomt"][ias], groundstate,
                                     ias)
                       for ias in range(int(groundstate["natmtot"]))])
    moved = symmetry.symmetrise(dense, groundstate)
    assert float(jnp.abs(moved - dense).max()) \
        / float(jnp.abs(dense).max()) < 1e-15


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_symmetrising_closes_the_pointwise_gap(request, name):
    """The result: 5.3e-3 before, 6.4e-14 after.

    Both halves are asserted.  Without the "before" this would pass on a cell
    with no symmetry to apply, where S is the identity and the pointwise
    potential was already right.
    """
    import jax.numpy as jnp
    from elkjax import energy, poisson, symmetry
    groundstate = _fixture(request, name)
    natmtot = int(groundstate["natmtot"])
    pointwise = jnp.stack([
        poisson.dense(energy.kohn_sham_potentials(groundstate)["vxcmt"][ias],
                      groundstate, ias)
        for ias in range(natmtot)])
    moved = symmetry.symmetrise(pointwise, groundstate)

    before = after = 0.0
    scale = 0.0
    for ias in range(natmtot):
        elk = np.asarray(poisson.dense(groundstate["vxcmt"][ias], groundstate,
                                       ias))
        scale = max(scale, np.abs(elk).max())
        before = max(before, np.abs(np.asarray(pointwise[ias]) - elk).max())
        after = max(after, np.abs(np.asarray(moved[ias]) - elk).max())
    assert before / scale > 1e-5, "no gap to close on this fixture"
    assert after / scale < 1e-14


def test_without_crystal_symmetry_the_operator_is_the_identity(unsymmetrised):
    """`symtype = 0` leaves one operation, so S must be exactly the identity --
    including zero between atoms, which is the part a wrong normalisation
    (dividing by the wrong `nsymcrys`) would break while leaving the diagonal
    plausible."""
    from elkjax import symmetry
    operator = np.asarray(unsymmetrised["symop"])
    natmtot, _, n, _ = operator.shape
    for ias in range(natmtot):
        for jas in range(natmtot):
            expected = np.eye(n) if ias == jas else np.zeros((n, n))
            assert np.abs(operator[ias, jas] - expected).max() < 1e-14
    assert symmetry.is_projection(unsymmetrised)


def test_the_packed_round_trip_preserves_the_inner_region(silicon):
    """`symmetrise_packed` must not leak weight into harmonics the inner region
    does not store.  The operator does not mix l, so its top-left block is
    closed on them -- but a transcription that applied the full lmmaxo matrix
    to the inner region would silently write past what Elk packs there, and the
    packed round trip is what notices."""
    import jax.numpy as jnp
    from elkjax import energy, symmetry
    packed = energy.kohn_sham_potentials(silicon)["vxcmt"]
    moved = symmetry.symmetrise_packed(packed, silicon)
    for ias in range(int(silicon["natmtot"])):
        n = int(silicon["npmt"][int(silicon["idxis"][ias]) - 1])
        reference = np.asarray(silicon["vxcmt"][ias])[:n]
        assert np.abs(np.asarray(moved[ias])[:n] - reference).max() \
            / np.abs(reference).max() < 1e-14
    assert jnp.asarray(moved).shape[1] >= int(silicon["npmt"].max())


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_symmetrising_reproduces_elks_potential_through_the_energy_module(
        request, name):
    """`kohn_sham_potentials(symmetrise=True)` gives Elk's own `vxcmt`.

    The composition check: section 2g's operator wired into the place an SCF
    iteration would take the potential from.  Without the flag the same call
    misses by 5.3e-3, which the file above measures directly on the raw arrays;
    this asserts the wiring, not the operator.
    """
    from elkjax import energy
    groundstate = _fixture(request, name)
    moved = np.asarray(
        energy.kohn_sham_potentials(groundstate, symmetrise=True)["vxcmt"])
    for ias in range(int(groundstate["natmtot"])):
        n = int(groundstate["npmt"][int(groundstate["idxis"][ias]) - 1])
        reference = np.asarray(groundstate["vxcmt"][ias])[:n]
        assert np.abs(moved[ias, :n] - reference).max() \
            / np.abs(reference).max() < 1e-14


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_symmetrising_does_not_change_the_energy(request, name):
    """Section 2f's orthogonality, seen from the other side.

    E_vxc is an integral against rho, and the leak S removes lives in the
    harmonics rho does not have, so switching the operator on must change
    nothing -- even though it changes the potential by 5.3e-3 pointwise.  This
    is the same fact as the overlap measurement in test_calculation_energy.py,
    but stated where someone reaching for `symmetrise=True` will see it, and it
    would catch a wiring error that the pointwise test above cannot: applying
    the operator to the ENERGY densities too, which Elk does not do.
    """
    from elkjax import energy
    groundstate = _fixture(request, name)
    plain = energy.terms(groundstate, symmetrise=False)
    moved = energy.terms(groundstate, symmetrise=True)
    for term in ("engyvxc", "engyx", "engyc", "engytot"):
        assert abs(float(plain[term]) - float(moved[term])) \
            / abs(float(plain[term])) < 1e-14, term


@pytest.mark.parametrize("name", ["silicon", "hbn"])
def test_the_operator_does_not_mix_angular_momenta(request, name):
    """Exactly block-diagonal in l -- the structural fact the inner/outer split
    rests on.

    `elkjax.symmetry.symmetrise` applies the operator's top-left lmmaxi block
    to the inner region, which is only legitimate because a rotation cannot
    take an l <= lmaxi harmonic into an l > lmaxi one.  Measured as EXACTLY
    zero, not small: a rotation acts within each (2l+1)-dimensional irreducible
    space, so any nonzero entry off the block diagonal would mean the exported
    matrices are not rotations at all.
    """
    from elkjax import symmetry  # noqa: F401  (import kept for symmetry-module parity)
    groundstate = _fixture(request, name)
    operator = np.asarray(groundstate["symop"])
    lmaxo = int(groundstate["lmaxo"])
    for l1 in range(lmaxo + 1):
        for l2 in range(lmaxo + 1):
            if l1 == l2:
                continue
            block = operator[:, :, l1 * l1:(l1 + 1) ** 2,
                             l2 * l2:(l2 + 1) ** 2]
            assert np.abs(block).max() == 0.0, (l1, l2)
