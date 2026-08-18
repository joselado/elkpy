"""Synthetic pins for the four-state energy-mapping arithmetic
(parsers/exchange.py). No Elk binary involved: every energy here comes from
an explicit classical spin model, so the extraction can be checked against a
tensor that is known exactly.

The central test is a round trip -- build a model with a known exchange
tensor on the target pair PLUS the things the method claims to cancel
(single-ion anisotropy, other pairs, spectator-spectator terms), generate the
four energies the way elkpy will, and check the tensor comes back exactly.
That is what the cancellation argument in docs/design.md #29 asserts, tested
numerically rather than trusted.
"""

import numpy as np
import pytest

from elkpy.parsers import exchange as ex


def model_energy(spins, pair_tensors, site_tensors=None):
    """E = sum_{i<j} S_i . J_ij . S_j + sum_i S_i . A_i . S_i, in Hartree."""
    energy = 0.0
    for (i, j), tensor in pair_tensors.items():
        energy += spins[i] @ np.asarray(tensor) @ spins[j]
    for i, tensor in (site_tensors or {}).items():
        energy += spins[i] @ np.asarray(tensor) @ spins[i]
    return energy


def four_state_energies(a, b, i, j, spectators, pair_tensors, site_tensors, spin):
    """The four total energies elkpy's configurations would produce."""
    energies = []
    for directions in ex.four_state_directions(a, b, spin=spin):
        spins = {k: np.asarray(directions["spectator"]) for k in spectators}
        spins[i] = directions["i"]
        spins[j] = directions["j"]
        energies.append(model_energy(spins, pair_tensors, site_tensors))
    return energies


TARGET = np.array([
    [1.0e-4, 3.0e-5, -2.0e-5],
    [-1.0e-5, 2.0e-4, 4.0e-5],
    [5.0e-5, -3.0e-5, 3.0e-4],
])


def test_round_trip_recovers_a_known_tensor():
    """Every one of the nine components, with nothing else in the model."""
    spin = 1.0
    quads = {
        (a, b): four_state_energies(a, b, 0, 1, [2, 3], {(0, 1): TARGET}, {}, spin)
        for a in range(3)
        for b in range(3)
    }
    recovered = ex.assemble_tensor(quads, spin=spin)
    assert np.allclose(recovered, TARGET * ex.HARTREE_TO_MEV, rtol=1e-10, atol=1e-10)


def test_single_ion_anisotropy_and_other_pairs_cancel():
    """The claim that makes the method usable: adding single-ion anisotropy on
    every site, couplings from the pair to the spectators, and couplings
    among the spectators themselves leaves the extracted tensor untouched."""
    spin = 1.5
    spectators = [2, 3]
    rng = np.random.default_rng(20260818)
    contaminants = {
        (0, 2): rng.normal(scale=1e-4, size=(3, 3)),
        (0, 3): rng.normal(scale=1e-4, size=(3, 3)),
        (1, 2): rng.normal(scale=1e-4, size=(3, 3)),
        (1, 3): rng.normal(scale=1e-4, size=(3, 3)),
        (2, 3): rng.normal(scale=1e-4, size=(3, 3)),
    }
    site_tensors = {k: rng.normal(scale=1e-4, size=(3, 3)) for k in range(4)}
    pair_tensors = {(0, 1): TARGET, **contaminants}
    quads = {
        (a, b): four_state_energies(a, b, 0, 1, spectators, pair_tensors, site_tensors, spin)
        for a in range(3)
        for b in range(3)
    }
    recovered = ex.assemble_tensor(quads, spin=spin)
    assert np.allclose(recovered, TARGET * ex.HARTREE_TO_MEV, rtol=1e-9, atol=1e-9)


def test_multiplicity_divides_out():
    """A pair whose periodic images make it m equivalent bonds gives m times
    the energy difference, and must be divided by m."""
    spin = 1.0
    quads = {
        (a, b): [6.0 * e for e in
                 four_state_energies(a, b, 0, 1, [2], {(0, 1): TARGET}, {}, spin)]
        for a in range(3)
        for b in range(3)
    }
    recovered = ex.assemble_tensor(quads, spin=spin, multiplicity=6)
    assert np.allclose(recovered, TARGET * ex.HARTREE_TO_MEV, rtol=1e-10, atol=1e-10)


def test_xiang_extra_sign_manufactures_a_spurious_dm_vector():
    """Sabani et al.'s correction to Xiang et al., reproduced.

    Xiang's DM formulas carry an extra minus sign on one component, inherited
    from a Hamiltonian that had already assumed J^{ab} = -J^{ba}. On a
    SYMMETRIC tensor -- the case an inversion centre at the bond midpoint
    forces -- the correct formulas give D = 0, while the extra sign turns the
    symmetric off-diagonal entries into a fictitious DM vector.
    """
    symmetric = np.array([
        [-4.12e-4, -0.58e-4, 0.74e-4],
        [-0.58e-4, -4.79e-4, -0.40e-4],
        [0.74e-4, -0.40e-4, -4.63e-4],
    ])
    spin = 1.5
    quads = {
        (a, b): four_state_energies(a, b, 0, 1, [2], {(0, 1): symmetric}, {}, spin)
        for a in range(3)
        for b in range(3)
    }
    tensor = ex.assemble_tensor(quads, spin=spin)
    assert np.allclose(tensor, tensor.T, atol=1e-9)
    assert np.allclose(ex.decompose(tensor)["dm"], 0.0, atol=1e-9)

    # the same energies with Xiang's extra minus sign on the "yx"-type
    # component: the antisymmetric part is no longer zero
    flipped = tensor.copy()
    flipped[1, 0] *= -1.0
    flipped[2, 0] *= -1.0
    flipped[2, 1] *= -1.0
    spurious = ex.decompose(flipped)["dm"]
    assert np.linalg.norm(spurious) > 1.0  # meV -- large, not a rounding artefact


def test_decompose_reassembles():
    parts = ex.decompose(TARGET)
    dm = parts["dm"]
    antisym = np.array([
        [0.0, dm[2], -dm[1]],
        [-dm[2], 0.0, dm[0]],
        [dm[1], -dm[0], 0.0],
    ])
    rebuilt = parts["isotropic"] * np.eye(3) + parts["symmetric"] + antisym
    assert np.allclose(rebuilt, TARGET)


def test_dm_vector_matches_the_cross_product_definition():
    """D is defined by S_i . A . S_j = D . (S_i x S_j); check that literally."""
    dm = ex.decompose(TARGET)["dm"]
    antisym = 0.5 * (TARGET - TARGET.T)
    rng = np.random.default_rng(7)
    for _ in range(5):
        si, sj = rng.normal(size=3), rng.normal(size=3)
        assert np.isclose(si @ antisym @ sj, dm @ np.cross(si, sj))


def test_kitaev_parameters_from_an_ideal_bond_tensor():
    j, k, g, gp = -1.8, -10.6, 3.8, -0.9
    ideal = np.array([[j, g, gp], [g, j, gp], [gp, gp, j + k]])
    got = ex.kitaev_parameters(ideal)
    assert np.isclose(got["J"], j)
    assert np.isclose(got["K"], k)
    assert np.isclose(got["Gamma"], g)
    assert np.isclose(got["Gamma_p1"], gp)
    assert np.isclose(got["Gamma_p2"], gp)
    assert got["residual"] < 1e-12


def test_kitaev_residual_flags_a_non_ideal_bond():
    ideal = np.array([[-1.8, 3.8, -0.9], [3.8, -1.8, -0.9], [-0.9, -0.9, -12.4]])
    distorted = ideal.copy()
    distorted[0, 0] += 0.5  # breaks the J^{xx} = J^{yy} the ideal form requires
    assert ex.kitaev_parameters(distorted)["residual"] > 0.2


def test_rotate_tensor_is_a_similarity_transform():
    u = ex.honeycomb_cubic_frame()
    assert np.allclose(u @ u.T, np.eye(3))
    rotated = ex.rotate_tensor(TARGET, u)
    assert np.isclose(np.trace(rotated), np.trace(TARGET))
    assert np.allclose(ex.rotate_tensor(rotated, u.T), TARGET)


def test_rotate_tensor_rejects_a_non_rotation():
    with pytest.raises(ValueError, match="orthonormal"):
        ex.rotate_tensor(TARGET, np.eye(3) * 2.0)


def test_single_ion_formulas_recover_a_known_anisotropy():
    """SIA needs its own configurations and its own formulas -- a different
    sign pattern and a different denominator from the exchange one."""
    spin = 1.5
    a_tensor = np.array([[0.0, 2.0e-5, 0.0], [2.0e-5, 1.0e-4, 0.0], [0.0, 0.0, 3.0e-4]])
    neighbour = np.array([[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 1e-4]])

    # off-diagonal A^{xy}: site 0 at 45 degrees in the xy plane, others along z
    root = spin / np.sqrt(2.0)
    energies = []
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        spins = {0: np.array([sx * root, sy * root, 0.0]), 1: np.array([0.0, 0.0, spin])}
        energies.append(model_energy(spins, {(0, 1): neighbour}, {0: a_tensor}))
    got = ex.single_ion_offdiagonal(*energies, spin=spin)
    # the cross term is A^{xy} S_x S_y + A^{yx} S_y S_x, and the 45-degree
    # placement makes the sign product +,-,-,+ -- so the formula returns
    # A^{xy} itself, not twice it
    assert np.isclose(got, a_tensor[0, 1] * ex.HARTREE_TO_MEV, rtol=1e-9)

    # diagonal difference A^{yy} - A^{xx}
    energies = []
    for vector in ([0, spin, 0], [0, -spin, 0], [spin, 0, 0], [-spin, 0, 0]):
        spins = {0: np.array(vector, dtype=float), 1: np.array([0.0, 0.0, spin])}
        energies.append(model_energy(spins, {(0, 1): neighbour}, {0: a_tensor}))
    got = ex.single_ion_difference(*energies, spin=spin)
    expected = (a_tensor[1, 1] - a_tensor[0, 0]) * ex.HARTREE_TO_MEV
    assert np.isclose(got, expected, rtol=1e-9)


def test_partial_extraction_stays_visibly_partial():
    quads = {(0, 0): (0.0, 1.0e-4, 0.0, 0.0)}
    tensor = ex.assemble_tensor(quads, spin=1.0)
    assert np.isfinite(tensor[0, 0])
    assert np.isnan(tensor[1, 1])


def test_spectator_axis_is_perpendicular_and_deterministic():
    for a in range(3):
        for b in range(3):
            c = ex.spectator_axis(a, b)
            assert c != a and c != b if a != b else c != a
    assert ex.spectator_axis("x", "z") == 1
    assert ex.spectator_axis("x", "x") == 1


# --- supercell bookkeeping -------------------------------------------------


def test_bond_multiplicity_counts_periodic_images():
    avec = np.eye(3) * 4.0
    # a pair 1 apart in a cell of side 4: the image at distance 3 is farther,
    # so only the direct bond counts
    assert ex.bond_multiplicity(avec, [0, 0, 0], [1.0, 0, 0]) == 1
    # exactly half a cell apart: the +x and -x images are equidistant
    assert ex.bond_multiplicity(avec, [0, 0, 0], [2.0, 0, 0]) == 2


def test_check_supercell_rejects_a_contaminated_pair():
    avec = np.eye(3) * 4.0
    with pytest.raises(ValueError, match="supercell too small"):
        ex.check_supercell(avec, [0, 0, 0], [1.0, 0, 0], shell_cutoff=3.5)
    multiplicity, distance = ex.check_supercell(
        avec, [0, 0, 0], [1.0, 0, 0], shell_cutoff=1.5
    )
    assert multiplicity == 1
    assert np.isclose(distance, 1.0)


def test_image_distances_excludes_the_atom_itself():
    avec = np.eye(3) * 4.0
    distances = ex.image_distances(avec, [0, 0, 0], [0, 0, 0])
    assert np.isclose(distances[0], 4.0)
