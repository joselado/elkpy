"""Integration test for four-state exchange mapping (docs/design.md #29) on
NiO -- the classic energy-mapping benchmark: rocksalt antiferromagnet whose
dominant coupling is the 180-degree Ni-O-Ni superexchange J2, with Ni(II) a
clean local S = 1 moment.

Slow: each tensor component costs four constrained non-collinear SCF runs on
a 16-atom all-electron cell, so the full nine-component tensor is 36 runs.
Gated behind ELKPY_RUN_SLOW_TESTS=1 like the phonon suite.

The sharpest check here does not depend on any literature convention: with
spin-orbit coupling OFF the total energy is invariant under a global spin
rotation, so the exchange tensor must be EXACTLY isotropic -- diagonal
entries equal, off-diagonal entries zero. Whatever deviation appears is the
method's numerical noise floor, measured rather than assumed.
"""

import os

import numpy as np
import pytest

from elkpy import config
from elkpy.parsers import exchange as ex
from elkpy.structure import Structure

# The supercell-bookkeeping test below needs neither the binary nor the sweep,
# so the gates go on the tests that actually run Elk rather than the module.
needs_elk = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)
slow = pytest.mark.skipif(
    os.environ.get("ELKPY_RUN_SLOW_TESTS") != "1",
    reason="36 constrained SCF runs; set ELKPY_RUN_SLOW_TESTS=1 to run",
)

BOHR_PER_ANGSTROM = 1.8897261246
A_NIO = 4.17


def nio_supercell():
    ase = pytest.importorskip("ase.build")
    return Structure.from_ase(ase.bulk("NiO", "rocksalt", a=A_NIO, cubic=False) * (2, 2, 2))


def j2_pair(structure):
    """(i, j) for the 180-degree superexchange pair, at the cubic lattice
    constant, found by distance rather than by hard-coded index."""
    from elkpy import exchange as exchange_module

    avec = np.array(structure.avec, dtype=float)
    nickel = exchange_module.magnetic_indices(structure, "Ni")
    positions = {k: exchange_module.atom_cartesian(structure, k) for k in nickel}
    target = A_NIO * BOHR_PER_ANGSTROM
    partner = min(
        nickel[1:],
        key=lambda k: abs(ex.image_distances(avec, positions[nickel[0]], positions[k])[0] - target),
    )
    return nickel[0], partner


def test_supercell_is_large_enough_for_the_j2_pair():
    """Pure bookkeeping, no Elk: in this cell all six J2 neighbours fold onto
    one site (multiplicity 6) and the next image is far outside the retained
    range, so the pair is extractable -- while the J2 partner is NOT also a
    nearest neighbour, which would mix two shells irrecoverably."""
    structure = nio_supercell()
    from elkpy import exchange as exchange_module

    avec = np.array(structure.avec, dtype=float)
    i, j = j2_pair(structure)
    multiplicity, distance = ex.check_supercell(
        avec,
        exchange_module.atom_cartesian(structure, i),
        exchange_module.atom_cartesian(structure, j),
        shell_cutoff=5.0 * BOHR_PER_ANGSTROM,
    )
    assert multiplicity == 6
    assert np.isclose(distance / BOHR_PER_ANGSTROM, A_NIO, atol=1e-3)


@pytest.fixture(scope="module")
def nio_tensor(tmp_path_factory):
    structure = nio_supercell()
    i, j = j2_pair(structure)
    calc = structure.get_calculation(
        tmp_path_factory.mktemp("nio"),
        xc="PW",
        spinpol=True,
        spinorb=False,
        ngridk=(2, 2, 2),
        extra_blocks={
            "dft+u": [(1, 1), (1, 2, 0.2205, 0.0367)],  # FLL, U = 6 eV, J = 1 eV on Ni d
            "nempty": [20],
            "maxscl": [100],
        },
    )
    return calc.get_exchange_tensor(
        i, j, magnetic="Ni", spin=1.0, components="all",
        shell_cutoff=5.0 * BOHR_PER_ANGSTROM, workers=7,
        epsengy=1.0e-7, magnitude=1.8, seed_field=2.0,
    )


@needs_elk
@slow
def test_j2_superexchange_is_antiferromagnetic_and_the_right_size(nio_tensor):
    """NiO's dominant coupling: strongly antiferromagnetic (J > 0 in the
    convention of docs/physics.tex Part XVI), of order 20 meV at U = 6 eV."""
    assert nio_tensor["multiplicity"] == 6
    j_iso = nio_tensor["isotropic"]
    assert j_iso > 0.0
    assert 5.0 < j_iso < 50.0


@needs_elk
@slow
def test_tensor_is_isotropic_without_spin_orbit_coupling(nio_tensor):
    """The convention-free check. With SOC off nothing ties spin to the
    lattice, so the energy is invariant under a global spin rotation and the
    tensor must be exactly J_iso * identity. The residual measures the noise
    floor of four summed total energies."""
    tensor = nio_tensor["tensor"]
    j_iso = nio_tensor["isotropic"]
    residual = np.max(np.abs(tensor - j_iso * np.eye(3)))
    assert residual < 0.05 * abs(j_iso)


@needs_elk
@slow
def test_dm_vector_vanishes_on_an_inversion_symmetric_bond(nio_tensor):
    """Moriya's theorem: the midpoint of the J2 bond is an inversion centre
    (it is the bridging oxygen), so D = 0 identically. This is a null test of
    the antisymmetric extraction -- and the one that Xiang et al.'s
    uncorrected DM formulas famously fail (see tests/test_parsers_exchange.py
    ::test_xiang_extra_sign_manufactures_a_spurious_dm_vector)."""
    assert np.linalg.norm(nio_tensor["dm"]) < 0.05 * abs(nio_tensor["isotropic"])
