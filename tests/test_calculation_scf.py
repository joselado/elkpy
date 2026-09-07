r"""Phase 3: the Kohn-Sham loop closed and iterated.

Phase 2 left two half-steps that each reproduce Elk pointwise and had never
been composed into a cycle -- density to potential (2i) and potential to
eigenvectors to density (2j).  `elkjax.occupations` supplied what joined them,
and `elkjax.scf` closes it:

    v -> radial functions -> H, O at every k -> eigenvalues, eigenvectors
      -> mu, occupations -> rho -> v_cl[rho] + S v_xc[rho] = F(v).

The tests are in the order the pieces enter.  Two forward pins first -- Elk's
`vsig` and Elk's `vsmt`/`vsir`, each against an array already exported -- then
the statement that Elk's own converged potential is a FIXED POINT of the map,
then the energy from this port's own occupations.  The convergence test itself
takes about eight minutes and is gated behind ELKPY_RUN_SLOW_TESTS.

Skipped without the elk binary, and without jax.
"""

import importlib.util
import os

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
    """Bulk Si on Elk's own defaults, reduced mesh included.

    Nothing here switches `symtype` off: the loop runs `symrf` on both regions
    (patches 0018 and 0022), so a symmetry-reduced k-set is the case it has to
    handle rather than one to avoid.
    """
    calculation = Structure(
        avec=SI_AVEC,
        species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    ).get_calculation(tmp_path_factory.mktemp("scf") / "si", xc="PW",
                      ngridk=(2, 2, 2), rgkmax=7.0)
    calculation.ensure_ground_state()
    with calculation.eigenstate_session() as session:
        return (session.ground_state(), session.density_k(),
                session.lapw_problem((0.0, 0.0, 0.0)))


def test_genvsig_transforms_the_potential_times_the_characteristic_function(
        silicon):
    """`vsig` is the transform of v_s * Theta, not of v_s.

    `rfirftoc.f90`'s first line multiplies by `cfunir`; `rfirctof`, the map the
    density uses in the other direction, does not.  So `density.coarsen` --
    which IS `rfirctof`'s exact inverse -- is not `rfirftoc`, and using it here
    is wrong by a factor of order ten.  That mutation is asserted, because the
    symptom downstream is not a slightly wrong potential: it drives the whole
    spectrum below `e0min`, every occupancy is gated to zero, and the density
    comes back empty.

    The tolerance is 1e-6 and not 1e-15 for a reason in Elk's own bookkeeping,
    measured rather than tolerated -- see the test below.
    """
    from elkjax import density, scf
    groundstate, densityk, _ = silicon
    reference = np.asarray(groundstate["vsig"])

    mine = np.asarray(scf.interstitial_potential_transform(
        groundstate["vsir"], groundstate))
    assert np.abs(mine - reference).max() / np.abs(reference).max() < 1e-6

    coarse_grid = tuple(int(n) for n in densityk["ngdgc"])
    igfc = np.asarray(densityk["igfc"])[:int(densityk["ngvc"])] - 1
    mutant = np.asarray(density._forward_fft(
        density.coarsen(groundstate["vsir"], groundstate, densityk),
        coarse_grid))[igfc]
    assert np.abs(mutant - reference).max() / np.abs(reference).max() > 1.0


def test_the_vsig_residual_is_elks_own_mixing_step(silicon):
    """Why the tolerance above is 1e-6, and what would make it 1e-15.

    `init0.f90` makes the mixer's target array `vsbs` = [`vsmt`, `vsirc`] --
    the COARSE interstitial potential -- while `vsir` is a separate array that
    `potks` fills from `vclir + vxcir` and nothing mixes.  `genvsig` reads
    `vsirc`.  So the export carries a `vsir` one un-mixed step ahead of the
    `vsig` built from it, and the gap IS the last mixing step.

    Asserted as a size rather than a mechanism, since the mechanism is a claim
    about Elk's source: the residual must be far above roundoff and far below
    anything that would matter.  Measured 2.6e-8 at Elk's default `epspot`
    (1e-6) and 2.0e-9 at `epspot=1e-8`, i.e. it tracks where Elk's own SCF
    stopped.
    """
    from elkjax import scf
    groundstate, _, _ = silicon
    reference = np.asarray(groundstate["vsig"])
    mine = np.asarray(scf.interstitial_potential_transform(
        groundstate["vsir"], groundstate))
    error = np.abs(mine - reference).max() / np.abs(reference).max()
    assert 1e-12 < error < 1e-6, (
        "the vsig residual is no longer Elk's mixing step; if it has become "
        "roundoff, Elk has changed what genvsig reads and this test should "
        "become an equality")


def test_the_composed_potential_lands_in_lapws_own_packing(silicon):
    """`potks` against `lapw["vsmt"]` and `groundstate["vsir"]`.

    §2i already compared v_cl + S v_xc against `vclmt + vxcmt`.  This compares
    it against the array `radial.potential_arrays` actually unpacks, which is
    the one the loop feeds back: a right potential in the wrong packing is a
    failure this catches and that one does not.
    """
    from elkjax import scf
    groundstate, _, lapw = silicon
    vsmt, vsir = scf.potential_from_density(
        groundstate["rhomt"], groundstate["rhoir"], groundstate)

    reference = np.asarray(lapw["vsmt"])
    assert np.asarray(vsmt).shape == reference.shape
    assert np.abs(np.asarray(vsmt) - reference).max() \
        / np.abs(reference).max() < 1e-14
    reference = np.asarray(groundstate["vsir"])
    assert np.abs(np.asarray(vsir) - reference).max() \
        / np.abs(reference).max() < 1e-14


def test_elks_converged_potential_is_a_fixed_point_of_the_map(silicon):
    """One whole iteration of F, starting at Elk's own answer.

    This is the statement that the cycle closes: radial functions, `apwalm`,
    both interstitial blocks, the zone eigensolve, `occupy`, all of `rhomag`
    including both symmetrisations, the Weinert solve and the functional --
    composed once and landing back where they started.

    Four things are held fixed that `gndstate` recomputes every iteration
    (`rhocr`, `evalsumcr`, `chgtot` and the linearisation energies), so this
    is not Elk's own map; but every one of them is evaluated at Elk's
    converged potential, which is what makes Elk's `v*` a fixed point of THIS
    map and the check meaningful.
    """
    import jax.numpy as jnp
    from elkjax import scf
    groundstate, densityk, lapw = silicon
    start = scf.pack(np.asarray(lapw["vsmt"]), groundstate["vsir"])
    residual = scf.step(start, (lapw, groundstate, densityk)) - start
    assert float(jnp.linalg.norm(residual)) \
        / float(jnp.linalg.norm(start)) < 1e-14


def test_the_energy_no_longer_imports_the_eigenvalue_sum_or_the_entropy(
        silicon):
    """§2f had to import `evalsum` and `engyts`; this computes both.

    They needed a zone sum and the occupations, which is what Phase 3 supplies.
    What is still imported is the CORE half of `evalsum` (an input at fixed
    potential, exactly as `rhocr` is) and `engynn`, a property of the lattice
    rather than of the density.

    `engyts` is checked in RELATIVE terms even though it is tiny (-9.5e-6 Ha
    on this fixture): an absolute tolerance on it would pass for a term that
    was simply zero, which is what `energy.f90` returns for every `stype` but
    Fermi-Dirac.
    """
    from elkjax import scf
    groundstate, densityk, lapw = silicon
    terms, got = scf.total_energy(lapw, groundstate, densityk,
                                  np.asarray(lapw["vsmt"]),
                                  groundstate["vsir"])
    for name in ("evalsum", "engyts", "engykn", "engytot"):
        mine, reference = float(terms[name]), float(groundstate[name])
        assert abs(mine - reference) / abs(reference) < 1e-6, name
    assert abs(float(terms["engyts"])) > 1e-9, (
        "engyts is zero on this fixture, so the relative check above is "
        "vacuous; Elk's default stype is 3 and this should not happen")
    assert abs(float(got["mu"]) - float(densityk["efermi"])) < 1e-8


@pytest.mark.skipif(os.environ.get("ELKPY_RUN_SLOW_TESTS") != "1",
                    reason="the SCF loop is ~8 min; set ELKPY_RUN_SLOW_TESTS=1")
def test_the_loop_converges_to_elks_ground_state(silicon):
    """The study's Phase 3 forward criterion, from a perturbed start.

    Starting at Elk's converged potential tests nothing -- the test above
    already measures that point as a fixed point to 1e-15.  The start here is
    §1k's smooth valence-region bump at 5% of the muffin-tin potential, which
    puts the initial residual at 0.80 and the initial distance from Elk's own
    potential at 0.30.

    Linear mixing at beta = 0.4 converges geometrically at about 0.62 per
    iteration after the first few, and `|v - v*|` tracks `|F(v) - v|` all the
    way down -- so it converges to ELK'S fixed point rather than to one of its
    own.  Measured after 30 iterations: residual 1.8e-6, `engytot` within
    3.0e-8 Ha of Elk's exported value and the Fermi level within 1.4e-9 Ha.
    Both are inside the study's own 1e-6 Ha and 1e-8 Ha, and both are set by
    where the iteration was stopped rather than by the map.

    The iteration count is deliberately NOT compared with Elk's: the study
    withdraws that criterion itself, since a different eigensolver and a
    different mixer make the step at which the residual crosses a threshold a
    coin flip.
    """
    import jax.numpy as jnp
    from elkjax import phase1_potential, scf
    groundstate, densityk, lapw = silicon

    bump = np.asarray(phase1_potential.split_directions(lapw)["valence"])
    start = scf.pack(np.asarray(lapw["vsmt"]) + 0.05 * bump,
                     groundstate["vsir"])
    reference = scf.pack(np.asarray(lapw["vsmt"]), groundstate["vsir"])
    initial = float(jnp.linalg.norm(
        scf.step(start, (lapw, groundstate, densityk)) - start))
    assert initial > 1e-2, "the start is too close to converged to be a test"

    vsmt, vsir, iterations, residual = scf.run(
        lapw, groundstate, densityk,
        vsmt=np.asarray(lapw["vsmt"]) + 0.05 * bump, vsir=groundstate["vsir"],
        mixing=0.4, tol=1e-5, maxiter=40)
    assert residual < 1e-5 and iterations < 40
    assert float(jnp.linalg.norm(scf.pack(vsmt, vsir) - reference)) < 1e-4

    terms, got = scf.total_energy(lapw, groundstate, densityk, vsmt, vsir)
    assert abs(float(terms["engytot"])
               - float(groundstate["engytot"])) < 1e-6
    assert abs(float(got["mu"]) - float(densityk["efermi"])) < 1e-8
