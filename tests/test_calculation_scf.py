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
def silicon_calculation(tmp_path_factory):
    """Bulk Si on Elk's own defaults, reduced mesh included.

    Nothing here switches `symtype` off: the loop runs `symrf` on both regions
    (patches 0018 and 0022), so a symmetry-reduced k-set is the case it has to
    handle rather than one to avoid.

    Nothing has been RUN on it -- `silicon` converges it, `silicon_initial`
    deliberately does not.
    """
    return Structure(
        avec=SI_AVEC,
        species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    ).get_calculation(tmp_path_factory.mktemp("scf") / "si", xc="PW",
                      ngridk=(2, 2, 2), rgkmax=7.0)


@pytest.fixture(scope="module")
def silicon(silicon_calculation):
    """The three export dicts at Elk's own CONVERGED ground state."""
    from elkjax import driver
    return driver.converged_state(silicon_calculation)


@pytest.fixture(scope="module")
def silicon_initial(silicon_calculation):
    """The same three, at the TOP of Elk's first iteration (task 9006).

    The density is the superposition of free atomic densities `rhoinit`
    builds; no `STATE.OUT` is read and no SCF has run.  This is what
    `elkjax.driver` starts from.
    """
    from elkjax import driver
    return driver.initial_state(silicon_calculation)


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

    **The two halves are asserted separately, and they are not the same
    number.**  The packed norm is dominated by the muffin tin, which carries
    the nuclear -Z/r at the first radial point and is of order 4.8e8 against
    the interstitial's 1.1e2 -- so a single relative bound on the packed
    vector would admit an interstitial error of 1e-6 absolute without
    noticing.  Measured: 1.8e-15 relative in the muffin tin and 1.0e-9 in the
    interstitial.  The second is not roundoff and is not claimed to be: the
    start mixes Elk's own MIXED `vsmt` with its UNMIXED `vsir` (see the test
    above), so a residual at the scale of Elk's last mixing step is what this
    start point can give.
    """
    import jax.numpy as jnp
    from elkjax import scf
    groundstate, densityk, lapw = silicon
    shape = tuple(int(n) for n in np.asarray(lapw["vsmt"]).shape)
    start = scf.pack(np.asarray(lapw["vsmt"]), groundstate["vsir"])
    residual = scf.step(start, (lapw, groundstate, densityk)) - start

    for got, reference, bound in zip(
            scf.unpack(residual, shape, int(groundstate["ngtot"])),
            scf.unpack(start, shape, int(groundstate["ngtot"])),
            (1e-14, 1e-8)):
        assert float(jnp.linalg.norm(got)) \
            / float(jnp.linalg.norm(reference)) < bound


def test_the_energy_no_longer_imports_the_eigenvalue_sum_or_the_entropy(
        silicon):
    """§2f had to import `evalsum` and `engyts`; this computes both.

    They needed a zone sum and the occupations, which is what Phase 3 supplies.
    What is still imported is the CORE half of `evalsum` (an input at fixed
    potential, exactly as `rhocr` is) and `engynn`, a property of the lattice
    rather than of the density.

    The big terms are checked in ABSOLUTE Hartrees, because that is the unit
    the study's own criterion is in and because a relative bound on `engytot`
    (-578 Ha) would admit 6e-4 Ha.  Measured 1.3e-8 Ha on `engytot`; 1e-7 is
    a regression guard with room for a different BLAS, not the measurement.

    `engyts` is the one checked relatively, because it is tiny (-9.5e-6 Ha
    here) and an absolute bound on it would pass for a term that was simply
    zero -- which is what `energy.f90` returns for every `stype` but
    Fermi-Dirac.  Its 1e-5 is loose on purpose: it inherits mu, which inherits
    the `vsig` mixing step, and this session measured twice that anything set
    by Elk's stopping point moves between builds.  The floor below keeps it
    from being vacuous.
    """
    from elkjax import scf
    groundstate, densityk, lapw = silicon
    terms, got = scf.total_energy(lapw, groundstate, densityk,
                                  np.asarray(lapw["vsmt"]),
                                  groundstate["vsir"])
    for name in ("evalsum", "engykn", "engytot"):
        assert abs(float(terms[name]) - float(groundstate[name])) < 1e-7, name
    assert abs(float(terms["engyts"]) - float(groundstate["engyts"])) \
        / abs(float(groundstate["engyts"])) < 1e-5
    assert abs(float(terms["engyts"])) > 1e-9, (
        "engyts is zero on this fixture, so the relative check above is "
        "vacuous; Elk's default stype is 3 and this should not happen")
    assert abs(float(got["mu"]) - float(densityk["efermi"])) < 1e-8


def test_the_step_is_differentiable_and_the_derivative_is_the_right_one(
        silicon):
    """`jvp(F)` against a central difference of F, along a random direction.

    The direction is random and therefore breaks every crystal symmetry, which
    matters: a symmetry-respecting perturbation makes the off-diagonal matrix
    elements inside silicon's `Gamma_25'` multiplet vanish and hides exactly the
    term that `elkjax.response` exists to get right.

    **The two halves are compared separately and only one of them is a test of
    the derivative.**  The muffin-tin half of the potential carries the nuclear
    -Z/r at the first radial point and is 4.8e8 in norm, so a central difference
    of it cannot resolve better than 4.8e8 * eps / (2h) -- at h = 1e-3 that is
    2.4e-5 against a tangent of norm 3.8e-2, i.e. 6e-4, and the measured 5.1e-4
    is that floor and nothing else (it grows as 1/h, measured 5.9e-3 at 1e-4 and
    9.2e-2 at 1e-5, which is the difference degrading and not the derivative).
    The interstitial half is 1e2 in norm and resolves cleanly: measured 6.6e-9.

    With JAX's own `eigh` rule in place of `elkjax.response` the same
    interstitial number is **1.9e-3, and does not move with h** -- flat across
    1e-3, 1e-4 and 1e-5, which is the signature of a wrong derivative rather
    than a noisy difference.
    """
    import jax
    import jax.numpy as jnp
    from elkjax import scf
    groundstate, densityk, lapw = silicon
    shape = tuple(int(n) for n in np.asarray(lapw["vsmt"]).shape)
    ngtot = int(groundstate["ngtot"])

    step = jax.jit(lambda v: scf.step(v, (lapw, groundstate, densityk)))
    start = jnp.asarray(scf.pack(np.asarray(lapw["vsmt"]),
                                 groundstate["vsir"]))
    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.normal(size=start.shape))
    direction = direction / jnp.linalg.norm(direction)

    _, tangent = jax.jvp(step, (start,), (direction,))
    h = 1e-3
    difference = (step(start + h * direction)
                  - step(start - h * direction)) / (2 * h)
    for name, bound, mine, theirs in zip(
            ("muffin tin", "interstitial"), (2e-3, 1e-7),
            scf.unpack(tangent, shape, ngtot),
            scf.unpack(difference, shape, ngtot)):
        error = float(jnp.linalg.norm(mine - theirs)
                      / jnp.linalg.norm(theirs))
        assert error < bound, f"{name}: {error:.3e}"


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


# ------------------------------------- §3c: a run from the input file alone


def test_the_initial_state_needs_no_converged_ground_state(silicon_initial):
    """Task 9006 comes up on an `elk.in` and nothing else.

    The fixture itself is most of the test -- `initial_state_session()` does
    not call `ensure_ground_state()`, so if task 9006 needed a `STATE.OUT`
    this would not have got here.  What is checked beyond that is that the
    state really is Elk's INITIALISATION and not its answer.

    The electron count is the sharp version of that.  `rhoinit` superposes
    free ATOMIC densities and does not normalise them -- `rhonorm` acts on the
    density the SCF produces, not on the starting guess -- so the count comes
    out 6.4e-3 SHORT of 28 on Si.  A converged density is normalised to
    machine precision, so the assertion is two-sided on purpose: close enough
    to be a real density, and far enough to prove this is not one Elk has
    iterated.
    """
    from elkjax import driver, integrate
    groundstate, densityk, lapw = silicon_initial

    driver.check_linearisation_frozen(lapw)
    assert not np.asarray(lapw["apwve"]).any()
    assert not np.asarray(lapw["lorbve"]).any()

    count = float(integrate.cell_integral(
        groundstate["rhomt"], groundstate["rhoir"], groundstate))
    missing = abs(count - float(densityk["chgtot"]))
    assert missing < 1e-2, "this is not a plausible starting density"
    assert missing > 1e-6, (
        "the starting density is normalised to the electron count, so this is "
        "a converged density rather than rhoinit's superposition")


def test_the_core_integral_reproduces_elks_own_energykncr(silicon_initial,
                                                          silicon):
    """`int rho_core v_s`, at BOTH exports, against `evalsumcr - engykncr`.

    This is the whole transcription behind `core_eigenvalue_sum`, and it is
    checked at two different potentials on purpose: the quantity it exists to
    correct is the one that MOVES between them.
    """
    from elkjax import energy
    for groundstate, densityk, lapw in (silicon_initial, silicon):
        mine = float(energy.core_potential_energy(
            lapw["vsmt"], densityk, groundstate))
        theirs = float(densityk["evalsumcr"]) - float(densityk["engykncr"])
        assert abs(mine - theirs) / abs(theirs) < 1e-14


def test_the_frozen_core_quantity_is_the_kinetic_energy_not_the_eigenvalues(
        silicon_initial, silicon):
    """Which core scalar may be held fixed across the loop, measured.

    `energy.f90` builds the kinetic energy as
    `evalsum - engyvcl - engyvxc`, so the core's share of it is
    `evalsumcr - int rho_core v_s` -- BOTH halves at the current potential.
    Freezing `evalsumcr` alone integrates the same core density against a
    potential its eigenvalues never saw.  Between the atomic superposition and
    the converged answer the two scalars move by three orders of magnitude
    apart, which is why one of them is a wrong formula and the other is the
    approximation this port actually makes.
    """
    _, initial, _ = silicon_initial
    _, converged, _ = silicon
    moved_eigenvalues = abs(float(converged["evalsumcr"])
                            - float(initial["evalsumcr"]))
    moved_kinetic = abs(float(converged["engykncr"])
                        - float(initial["engykncr"]))
    assert moved_eigenvalues > 1.0
    assert moved_kinetic < 1e-2
    assert moved_eigenvalues / moved_kinetic > 100


@pytest.mark.skipif(os.environ.get("ELKPY_RUN_SLOW_TESTS") != "1",
                    reason="the run-from-input loop is ~3 min; set "
                           "ELKPY_RUN_SLOW_TESTS=1")
def test_a_run_from_the_input_file_reaches_elks_ground_state(
        silicon_initial, silicon):
    """§3c: 40 iterations from `rhoinit`'s atomic superposition.

    The start is not a perturbation of the answer -- its Fermi level is
    0.1249 Ha against the converged 0.2140 -- and linear mixing at 0.4 reaches
    a residual of 9.3e-8 in 40 iterations.  Measured: `engytot` within
    3.6e-4 Ha of Elk's own and the Fermi level within 4.3e-5 Ha.

    **Both of those bounds are the FROZEN CORE and nothing else**, which is
    the second half of this test: putting the converged `rhocr`/`engykncr`
    into the same initial triple, and changing nothing else, takes the same
    run to 3.8e-8 Ha and 4.5e-9 Ha -- §3b's own agreement, from a cold start.
    So a regression here that moves only the first pair is core physics, and
    one that moves the second pair is the map.
    """
    from elkjax import driver
    groundstate, densityk, lapw = silicon_initial
    reference, converged, _ = silicon
    energy, fermi = float(reference["engytot"]), float(converged["efermi"])

    frozen = driver.run(state=silicon_initial, mixing=0.4, tol=1e-7,
                        maxiter=60)
    assert frozen.iterations < 60 and float(frozen.residual) < 1e-7
    assert abs(frozen.energy - energy) < 1e-3
    assert abs(frozen.mu - fermi) < 1e-4

    swapped = dict(densityk)
    for key in ("rhocr", "evalsumcr", "engykncr"):
        swapped[key] = converged[key]
    exact = driver.run(state=(groundstate, swapped, lapw), mixing=0.4,
                       tol=1e-7, maxiter=60)
    assert abs(exact.energy - energy) < 1e-6
    assert abs(exact.mu - fermi) < 1e-7
