"""Phase 1i: smeared occupations and the self-consistent Fermi level, on Elk's matrices.

`docs/continue_here.md` §3 item 4, and the reason it was still open after §1f: for a
HARD integer window the divided-difference kernel's near-degenerate branch is inert,
because both branches of a same-side pair are identically zero.  Fermi-Dirac
occupations make the branch value f' -- large, not zero -- so the threshold is
load-bearing for the first time, and the fixed-electron-number Fermi level (study
§8(b)'s dmu/deps = w_i f'_i / sum_j w_j f'_j, listed as untested) becomes reachable.

The fixture is graphene at K, and it is physically right rather than engineered: the
two pi bands are degenerate there AND the Fermi level sits on them, so f = 1/2 exactly
and f' is at its maximum.  Measured on this build, the single-k chemical potential
reproduces Elk's own zone-integrated `EFERMI.OUT` to 3.4e-9 Ha.

Every assertion here uses `elkjax.reference.fermi_divided_difference_kernel` as its
reference -- a closed form for the logistic difference quotient with no subtraction in
it, hence accurate at any splitting.  That matters because the two things under test
(the direct quotient and the f' branch) are exactly the two candidate answers, so a
reference built from either would beg the question.  Central finite differences are
carried alongside, and are legitimate here in a way they are not in §1h: P = f(H) is a
smooth matrix function, so there is no branch exchange to average over.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import elkjax  # noqa: E402  (sets jax_enable_x64 on import)
from elkjax import hamiltonian as ham, memory, projector as pj, reference as ref  # noqa: E402
from elkpy import config  # noqa: E402
from elkpy.structure import Structure  # noqa: E402

memory.limit_address_space(16.0)

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

A = 4.6511                                   # graphene, a = 2.461 Angstrom
AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3 ** 0.5 / 2, 0.0), (0.0, 0.0, 20.0)]
K_POINT = (1 / 3, 1 / 3, 0.0)
NELEC = 4.0            # 8 valence electrons at occmax = 2, i.e. 4 STATES
WIDTHS = (1e-3, 1e-2, 1e-1)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


@pytest.fixture(scope="module")
def graphene(tmp_path_factory):
    """The LAPW eigenproblem at K, plus Elk's own Fermi level.

    Same cell and cutoff as `test_calculation_lapw_projector.py`'s Dirac fixture, so
    the ground state is the one §1h already characterised: two atoms, rgkmax 6, under
    two minutes.
    """
    workdir = tmp_path_factory.mktemp("lapw_smearing") / "graphene"
    calculation = Structure(
        avec=AVEC, species={"C": [(0.0, 0.0, 0.0), (1 / 3, 2 / 3, 0.0)]}
    ).get_calculation(workdir, xc="PW", ngridk=(6, 6, 1), rgkmax=6.0)
    calculation.ensure_ground_state()
    efermi = float((calculation.workdir / "EFERMI.OUT").read_text().split()[0])
    with calculation.eigenstate_session() as session:
        export = session.lapw_problem(K_POINT)
    h, o = ham.eigenproblem_at(export, np.asarray(export["vkc"]))
    reduced, _ = ham.cholesky_reduce(h, o)
    return dict(export=export, efermi=efermi, reduced=np.asarray(reduced),
                tol=ham.projector_tolerance(reduced, o),
                evals=np.linalg.eigvalsh(np.asarray(reduced)))


def _directions(n, count, seed=2100):
    return [ref.random_hermitian_direction(n, seed + j) for j in range(count)]


def _routes(reduced, direction, mu, width, tol, observable):
    """The three JAX routes and the exact kernel, on one Hermitian direction."""
    hj, dj, mj = (jnp.asarray(reduced), jnp.asarray(direction),
                  jnp.asarray(observable))
    loss = lambda p: jnp.real(jnp.trace(p @ mj))
    naive = lambda t: loss(pj.naive_smeared_projector(hj + t * dj, mu, width))
    quotient = lambda t: loss(pj.direct_quotient_projector(hj + t * dj, mu, width))
    safe = lambda t: loss(pj.smeared_projector(hj + t * dj, mu, width, tol))
    return dict(
        exact=float(np.real(np.trace(
            ref.dprojector_fermi(reduced, direction, mu, width) @ observable))),
        naive_fwd=float(jax.jvp(naive, (0.0,), (1.0,))[1]),
        naive_rev=float(jax.grad(naive)(0.0)),
        quotient_rev=float(jax.grad(quotient)(0.0)),
        safe_fwd=float(jax.jvp(safe, (0.0,), (1.0,))[1]),
        safe_rev=float(jax.grad(safe)(0.0)))


def test_the_single_k_fermi_level_reproduces_elks_own(graphene):
    """The physical anchor, and the reason this fixture is not engineered.

    Requiring sum f = 4 at K alone gives the same chemical potential as Elk's
    zone-integrated `EFERMI.OUT`, because particle-hole symmetry at the Dirac point
    puts the local answer on the global one.  A cell where that failed would still
    exercise the kernel, but the smearing would then be a knob rather than physics.
    """
    evals, efermi = graphene["evals"], graphene["efermi"]
    occupancy = ref.fermi_dirac(evals, efermi, 1e-3)[0].sum()
    assert abs(occupancy - NELEC) < 1e-4, occupancy
    mu = float(pj.fermi_level(jnp.asarray(graphene["reduced"]), NELEC, 1e-3))
    assert abs(mu - efermi) < 1e-7, (mu, efermi)


def test_a_real_dirac_point_is_split_far_above_the_resolution(graphene):
    """The finding that decides which failure this fixture can show, asserted.

    Symmetry makes the two pi bands degenerate at K, but Elk's assembly splits them by
    ~3e-7 Ha -- five decades ABOVE this run's eps kappa(O) |H~| resolution.  So the
    kernel's near-degenerate branch never fires here and the direct quotient is taken,
    which is why the route that fails below is JAX's eigenvector rule and not the
    numerator's cancellation.  §1f measured the same statistic on Si's Gamma_25'
    triplet and found 5.1e-15 for one pair and 3.53e-9 for the other: a real LAPW
    multiplet's splitting is a property of the assembly's roundoff, not of the
    symmetry, and spans at least eight decades.
    """
    split = graphene["evals"][4] - graphene["evals"][3]
    assert split < 1e-5, "the Dirac point is no longer degenerate"
    assert split > 1e3 * graphene["tol"], (split, graphene["tol"])
    assert graphene["evals"][3] - graphene["evals"][2] > 0.1     # cleanly isolated


def test_the_safe_rule_matches_the_exact_kernel_on_a_real_matrix(graphene):
    """The Phase 1i forward-and-gradient criterion, at Elk's own swidth default.

    Forward and reverse mode are compared against each other as well as against the
    reference: for a scalar-in scalar-out function they are the same number, so a
    disagreement is proof on its own (CLAUDE.md, "JAX port").
    """
    reduced, tol, mu = graphene["reduced"], graphene["tol"], graphene["efermi"]
    observable = np.diag(np.linspace(-1.0, 1.0, reduced.shape[0])).astype(complex)
    for direction in _directions(reduced.shape[0], 3):
        r = _routes(reduced, direction, mu, 1e-3, tol, observable)
        assert abs(r["exact"]) > 1e-3, "this direction carries no signal"
        assert _rel(r["safe_rev"], r["exact"]) < 1e-11, r
        assert _rel(r["safe_fwd"], r["safe_rev"]) < 1e-11, r


def test_the_eigenvector_rule_degrades_as_the_smearing_widens(graphene):
    """JAX's own `eigh` rule is the route that fails here, and it fails PROPORTIONALLY.

    Its error is ~eps ||A - A^dag|| / dlambda absolute against a derivative of order
    f' = 1/4w, so the RELATIVE error grows linearly in the smearing width -- the
    opposite of the intuition that broader smearing is gentler.  Measured on this
    build: 3.9e-9, 3.9e-8, 2.8e-7 in forward mode at w = 1e-3, 1e-2, 1e-1, against
    1.0e-13, 7.7e-14, 2.0e-10 for the safe rule.  Asserted as a growth rate and a
    separation, not as thresholds, since both depend on the eigensolver's own splitting
    of the pair.
    """
    reduced, tol, mu = graphene["reduced"], graphene["tol"], graphene["efermi"]
    observable = np.diag(np.linspace(-1.0, 1.0, reduced.shape[0])).astype(complex)
    directions = _directions(reduced.shape[0], 3)
    naive, safe = [], []
    for width in WIDTHS:
        rows = [_routes(reduced, d, mu, width, tol, observable) for d in directions]
        naive.append(max(_rel(r["naive_fwd"], r["exact"]) for r in rows))
        safe.append(max(_rel(r["safe_rev"], r["exact"]) for r in rows))
    assert naive[2] > 20 * naive[0], naive          # ~100x over two decades of width
    assert all(n > 1e3 * s for n, s in zip(naive, safe)), (naive, safe)


def test_the_direct_quotient_is_enough_at_this_splitting(graphene):
    """Switching the branch off changes nothing HERE, and that is the point.

    3e-7 Ha is far too wide for the numerator's cancellation to bite: the loss is
    ~eps w / dlambda, i.e. 1e-12 at Elk's default width.  So the tolerance's value is
    not what protects this fixture -- avoiding the eigenvector derivative is -- and a
    test that conflated the two would credit the wrong mechanism.  The regime where the
    branch itself is load-bearing is pinned separately on synthetic spectra with a
    1e-15 splitting (`test_jax_projector.py`).
    """
    reduced, tol, mu = graphene["reduced"], graphene["tol"], graphene["efermi"]
    observable = np.diag(np.linspace(-1.0, 1.0, reduced.shape[0])).astype(complex)
    for direction in _directions(reduced.shape[0], 2):
        r = _routes(reduced, direction, mu, 1e-3, tol, observable)
        assert _rel(r["quotient_rev"], r["exact"]) < 1e-11, r
        assert _rel(r["quotient_rev"], r["safe_rev"]) < 1e-11, r


def test_the_chemical_potential_term_dominates_at_a_half_filled_level(graphene):
    """Fixed N against fixed mu: the term study §8(b) supplies is not a correction.

    At a level pinned to the Fermi energy, holding mu fixed while H moves is not a
    small error -- measured here, the fixed-mu derivative is wrong by ~70x.  The
    reference re-solves mu at each displaced matrix, so it carries the constraint
    rather than the formula being tested.

    Tested along Hermitian directions and deliberately NOT along k: at K the pi pair's
    trace is stationary by symmetry, so dmu/dk vanishes and a k-direction test would
    pass with the correction identically zero -- the same "perturbation respects the
    protecting symmetry" trap Phase 0a records.
    """
    reduced, tol = graphene["reduced"], graphene["tol"]
    n = reduced.shape[0]
    observable = np.diag(np.linspace(-1.0, 1.0, n)).astype(complex)
    hj, mj = jnp.asarray(reduced), jnp.asarray(observable)
    width = 1e-3
    mu, response = pj.check_fermi_level_determined(hj, NELEC, width)
    assert response > 1e2, response          # a half-filled level responds strongly
    dominated = False
    for direction in _directions(n, 3, seed=2300):
        dj = jnp.asarray(direction)
        loss = lambda t: jnp.real(jnp.trace(
            pj.fixed_number_projector(hj + t * dj, NELEC, width, tol) @ mj))
        exact = float(np.real(np.trace(ref.dprojector_fermi(
            reduced, direction, mu, width, dmu="selfconsistent") @ observable)))
        fixed_mu = float(np.real(np.trace(
            ref.dprojector_fermi(reduced, direction, mu, width) @ observable)))
        assert _rel(float(jax.grad(loss)(0.0)), exact) < 1e-10
        assert _rel(float(jax.jvp(loss, (0.0,), (1.0,))[1]), exact) < 1e-10
        if _rel(fixed_mu, exact) > 1.0:
            dominated = True
    assert dominated, "the mu term is negligible here, so this fixture proves nothing"


def test_the_k_derivative_carries_the_smeared_occupations(graphene):
    """The whole differentiable pipeline, with occupations that actually respond.

    §1f's k-derivative went through a hard window, where a same-side multiplet
    contributes exactly nothing; here the states at the Fermi level carry the largest
    kernel entries in the matrix, so the k-tangent passes through them.  The reference
    is the exact kernel fed with dH~/dk from a matrix-valued jvp -- legitimate because
    the projector is the only non-smooth step -- with central FD as the control that
    separates an AD bug from a broken test.  The FD step must satisfy v_F h << w.
    """
    export, tol, mu = graphene["export"], graphene["tol"], graphene["efermi"]
    kc = np.asarray(export["vkc"])
    width, step = 1e-3, 1e-5

    def reduced_at(kvec):
        return ham.cholesky_reduce(*ham.eigenproblem_at(export, kvec))[0]

    reduced0 = np.asarray(reduced_at(jnp.asarray(kc)))
    observable = np.diag(np.linspace(-1.0, 1.0, reduced0.shape[0])).astype(complex)
    mj = jnp.asarray(observable)
    dk = np.array([0.3, -0.5, 0.0])
    dk /= np.linalg.norm(dk)
    dh = np.asarray(jax.jvp(reduced_at, (jnp.asarray(kc),), (jnp.asarray(dk),))[1])
    dh = 0.5 * (dh + dh.conj().T)
    exact = float(np.real(np.trace(
        ref.dprojector_fermi(reduced0, dh, mu, width) @ observable)))

    safe = lambda t: jnp.real(jnp.trace(pj.smeared_projector(
        reduced_at(jnp.asarray(kc) + t * jnp.asarray(dk)), mu, width, tol) @ mj))

    def np_loss(matrix):
        evals, evecs = np.linalg.eigh(np.asarray(matrix))
        f, _ = ref.fermi_dirac(evals, mu, width)
        return float(np.real(np.trace(((evecs * f) @ evecs.conj().T) @ observable)))

    fd = (np_loss(reduced_at(jnp.asarray(kc + step * dk)))
          - np_loss(reduced_at(jnp.asarray(kc - step * dk)))) / (2 * step)
    assert abs(exact) > 1e-2, "this k-direction carries no signal"
    assert _rel(float(jax.grad(safe)(0.0)), exact) < 1e-10
    assert _rel(fd, exact) < 1e-5, (fd, exact)
