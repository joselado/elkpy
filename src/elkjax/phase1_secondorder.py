r"""Phase 1j: second derivatives of the occupied projector, on Elk's own matrices.

``docs/continue_here.md`` §3 item 5, and the last Phase 1 item reachable without Phase 2.
The first-order safe-:math:`K` rule of §1f is **first order only** by construction: its
own JVP body calls ``jnp.linalg.eigh``, so a second derivative falls back on JAX's
default eigenvector rule and the hazard the rule exists to remove comes straight back.
:func:`elkjax.projector.sign_projector` is the eigensolver-free route — the occupied
projector as a matrix sign function,

.. math::

    P=\tfrac12\big(\mathbb 1-\mathrm{sign}(\tilde H-\mu)\big),
    \qquad X\leftarrow\tfrac12(3X-X^3),\ \ X_0=(\tilde H-\mu)/\lVert\tilde H-\mu\rVert_2,

which contains no eigendecomposition, hence no gauge and no
:math:`1/(\lambda_i-\lambda_j)` anywhere, and which JAX therefore differentiates to any
order natively.  It has only ever run on Phase 0a′'s toy.

**Why a real matrix is not just a bigger toy.**  Newton-Schulz needs about
:math:`\log(\lVert\tilde H-\mu\rVert_2/\Delta)/\log(3/2)` iterations for a boundary gap
:math:`\Delta`, and the numerator is set by **the extreme of the reduced spectrum, not by
the valence bandwidth**: measured on bulk Si at :math:`\Gamma` with ``rgkmax=7``, the
reduced matrix runs to 17.9 Ha against a 0.35 Ha valence manifold, so the ratio is 190
rather than about 4 and the count is 13 rather than 3.  (Note this is the
*first-variational* matrix, which has no core states in it at all -- the study's own
"deepest state" phrasing anticipates a second-variational or all-electron block, where
the numerator would be far larger still.)  So this is a cost measurement as much as a
correctness one, and the cost is the study's own concern about unrolled tapes: the
iteration is unrolled, and Phase 0e measured compile time as superlinear in HLO op
count.

Four experiments::

    PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase1_secondorder [workdir]

* **A, the step count.**  :math:`\lVert\tilde H-\mu\rVert_2/\Delta` on real matrices,
  the iteration count it predicts, and where ``sign_projector`` actually converges onto
  ``hard_window_projector`` and onto Elk's own occupied subspace.
* **B, first derivative.**  ``sign`` against the already-validated safe-:math:`K` rule
  and against the closed form.  A third, eigensolver-free code path agreeing is
  independent confirmation; it also has to pass before C means anything.
* **C, second derivative.**  ``grad(grad)`` of the sign route against a central
  difference of the *safe rule's own first derivative* — an independent reference,
  since that first derivative is validated separately and shares no code with this
  one — with a second difference of the loss as the coarse control, and the
  ``eigh``-based route alongside to show what it does at the same multiplet.
  ``jax.hessian`` is not used anywhere: a ``custom_vjp`` cannot be forward-differentiated
  (Phase 0a′), so the composition is ``grad(grad)``.
* **D, in** :math:`k`.  The same through the whole assembly, which is what a
  second-order response would need.

Results: ``docs/jax_port_phase1.md`` §1j.
"""

import sys
import time

import numpy as np

import jax
import jax.numpy as jnp

from . import hamiltonian as ham, memory, projector as pj, reference as ref

SI = (
    [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)],
    {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    (4, 4, 4), 7.0,
)
# (label, k, whether the occupied window encloses a symmetry multiplet)
CASES = [
    ("Si Gamma", (0.0, 0.0, 0.0), True),
    ("Si generic", (0.1, 0.2, 0.05), False),
]
STEP_COUNTS = (10, 20, 30, 40, 60)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def _observable(n):
    return np.diag(np.linspace(-1.0, 1.0, n)).astype(complex)


def _loss(p, m):
    return jnp.real(jnp.trace(p @ m))


# --------------------------------------------------------------------------- A


def step_count(reduced, nocc, tol, elk_projector):
    r"""What the iteration costs here, and where it converges.

    The predicted count is :math:`\log(\lVert\tilde H-\mu\rVert_2/\Delta)/\log(3/2)`;
    the measured one is where the forward projector stops moving.  Reporting both is
    the point -- the study's estimate is about the *spectrum*, and a real LAPW matrix's
    extreme eigenvalue is nothing like its valence bandwidth.
    """
    hj = jnp.asarray(reduced)
    evals = np.linalg.eigvalsh(reduced)
    gap = float(evals[nocc] - evals[nocc - 1])
    mu = 0.5 * (evals[nocc - 1] + evals[nocc])
    ratio = float(np.linalg.norm(reduced - mu * np.eye(reduced.shape[0]), 2) / gap)
    safe = np.asarray(pj.hard_window_projector(hj, nocc, tol))
    rows = []
    for steps in STEP_COUNTS:
        start = time.perf_counter()
        p = np.asarray(pj.sign_projector(hj, nocc, steps=steps))
        rows.append(dict(steps=steps, seconds=time.perf_counter() - start,
                         vs_safe=float(np.linalg.norm(p - safe)),
                         vs_elk=float(np.linalg.norm(p - elk_projector)),
                         idempotency=float(np.linalg.norm(p @ p - p))))
    return dict(gap=gap, ratio=ratio,
                predicted=float(np.log(ratio) / np.log(1.5)),
                spectrum=(float(evals[0]), float(evals[-1])), rows=rows)


# --------------------------------------------------------------------------- B


def first_derivative(reduced, nocc, tol, steps, ndir=3, seed=3100):
    """The eigensolver-free route against the validated first-order one."""
    n = reduced.shape[0]
    m = _observable(n)
    hj, mj = jnp.asarray(reduced), jnp.asarray(m)
    occ = ref.hard_occupations(n, nocc)
    rows = []
    for j in range(ndir):
        d = ref.random_hermitian_direction(n, seed + j)
        dj = jnp.asarray(d)
        rows.append(dict(
            exact=ref.directional_derivative(reduced, d, m, occ, tol=tol),
            sign=float(jax.grad(lambda t: _loss(
                pj.sign_projector(hj + t * dj, nocc, steps=steps), mj))(0.0)),
            safe=float(jax.grad(lambda t: _loss(
                pj.hard_window_projector(hj + t * dj, nocc, tol), mj))(0.0))))
    return rows


# --------------------------------------------------------------------------- C


def second_derivative(reduced, nocc, tol, steps, ndir=3, seed=3200, fd_step=1e-4):
    r"""``grad(grad)`` against a central difference of the first derivative.

    The reference is *not* a second difference of the loss, which loses four digits;
    it is a central difference of the safe-:math:`K` rule's first derivative, which is
    itself validated in §1f against the closed form and shares no code with the sign
    route.  The second difference is carried alongside as the coarse control.
    """
    n = reduced.shape[0]
    m = _observable(n)
    hj, mj = jnp.asarray(reduced), jnp.asarray(m)
    rows = []
    for j in range(ndir):
        d = ref.random_hermitian_direction(n, seed + j)
        dj = jnp.asarray(d)

        def sign_loss(t):
            return _loss(pj.sign_projector(hj + t * dj, nocc, steps=steps), mj)

        def safe_loss(t):
            return _loss(pj.hard_window_projector(hj + t * dj, nocc, tol), mj)

        def naive_loss(t):
            return _loss(pj.naive_hard_window_projector(hj + t * dj, nocc), mj)

        safe_grad = jax.grad(safe_loss)
        reference = float((safe_grad(fd_step) - safe_grad(-fd_step)) / (2 * fd_step))
        loss0 = float(safe_loss(0.0))
        second_difference = float(
            (safe_loss(fd_step) - 2 * loss0 + safe_loss(-fd_step)) / fd_step ** 2)
        rows.append(dict(
            reference=reference, second_difference=second_difference,
            sign=float(jax.grad(jax.grad(sign_loss))(0.0)),
            safe=float(jax.grad(jax.grad(safe_loss))(0.0)),
            naive=float(jax.grad(jax.grad(naive_loss))(0.0))))
    return rows


# --------------------------------------------------------------------------- D


def second_derivative_in_k(export, nocc, tol, steps, seed=3300,
                           fd_steps=(1e-3, 3e-4, 1e-4, 3e-5)):
    r"""The same through the whole k-dependent assembly, with the FD step REFINED.

    A single step cannot say which side of a disagreement is wrong.  Refining it can:
    if the reference marches toward the AD value as :math:`O(h^2)`, it is the finite
    difference converging onto AD and not the other way round -- the same argument
    Phase 0a′ used on the toy.
    """
    kc = np.asarray(export["vkc"])

    def reduced_at(kvec):
        return ham.cholesky_reduce(*ham.eigenproblem_at(export, kvec))[0]

    n = int(reduced_at(jnp.asarray(kc)).shape[0])
    mj = jnp.asarray(_observable(n))
    rng = np.random.default_rng(seed)
    dk = rng.normal(size=3)
    dk /= np.linalg.norm(dk)
    kj, dkj = jnp.asarray(kc), jnp.asarray(dk)

    def sign_loss(t):
        return _loss(pj.sign_projector(reduced_at(kj + t * dkj), nocc, steps=steps), mj)

    def safe_loss(t):
        return _loss(pj.hard_window_projector(reduced_at(kj + t * dkj), nocc, tol), mj)

    safe_grad = jax.grad(safe_loss)
    start = time.perf_counter()
    value = float(jax.grad(jax.grad(sign_loss))(0.0))
    seconds = time.perf_counter() - start
    references = [float((safe_grad(h) - safe_grad(-h)) / (2 * h)) for h in fd_steps]
    return dict(
        sign=value, seconds=seconds, fd_steps=fd_steps, references=references,
        reference=references[-1],
        first_sign=float(jax.grad(sign_loss)(0.0)),
        first_safe=float(safe_grad(0.0)))


# --------------------------------------------------------------------------- main


def _ground_state(workdir):
    from elkpy.structure import Structure

    avec, species, ngridk, rgkmax = SI
    calculation = Structure(avec, species).get_calculation(
        f"{workdir}/Si", xc="PW", ngridk=ngridk, rgkmax=rgkmax)
    calculation.ensure_ground_state()
    efermi = float((calculation.workdir / "EFERMI.OUT").read_text().split()[0])
    return calculation, efermi


def main(workdir="phase1_secondorder", steps=40):
    memory.limit_address_space(16.0)
    calculation, efermi = _ground_state(workdir)
    for label, k, multiplet in CASES:
        with calculation.eigenstate_session() as session:
            export = session.lapw_problem(k)
        kc = np.asarray(export["vkc"])
        h, o = ham.eigenproblem_at(export, kc)
        reduced, _ = ham.cholesky_reduce(h, o)
        reduced = np.asarray(reduced)
        tol = ham.projector_tolerance(reduced, o)
        nocc = ham.occupied_band_count(export["evalfv"], efermi)
        print(f"\n{'=' * 78}\n{label}   k={k}   nmat={export['nmatp']}   nocc={nocc}   "
              f"{'multiplet enclosed' if multiplet else 'no degeneracy'}\n{'=' * 78}",
              flush=True)

        a = step_count(reduced, nocc, tol, ham.elk_occupied_projector(export, nocc))
        print(f"A. spectrum [{a['spectrum'][0]:.3f}, {a['spectrum'][1]:.1f}] Ha   "
              f"boundary gap {a['gap']:.4f} Ha   |H~-mu|/gap = {a['ratio']:.3e}")
        print(f"   predicted Newton-Schulz steps log(ratio)/log(1.5) = "
              f"{a['predicted']:.1f}")
        for r in a["rows"]:
            print(f"     steps {r['steps']:3d}   vs safe-K {r['vs_safe']:.3e}   "
                  f"vs Elk {r['vs_elk']:.3e}   |P^2-P| {r['idempotency']:.3e}   "
                  f"{r['seconds'] * 1e3:.0f} ms")

        rows = first_derivative(reduced, nocc, tol, steps)
        print(f"B. first derivative, 3 Hermitian directions (steps={steps}):")
        for r in rows:
            print(f"     exact {r['exact']: .10f}   sign {_rel(r['sign'], r['exact']):.2e}"
                  f"   safe {_rel(r['safe'], r['exact']):.2e}")

        rows = second_derivative(reduced, nocc, tol, steps)
        print(f"C. second derivative, grad(grad) vs central FD of the first:")
        for r in rows:
            print(f"     reference {r['reference']: .8f}   sign {r['sign']: .8f} "
                  f"({_rel(r['sign'], r['reference']):.2e})   "
                  f"safe {r['safe']: .6f} ({_rel(r['safe'], r['reference']):.2e})   "
                  f"naive {r['naive']: .6f}")
            print(f"       second difference of the loss {r['second_difference']: .6f} "
                  f"({_rel(r['second_difference'], r['reference']):.2e})")

        d = second_derivative_in_k(export, nocc, tol, steps)
        print(f"D. in k: first sign {d['first_sign']: .10f}  safe {d['first_safe']: .10f}"
              f"  ({_rel(d['first_sign'], d['first_safe']):.2e})")
        print(f"   second sign {d['sign']: .10f}   ({d['seconds']:.1f} s)")
        for h, value in zip(d["fd_steps"], d["references"]):
            print(f"     central FD of the first derivative at h={h:.0e}: "
                  f"{value: .10f}   rel to AD {_rel(value, d['sign']):.2e}")

    print(f"\npeak RSS {memory.peak_rss_bytes() / memory.GB:.2f} GB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "phase1_secondorder")
