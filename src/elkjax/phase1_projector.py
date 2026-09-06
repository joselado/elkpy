r"""Phase 1f: the Cholesky-reduced eigensolve wired to the safe-$K$ projector rule.

``docs/continue_here.md`` §3 ranks this first among what is left of Phase 1, and says
why: :func:`elkjax.hamiltonian.first_variational_eigenvalues` closes with a plain
``eigvalsh``, so at a multiplet it fails exactly the way item 0b describes, while
``projector.py``'s rule -- which exists and works -- had **only ever been exercised at a
synthetic overlap with a prescribed** :math:`\kappa(S)`.  §3 calls that the biggest hole
left in Phase 0.  Everything needed to close it is now in place: real :math:`H` and
:math:`O` at any :math:`k` (patch 0013), a real :math:`\kappa(O)`, and the tolerance
:math:`\epsilon\,\kappa(O)\,\lVert L^{-1}HL^{-\dagger}\rVert`.

What a real matrix supplies that a synthetic one cannot is a **real multiplet**.  Bulk
silicon at :math:`\Gamma` carries the :math:`\Gamma_{25'}` valence triplet *inside* the
occupied window, with the window boundary gapped by 0.09 Ha -- which is the disputed
configuration of experiment 0b-A, arrived at by symmetry rather than by engineering a
spectrum.  A generic :math:`k` on the same ground state is the control: no degeneracy,
so both routes must agree there or the comparison means nothing.

Four experiments::

    PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase1_projector [workdir]

* **A, forward.** :math:`\lVert\tilde P_{\rm JAX}-\tilde P_{\rm Elk}\rVert_F` with
  :math:`\tilde P_{\rm Elk}=YY^\dagger`, :math:`Y=L^\dagger C`.  This is the study's own
  Phase 1 forward criterion, and it is a *projector* comparison because ``evecfv`` is
  arbitrary inside the triplet.  Phase 0's first carried-forward finding is that a green
  gradient test does not validate a transcription, so this runs first.
* **B, gradient along a Hermitian matrix direction.**  The experiment-0b-B analogue with
  Elk's own :math:`\tilde H` in place of a synthetic one: it isolates the projector rule
  from the :math:`k`-pipeline, so a failure names one of them.
* **C, gradient in** :math:`k`.  The composition that Phase 1 actually wants.  The
  reference is *not* finite differences of the loss: :math:`d\tilde H/dk` comes from a
  matrix-valued ``jvp`` (every step from ``match`` to the Cholesky reduction is smooth --
  only the projector is not) and is then fed to the closed form.  Central FD of the same
  loss is carried alongside as the control that separates an AD bug from a broken test.
* **D, the tolerance.**  What :math:`\epsilon\,\kappa(O)\,\lVert\tilde H\rVert` actually
  is on these runs, next to the splittings the matrices actually carry.

Results and their conclusions: ``docs/jax_port_phase1.md`` §1f.
"""

import sys

import numpy as np

import jax
import jax.numpy as jnp

from . import hamiltonian as ham, memory, projector as pj, reference as ref

SI = (
    [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)],
    {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    (4, 4, 4),
)
_A = 4.7443                      # monolayer h-BN, a = 2.511 Angstrom
HBN = (
    [(_A, 0.0, 0.0), (-_A / 2, _A * np.sqrt(3) / 2, 0.0), (0.0, 0.0, 20.0)],
    {"B": [(0.0, 0.0, 0.0)], "N": [(1 / 3, 2 / 3, 0.0)]},
    (4, 4, 1),
)
# (label, structure, k, whether the occupied window encloses a symmetry multiplet)
CASES = [
    ("Si Gamma", SI, (0.0, 0.0, 0.0), True),
    ("Si generic", SI, (0.1, 0.2, 0.05), False),
    ("hBN Gamma", HBN, (0.0, 0.0, 0.0), True),
]


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def _loss(p, m):
    return jnp.real(jnp.trace(p @ m))


def _np_loss(reduced, nocc, m):
    _, v = np.linalg.eigh(np.asarray(reduced))
    p = v[:, :nocc] @ v[:, :nocc].conj().T
    return float(np.real(np.trace(p @ m)))


def _observable(n):
    return np.diag(np.linspace(-1.0, 1.0, n)).astype(complex)


# --------------------------------------------------------------------------- A


def forward(export, nocc, tol):
    """Elk's own occupied subspace against the one assembled here."""
    kc = np.asarray(export["vkc"])
    reduced, _ = ham.cholesky_reduce(*ham.eigenproblem_at(export, kc))
    p_jax = np.asarray(pj.hard_window_projector(reduced, nocc, tol))
    p_elk = ham.elk_occupied_projector(export, nocc)
    evals = np.asarray(jnp.linalg.eigvalsh(reduced))
    return dict(
        projector_error=float(np.linalg.norm(p_jax - p_elk)),
        eigenvalue_error=float(np.abs(evals[:export["nstfv"]]
                                      - export["evalfv"]).max()),
        idempotency=float(np.linalg.norm(p_jax @ p_jax - p_jax)),
    )


# --------------------------------------------------------------------------- B


def matrix_direction(export, nocc, tol, ndir=5, seed=800):
    """The projector rule alone, on Elk's own reduced matrix."""
    kc = np.asarray(export["vkc"])
    reduced, _ = ham.cholesky_reduce(*ham.eigenproblem_at(export, kc))
    reduced = np.asarray(reduced)
    n = reduced.shape[0]
    m = _observable(n)
    occ = ref.hard_occupations(n, nocc)
    hj, mj = jnp.asarray(reduced), jnp.asarray(m)
    rows = []
    for j in range(ndir):
        d = ref.random_hermitian_direction(n, seed + j)
        dj = jnp.asarray(d)
        exact = ref.directional_derivative(reduced, d, m, occ, tol=tol)
        naive = lambda t: _loss(pj.naive_hard_window_projector(hj + t * dj, nocc), mj)
        safe = lambda t: _loss(pj.hard_window_projector(hj + t * dj, nocc, tol), mj)
        fd = [(_np_loss(reduced + s * d, nocc, m)
               - _np_loss(reduced - s * d, nocc, m)) / (2 * s)
              for s in (1e-4, 1e-5, 1e-6)]
        rows.append(dict(
            exact=exact, fd=fd[1],
            naive_fwd=float(jax.jvp(naive, (0.0,), (1.0,))[1]),
            naive_rev=float(jax.grad(naive)(0.0)),
            safe_fwd=float(jax.jvp(safe, (0.0,), (1.0,))[1]),
            safe_rev=float(jax.grad(safe)(0.0))))
    return rows


# --------------------------------------------------------------------------- C


def k_direction(export, nocc, tol, ndir=3, seed=900):
    r"""The whole pipeline differentiated in :math:`k`.

    The reference is the closed form fed with :math:`d\tilde H/dk` from a
    matrix-valued ``jvp``.  That composition is legitimate precisely because the
    projector is the ONLY non-smooth step: ``match``, the Gaunt contractions, the
    Cholesky and the two solves are all analytic in :math:`k`, so their tangent is
    exact and carries no degeneracy.
    """
    kc = np.asarray(export["vkc"])
    reduced0, _ = ham.cholesky_reduce(*ham.eigenproblem_at(export, kc))
    n = int(reduced0.shape[0])
    m = _observable(n)
    occ = ref.hard_occupations(n, nocc)
    mj = jnp.asarray(m)

    def reduced_at(kvec):
        return ham.cholesky_reduce(*ham.eigenproblem_at(export, kvec))[0]

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(ndir):
        dk = rng.normal(size=3)
        dk /= np.linalg.norm(dk)
        # dH~/dk along dk, exactly -- the smooth half of the chain
        dh = np.asarray(jax.jvp(reduced_at, (jnp.asarray(kc),),
                                (jnp.asarray(dk),))[1])
        dh = 0.5 * (dh + dh.conj().T)      # Hermitian by construction; enforce it
        exact = ref.directional_derivative(np.asarray(reduced0), dh, m, occ, tol=tol)
        naive = lambda t: _loss(pj.naive_hard_window_projector(
            reduced_at(jnp.asarray(kc) + t * jnp.asarray(dk)), nocc), mj)
        safe = lambda t: _loss(pj.hard_window_projector(
            reduced_at(jnp.asarray(kc) + t * jnp.asarray(dk)), nocc, tol), mj)
        fd = [(_np_loss(reduced_at(jnp.asarray(kc + s * dk)), nocc, m)
               - _np_loss(reduced_at(jnp.asarray(kc - s * dk)), nocc, m)) / (2 * s)
              for s in (1e-4, 1e-5, 1e-6)]
        rows.append(dict(
            exact=exact, fd=fd,
            naive_fwd=float(jax.jvp(naive, (0.0,), (1.0,))[1]),
            naive_rev=float(jax.grad(naive)(0.0)),
            safe_fwd=float(jax.jvp(safe, (0.0,), (1.0,))[1]),
            safe_rev=float(jax.grad(safe)(0.0))))
    return rows


# --------------------------------------------------------------------------- D


def resolution(export, nocc):
    """The tolerance, and the splittings the matrices actually carry."""
    kc = np.asarray(export["vkc"])
    h, o = ham.eigenproblem_at(export, kc)
    reduced, _ = ham.cholesky_reduce(h, o)
    tol = ham.projector_tolerance(reduced, o)
    evals = np.asarray(jnp.linalg.eigvalsh(reduced))
    inside = np.diff(evals[:nocc]) if nocc > 1 else np.array([np.inf])
    return dict(tol=tol, nocc=nocc, n=int(evals.size),
                kappa=float(np.linalg.eigvalsh(np.asarray(o)).max()
                            / np.linalg.eigvalsh(np.asarray(o)).min()),
                norm_reduced=float(np.linalg.norm(np.asarray(reduced), 2)),
                boundary_gap=float(evals[nocc] - evals[nocc - 1]),
                tightest_inside=float(inside.min()),
                evals=evals[:nocc + 2])


# --------------------------------------------------------------------------- main


def _ground_state(workdir, label, structure):
    from elkpy.structure import Structure

    avec, species, ngridk = structure
    calculation = Structure(avec, species).get_calculation(
        f"{workdir}/{label.replace(' ', '_')}", xc="PW",
        ngridk=ngridk, rgkmax=7.0)
    calculation.ensure_ground_state()
    efermi = float((calculation.workdir / "EFERMI.OUT").read_text().split()[0])
    return calculation, efermi


def main(workdir="phase1_projector"):
    memory.limit_address_space(16.0)
    for label, structure, k, multiplet in CASES:
        calculation, efermi = _ground_state(workdir, label.split()[0], structure)
        with calculation.eigenstate_session() as session:
            export = session.lapw_problem(k)
        nocc = ham.occupied_band_count(export["evalfv"], efermi)
        print(f"\n{'=' * 78}\n{label}   k={k}   nmat={export['nmatp']}   "
              f"nocc={nocc}   {'multiplet enclosed' if multiplet else 'no degeneracy'}"
              f"\n{'=' * 78}", flush=True)

        d = resolution(export, nocc)
        print(f"D. kappa(O)={d['kappa']:.3e}  |H~|={d['norm_reduced']:.3f}  "
              f"tol = eps kappa |H~| = {d['tol']:.3e} Ha")
        print(f"   occupied eigenvalues {np.array2string(d['evals'][:nocc], precision=9)}")
        print(f"   tightest split inside the window {d['tightest_inside']:.3e} Ha "
              f"({d['tightest_inside'] / d['tol']:.2e} x tol)")
        print(f"   boundary gap {d['boundary_gap']:.3e} Ha "
              f"({d['boundary_gap'] / d['tol']:.2e} x tol)")
        tol = d["tol"]

        try:
            checked_tol, gap = ham.occupied_window(export, np.asarray(export["vkc"]), nocc)
            print(f"   occupied_window accepts: gap {gap:.3e} > tol {checked_tol:.3e}")
        except ValueError as exc:
            print(f"   occupied_window REFUSES: {exc}")

        f = forward(export, nocc, tol)
        print(f"A. |P_JAX - P_Elk|_F = {f['projector_error']:.3e}   "
              f"max |eval - evalfv| = {f['eigenvalue_error']:.3e} Ha   "
              f"|P^2 - P|_F = {f['idempotency']:.3e}")

        rows = matrix_direction(export, nocc, tol)
        worst = {key: max(_rel(r[key], r["exact"]) for r in rows)
                 for key in ("fd", "naive_fwd", "naive_rev", "safe_fwd", "safe_rev")}
        print(f"B. Hermitian matrix directions (5): worst relative error")
        print(f"     central FD {worst['fd']:.2e}   "
              f"naive fwd {worst['naive_fwd']:.2e}  naive rev {worst['naive_rev']:.2e}   "
              f"safe fwd {worst['safe_fwd']:.2e}  safe rev {worst['safe_rev']:.2e}")

        rows = k_direction(export, nocc, tol)
        print(f"C. k directions (3): exact / naive rev / safe rev, and FD at 3 steps")
        for r in rows:
            print(f"     exact {r['exact']: .10f}   naive rev {r['naive_rev']: .6f}   "
                  f"safe rev {r['safe_rev']: .10f}")
            print(f"       FD {['%.10f' % x for x in r['fd']]}   "
                  f"safe rel {_rel(r['safe_rev'], r['exact']):.2e}   "
                  f"naive rel {_rel(r['naive_rev'], r['exact']):.2e}   "
                  f"fwd-vs-rev safe {_rel(r['safe_fwd'], r['safe_rev']):.2e}")

    print(f"\npeak RSS {memory.peak_rss_bytes() / memory.GB:.2f} GB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "phase1_projector")
