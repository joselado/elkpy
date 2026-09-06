r"""Phase 1i: smeared occupations, and the self-consistent Fermi level, on real matrices.

``docs/continue_here.md`` §3 item 4 ranks this next, and gives the reason §1f leaves it
open: for a **hard** integer window the near-degenerate branch of the divided-difference
kernel is *inert*.  Both branches of a same-side pair are identically zero
(:math:`f_i-f_j=0` exactly and :math:`f'=0`), so §1f's plateau in ``tol`` measures
nothing.  With Fermi-Dirac occupations the branch value is :math:`f'(\bar\lambda)`,
which at a half-filled level is :math:`-1/4w` -- large, not zero -- so the threshold is
load-bearing for the first time.  **That is the history rather than the current state**:
what the measurements below showed is that a threshold was the wrong instrument, and
:func:`elkjax.projector.smeared_projector` now carries the cancellation-free closed form
in its JVP instead, which makes ``tol`` inert for smeared occupations.  The experiments
are kept as they were run, because the route they retire is still the one a Phase 1
developer writes first.

**The fixture is graphene at** :math:`K`, and it is the physically right one rather than
an engineered one: the two :math:`\pi` bands are degenerate there *and* the Fermi level
sits on them, so :math:`f=\tfrac12` exactly and :math:`f'` is at its maximum.  Bulk
silicon at :math:`\Gamma` is the second fixture, for the opposite reason -- §1f measured
its :math:`\Gamma_{25'}` triplet split by 5.1e-15 Ha for one pair, at the assembly's
roundoff floor, where graphene's Dirac pair is split by 3.4e-7 Ha.  The two together
span the range a real LAPW multiplet actually occupies.

**The oracle is analytic, not finite differences.**  :math:`P=f(H)` is a smooth matrix
function when :math:`f` is smooth, so unlike §1f's hard window the derivative *exists* at
every degeneracy, and unlike §1h there is no branch exchange for central differences to
average over -- FD is a legitimate second check here.  But the primary reference is
:func:`elkjax.reference.fermi_divided_difference_kernel`, a closed form for the logistic
difference quotient with **no subtraction in it**, so it is accurate at any splitting and
can arbitrate between the two branches instead of assuming one.

Five experiments::

    PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase1_smearing [workdir]

* **A, forward and the physical anchor.**  :math:`\sum_i f_i` against the cell's own
  electron count at Elk's ``EFERMI.OUT`` and ``swidth``; and, for graphene at
  :math:`K`, the single-:math:`k` :math:`\mu` from :func:`elkjax.projector.fermi_level`
  against that same Elk Fermi level -- which agree only because particle-hole symmetry
  at the Dirac point puts the zone-integrated answer at the local one.
* **B, kernel anatomy.**  For every close pair: the splitting, the run's tolerance, the
  direct quotient, :math:`f'(\bar\lambda)`, and which of the two the exact kernel
  agrees with.  This is what says whether the branch fires at all -- a diagnostic of the
  retired mechanism, kept because it is also what shows the closed form agreeing with
  *both* candidates where each is right, which is why it can replace them.
* **C, gradient along Hermitian directions, swept in** :math:`w`.  Three routes --
  JAX's own ``eigh`` rule, the literal difference quotient
  (:func:`elkjax.projector.direct_quotient_projector`) and the safe rule -- against the
  exact kernel and against central FD.  Sweeping :math:`w` is the point: the quotient's
  relative error is :math:`\sim\epsilon\,w/\Delta\lambda`, so it *grows* with the
  smearing width, which is the opposite of the intuition that broader smearing is safer.
  Since the measurements below were first taken, :func:`elkjax.projector.smeared_projector`
  has been changed to carry the cancellation-free kernel in its JVP, so ``tol`` no longer
  selects anything for it -- the quotient route is kept because it is the thing under
  test, not because it is an implementation.
* **C(ii), what is left after that.**  The residual is the two eigensolvers disagreeing
  about the eigenvalues, amplified by the kernel's :math:`1/w` relative sensitivity --
  measured, not inferred, by rerunning the same closed form on XLA's own decomposition.
* **D, gradient in** :math:`k`.  The same at the end of the whole differentiable
  pipeline.  The FD step must satisfy :math:`v_F h\ll w` or the cubic term dominates.
* **E, the self-consistent Fermi level.**  Fixed :math:`N` against fixed :math:`\mu`,
  with §8(b)'s closed form :math:`d\mu=\sum_j f'_jA_{jj}/\sum_j f'_j` as
  :func:`elkjax.projector.fermi_level`'s own JVP.  Along random Hermitian directions,
  never along :math:`k`: at :math:`K` the pair's trace is stationary by symmetry, so a
  :math:`k`-direction test would pass with the correction identically zero -- the
  "perturbation respects the protecting symmetry" trap of Phase 0a in a new place.

Results and their conclusions: ``docs/jax_port_phase1.md`` §1i.
"""

import sys

import numpy as np

import jax
import jax.numpy as jnp

from . import hamiltonian as ham, memory, projector as pj, reference as ref

_A = 4.6511                                  # graphene, a = 2.461 Angstrom
GRAPHENE = (
    [(_A, 0.0, 0.0), (-_A / 2, _A * np.sqrt(3) / 2, 0.0), (0.0, 0.0, 20.0)],
    {"C": [(0.0, 0.0, 0.0), (1 / 3, 2 / 3, 0.0)]},
    (6, 6, 1), 6.0, 4.0,
)
SI = (
    [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)],
    {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    (4, 4, 4), 7.0, 4.0,
)
# (label, structure, k, does the Fermi level sit ON the degeneracy)
CASES = [
    ("graphene K", GRAPHENE, (1 / 3, 1 / 3, 0.0), True),
    ("Si Gamma", SI, (0.0, 0.0, 0.0), False),
]
WIDTHS = (1e-3, 1e-2, 1e-1)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def _observable(n):
    return np.diag(np.linspace(-1.0, 1.0, n)).astype(complex)


def _loss(p, m):
    return jnp.real(jnp.trace(p @ m))


def _np_loss(reduced, mu, width, m):
    evals, evecs = np.linalg.eigh(np.asarray(reduced))
    f, _ = ref.fermi_dirac(evals, mu, width)
    return float(np.real(np.trace(((evecs * f) @ evecs.conj().T) @ m)))


def _np_loss_fixed_n(reduced, nelec, width, m):
    evals, evecs = np.linalg.eigh(np.asarray(reduced))
    f, _ = ref.fermi_dirac(evals, ref.fermi_level(evals, nelec, width), width)
    return float(np.real(np.trace(((evecs * f) @ evecs.conj().T) @ m)))


# --------------------------------------------------------------------------- A


def forward(reduced, export, efermi, width, nelec):
    """Occupations against the cell's own electron count, and mu against Elk's."""
    evals = np.linalg.eigvalsh(np.asarray(reduced))
    f, _ = ref.fermi_dirac(evals, efermi, width)
    mu_local = float(pj.fermi_level(jnp.asarray(reduced), nelec, width))
    p = np.asarray(pj.smeared_projector(jnp.asarray(reduced), efermi, width, 0.0))
    return dict(
        occupancy=float(f.sum()), nelec=nelec,
        mu_local=mu_local, efermi=efermi,
        hermiticity=float(np.linalg.norm(p - p.conj().T)),
        trace=float(np.real(np.trace(p))))


# --------------------------------------------------------------------------- B


def kernel_anatomy(reduced, tol, mu, width, nshow=3):
    """Which pairs the near-degenerate branch fires on, and what it is worth.

    Adjacent pairs ranked by :math:`|K_{i,i+1}|`, i.e. by how much of the derivative
    they carry -- NOT by how close they are.  Ranking by the splitting alone selects
    the deep and the far-empty states, where the two branches are both ~1e-200 and
    agreeing or disagreeing about them means nothing; the pair that matters is the one
    at the Fermi level.
    """
    evals = np.linalg.eigvalsh(np.asarray(reduced))
    exact = ref.fermi_divided_difference_kernel(evals, mu, width)
    f, docc = ref.fermi_dirac(evals, mu, width)
    weight = np.abs(np.diag(exact, 1))
    order = np.argsort(-weight)[:nshow]
    rows = []
    for i in sorted(int(x) for x in order):
        dl = evals[i] - evals[i + 1]          # the kernel's own denominator order
        direct = (f[i] - f[i + 1]) / dl if dl != 0.0 else np.inf
        branch = 0.5 * (docc[i] + docc[i + 1])
        rows.append(dict(pair=(int(i), int(i) + 1), split=float(dl), tol=tol,
                         fires=bool(abs(dl) <= tol), split_abs=float(abs(dl)),
                         direct=float(direct), branch=float(branch),
                         exact=float(exact[i, i + 1]),
                         occ=(float(f[i]), float(f[i + 1]))))
    return rows


# --------------------------------------------------------------------------- C


def matrix_direction(reduced, tol, mu, width, ndir=5, seed=1400):
    """The three routes on Elk's own reduced matrix, against the exact kernel."""
    reduced = np.asarray(reduced)
    n = reduced.shape[0]
    m = _observable(n)
    hj, mj = jnp.asarray(reduced), jnp.asarray(m)
    rows = []
    for j in range(ndir):
        d = ref.random_hermitian_direction(n, seed + j)
        dj = jnp.asarray(d)
        exact = float(np.real(np.trace(
            ref.dprojector_fermi(reduced, d, mu, width) @ m)))
        naive = lambda t: _loss(pj.naive_smeared_projector(hj + t * dj, mu, width), mj)
        direct = lambda t: _loss(pj.direct_quotient_projector(hj + t * dj, mu, width), mj)
        safe = lambda t: _loss(pj.smeared_projector(hj + t * dj, mu, width, tol), mj)
        step = min(1e-5, 0.1 * width)
        fd = (_np_loss(reduced + step * d, mu, width, m)
              - _np_loss(reduced - step * d, mu, width, m)) / (2 * step)
        rows.append(dict(
            exact=exact, fd=fd,
            naive_fwd=float(jax.jvp(naive, (0.0,), (1.0,))[1]),
            naive_rev=float(jax.grad(naive)(0.0)),
            direct_rev=float(jax.grad(direct)(0.0)),
            safe_fwd=float(jax.jvp(safe, (0.0,), (1.0,))[1]),
            safe_rev=float(jax.grad(safe)(0.0))))
    return rows


# ------------------------------------------------------------------------ C(ii)


def eigensolver_floor(reduced, mu, width, ndir=3, seed=1450):
    r"""What is left after the closed-form kernel: the two eigensolvers, not the rule.

    The reference builds its own decomposition with LAPACK while ``smeared_projector``
    uses XLA's, and the two disagree at :math:`\sim10^{-14}` Ha.  The kernel's
    sensitivity to an eigenvalue is :math:`\sim f''\sim1/w^2` against a value
    :math:`\sim f'\sim1/w`, so that disagreement enters the RELATIVE error divided by
    :math:`w` -- which is why the residual shrinks as the smearing widens, the opposite
    of every other effect in §1i.  Rerunning the same closed form on XLA's own
    decomposition removes it and is the measurement that says so.
    """
    n = reduced.shape[0]
    m = _observable(n)
    hj, mj = jnp.asarray(reduced), jnp.asarray(m)
    evals, evecs = (np.asarray(x) for x in jnp.linalg.eigh(hj))
    kernel = ref.fermi_divided_difference_kernel(evals, mu, width)
    rows = []
    for j in range(ndir):
        d = ref.random_hermitian_direction(n, seed + j)
        dj = jnp.asarray(d)
        ad = float(jax.grad(lambda t: _loss(
            pj.smeared_projector(hj + t * dj, mu, width, 0.0), mj))(0.0))
        lapack = float(np.real(np.trace(
            ref.dprojector_fermi(reduced, d, mu, width) @ m)))
        a = evecs.conj().T @ d @ evecs
        xla = float(np.real(np.trace((evecs @ (kernel * a) @ evecs.conj().T) @ m)))
        rows.append(dict(vs_lapack=_rel(ad, lapack), vs_xla=_rel(ad, xla),
                         between=_rel(lapack, xla)))
    return rows


# --------------------------------------------------------------------------- D


def k_direction(export, tol, mu, width, ndir=2, seed=1500):
    """The whole pipeline differentiated in k, with smeared occupations."""
    kc = np.asarray(export["vkc"])

    def reduced_at(kvec):
        return ham.cholesky_reduce(*ham.eigenproblem_at(export, kvec))[0]

    reduced0 = np.asarray(reduced_at(jnp.asarray(kc)))
    m = _observable(reduced0.shape[0])
    mj = jnp.asarray(m)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(ndir):
        dk = rng.normal(size=3)
        dk /= np.linalg.norm(dk)
        dh = np.asarray(jax.jvp(reduced_at, (jnp.asarray(kc),),
                                (jnp.asarray(dk),))[1])
        dh = 0.5 * (dh + dh.conj().T)
        exact = float(np.real(np.trace(
            ref.dprojector_fermi(reduced0, dh, mu, width) @ m)))
        naive = lambda t: _loss(pj.naive_smeared_projector(
            reduced_at(jnp.asarray(kc) + t * jnp.asarray(dk)), mu, width), mj)
        safe = lambda t: _loss(pj.smeared_projector(
            reduced_at(jnp.asarray(kc) + t * jnp.asarray(dk)), mu, width, tol), mj)
        # v_F h << w, or the cubic term of the band dispersion dominates the FD
        step = min(1e-5, 1e-2 * width)
        fd = (_np_loss(reduced_at(jnp.asarray(kc + step * dk)), mu, width, m)
              - _np_loss(reduced_at(jnp.asarray(kc - step * dk)), mu, width, m)) / (2 * step)
        rows.append(dict(exact=exact, fd=fd, step=step,
                         naive_rev=float(jax.grad(naive)(0.0)),
                         safe_fwd=float(jax.jvp(safe, (0.0,), (1.0,))[1]),
                         safe_rev=float(jax.grad(safe)(0.0))))
    return rows


# --------------------------------------------------------------------------- E


def fermi_level_rule(reduced, tol, nelec, width, ndir=5, seed=1600):
    r"""Fixed :math:`N` against fixed :math:`\mu`, and both against FD.

    The reference for the fixed-:math:`N` column re-solves :math:`\mu` at each
    displaced matrix, so it carries the constraint that the closed form encodes.
    """
    reduced = np.asarray(reduced)
    n = reduced.shape[0]
    m = _observable(n)
    hj, mj = jnp.asarray(reduced), jnp.asarray(m)
    mu = float(pj.fermi_level(hj, nelec, width))
    rows = []
    for j in range(ndir):
        d = ref.random_hermitian_direction(n, seed + j)
        dj = jnp.asarray(d)
        fixed_mu = float(np.real(np.trace(
            ref.dprojector_fermi(reduced, d, mu, width) @ m)))
        exact = float(np.real(np.trace(
            ref.dprojector_fermi(reduced, d, mu, width, dmu="selfconsistent") @ m)))
        loss_n = lambda t: _loss(
            pj.fixed_number_projector(hj + t * dj, nelec, width, tol), mj)
        step = min(1e-5, 0.1 * width)
        fd = (_np_loss_fixed_n(reduced + step * d, nelec, width, m)
              - _np_loss_fixed_n(reduced - step * d, nelec, width, m)) / (2 * step)
        dmu = float(jax.jvp(lambda t: pj.fermi_level(hj + t * dj, nelec, width),
                            (0.0,), (1.0,))[1])
        dmu_fd = (ref.fermi_level(np.linalg.eigvalsh(reduced + step * d), nelec, width)
                  - ref.fermi_level(np.linalg.eigvalsh(reduced - step * d), nelec, width)
                  ) / (2 * step)
        rows.append(dict(exact=exact, fixed_mu=fixed_mu, fd=fd,
                         ad_rev=float(jax.grad(loss_n)(0.0)),
                         ad_fwd=float(jax.jvp(loss_n, (0.0,), (1.0,))[1]),
                         dmu=dmu, dmu_fd=dmu_fd))
    return rows


# --------------------------------------------------------------------------- main


def _ground_state(workdir, label, structure):
    from elkpy.structure import Structure

    avec, species, ngridk, rgkmax, _ = structure
    calculation = Structure(avec, species).get_calculation(
        f"{workdir}/{label}", xc="PW", ngridk=ngridk, rgkmax=rgkmax)
    calculation.ensure_ground_state()
    efermi = float((calculation.workdir / "EFERMI.OUT").read_text().split()[0])
    return calculation, efermi


def main(workdir="phase1_smearing"):
    memory.limit_address_space(16.0)
    for label, structure, k, on_degeneracy in CASES:
        nelec = structure[4]
        calculation, efermi = _ground_state(workdir, label.split()[0], structure)
        with calculation.eigenstate_session() as session:
            export = session.lapw_problem(k)
        kc = np.asarray(export["vkc"])
        h, o = ham.eigenproblem_at(export, kc)
        reduced, _ = ham.cholesky_reduce(h, o)
        tol = ham.projector_tolerance(reduced, o)
        evals = np.linalg.eigvalsh(np.asarray(reduced))
        print(f"\n{'=' * 78}\n{label}   k={k}   nmat={export['nmatp']}   "
              f"E_F={efermi:.9f} Ha   tol={tol:.3e} Ha"
              f"\n{'Fermi level ON the degeneracy' if on_degeneracy else 'gapped at this k'}"
              f"\n{'=' * 78}", flush=True)

        f = forward(reduced, export, efermi, 1e-3, nelec)
        print(f"A. sum f at Elk's E_F and swidth=1e-3: {f['occupancy']:.10f} "
              f"(electron count {f['nelec']:.1f})")
        print(f"   single-k mu from the same count:    {f['mu_local']:.10f}   "
              f"Elk E_F {f['efermi']:.10f}   diff {f['mu_local'] - f['efermi']:+.2e} Ha")
        print(f"   |P - P^dag| {f['hermiticity']:.2e}   Tr P {f['trace']:.10f}")

        for width in WIDTHS:
            mu = efermi
            print(f"\n--- swidth = {width:.0e} Ha "
                  f"({width * 315775.0:.0f} K) {'-' * 40}")
            for r in kernel_anatomy(reduced, tol, mu, width):
                which = ("branch" if abs(r["exact"] - r["branch"])
                         < abs(r["exact"] - r["direct"]) else "quotient")
                print(f"B. pair {r['pair']}  split {r['split_abs']:.3e} Ha "
                      f"({r['split_abs'] / tol:.2e} x tol)  fires={r['fires']}  "
                      f"f=({r['occ'][0]:.4f},{r['occ'][1]:.4f})")
                print(f"     quotient {r['direct']: .8e}   f' {r['branch']: .8e}   "
                      f"exact {r['exact']: .8e}   -> {which}, "
                      f"rel diff {_rel(r['direct'], r['exact']):.2e}")

            rows = matrix_direction(reduced, tol, mu, width)
            worst = {key: max(_rel(r[key], r["exact"]) for r in rows)
                     for key in ("fd", "naive_fwd", "naive_rev", "direct_rev",
                                 "safe_fwd", "safe_rev")}
            print(f"C. 5 Hermitian directions, worst relative error vs the exact kernel:")
            print(f"     central FD {worst['fd']:.2e}   "
                  f"naive fwd {worst['naive_fwd']:.2e}  rev {worst['naive_rev']:.2e}   "
                  f"quotient rev {worst['direct_rev']:.2e}   "
                  f"safe fwd {worst['safe_fwd']:.2e}  rev {worst['safe_rev']:.2e}")

            rows = eigensolver_floor(reduced, mu, width)
            worst = {key: max(r[key] for r in rows)
                     for key in ("vs_lapack", "vs_xla", "between")}
            print(f"C(ii). the residual is the eigensolver, not the kernel: "
                  f"AD vs LAPACK-built reference {worst['vs_lapack']:.2e}, "
                  f"vs XLA-built {worst['vs_xla']:.2e}, "
                  f"the two references differ by {worst['between']:.2e}")

            rows = k_direction(export, tol, mu, width)
            for r in rows:
                print(f"D. k: exact {r['exact']: .10f}  safe rev {r['safe_rev']: .10f} "
                      f"({_rel(r['safe_rev'], r['exact']):.2e})  "
                      f"naive rev {r['naive_rev']: .10f} "
                      f"({_rel(r['naive_rev'], r['exact']):.2e})  "
                      f"FD@{r['step']:.0e} {r['fd']: .8f} "
                      f"({_rel(r['fd'], r['exact']):.2e})")

            try:
                mu_n, response = pj.check_fermi_level_determined(
                    jnp.asarray(reduced), nelec, width)
            except ValueError as exc:
                print(f"E. check_fermi_level_determined REFUSES: {str(exc)[:110]}")
                continue
            rows = fermi_level_rule(reduced, tol, nelec, width)
            worst = {key: max(_rel(r[key], r["exact"]) for r in rows)
                     for key in ("fd", "ad_rev", "ad_fwd", "fixed_mu")}
            print(f"E. fixed N: mu={mu_n:.9f}  sum|f'|={response:.3e}   "
                  f"worst relative error vs the exact fixed-N derivative:")
            print(f"     AD fwd {worst['ad_fwd']:.2e}  rev {worst['ad_rev']:.2e}   "
                  f"central FD {worst['fd']:.2e}   "
                  f"fixed-mu (the term omitted) {worst['fixed_mu']:.2e}")
            print(f"     dmu: AD {rows[0]['dmu']: .8e}  FD {rows[0]['dmu_fd']: .8e}  "
                  f"rel {_rel(rows[0]['dmu'], rows[0]['dmu_fd']):.2e}")

    print(f"\npeak RSS {memory.peak_rss_bytes() / memory.GB:.2f} GB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "phase1_smearing")
