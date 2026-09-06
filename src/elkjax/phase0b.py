r"""Phase 0b: does the safe-:math:`K` projector rule work, and is it needed?

Study §6 item 0b, plus the unresolved measurement disagreement recorded in
``docs/continue_here.md`` §3.  Run it with::

    PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase0b

Five experiments, each printing the numbers that go into ``docs/jax_port_phase0.md``.
Every one uses the closed-form derivative of :mod:`elkjax.reference` as its reference
rather than finite differences, for the reason study §8(b) gives: at a multiplet, FD of
the sorted spectrum returns the branch average and cannot detect a wrong gradient.  (For
the *projector* the situation is better than that — see experiment A, where FD is stable
across three step sizes and agrees with the closed form, because a gapped window
boundary makes :math:`\mathrm{Tr}[PM]` smooth in :math:`H` whatever happens inside the
window.  That is what makes A decisive rather than suggestive.)

The tolerance sweep in experiment F is the "flat over at least two decades" requirement
study §8(b) sets, at a size where the plateau can actually vanish.

Every ``tol`` below is ``1e3 *`` the backward error :math:`\epsilon\,\kappa(S)\,\|H\|`.
That factor is a placeholder, not a derived number: the Cholesky-diagonal estimate of
:math:`\kappa(S)` is a provable *lower* bound (each :math:`L_{ii}^2` is a Schur pivot,
hence between :math:`\lambda_{\min}` and :math:`\lambda_{\max}`), and it came out 140x
low on a synthetic :math:`S` with :math:`\kappa_2=10^6`.  Experiment F is what shows the
answer does not depend on it over eight decades.
"""

import numpy as np

import jax
import jax.numpy as jnp

from . import memory, projector as pj, reference as ref


def _loss(p, m):
    return jnp.real(jnp.trace(p @ m))


def _rel(a, b):
    scale = max(abs(b), 1e-30)
    return abs(a - b) / scale


def _modes(fn):
    """Forward and reverse directional derivative of a scalar-in scalar-out ``fn``."""
    forward = float(jax.jvp(fn, (0.0,), (1.0,))[1])
    reverse = float(jax.grad(fn)(0.0))
    return forward, reverse


def _case(h, d, m, nocc, tol):
    hj, dj, mj = jnp.array(h), jnp.array(d), jnp.array(m)
    naive = lambda t: _loss(pj.naive_hard_window_projector(hj + t * dj, nocc), mj)
    safe = lambda t: _loss(pj.hard_window_projector(hj + t * dj, nocc, tol), mj)
    return _modes(naive), _modes(safe)


# --------------------------------------------------------------------------- A


def experiment_a(n=6, nocc=3, ndir=20, seeds=(0, 1, 2)):
    """Hard window, degenerate pair fully **enclosed** — the disputed case.

    ``docs/continue_here.md`` §3 reports the naive route agreeing with central FD to
    1.5e-9 here and concludes the hard window is safe.  That check used one direction
    (a single real diagonal entry) and reverse mode only.
    """
    evals = np.array([-2.0, 1.0, 1.0, 3.0, 4.0, 5.0])
    m = np.diag(np.arange(float(n))).astype(complex)
    occ = ref.hard_occupations(n, nocc)
    rows = []
    for seed in seeds:
        h = ref.hermitian_from_spectrum(evals, seed)
        w = np.linalg.eigvalsh(h)
        tol = 1e3 * ref.degeneracy_tolerance(np.linalg.norm(h, 2))
        directions = [("e00", np.diag([1.0] + [0.0] * (n - 1)).astype(complex))]
        directions += [("random", ref.random_hermitian_direction(n, 100 + k)) for k in range(ndir)]
        for label, d in directions:
            exact = ref.directional_derivative(h, d, m, occ)
            fd = [(_np_loss(h + s * d, nocc, m) - _np_loss(h - s * d, nocc, m)) / (2 * s)
                  for s in (1e-4, 1e-5, 1e-6)]
            (nf, nr), (sf, sr) = _case(h, d, m, nocc, tol)
            rows.append(dict(seed=seed, direction=label, split=float(w[2] - w[1]),
                             exact=exact, fd=fd, naive_fwd=nf, naive_rev=nr,
                             safe_fwd=sf, safe_rev=sr))
    return rows


def _np_loss(h, nocc, m):
    _, v = np.linalg.eigh(h)
    p = v[:, :nocc] @ v[:, :nocc].conj().T
    return float(np.real(np.trace(p @ m)))


# --------------------------------------------------------------------------- B

def reduced_pair_case(n, nocc, condition, seed, degenerate=True, boundary_gap=2.0):
    r"""Build a generalized problem whose **reduced** matrix carries the degeneracy.

    The trap this avoids: assembling ``H`` with a degenerate spectrum and *then*
    reducing destroys it, because the eigenvalues of ``H c = lambda S c`` are not the
    eigenvalues of ``H``.  So the spectrum is engineered in the reduced problem and
    pushed outward, ``H = L H_red L^H``; recomputing the reduction numerically then
    splits the pair at exactly the backward error study §8(b) is about,
    :math:`\epsilon\,\kappa(S)\,\|H\|`.

    Returns ``(h_red, spectrum, kappa_true, kappa_cholesky, measured_split)``.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.sort(rng.normal(scale=10.0, size=n))
    if degenerate:
        spectrum[nocc - 3] = spectrum[nocc - 4]
    spectrum[nocc] = spectrum[nocc - 1] + boundary_gap
    h_red0 = ref.hermitian_from_spectrum(spectrum, seed)
    s = ref.random_overlap(n, seed + 1, condition=condition)
    l = np.linalg.cholesky(s)
    h = l @ h_red0 @ l.conj().T
    h_red, l2 = ref.cholesky_reduce(h, s)
    w = np.linalg.eigvalsh(h_red)
    kappa_true = float(np.linalg.cond(s))
    return (h_red, w, kappa_true, ref.condition_from_cholesky(l2),
            float(w[nocc - 3] - w[nocc - 4]))




def experiment_b(n=1000, condition=1.0e3, seed=0, ndir=3):
    """The same at production-ish size, through a Cholesky-reduced overlap.

    Study §8(b): the degeneracy tolerance scales as :math:`\\epsilon\\,\\kappa(S)\\,\\|H\\|`
    for the generalized problem, and :math:`\\kappa(S)` for Elk's APW+lo overlap was
    never measured -- so it is a knob here, and the tolerance is derived from the
    Cholesky factor the solver already forms.

    Memory: at ``n=1000``, complex128, one matrix is 16 MB and this holds a handful.
    The production shape (``n=3000`` x 100 k-points, ~29 GB) is not runnable here at
    all -- see CLAUDE.md, "JAX port".
    """
    nocc = n // 2
    h_red, w, kappa_true, kappa_chol, split = reduced_pair_case(n, nocc, condition, seed)
    tol = 1e3 * ref.degeneracy_tolerance(np.linalg.norm(h_red, 2), kappa_chol)
    occ = ref.hard_occupations(n, nocc)
    m = np.diag(np.linspace(-1.0, 1.0, n)).astype(complex)
    rows = []
    for k in range(ndir):
        d = ref.random_hermitian_direction(n, 200 + k)
        exact = ref.directional_derivative(h_red, d, m, occ)
        (nf, nr), (sf, sr) = _case(h_red, d, m, nocc, tol)
        rows.append(dict(kappa_true=kappa_true, kappa_chol=kappa_chol, split=split,
                         tol=tol, exact=exact, naive_fwd=nf, naive_rev=nr,
                         safe_fwd=sf, safe_rev=sr))
    return rows


# --------------------------------------------------------------------------- C


def experiment_c(n=6, nocc=3, jitter=1e-16, nrepeat=6):
    """Reassembly jitter: perturb :math:`H` at 1e-16 and watch the gradient move.

    The true answer moves by ~1e-16; anything larger is the AD route amplifying
    rounding.  Study §8(b) reports 159% spread naive, 4.3e-15 with the rule;
    measured here, 64%.  The naive figure is a rounding accident and is not stable
    across builds -- only its order of magnitude means anything, which is why the
    test asserts a loose bound rather than the number.
    """
    evals = np.array([-2.0, 1.0, 1.0, 3.0, 4.0, 5.0])
    m = np.diag(np.arange(float(n))).astype(complex)
    occ = ref.hard_occupations(n, nocc)
    h0 = ref.hermitian_from_spectrum(evals, 0)
    d = ref.random_hermitian_direction(n, 1)
    tol = 1e3 * ref.degeneracy_tolerance(np.linalg.norm(h0, 2))
    exact, naive, safe = [], [], []
    for k in range(nrepeat):
        h = h0 + jitter * ref.random_hermitian_direction(n, 300 + k)
        exact.append(ref.directional_derivative(h, d, m, occ))
        (_, nr), (_, sr) = _case(h, d, m, nocc, tol)
        naive.append(nr)
        safe.append(sr)
    spread = lambda xs: (max(xs) - min(xs)) / max(abs(np.mean(xs)), 1e-30)
    return dict(exact=exact, naive=naive, safe=safe, spread_exact=spread(exact),
                spread_naive=spread(naive), spread_safe=spread(safe))


# --------------------------------------------------------------------------- D


def experiment_d(nphys=4, npad=4, e_big=1.0e3, nocc=2):
    """The padding block: :math:`H_{\\rm pad}=E_{\\rm big}\\mathbb 1`, bitwise degenerate.

    Study §8(b): "do not manufacture degeneracies in the padding".  §3.2's obvious
    padding choice makes an ``(nmatmax - nmat)``-fold **bitwise** degeneracy at every
    k-point, unconditionally, and a Phase 1 developer meets it on day one.
    """
    n = nphys + npad
    spectrum = np.concatenate([np.array([-3.0, -1.0, 0.5, 2.0])[:nphys],
                               np.full(npad, e_big)])
    h = np.zeros((n, n), dtype=complex)
    h[:nphys, :nphys] = ref.hermitian_from_spectrum(spectrum[:nphys], 7)
    h[nphys:, nphys:] = e_big * np.eye(npad)
    d = ref.random_hermitian_direction(n, 8)
    m = np.diag(np.arange(float(n))).astype(complex)
    occ = ref.hard_occupations(n, nocc)
    exact = ref.directional_derivative(h, d, m, occ)
    tol = 1e3 * ref.degeneracy_tolerance(np.linalg.norm(h, 2))
    (nf, nr), (sf, sr) = _case(h, d, m, nocc, tol)
    return dict(exact=exact, naive_fwd=nf, naive_rev=nr, safe_fwd=sf, safe_rev=sr)


# --------------------------------------------------------------------------- E


def experiment_e(n=6, mu=2.0, width=0.3, ndir=5):
    """Fermi-Dirac smearing — the metallic case, where §8(b) and the later check agree.

    Here :math:`f_i-f_j` is a small nonzero number rather than an exact zero, so the
    :math:`1/(\\lambda_i-\\lambda_j)` amplifies it instead of annihilating it.
    """
    evals = np.array([-2.0, 1.0, 1.0, 3.0, 4.0, 5.0])
    m = np.diag(np.arange(float(n))).astype(complex)
    rows = []
    for seed in (0, 1, 2):
        h = ref.hermitian_from_spectrum(evals, seed)
        w = np.linalg.eigvalsh(h)
        occ, docc = ref.fermi_dirac(w, mu, width)
        tol = 1e3 * ref.degeneracy_tolerance(np.linalg.norm(h, 2))
        hj, mj = jnp.array(h), jnp.array(m)
        for k in range(ndir):
            d = ref.random_hermitian_direction(n, 400 + k)
            exact = ref.directional_derivative(h, d, m, occ, docc, tol)
            dj = jnp.array(d)
            naive = lambda t: _loss(pj.naive_smeared_projector(hj + t * dj, mu, width), mj)
            safe = lambda t: _loss(pj.smeared_projector(hj + t * dj, mu, width, tol), mj)
            (nf, nr), (sf, sr) = _modes(naive), _modes(safe)
            rows.append(dict(seed=seed, exact=exact, naive_fwd=nf, naive_rev=nr,
                             safe_fwd=sf, safe_rev=sr))
    return rows


# --------------------------------------------------------------------------- F


def experiment_f(n=400, condition=1.0e6, seed=3, mu=0.0, width=0.5, decades=9):
    """Is the safe-:math:`K` gradient flat over two decades of ``tol``?

    Run **with smearing**, because for a hard window the tolerance turns out to be inert
    everywhere except at the window boundary: both branches of the kernel are then
    identically zero for a same-side pair (:math:`f_i-f_j=0` exactly, and :math:`f'=0`),
    so the sweep is flat for a trivial reason and settles nothing.  With Fermi-Dirac
    occupations :math:`f_i-f_j` is a genuine small number and the branch matters.

    Study §8(b) requires this at production :math:`n` with a real LAPW :math:`S`; this
    is the synthetic stand-in until Phase 1 supplies one.
    """
    nocc = n // 2
    h_red, w, kappa_true, kappa_chol, split = reduced_pair_case(n, nocc, condition, seed)
    base = ref.degeneracy_tolerance(np.linalg.norm(h_red, 2), kappa_chol)
    m = np.diag(np.linspace(-1.0, 1.0, n)).astype(complex)
    d = ref.random_hermitian_direction(n, 500)
    occ, docc = ref.fermi_dirac(w, mu, width)
    exact = ref.directional_derivative(h_red, d, m, occ, docc, base)
    hj, dj, mj = jnp.array(h_red), jnp.array(d), jnp.array(m)
    naive = float(jax.grad(lambda t: _loss(
        pj.naive_smeared_projector(hj + t * dj, mu, width), mj))(0.0))
    out = []
    for p in range(decades):
        tol = base * 10.0 ** (p - 2)
        fn = lambda t: _loss(pj.smeared_projector(hj + t * dj, mu, width, tol), mj)
        out.append((tol, float(jax.grad(fn)(0.0))))
    return dict(kappa_true=kappa_true, kappa_chol=kappa_chol, split=split,
                base_tol=base, exact=exact, naive=naive, sweep=out)


# --------------------------------------------------------------------------- G


def experiment_g(n=400, seed=11, nocc=200,
                 splits=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 0.0)):
    """How the naive route fails as the enclosed pair closes, hard window.

    Study §8(b) tabulates this for *individual eigenvalues*.  The disputed case is the
    projector with the multiplet **inside** the window, where the divergent terms are
    supposed to cancel; this is the same sweep for that quantity.  The achieved split is
    reported rather than the requested one, since assembling through a unitary cannot
    resolve below :math:`\epsilon\|H\|`.
    """
    rng = np.random.default_rng(seed)
    base = np.sort(rng.normal(scale=10.0, size=n))
    base[nocc] = base[nocc - 1] + 2.0
    m = np.diag(np.linspace(-1.0, 1.0, n)).astype(complex)
    occ = ref.hard_occupations(n, nocc)
    d = ref.random_hermitian_direction(n, 600)
    rows = []
    for requested in splits:
        spectrum = base.copy()
        spectrum[nocc - 3] = spectrum[nocc - 4] + requested
        h = ref.hermitian_from_spectrum(spectrum, seed)
        w = np.linalg.eigvalsh(h)
        achieved = float(w[nocc - 3] - w[nocc - 4])
        exact = ref.directional_derivative(h, d, m, occ)
        tol = 1e3 * ref.degeneracy_tolerance(np.linalg.norm(h, 2))
        (nf, nr), (sf, sr) = _case(h, d, m, nocc, tol)
        rows.append(dict(requested=requested, achieved=achieved, exact=exact,
                         naive_fwd=nf, naive_rev=nr, safe_rev=sr))
    return rows



# --------------------------------------------------------------------------- H


def experiment_h(sizes=(200, 400, 600, 800, 1000), condition=1.0e3, seed=0):
    """Which failure mode the naive route takes is decided by the **eigensolver**.

    The same matrix, an engineered pair that LAPACK resolves at ~1e-14, can come back
    from XLA's own ``eigh`` as a *bitwise* equal pair -- and JAX's eigenvector rule then
    forms ``1/(lambda_i - lambda_j) = inf`` and multiplies it by an exactly-zero
    numerator, giving ``NaN`` instead of a wrong number.  So "naive AD returns garbage"
    and "naive AD returns NaN" are the same bug wearing whichever face the backend's
    rounding picks, and neither is reproducible across LAPACK and XLA on one matrix.
    """
    rows = []
    for n in sizes:
        h_red, w_np, kappa_true, kappa_chol, split = reduced_pair_case(
            n, n // 2, condition, seed)
        w_jax = np.asarray(jnp.linalg.eigvalsh(jnp.array(h_red)))
        d = ref.random_hermitian_direction(n, 700)
        m = np.diag(np.linspace(-1.0, 1.0, n)).astype(complex)
        occ = ref.hard_occupations(n, n // 2)
        exact = ref.directional_derivative(h_red, d, m, occ)
        tol = 1e3 * ref.degeneracy_tolerance(np.linalg.norm(h_red, 2), kappa_chol)
        (_, nr), (_, sr) = _case(h_red, d, m, n // 2, tol)
        rows.append(dict(n=n, split_lapack=float(np.diff(w_np).min()),
                         split_xla=float(np.diff(w_jax).min()), exact=exact,
                         naive_rev=nr, safe_rev=sr))
    return rows


# --------------------------------------------------------------------------- main


def main():
    memory.limit_address_space(16.0)
    print("== A. hard window, degenerate pair fully enclosed (n=6) ==")
    print(f"{'seed':>4} {'direction':>9} {'exact':>14} {'FD(1e-5)':>14} "
          f"{'naive fwd':>12} {'naive rev':>12} {'safe fwd':>12} {'safe rev':>12}")
    worst = {"naive_fwd": 0.0, "naive_rev": 0.0, "safe_fwd": 0.0, "safe_rev": 0.0}
    fd_worst = 0.0
    for r in experiment_a():
        for key in worst:
            worst[key] = max(worst[key], _rel(r[key], r["exact"]))
        fd_worst = max(fd_worst, _rel(r["fd"][1], r["exact"]))
        if r["direction"] == "e00" or r["seed"] == 0:
            print(f"{r['seed']:>4} {r['direction']:>9} {r['exact']:>14.10f} {r['fd'][1]:>14.10f} "
                  f"{r['naive_fwd']:>12.6f} {r['naive_rev']:>12.6f} "
                  f"{r['safe_fwd']:>12.8f} {r['safe_rev']:>12.8f}")
    print(f"  worst relative error over 3 seeds x 21 directions:")
    print(f"    central FD  {fd_worst:.2e}   naive fwd {worst['naive_fwd']:.2e}   "
          f"naive rev {worst['naive_rev']:.2e}")
    print(f"    safe fwd    {worst['safe_fwd']:.2e}   safe rev  {worst['safe_rev']:.2e}")

    print("\n== B. same, n=1000 through a Cholesky-reduced overlap ==")
    rows = experiment_b()
    r0 = rows[0]
    print(f"  kappa(S) true {r0['kappa_true']:.3e}, Cholesky estimate {r0['kappa_chol']:.3e}; "
          f"engineered pair splits at {r0['split']:.3e}; tol {r0['tol']:.3e}")
    for r in rows:
        print(f"    exact={r['exact']: .10f}  "
              f"naive fwd {_rel(r['naive_fwd'], r['exact']):.2e}  "
              f"naive rev {_rel(r['naive_rev'], r['exact']):.2e}  "
              f"safe rev {_rel(r['safe_rev'], r['exact']):.2e}")

    print("\n== C. reassembly jitter at 1e-16 ==")
    c = experiment_c()
    print(f"  relative spread of the gradient: exact {c['spread_exact']:.2e}  "
          f"naive {c['spread_naive']:.2e}  safe {c['spread_safe']:.2e}")

    print("\n== D. padding block H_pad = E_big * I (bitwise degenerate) ==")
    d = experiment_d()
    print(f"  exact {d['exact']:.10f}  naive fwd {d['naive_fwd']}  naive rev {d['naive_rev']}  "
          f"safe rev {d['safe_rev']:.10f}")

    print("\n== E. Fermi-Dirac smearing (mu=2.0, width=0.3) ==")
    rows = experiment_e()
    nf = max(_rel(r["naive_fwd"], r["exact"]) for r in rows)
    nr = max(_rel(r["naive_rev"], r["exact"]) for r in rows)
    sr = max(_rel(r["safe_rev"], r["exact"]) for r in rows)
    print(f"  worst relative error: naive fwd {nf:.2e}  naive rev {nr:.2e}  safe rev {sr:.2e}")

    print("\n== F. tolerance plateau, Fermi-Dirac smearing (n=400, kappa(S)=1e6) ==")
    f = experiment_f()
    print(f"  kappa(S) true {f['kappa_true']:.3e}, Cholesky estimate {f['kappa_chol']:.3e}; "
          f"pair splits at {f['split']:.3e}")
    print(f"  eps*kappa*|H| = {f['base_tol']:.3e}  exact={f['exact']:.10f}  "
          f"naive rev rel {_rel(f['naive'], f['exact']):.2e}")
    for tol, value in f["sweep"]:
        print(f"    tol {tol:.3e}   grad {value:.12f}   rel {_rel(value, f['exact']):.2e}")
    print("\n== G. naive error vs the enclosed pair's splitting (n=400, hard window) ==")
    print(f"  {'requested':>10} {'achieved':>11} {'exact':>15} {'naive fwd':>11} "
          f"{'naive rev':>11} {'safe rev':>11}")
    for r in experiment_g():
        print(f"  {r['requested']:>10.0e} {r['achieved']:>11.3e} {r['exact']:>15.10f} "
              f"{_rel(r['naive_fwd'], r['exact']):>11.2e} {_rel(r['naive_rev'], r['exact']):>11.2e} "
              f"{_rel(r['safe_rev'], r['exact']):>11.2e}")

    print("\n== H. which failure mode appears is the eigensolver's choice ==")
    print(f"  {'n':>5} {'split LAPACK':>13} {'split XLA':>13} {'naive rev':>22} {'safe rev rel':>13}")
    for r in experiment_h():
        naive = ("nan" if np.isnan(r["naive_rev"])
                 else f"{_rel(r['naive_rev'], r['exact']):.2e} rel")
        print(f"  {r['n']:>5} {r['split_lapack']:>13.3e} {r['split_xla']:>13.3e} "
              f"{naive:>22} {_rel(r['safe_rev'], r['exact']):>13.2e}")

    print(f"\npeak RSS {memory.peak_rss_bytes() / memory.GB:.2f} GB")


if __name__ == "__main__":
    main()
