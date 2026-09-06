r"""The safe-:math:`K` spectral projector rule — study §8(b), Phase 0b.

The occupied density matrix :math:`P=\sum_i f_i\,|i\rangle\langle i|` is the object the
whole SCF fixed point is built out of: study §8(a) notes that the independent-particle
susceptibility :math:`\chi_0` *is* its derivative, so without a working rule here the
implicit Jacobian :math:`(\mathbb 1-K_{\rm Hxc}\chi_0)` is not merely inaccurate, it is
undefined at every multiplet.

**The failure it fixes.**  JAX's default VJP for ``eigh`` differentiates the
eigenvectors, and that rule divides by :math:`\lambda_i-\lambda_j` *before* the
:math:`f_i-f_j` numerator supplied by the occupations can cancel against it.  At a
degeneracy the division is :math:`0/0`; at a roundoff-split degeneracy — which is what a
real Hamiltonian has, since assembling a spectrum through a unitary splits an exact pair
at ~1e-15 — it is a ratio of two rounding errors, finite and meaningless.

**The fix** is to form the whole divided difference as *one* expression,

.. math::

    K_{ij} = \frac{f_i - f_j}{\lambda_i - \lambda_j}
    \quad\text{with}\quad
    K_{ij} \to f'(\bar\lambda) \ \text{ when } |\lambda_i-\lambda_j| \le \texttt{tol},
    \qquad
    dP = V\,(K \circ V^\dagger\,\delta H\,V)\,V^\dagger ,

and to install it as a ``custom_jvp`` so JAX never sees the eigenvector derivative at
all.  The two ``jnp.where`` calls are not redundant: the *denominator* is made safe
before the division, because ``jnp.where`` evaluates both branches and a ``NaN`` in the
discarded one still propagates through the gradient.

**Two refusals rather than two silently wrong numbers.**

* For a hard integer window, a pair straddling the window boundary that is closer than
  ``tol`` has no differentiable occupied subspace at all — the derivative does not exist,
  as mathematics and not as numerics.  This returns ``NaN`` there deliberately.  The
  project's standing mitigation is the one CLAUDE.md §13 already applies to Berry
  curvature: window the whole degenerate group together.
* ``tol`` must come from the eigenvalue backward error :math:`\epsilon\,\kappa(S)\,\|H\|`
  for the *generalized* problem, not :math:`\epsilon\|H\|`
  (:func:`elkjax.reference.degeneracy_tolerance`).  Too small and genuinely degenerate
  pairs are treated as split, which is hazard A reintroduced exactly where the rule was
  installed to prevent it.

**Known limitation, and it is Phase 0a′'s problem.**  This rule is first-order only: the
JVP body itself calls ``jnp.linalg.eigh``, so differentiating the rule a second time
(``jax.hessian``) falls back on JAX's default eigenvector rule and the hazard returns.
Second-order work needs the rule made recursive, or forward-over-forward.
"""

import functools

import numpy as np

import jax
import jax.numpy as jnp

__all__ = [
    "sign_projector",
    "check_sign_window",
    "divided_difference_kernel",
    "hard_window_projector",
    "naive_hard_window_projector",
    "smeared_projector",
    "naive_smeared_projector",
    "direct_quotient_projector",
    "fermi_level",
    "fixed_number_projector",
    "check_fermi_level_determined",
    "fermi_dirac",
    "window_gap",
]


def divided_difference_kernel(evals, occ, docc, tol):
    """:math:`K_{ij}`, with the near-degenerate block replaced by :math:`f'`."""
    dl = evals[:, None] - evals[None, :]
    df = occ[:, None] - occ[None, :]
    near = jnp.abs(dl) <= tol
    safe = jnp.where(near, jnp.ones_like(dl), dl)
    fp = 0.5 * (docc[:, None] + docc[None, :])
    return jnp.where(near, fp, df / safe)


def window_gap(evals, nocc):
    """``evals[nocc] - evals[nocc-1]``: the gap the hard-window projector needs."""
    return evals[nocc] - evals[nocc - 1]


def fermi_dirac(evals, mu, width):
    """Occupations and :math:`f'` for ``stype=3`` smearing, as ``(f, df/de)``."""
    f = jax.nn.sigmoid(-(evals - mu) / width)
    return f, -f * (1.0 - f) / width


def _projector(evecs, occ):
    return (evecs * occ) @ evecs.conj().T


# ---------------------------------------------------------------- hard window


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def hard_window_projector(h, nocc, tol=0.0):
    r""":math:`P=\sum_{i<n_{\rm occ}}|i\rangle\langle i|`, with the safe-:math:`K` JVP."""
    _, evecs = jnp.linalg.eigh(h)
    occupied = evecs[:, :nocc]
    return occupied @ occupied.conj().T


@hard_window_projector.defjvp
def _hard_window_projector_jvp(nocc, tol, primals, tangents):
    (h,), (dh,) = primals, tangents
    evals, evecs = jnp.linalg.eigh(h)
    occ = (jnp.arange(evals.shape[0]) < nocc).astype(evals.dtype)
    # docc = 0: a hard window's occupation is locally constant in the eigenvalue, so a
    # near-degenerate pair on the SAME side of the boundary contributes nothing.  A pair
    # straddling it is a different matter and must not be papered over -- hence the NaN.
    kernel = divided_difference_kernel(evals, occ, jnp.zeros_like(evals), tol)
    dl = evals[:, None] - evals[None, :]
    straddles = occ[:, None] != occ[None, :]
    kernel = jnp.where(straddles & (jnp.abs(dl) <= tol), jnp.nan, kernel)
    a = evecs.conj().T @ dh @ evecs
    return _projector(evecs, occ), evecs @ (kernel * a) @ evecs.conj().T


def naive_hard_window_projector(h, nocc):
    """The plain transcription, differentiated by JAX's own ``eigh`` rule.

    Kept as the thing under test, not as an implementation: this is what a Phase 1
    developer writes on day one, and what study §8(b) claims returns garbage.
    """
    _, evecs = jnp.linalg.eigh(h)
    occupied = evecs[:, :nocc]
    return occupied @ occupied.conj().T


# ------------------------------------------------------------ smeared occupations


@functools.partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def smeared_projector(h, mu, width, tol=0.0):
    r""":math:`P=\sum_i f(\lambda_i-\mu)|i\rangle\langle i|`, Fermi-Dirac occupations.

    ``mu`` is a **differentiable** primal, not a constant.  That is what lets the
    fixed-electron-number Fermi level of :func:`fermi_level` be composed here and get
    the right chain rule for free: study §8(b)'s
    :math:`d\mu/d\varepsilon_i = w_if'_i/\sum_j w_jf'_j` is that function's own JVP,
    and :func:`fixed_number_projector` is nothing but the composition of the two.

    The :math:`\mu` tangent enters as :math:`-V\,\mathrm{diag}(f')\,V^\dagger\,d\mu`,
    since :math:`\partial f(\lambda-\mu)/\partial\mu = -f'(\lambda)`.
    """
    evals, evecs = jnp.linalg.eigh(h)
    occ, _ = fermi_dirac(evals, mu, width)
    return _projector(evecs, occ)


@smeared_projector.defjvp
def _smeared_projector_jvp(width, tol, primals, tangents):
    (h, mu), (dh, dmu) = primals, tangents
    evals, evecs = jnp.linalg.eigh(h)
    occ, docc = fermi_dirac(evals, mu, width)
    kernel = divided_difference_kernel(evals, occ, docc, tol)
    a = evecs.conj().T @ dh @ evecs
    dp = evecs @ (kernel * a) @ evecs.conj().T
    return _projector(evecs, occ), dp - _projector(evecs, docc) * dmu


def naive_smeared_projector(h, mu, width):
    """The plain transcription with smearing, for the same comparison."""
    evals, evecs = jnp.linalg.eigh(h)
    occ, _ = fermi_dirac(evals, mu, width)
    return _projector(evecs, occ)


def direct_quotient_projector(h, mu, width):
    r"""The safe-:math:`K` rule with the near-degenerate branch **switched off**.

    ``smeared_projector(h, mu, width, tol=0.0)``, given a name because it is the
    third route Phase 1i needs and it is not the same thing as
    :func:`naive_smeared_projector`.  Both avoid the eigenvector derivative; this one
    still forms :math:`(f_i-f_j)/(\lambda_i-\lambda_j)` literally, so it is exposed to
    the *numerator's* cancellation but not to the eigenvector rule's :math:`1/\delta\lambda`
    amplification.  Separating the two is what shows which of them a real LAPW multiplet
    actually trips (`docs/jax_port_phase1.md` §1i).
    """
    return smeared_projector(h, mu, width, 0.0)


# ------------------------------------------------ the self-consistent Fermi level


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def fermi_level(h, nelec, width, steps=100):
    r""":math:`\mu` fixed by the electron count, :math:`\sum_i f(\lambda_i-\mu)=N`.

    Study §8(b) gives the derivative in closed form and
    ``docs/continue_here.md`` §3 records it as untested:

    .. math::

        \frac{d\mu}{d\varepsilon_i} = \frac{w_i f'_i}{\sum_j w_j f'_j}
        \qquad\Longrightarrow\qquad
        d\mu = \frac{\sum_j f'_j\,(V^\dagger\,\delta H\,V)_{jj}}{\sum_j f'_j},

    which follows from differentiating the constraint: :math:`dN=\sum_j f'_j(d\lambda_j
    - d\mu)=0`.  The weights :math:`w_j` are the k-point weights of a full Brillouin-zone
    sum; at the single k-point of Phase 1 they are equal and cancel, so what is tested
    here is the *shape* of the rule and not the zone integration.

    **It is gauge-invariant at a multiplet even though it uses** :math:`A_{jj}`.  Inside a
    degenerate group :math:`f'_j` is constant, so :math:`\sum_j f'_j A_{jj}` over that
    group is :math:`f'\,\mathrm{Tr}_{\rm group}A` — a trace, invariant under the arbitrary
    unitary ``eigh`` picks there.  That is the same reason study §8(b) lists
    :math:`\sum_i f_i\lambda_i` as the one eigenvalue quantity safe *at* a degeneracy, and
    it is why no :math:`1/(\lambda_i-\lambda_j)` appears in this rule at all.

    **The denominator is a physical singularity, not a numerical one.**  :math:`\sum_j
    f'_j\to0` for a gapped system at small ``width``: no state responds, the constraint
    stops determining :math:`\mu`, and :math:`d\mu` is genuinely :math:`0/0`.
    :func:`check_fermi_level_determined` is the host-side refusal.

    The primal is a bisection, which is never differentiated -- that is the whole point of
    a ``custom_jvp`` here, since unrolling a root find is the Phase 0a lesson about
    unrolled solvers repeated in miniature.
    """
    evals = jnp.linalg.eigvalsh(h)
    lo = evals[0] - 50.0 * width
    hi = evals[-1] + 50.0 * width

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        count = jnp.sum(fermi_dirac(evals, mid, width)[0])
        return jnp.where(count < nelec, mid, lo), jnp.where(count < nelec, hi, mid)

    lo, hi = jax.lax.fori_loop(0, steps, body, (lo, hi))
    return 0.5 * (lo + hi)


@fermi_level.defjvp
def _fermi_level_jvp(nelec, width, steps, primals, tangents):
    (h,), (dh,) = primals, tangents
    mu = fermi_level(h, nelec, width, steps)
    evals, evecs = jnp.linalg.eigh(h)
    _, docc = fermi_dirac(evals, mu, width)
    a = jnp.real(jnp.einsum("ji,jk,ki->i", evecs.conj(), dh, evecs))
    return mu, jnp.sum(docc * a) / jnp.sum(docc)


def fixed_number_projector(h, nelec, width, tol=0.0, steps=100):
    r""":math:`P` at **fixed electron number** rather than fixed :math:`\mu`.

    Nothing but :func:`smeared_projector` composed with :func:`fermi_level`; the chain
    rule then supplies the extra term the fixed-:math:`\mu` derivative is missing,

    .. math::

        dP\big|_N = dP\big|_\mu - V\,\mathrm{diag}(f')\,V^\dagger\,d\mu ,

    which is not a correction of the same order as the rest -- at a half-filled level it
    is comparable to the whole fixed-:math:`\mu` derivative.
    """
    return smeared_projector(h, fermi_level(h, nelec, width, steps), width, tol)


def check_fermi_level_determined(h, nelec, width, steps=100, tol=1e-8):
    r"""Refuse a Fermi level the electron count does not actually determine.

    Returns :math:`(\mu, \sum_j |f'_j|)`; raises ``ValueError`` when the sum is below
    ``tol``, i.e. when every state is more than a few ``width`` from :math:`\mu` and
    :math:`d\mu` is :math:`0/0`.  This is the gapped-insulator case, where the Fermi
    level may be placed anywhere in the gap and its derivative is not a number --
    the same host-side shape as :func:`check_sign_window` and
    ``elkjax.hamiltonian.occupied_window``.
    """
    mu = float(fermi_level(h, nelec, width, steps))
    evals = np.asarray(jnp.linalg.eigvalsh(h))
    x = np.clip((evals - mu) / width, -600.0, 600.0)
    f = 1.0 / (1.0 + np.exp(x))
    response = float(np.sum(f * (1.0 - f)) / width)
    if response < tol:
        raise ValueError(
            f"sum |f'| = {response:.3e} is below {tol:.3e}: no state lies within a few "
            f"smearing widths of mu = {mu:.6f}, so the electron-number constraint does "
            f"not determine the Fermi level and dmu/dtheta is 0/0. This is a gapped "
            f"system at this width; use a fixed mu, or a hard window.")
    return mu, response


# ------------------------------------------------------- the eigensolver-free route


def sign_projector(h, nocc, *, steps=30, mu=None):
    r"""The occupied projector as a **matrix sign function**, built from matmuls only.

    For a hard window with a gapped boundary,

    .. math::

        P = \tfrac12\big(\mathbb 1 - \mathrm{sign}(H-\mu\mathbb 1)\big),

    with :math:`\mu` anywhere in the gap, and the sign is obtained by Newton-Schulz
    iteration :math:`X\leftarrow\tfrac12(3X-X^3)` from
    :math:`X_0=(H-\mu)/\|H-\mu\|_2`.

    **Why this exists: it is differentiable to any order.**  The safe-:math:`K` rule of
    :func:`hard_window_projector` is first-order only, because its own JVP body calls
    ``eigh`` — so a second derivative falls back on JAX's default eigenvector rule and
    returns ``NaN`` at a multiplet (measured, ``docs/jax_port_phase0.md`` §0a′).  This
    route contains no eigendecomposition at all, so there is no gauge to be arbitrary
    and no :math:`1/(\lambda_i-\lambda_j)` anywhere; JAX differentiates a chain of
    matrix products natively, however many times it is asked.

    Two shortcuts here are **exact, not approximations**.  :math:`P` is locally constant
    in :math:`\mu` while the gap stays open, so :math:`dP/d\mu=0` and taking :math:`\mu`
    from a ``stop_gradient``-ed spectrum loses nothing — which also keeps the ill-posed
    derivative of an individual eigenvalue out of the graph entirely.  Likewise
    :math:`\mathrm{sign}(A/s)=\mathrm{sign}(A)` for :math:`s>0`, so the normalisation is
    ``stop_gradient``-ed too.

    Cost and limits.  Convergence needs roughly
    :math:`\log(\|H-\mu\|_2/\Delta)/\log(3/2)` iterations for a gap :math:`\Delta`,
    and for an all-electron Hamiltonian **the numerator is set by the deepest core
    level, not by the valence bandwidth** -- with core states 1000+ Ha below
    :math:`\mu` and a 1 eV gap that ratio is :math:`\sim3\times10^4`, so ~25-30
    steps.  Keeping the core out of the second-variational block, as Elk does, cuts
    it to ~10.  The iteration is unrolled, which is exactly the tape the study warns
    about at production shapes.

    Two limits.  **Hard windows only** -- smeared occupations would need a Chebyshev
    expansion of the Fermi function instead.  And if :math:`\mu` lands *outside* the
    gap, because ``evals[nocc-1:nocc+1]`` straddles a multiplet rather than the two
    sides of a boundary, this returns a wrong projector **silently** -- nothing in
    the Newton-Schulz iteration notices.  :func:`check_sign_window` is the host-side
    guard, in the spirit of ``elkpy.parsers.symmetry.check_window_gap``.
    """
    n = h.shape[-1]
    eye = jnp.eye(n, dtype=h.dtype)
    frozen = jax.lax.stop_gradient(h)
    if mu is None:
        evals = jnp.linalg.eigvalsh(frozen)
        mu = 0.5 * (evals[nocc - 1] + evals[nocc])
    shifted = h - mu * eye
    scale = jnp.linalg.norm(jax.lax.stop_gradient(shifted), 2)
    x = shifted / scale
    for _ in range(steps):
        x = 1.5 * x - 0.5 * (x @ x @ x)
    return 0.5 * (eye - x)


def check_sign_window(h, nocc, tol=None):
    r"""Refuse a band window whose boundary is not gapped, before it fails silently.

    :func:`sign_projector` places :math:`\mu` midway between ``evals[nocc-1]`` and
    ``evals[nocc]``; if those two are a degenerate pair rather than the two sides of a
    gap, :math:`\mu` is inside a multiplet and the returned operator is not the
    projector onto anything.  Returns the gap; raises ``ValueError`` when it is below
    ``tol`` (default: :math:`10^3\,\epsilon\|H\|`).
    """
    evals = np.asarray(jnp.linalg.eigvalsh(h))
    if nocc <= 0 or nocc >= evals.size:
        raise ValueError(f"nocc={nocc} is not a proper window of {evals.size} states")
    gap = float(evals[nocc] - evals[nocc - 1])
    if tol is None:
        tol = 1e3 * np.finfo(np.float64).eps * float(np.abs(evals).max())
    if gap <= tol:
        raise ValueError(
            f"band window boundary gap {gap:.3e} is below {tol:.3e}: mu would sit "
            f"inside a multiplet and sign_projector would return a wrong operator "
            f"without failing. Window the whole degenerate group instead.")
    return gap
