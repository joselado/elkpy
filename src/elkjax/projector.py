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

import jax
import jax.numpy as jnp

__all__ = [
    "divided_difference_kernel",
    "hard_window_projector",
    "naive_hard_window_projector",
    "smeared_projector",
    "naive_smeared_projector",
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


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def smeared_projector(h, mu, width, tol=0.0):
    r""":math:`P=\sum_i f(\lambda_i)|i\rangle\langle i|` with Fermi-Dirac occupations.

    ``mu`` is held fixed.  The self-consistent Fermi level is a separate custom rule —
    study §8(b) gives it in closed form, :math:`d\mu/d\varepsilon_i = w_if'_i/\sum_j
    w_jf'_j` — and belongs to Phase 0a, where the electron-number constraint enters.
    """
    evals, evecs = jnp.linalg.eigh(h)
    occ, _ = fermi_dirac(evals, mu, width)
    return _projector(evecs, occ)


@smeared_projector.defjvp
def _smeared_projector_jvp(mu, width, tol, primals, tangents):
    (h,), (dh,) = primals, tangents
    evals, evecs = jnp.linalg.eigh(h)
    occ, docc = fermi_dirac(evals, mu, width)
    kernel = divided_difference_kernel(evals, occ, docc, tol)
    a = evecs.conj().T @ dh @ evecs
    return _projector(evecs, occ), evecs @ (kernel * a) @ evecs.conj().T


def naive_smeared_projector(h, mu, width):
    """The plain transcription with smearing, for the same comparison."""
    evals, evecs = jnp.linalg.eigh(h)
    occ, _ = fermi_dirac(evals, mu, width)
    return _projector(evecs, occ)
