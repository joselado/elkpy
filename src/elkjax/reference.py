r"""Analytic references for the eigenproblem derivatives, in plain NumPy.

**Why this module exists.**  ``docs/continue_here.md`` §3 records an unresolved
disagreement about whether JAX's default VJP through ``eigh`` is safe for a hard
integer occupied window with a degenerate multiplet fully enclosed, and says what
would settle it: *a reference that is not finite differences*.  Study §8(b) measures
why FD cannot serve — central FD of the **sorted** spectrum returns the branch average
at a multiplet (measured there as ``(-0.498, -0.498)`` against true one-sided
``(-1.553, +0.557)``), so it cannot detect a wrong individual-eigenvalue gradient at
all, and near a degeneracy it is noise.

The reference used here is the closed form instead.  For any spectral function
:math:`P = \\sum_i f_i\\,|i\\rangle\\langle i|` of a Hermitian :math:`H`, the
Daleckii-Krein formula gives the directional derivative along a Hermitian
:math:`\\delta H` exactly:

.. math::

    dP = V\\,\\big(K \\circ (V^\\dagger\\,\\delta H\\,V)\\big)\\,V^\\dagger,
    \\qquad
    K_{ij} = \\frac{f_i - f_j}{\\lambda_i - \\lambda_j},
    \\qquad
    K_{ii} = f'(\\lambda_i),

with :math:`\\circ` the elementwise product.  For a hard window
(:math:`f_i = 1` inside, 0 outside) every same-side entry of :math:`K` vanishes
identically — including inside a degenerate multiplet, which is the whole content of
the claim under dispute — and the formula collapses to

.. math::

    dP = \\sum_{i \\in W,\; j \\notin W}
    \\frac{|i\\rangle\\langle i|\\,\\delta H\\,|j\\rangle\\langle j| + \\mathrm{h.c.}}
         {\\lambda_i - \\lambda_j},

which is gauge-invariant (it depends on :math:`H` only through the two spectral
projectors), exact, and valid at any matrix size.  It is undefined, as a matter of
mathematics rather than numerics, when the window boundary itself is degenerate: there
is then no differentiable "occupied subspace" to speak of.  :func:`window_gap` is what
detects that, and the callers here refuse rather than return a number.

Everything is expressed as a **directional** derivative :math:`d/dt` along a Hermitian
direction at real :math:`t`.  That sidesteps the Wirtinger convention question entirely
— ``jax.grad`` of a real loss over a complex array returns a conjugated cotangent whose
factor conventions are easy to get subtly wrong, and a directional derivative of a real
scalar with respect to a real parameter has no convention to get wrong.
"""

import numpy as np


def hermitian_from_spectrum(evals, seed, dtype=np.complex128):
    """Assemble ``U diag(evals) U^H`` with a Haar-ish random unitary from ``seed``.

    This is the *realistic* route, and the distinction matters: a diagonal test matrix
    carrying a degeneracy is degenerate **bitwise**, while the same spectrum assembled
    through a unitary splits at ~1e-15 (measured, ``docs/continue_here.md`` §3).  Naive
    AD returns ``NaN`` on the first and a finite gradient of order 1e14 whose sign
    flips between assemblies on the second.  Unit tests reach for the first shape; real
    Hamiltonians are the second.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(len(evals), len(evals)))
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * rng.normal(size=(len(evals), len(evals)))
    q, r = np.linalg.qr(a)
    q = q * (np.diag(r) / np.abs(np.diag(r)))  # fix the QR sign convention
    return ((q * np.asarray(evals)) @ q.conj().T).astype(dtype)


def random_hermitian_direction(n, seed, dtype=np.complex128):
    """A Hermitian perturbation direction, normalised to unit Frobenius norm."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * rng.normal(size=(n, n))
    a = a + a.conj().T
    return (a / np.linalg.norm(a)).astype(dtype)


def random_overlap(n, seed, condition=1.0e3, dtype=np.complex128):
    """A Hermitian positive-definite overlap matrix with a prescribed 2-norm condition.

    Study §8(b) makes the degeneracy tolerance scale as :math:`\\epsilon\\,\\kappa(S)\\,
    \\|H\\|` for the **generalized** problem the Cholesky reduction of B4 solves, and
    notes that :math:`\\kappa(S)` for Elk's APW+lo overlap was never measured.  Until it
    is, the honest thing is to make it a knob and report the tolerance as a function of
    it, which is what this supports.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * rng.normal(size=(n, n))
    q, _ = np.linalg.qr(a)
    spectrum = np.geomspace(1.0 / condition, 1.0, n)
    return ((q * spectrum) @ q.conj().T).astype(dtype)


def cholesky_reduce(h, s):
    """Reduce ``H c = lambda S c`` to a standard problem, Elk's own route (B4).

    Returns ``(h_reduced, l)`` with ``s = l l^H`` and
    ``h_reduced = l^-1 H l^-H``; eigenvectors map back as ``c = l^-H y``.
    """
    l = np.linalg.cholesky(s)
    y = np.linalg.solve(l, h)
    return np.linalg.solve(l, y.conj().T).conj().T, l


def condition_from_cholesky(l):
    """Cheap :math:`\\kappa(S)` estimate from the Cholesky factor the solver already forms.

    Study §8(b) suggests ``min diag(L)^-2``; that is the estimate for ``||S|| ~ 1``, and
    the scale-free version used here is ``(max diag(L) / min diag(L))^2``.
    """
    d = np.abs(np.diag(l))
    return float((d.max() / d.min()) ** 2)


def degeneracy_tolerance(h_norm, condition=1.0, eps=None):
    """Eigenvalue backward error :math:`\\epsilon\\,\\kappa(S)\\,\\|H\\|`.

    This is the scale below which two computed eigenvalues cannot be distinguished, and
    therefore the only defensible threshold for "degenerate" in a divided-difference
    kernel.  Study §8(b): the naive figure without :math:`\\kappa(S)` is a *lower bound*,
    and setting the tolerance there would treat genuinely-degenerate pairs as split,
    reintroducing the hazard the rule exists to prevent.
    """
    if eps is None:
        eps = np.finfo(np.float64).eps
    return eps * condition * h_norm


def window_gap(evals, nocc):
    """The gap across the occupied-window boundary, ``evals[nocc] - evals[nocc-1]``.

    ``inf`` when the window is everything or nothing.  The projector derivative exists
    only when this is nonzero, and is contaminated at the level of
    :func:`degeneracy_tolerance` divided by it.
    """
    evals = np.asarray(evals)
    if nocc <= 0 or nocc >= len(evals):
        return np.inf
    return float(evals[nocc] - evals[nocc - 1])


def divided_difference_kernel(evals, occ, docc=None, tol=0.0):
    """The Daleckii-Krein kernel :math:`K_{ij} = (f_i - f_j)/(\\lambda_i - \\lambda_j)`.

    ``occ`` are the occupation numbers :math:`f_i`; ``docc`` the analytic
    :math:`f'(\\lambda_i)`, used on any pair closer than ``tol`` (and on the diagonal).
    For a hard window ``docc`` is zero, which is correct: the occupation is locally
    constant in the eigenvalue.
    """
    evals = np.asarray(evals, dtype=float)
    occ = np.asarray(occ, dtype=float)
    docc = np.zeros_like(evals) if docc is None else np.asarray(docc, dtype=float)
    dl = evals[:, None] - evals[None, :]
    df = occ[:, None] - occ[None, :]
    near = np.abs(dl) <= tol
    safe = np.where(near, 1.0, dl)
    fp = 0.5 * (docc[:, None] + docc[None, :])
    return np.where(near, fp, df / safe)


def dprojector(h, dh, occ, docc=None, tol=0.0, s=None):
    """:math:`dP` along ``dh``, from the closed form above.  ``h``/``dh`` Hermitian.

    With ``s`` given the generalized problem is solved by Cholesky reduction and the
    returned derivative is that of the *reduced* projector — an honest orthogonal
    projector, which is what the AD comparison needs; ``dh`` is then also understood in
    the reduced basis.
    """
    if s is not None:
        h, _ = cholesky_reduce(h, s)
    evals, evecs = np.linalg.eigh(h)
    kernel = divided_difference_kernel(evals, occ, docc, tol)
    a = evecs.conj().T @ dh @ evecs
    return evecs @ (kernel * a) @ evecs.conj().T


def hard_occupations(n, nocc):
    """Integer occupations: 1 on the first ``nocc`` states, 0 above."""
    occ = np.zeros(n)
    occ[:nocc] = 1.0
    return occ


def fermi_dirac(evals, mu, width):
    """Occupations and their analytic derivative for Fermi-Dirac smearing.

    ``f = 1/(1 + exp((e-mu)/w))`` and ``f' = -f(1-f)/w``.  This is Elk's ``stype=3``;
    the point of carrying ``f'`` explicitly is that it is the correct value of the
    divided difference *at* a degeneracy, where the difference quotient is 0/0.
    """
    x = (np.asarray(evals, dtype=float) - mu) / width
    f = 1.0 / (1.0 + np.exp(np.clip(x, -600.0, 600.0)))
    return f, -f * (1.0 - f) / width


def directional_derivative(h, dh, observable, occ, docc=None, tol=0.0, s=None):
    """:math:`\\frac{d}{dt}\\,\\mathrm{Re}\\,\\mathrm{Tr}[P(H+t\\,\\delta H)\\,M]` at ``t=0``.

    The scalar loss used throughout the Phase 0b tests, with ``observable`` playing
    :math:`M`.  Real by construction for Hermitian ``M``, and — being built from the
    projector rather than from individual eigenvectors — invariant under the arbitrary
    unitary ``eigh`` picks inside any degenerate multiplet (study §8(b), hazard C).
    """
    dp = dprojector(h, dh, occ, docc, tol, s)
    return float(np.real(np.trace(dp @ observable)))


def multiplet_sum_derivative(h, dh, occ, s=None):
    """:math:`\\frac{d}{dt}\\sum_i f_i\\,\\lambda_i`, which is :math:`\\mathrm{Tr}[P\\,\\delta H]`.

    Study §8(b)'s "exact and free" row: no :math:`1/(\\lambda_i-\\lambda_j)` appears
    anywhere, so this is the one eigenvalue quantity that is safe *at* a degeneracy.
    Kept here as the control that separates "AD is broken" from "the test is broken".
    """
    if s is not None:
        h, _ = cholesky_reduce(h, s)
    evals, evecs = np.linalg.eigh(h)
    p = (evecs * np.asarray(occ, dtype=float)) @ evecs.conj().T
    return float(np.real(np.trace(p @ dh)))
