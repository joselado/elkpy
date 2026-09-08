r"""The density's derivative through a degenerate spectrum — the safe-:math:`K`
rule of :mod:`elkjax.projector`, carried from a model projector to Elk's own
generalized eigenproblem and to the density built on the grid.

**What breaks without it.**  The valence density is
:math:`\rho(\mathbf r)=\sum_a f_a|\psi_a(\mathbf r)|^2`, and JAX's own rule for
``eigh`` differentiates each :math:`\psi_a` separately:

.. math::

    d\psi_a = \sum_{b\neq a}\psi_b\,
    \frac{\langle b|dH-\varepsilon_a\,dO|a\rangle}
         {\varepsilon_a-\varepsilon_b} - \tfrac12\psi_a\langle a|dO|a\rangle .

Inside a degenerate multiplet that denominator is a rounding error.  The
density's derivative is finite there — the pair :math:`(a,b)` and the pair
:math:`(b,a)` combine into a *divided difference* whose numerator vanishes with
its denominator — but only if the two are formed as one expression.  Formed
separately, each is :math:`10^{12}` and their sum is :math:`O(1)`: the answer
survives with the relative accuracy of the cancellation, not of the arithmetic.
Bulk silicon has a three-fold :math:`\Gamma_{25'}` valence top split by
roundoff, so this is not a corner case here, it is the first k-point.

**The combination, written once.**  Pairing the two terms and using
:math:`dH,dO` Hermitian,

.. math::

    d\rho(\mathbf r)=\sum_{ab}G_{ba}\,\overline{\psi_a(\mathbf r)}\,
    \psi_b(\mathbf r) + d\mu\sum_a\frac{\partial f_a}{\partial\mu}
    |\psi_a(\mathbf r)|^2 ,
    \qquad
    G = K^H\!\circ(C^\dagger dH\,C) - K^S\!\circ(C^\dagger dO\,C),

with the two divided differences

.. math::

    K^H_{ab}=\frac{f_a-f_b}{\varepsilon_a-\varepsilon_b},\qquad
    K^S_{ab}=\frac{f_a\varepsilon_a-f_b\varepsilon_b}{\varepsilon_a-\varepsilon_b}
    =\frac{\varepsilon_a+\varepsilon_b}{2}K^H_{ab}+\frac{f_a+f_b}{2} .

:math:`K^H` is :func:`elkjax.projector.fermi_kernel` — the Fermi-Dirac quotient
in closed form, with **no subtraction in it**, exact at every splitting — and
the second identity carries that same freedom from cancellation into
:math:`K^S`.  The diagonal is not a special case: :math:`K^H_{aa}=f'` and
:math:`K^S_{aa}=f'\varepsilon_a+f_a` are the limits the same expressions take.

**Why this module returns two factors rather than a density.**  Written as

.. math::

    \rho(\mathbf r)=\sum_{a<n}\mathrm{Re}\big[
    \overline{(W\bar c_a)(\mathbf r)}\,(Wx_a)(\mathbf r)\big],
    \qquad \bar c_a = c_a,\quad x_a = f_a c_a ,

the density is *bilinear* in two sets of coefficients, and :math:`W` — the map
from a coefficient vector to a wavefunction on Elk's grids — is linear and
depends on the potential only through the radial functions.  Giving
:math:`(\bar C, X)` the tangent :math:`(0,\,dX)` with

.. math::

    dX = \bar C\,G_{\rm oo} + 2\,C_{\rm e}\,G_{\rm eo}
       + \bar C\,\mathrm{diag}(\partial_\mu f\;d\mu)

then reproduces the whole of :math:`d\rho` with **no LAPW machinery inside the
rule**: the :math:`dW` half comes out of ordinary automatic differentiation of
:math:`W`, and the :math:`n` extra wavefunctions the rule needs are JAX applying
that same linear :math:`W` to :math:`dX`.  The factor 2 on the empty block is
:math:`G`'s hermiticity: the ``eo`` and ``oe`` halves of the sum are conjugates
and only their real part survives.  The identity that pins both it and the
:math:`\mathrm{Re}` is

.. math::

    \tfrac12\big(dX\,\bar C^\dagger + \bar C\,dX^\dagger\big) = dP ,

the tangent of the density matrix, which :func:`elkjax.projector.smeared_projector`
computes independently for :math:`O=\mathbb 1`.

**The subscripts o and e are Elk's truncation, not a physical occupation.**
``eveqnfv`` keeps the lowest ``nstfv`` states and the density sums over those,
so :math:`f_b\equiv0` for :math:`b\ge n` as a matter of what the code computes;
the sum over :math:`b` in :math:`G` nonetheless runs over the whole
:math:`n_{\rm mat}`-dimensional basis, because the *response* of an occupied
state reaches every basis function.  Dropping that tail would be a Sternheimer
equation solved in the occupied subspace only, and wrong at the percent level.

**Fermi-Dirac only.**  ``stype`` 0-2 (Methfessel-Paxton) have no closed form for
:math:`K^H` and would need the ``tol`` cliff Phase 1i measured and removed, so
they are refused rather than approximated.  Elk's default is ``stype=3``.
"""

import functools

import numpy as np

import jax
import jax.numpy as jnp

from . import occupations as _occ
from .projector import fermi_kernel

__all__ = ["eigenvalues", "occupation_weights", "pair_kernels",
           "density_factors", "density_matrix_tangent"]


def _solve(h, o):
    r"""The generalized eigenproblem, all :math:`n_{\rm mat}` states.

    Elk's own route (``zhegv``): :math:`O=LL^\dagger`,
    :math:`\tilde H=L^{-1}HL^{-\dagger}`, :math:`c=L^{-\dagger}y`, which leaves
    the eigenvectors in Elk's normalisation :math:`c^\dagger Oc=\mathbb 1`.
    Unlike `elkjax.density.solve_zone` this keeps every column: the empty
    states are what the response of the occupied ones is expanded in.
    """
    from .hamiltonian import cholesky_reduce

    reduced, chol = cholesky_reduce(h, o)
    values, y = jnp.linalg.eigh(reduced)
    return values, jnp.linalg.solve(chol.conj().T, y)


@jax.custom_jvp
def eigenvalues(h, o):
    r""":math:`\varepsilon_a` of :math:`Hc=\varepsilon Oc`, with its own rule.

    The eigenvalue derivative :math:`d\varepsilon_a=\langle a|dH-\varepsilon_a
    dO|a\rangle` has no denominator in it and is therefore safe at a
    degeneracy — but JAX's ``eigh`` rule computes the eigenvector tangent in
    the same primitive, and that one is not.  Writing the rule out keeps the
    unsafe expression from being formed at all rather than relying on it being
    eliminated as dead code.

    At an exact degeneracy the individual :math:`d\varepsilon_a` are basis
    dependent; every use here is through a symmetric function of the spectrum
    (the Fermi level, the eigenvalue sum), which is not.
    """
    return _solve(h, o)[0]


@eigenvalues.defjvp
def _eigenvalues_jvp(primals, tangents):
    (h, o), (dh, do) = primals, tangents
    values, c = _solve(h, o)
    left = c.conj().T
    tangent = (jnp.einsum("ai,ia->a", left, dh @ c)
               - values * jnp.einsum("ai,ia->a", left, do @ c))
    return values, jnp.real(tangent)


def occupation_weights(values, mu, swidth, occmax, e0min, stype, nstfv):
    r"""``occupy``'s :math:`f_a` and :math:`\partial f_a/\partial\mu`, truncated.

    The truncation is Elk's: ``eveqnfv`` returns ``nstfv`` states and the
    density sums over those, so anything above is identically empty and its
    derivative is identically zero — a *fact about the code*, not a limit taken
    numerically.  The ``e0min`` gate is the same statement about states below
    the lowest linearisation energy (see `elkjax.occupations.occupations`).
    """
    values = jnp.asarray(values)
    gate = values >= e0min
    x = (mu - values) / swidth
    f = jnp.where(gate, occmax * _occ.stheta(stype, x), 0.0)
    dfdmu = jnp.where(gate, occmax * _occ.sdelta(stype, x) / swidth, 0.0)
    keep = jnp.asarray(np.arange(values.shape[-1]) < nstfv)
    return jnp.where(keep, f, 0.0), jnp.where(keep, dfdmu, 0.0)


def pair_kernels(values, mu, swidth, occmax, e0min, stype, nstfv):
    r""":math:`(K^H,K^S,f)` — the two divided differences and the occupations.

    The closed form is used wherever it applies: both states inside the
    ``nstfv`` window and above ``e0min``, so that :math:`f_a` really is the
    smooth function of :math:`\varepsilon_a` the closed form assumes.  A pair
    with either state outside takes the plain quotient, which is safe there
    because such a pair is separated by at least the distance to the truncation
    or to ``e0min`` — never by a rounding error.
    """
    if int(stype) != 3:
        raise ValueError(
            f"stype={int(stype)}: the divided difference K^H has a closed form "
            f"only for Fermi-Dirac smearing (stype=3). Methfessel-Paxton would "
            f"need the tol threshold Phase 1i measured to be a cliff, so it is "
            f"refused rather than approximated.")
    values = jnp.asarray(values)
    f, dfdmu = occupation_weights(values, mu, swidth, occmax, e0min, stype,
                                  nstfv)
    smooth = ((values >= e0min)
              & jnp.asarray(np.arange(values.shape[-1]) < nstfv))
    both = smooth[:, None] & smooth[None, :]

    gap = values[:, None] - values[None, :]
    quotient = ((f[:, None] - f[None, :])
                / jnp.where(gap == 0.0, 1.0, gap))
    quotient = jnp.where(gap == 0.0, 0.0, quotient)
    kh = jnp.where(both, occmax * fermi_kernel(values, mu, swidth), quotient)
    ks = (0.5 * (values[:, None] + values[None, :]) * kh
          + 0.5 * (f[:, None] + f[None, :]))
    return kh, ks, f, dfdmu


@functools.partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5, 6, 7))
def density_factors(h, o, mu, nstfv, swidth, occmax, e0min, stype):
    r"""The two coefficient factors the valence density is bilinear in.

    Returns :math:`(\bar C, X)`, both ``(nmat, nstfv)``, with
    :math:`\bar c_a=c_a` and :math:`x_a=f_ac_a`; the density is then
    :math:`\sum_a\mathrm{Re}[\overline{W\bar c_a}\,(Wx_a)]`, which is
    :math:`\sum_af_a|\psi_a|^2` as a value and carries the whole of
    :math:`d\rho` as a derivative.  See the module docstring for why the
    boundary is drawn here.
    """
    values, c = _solve(h, o)
    f, _ = occupation_weights(values, mu, swidth, occmax, e0min, stype, nstfv)
    bra = c[:, :nstfv]
    return bra, bra * f[:nstfv][None, :]


@density_factors.defjvp
def _density_factors_jvp(nstfv, swidth, occmax, e0min, stype, primals,
                         tangents):
    h, o, mu = primals
    dh, do, dmu = tangents
    values, c = _solve(h, o)
    kh, ks, f, dfdmu = pair_kernels(values, mu, swidth, occmax, e0min, stype,
                                    nstfv)
    left = c.conj().T
    g = kh * (left @ dh @ c) - ks * (left @ do @ c)
    bra = c[:, :nstfv]
    tangent = (bra @ g[:nstfv, :nstfv]
               + 2.0 * c[:, nstfv:] @ g[nstfv:, :nstfv]
               + bra * (dfdmu[:nstfv] * dmu)[None, :])
    return (bra, bra * f[:nstfv][None, :]), (jnp.zeros_like(bra), tangent)


def density_matrix_tangent(bra, tangent):
    r""":math:`\tfrac12(dX\bar C^\dagger+\bar CdX^\dagger)`, the tangent of
    :math:`P=\sum_af_a|a\rangle\langle a|`.

    Not used by the density itself — the grid never sees an
    :math:`n_{\rm mat}`-square matrix — but it is the object
    :func:`density_factors`' rule can be checked against without any of Elk in
    the way, and the check pins the factor 2 and the :math:`\mathrm{Re}` that
    are otherwise invisible.
    """
    product = jnp.asarray(tangent) @ jnp.asarray(bra).conj().T
    return 0.5 * (product + product.conj().T)
