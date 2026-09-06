r"""Implicit differentiation of an SCF fixed point — study §8(a), Phase 0a.

The ground state is defined by :math:`v^*=F(v^*;\theta)`, and differentiating that
condition gives

.. math::

    \Big(\mathbb 1 - \tfrac{\partial F}{\partial v}\Big)\frac{dv^*}{d\theta}
    = \frac{\partial F}{\partial\theta},
    \qquad
    \frac{\partial F}{\partial v} = K_{\rm Hxc}\,\chi_0 ,

so for a scalar loss reverse mode solves the **transposed** system
:math:`(\mathbb 1-K_{\rm Hxc}\chi_0)^{\mathsf T}\lambda=\partial L/\partial v` once and
contracts :math:`\lambda` with :math:`\partial F/\partial\theta`.

**Why this is the item that decides the port.**  :math:`\chi_0` *is* the derivative of
the occupied projector, so the matvec of that linear system is the JVP of one
Kohn-Sham step — it passes through ``eigh``'s derivative at every multiplet, on every
GMRES iteration, and is then transposed.  Study §6 item 0a is exactly whether the
safe-:math:`K` rule of :mod:`elkjax.projector` (blocker B1) survives being used this way
(blocker B2).  A smooth scalar fixed point with no eigensolve is already known to work
and settles nothing.

Three deliberate choices, each from the study:

* ``jax.custom_vjp`` and an explicit GMRES, **not** ``lax.custom_root`` with an
  iterative ``tangent_solve`` — measured there to raise ``NotImplementedError``, which
  would read as a kill and is not one.
* The mixer is the *solver*, not the equation.  :math:`F` is defined without it, the
  converged answer does not depend on it, and neither may the gradient — which is a
  cheap, reference-free assertion (study §8a, Phase 3 "Gradient A").
* The forward iteration is a plain Python loop with a data-dependent stopping test.  It
  is therefore not ``jit``-able as written; that is Phase 0e's problem, not this one,
  and ``lax.while_loop`` has no reverse rule anyway.
"""

import numpy as np

import jax
import jax.numpy as jnp

__all__ = ["implicit_fixed_point", "iterate", "adjoint_residual"]


def iterate(step, theta, v0, *, mixing=0.5, tol=1e-12, maxiter=500, history=0,
            ridge=1e-10):
    r"""Solve :math:`v=F(v;\theta)` by damped iteration, or Anderson if ``history>0``.

    ``mixing`` is Elk's ``beta0`` in spirit: :math:`v\leftarrow(1-\beta)v+\beta F(v)`.
    ``history>0`` switches to Anderson/Pulay over that many stored residuals, which
    reaches a *different point inside the same tolerance ball* — the whole point of the
    mixer-independence check, since the gradient must not notice.
    """
    v = v0
    iterates, residuals = [], []
    norm = float("inf")
    for it in range(maxiter):
        residual = step(v, theta) - v
        norm = float(jnp.linalg.norm(residual))
        if not np.isfinite(norm):
            raise FloatingPointError(f"fixed-point iteration diverged at step {it}")
        if norm < tol:
            return v, it, norm
        iterates.append(v)
        residuals.append(residual)
        keep = max(history, 1)
        if len(iterates) > keep:
            iterates, residuals = iterates[-keep:], residuals[-keep:]
        if history and len(iterates) > 1:
            v = _anderson(iterates, residuals, mixing, ridge)
        else:
            v = v + mixing * residual
    return v, maxiter, norm


def _anderson(iterates, residuals, mixing, ridge=1e-10):
    r"""Least-squares extrapolation over the stored ``(x_k, r_k)`` pairs.

    Minimise :math:`\|\sum_k\alpha_kr_k\|` subject to :math:`\sum_k\alpha_k=1`, then take
    :math:`x\leftarrow\sum_k\alpha_kx_k+\beta\sum_k\alpha_kr_k`.  The Gram matrix goes
    singular as the residuals become parallel near convergence, so it carries a
    relative ridge rather than an absolute one.
    """
    r = jnp.stack(residuals, axis=1)
    x = jnp.stack(iterates, axis=1)
    gram = r.T @ r
    scale = jnp.trace(gram) / gram.shape[0]
    weights = jnp.linalg.solve(gram + ridge * scale * jnp.eye(gram.shape[0]),
                               jnp.ones(gram.shape[0]))
    weights = weights / jnp.sum(weights)
    return x @ weights + mixing * (r @ weights)


def implicit_fixed_point(step, *, solver_kwargs=None, gmres_kwargs=None):
    r"""Wrap ``step(v, theta)`` as a differentiable ``solve(theta, v0) -> v*``.

    The returned function is differentiable in ``theta`` only; ``v0`` is a starting
    guess and receives a zero cotangent, which is correct — the fixed point does not
    depend on it.
    """
    solver_kwargs = dict(solver_kwargs or {})
    linear = dict(tol=1e-10, atol=0.0, restart=20, maxiter=200)
    linear.update(gmres_kwargs or {})

    @jax.custom_vjp
    def solve(theta, v0):
        return iterate(step, theta, v0, **solver_kwargs)[0]

    def solve_fwd(theta, v0):
        v = solve(theta, v0)
        return v, (theta, v)

    def solve_bwd(res, cotangent):
        theta, v = res
        # (1 - dF/dv)^T u = dL/dv.  vjp_v(u) IS (dF/dv)^T u, and it runs one full
        # Kohn-Sham JVP -- eigensolve included -- per GMRES iteration.
        _, vjp_v = jax.vjp(lambda vv: step(vv, theta), v)
        matvec = lambda u: u - vjp_v(u)[0]
        u, _ = jax.scipy.sparse.linalg.gmres(matvec, cotangent, **linear)
        _, vjp_theta = jax.vjp(lambda th: step(v, th), theta)
        return vjp_theta(u)[0], jnp.zeros_like(v)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve


def adjoint_residual(step, theta, v, cotangent, **gmres_kwargs):
    r"""Re-solve the adjoint system and report how well GMRES actually did.

    ``jax.scipy.sparse.linalg.gmres`` returns ``info=0`` unconditionally, so the only
    way to know whether the transposed operator was inverted — as opposed to returning
    whatever 20 restarts reached — is to form the residual
    :math:`\|(\mathbb 1-\partial_vF)^{\mathsf T}u - \partial L/\partial v\|` directly.
    """
    linear = dict(tol=1e-10, atol=0.0, restart=20, maxiter=200)
    linear.update(gmres_kwargs)
    _, vjp_v = jax.vjp(lambda vv: step(vv, theta), v)
    matvec = lambda u: u - vjp_v(u)[0]
    u, _ = jax.scipy.sparse.linalg.gmres(matvec, cotangent, **linear)
    return u, float(jnp.linalg.norm(matvec(u) - cotangent) / jnp.linalg.norm(cotangent))
