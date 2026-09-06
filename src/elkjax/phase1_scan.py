r"""Phase 1, leftover item: the Newton-Schulz tape as a ``lax.scan``.

:func:`elkjax.projector.sign_projector` is the only projector here that survives a
second derivative at a multiplet (Phase 1j), and it got there by containing no
eigensolve -- just a fixed number of Newton-Schulz steps.  Written as a Python
``for`` loop those steps are **unrolled** into the graph, which is precisely the
construct Phase 0e measured as expensive: compile time is flat in the tensor extents
and *superlinear* in HLO op count, exponent :math:`\approx1.85`, so a tape whose op
count is proportional to ``steps`` costs more than proportionally to compile.

``lax.scan`` emits the body once.  This module measures what that is worth, and --
more importantly -- that it costs nothing in accuracy, since a compile-time
optimisation that perturbs a second derivative at a degeneracy would not be worth
having.

Two things are deliberately measured rather than assumed:

* **The rewrite is NOT bitwise, and the reason is worth knowing.**  Both branches
  call the same ``_newton_schulz_step`` the same number of times in the same order,
  so it is tempting to assert equality -- and the projector itself differs by
  :math:`2.8\times10^{-16}` (:math:`6\times10^{-16}` relative).  Calling a function
  the same number of times does not fix the arithmetic: XLA fuses a ``scan`` body
  and an unrolled chain into different regions and is free to reassociate inside
  each.  So the check is a relative tolerance at all three orders (measured
  :math:`9\times10^{-16}`, :math:`1.5\times10^{-16}`, :math:`4.5\times10^{-15}`),
  and asserting exact equality would be a test that fails on a compiler upgrade for
  no reason.
* **Reverse mode does not obviously save memory.**  ``scan``'s backward pass stores
  one residual per iteration exactly as the unrolled tape does; the win is the
  compile, not the tape.  ``temp_size_in_bytes`` is reported so the claim is a
  number rather than a hope.

Run: ``PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase1_scan``.  Needs no Elk
binary -- the matrix is synthetic, with a spectrum shaped like the real one Phase 1j
measured (occupied manifold 0.35 Ha wide, basis top 18 Ha, so a norm-to-gap ratio of
order 100 and ~13 steps needed).
"""

import numpy as np

import jax
import jax.numpy as jnp

from . import memory, projector

__all__ = ["hamiltonian", "agreement", "compile_cost", "sweep", "report", "main"]

DEFAULT_STEPS = 20


def hamiltonian(n=64, nocc=16, seed=0, occupied=(-1.0, -0.2),
                unoccupied=(0.2, 18.0)):
    r"""A symmetric matrix with a Phase-1j-shaped spectrum.

    The occupied manifold is narrow and the unoccupied one reaches far above it,
    which is what makes the Newton-Schulz count interesting: it is set by
    :math:`\log(\lVert H-\mu\rVert/\Delta)`, so the *top of the basis* and not the
    valence bandwidth decides how many steps are needed (Phase 1j measured a ratio of
    190 and a count of 13 on bulk silicon).  A toy with a symmetric spectrum would
    converge in five steps and say nothing about the tape length that matters.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    evals = np.concatenate([np.linspace(*occupied, nocc),
                            np.linspace(*unoccupied, n - nocc)])
    return jnp.asarray(q @ np.diag(evals) @ q.T), nocc


def _line(h, direction, weight, nocc, steps, unroll):
    r"""A scalar function of one real parameter, to differentiate twice.

    The observable is :math:`\mathrm{tr}(PW)` for a fixed symmetric :math:`W`, and
    the choice matters more than it looks.  The obvious
    :math:`\sum_{ij}P_{ij}^2=\mathrm{tr}(P^2)=\mathrm{tr}(P)=n_{\rm occ}` is
    **constant** while the gap stays open, so differentiating it compares two noise
    floors around zero and would report agreement whatever the two branches did.
    Measured before the fix: first derivative :math:`7\times10^{-16}`, second
    :math:`1.6\times10^{-14}`, both of a function that is exactly 16.
    """
    def scalar(t):
        p = projector.sign_projector(h + t * direction, nocc, steps=steps,
                                     unroll=unroll)
        return jnp.sum(p * weight)
    return scalar


def agreement(n=64, nocc=16, steps=DEFAULT_STEPS, seed=0):
    """Value, first and second derivative, scanned against unrolled.

    Returns the three values and their differences, all expected at roundoff
    relative to the value -- not exactly zero, for the reason in the module
    docstring.
    """
    h, nocc = hamiltonian(n=n, nocc=nocc, seed=seed)
    rng = np.random.default_rng(seed + 1)
    a = rng.standard_normal((n, n))
    direction = jnp.asarray(a + a.T)
    b = rng.standard_normal((n, n))
    weight = jnp.asarray(b + b.T)

    out = {}
    for order, transform in ((0, lambda f: f),
                             (1, jax.grad),
                             (2, lambda f: jax.grad(jax.grad(f)))):
        values = [float(transform(
            _line(h, direction, weight, nocc, steps, unroll))(0.0))
            for unroll in (False, True)]
        out[order] = dict(scanned=values[0], unrolled=values[1],
                          difference=abs(values[0] - values[1]))
    return out


def _instructions(fn, aval):
    """Optimised-HLO instruction count: one per ``=`` assignment in the text.

    A proxy, and labelled as one.  Phase 0e used the *pass count* of an unrolled
    construct as its proxy; counting the emitted text is a step closer to the thing
    XLA's pass pipeline actually walks, and the two agree on the direction.
    """
    text = jax.jit(fn).lower(aval).compile().as_text()
    return sum(1 for line in text.splitlines() if " = " in line)


def compile_cost(n=64, nocc=16, steps=DEFAULT_STEPS, order=1, unroll=False):
    """Compile time, instruction count and scratch bytes, without executing."""
    h, nocc = hamiltonian(n=n, nocc=nocc)
    direction = jnp.eye(n)
    scalar = _line(h, direction, jnp.eye(n), nocc, steps, unroll)
    fn = {0: scalar, 1: jax.grad(scalar),
          2: jax.grad(jax.grad(scalar))}[order]
    aval = jax.ShapeDtypeStruct((), jnp.float64)
    seconds, stats = memory.compiled_cost(fn, aval)
    return dict(n=n, steps=steps, order=order,
                mode="unrolled" if unroll else "scan",
                seconds=seconds, instructions=_instructions(fn, aval),
                temp=stats.temp_size_in_bytes)


def sweep(step_counts=(10, 20, 40, 80), orders=(0, 1, 2), n=64):
    rows = []
    for steps in step_counts:
        for order in orders:
            for unroll in (False, True):
                rows.append(compile_cost(n=n, steps=steps, order=order,
                                         unroll=unroll))
    return rows


def report():
    print("agreement, scanned vs unrolled (n=64, 20 steps)")
    for order, row in agreement().items():
        rel = row["difference"] / max(abs(row["scanned"]), 1e-300)
        print("  order %d   scanned %+.12e   difference %.3e (rel %.1e)"
              % (order, row["scanned"], row["difference"], rel))

    print("\ncompile cost, n=64")
    print("  %-6s %-5s %-9s %10s %8s %10s" %
          ("steps", "order", "mode", "seconds", "instrs", "temp (B)"))
    rows = sweep()
    for row in rows:
        print("  %-6d %-5d %-9s %10.3f %8d %10d" %
              (row["steps"], row["order"], row["mode"], row["seconds"],
               row["instructions"], row["temp"]))

    print("\nratio unrolled/scan")
    for order in (0, 1, 2):
        for steps in (10, 20, 40, 80):
            pair = {r["mode"]: r for r in rows
                    if r["steps"] == steps and r["order"] == order}
            print("  order %d steps %3d   time x%6.2f   instructions x%6.2f"
                  % (order, steps,
                     pair["unrolled"]["seconds"] / pair["scan"]["seconds"],
                     pair["unrolled"]["instructions"]
                     / pair["scan"]["instructions"]))
    return rows


def main():
    memory.limit_address_space()
    report()


if __name__ == "__main__":
    main()
