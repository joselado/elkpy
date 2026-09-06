r"""Phase 0e: what does one traced SCF step cost to compile, and to hold?

Study §6 item 0e asks for ``jit`` compile time and peak device memory for **one** traced
SCF step at production shapes (:math:`n_{\rm mat}\approx3000`, :math:`n_{\bf k}\approx100`),
and warns that §8(d)'s reassuring toy-scale numbers (:math:`n\le400`, 4 k-points) must not
be extrapolated, because §3 proposes unrolled constructs — an 8-pass corrector, unrolled
Gram-Schmidt, a scan over ~700 radial points — inside a step that also contains per-k
eigensolves, and XLA compile time grows superlinearly in HLO op count.

**Nothing here is executed.**  ``jax.jit(f).lower(*avals).compile()`` builds the
executable from abstract shapes, so the 26.8 GiB of :math:`H` and :math:`S` this machine
cannot hold is never allocated (CLAUDE.md, "JAX port").  ``memory_analysis()`` then
reports what a device *would* need, which is the number the item wants and the one an
executed small case cannot give.

**What is and is not faithful.**  The shapes, the dtype, the loop structure and the
op-count-inflating constructs are the real ones.  The *arithmetic* inside them is a
stand-in: there is no ``match``, no Weinert Poisson and no XC functional here, because
those are Phase 1 and 2.  So these numbers bound the cost of the skeleton every real
implementation must carry, and are a lower bound on the real thing — which is the useful
direction for a "is this a wall?" question.

It also answers the **memory half** of item 0d without the GPU that item needs:
``vmap`` over the k-axis versus ``lax.map`` is a compile-time-visible difference in
buffer sizes, even though the timing question is not.
"""

import numpy as np

import jax
import jax.numpy as jnp

from . import memory

__all__ = ["scf_step", "measure", "sweep"]


def _muffin_tin_pass(v, nr, lmmax):
    """A scan over radial points with a small dense solve at each — §3.3's shape."""
    def body(carry, row):
        carry = carry + row
        return carry * 0.999 + 0.001 * jnp.tanh(carry), carry.sum()

    rows = jnp.reshape(v[:nr * lmmax] if v.size >= nr * lmmax
                       else jnp.resize(v, nr * lmmax), (nr, lmmax))
    _, out = jax.lax.scan(body, jnp.zeros(lmmax), rows)
    return out


def _corrector(rho, passes):
    """An unrolled multi-pass corrector, §3.3's 8-pass shape."""
    for _ in range(passes):
        rho = rho - 0.1 * (rho - jnp.roll(rho, 1))
    return rho


def _gram_schmidt(basis):
    """Unrolled modified Gram-Schmidt over the local-orbital block, §3.2's shape."""
    if basis.shape[1] == 0:
        return basis
    columns = []
    for j in range(basis.shape[1]):
        column = basis[:, j]
        for previous in columns:
            column = column - previous * jnp.vdot(previous, column)
        columns.append(column / jnp.linalg.norm(column))
    return jnp.stack(columns, axis=1)


def scf_step(n_mat, n_k, *, nocc=None, k_axis="scan", radial=700, lmmax=49,
             corrector_passes=8, n_lo=8):
    r"""One traced Kohn-Sham step at the given shapes, as a function of ``(v, h, s)``.

    ``h``/``s`` carry the per-k matrices, which is what makes the k-axis choice a memory
    decision.  ``k_axis`` picks between ``vmap`` (every k-point's intermediates live at
    once), ``"map"`` (``lax.map``: one k-point's matrices, but the per-k *outputs* are
    still stacked) and ``"scan"`` (an accumulator, so nothing per-k survives the loop).
    ``nocc`` defaults to a quarter filling.
    """
    nocc = nocc or max(1, n_mat // 4)

    def one_kpoint(hk_sk):
        hk, sk = hk_sk
        chol = jnp.linalg.cholesky(sk)
        reduced = jax.scipy.linalg.solve_triangular(chol, hk, lower=True)
        reduced = jax.scipy.linalg.solve_triangular(chol, reduced.conj().T,
                                                    lower=True).conj().T
        evals, evecs = jnp.linalg.eigh(reduced)
        occupied = evecs[:, :nocc]
        if n_lo:
            occupied = occupied.at[:, :n_lo].set(_gram_schmidt(occupied[:, :n_lo]))
        return jnp.real(jnp.sum(occupied * occupied.conj(), axis=1)), evals[:nocc].sum()

    def step(v, h, s):
        radial_part = _muffin_tin_pass(v, radial, lmmax)
        if k_axis == "vmap":
            density, band = jax.vmap(one_kpoint)((h, s))
            density, band = jnp.sum(density, axis=0), jnp.sum(band)
        elif k_axis == "map":
            # lax.map still STACKS the per-k outputs -- n_k x n_mat, not n_k x n_mat^2,
            # so it is small, but it is not nothing.
            density, band = jax.lax.map(one_kpoint, (h, s))
            density, band = jnp.sum(density, axis=0), jnp.sum(band)
        else:
            # scan with an accumulator: nothing per-k survives the loop at all
            def accumulate(carry, hk_sk):
                d, b = one_kpoint(hk_sk)
                return (carry[0] + d, carry[1] + b), None
            (density, band), _ = jax.lax.scan(
                accumulate, (jnp.zeros(n_mat), jnp.zeros(())), (h, s))
        density = density / n_k
        density = _corrector(density, corrector_passes)
        return density + jnp.sum(radial_part) * 0.0 + band * 0.0

    return step


def measure(n_mat, n_k, *, k_axis="scan", differentiate=False, **options):
    """Compile at these shapes without executing, returning time and buffer sizes.

    ``differentiate=True`` compiles ``grad`` of a scalar reduction of the step instead,
    which is the unit a Phase 3 SCF gradient actually has to build.
    """
    step = scf_step(n_mat, n_k, k_axis=k_axis, **options)
    if differentiate:
        inner = step
        step = jax.grad(lambda v, h, s: jnp.sum(inner(v, h, s) ** 2))
    avals = (jax.ShapeDtypeStruct((n_mat,), jnp.float64),
             jax.ShapeDtypeStruct((n_k, n_mat, n_mat), jnp.complex128),
             jax.ShapeDtypeStruct((n_k, n_mat, n_mat), jnp.complex128))
    seconds, stats = memory.compiled_cost(step, *avals)
    return dict(n_mat=n_mat, n_k=n_k, k_axis=k_axis, differentiate=differentiate,
                seconds=seconds,
                arguments=stats.argument_size_in_bytes,
                temp=stats.temp_size_in_bytes,
                output=stats.output_size_in_bytes,
                code=stats.generated_code_size_in_bytes)


def sweep(sizes=((200, 4), (400, 4), (800, 4), (1500, 4), (3000, 4),
                 (3000, 25), (3000, 100)), k_axis="scan", budget=600.0, **options):
    """Walk up to the production shape, stopping if a compile exceeds ``budget`` seconds."""
    rows = []
    for n_mat, n_k in sizes:
        row = measure(n_mat, n_k, k_axis=k_axis, **options)
        rows.append(row)
        if row["seconds"] > budget:
            row["stopped"] = True
            break
    return rows


def main():
    memory.limit_address_space(16.0)
    gb = memory.GB
    print("== one traced SCF step, compiled but NEVER executed ==")
    print(f"  {'n_mat':>6} {'n_k':>5} {'k axis':>7} {'compile s':>10} "
          f"{'args GiB':>9} {'temp GiB':>9}")
    for k_axis in ("scan", "map", "vmap"):
        for row in sweep(k_axis=k_axis):
            print(f"  {row['n_mat']:>6} {row['n_k']:>5} {row['k_axis']:>7} "
                  f"{row['seconds']:>10.2f} {row['arguments']/gb:>9.3f} "
                  f"{row['temp']/gb:>9.3f}"
                  + ("   [over budget, stopped]" if row.get("stopped") else ""))

    print("\n== op-count sensitivity at a FIXED shape (n_mat=800, n_k=4) ==")
    print("   compile time is flat in the shapes above, so this is where hazard K lives")
    print(f"  {'construct':>34} {'compile s':>10} {'temp GiB':>9}")
    for label, options in (("no unrolled Gram-Schmidt", dict(n_lo=0)),
                           ("baseline (8 lo, 8-pass)", {}),
                           ("32 local orbitals", dict(n_lo=32)),
                           ("64 local orbitals", dict(n_lo=64)),
                           ("128 local orbitals", dict(n_lo=128)),
                           ("64-pass corrector", dict(corrector_passes=64)),
                           ("256-pass corrector", dict(corrector_passes=256)),
                           ("2800-point radial scan", dict(radial=2800))):
        row = measure(800, 4, **options)
        print(f"  {label:>34} {row['seconds']:>10.2f} {row['temp']/gb:>9.3f}")

    print(f"\npeak RSS {memory.peak_rss_bytes() / gb:.2f} GB (nothing was executed)")


if __name__ == "__main__":
    main()
