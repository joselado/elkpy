"""Guardrails so a JAX experiment cannot take the workstation down.

The design study's Phase 0e (``docs/jax_port.md`` §6) asks for compile time and peak
device memory of one traced SCF step at production shapes, and those shapes do not fit
here.  At the study's own figures — ``n_mat`` ~ 3000, ``n_k`` ~ 100, complex128 — a
single k-point's :math:`H` or :math:`S` is 144 MB, so :math:`H+S` over the k-set is
**26.8 GiB** before the eigenvectors (another 13.4 GiB) and before any eigensolver
workspace, against ~28 GiB available on this machine.  It does not fit, and the margin
is not the kind worth testing empirically.

The answer is not to run it smaller and hope: it is to never execute it at all.
:func:`compiled_cost` lowers and compiles at the real shapes without allocating a byte
of them, which is where XLA's own ``memory_analysis()`` reports the buffer sizes.
:func:`limit_address_space` is the backstop for everything else — verified to leave a
CPU ``eigh`` untouched at n=800 while turning a 25 TB allocation into a prompt
``XlaRuntimeError`` instead of an hour of swapping.

Nothing in this module imports JAX at module scope, so it is cheap to import.
"""

import resource
import time

GB = 1024 ** 3

#: Study §6, item 0e: the shapes a production SCF step would be traced at.
PRODUCTION_SHAPE = {"n_mat": 3000, "n_k": 100}

#: What this box can actually be asked for.  See CLAUDE.md, "JAX port".
LOCAL_BUDGET = {"n_mat": 1500, "n_k": 4}


def matrix_bytes(n, count=1, itemsize=16):
    """Bytes held by ``count`` dense ``n`` x ``n`` matrices (complex128 by default)."""
    return count * itemsize * n * n


def kpoint_set_bytes(n_mat, n_k, matrices_per_k=2, itemsize=16):
    """Bytes for ``matrices_per_k`` matrices at each of ``n_k`` k-points, held at once.

    This is exactly the quantity that makes ``vmap(eigh)`` over the k-axis a memory
    decision rather than a style one: ``vmap`` materialises every k-point's matrix
    simultaneously, ``lax.map``/``lax.scan`` hold one.  Study §6 item 0d is the fork
    that would justify ``vmap``, and it needs a GPU this machine does not have.
    """
    return n_k * matrix_bytes(n_mat, matrices_per_k, itemsize)


def limit_address_space(gb=16.0):
    """Cap this process's virtual address space, returning the previous ``(soft, hard)``.

    A runaway allocation then raises immediately rather than pushing the machine into
    swap.  The cap is deliberately below the ~28 GB available so that hitting it is a
    failed experiment, not a failed workstation.
    """
    previous = resource.getrlimit(resource.RLIMIT_AS)
    limit = int(gb * GB)
    hard = previous[1]
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return previous


def peak_rss_bytes():
    """Peak resident set size of this process so far (``ru_maxrss`` is in KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def compiled_cost(fn, *avals, **kwargs):
    """Compile ``fn`` at the given abstract shapes **without executing it**.

    ``avals`` are :class:`jax.ShapeDtypeStruct` (or anything ``jax.eval_shape`` accepts).
    Returns ``(seconds, jax.stages.CompiledMemoryStats)``; the memory record's
    ``temp_size_in_bytes`` is XLA's own scratch estimate, which is the number Phase 0e
    wants and the one an executed small case cannot give.

    Verified present in JAX 0.7.1.  Compile time here is wall time for the XLA pass
    pipeline, which the study warns grows superlinearly in HLO op count — the whole
    reason for measuring it before Phase 3 designs around its absence.
    """
    import jax

    lowered = jax.jit(fn, **kwargs).lower(*avals)
    started = time.perf_counter()
    compiled = lowered.compile()
    elapsed = time.perf_counter() - started
    return elapsed, compiled.memory_analysis()
