r"""Phase 2: `symrfmt`, the muffin-tin symmetrisation Elk applies to $v_{xc}$.

`potxc.f90` lines 55-58 symmetrise `vxcmt` and `bxcmt` and **not** `exmt`/`ecmt`
(`docs/jax_port_phase2.md` §2d), so a transcription that applies the functional
pointwise reproduces the energy densities exactly and misses `vxcmt` by 1.2e-4
relative.  §2f then measured that this costs nothing in any integral against
$\rho$ -- symmetrisation is a group average, hence an orthogonal projection, and
$\rho$ is already in its range -- but an SCF iteration compares potentials
*pointwise*, so closing the loop needs the operator.

**It is exported, not transcribed, and that is the better route rather than the
cheaper one.**  `rotrflm`'s Euler-angle and Wigner-$D$ construction is two
hundred lines whose only consumer inside Elk is `symrfmt` itself, so a Python
re-derivation would have no independent check except agreement with the thing it
replaces; and Elk's atom bookkeeping would have to be transcribed with it --
`ieqatom`, `tfeqat`, and the *inverse* lattice rotation in the
rotate-into-equivalent-atoms loop, none of which the available fixtures exercise
in both directions.  Patch 0018 instead calls `symrfmt` on basis vectors and
exports the resulting linear operator, so what is tested here is its
*application*, with no convention in it at all.

The operator is one $l_{\max}^{\rm o}$-square matrix per **ordered atom pair**,
which is the whole of it: a rotation is diagonal in the radial index and does not
mix $l$, so the radial mesh does not enter, and the inner region -- carrying only
$l_{\max}^{\rm i}$ harmonics -- uses the same matrix's top-left block, since
`rotrfmt` calls `rotrflm` separately on the two regions with the same rotation.
"""

import numpy as np

import jax.numpy as jnp

from . import poisson

__all__ = ["symmetrise", "symmetrise_packed", "is_projection",
           "idempotence_residual"]


def symmetrise(values, groundstate):
    r"""$\hat Sf$, for a stack of dense ``(nr, lmmaxo)`` muffin-tin functions.

    ``values[ias]`` is atom ``ias``'s function; the result mixes atoms, since
    `symrfmt` averages over operations that permute equivalent sites.

    The inner region's harmonics above $l_{\max}^{\rm i}$ are zero in the dense
    representation and stay zero: the operator's top-left block is closed on
    them, a rotation not mixing $l$.  That is asserted rather than assumed by
    :func:`elkjax.poisson.pack`'s round trip in the tests.
    """
    operator = jnp.asarray(groundstate["symop"])
    values = jnp.asarray(values)
    natmtot = int(groundstate["natmtot"])
    lmmaxi = int(groundstate["lmmaxi"])

    out = []
    for ias in range(natmtot):
        isp = int(groundstate["idxis"][ias]) - 1
        nri = int(groundstate["nrmti"][isp])
        outer = sum(values[jas] @ operator[ias, jas].T
                    for jas in range(natmtot))
        inner = sum(values[jas][:nri, :lmmaxi]
                    @ operator[ias, jas][:lmmaxi, :lmmaxi].T
                    for jas in range(natmtot))
        out.append(outer.at[:nri, :].set(0.0).at[:nri, :lmmaxi].set(inner))
    return jnp.stack(out)


def symmetrise_packed(packed, groundstate):
    """:func:`symmetrise`, in Elk's own packing on both ends."""
    natmtot = int(groundstate["natmtot"])
    dense = jnp.stack([poisson.dense(packed[ias], groundstate, ias)
                       for ias in range(natmtot)])
    values = symmetrise(dense, groundstate)
    return jnp.stack([poisson.pack(values[ias], groundstate, ias)
                      for ias in range(natmtot)])


def idempotence_residual(groundstate):
    r"""$\lVert\hat S^2-\hat S\rVert_\infty$, on the exported matrices alone.

    A group average is idempotent, so this is a cheap check that the exported
    operator really is an average over a *closed* set of operations -- it
    catches a missing element or a double-counted one without needing a ground
    state, a functional, or Elk's own ``vxcmt``.

    **It is not the same number on every lattice**, and the difference is
    Elk's, not this module's.  Measured: $10^{-16}$ at every $l$ on bulk
    silicon, and $2.5\times10^{-12}$ at $l=1$ rising to $1.2\times10^{-11}$ at
    $l=5$ on monolayer h-BN.  ``symrfmt`` builds each rotation through
    ``roteuler``, which extracts Euler angles from the Cartesian ``symlatc`` by
    inverse trigonometry; for a cubic lattice those entries are exactly $0$ and
    $\pm1$ and the angles are exact multiples of $\pi/2$, and for a hexagonal
    one they are not.  The growth with $l$ is the Wigner-$D$ matrices of
    increasing order compounding that angle error.

    This bounds how idempotent ``symrfmt`` can be at all; it does **not** bound
    the operator's accuracy in use, which applies it once -- the ``vxcmt``
    comparison holds at $10^{-14}$ on both structures.
    """
    operator = np.asarray(groundstate["symop"])
    natmtot, _, n, _ = operator.shape
    flat = operator.transpose(0, 2, 1, 3).reshape(natmtot * n, natmtot * n)
    return float(np.abs(flat @ flat - flat).max())


def is_projection(groundstate, tol=1e-10):
    """:func:`idempotence_residual` against a tolerance; see it for why 1e-10."""
    return idempotence_residual(groundstate) < tol
