r"""Phase 1 of the JAX port: build the APW and local-orbital radial functions.

`test_calculation_lapw_radial.py` builds the radial INTEGRALS from the radial
functions.  This builds the radial functions themselves from the potential --
`src/elkjax/radial_functions.py`, a transcription of `rschrodint.f90`,
`genapwfr.f90` and `genlofr.f90` -- and checks them against Elk's own, which
patch 0015 exports in full.  With this the chain

    vsmt  ->  apwfr / lofr  ->  radial integrals  ->  H, O  ->  evalfv

is closed, and no part of it is an imported number any more.

The comparison is element-wise on the mesh, so it is a transcription check and
not a physics one: an integrator that converged to the same continuum solution
by a different route would disagree at the mesh's own truncation error, four
or five decades above what is asserted here.

What each fixture catches, beyond what the assembly test already needs them
for:

  * Si at apword=1 -- `apwfr`'s second component is exactly `e` times the
    first, and `apwdfr` is a single unmixed surface derivative, so this
    fixture cannot see the Gram-Schmidt at all.
  * Si at apword=2 -- it can.  `genapwfr` carries `e*u` and the surface
    derivative through the SAME orthonormalisation as `u`, so at order 2 the
    second component is a COMBINATION of `e_i u_i` and not `e` times anything.
  * h-BN -- two species with different meshes, and a nitrogen carrying two
    l=0 local orbitals, which is the only place `genlofr`'s
    same-l orthogonalisation (in Elk's ascending-energy `idxelo` order) does
    any work.

The rebuilt `D` is checked here too.  It is what carries a perturbed `apwfr`
into `apwalm`, and it is invisible to any gradient check: a chain that rebuilt
the radial functions and left `D` alone differentiates a truncated function
consistently and agrees with its own finite differences perfectly.

`rschrodint`'s node count and its overflow freeze are deliberately absent
from the transcription; both are non-differentiable branches, and the first
is only used by `linengy`, which this port does not run.

Skipped without the elk binary, and without jax.
"""

import importlib.util

import numpy as np
import pytest

from elkpy import config

pytestmark = [
    pytest.mark.skipif(not config.default_elk_binary().is_file(),
                       reason="elk binary not built; see docs/design.md #8"),
    pytest.mark.skipif(importlib.util.find_spec("jax") is None,
                       reason="jax not installed; pip install -e .[jax]"),
]

from conftest import LAPW_CASES as CASES  # noqa: F401

# Every radial function is a few hundred predictor-corrector steps in double
# precision; measured worst case across the three fixtures is 7e-13 absolute
# on a scale of 56, i.e. 1.3e-14 relative.
TOL_REL = 1e-11


def _skip_without_potential(export):
    if "vsmt" not in export:
        pytest.skip("binary predates patch 0015 (no potential exported)")


@pytest.mark.parametrize("case", CASES)
def test_apw_radial_functions_match_elk(case, exports):
    """`genapwfr`, element-wise: both components of `apwfr` and `apwdfr`."""
    from elkjax import radial_functions as rf
    export = exports[case]
    _skip_without_potential(export)
    fr, dfr = rf.apw_radial_functions(export)
    for key, got in (("apwfr_full", fr), ("apwdfr", dfr)):
        ref = np.asarray(export[key])
        scale = np.abs(ref).max()
        assert scale > 1e-3, f"{key} reference is trivially small"
        rel = np.abs(np.asarray(got) - ref).max() / scale
        assert rel < TOL_REL, f"{key}: {rel:.3e}"


@pytest.mark.parametrize("case", CASES)
def test_lo_radial_functions_match_elk(case, exports):
    """`genlofr`, element-wise."""
    from elkjax import radial_functions as rf
    export = exports[case]
    _skip_without_potential(export)
    ref = np.asarray(export["lofr"])
    scale = np.abs(ref).max()
    assert scale > 1e-3
    rel = np.abs(np.asarray(rf.lo_radial_functions(export)) - ref).max() / scale
    assert rel < TOL_REL


def test_apword_two_actually_mixes_the_orders(exports):
    """The premise of the apword=2 fixture, asserted rather than assumed.

    At APW order 1 the second component of `apwfr` is exactly `e` times the
    first, so a transcription that ignored `genapwfr`'s Gram-Schmidt entirely
    would still pass the two Si-apword1 and h-BN checks above.  At order 2 it
    must not be -- the orthogonalisation mixes `e_1 u_1` into `u_2`'s partner
    -- and this fails loudly if a future species file quietly drops back to
    order 1.
    """
    export = exports["si_apword2"]
    _skip_without_potential(export)
    apwe = np.asarray(export["apwe"])
    apwdm = np.asarray(export["apwdm"])
    deapw = float(export["deapw"])
    fr = np.asarray(export["apwfr_full"])
    assert int(np.asarray(export["apword"]).max()) == 2
    l, io, ias = 0, 1, 0
    e = apwe[io, l, ias] + apwdm[io, l, 0] * deapw
    u, hu = fr[:, 0, io, l, ias], fr[:, 1, io, l, ias]
    assert np.abs(hu - e * u).max() > 1e-3 * np.abs(hu).max()
    # ... while at order 1 it IS exactly e * u, which is the contrast
    e0 = apwe[0, l, ias] + apwdm[0, l, 0] * deapw
    u0, hu0 = fr[:, 0, 0, l, ias], fr[:, 1, 0, l, ias]
    assert np.abs(hu0 - e0 * u0).max() < 1e-14 * np.abs(hu0).max()


@pytest.mark.parametrize("case", CASES)
def test_the_derivative_matrices_match_elk(case, exports):
    """`D`, the matrix `match` inverts, rebuilt from the radial functions.

    This is the ONLY route by which a rebuilt `apwfr` reaches the matching
    coefficients, and therefore every APW block of both H and O.  It is
    checked against Elk's own `dmat` rather than against a finite difference,
    because a chain that rebuilt `apwfr` and left `D` alone would still pass
    every AD-versus-FD comparison -- both sides differentiate the same
    truncated function.  That is Phase 0's standing finding, and it happened
    here (see `docs/jax_port_phase1.md` §1k).

    Only the apword=2 fixture makes `D` a real matrix; at order 1 it is the
    single number u(R_MT) and the `polynm` derivative rows are never built.
    """
    from elkjax import radial_functions as rf
    export = exports[case]
    _skip_without_potential(export)
    built = rf.derivative_matrices(export)
    worst = 0.0
    for ias, per_l in enumerate(built):
        for l, matrix in enumerate(per_l):
            ref = np.asarray(export["dmat"][ias][l])
            worst = max(worst, np.abs(np.asarray(matrix) - ref).max()
                        / max(np.abs(ref).max(), 1e-30))
    assert worst < 1e-13, worst


@pytest.mark.parametrize("case", CASES)
def test_the_potential_alone_reproduces_elks_eigenvalues(case, exports):
    """The end-to-end statement: from `vsmt` to Elk's own `evalfv`.

    Radial functions, then radial integrals, then the assembly -- with only
    the interstitial blocks (which need the interstitial potential, Phase 2)
    and the linearisation energies taken from the export.
    """
    import jax.numpy as jnp
    from elkjax import hamiltonian as ham, radial, radial_functions as rf
    export = exports[case]
    _skip_without_potential(export)
    rebuilt = radial.integrals_from_export(
        rf.radial_functions_from_export(export))
    h, o = ham.assemble_from_export(rebuilt)
    reduced, _ = ham.cholesky_reduce(h, o)
    evals = np.asarray(jnp.linalg.eigvalsh(reduced))
    ref = np.asarray(export["evalfv"])
    assert np.abs(evals[:len(ref)] - ref).max() < 1e-8


def test_polynomial_helpers_are_exact():
    """`poly3`, `poly4i` and `polynm` on polynomials they must reproduce
    exactly -- the cheapest place a stencil bug shows, and the one that does
    not need Elk at all."""
    import jax.numpy as jnp
    from elkjax import radial_functions as rf
    xa = jnp.asarray([0.3, 0.7, 1.1])
    co = np.array([0.4, -1.2, 2.3])
    ya = sum(co[i] * xa ** i for i in range(3))
    exact = sum(co[i] * 0.9 ** i for i in range(3))
    assert abs(float(rf.poly3(xa, ya, 0.9)) - exact) < 1e-14
    xa4 = jnp.asarray([0.3, 0.7, 1.1, 1.6])
    co4 = np.array([0.4, -1.2, 2.3, -0.8])
    ya4 = sum(co4[i] * xa4 ** i for i in range(4))
    exact = sum(co4[i] * (1.4 ** (i + 1) - 0.3 ** (i + 1)) / (i + 1)
                for i in range(4))
    assert abs(float(rf.poly4i(xa4, ya4, 1.4)) - exact) < 1e-14
    x = jnp.asarray(np.linspace(0.5, 2.0, 8))
    c8 = np.array([1.0, -2.0, 0.7, 0.3, -0.1, 0.05, 0.02, -0.01])
    y = sum(c8[i] * x ** i for i in range(8))
    for m in range(4):
        exact = sum(c8[i] * np.prod([i - j for j in range(m)]) * 1.3 ** (i - m)
                    for i in range(m, 8))
        assert abs(float(rf.polynm(m, x, y, 1.3)) - exact) < 1e-12 * abs(exact)
