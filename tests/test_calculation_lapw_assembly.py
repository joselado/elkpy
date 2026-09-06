"""Phase 1 of the JAX port: assemble H and O in JAX, compare against Elk's.

`src/elkjax/hamiltonian.py` transcribes olpfv/hmlfv -- the muffin-tin half of
the first-variational LAPW eigenproblem -- and this checks it against the
matrices Elk itself built at the same k-point, block by block, through the
LAPW export (patches/0013 + 0014, docs/design.md #33).

Six blocks are compared SEPARATELY rather than only in total, so a failure
names one upstream routine (olpaa, olpalo, olplolo, hmlaa, hmlalo, hmllolo)
instead of "H is wrong somewhere". The interstitial contributions are taken
from the export and not rebuilt: the Hamiltonian's needs the interstitial
Kohn-Sham potential, which is Phase 2 of the port.

Three fixtures (built once per session in `tests/conftest.py`), each catching
something the others cannot:

  * bulk Si at apword=1  -- the stock case, every species file Elk ships.
  * bulk Si at apword=2  -- the only way to exercise haa's and hloa's APW
    ORDER axes, which are length 1 otherwise, so an io/l index swap is
    invisible at apword=1. Same generated species file the export test uses.
  * monolayer h-BN       -- two species (idxis indexing into rmt/apword/nlorb)
    and, decisively, a nitrogen carrying TWO l=0 local orbitals. hlolo's
    l2=0 element is <u_i|H u_j>, built with no symmetrisation, so the two
    orderings differ (measured: 1.3e-2 Ha) and only the half Elk evaluates
    may be used. Si cannot see this at all -- its local orbitals are one s
    and one p, so no pair shares an l.

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

from conftest import LAPW_KPOINT as KPOINT      # noqa: F401

# Machine precision is the right scale for every block: each is a small dense
# contraction of exported doubles. The one exception is Elk's own side --
# hmlaa/hmlalo skip any Gaunt-contracted z1 with |Re| + |Im| <= 1e-12
# (hmlaa.f90's zaxpy guard), which the dense transcription here keeps. Measured
# on h-BN that costs 3.3e-14, and reproducing the guard recovers 3.0e-17, so
# the tolerance is set by the guard and not by the transcription.
TOL = 1e-12


CASES = ["si_apword1", "si_apword2", "hbn"]


def _reference_blocks(export):
    """Elk's own muffin-tin blocks: its H and O with the exported interstitial
    contribution (which lives in the APW-APW corner alone -- a local orbital
    vanishes outside its sphere) subtracted off."""
    ngp = int(export["ngp"])
    h = np.array(export["hmat"])
    o = np.array(export["omat"])
    h[:ngp, :ngp] -= export["hmat_istl"]
    o[:ngp, :ngp] -= export["omat_istl"]
    return h, o


def _compare_blocks(export):
    from elkjax import hamiltonian as ham
    ngp = int(export["ngp"])
    h_ref, o_ref = _reference_blocks(export)
    o_jax = np.asarray(ham.muffin_tin_overlap(export))
    h_jax = np.asarray(ham.muffin_tin_hamiltonian(export))
    corners = {"apw_apw": (slice(None, ngp), slice(None, ngp)),
               "apw_lo": (slice(None, ngp), slice(ngp, None)),
               "lo_lo": (slice(ngp, None), slice(ngp, None))}
    out = {}
    for name, sel in corners.items():
        for tag, jax_mat, ref in (("olp", o_jax, o_ref), ("hml", h_jax, h_ref)):
            out[f"{tag}_{name}"] = (np.abs(jax_mat[sel] - ref[sel]).max(),
                                    np.abs(ref[sel]).max())
    return out


@pytest.mark.parametrize("case", CASES)
def test_muffin_tin_blocks_match_elk(case, exports):
    """Each of the six muffin-tin blocks, separately.

    Every block is also required to be non-trivial: a transcription that
    returns zeros would otherwise pass the ones Elk also leaves near zero.
    """
    export = exports[case]
    for name, (diff, scale) in _compare_blocks(export).items():
        assert scale > 1e-3, f"{name} reference block is trivially small"
        assert diff < TOL, f"{name}: max|diff| = {diff:.3e}"


@pytest.mark.parametrize("case", CASES)
def test_assembled_matrices_reproduce_elks_eigenvalues(case, exports):
    """H and O assembled here, diagonalised, against Elk's own evalfv.

    This is the whole-object check the block-by-block one cannot be: evalfv
    comes from eveqnfv through Elk's own configured path, so agreement pins
    the assembly, the local-orbital column numbering and the interstitial
    padding together.
    """
    import scipy.linalg as sla
    from elkjax import hamiltonian as ham

    export = exports[case]
    h, o = (np.asarray(m) for m in ham.assemble_from_export(export))
    evalfv = np.asarray(export["evalfv"])
    computed = sla.eigh(h, o, eigvals_only=True)[:len(evalfv)]
    assert np.abs(computed - evalfv).max() < 1e-10


def test_hbn_nitrogen_has_two_l0_local_orbitals(exports):
    """The premise of the h-BN fixture, asserted rather than assumed.

    hmllolo evaluates a local-orbital pair in one order only, and hlolo's
    l2 = 0 element <u_i|H u_j> is not symmetric under exchanging them. That
    can only be caught by a species with two local orbitals sharing an l --
    which is why this fixture exists, and which stops being true silently if
    Elk's shipped N.in changes.
    """
    hbn = exports["hbn"]
    lorbl = [np.asarray(x).tolist() for x in hbn["lorbl"]]
    repeated = [l for species in lorbl for l in set(species)
                if species.count(l) > 1]
    assert repeated, f"no species has a repeated l: {lorbl}"

    hlolo = np.asarray(hbn["hlolo"])
    worst = 0.0
    for ias, is_ in enumerate(np.asarray(hbn["idxis"]) - 1):
        n = len(lorbl[is_])
        block = hlolo[0, :n, :n, ias]
        worst = max(worst, np.abs(block - block.T).max())
    assert worst > 1e-4, (
        f"hlolo's l2=0 block is symmetric to {worst:.1e}; the ordering this "
        "test exists to pin would then be unobservable")



# ---------------------------------------------------------------------------
# The k-dependent pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_k_pipeline_reproduces_elk_at_the_exported_point(case, exports):
    """H, O and the spectrum rebuilt as a FUNCTION of k, evaluated back at
    the exported k.

    Everything k-dependent is rebuilt here -- `apwalm` through
    `elkjax.lapw.match` and the interstitial kinetic term -- so this is a much
    stronger statement than the frozen-apwalm assembly above: it also pins
    the recovery of V_s from the two exported interstitial blocks, and the
    Cholesky reduction.
    """
    from elkjax import hamiltonian as ham

    export = exports[case]
    kc = np.asarray(export["vkc"])
    h, o = (np.asarray(m) for m in ham.eigenproblem_at(export, kc))
    # Looser than TOL, and only at apword=2: `match`'s general branch solves a
    # 2x2 whose rows are u(R) and u'(R), whose magnitudes differ by orders, so
    # it is far worse conditioned than the omax==1 division. Phase 0c measured
    # exactly that -- 1.9e-15 against Elk's apwalm in the fast branch, 8.0e-13
    # in this one -- and the matrices inherit it. Measured here: 9.8e-15 (Si,
    # apword=1), 3.3e-14 (h-BN), 1.3e-12 (Si, apword=2).
    matrix_tol = 1e-11 if int(np.max(export["apword"])) > 1 else TOL
    assert np.abs(h - export["hmat"]).max() < matrix_tol
    assert np.abs(o - export["omat"]).max() < matrix_tol
    # The SPECTRUM does not inherit it: 3.8e-15 even at apword=2, because the
    # conditioning shows up as a near-null-space rotation inside the APW order
    # space, which the eigenvalues are blind to.
    evalfv = np.asarray(export["evalfv"])
    w = np.asarray(ham.first_variational_eigenvalues(export, kc))
    assert np.abs(w[:len(evalfv)] - evalfv).max() < 1e-13


def test_k_derivative_matches_finite_differences_of_itself(exports):
    """jax.grad through match, the assembly, the Cholesky and eigvalsh.

    Against central differences of the SAME function, so this tests the AD
    plumbing and nothing else -- no physics reference, no convention. That
    separation is deliberate: the comparison against Elk's own momentum
    matrix below does NOT agree to machine precision, and without this
    control there would be no way to tell an AD bug from the real effect.

    Bulk Si at a generic k has no degeneracy in the low bands (checked), so
    individual eigenvalues are differentiable; at a multiplet neither the
    sorted branch nor eigh's own derivative rule would be (Phase 0b).
    """
    import jax
    import jax.numpy as jnp
    from elkjax import hamiltonian as ham

    export = exports["si_apword1"]
    kc = np.asarray(export["vkc"])
    vsig = ham.interstitial_potential_matrix(export)
    w = np.asarray(ham.first_variational_eigenvalues(export, kc, vsig=vsig))
    step = 1e-5
    for n in (0, 3, 7):
        assert min(w[n] - w[n - 1] if n else np.inf, w[n + 1] - w[n]) > 1e-3
        grad = np.asarray(jax.grad(
            lambda k: ham.first_variational_eigenvalues(export, k, vsig=vsig)[n]
        )(jnp.asarray(kc)))
        fd = np.array([
            (float(ham.first_variational_eigenvalues(
                export, kc + np.eye(3)[a] * step, vsig=vsig)[n])
             - float(ham.first_variational_eigenvalues(
                 export, kc - np.eye(3)[a] * step, vsig=vsig)[n])) / (2 * step)
            for a in range(3)])
        assert np.abs(grad - fd).max() < 1e-7


def _velocity_gap(export, bands=(0, 3, 7)):
    """max relative difference between d(eps)/dk from AD and Elk's p_nn."""
    import jax
    import jax.numpy as jnp
    from elkjax import hamiltonian as ham

    kc = np.asarray(export["vkc"])
    vsig = ham.interstitial_potential_matrix(export)
    pmat = export["_pmat"]
    out = {}
    for n in bands:
        grad = np.asarray(jax.grad(
            lambda k: ham.first_variational_eigenvalues(export, k, vsig=vsig)[n]
        )(jnp.asarray(kc)))
        p = np.array([pmat[a, n, n].real for a in range(3)])
        out[n] = np.abs(grad - p).max() / np.abs(p).max()
    return out


def test_k_derivative_and_elks_momentum_differ_by_muffin_tin_incompleteness(
        exports):
    """d(eps)/dk and <p> are NOT the same object in a finite LAPW basis.

    The Hellmann-Feynman identity v_nn = d(eps_n)/dk holds for a complete,
    k-INDEPENDENT basis. The LAPW basis is neither: H and O both depend on k
    (through `match`), so what jax.grad returns is
    v^dag (dH/dk - eps dO/dk) v, while genpmatk returns <psi|-i grad|psi>.

    Measured on bulk Si: they agree in sign and to 0.2-1.4%, and the gap is
    FLAT in rgkmax (2.336e-3, 2.339e-3, 2.340e-3 at 7, 8, 9 for band 0, while
    the eigenvalue itself converges), so it is not the plane-wave cutoff. It
    is the muffin-tin linearisation -- raising `apword` from 1 to 2, i.e.
    augmenting with du/dE as well as u, cuts it by up to 4x. That is what
    this asserts: the direction of the effect, not a tolerance.

    Practical consequence for the port, and for anyone using elkpy's own
    get_momentum_matrix as a band velocity: genpmatk is not a
    machine-precision reference for a k-derivative, which is why
    tests/test_calculation_momentum.py's own Hellmann-Feynman check needs
    rel=2e-2.
    """
    plain = _velocity_gap(exports["si_apword1"])
    augmented = _velocity_gap(exports["si_apword2"])
    for n, gap in plain.items():
        assert gap < 5e-2, f"band {n}: {gap:.2e} is too large to be incompleteness"
        assert gap > 1e-4, (
            f"band {n}: {gap:.2e} -- if the two agreed to machine precision "
            "the premise of this test would be wrong, not merely its bound")
    assert augmented[7] < 0.5 * plain[7], (
        f"band 7: apword=2 gave {augmented[7]:.2e} against apword=1's "
        f"{plain[7]:.2e}; the muffin-tin attribution rests on this shrinking")
