"""Synthetic pins for the LAPW assembly (src/elkjax/hamiltonian.py).

tests/test_calculation_lapw_assembly.py checks the whole thing against a real
Elk run; this checks the pieces that are pure index bookkeeping, needs no
binary, and is where a convention gets stated once rather than inferred from
a passing end-to-end comparison.

Two of these pin a bug class the end-to-end test found the hard way and the
other cannot localise:

  * `_hermitise_evaluated_half`, because Elk evaluates a local-orbital pair
    in ONE order and hlolo is not symmetric under exchanging them;
  * `_gaunt_contract`, because Fortran's `sum(gntyry * haa)` is an
    UNCONJUGATED elementwise product -- an inner product would be a plausible
    reading and is wrong.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import elkjax                                   # noqa: E402  (sets x64 first)
from elkjax import hamiltonian as ham           # noqa: E402


def test_apw_index_table_is_elks_loop_nest():
    """`do l = 0, lmaxapw; do lm = l^2+1, (l+1)^2; do io = 1, apword(l)`.

    Written out for lmaxapw = 1 with a DIFFERENT order per l, which is the
    case an `apword`-is-a-scalar misreading would get wrong and a uniform
    apword would hide.
    """
    l, lm, io = ham.apw_index_table(np.array([2, 1]), lmaxapw=1)
    assert l.tolist() == [0, 0, 1, 1, 1]
    assert lm.tolist() == [0, 0, 1, 2, 3]
    assert io.tolist() == [0, 1, 0, 0, 0]
    assert len(l) == 2 * 1 + 1 * 3          # lmoapw = sum_l apword(l)(2l+1)


def test_lo_index_table_reproduces_genidxlo():
    """genidxlo numbers columns in increasing (ias, ilo, lm), 1-based.

    The table returns 0-based columns, and -- the property
    `_hermitise_evaluated_half` relies on -- they increase with ilo, so
    "ilo <= jlo and i <= j" is exactly "col_i <= col_j".
    """
    lorbl = np.array([0, 1])                 # one s and one p local orbital
    idxlo = np.zeros((4, 2, 1), dtype=int)
    i = 0
    for ilo, l in enumerate(lorbl):
        for lm in range(l * l, (l + 1) ** 2):
            i += 1
            idxlo[lm, ilo, 0] = i
    lm, ilo, col = ham.lo_index_table(idxlo, lorbl, ias=0)
    assert lm.tolist() == [0, 1, 2, 3]
    assert ilo.tolist() == [0, 1, 1, 1]
    assert col.tolist() == [0, 1, 2, 3]
    assert np.all(np.diff(ilo) >= 0) and np.all(np.diff(col) > 0)


def test_hermitise_evaluated_half_keeps_the_upper_triangle():
    """Elk's half, not the average and not the other half.

    Built on a matrix that is deliberately NOT Hermitian, so all three
    readings give different answers: taking the input as it stands, taking
    its lower half, or symmetrising.
    """
    contrib = np.array([[1.0 + 0.0j, 2.0], [5.0, 3.0 + 0.0j]])
    col = np.array([0, 1])
    out = np.asarray(ham._hermitise_evaluated_half(contrib, col))
    assert np.allclose(out, [[1.0, 2.0], [2.0, 3.0]])
    assert np.allclose(out, out.conj().T)
    assert not np.allclose(out, contrib)             # not the raw input
    assert not np.allclose(out, 0.5 * (contrib + contrib.conj().T))


def test_gaunt_contract_is_unconjugated():
    """sum_lm2 g[lm2, lm_col, lm_row] * radial[lm2, row, col].

    Checked against an explicit loop, and separately shown to DIFFER from the
    conjugated (inner-product) reading -- which for a complex Gaunt array is
    the failure a Hermiticity check would not catch, since both readings give
    a Hermitian result when the radial array is symmetric.
    """
    rng = np.random.default_rng(0)
    nlm2, nlm = 5, 3
    gnt = rng.normal(size=(nlm2, nlm, nlm)) + 1j * rng.normal(size=(nlm2, nlm, nlm))
    radial = rng.normal(size=(nlm2, 2, 2))
    lm_row, lm_col = np.array([0, 2]), np.array([1, 2])
    out = np.asarray(ham._gaunt_contract(gnt, radial, lm_row, lm_col))
    expected = np.array([[sum(gnt[lm2, lm_col[j], lm_row[i]] * radial[lm2, i, j]
                              for lm2 in range(nlm2))
                          for j in range(2)] for i in range(2)])
    assert np.allclose(out, expected)
    conjugated = np.array([[sum(np.conj(gnt[lm2, lm_col[j], lm_row[i]]) * radial[lm2, i, j]
                                for lm2 in range(nlm2))
                            for j in range(2)] for i in range(2)])
    assert not np.allclose(out, conjugated)


def test_apw_overlap_is_the_gram_matrix_of_the_matching_coefficients():
    """O^MT = A^dagger A, with A flattened in Elk's own (l, lm, io) order.

    No radial integral appears because Elk normalises the APW radial
    functions on the sphere; the whole content of olpaa is the matching
    coefficients, and this states that rather than leaving it implicit in an
    end-to-end number.
    """
    rng = np.random.default_rng(1)
    ngp, apwordmax, lmmaxapw = 6, 2, 4
    apwalm = (rng.normal(size=(ngp, apwordmax, lmmaxapw, 1))
              + 1j * rng.normal(size=(ngp, apwordmax, lmmaxapw, 1)))
    table = ham.apw_index_table(np.array([2, 1]), lmaxapw=1)
    out = np.asarray(ham.olp_apw_apw(jax.numpy.asarray(apwalm), 0, table))
    l, lm, io = table
    a = np.stack([apwalm[:, io[i], lm[i], 0] for i in range(len(l))])
    assert np.allclose(out, a.conj().T @ a)
    assert np.allclose(out, out.conj().T)
