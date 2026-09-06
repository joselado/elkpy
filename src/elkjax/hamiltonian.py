"""The first-variational LAPW Hamiltonian and overlap, in JAX.

This is Phase 1's first build step (`docs/jax_port.md` section 6): a
transcription of Elk's ``olpfv``/``hmlfv`` -- the assembly of

.. math::

    (H - \\varepsilon O)\\,v = 0

in the LAPW basis at one k-point -- checked element by element against Elk's
own matrices through the LAPW export (``patches/0013``, extended by
``patches/0014``; `docs/design.md` section 33).

What is here and what is not
----------------------------

Both matrices split into an interstitial part and one muffin-tin part per
atom.  **Only the muffin-tin halves are built here.**  The interstitial
blocks

.. math::

    O^{\\rm I}_{ij} = \\tilde\\Theta(\\mathbf G_i - \\mathbf G_j), \\qquad
    H^{\\rm I}_{ij} = V_s(\\mathbf G_i - \\mathbf G_j)
      + \\tfrac12 (\\mathbf G_i + \\mathbf k)\\cdot(\\mathbf G_j + \\mathbf k)\\,
        \\tilde\\Theta(\\mathbf G_i - \\mathbf G_j)

are taken from the export as given.  That is not laziness: :math:`V_s` is the
*interstitial Kohn-Sham potential*, which is an output of the density and
potential half of the port (Phase 2) and does not exist yet, so the
Hamiltonian's interstitial block could not be built here whatever the effort.
The characteristic function :math:`\\tilde\\Theta` alone is closed-form and
could be (``gencfun``), but building half of a block buys nothing.

The muffin-tin part of a sphere :math:`\\alpha` is, with
:math:`A_{i,(\\ell m,o)}` the APW matching coefficients (``match``, and
`elkjax.lapw`),

.. math::

    O^{\\rm MT} = A^\\dagger A, \\qquad H^{\\rm MT} = A^\\dagger Z A,

the first because Elk's APW radial functions are normalised so that the
radial overlap is the identity, and the second with

.. math::

    Z_{(\\ell_1 m_1, o),(\\ell_3 m_3, o')} = \\sum_{\\ell_2 m_2}
      \\langle Y_{\\ell_1 m_1} | R_{\\ell_2 m_2} | Y_{\\ell_3 m_3}\\rangle\\,
      h^{\\alpha}_{\\ell_2 m_2, o' \\ell_3, o \\ell_1},

where :math:`h` is Elk's ``haa`` -- the radial integral of the muffin-tin
potential (expanded in **real** harmonics :math:`R_{\\ell_2 m_2}`) between two
APW radial functions, plus the kinetic term -- and the bracket is ``gntyry``.
The angular selection rule :math:`\\ell_1 + \\ell_2 + \\ell_3` even is not
imposed by hand: ``gntyry`` is exactly zero otherwise, which is what Elk's
own ``do l2 = l0, lmaxo, 2`` stride exploits.

Local orbitals extend the basis by ``nlotot`` columns whose matching
coefficients are trivial (a local orbital is confined to one sphere), so their
blocks are the radial integrals themselves: ``oalo``/``ololo`` for the
overlap, and ``hloa``/``hlolo`` contracted with the same Gaunt array for the
Hamiltonian.

Everything below takes the radial integrals as inputs.  Building *them* from
the radial Schrodinger solutions (``genapwfr``, ``genlofr``, ``hmlrad``,
``olprad``) is the next step and needs the muffin-tin potential, i.e. Phase 2.
"""

import numpy as np

import jax.numpy as jnp

__all__ = [
    "apw_index_table",
    "lo_index_table",
    "olp_apw_apw",
    "hml_apw_apw",
    "olp_apw_lo",
    "hml_apw_lo",
    "olp_lo_lo",
    "hml_lo_lo",
    "muffin_tin_overlap",
    "muffin_tin_hamiltonian",
    "assemble_from_export",
]


def apw_index_table(apword_l, lmaxapw):
    """Elk's own ordering of the APW functions of one sphere.

    ``olpaa``/``hmlaa`` flatten the sphere's APW basis with the loop nest
    ``do l; do lm = l^2+1, (l+1)^2; do io = 1, apword(l)``, giving
    ``lmoapw(is) = sum_l apword(l) * (2l+1)`` functions.  Every contraction
    below is written against that flat index, so this table is the one place
    the ordering is stated.

    Parameters
    ----------
    apword_l : array of int, shape (lmaxapw + 1,)
        APW order per l for this species (Elk's ``apword(:, is)``).
    lmaxapw : int

    Returns
    -------
    l, lm, io : int arrays of shape (lmoapw,)
        0-based angular momentum, 0-based combined lm index
        (``l^2 + l + m``), and 0-based APW order index.
    """
    l_list, lm_list, io_list = [], [], []
    for l in range(lmaxapw + 1):
        for lm in range(l * l, (l + 1) ** 2):
            for io in range(int(apword_l[l])):
                l_list.append(l)
                lm_list.append(lm)
                io_list.append(io)
    return (np.array(l_list, dtype=int), np.array(lm_list, dtype=int),
            np.array(io_list, dtype=int))


def lo_index_table(idxlo, lorbl_is, ias):
    """The local orbitals of one sphere, and the columns they occupy.

    Local-orbital columns are numbered globally across the cell by Elk's
    ``idxlo(lm, ilo, ias)`` (``genidxlo``), 1-based within the ``nlotot``-long
    tail of the matrix, so this returns the column each one owns rather than
    assuming they are contiguous per atom.

    Returns
    -------
    lm, ilo, col : int arrays of shape (n_lo_atom,)
        0-based combined lm index, 0-based local-orbital number within the
        species, and 0-based column within the local-orbital block.
    """
    lm_list, ilo_list, col_list = [], [], []
    for ilo, l in enumerate(np.asarray(lorbl_is, dtype=int)):
        for lm in range(l * l, (l + 1) ** 2):
            lm_list.append(lm)
            ilo_list.append(ilo)
            col_list.append(int(idxlo[lm, ilo, ias]) - 1)
    return (np.array(lm_list, dtype=int), np.array(ilo_list, dtype=int),
            np.array(col_list, dtype=int))


def _apw_matrix(apwalm, ias, table):
    """A, shape (lmoapw, ngp): the sphere's matching coefficients, flattened
    into ``apw_index_table``'s order.  ``apwalm`` is Elk's own array,
    (ngp, apwordmax, lmmaxapw, natmtot)."""
    l, lm, io = table
    return jnp.transpose(apwalm[:, io, lm, ias])


def olp_apw_apw(apwalm, ias, table):
    """The APW-APW muffin-tin overlap of one sphere, :math:`A^\\dagger A`.

    This is ``olpaa``.  No radial integral appears because Elk normalises the
    APW radial functions on the sphere; the whole content is the matching
    coefficients.
    """
    a = _apw_matrix(apwalm, ias, table)
    return jnp.conjugate(a).T @ a


def _gaunt_contract(gntyry, radial, lm_row, lm_col):
    """sum over lm2 of ``gntyry[lm2, lm_col, lm_row] * radial[lm2, row, col]``.

    The product is elementwise and **unconjugated** -- Fortran's
    ``sum(gntyry(lma:lmb, lm3, lm1) * haa(lma:lmb, ...))`` -- so this is not
    an inner product and ``vdot`` would be wrong.
    """
    g = gntyry[:, lm_col[None, :], lm_row[:, None]]
    return jnp.sum(g * radial, axis=0)


def hml_apw_apw(apwalm, ias, table, haa, gntyry):
    """The APW-APW muffin-tin Hamiltonian of one sphere,
    :math:`A^\\dagger Z A` (``hmlaa``).

    Elk skips any :math:`Z` element with
    ``|Re z| + |Im z| <= 1e-12`` (``hmlaa.f90``'s ``zaxpy`` guard), so an
    element-wise comparison against its output has a floor of that order and
    not of machine precision.  The dense contraction here is the more accurate
    of the two.
    """
    l, lm, io = table
    radial = haa[:, io[None, :], l[None, :], io[:, None], l[:, None], ias]
    z = _gaunt_contract(gntyry, radial, lm, lm)
    a = _apw_matrix(apwalm, ias, table)
    return jnp.conjugate(a).T @ (z @ a)


def olp_apw_lo(apwalm, ias, lo_table, oalo, nlotot):
    """The APW-local-orbital overlap block, shape (ngp, nlotot) (``olpalo``).

    :math:`O_{\\mathbf G, (\\ell m, i_{\\rm lo})} =
    \\sum_o \\langle u_{o\\ell} | u^{\\rm lo}_{i}\\rangle\\,
    \\overline{A_{\\mathbf G, (\\ell m, o)}}` -- note ``oalo`` is real, so
    only the matching coefficient is conjugated.
    """
    lm, ilo, col = lo_table
    ngp = apwalm.shape[0]
    sel = jnp.conjugate(apwalm[:, :, lm, ias])          # (ngp, apword, n_lo)
    weight = jnp.asarray(oalo[:, ilo, ias])             # (apword, n_lo)
    contrib = jnp.einsum("gon,on->gn", sel, weight)
    return jnp.zeros((ngp, nlotot), dtype=contrib.dtype).at[:, col].add(contrib)


def hml_apw_lo(apwalm, ias, table, lo_table, hloa, gntyry, nlotot):
    """The APW-local-orbital Hamiltonian block, shape (ngp, nlotot)
    (``hmlalo``).

    ``hmlalo`` adds ``conjg(z1 * apwalm)`` -- the conjugation is of the whole
    product, not of the matching coefficient alone, because what the routine
    actually forms is the Hermitian conjugate of
    :math:`\\langle {\\rm lo} | H | {\\rm APW}\\rangle`.  Contrast
    ``olp_apw_lo``, whose weight is real and where the distinction does not
    arise.
    """
    l, lm, io = table
    lm_lo, ilo, col = lo_table
    radial = hloa[:, io[None, :], l[None, :], ilo[:, None], ias]
    z = _gaunt_contract(gntyry, radial, lm_lo, lm)      # (n_lo, lmoapw)
    a = _apw_matrix(apwalm, ias, table)                 # (lmoapw, ngp)
    contrib = jnp.conjugate(jnp.transpose(z @ a))       # (ngp, n_lo)
    ngp = apwalm.shape[0]
    return jnp.zeros((ngp, nlotot), dtype=contrib.dtype).at[:, col].add(contrib)


def _hermitise_evaluated_half(contrib, col):
    """Keep only the entries Elk evaluates, and fill the rest by Hermiticity.

    ``olplolo``/``hmllolo`` run ``do jlo; do ilo = 1, jlo`` and then
    ``if (i > j) cycle``, so they touch a local-orbital pair in ONE order
    only.  Since ``genidxlo`` numbers the columns in increasing ``(ilo, lm)``,
    that set is exactly the matrix upper triangle ``col_i <= col_j``, and the
    other half of the matrix is its Hermitian conjugate rather than a second
    evaluation.

    For the overlap this is a formality: ``ololo`` is
    :math:`\\int u^{\\rm lo}_i u^{\\rm lo}_j r^2 dr`, symmetric by
    inspection.  For the Hamiltonian it is not.  ``hmlrad.f90``'s
    :math:`\\ell_2 = 0` element of ``hlolo`` is

    .. math::

        \\int u^{\\rm lo}_i \\, (\\hat H u^{\\rm lo}_j)\\, r^2 dr ,

    built with no symmetrisation and no kinetic surface term -- unlike
    ``haa``, whose counterpart is explicitly averaged over the two orderings
    and whose transpose is explicitly assigned.  Two local orbitals with the
    same :math:`\\ell` and different linearisation energies therefore give
    genuinely different numbers in the two orderings, and using the wrong one
    is a silent, plausible-looking error.  Measured on monolayer h-BN, whose
    nitrogen carries two :math:`\\ell = 0` local orbitals: the two orderings
    differ by 1.3e-2 Ha, which reaches the assembled H as 3.7e-3 Ha and the
    first-variational eigenvalues as 4.3e-7 Ha.  Bulk silicon cannot see it --
    its local orbitals are one s and one p, so no pair shares an
    :math:`\\ell`.
    """
    keep = jnp.asarray(col[:, None] <= col[None, :])
    return jnp.where(keep, contrib, jnp.conjugate(contrib).T)


def olp_lo_lo(ias, lo_table, ololo, nlotot):
    """The local-orbital-local-orbital overlap block, shape
    (nlotot, nlotot) (``olplolo``).

    Diagonal in :math:`\\ell m` -- two local orbitals overlap only if they
    carry the same angular function -- with the radial overlap
    :math:`\\langle u^{\\rm lo}_i | u^{\\rm lo}_j \\rangle` as the value.
    """
    lm, ilo, col = lo_table
    same_lm = (lm[:, None] == lm[None, :])
    value = jnp.asarray(ololo[ilo[:, None], ilo[None, :], ias])
    contrib = jnp.where(jnp.asarray(same_lm), value, 0.0).astype(complex)
    contrib = _hermitise_evaluated_half(contrib, col)
    out = jnp.zeros((nlotot, nlotot), dtype=contrib.dtype)
    return out.at[jnp.asarray(col)[:, None], jnp.asarray(col)[None, :]].add(
        contrib)


def hml_lo_lo(ias, lo_table, hlolo, gntyry, nlotot):
    """The local-orbital-local-orbital Hamiltonian block, shape
    (nlotot, nlotot) (``hmllolo``).

    Unlike the overlap this is not diagonal in :math:`\\ell m`: the muffin-tin
    potential's own non-spherical harmonics :math:`R_{\\ell_2 m_2}` couple
    different :math:`\\ell m` of the same sphere, which is exactly what the
    Gaunt contraction carries.

    ``hlolo`` is NOT symmetric under exchanging the two local orbitals, so
    only the half Elk evaluates may be used; see
    ``_hermitise_evaluated_half``.
    """
    lm, ilo, col = lo_table
    radial = hlolo[:, ilo[None, :], ilo[:, None], ias]
    contrib = _gaunt_contract(gntyry, radial, lm, lm)
    contrib = _hermitise_evaluated_half(contrib, col)
    out = jnp.zeros((nlotot, nlotot), dtype=contrib.dtype)
    return out.at[jnp.asarray(col)[:, None], jnp.asarray(col)[None, :]].add(
        contrib)


def _tables(export):
    """Per-atom (apw_index_table, lo_index_table), from a parsed export."""
    idxis = np.asarray(export["idxis"]) - 1
    apword = np.asarray(export["apword"])
    idxlo = np.asarray(export["idxlo"])
    lorbl = export["lorbl"]
    out = []
    for ias in range(int(export["natmtot"])):
        is_ = int(idxis[ias])
        out.append((apw_index_table(apword[:, is_], int(export["lmaxapw"])),
                    lo_index_table(idxlo, lorbl[is_], ias)))
    return out


def muffin_tin_overlap(export):
    """The full muffin-tin contribution to O, shape (nmatp, nmatp).

    Summed over spheres, with the local-orbital blocks in place; the
    interstitial part is NOT included (see the module docstring).
    """
    apwalm = jnp.asarray(export["apwalm"])
    ngp, nlotot = int(export["ngp"]), int(export["nlotot"])
    oalo, ololo = export["oalo"], export["ololo"]
    aa = jnp.zeros((ngp, ngp), dtype=complex)
    al = jnp.zeros((ngp, nlotot), dtype=complex)
    ll = jnp.zeros((nlotot, nlotot), dtype=complex)
    for ias, (table, lo_table) in enumerate(_tables(export)):
        aa = aa + olp_apw_apw(apwalm, ias, table)
        if lo_table[0].size:
            al = al + olp_apw_lo(apwalm, ias, lo_table, oalo, nlotot)
            ll = ll + olp_lo_lo(ias, lo_table, ololo, nlotot)
    return jnp.block([[aa, al], [jnp.conjugate(al).T, ll]])


def muffin_tin_hamiltonian(export):
    """The full muffin-tin contribution to H, shape (nmatp, nmatp)."""
    apwalm = jnp.asarray(export["apwalm"])
    ngp, nlotot = int(export["ngp"]), int(export["nlotot"])
    gntyry = jnp.asarray(export["gntyry"])
    haa, hloa, hlolo = (jnp.asarray(export[k])
                        for k in ("haa", "hloa", "hlolo"))
    aa = jnp.zeros((ngp, ngp), dtype=complex)
    al = jnp.zeros((ngp, nlotot), dtype=complex)
    ll = jnp.zeros((nlotot, nlotot), dtype=complex)
    for ias, (table, lo_table) in enumerate(_tables(export)):
        aa = aa + hml_apw_apw(apwalm, ias, table, haa, gntyry)
        if lo_table[0].size:
            al = al + hml_apw_lo(apwalm, ias, table, lo_table, hloa, gntyry,
                                 nlotot)
            ll = ll + hml_lo_lo(ias, lo_table, hlolo, gntyry, nlotot)
    return jnp.block([[aa, al], [jnp.conjugate(al).T, ll]])


def assemble_from_export(export):
    """(H, O) at the exported k-point, muffin-tin blocks built here and the
    interstitial blocks taken from Elk.

    The interstitial contribution occupies the APW-APW corner only -- a local
    orbital vanishes outside its sphere -- so it is padded rather than added
    everywhere.
    """
    nmatp = int(export["nmatp"])
    ngp = int(export["ngp"])

    def _pad(block):
        return jnp.zeros((nmatp, nmatp), dtype=complex).at[:ngp, :ngp].add(
            jnp.asarray(block))

    return (muffin_tin_hamiltonian(export) + _pad(export["hmat_istl"]),
            muffin_tin_overlap(export) + _pad(export["omat_istl"]))
