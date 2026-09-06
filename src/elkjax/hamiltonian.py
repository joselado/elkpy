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

The last section of this module closes the eigenproblem: the Cholesky
reduction :math:`\\tilde H=L^{-1}HL^{-\\dagger}` that Elk's own ``eveqnfv``
makes through ``zhegv``, the tolerance
:math:`\\epsilon\\,\\kappa(O)\\,\\lVert\\tilde H\\rVert` measured on
*this* run, and the occupied projector built through
`elkjax.projector`'s safe-:math:`K` rule.  That last piece is not a
convenience: an individual eigenvalue's derivative does not exist inside a
multiplet, and a real ground state puts one there by symmetry -- bulk silicon
at :math:`\\Gamma` carries the three-fold :math:`\\Gamma_{25'}` valence level
inside its occupied window.  The projector is the object that survives, and
Phase 1f measures that it does (`docs/jax_port_phase1.md`).
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
    "interstitial_potential_matrix",
    "matching_coefficients",
    "eigenproblem_at",
    "cholesky_reduce",
    "first_variational_eigenvalues",
    "projector_tolerance",
    "occupied_window",
    "occupied_projector",
    "smeared_occupied_projector",
    "fixed_number_occupied_projector",
    "elk_occupied_projector",
    "occupied_band_count",
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


# ---------------------------------------------------------------------------
# The k-dependent pipeline
# ---------------------------------------------------------------------------
#
# Everything above takes `apwalm` from the export and so is frozen at one
# k-point.  The functions below rebuild it with `elkjax.lapw.match` instead,
# which makes the whole assembly a differentiable function of k -- the first
# thing in the port that is.  Two observations make it possible without any
# new Fortran:
#
#   * The overlap's interstitial block IS the characteristic function,
#     O^I_ij = Theta(G_i - G_j), and Theta does not depend on k at all.
#   * The Hamiltonian's is
#     H^I_ij = V_s(G_i - G_j) + 1/2 (G_i+k).(G_j+k) Theta(G_i - G_j),
#     so V_s -- the one ingredient that belongs to Phase 2 -- can be RECOVERED
#     as a matrix from the two exported blocks at the exported k, and then
#     held fixed while k varies.  No Fourier index mapping is needed: the
#     subtraction is elementwise in (i, j).
#
# The radial integrals and the Gaunt array are k-independent, so the only
# k-dependence left is through `match` (item 0c, already checked against Elk
# element-wise) and that explicit kinetic term.


def interstitial_potential_matrix(export):
    """V_s(G_i - G_j) as a matrix, recovered from the exported blocks.

    ``hmlistl`` computes ``vsig(ig) + (G_i+k).(G_j+k)/2 * cfunig(ig)`` and
    ``olpistl`` computes ``cfunig(ig)`` for the same ``ig``, so subtracting
    the second (scaled) from the first leaves ``vsig`` with no need to know
    which pair (i, j) maps to which G-vector difference.

    This is a way of holding a Phase 2 quantity fixed, not of computing it:
    :math:`V_s` is the interstitial Kohn-Sham potential and follows the
    density.  It is exactly right for varying k at a fixed ground state,
    which is what a band velocity is.
    """
    ngp = int(export["ngp"])
    gk = np.asarray(export["vgpc"])[:, :ngp]                 # (3, ngp), G + k0
    kinetic = 0.5 * (gk.T @ gk)
    return np.asarray(export["hmat_istl"]) - kinetic * np.asarray(
        export["omat_istl"])


def _reciprocal_vectors(export):
    """The pure G-vectors of the basis, i.e. the exported G+k minus k."""
    ngp = int(export["ngp"])
    return (np.asarray(export["vgpc"])[:, :ngp]
            - np.asarray(export["vkc"])[:, None]).T            # (ngp, 3)


def matching_coefficients(export, vgkc):
    """`apwalm` at an arbitrary k, in Elk's own array layout.

    Rebuilds it through `elkjax.lapw.match` -- verified element-wise against
    Elk's array by patch 0013 -- rather than taking the exported one, so the
    result is differentiable in k.  The G-vector SET is held fixed: it is a
    property of the cutoff and changes discontinuously with k, which is not a
    problem for a derivative at a point but does mean this is only valid for
    k near the exported one.
    """
    from . import lapw

    idxis = np.asarray(export["idxis"]) - 1
    atposc = np.asarray(export["atposc"])
    rmt = np.asarray(export["rmt"])
    omega = float(export["omega"])
    lmaxapw = int(export["lmaxapw"])
    per_atom = []
    for ias in range(int(export["natmtot"])):
        dmat = [jnp.asarray(m, dtype=complex) for m in export["dmat"][ias]]
        # `export` may carry a REBUILT dmat (elkjax.radial_functions), which is
        # how a perturbed potential reaches the matching coefficients.
        per_atom.append(lapw.match(
            lmaxapw, vgkc, jnp.asarray(atposc[:, ias]), dmat,
            float(rmt[int(idxis[ias])]), omega))
    return jnp.stack(per_atom, axis=-1)     # (ngp, ordmax, lmmaxapw, natmtot)


def eigenproblem_at(export, kc, vsig=None):
    """(H, O) at an arbitrary Cartesian k-point, built entirely here.

    Only the ground state is imported: the radial integrals, the Gaunt array,
    the recovered :math:`V_s` and the characteristic function are all
    k-independent and come from the export, while `apwalm` and the kinetic
    term are rebuilt.  At ``kc = export["vkc"]`` this must reproduce Elk's own
    matrices, which is the forward check that makes the derivative meaningful.
    """
    if vsig is None:
        vsig = interstitial_potential_matrix(export)
    gvec = jnp.asarray(_reciprocal_vectors(export))
    vgkc = gvec + jnp.asarray(kc)[None, :]
    apwalm = matching_coefficients(export, vgkc)
    local = dict(export)
    local["apwalm"] = apwalm
    kinetic = 0.5 * (vgkc @ vgkc.T)
    istl_h = jnp.asarray(vsig) + kinetic * jnp.asarray(export["omat_istl"])
    nmatp, ngp = int(export["nmatp"]), int(export["ngp"])

    def _pad(block):
        return jnp.zeros((nmatp, nmatp), dtype=complex).at[:ngp, :ngp].add(block)

    return (muffin_tin_hamiltonian(local) + _pad(istl_h),
            muffin_tin_overlap(local) + _pad(jnp.asarray(export["omat_istl"])))


def cholesky_reduce(h, o):
    r"""Reduce :math:`Hv=\varepsilon Ov` to a standard problem, Elk's own route.

    Returns :math:`(\tilde H, L)` with :math:`O=LL^\dagger` and
    :math:`\tilde H=L^{-1}HL^{-\dagger}`, which is what LAPACK's ``zhegv``
    does inside `eveqnfv`.  Eigenvectors map back as :math:`c=L^{-\dagger}y`,
    and conversely a normalised Elk eigenvector (:math:`c^\dagger Oc=1`)
    becomes the orthonormal :math:`y=L^\dagger c` -- which is how the
    projector below is compared against Elk's own `evecfv`.

    Both solves are triangular in exact arithmetic but are written as general
    ``solve`` calls because that is what JAX differentiates without a
    structured-matrix rule; the cost is irrelevant at Phase 1 sizes.
    """
    chol = jnp.linalg.cholesky(o)
    reduced = jnp.linalg.solve(chol, jnp.linalg.solve(chol, h).conj().T).conj().T
    return reduced, chol


def first_variational_eigenvalues(export, kc, vsig=None):
    """The first-variational spectrum at an arbitrary Cartesian k.

    Differentiable in ``kc``.  An INDIVIDUAL eigenvalue's derivative is only
    meaningful where that eigenvalue is non-degenerate -- at a multiplet
    ``eigh``'s own derivative rule divides by a vanishing gap (Phase 0b) and
    the sorted branch is not differentiable at all -- so a caller wanting a
    degenerate group must differentiate its trace, or use
    :func:`occupied_projector`, which is well defined there.
    """
    reduced, _ = cholesky_reduce(*eigenproblem_at(export, kc, vsig=vsig))
    return jnp.linalg.eigvalsh(reduced)


# ---------------------------------------------------------------------------
# The occupied projector, and the tolerance it needs
# ---------------------------------------------------------------------------
#
# `first_variational_eigenvalues` closes with a plain `eigvalsh`, which is the
# route Phase 0b showed is unsafe the moment two eigenvalues approach: JAX's
# own rule differentiates the eigenVECTORS and divides by the gap before the
# occupation difference can cancel against it.  The occupied projector
#
#     P = sum_{i < nocc} |i><i|
#
# is the object that IS well defined at an enclosed multiplet -- it does not
# depend on how `eigh` resolves the degenerate subspace -- and
# `projector.hard_window_projector` carries the rule that keeps it that way.
# The two had never met: the rule was only ever exercised on a synthetic
# overlap with a PRESCRIBED condition number, which `docs/continue_here.md` §3
# flags as the biggest hole left in Phase 0.
#
# What real matrices supply that a synthetic pair cannot is the tolerance.  It
# is a property of the run -- kappa(O) is set by the plane-wave cutoff, not by
# the matrix size (Phase 0b(ii)) -- so it is measured here rather than passed
# in, and it is measured on the REDUCED matrix, whose norm is about 3x |H|.


def projector_tolerance(reduced, o):
    r""":math:`\epsilon\,\kappa(O)\,\lVert\tilde H\rVert_2`, measured on this run.

    The scale below which two computed eigenvalues of the reduced problem
    cannot be told apart, and therefore the only defensible threshold for
    "degenerate" in the divided-difference kernel (study §8(b)).  Three
    choices here are conclusions of Phase 0b(ii), not preferences:

    * :math:`\kappa(O)` comes from a dense ``eigvalsh``.  §8(b)'s cheap
      Cholesky-diagonal estimate is **uninformative** on real overlaps -- it
      moved 8.05 to 9.25 while the truth moved over a 74-fold range -- so it
      cannot set a threshold.
    * the norm is :math:`\lVert L^{-1}HL^{-\dagger}\rVert`, not
      :math:`\lVert H\rVert`; the reduced matrix is the one actually
      diagonalised and is consistently the larger.
    * it is recomputed per run, because :math:`\kappa(O)` is a **cutoff**
      property: ``rgkmax`` 7 to 9 takes bulk Si from 5e3 to 2.1e5 while the
      matrix size is nearly irrelevant.

    Host-side and non-differentiable by construction: a threshold that moved
    with the parameter would make the kernel's branch a function of the
    perturbation.
    """
    eigenvalues = np.linalg.eigvalsh(np.asarray(o))
    kappa = float(eigenvalues.max() / eigenvalues.min())
    return float(np.finfo(np.float64).eps * kappa
                 * np.linalg.norm(np.asarray(reduced), 2))


def occupied_window(export, kc, nocc, vsig=None, tol=None):
    r"""Refuse an occupied window whose boundary is not resolvably gapped.

    Returns ``(tol, gap)``.  Raises ``ValueError`` when
    :math:`\varepsilon_{n_{\rm occ}}-\varepsilon_{n_{\rm occ}-1}` is at or
    below ``tol``, because there is then no differentiable occupied subspace
    at all -- as mathematics, not as numerics -- and the honest answer is a
    refusal rather than a number.  :func:`occupied_projector`'s in-graph
    ``NaN`` is the safety net for a caller who skips this; a ``NaN`` is still
    a returned non-number, and the study's Phase 1 criterion asks for an
    explicit refusal.

    This is the same shape as `elkpy.parsers.symmetry.check_window_gap` and
    `elkjax.projector.check_sign_window`: host-side, before any tracing, since
    a traced predicate cannot raise.
    """
    h, o = eigenproblem_at(export, kc, vsig=vsig)
    reduced, _ = cholesky_reduce(h, o)
    if tol is None:
        tol = projector_tolerance(reduced, o)
    evals = np.asarray(jnp.linalg.eigvalsh(reduced))
    if not 0 < nocc < evals.size:
        raise ValueError(
            f"nocc={nocc} is not a proper window of {evals.size} states")
    gap = float(evals[nocc] - evals[nocc - 1])
    if gap <= tol:
        raise ValueError(
            f"occupied-window boundary gap {gap:.3e} Ha is at or below the "
            f"resolution {tol:.3e} Ha of this run (eps kappa(O) |H~|): the "
            f"occupied subspace is not differentiable here. Window the whole "
            f"degenerate group together instead.")
    return tol, gap


def occupied_projector(export, kc, nocc, tol, vsig=None):
    r""":math:`\tilde P=\sum_{i<n_{\rm occ}}|y_i\rangle\langle y_i|`, differentiable in ``kc``.

    In the Cholesky-REDUCED basis, where the projector is an honest orthogonal
    one; Elk's own eigenvectors map into it as :math:`y=L^\dagger c`.  Built
    through :func:`elkjax.projector.hard_window_projector`, so JAX never sees
    the eigenvector derivative and an enclosed multiplet contributes exactly
    zero to :math:`dP` instead of a ratio of two rounding errors.

    ``tol`` is required rather than defaulted: it must come from
    :func:`occupied_window`, which also performs the refusal, and defaulting
    it here would let a caller differentiate through an ungapped boundary
    while believing a threshold had been checked.
    """
    from .projector import hard_window_projector

    reduced, _ = cholesky_reduce(*eigenproblem_at(export, kc, vsig=vsig))
    return hard_window_projector(reduced, nocc, tol)


def smeared_occupied_projector(export, kc, mu, width, tol, vsig=None):
    r""":math:`\tilde P=\sum_i f(\varepsilon_i-\mu)\,|y_i\rangle\langle y_i|` at fixed :math:`\mu`.

    The smeared sibling of :func:`occupied_projector`.  Fermi-Dirac is Elk's own
    ``stype=3`` and ``width`` is its ``swidth``.

    ``tol`` is accepted so the two signatures stay parallel but is **inert**:
    :func:`elkjax.projector.smeared_projector` carries the cancellation-free closed form
    of the kernel (`docs/jax_port_phase1.md` §1i), which is exact at every splitting, so
    nothing is left for a threshold to select.  It remains load-bearing for
    :func:`occupied_projector`, whose occupation is a step rather than a smooth function
    of the eigenvalue and which therefore has no closed form to reach for.
    """
    from .projector import smeared_projector

    reduced, _ = cholesky_reduce(*eigenproblem_at(export, kc, vsig=vsig))
    return smeared_projector(reduced, mu, width, tol)


def fixed_number_occupied_projector(export, kc, nelec, width, tol, vsig=None):
    r"""The same at fixed **electron number**, with :math:`\mu` solved for.

    ``nelec`` counts states, not electrons: Elk's ``occmax`` is 2 without spin-orbit,
    so a cell with 8 valence electrons has ``nelec=4`` here.  The self-consistent Fermi
    level's derivative is :func:`elkjax.projector.fermi_level`'s ``custom_jvp``; this is
    the composition, so the chain rule supplies the extra term.  ``tol`` is inert here
    for the same reason as in :func:`smeared_occupied_projector`.
    """
    from .projector import fixed_number_projector

    reduced, _ = cholesky_reduce(*eigenproblem_at(export, kc, vsig=vsig))
    return fixed_number_projector(reduced, nelec, width, tol)


def elk_occupied_projector(export, nocc):
    r"""The same projector built from Elk's own ``evecfv``, for the forward check.

    :math:`Y=L^\dagger C` is orthonormal because Elk normalises
    :math:`C^\dagger OC=\mathbb 1`, so :math:`YY^\dagger` is the reduced-basis
    projector onto exactly the subspace Elk's eigenvectors span.  Comparing
    *projectors* rather than coefficients is what makes the check meaningful
    at all: ``evecfv`` is arbitrary within any degenerate multiplet (study
    §8(b), hazard C), and bulk Si at :math:`\Gamma` has a three-fold one
    inside the occupied window.
    """
    chol = np.linalg.cholesky(np.asarray(export["omat"]))
    y = chol.conj().T @ np.asarray(export["evecfv"])[:, :nocc]
    return y @ y.conj().T


def occupied_band_count(evalfv, efermi):
    """How many first-variational bands lie below the Fermi level.

    The export carries no electron count, and assuming one is a trap this
    project has already paid for elsewhere (`docs/design.md` §13: core states
    are not among the bands ``nstsv`` indexes).  Counting Elk's own
    eigenvalues against Elk's own ``EFERMI.OUT`` is exact for a gapped system
    at any k-point and needs no chemistry.
    """
    return int(np.sum(np.asarray(evalfv) < float(efermi)))
