"""Parse responses from the elkpy interactive eigenstate/overlap session
(task 9002, src/elkpy_eigenstates.f90 -- patches/0003-eigenstate-session.patch).

Pure token-list parsing, no subprocess/file dependency, mirroring
parsers/berry.py's style -- session.py owns reading lines from the running
elk process and handing the accumulated tokens to the functions here, so
these are independently unit-testable (tests/test_parsers_eigenstates.py)
without needing a live Elk process.
"""

import numpy as np


def _take(tokens, pos, n, cast):
    vals = [cast(t) for t in tokens[pos : pos + n]]
    return vals, pos + n


def parse_eigenstates_response(tokens):
    """Parse the token stream of an EIGENSTATES response: nstsv, then nstsv
    eigenvalues (Hartree), then the nstsv x nstsv evecsv matrix as real/imag
    pairs.

    Returns (energies, evecsv): energies shape (nstsv,), evecsv shape
    (nstsv, nstsv) complex -- evecsv[:, i] is the i-th second-variational
    eigenvector's coefficients in the orthonormal spinor basis of this one
    diagonalisation. Valid for overlap-style inner products only among
    columns of this SAME array -- comparing evecsv across two different
    queries (different k, or a separate diagonalisation at the same k) is
    not meaningful this way; use EigenstateSession.overlap() for that (see
    docs/design.md #14).
    """
    pos = 0
    (nstsv,), pos = _take(tokens, pos, 1, int)
    energies, pos = _take(tokens, pos, nstsv, float)
    energies = np.array(energies)
    flat, pos = _take(tokens, pos, 2 * nstsv * nstsv, float)
    reim = np.array(flat).reshape(nstsv * nstsv, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    # Fortran wrote "do b; do a" with a innermost -- column-major, matching
    # evecsv(a, b)'s own natural storage order (same convention as
    # parsers/berry.py).
    evecsv = values.reshape(nstsv, nstsv, order="F")
    return energies, evecsv


def parse_overlap_response(tokens):
    """Parse the token stream of an OVERLAP response: nst, then the
    nst x nst overlap matrix M(a, b) = <psi_a(k_a)|psi_b(k_b)> as real/imag
    pairs (same column-major convention as parse_eigenstates_response)."""
    pos = 0
    (nst,), pos = _take(tokens, pos, 1, int)
    flat, pos = _take(tokens, pos, 2 * nst * nst, float)
    reim = np.array(flat).reshape(nst * nst, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    return values.reshape(nst, nst, order="F")


def parse_projection_response(tokens):
    """Parse the token stream of a PROJECTION response: nst, natmtot, then
    natmtot consecutive nst x nst atom-projection matrices (real/imag pairs,
    same column-major convention as parse_eigenstates_response/
    parse_overlap_response), one per atom, in Fortran's global 1-based atom
    order (species in declaration order, then atoms within each species in
    order -- see Calculation.get_forces()'s docstring for the same
    convention, and Calculation.get_atom_projection() for how to look up a
    particular atom's matrix by (species, index)).

    Returns an (natmtot, nst, nst) complex array; matrices[ias] is the
    atom-projection operator P_ias (0-based ias here, 1-based in the
    Fortran/protocol sense) -- see EigenstateSession.atom_projection() for
    what each matrix element means and its gauge caveat.
    """
    pos = 0
    (nst, natmtot), pos = _take(tokens, pos, 2, int)
    flat, pos = _take(tokens, pos, 2 * nst * nst * natmtot, float)
    reim = np.array(flat).reshape(nst * nst * natmtot, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    # Fortran wrote "do ias; do b; do a" with a innermost -- so within each
    # atom's block the layout is column-major (matching the other parsers
    # here), and the natmtot blocks themselves are consecutive.
    return values.reshape(nst, nst, natmtot, order="F").transpose(2, 0, 1)


def parse_orbital_projection_response(tokens):
    """Parse the token stream of an ORBITAL response: nst, natmtot, nl (always
    4: s, p, d, f), then natmtot*nl consecutive nst x nst orbital-projection
    matrices (real/imag pairs, same column-major convention as the other
    parsers here), in Fortran's "do ias; do lsel" nesting -- atom-major,
    l-minor (l=0..3 i.e. s,p,d,f, in that order) within each atom's block --
    same global 1-based atom order as parse_projection_response.

    Returns an (natmtot, 4, nst, nst) complex array; matrices[ias, l] is the
    l-resolved atom-projection operator P_{ias,l} (0-based ias/l here) --
    see EigenstateSession.orbital_projection() for what each matrix element
    means and its gauge caveat (same as atom_projection()'s).
    """
    pos = 0
    (nst, natmtot, nl), pos = _take(tokens, pos, 3, int)
    flat, pos = _take(tokens, pos, 2 * nst * nst * nl * natmtot, float)
    reim = np.array(flat).reshape(nst * nst * nl * natmtot, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    # column-major within each (ias, l) block, blocks consecutive with l
    # fastest-varying after (a, b) -- i.e. Fortran's "do ias; do lsel; do b;
    # do a" nesting, a innermost.
    return values.reshape(nst, nst, nl, natmtot, order="F").transpose(3, 2, 0, 1)


def parse_momentum_response(tokens):
    """Parse the token stream of a MOMENTUM response: nstsv, then nstsv
    eigenvalues (Hartree), then the nstsv x nstsv evecsv matrix, then the
    three nstsv x nstsv Cartesian momentum matrices p^x, p^y, p^z, all as
    real/imag pairs (same column-major convention as the other parsers
    here, Cartesian component slowest-varying).

    The leading nstsv/eigenvalues/evecsv block is deliberately byte-for-byte
    the shape parse_eigenstates_response() reads, with the momentum
    components appended -- a MOMENTUM response IS an EIGENSTATES response
    plus p^a. evecsv was added by patches/0009 for the reason docs/design.md
    #24 explains: S_a is built from evecsv alone (parsers.spin), so the spin
    current operator J^z_a = (1/2){S_z, v_a} is only well defined if that
    evecsv and this pmat come from the SAME diagonalisation. Taking evecsv
    from a separate EIGENSTATES query would silently mix two arbitrary
    resolutions of any degenerate multiplet (docs/design.md #14) while
    leaving every Hermiticity and unitarity check intact.

    Returns (energies, evecsv, pmat): energies shape (nstsv,) Hartree,
    evecsv shape (nstsv, nstsv) complex (evecsv[:, i] the i-th
    eigenvector, row index i = p + (ispn-1)*nstfv), pmat shape
    (3, nstsv, nstsv) complex, pmat[a] the a-th Cartesian component
    (a = 0, 1, 2 for x, y, z) in atomic units, indexed
    pmat[a][n, m] = <psi_n|p_a|psi_m> -- i.e. the CONJUGATED (bra) state
    on the row. That index convention is not a guess: genpmatk.f90's own
    header defines P_ij = integral Psi_i^*(-i grad + ...) Psi_j, and its
    accumulation is zgemv('C', ...) into pmat(1, jst, i), which places
    the conjugated factor on the first (row) index. It matters -- a
    transposed pmat flips both the Kubo Berry curvature and the circular
    polarization together, so no internal consistency check between them
    would catch it. Unlike the projection
    responses this carries NO band window -- genpmatk's array is
    hard-dimensioned nstsv (see elkpy_momentum), and the Kubo-style sums
    this feeds need states outside any window of interest anyway.

    The energies are returned alongside deliberately: they come from the
    same diagonalisation as pmat, which is what makes them safe to use as
    the energy denominators of those sums. Pairing pmat with a separate
    EIGENSTATES query's energies would reintroduce the degenerate-subspace
    basis ambiguity of docs/design.md #14.
    """
    pos = 0
    (nstsv,), pos = _take(tokens, pos, 1, int)
    energies, pos = _take(tokens, pos, nstsv, float)
    energies = np.array(energies)
    flat, pos = _take(tokens, pos, 2 * nstsv * nstsv, float)
    reim = np.array(flat).reshape(nstsv * nstsv, 2)
    # same "do b; do a" column-major block parse_eigenstates_response reads
    evecsv = (reim[:, 0] + 1j * reim[:, 1]).reshape(nstsv, nstsv, order="F")
    flat, pos = _take(tokens, pos, 2 * 3 * nstsv * nstsv, float)
    reim = np.array(flat).reshape(3 * nstsv * nstsv, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    # Fortran wrote "do comp; do b; do a" with a innermost -- column-major
    # within each Cartesian component's block, the three blocks consecutive.
    pmat = values.reshape(nstsv, nstsv, 3, order="F").transpose(2, 0, 1)
    return energies, evecsv, pmat


def parse_parity_response(tokens):
    """Parse the token stream of a PARITY response: nst, nstsv, then nstsv
    eigenvalues (Hartree), then the nst x nst inversion-operator matrix
    P(a, b) = <psi_a|I|psi_b> as real/imag pairs (same column-major
    convention as the other parsers here).

    Returns (energies, pmat): energies shape (nstsv,) Hartree -- ALL states,
    not just the window -- and pmat shape (nst, nst) complex over the
    requested band window. The full eigenvalue list travels with the
    windowed operator (same reasoning as parse_momentum_response) so the
    caller can see the degeneracy structure around the window and confirm
    its boundary does not cut a Kramers pair.

    P is Hermitian with P^2 = 1 and eigenvalues +-1 whenever the window is
    gapped from the rest of the spectrum -- see
    EigenstateSession.parity() for why that survives the truncation here
    when the analogous angular-momentum identities (docs/design.md #19) do
    not.
    """
    pos = 0
    (nst, nstsv), pos = _take(tokens, pos, 2, int)
    energies, pos = _take(tokens, pos, nstsv, float)
    energies = np.array(energies)
    flat, pos = _take(tokens, pos, 2 * nst * nst, float)
    reim = np.array(flat).reshape(nst * nst, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    return energies, values.reshape(nst, nst, order="F")


def parse_angular_momentum_response(tokens):
    """Parse the token stream of an ANGMOM response: nst, natmtot, nl (always
    4: s, p, d, f), ncomp (always 3: x, y, z), then natmtot*nl*ncomp
    consecutive nst x nst angular-momentum matrices (real/imag pairs, same
    column-major convention as the other parsers here), in Fortran's
    "do ias; do lsel; do comp" nesting -- atom-major, then l (l=0..3 i.e.
    s,p,d,f), then Cartesian component (x,y,z) minor -- same global 1-based
    atom order as parse_projection_response.

    Returns an (natmtot, 4, 3, nst, nst) complex array; matrices[ias, l, c]
    is the l-resolved angular-momentum operator (L_x, L_y or L_z, c=0,1,2)
    for atom ias -- see EigenstateSession.angular_momentum() for what each
    matrix element means.
    """
    pos = 0
    (nst, natmtot, nl, ncomp), pos = _take(tokens, pos, 4, int)
    flat, pos = _take(tokens, pos, 2 * nst * nst * ncomp * nl * natmtot, float)
    reim = np.array(flat).reshape(nst * nst * ncomp * nl * natmtot, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    # column-major within each (ias, l, comp) block, blocks consecutive with
    # comp fastest-varying after (a, b), then l, then ias -- i.e. Fortran's
    # "do ias; do lsel; do comp; do b; do a" nesting, a innermost.
    return values.reshape(nst, nst, ncomp, nl, natmtot, order="F").transpose(
        4, 3, 2, 0, 1
    )


def parse_symlist_response(tokens):
    """Parse a SYMLIST response: nsymcrys, nspinor, then per operation its
    index, a zero-translation flag, three rows of the integer lattice
    rotation matrix, and the fractional translation.

    Returns (nspinor, ops) with ops a list of dicts
    {"isym": 1-based index, "rotation": (3,3) int array in LATTICE
    coordinates, "translation": (3,) float, "symmorphic": bool}.

    The rotation is Elk's own `symlat(:,:,lsplsymc(isym))` -- the spatial
    part of the space-group element, in lattice coordinates, so it is an
    integer matrix. Working in lattice coordinates is what lets the Python
    side test R.k == k exactly, with no tolerance on a Cartesian rotation.
    """
    pos = 0
    (nsymcrys, nspinor), pos = _take(tokens, pos, 2, int)
    ops = []
    for _ in range(nsymcrys):
        (isym, tv0), pos = _take(tokens, pos, 2, int)
        rows, pos = _take(tokens, pos, 9, int)
        trans, pos = _take(tokens, pos, 3, float)
        ops.append({
            "isym": isym,
            "rotation": np.array(rows, dtype=int).reshape(3, 3),
            "translation": np.array(trans),
            "symmorphic": bool(tv0),
        })
    return nspinor, ops


def parse_symmetry_response(tokens):
    """Parse a SYMMETRY response: identical in shape to a PARITY response
    (nst, nstsv, nstsv eigenvalues, then the nst x nst operator matrix as
    real/imag pairs, column-major).

    Returns (energies, smat) with smat[a, b] = <psi_a|O|psi_b> for the
    requested crystal symmetry -- see EigenstateSession.symmetry_operator().
    """
    return parse_parity_response(tokens)


def _upper_triangle(tokens, pos, n):
    """Read n*(n+1)/2 real/imag pairs written as `do j=1,n; do i=1,j` and
    return the full Hermitian (n, n) matrix.

    Elk fills only the upper triangle of H and O -- olpistl/hmlistl run
    `do i=1,j`, and every muffin-tin contribution added afterwards does the
    same -- so the lower triangle of the allocated array is never assigned
    and elkpy_lapwexport deliberately does not write it. Hermitising here is
    therefore reconstruction, not a convenience: taking the array as it
    stands would put uninitialised memory into every subsequent norm,
    condition number and eigenvalue.
    """
    ntri = n * (n + 1) // 2
    flat, pos = _take(tokens, pos, 2 * ntri, float)
    reim = np.array(flat).reshape(ntri, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    mat = np.zeros((n, n), dtype=complex)
    # `do j; do i=1,j` walks the upper triangle COLUMN by column, i.e. sorted
    # by (column, row). np.triu_indices sorts by (row, column) instead, so it
    # is the wrong order and silently produces a non-Hermitian matrix.
    # np.tril_indices with its two index arrays swapped is exactly the order
    # wanted: the lower triangle read row-major is (0,0),(1,0),(1,1),... whose
    # transpose is (0,0),(0,1),(1,1),... -- upper, column-major.
    rows, cols = np.tril_indices(n)
    mat[cols, rows] = values
    lower = np.tril_indices(n, -1)
    mat[lower] = np.conj(mat.T[lower])
    return mat, pos


def parse_lapw_response(tokens):
    """Parse the token stream of a LAPW response -- every ingredient of the
    first-variational LAPW eigenvalue problem at one k-point (see
    elkpy_lapwexport, patches/0013).

    Returns a dict with, in the order the Fortran writes them:

      ngp, nlotot, nmatp, nstfv, apwordmax, lmmaxapw, natmtot, nspecies,
      lmaxapw, npapw, nmatmax   -- ints, the shapes
      omega                     -- unit cell volume (Bohr^3)
      avec, bvec                -- (3, 3), columns are the lattice /
                                   reciprocal lattice vectors (Elk's own
                                   avec(:, j) convention, atomic units).
                                   Needed to put `ivg` on the FFT grid: `vgc`
                                   only covers the first `ngvec` entries.
      avec, bvec                -- (3, 3), columns are the lattice /
                                   reciprocal lattice vectors (Elk's own
                                   avec(:, j) convention, atomic units)
      vkc                       -- (3,) the k-point in Cartesian a.u.
      idxis                     -- (natmtot,) 1-based species of each atom
      rbshti, rfshti,
      rbshto, rfshto            -- (lmmaxi, lmmaxi) / (lmmaxo, lmmaxo), the
                                   backward and forward spherical-harmonic
                                   transforms (`genshtmat`).  `rbsht` maps
                                   harmonic coefficients to values on an
                                   angular grid so a nonlinear functional can
                                   be applied pointwise inside a sphere;
                                   `rfsht` maps back
      rlmt, wr2mt               -- (nspecies, max nrmt) the radial mesh and
                                   its r^2-weighted quadrature weights
                                   (`wsplint`, a Simpson-like rule -- not
                                   r^2 dr), zero-padded
      rmt                       -- (nspecies,) muffin-tin radii
      nrmt                      -- (nspecies,) radial points per sphere
      apword                    -- (lmaxapw+1, nspecies) APW order per l
      atposc                    -- (3, natmtot) Cartesian atomic positions,
                                   AS ELK HOLDS THEM: `tshift` may have moved
                                   the origin relative to the input file, and
                                   the matching coefficients' structure factor
                                   uses these, not the input positions
      igpig                     -- (ngp,) 1-based index into Elk's G-vector set
      vgpl, vgpc                -- (3, ngp) G+k in lattice / Cartesian coords
      gpc                       -- (ngp,) |G+k|
      apwalm                    -- (ngp, apwordmax, lmmaxapw, natmtot) complex
      dmat                      -- list over atoms of a list over l of the
                                   (ord, ord) real derivative matrix D that
                                   match inverts; ord = apword[l, is] varies
                                   with l, hence a nested list rather than an
                                   array
      apwfr                     -- same nesting, the last npapw radial points
                                   of each APW radial function u_{jl}(r)
      rsp                       -- (nspecies, npapw) the radial mesh points
                                   those sit on
      hmat, omat                -- (nmatp, nmatp) complex Hermitian
      hmat_istl, omat_istl      -- (ngp, ngp) complex Hermitian, the
                                   interstitial contributions ALONE, so the
                                   muffin-tin APW-APW block can be isolated:
                                   omat[:ngp, :ngp] - omat_istl is exactly
                                   sum_{lm,io} conj(A_{i,io,lm}) A_{j,io,lm}
                                   (olpaa's zmctmu), which is the sharpest
                                   available check of apwalm against Elk's own
                                   assembly
      evalfv                    -- (nstfv,) first-variational eigenvalues (Ha)
      evecfv                    -- (nmatp, nstfv) complex eigenvectors

    hmat/omat are returned Hermitised from the upper triangle Elk writes; see
    _upper_triangle. evalfv/evecfv come from eveqnfv through Elk's OWN
    configured path (before elkpy_lapwexport disables the tefvr real-matrix
    shortcut to build the exported matrices), so diagonalising hmat/omat and
    recovering evalfv is a real check on the export rather than a tautology.
    """
    pos = 0
    head, pos = _take(tokens, pos, 11, int)
    (ngp, nlotot, nmatp, nstfv, apwordmax, lmmaxapw, natmtot, nspecies,
     lmaxapw, npapw, nmatmax) = head
    out = dict(zip(
        ("ngp", "nlotot", "nmatp", "nstfv", "apwordmax", "lmmaxapw",
         "natmtot", "nspecies", "lmaxapw", "npapw", "nmatmax"), head))
    (out["omega"],), pos = _take(tokens, pos, 1, float)
    flat, pos = _take(tokens, pos, 9, float)
    out["avec"] = np.array(flat).reshape(3, 3, order="F")
    flat, pos = _take(tokens, pos, 9, float)
    out["bvec"] = np.array(flat).reshape(3, 3, order="F")
    flat, pos = _take(tokens, pos, 3, float)
    out["vkc"] = np.array(flat)
    flat, pos = _take(tokens, pos, natmtot, int)
    out["idxis"] = np.array(flat)
    flat, pos = _take(tokens, pos, nspecies, float)
    out["rmt"] = np.array(flat)
    flat, pos = _take(tokens, pos, nspecies, int)
    out["nrmt"] = np.array(flat)
    flat, pos = _take(tokens, pos, (lmaxapw + 1) * nspecies, int)
    # written `do is; do l` with l innermost
    apword = np.array(flat).reshape(lmaxapw + 1, nspecies, order="F")
    out["apword"] = apword
    flat, pos = _take(tokens, pos, 3 * natmtot, float)
    out["atposc"] = np.array(flat).reshape(3, natmtot, order="F")
    flat, pos = _take(tokens, pos, ngp, int)
    out["igpig"] = np.array(flat)
    flat, pos = _take(tokens, pos, 3 * ngp, float)
    out["vgpl"] = np.array(flat).reshape(3, ngp, order="F")
    flat, pos = _take(tokens, pos, 3 * ngp, float)
    out["vgpc"] = np.array(flat).reshape(3, ngp, order="F")
    flat, pos = _take(tokens, pos, ngp, float)
    out["gpc"] = np.array(flat)
    n = ngp * apwordmax * lmmaxapw * natmtot
    flat, pos = _take(tokens, pos, 2 * n, float)
    reim = np.array(flat).reshape(n, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    # `do ias; do lm; do io; do igp` -- exactly column-major over
    # apwalm(1:ngp, :, :, :), so an order="F" reshape restores the array
    out["apwalm"] = values.reshape(
        ngp, apwordmax, lmmaxapw, natmtot, order="F")
    dmat = []
    for ias in range(natmtot):
        per_l = []
        for l in range(lmaxapw + 1):
            (ord_,), pos = _take(tokens, pos, 1, int)
            flat, pos = _take(tokens, pos, ord_ * ord_, float)
            per_l.append(np.array(flat).reshape(ord_, ord_, order="F"))
        dmat.append(per_l)
    out["dmat"] = dmat
    apwfr = []
    for ias in range(natmtot):
        is_ = out["idxis"][ias] - 1
        per_l = []
        for l in range(lmaxapw + 1):
            per_o = []
            for _ in range(apword[l, is_]):
                flat, pos = _take(tokens, pos, npapw, float)
                per_o.append(np.array(flat))
            per_l.append(np.array(per_o))
        apwfr.append(per_l)
    out["apwfr"] = apwfr
    flat, pos = _take(tokens, pos, nspecies * npapw, float)
    out["rsp"] = np.array(flat).reshape(nspecies, npapw)
    out["hmat"], pos = _upper_triangle(tokens, pos, nmatp)
    out["omat"], pos = _upper_triangle(tokens, pos, nmatp)
    out["hmat_istl"], pos = _upper_triangle(tokens, pos, ngp)
    out["omat_istl"], pos = _upper_triangle(tokens, pos, ngp)
    flat, pos = _take(tokens, pos, nstfv, float)
    out["evalfv"] = np.array(flat)
    flat, pos = _take(tokens, pos, 2 * nmatp * nstfv, float)
    reim = np.array(flat).reshape(nmatp * nstfv, 2)
    values = reim[:, 0] + 1j * reim[:, 1]
    out["evecfv"] = values.reshape(nmatp, nstfv, order="F")
    _parse_lapw_radial(tokens, pos, out)
    return out


def _parse_lapw_radial(tokens, pos, out):
    """The muffin-tin radial integrals and the Gaunt array (patch 0014).

    These are every remaining input of olpfv/hmlfv beyond `apwalm` and the
    interstitial blocks already parsed above, and they are what lets the JAX
    port assemble the six muffin-tin blocks of H and O and compare each
    against Elk's own, one Fortran routine at a time. Adds:

      lmaxo, lmmaxo, nlomax, lolmmax  -- ints, the shapes
      nlorb          -- (nspecies,) local orbitals per species
      lorbl          -- list over species of that species' (nlorb,) l values
      idxlo          -- (lolmmax, nlomax, natmtot) 1-based index of a local
                        orbital within the lo block, or 0
      oalo           -- (apwordmax, nlomax, natmtot) <u_APW|u_lo> radial
                        overlap (olprad.f90)
      ololo          -- (nlomax, nlomax, natmtot) <u_lo|u_lo>
      haa            -- (lmmaxo, apwordmax, lmaxapw+1, apwordmax,
                        lmaxapw+1, natmtot) APW-APW radial Hamiltonian
                        integrals, indexed haa[lm2, jo, l3, io, l1, ias]
      hloa           -- (lmmaxo, apwordmax, lmaxapw+1, nlomax, natmtot),
                        indexed hloa[lm2, io, l3, ilo, ias]
      hlolo          -- (lmmaxo, nlomax, nlomax, natmtot), indexed
                        hlolo[lm2, jlo, ilo, ias]
      gntyry         -- (lmmaxo, lmmaxapw, lmmaxapw) complex, indexed
                        gntyry[lm2, lm3, lm1] = <Y_{l1 m1}|R_{l2 m2}|Y_{l3 m3}>

    Note the REAL spherical harmonic in gntyry's middle slot: the muffin-tin
    potential is stored in a real harmonic basis, the wavefunctions in a
    complex one. All l indices here are 0-based array positions, i.e. Elk's
    `0:lmaxapw` dimensions become `lmaxapw + 1` long.

    Absent on a binary built before patch 0014, in which case the keys are
    simply not set -- the fields above are appended after `evecfv`, so
    everything patch 0013 wrote still parses identically.
    """
    if pos >= len(tokens):
        return
    head, pos = _take(tokens, pos, 4, int)
    lmaxo, lmmaxo, nlomax, lolmmax = head
    out.update(zip(("lmaxo", "lmmaxo", "nlomax", "lolmmax"), head))
    nspecies = out["nspecies"]
    natmtot = out["natmtot"]
    apwordmax = out["apwordmax"]
    nl = out["lmaxapw"] + 1
    flat, pos = _take(tokens, pos, nspecies, int)
    nlorb = np.array(flat)
    out["nlorb"] = nlorb
    lorbl = []
    for is_ in range(nspecies):
        flat, pos = _take(tokens, pos, int(nlorb[is_]), int)
        lorbl.append(np.array(flat, dtype=int))
    out["lorbl"] = lorbl

    def _block(shape, kind):
        nonlocal pos
        n = int(np.prod(shape))
        values, pos = _take(tokens, pos, n, kind)
        return np.array(values).reshape(shape, order="F")

    out["idxlo"] = _block((lolmmax, nlomax, natmtot), int)
    out["oalo"] = _block((apwordmax, nlomax, natmtot), float)
    out["ololo"] = _block((nlomax, nlomax, natmtot), float)
    out["haa"] = _block(
        (lmmaxo, apwordmax, nl, apwordmax, nl, natmtot), float)
    out["hloa"] = _block((lmmaxo, apwordmax, nl, nlomax, natmtot), float)
    out["hlolo"] = _block((lmmaxo, nlomax, nlomax, natmtot), float)
    lmmaxapw = out["lmmaxapw"]
    n = lmmaxo * lmmaxapw * lmmaxapw
    flat, pos = _take(tokens, pos, 2 * n, float)
    reim = np.array(flat).reshape(n, 2)
    out["gntyry"] = (reim[:, 0] + 1j * reim[:, 1]).reshape(
        (lmmaxo, lmmaxapw, lmmaxapw), order="F")
    _parse_lapw_potential(tokens, pos, out)


def _parse_lapw_potential(tokens, pos, out):
    """The muffin-tin potential, the radial mesh and the radial functions
    (patch 0015).

    These are the INPUTS of the radial integrals parsed above -- what
    `genapwfr`, `genlofr`, `olprad` and `hmlrad` consume -- so with them the
    port can build `haa`/`hloa`/`hlolo`/`oalo`/`ololo` itself rather than
    importing them, and differentiate the spectrum with respect to the
    potential. Adds:

      lmaxi, lmmaxi  -- the inner muffin-tin region's angular cutoff, and
                        (lmaxi+1)^2. Elk stores a muffin-tin function with
                        only lmmaxi harmonics per radial point over points
                        1..nrmti and lmmaxo over the rest; that packing is
                        the load-bearing detail of hmlrad.
      nrmtmax, lorbordmax, nplorb, npmtmax  -- shapes
      y00            -- 1/sqrt(4 pi), the l=0 real spherical harmonic
      solsc          -- speed of light in atomic units, as scaled by `socscf`
      deapw, delorb  -- the finite-difference energy steps behind Elk's
                        energy-derivative radial functions
      nrmti          -- (nspecies,) inner-region radial points
      npmt           -- (nspecies,) packed length of one muffin-tin function
      rlmt           -- (nspecies, nrmtmax) the radial mesh (zero-padded)
      wr2mt          -- (nspecies, nrmtmax) the r^2-weighted quadrature
                        weights.  Written rather than rebuilt: these come
                        from `wsplint`, a Simpson-like rule, not r^2 dr.
      apwdm          -- (apwordmax, lmaxapw+1, nspecies) int
      apwe           -- (apwordmax, lmaxapw+1, natmtot)
      lorbord, idxelo -- (nlomax, nspecies) int
      lorbdm         -- (lorbordmax, nlomax, nspecies) int
      lorbe          -- (lorbordmax, nlomax, natmtot)
      vsmt           -- (natmtot, npmtmax) the muffin-tin Kohn-Sham
                        potential in Elk's own packing; see
                        `unpack_muffin_tin`
      apwfr          -- (nrmtmax, 2, apwordmax, lmaxapw+1, natmtot), Elk's
                        own index order.  [..., 0, ...] is u_{io,l}(r) and
                        [..., 1, ...] is the Gram-Schmidt combination of
                        e*u that genapwfr stores -- NOT e times the first
                        component, except at APW order 1
      apwdfr         -- (apwordmax, lmaxapw+1, natmtot), the surface
                        derivative term (p1s - p0(nr)) * rmt / 2
      lofr           -- (nrmtmax, 2, nlomax, natmtot), same pairing
      autolinengy    -- bool; apwve, lorbve -- bool arrays shaped like
                        `apwdm`/`lorbdm` (patch 0024).  True where
                        `linengy.f90` calls `findband` and re-searches the
                        linearisation energy every SCF iteration; false
                        where `apwe`/`lorbe` stay at the species file's own
                        `apwe0`/`lorbe0` for the whole run.  This is what
                        says whether freezing them outside Elk (as
                        `elkjax.driver` does) is exact or an approximation
                        -- `apwe` alone cannot say, a searched energy and a
                        default one being the same kind of number

    Ragged axes (`apword(l, is)` orders per l, `lorbord(ilo, is)` per local
    orbital) are zero-padded to the maximum, so every field above is a dense
    array; the counts needed to slice them are `apword`/`lorbord`/`nlorb`.

    Absent on a binary built before patch 0015, in which case the keys are
    simply not set.
    """
    if pos >= len(tokens):
        return
    head, pos = _take(tokens, pos, 6, int)
    out.update(zip(
        ("lmaxi", "lmmaxi", "nrmtmax", "lorbordmax", "nplorb", "npmtmax"),
        head))
    lmaxi, lmmaxi, nrmtmax, lorbordmax, nplorb, npmtmax = head
    vals, pos = _take(tokens, pos, 4, float)
    out.update(zip(("y00", "solsc", "deapw", "delorb"), vals))
    nspecies = out["nspecies"]
    natmtot = out["natmtot"]
    apwordmax = out["apwordmax"]
    nlomax = out["nlomax"]
    nl = out["lmaxapw"] + 1
    nrmt = out["nrmt"]
    npmt = np.zeros(nspecies, dtype=int)
    nrmti = np.zeros(nspecies, dtype=int)
    for is_ in range(nspecies):
        pair, pos = _take(tokens, pos, 2, int)
        nrmti[is_], npmt[is_] = pair
    out["nrmti"] = nrmti
    out["npmt"] = npmt
    for key in ("rlmt", "wr2mt"):
        arr = np.zeros((nspecies, nrmtmax))
        for is_ in range(nspecies):
            flat, pos = _take(tokens, pos, int(nrmt[is_]), float)
            arr[is_, :int(nrmt[is_])] = flat
        out[key] = arr
    apword = out["apword"]
    idxis = out["idxis"]
    apwdm = np.zeros((apwordmax, nl, nspecies), dtype=int)
    for is_ in range(nspecies):
        for l in range(nl):
            flat, pos = _take(tokens, pos, int(apword[l, is_]), int)
            apwdm[:int(apword[l, is_]), l, is_] = flat
    out["apwdm"] = apwdm
    apwe = np.zeros((apwordmax, nl, natmtot))
    for ias in range(natmtot):
        is_ = idxis[ias] - 1
        for l in range(nl):
            flat, pos = _take(tokens, pos, int(apword[l, is_]), float)
            apwe[:int(apword[l, is_]), l, ias] = flat
    out["apwe"] = apwe
    nlorb = out["nlorb"]
    lorbord = np.zeros((nlomax, nspecies), dtype=int)
    idxelo = np.zeros((nlomax, nspecies), dtype=int)
    for is_ in range(nspecies):
        for ilo in range(int(nlorb[is_])):
            pair, pos = _take(tokens, pos, 2, int)
            lorbord[ilo, is_], idxelo[ilo, is_] = pair
    out["lorbord"] = lorbord
    out["idxelo"] = idxelo
    lorbdm = np.zeros((lorbordmax, nlomax, nspecies), dtype=int)
    for is_ in range(nspecies):
        for ilo in range(int(nlorb[is_])):
            n = int(lorbord[ilo, is_])
            flat, pos = _take(tokens, pos, n, int)
            lorbdm[:n, ilo, is_] = flat
    out["lorbdm"] = lorbdm
    lorbe = np.zeros((lorbordmax, nlomax, natmtot))
    for ias in range(natmtot):
        is_ = idxis[ias] - 1
        for ilo in range(int(nlorb[is_])):
            n = int(lorbord[ilo, is_])
            flat, pos = _take(tokens, pos, n, float)
            lorbe[:n, ilo, ias] = flat
    out["lorbe"] = lorbe
    vsmt = np.zeros((natmtot, npmtmax))
    for ias in range(natmtot):
        is_ = idxis[ias] - 1
        flat, pos = _take(tokens, pos, int(npmt[is_]), float)
        vsmt[ias, :int(npmt[is_])] = flat
    out["vsmt"] = vsmt
    apwfr = np.zeros((nrmtmax, 2, apwordmax, nl, natmtot))
    for ias in range(natmtot):
        is_ = idxis[ias] - 1
        nr = int(nrmt[is_])
        for l in range(nl):
            for io in range(int(apword[l, is_])):
                for j in range(2):
                    flat, pos = _take(tokens, pos, nr, float)
                    apwfr[:nr, j, io, l, ias] = flat
    out["apwfr_full"] = apwfr
    apwdfr = np.zeros((apwordmax, nl, natmtot))
    for ias in range(natmtot):
        is_ = idxis[ias] - 1
        for l in range(nl):
            flat, pos = _take(tokens, pos, int(apword[l, is_]), float)
            apwdfr[:int(apword[l, is_]), l, ias] = flat
    out["apwdfr"] = apwdfr
    lofr = np.zeros((nrmtmax, 2, nlomax, natmtot))
    for ias in range(natmtot):
        is_ = idxis[ias] - 1
        nr = int(nrmt[is_])
        for ilo in range(int(nlorb[is_])):
            for j in range(2):
                flat, pos = _take(tokens, pos, nr, float)
                lofr[:nr, j, ilo, ias] = flat
    out["lofr"] = lofr
    # patches/0024: whether `linengy` would SEARCH these energies.  Elk only
    # calls `findband` where the species file sets apwve/lorbve true; with
    # every flag false and `autolinengy` off, `apwe`/`lorbe` are the species
    # file's own defaults and never move, so a loop run outside Elk can
    # freeze them exactly.  Written as 0/1 ints rather than Fortran T/F.
    autolinengy, pos = _take(tokens, pos, 1, int)
    out["autolinengy"] = bool(autolinengy[0])
    apwve = np.zeros((apwordmax, nl, nspecies), dtype=bool)
    for is_ in range(nspecies):
        for l in range(nl):
            flat, pos = _take(tokens, pos, int(apword[l, is_]), int)
            apwve[:int(apword[l, is_]), l, is_] = np.asarray(flat, dtype=bool)
    out["apwve"] = apwve
    lorbve = np.zeros((lorbordmax, nlomax, nspecies), dtype=bool)
    for is_ in range(nspecies):
        for ilo in range(int(nlorb[is_])):
            n = int(lorbord[ilo, is_])
            flat, pos = _take(tokens, pos, n, int)
            lorbve[:n, ilo, is_] = np.asarray(flat, dtype=bool)
    out["lorbve"] = lorbve


def parse_groundstate_response(tokens):
    """Parse the token stream of a GROUNDSTATE response -- the converged
    density and potentials on the grids Elk holds them on (see
    `elkpy_gsexport`, patches/0016).

    Returns a dict with:

      natmtot, nspecies, ngtot, ngvec, lmaxi, lmmaxi, lmaxo, lmmaxo,
      npmtmax                   -- ints, the shapes
      ngvc                      -- number of G-vectors with |G| <= 2 gkmax.
                                   `vxcir` has been passed through
                                   `trimrfg`, which zeroes every Fourier
                                   component beyond it (potks.f90), so
                                   reproducing `vxcir` from `rhoir` means
                                   reproducing that filter too.
      ngridg                    -- (3,) the real-space FFT grid
      omega                     -- unit cell volume (Bohr^3)
      avec, bvec                -- (3, 3), columns are the lattice /
                                   reciprocal lattice vectors (Elk's own
                                   avec(:, j) convention, atomic units).
                                   Needed to put `ivg` on the FFT grid: `vgc`
                                   only covers the first `ngvec` entries.
      nrmt, nrmti, npmt         -- (nspecies,) radial points, inner-region
                                   radial points, packed muffin-tin length
      idxis                     -- (natmtot,) 1-based species of each atom
      rbshti, rfshti,
      rbshto, rfshto            -- (lmmaxi, lmmaxi) / (lmmaxo, lmmaxo), the
                                   backward and forward spherical-harmonic
                                   transforms (`genshtmat`).  `rbsht` maps
                                   harmonic coefficients to values on an
                                   angular grid so a nonlinear functional can
                                   be applied pointwise inside a sphere;
                                   `rfsht` maps back
      rlmt, wr2mt               -- (nspecies, max nrmt) the radial mesh and
                                   its r^2-weighted quadrature weights
                                   (`wsplint`, a Simpson-like rule -- not
                                   r^2 dr), zero-padded
      ivg                       -- (3, ngtot) integer G-vectors
      igfft                     -- (ngtot,) 1-based map from the G-vector
                                   index to the FFT array position
      gc                        -- (ngvec,) |G|
      vgc                       -- (3, ngvec) G in Cartesian a.u.
      rhomt, vclmt, vxcmt,
      exmt, ecmt                -- (natmtot, npmtmax) muffin-tin density,
                                   Coulomb and exchange-correlation
                                   potentials and the exchange and
                                   correlation ENERGY densities, in ELK'S OWN
                                   PACKING (see `unpack_muffin_tin`)
      rhoir, vclir, vxcir,
      exir, ecir, vsir, cfunir  -- (ngtot,) the same functions plus the
                                   Kohn-Sham potential and the characteristic
                                   function, on the real-space FFT grid.
                                   `exir`/`ecir` are NOT passed through
                                   `trimrfg` (only `vxcir` is), so they are
                                   the raw pointwise output of the functional
                                   and the exact reference for a
                                   transcription of it
      cfunig                    -- (ngvec,) complex, in G-space
      vsig                      -- (NGVC,) complex, not (ngvec,): `genvsig`
                                   builds it on the COARSE grid, so it only
                                   carries |G| <= 2 gkmax, and `init0`
                                   allocates it that long
      npsd, lnpsd, wprmt,
      vcln, atposc              -- the Weinert Poisson solve's ingredients;
                                   see `_parse_poisson` for why these four and
                                   nothing else

    Two contents are what a transcription has to reproduce rather than what
    it might expect: `rhomt` INCLUDES the core density (`rhocore` adds it),
    and `vclmt` INCLUDES the nuclear -Z/r (`potnucl`).

    `vxcir` is the sharpest available check on an exchange-correlation
    transcription: it is the functional applied POINTWISE to `rhoir` on this
    same grid, so the comparison is exact -- unlike one made through
    `plot1d`, where a nonlinear functional does not commute with either the
    spherical-harmonic truncation or the Fourier interpolation.
    """
    pos = 0
    head, pos = _take(tokens, pos, 9, int)
    out = dict(zip(
        ("natmtot", "nspecies", "ngtot", "ngvec", "lmaxi", "lmmaxi",
         "lmaxo", "lmmaxo", "npmtmax"), head))
    (natmtot, nspecies, ngtot, ngvec, _, lmmaxi, _, lmmaxo, npmtmax) = head
    (out["ngvc"],), pos = _take(tokens, pos, 1, int)
    flat, pos = _take(tokens, pos, 3, int)
    out["ngridg"] = np.array(flat)
    (out["omega"],), pos = _take(tokens, pos, 1, float)
    for key in ("avec", "bvec"):
        flat, pos = _take(tokens, pos, 9, float)
        out[key] = np.array(flat).reshape(3, 3, order="F")
    nrmt = np.zeros(nspecies, dtype=int)
    nrmti = np.zeros(nspecies, dtype=int)
    npmt = np.zeros(nspecies, dtype=int)
    for is_ in range(nspecies):
        triple, pos = _take(tokens, pos, 3, int)
        nrmt[is_], nrmti[is_], npmt[is_] = triple
    out.update(nrmt=nrmt, nrmti=nrmti, npmt=npmt)
    flat, pos = _take(tokens, pos, natmtot, int)
    out["idxis"] = np.array(flat)
    nrmtmax = int(nrmt.max())
    for key in ("rlmt", "wr2mt"):
        arr = np.zeros((nspecies, nrmtmax))
        for is_ in range(nspecies):
            flat, pos = _take(tokens, pos, int(nrmt[is_]), float)
            arr[is_, :int(nrmt[is_])] = flat
        out[key] = arr
    for key, n in (("rbshti", lmmaxi), ("rfshti", lmmaxi),
                   ("rbshto", lmmaxo), ("rfshto", lmmaxo)):
        flat, pos = _take(tokens, pos, n * n, float)
        out[key] = np.array(flat).reshape(n, n, order="F")
    flat, pos = _take(tokens, pos, 3 * ngtot, int)
    out["ivg"] = np.array(flat).reshape(3, ngtot, order="F")
    flat, pos = _take(tokens, pos, ngtot, int)
    out["igfft"] = np.array(flat)
    flat, pos = _take(tokens, pos, ngvec, float)
    out["gc"] = np.array(flat)
    flat, pos = _take(tokens, pos, 3 * ngvec, float)
    out["vgc"] = np.array(flat).reshape(3, ngvec, order="F")
    idxis = out["idxis"]
    for key in ("rhomt", "vclmt", "vxcmt", "exmt", "ecmt"):
        arr = np.zeros((natmtot, npmtmax))
        for ias in range(natmtot):
            n = int(npmt[int(idxis[ias]) - 1])
            flat, pos = _take(tokens, pos, n, float)
            arr[ias, :n] = flat
        out[key] = arr
    for key in ("rhoir", "vclir", "vxcir", "exir", "ecir", "vsir", "cfunir"):
        flat, pos = _take(tokens, pos, ngtot, float)
        out[key] = np.array(flat)
    for key, count in (("cfunig", ngvec), ("vsig", int(out["ngvc"]))):
        flat, pos = _take(tokens, pos, 2 * count, float)
        reim = np.array(flat).reshape(count, 2)
        out[key] = reim[:, 0] + 1j * reim[:, 1]
    pos = _parse_poisson(tokens, pos, out, nspecies, natmtot, nrmt, nrmtmax)
    return out


def _parse_poisson(tokens, pos, out, nspecies, natmtot, nrmt, nrmtmax):
    """Patch 0017's tail: the Weinert solve's ingredients that cannot be rebuilt.

    Everything else `potcoul` needs IS rebuilt rather than exported --
    `rlmt(:,l,:)` is r**l and `rmtl(l,:)` is rmt**l from the mesh above,
    `gclg` is 4*pi/gc**2, and `ylmg`/`sfacg`/`jlgrmt` are `genylmv`,
    `gensfacgp` and `sbessel`, all of which `elkjax.lapw` transcribes and
    Phase 0c already pinned against Elk element-wise.  Exporting `ylmg`
    alone would be ~38 MB of text.

      npsd, lnpsd  -- the pseudocharge exponent and lmaxo+npsd+1 (`init0`)
      wprmt        -- (nspecies, 4, max nrmt) `wsplint`'s cumulative spline
                      weights, which `zpotclmt`'s `splintwp` consumes four at
                      a time.  NOT `wr2mt`, and no closed form worth retyping
      vcln         -- (nspecies, max nrmt) the nuclear potential (`potnucl`),
                      which `potcoul` adds to the l=0 channel BEFORE
                      `zpotcoul` reads the sphere-boundary multipoles -- so a
                      transcription that omits it gets every qlm wrong
      atposc       -- (3, natmtot) Cartesian atomic positions, for the
                      structure factors.  No other query carries them
      spzn         -- (nspecies,) the nuclear charge, NEGATIVE in Elk's
                      convention, for `energy.f90`'s Madelung term
      evalsum ...  -- `energy.f90`'s own converged decomposition, thirteen
      engytot         scalars, so a transcription can be checked TERM BY TERM
                      at full precision instead of against INFO.OUT's print
                      width.  `evalsum` and `engyts` need the
                      second-variational step and the zone sum, so they are
                      what a Phase 2 total energy imports rather than
                      reproduces
      symop        -- (natmtot, natmtot, lmmaxo, lmmaxo) `symrfmt`'s operator,
                      `symop[ias, jas]` mapping atom `jas`'s harmonics to atom
                      `ias`'s.  EXPORTED rather than transcribed: `rotrflm`'s
                      Euler-angle and Wigner-D construction has no consumer
                      but `symrfmt`, so a re-derivation would have no
                      independent check, and Elk's whole atom bookkeeping
                      would have to come with it.  The inner region uses this
                      matrix's top-left `lmmaxi` block, since `rotrfmt` calls
                      `rotrflm` separately on the two regions with the same
                      rotation
    """
    pair, pos = _take(tokens, pos, 2, int)
    out["npsd"], out["lnpsd"] = pair
    wprmt = np.zeros((nspecies, 4, nrmtmax))
    vcln = np.zeros((nspecies, nrmtmax))
    for is_ in range(nspecies):
        n = int(nrmt[is_])
        flat, pos = _take(tokens, pos, 4 * n, float)
        wprmt[is_, :, :n] = np.array(flat).reshape(4, n, order="F")
        flat, pos = _take(tokens, pos, n, float)
        vcln[is_, :n] = flat
    out.update(wprmt=wprmt, vcln=vcln)
    flat, pos = _take(tokens, pos, 3 * natmtot, float)
    out["atposc"] = np.array(flat).reshape(3, natmtot, order="F")
    flat, pos = _take(tokens, pos, nspecies, float)
    out["spzn"] = np.array(flat)
    names = ("evalsum", "engykn", "engyvcl", "engyvxc", "engymad", "engyen",
             "engyhar", "engycl", "engynn", "engyx", "engyc", "engyts",
             "engytot")
    flat, pos = _take(tokens, pos, len(names), float)
    out.update(zip(names, flat))
    lmmaxo = int(out["lmmaxo"])
    flat, pos = _take(tokens, pos, (natmtot * lmmaxo) ** 2, float)
    # written jas-slowest, then the source harmonic, then ias, then the
    # destination harmonics
    block = np.array(flat).reshape(natmtot, lmmaxo, natmtot, lmmaxo)
    out["symop"] = np.ascontiguousarray(block.transpose(2, 0, 3, 1))
    return pos


def unpack_muffin_tin(packed, nr, nri, lmmaxi, lmmaxo):
    """Elk's packed muffin-tin function -> a dense (nr, lmmaxo) array.

    A muffin-tin function is stored with only `lmmaxi` spherical-harmonic
    coefficients per radial point over the inner region (points 1..nri) and
    `lmmaxo` over the outer one, laid out radial-point-slowest.  Every
    consumer in `vendor/elk/src/` therefore addresses it with a stride
    (`vsmt(lm:i1:lmmaxi, ias)`), and reproducing that stride arithmetic is
    where a transcription of `hmlrad` goes wrong.  This returns the dense
    array instead, with the harmonics the inner region does not carry set to
    ZERO -- which is exactly what makes `hmlrad`'s `l2 <= lmaxi` guard
    automatic rather than a branch to remember.
    """
    out = np.zeros((nr, lmmaxo))
    inner = np.asarray(packed[:lmmaxi * nri]).reshape(nri, lmmaxi)
    out[:nri, :lmmaxi] = inner
    outer = np.asarray(
        packed[lmmaxi * nri:lmmaxi * nri + lmmaxo * (nr - nri)])
    out[nri:, :] = outer.reshape(nr - nri, lmmaxo)
    return out


def parse_densityk_response(tokens):
    """The `DENSITYK` query: what `rhomagv` feeds to `rhomagk`, per k-point.

    The valence density is a zone sum, so a transcription needs the k-set Elk
    actually used -- weights included.  Reconstructing it from `ngridk` would
    silently differ whenever `reducek` is nonzero, which is the default.

    Returns a dict:

      nkpt, nspnfv, nstfv, nstsv, ngkmax
      ngdgc                     -- (3,) the COARSE interstitial FFT grid, which
                                   is where `rhomagk` accumulates.  Not the
                                   `ngridg` the `GROUNDSTATE` query's `rhoir`
                                   lives on; `rfirctof` interpolates between
                                   them by zero-padding
      ngtc, ngvc                -- its size and its G-vector count
      igfc                      -- (ngvc,) 1-based map from the G index to the
                                   coarse FFT array position
      wkpt                      -- (nkpt,)
      vkl                       -- (3, nkpt) k in fractional coordinates
      occsv                     -- (nkpt, nstsv) second-variational occupations
      ngk                       -- (nkpt, nspnfv) the |G+k| < gkmax count
      igkig                     -- list per (ik, ispn) of 1-based indices into
                                   the global G list
      nmat                      -- (nkpt, nspnfv) = ngk + nlotot, the FULL
                                   basis size
      evecfv                    -- list per (ik, ispn) of (nstfv, nmat) complex
                                   first-variational eigenvectors, cut to this
                                   k-point's own `nmat` rather than padded to
                                   `nmatmax`.  NOT cut to `ngk`: the entries
                                   beyond it are the local-orbital
                                   coefficients, which `wfmtsv` reads as
                                   `evecfv(ngp + idxlo(...))`
    """
    pos = 0
    head, pos = _take(tokens, pos, 5, int)
    out = dict(zip(("nkpt", "nspnfv", "nstfv", "nstsv", "ngkmax"), head))
    nkpt, nspnfv, nstfv, nstsv, _ = head
    shapes, pos = _take(tokens, pos, 4, int)
    out.update(zip(("natmtot", "nspecies", "lmmaxi", "lmmaxo"), shapes))
    flat, pos = _take(tokens, pos, int(out["natmtot"]), int)
    out["idxis"] = np.array(flat)
    flat, pos = _take(tokens, pos, 3, int)
    out["ngdgc"] = np.array(flat)
    pair, pos = _take(tokens, pos, 2, int)
    out["ngtc"], out["ngvc"] = pair
    flat, pos = _take(tokens, pos, int(out["ngvc"]), int)
    out["igfc"] = np.array(flat)

    wkpt = np.zeros(nkpt)
    vkl = np.zeros((3, nkpt))
    occsv = np.zeros((nkpt, nstsv))
    ngk = np.zeros((nkpt, nspnfv), dtype=int)
    nmat = np.zeros((nkpt, nspnfv), dtype=int)
    igkig, evecfv = {}, {}
    for ik in range(nkpt):
        (wkpt[ik],), pos = _take(tokens, pos, 1, float)
        flat, pos = _take(tokens, pos, 3, float)
        vkl[:, ik] = flat
        flat, pos = _take(tokens, pos, nstsv, float)
        occsv[ik] = flat
        for ispn in range(nspnfv):
            pair, pos = _take(tokens, pos, 2, int)
            ngk[ik, ispn], nmat[ik, ispn] = pair
            flat, pos = _take(tokens, pos, int(ngk[ik, ispn]), int)
            igkig[(ik, ispn)] = np.array(flat)
        for ispn in range(nspnfv):
            n = int(nmat[ik, ispn])
            flat, pos = _take(tokens, pos, 2 * nstfv * n, float)
            reim = np.array(flat).reshape(nstfv, n, 2)
            evecfv[(ik, ispn)] = reim[:, :, 0] + 1j * reim[:, :, 1]
    out.update(wkpt=wkpt, vkl=vkl, occsv=occsv, ngk=ngk, nmat=nmat,
               igkig=igkig, evecfv=evecfv)

    triple, pos = _take(tokens, pos, 3, int)
    out["lradstp"], out["npcmtmax"], out["npmtmax"] = triple
    nrcmt = np.zeros(int(out["nspecies"]), dtype=int)
    nrcmti = np.zeros_like(nrcmt)
    npcmt = np.zeros_like(nrcmt)
    npcmti = np.zeros_like(nrcmt)
    nrmt = np.zeros_like(nrcmt)
    nrmti = np.zeros_like(nrcmt)
    npmt = np.zeros_like(nrcmt)
    for is_ in range(nrcmt.size):
        quad, pos = _take(tokens, pos, 4, int)
        nrcmt[is_], nrcmti[is_], npcmt[is_], npcmti[is_] = quad
        trio, pos = _take(tokens, pos, 3, int)
        nrmt[is_], nrmti[is_], npmt[is_] = trio
    out.update(nrcmt=nrcmt, nrcmti=nrcmti, npcmt=npcmt, npcmti=npcmti,
               nrmt=nrmt, nrmti=nrmti, npmt=npmt)

    for key, n in (("zbshti", int(out["lmmaxi"])),
                   ("zbshto", int(out["lmmaxo"]))):
        # written as the whole real part then the whole imaginary part, each
        # in Fortran order -- NOT interleaved like the complex arrays above
        flat, pos = _take(tokens, pos, 2 * n * n, float)
        real = np.array(flat[:n * n]).reshape(n, n, order="F")
        imag = np.array(flat[n * n:]).reshape(n, n, order="F")
        out[key] = real + 1j * imag

    natmtot = int(out["natmtot"])
    idxis = np.asarray(out["idxis"])
    rhomt = np.zeros((natmtot, int(out["npcmtmax"])))
    for ias in range(natmtot):
        n = int(npcmt[int(idxis[ias]) - 1])
        flat, pos = _take(tokens, pos, n, float)
        rhomt[ias, :n] = flat
    out["rhomt_coarse"] = rhomt
    flat, pos = _take(tokens, pos, int(out["ngtc"]), float)
    out["rhoir_coarse"] = np.array(flat)

    # the two post-processing intermediates, so each step is checkable alone
    for key, sizes, width in (("rhomt_sh", npcmt, int(out["npcmtmax"])),
                              ("rhomt_fine", npmt, int(out["npmtmax"]))):
        arr = np.zeros((natmtot, width))
        for ias in range(natmtot):
            n = int(sizes[int(idxis[ias]) - 1])
            flat, pos = _take(tokens, pos, n, float)
            arr[ias, :n] = flat
        out[key] = arr

    # rfmtctof as a matrix, per species: the full-range map and the
    # outer-region-only one.  Written one COARSE basis vector at a time, so
    # row i of the exported block is the image of coarse point i -- i.e. the
    # transpose of the operator as it multiplies a column vector.
    full, outer = [], []
    for is_ in range(int(out["nspecies"])):
        nrc, nrci = int(nrcmt[is_]), int(nrcmti[is_])
        nr, nri = int(out["nrmt"][is_]), int(out["nrmti"][is_])
        flat, pos = _take(tokens, pos, nrc * nr, float)
        full.append(np.array(flat).reshape(nrc, nr).T)
        flat, pos = _take(tokens, pos, (nrc - nrci) * (nr - nri), float)
        outer.append(np.array(flat).reshape(nrc - nrci, nr - nri).T)
    out["ctof_full"] = full
    out["ctof_outer"] = outer

    (nsymcrys,), pos = _take(tokens, pos, 1, int)
    out["nsymcrys"] = nsymcrys
    ngvc = int(out["ngvc"])
    symmap = np.zeros((nsymcrys, ngvc), dtype=int)
    symphase = np.zeros((nsymcrys, ngvc), dtype=complex)
    for isym in range(nsymcrys):
        for ig in range(ngvc):
            (jg,), pos = _take(tokens, pos, 1, int)
            pair, pos = _take(tokens, pos, 2, float)
            symmap[isym, ig] = jg
            symphase[isym, ig] = pair[0] + 1j * pair[1]
    out["symmap"], out["symphase"] = symmap, symphase

    (nspncr,), pos = _take(tokens, pos, 1, int)
    out["nspncr"] = nspncr
    (out["chgtot"],), pos = _take(tokens, pos, 1, float)
    rhocr = np.zeros((natmtot, nspncr, int(nrmt.max())))
    for ias in range(natmtot):
        n = int(nrmt[int(idxis[ias]) - 1])
        for ispn in range(nspncr):
            flat, pos = _take(tokens, pos, n, float)
            rhocr[ias, ispn, :n] = flat
    out["rhocr"] = rhocr

    # occupy.f90's own inputs and its own answer (patch 0023).  `evalsv` is
    # here and not on the `LAPW` query because it is per-k over the WHOLE set:
    # the Fermi level is a zone sum, so one k-point cannot supply it.
    (stype,), pos = _take(tokens, pos, 1, int)
    out["stype"] = stype
    scalars, pos = _take(tokens, pos, 6, float)
    out.update(zip(("swidth", "occmax", "chgval", "e0min", "epsocc",
                    "efermi"), scalars))
    evalsv = np.zeros((nkpt, nstsv))
    for ik in range(nkpt):
        flat, pos = _take(tokens, pos, nstsv, float)
        evalsv[ik] = flat
    out["evalsv"] = evalsv
    (out["evalsumcr"],), pos = _take(tokens, pos, 1, float)
    # patches/0024: `energykncr` at this potential, i.e. `evalsumcr` MINUS
    # the integral of the core density against v_s.  Elk recomputes both
    # halves every iteration; a loop that freezes the core must freeze this
    # difference rather than `evalsumcr` alone, or the same core density is
    # integrated against a potential its eigenvalues never saw.
    (out["engykncr"],), pos = _take(tokens, pos, 1, float)
    return out


