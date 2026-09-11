"""The momentum-resolved tunnelling Fermi surface: a PLANAR tip above a
two-dimensional material, a substrate plane below it (elkpy task 9007,
src/elkpy_fermitunnel.f90 -- docs/design.md #35).

Like `transport`, this is not just a parser: the contraction and every
diagnostic are done here rather than in Fortran, so they are unit-testable
against synthetic Gram matrices without an Elk run.

`transport` (docs/design.md #31) puts a POINT tip at `r` above the sheet and
gives the real-space map

    T(r; E) = int_exit |G(r, r'; E)|^2 d^2r'.

Replace the point tip by an infinite plane at height z_t and the same
Landauer-Buettiker trace becomes a double plane integral,

    T(E) = int_tip d^2r int_exit d^2r' |G(r, r'; E)|^2.

Both planes are invariant under every lateral lattice translation, so the
lateral momentum is conserved going IN as well as going OUT, the k-sum is
incoherent, and what is left is one number per k-point:

    T(E) = sum_k w_k W(k; E),     W(k; E) = Tr[D G_t D S],
    D = diag(g_n),  g_n = sqrt(occmax delta_eta(E - eps_kn))

with the two plane Gram matrices the Fortran task exports,

    G_t[n, m] = int_tip  psi*_kn Q_t psi_km d^2r      (the tip plane)
    S[n, m]   = int_exit psi*_kn P_s psi_km d^2r'     (the substrate plane).

**The contraction is a trace of a product, not an elementwise sum.** Both
matrices come out of the Fortran in the SAME convention (conjugate on the
first index), so the tip integral appearing in |G|^2 is

    int_tip psi_n psi*_m d^2r = G_t[m, n]

by hermiticity, and W = sum_nm g_n g_m G_t[m, n] S[n, m] = Tr[D G_t D S]. The
tempting elementwise form sum_nm g_n g_m G_t[n, m] S[n, m] is Tr[D G_t D S^T]
-- it is real, non-negative, exactly right whenever S is diagonal, and wrong
wherever S has a complex off-diagonal, which is precisely where the
interference this exists for lives. It is the same transpose trap
`parsers.transport` documents one level down, on the exit variable, and
`quadrature_weight()` is what pins it: it evaluates the double integral
literally.

**Two limits come free from the same export**, which is what makes the result
readable:

- `tip_region="cell"` and `exit_region="cell"` replace both matrices by the
  identity, so W = sum_n g_n^2 = occmax sum_n delta_eta(E - eps_kn) -- the
  PLAIN Fermi surface, the same quantity Elk's own task 103 writes.
- `exit_region="cell"` alone leaves W = sum_n g_n^2 G_t[n, n]: the Fermi
  surface weighted by how much of each state survives out at the tip plane.
  A planar Tersoff-Hamann image, in momentum space.

`compute_fermi_weight()` returns the plain Fermi surface alongside the
tunnelling one on every call, so the ratio -- which pocket the junction
actually sees -- costs nothing.
"""

import numpy as np

from .transport import DEGENERACY_TOL, amplitude_weights, channel_basis


def parse_fermitunnel(path):
    """Read ELKPY_FERMITUNNEL.OUT.

    Returns a dict with the run's geometry plus, per k-point, the states
    inside the exported energy window: their indices and eigenvalues and the
    two plane Gram matrices.

    Keys: nspinor, grid (n1, n2, n3), axis, efermi, tip_height, exit_height,
    area, omega, koffset, window (relative to E_F), tip_polarization,
    tip_direction, polarization, direction, kpoints (nk, 3) in lattice
    coordinates, kpoints_cartesian (nk, 3) in 1/Bohr, weights (nk,), and the
    per-k lists `states`, `eigenvalues`, `tip_overlaps` (nsel, nsel) and
    `overlaps` (nsel, nsel).

    `grid` is (n1, n2, n3) with the FIRST index running fastest over the
    k-list, matching the write loop in src/elkpy_fermitunnel.f90 -- so a
    per-k column reshapes as (n3, n2, n1). `reshape_grid()` does that.
    """
    def _numbers(chunk, ncol):
        if not chunk:
            return np.empty((0, ncol))
        return np.fromstring(" ".join(chunk), sep=" ").reshape(-1, ncol)

    with open(path) as fh:
        lines = fh.read().splitlines()
    ints = [int(x) for x in lines[1].split(":")[0].split()]
    nkpt, nspinor, n1, n2, n3, axis = ints
    reals = [float(x) for x in lines[2].split(":")[0].split()]
    efermi, tip_height, exit_height, area, omega = reals
    koffset = np.array([float(x) for x in lines[3].split(":")[0].split()])
    window = tuple(float(x) for x in lines[4].split(":")[0].split())
    reals = [float(x) for x in lines[5].split(":")[0].split()]
    tip_polarization, tip_direction = reals[0], np.array(reals[1:4])
    polarization, direction = reals[4], np.array(reals[5:8])

    i = 6
    kpoints = np.empty((nkpt, 3))
    kcart = np.empty((nkpt, 3))
    weights = np.empty(nkpt)
    states, eigenvalues, tip_overlaps, overlaps = [], [], [], []
    for ik in range(nkpt):
        head = lines[i].split(":")[0].split()
        kpoints[ik] = [float(x) for x in head[:3]]
        kcart[ik] = [float(x) for x in head[3:6]]
        weights[ik] = float(head[6])
        nsel = int(head[7])
        i += 1
        sel = _numbers(lines[i:i + nsel], 2)
        states.append(sel[:, 0].astype(int))
        eigenvalues.append(sel[:, 1])
        i += nsel
        n = nsel * nsel
        for target in (tip_overlaps, overlaps):
            raw = _numbers(lines[i:i + n], 2)
            # written column by column (Fortran's own storage order)
            target.append((raw[:, 0] + 1j * raw[:, 1]).reshape(nsel, nsel).T)
            i += n
    return {
        "nspinor": nspinor,
        "grid": (n1, n2, n3),
        "axis": axis,
        "efermi": efermi,
        "tip_height": tip_height,
        "exit_height": exit_height,
        "area": area,
        "omega": omega,
        "koffset": koffset,
        "window": window,
        "tip_polarization": tip_polarization,
        "tip_direction": tip_direction,
        "polarization": polarization,
        "direction": direction,
        "kpoints": kpoints,
        "kpoints_cartesian": kcart,
        "weights": weights,
        "states": states,
        "eigenvalues": eigenvalues,
        "tip_overlaps": tip_overlaps,
        "overlaps": overlaps,
    }


def reshape_grid(values, grid):
    """Fold a per-k column onto the k-grid.

    src/elkpy_fermitunnel.f90 runs the FIRST grid index fastest, so a (nk,)
    column is a (n3, n2, n1) array -- the same load-bearing ordering detail as
    parsers.volumetric.parse_plot2d and parsers.transport.parse_transport.
    A trailing axis of length one (a 2D material's k-mesh) is dropped, so a
    (n1, n2, 1) mesh comes back as (n2, n1), ready to imshow.
    """
    n1, n2, n3 = grid
    values = np.asarray(values)
    out = values.reshape(values.shape[:-1] + (n3, n2, n1))
    if n3 == 1:
        out = out[..., 0, :, :]
    return out


def _check_energy_window(data, energies):
    """Refuse energies outside the window src/elkpy_fermitunnel.f90 exported.

    Same trap, same reason as parsers.transport._check_energy_window: these
    `energies` are ABSOLUTE Hartree while
    Calculation.get_tunnelling_fermi_surface()'s identically named argument is
    a bias RELATIVE to the Fermi energy. Outside the exported window there are
    no states at all, but the smeared delta has exponential tails, so what
    comes back would be a plausible small map at the wrong energy rather than
    an error.
    """
    efermi = float(data["efermi"])
    lo, hi = (efermi + float(w) for w in data["window"])
    energies = np.asarray(energies, dtype=float)
    outside = energies[(energies < lo) | (energies > hi)]
    if outside.size == 0:
        return
    raise ValueError(
        "compute_fermi_weight() energies are ABSOLUTE Hartree, and "
        f"{np.array2string(outside, precision=6)} lies outside the exported "
        f"window [{lo:.6f}, {hi:.6f}] (E_F = {efermi:.6f}, window "
        f"[{data['window'][0]:.6f}, {data['window'][1]:.6f}] relative to it), "
        "which holds no states -- the result would be a plausible small map "
        "built from the tails of the smeared delta rather than an error. If "
        "these are a bias relative to the Fermi energy, pass "
        "data['efermi'] + bias; "
        "Calculation.get_tunnelling_fermi_surface() takes the relative form "
        "and does that shift for you."
    )


def weight_at_k(tip_overlap, overlap, weights, coherent=True,
                eigenvalues=None, tol=DEGENERACY_TOL):
    """One k-point's tunnelling weight, before its k-weight: a scalar.

    `weights` is the per-state on-shell amplitude g_n. The coherent form is
    Tr[D G_t D S]; see the module docstring for why it is that and not the
    elementwise product.

    The INCOHERENT form -- every substrate channel tunnelling on its own --
    needs a basis and the coherent one does not: Tr[D G_t D S] is invariant
    under any unitary mixing a degenerate multiplet (D is a multiple of the
    identity there), while a diagonal is not. It is taken in the basis
    diagonalising S within each multiplet, i.e. the substrate's own channels,
    which is exactly parsers.transport's convention for the same split.
    """
    tip_overlap = np.asarray(tip_overlap)
    overlap = np.asarray(overlap)
    weights = np.asarray(weights, dtype=float)
    nst = weights.shape[0]
    for name, m in (("tip", tip_overlap), ("exit", overlap)):
        if m.shape != (nst, nst):
            raise ValueError(
                f"the {name}-plane Gram matrix is {m.shape} and the "
                f"{nst} state weights want {(nst, nst)}"
            )
    if coherent:
        # Tr[D G_t D S]: the conjugation on the tip variable puts G_t's
        # TRANSPOSE against S, and G_t is Hermitian, so this is G_t itself
        dgd = weights[:, None] * tip_overlap * weights[None, :]
        return float(np.real(np.einsum("mn,nm->", dgd, overlap, optimize=True)))
    if eigenvalues is not None:
        u = channel_basis(overlap, eigenvalues, tol)
        tip_overlap = u.conj().T @ tip_overlap @ u
        overlap = u.conj().T @ overlap @ u
    return float(np.real(
        np.dot(weights**2, np.diag(tip_overlap) * np.diag(overlap))
    ))


def quadrature_weight(tip_amplitude, exit_amplitude, tip_area, exit_area,
                      weights):
    """W at one k-point by evaluating the DEFINITION, with a quadrature.

    Builds G(r, r') = sum_n g_n psi_n(r) psi*_n(r') on sampled tip and exit
    points and integrates |G|^2 over BOTH planes by the rectangle rule, which
    is exact for this integrand once each sampling exceeds twice the largest
    in-plane G-vector index (the integrand is periodic and band-limited).

    This is what discriminates Tr[D G_t D S] from Tr[D G_t D S^T]: the two
    differ by a transpose of a Hermitian matrix, so the wrong one is real,
    non-negative, blind to a degenerate rotation, and exactly right in the
    Tersoff-Hamann limit where S is the identity. Nothing but a literal check
    against the definition can tell them apart -- which is why the test that
    uses this must sample TWO DISTINCT planes, so the Gram matrices actually
    have complex off-diagonals.

    `tip_amplitude` and `exit_amplitude` are (nspinor, nst, npts) samples of
    the same states on the two planes; they may have different npts. The leads
    are taken unpolarised here (both Gram matrices spin-summed), which is what
    makes G a 2x2 matrix in spin space whose |G|^2 sums over BOTH spin
    indices -- and is why this reduces to the plane integral of
    parsers.transport's own point-tip map, whose exit Gram is likewise
    spin-summed.
    """
    tip = np.asarray(tip_amplitude) * np.asarray(weights)[None, :, None]
    exit_ = np.asarray(exit_amplitude)
    npt, npx = tip.shape[2], exit_.shape[2]
    g = np.einsum("snp,tnq->stpq", tip, exit_.conj(), optimize=True)
    return float(
        (tip_area / npt) * (exit_area / npx) * np.real(np.sum(g * g.conj()))
    )


def compute_fermi_weight(data, energies=None, broadening=0.005, stype=3,
                         tip_region="plane", exit_region="plane",
                         incoherent=True, degeneracy_tol=DEGENERACY_TOL):
    """The momentum-resolved tunnelling weight, from `parse_fermitunnel`.

    `energies` are absolute (Hartree); the default is the Fermi energy. They
    are checked against the exported window and an energy outside it is
    refused -- see `_check_energy_window`.

    `tip_region`/`exit_region` are "plane" (the real Gram matrix) or "cell"
    (the identity). Both "cell" is the plain Fermi surface, which is returned
    as "bare" on every call regardless, so no second run is needed for it.

    **Three levels of the same measurement come back on every call**, on the
    same k-points and the same eigenvalues, because they are three
    contractions of one export:

    - `"bare"`: both Gram matrices replaced by the identity,
      occmax sum_n delta_eta(E - eps_kn). The PLAIN density of states --
      what a band-structure calculation reports, with no junction in it.
    - `"tersoff_hamann"`: sum_n g_n^2 int_tip |psi_n|^2, the local density of
      states at the tip plane. This is the Tersoff-Hamann model of dI/dV: each
      band's DOS weighted by how much of it survives out at the tip, with
      bands adding as PROBABILITIES. Always built from the real exported tip
      Gram, whatever `tip_region` is set to, so it stays a fixed reference.
    - `"weight"`: the full Tr[D G_t D S]. Bands add as AMPLITUDES, and the
      off-diagonals of both Gram matrices contribute -- the interference a
      local density of states cannot represent.

    Their k-integrals are `"total_bare"`, `"total_tersoff_hamann"` and
    `"total"` (nE,). `"total"` is the plane integral of parsers.transport's
    own point-tip map, which is the test that ties the two tasks together.

    Returns a dict with those, plus "energies" (nE,),
    "incoherent"/"interference" (nE, nk) or None, "kpoints",
    "kpoints_cartesian", "weights", "grid", and the diagnostics
    "least_eigenvalue", "hermiticity", "offdiagonal_weight" and "channels"
    (all taken on the EXIT plane, as in parsers.transport).
    """
    for name, region in (("tip_region", tip_region),
                         ("exit_region", exit_region)):
        if region not in ("plane", "cell"):
            raise ValueError(
                f"unknown {name} {region!r}: use 'plane' (the real Gram "
                "matrix) or 'cell' (the identity)"
            )
    if energies is None:
        energies = [data["efermi"]]
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    _check_energy_window(data, energies)
    occmax = 2.0 if data["nspinor"] == 1 else 1.0
    nk = data["kpoints"].shape[0]

    weight = np.zeros((energies.size, nk))
    bare = np.zeros_like(weight)
    tersoff = np.zeros_like(weight)
    inc = np.zeros_like(weight) if incoherent else None
    least, offdiag, channels, hermiticity = np.inf, [], [], 0.0
    for ik in range(nk):
        eig = data["eigenvalues"][ik]
        if eig.size == 0:
            continue
        gtip = data["tip_overlaps"][ik]
        ovl = data["overlaps"][ik]
        hermiticity = max(
            hermiticity,
            float(np.abs(ovl - ovl.conj().T).max()),
            float(np.abs(gtip - gtip.conj().T).max()),
        )
        spectrum = np.linalg.eigvalsh(0.5 * (ovl + ovl.conj().T))
        least = min(least, float(spectrum.min()))
        # how many independent ways through the substrate there are: the
        # participation ratio of S_k's spectrum
        positive = np.clip(spectrum, 0.0, None)
        norm2 = float((positive**2).sum())
        channels.append(positive.sum()**2 / norm2 if norm2 > 0.0 else 0.0)
        # in the channel basis, so that "how much sits off the diagonal" is a
        # property of the substrate and not of the eigensolver's choice
        u = channel_basis(ovl, eig, degeneracy_tol)
        rotated = u.conj().T @ ovl @ u
        norm = float(np.linalg.norm(rotated))
        diag = float(np.linalg.norm(np.diag(rotated)))
        offdiag.append(
            np.sqrt(max(norm**2 - diag**2, 0.0)) / norm if norm > 0.0 else 0.0
        )
        # the Tersoff-Hamann reference, taken from the REAL tip Gram before any
        # region switch: sum_n g_n^2 int_tip |psi_n|^2, the plane-integrated
        # local density of states at the tip height. Bands add as
        # probabilities here and as amplitudes in `weight` -- that difference
        # is the whole point of comparing the two
        tersoff_diag = np.real(np.diag(gtip)).copy()
        if tip_region == "cell":
            gtip = np.eye(eig.size, dtype=complex)
        if exit_region == "cell":
            ovl = np.eye(eig.size, dtype=complex)
        identity = np.eye(eig.size, dtype=complex)
        for ie, energy in enumerate(energies):
            g = amplitude_weights(eig, energy, broadening, stype, occmax)
            weight[ie, ik] = weight_at_k(gtip, ovl, g, True)
            bare[ie, ik] = weight_at_k(identity, identity, g, True)
            tersoff[ie, ik] = float(np.dot(g**2, tersoff_diag))
            if incoherent:
                inc[ie, ik] = weight_at_k(
                    gtip, ovl, g, False, eig, degeneracy_tol
                )
    return {
        "energies": energies,
        "weight": weight,
        "bare": bare,
        "tersoff_hamann": tersoff,
        "incoherent": inc,
        "interference": None if inc is None else weight - inc,
        "total": weight @ data["weights"],
        "total_bare": bare @ data["weights"],
        "total_tersoff_hamann": tersoff @ data["weights"],
        "kpoints": data["kpoints"],
        "kpoints_cartesian": data["kpoints_cartesian"],
        "weights": data["weights"],
        "grid": data["grid"],
        # 0.0, not None, when no k-point carried a state in the window: this
        # is compared against a tolerance by every caller
        "least_eigenvalue": 0.0 if least == np.inf else least,
        "hermiticity": hermiticity,
        "offdiagonal_weight": float(np.mean(offdiag)) if offdiag else 0.0,
        "channels": float(np.mean(channels)) if channels else 0.0,
    }
