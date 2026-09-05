"""Vertical tunnelling transport: point tip -> substrate plane (elkpy task
9005, src/elkpy_transport.f90 -- docs/design.md #31).

Like `berry`, `wilson`, `optical` and `exchange`, this is not just a parser:
it also does all of the transport arithmetic, deliberately kept out of Fortran
so it is unit-testable against synthetic overlaps and amplitudes
(tests/test_parsers_transport.py) without an Elk run.

An electron enters at a point `r` above a two-dimensional material (the STM
tip) and leaves into an infinite, featureless plane below it (the substrate).
A point tip makes the tip coupling rank one, which collapses the
Landauer-Buettiker transmission T = Tr[Gamma_t G^r Gamma_s G^a] exactly to an
integral of the nonlocal Green's function over the exit plane,

    T(r; E) = int_plane |G(r, r'; E)|^2 d^2r',
    G(r, r') = sum_kn psi_kn(r) psi*_kn(r') / (E - eps_kn + i eta)

A substrate invariant under every lateral lattice translation conserves the
lateral momentum, so the k-sum is incoherent and what survives is one
Hermitian Gram matrix per k-point,

    S_k[n, n'] = int_plane psi*_kn(r') P_s psi_k n'(r') d^2r'
    T(r; E)    = sum_k w_k sum_nm a_n(r) a*_m(r) S_k[n, m]

Two conventions here are load-bearing and are taken from the sister
plane-wave code (defumat), which measured both:

**The amplitude is the on-shell one.** Writing G with its literal energy
denominator 1/(E - eps + i eta) is the exact Landauer expression, and a sum
over a truncated band set CANNOT evaluate it: the states far from E carry the
barrier's evanescent decay entirely through cancellation between themselves
(measured cancellation ratio 349 on a cell diagonalised completely). What
converges is the modulus of that denominator with its phase held fixed --
Bardeen's golden rule, the sample visited on shell,

    a_kn(r) = psi_kn(r) sqrt(occmax delta_eta(E - eps_kn) / eta)

which is *exactly* the resolvent's modulus for a Lorentzian; all that is
dropped is the arctan its phase sweeps across a resonance. The interference it
keeps is between bands degenerate at the tip energy -- the interference a
tunnelling experiment actually lets happen, and the part no local density of
states can represent.

**The conjugation sits on the exit variable.** G(r, r') carries psi conjugated
in r', so the plane integral of |G|^2 is sum_nm a_n a*_m S[n, m] and not
sum_nm a*_n a_m S[n, m]. The two differ by a transpose of a Hermitian matrix,
so the wrong one is real, non-negative, exactly reproduces the
Tersoff-Hamann limit (where S is the identity and they coincide), and is wrong
wherever S has an off-diagonal -- that is, wherever the interference this
exists for actually lives. `green_function_transmission()` evaluates the
definition literally, by quadrature, and is what pins it.
"""

import numpy as np

#: Two eigenvalues closer than this (Hartree) count as one multiplet.
DEGENERACY_TOL = 1.0e-6


def parse_transport(path):
    """Read ELKPY_TRANSPORT.OUT.

    Returns a dict with the run's geometry plus, per k-point, the states
    inside the exported energy window: their indices and eigenvalues, the
    exit-plane Gram matrix and the tip amplitudes.

    Keys: nspinor, npoints, grid (n2, n1 -- see below), axis, efermi,
    height, area, omega, window, polarization, direction, points (np, 2)
    in-plane Cartesian Bohr in the plotting parallelogram's own frame,
    points_lattice (np, 3), kpoints (nk, 3), weights (nk,), and the
    per-k lists `states`, `eigenvalues`, `overlaps` (nsel, nsel) and
    `amplitudes` (nspinor, nsel, np).

    `grid` is returned as (n2, n1) rather than np2d's own (n1, n2) because
    src/plotpt2d.f90 runs the FIRST plotting vector's index fastest, so a
    per-point column reshapes as (n2, n1) -- the same load-bearing ordering
    detail as parsers.volumetric.parse_plot2d.
    """
    def _numbers(chunk, ncol):
        if not chunk:
            return np.empty((0, ncol))
        return np.fromstring(" ".join(chunk), sep=" ").reshape(-1, ncol)

    with open(path) as fh:
        lines = fh.read().splitlines()
    ints = [int(x) for x in lines[1].split(":")[0].split()]
    nkpt, nspinor, npoints, n1, n2, axis = ints
    reals = [float(x) for x in lines[2].split(":")[0].split()]
    efermi, height, area, omega = reals
    reals = [float(x) for x in lines[3].split(":")[0].split()]
    window = (reals[0], reals[1])
    polarization = reals[2]
    direction = np.array(reals[3:6])

    i = 5
    block = _numbers(lines[i:i + npoints], 5)
    points, points_lattice = block[:, :2], block[:, 2:]
    i += npoints

    kpoints = np.empty((nkpt, 3))
    weights = np.empty(nkpt)
    states, eigenvalues, overlaps, amplitudes = [], [], [], []
    for ik in range(nkpt):
        head = lines[i].split(":")[0].split()
        kpoints[ik] = [float(x) for x in head[:3]]
        weights[ik] = float(head[3])
        nsel = int(head[4])
        i += 1
        sel = _numbers(lines[i:i + nsel], 2)
        states.append(sel[:, 0].astype(int))
        eigenvalues.append(sel[:, 1])
        i += nsel
        n = nsel * nsel
        raw = _numbers(lines[i:i + n], 2)
        # written column by column (Fortran's own storage order)
        overlaps.append((raw[:, 0] + 1j * raw[:, 1]).reshape(nsel, nsel).T)
        i += n
        n = nspinor * nsel * npoints
        raw = _numbers(lines[i:i + n], 2)
        amplitudes.append(
            (raw[:, 0] + 1j * raw[:, 1]).reshape(nspinor, nsel, npoints)
        )
        i += n
    return {
        "nspinor": nspinor,
        "npoints": npoints,
        "grid": (n2, n1),
        "axis": axis,
        "efermi": efermi,
        "height": height,
        "area": area,
        "omega": omega,
        "window": window,
        "polarization": polarization,
        "direction": direction,
        "points": points,
        "points_lattice": points_lattice,
        "kpoints": kpoints,
        "weights": weights,
        "states": states,
        "eigenvalues": eigenvalues,
        "overlaps": overlaps,
        "amplitudes": amplitudes,
    }


def smeared_delta(x, stype=3):
    """Elk's own smeared delta function, src/sdelta.f90, as a function of the
    dimensionless argument x = (E - eps)/swidth.

    Transcribed rather than replaced by a convenient Gaussian so that the
    S -> identity limit reproduces the spin-polarised STM task's own LDOS
    (elkpy task 9003, docs/design.md #30) with NO factor between them -- that
    comparison is the one check a wrong overall normalisation could not hide
    in, since everything else here is a ratio, a correlation or a partition.

    Methfessel-Paxton of order 1 or 2 (`stype` 1, 2) is refused: it goes
    negative on its wings and an amplitude is its square root.
    """
    x = np.asarray(x, dtype=float)
    if stype == 0:                                    # Gaussian (MP order 0)
        return np.exp(-x**2) / np.sqrt(np.pi)
    if stype in (1, 2):
        raise ValueError(
            f"stype={stype} is Methfessel-Paxton of order {stype}, which goes "
            "negative on its wings; the on-shell tunnelling amplitude is the "
            "SQUARE ROOT of the delta, so it has no real value there. Use "
            "stype 0 (Gaussian), 3 (Fermi-Dirac, Elk's default), 4 (square) "
            "or 5 (Lorentzian)"
        )
    if stype == 3:                                    # Fermi-Dirac
        t = np.clip(np.abs(x), None, 200.0)
        e = np.exp(-t)
        return e / (1.0 + e)**2
    if stype == 4:                                    # square-wave
        return np.where(np.abs(x) <= 0.5, 1.0, 0.0)
    if stype == 5:                                    # Lorentzian
        return (1.0 / np.pi) / (x**2 + 1.0)
    raise ValueError(f"unknown Elk stype {stype!r}")


def amplitude_weights(eigenvalues, energy, broadening, stype=3, occmax=2.0):
    """sqrt(occmax delta_eta(E - eps) / eta): the real, non-negative factor
    multiplying psi_kn(r) before anything is squared.

    `occmax` is Elk's own spin-degeneracy factor -- 2 for nspinor=1, 1 for
    nspinor=2 -- carried here for exactly the reason src/occupy.f90 carries
    it, so that the whole-cell (S = identity) limit is the density of states
    and not half of it.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if broadening <= 0.0:
        raise ValueError(f"the broadening must be positive, got {broadening}")
    delta = smeared_delta((float(energy) - eigenvalues) / broadening, stype)
    return np.sqrt(occmax * delta / broadening)


def channel_basis(overlap, eigenvalues, tol=DEGENERACY_TOL):
    """A unitary diagonalising `overlap` inside each degenerate multiplet.

    The INCOHERENT map needs this and the coherent one does not. T_coherent is
    a quadratic form and is invariant under the rotation a degenerate
    eigensolver is free in (docs/design.md #14 documents that freedom for
    Elk's `evecsv` specifically); sum_n |a_n|^2 S[n,n] is a DIAGONAL, and the
    diagonal of an operator is not invariant. Reported as it stands, "how much
    interference there is" would depend on which basis Elk's diagonalisation
    happened to return inside a multiplet.

    Diagonalising S within each multiplet fixes that and gives the quantity
    its meaning back: the incoherent map becomes the substrate's own channels,
    each taken on its own, and the interference becomes the part that runs
    BETWEEN multiplets, which no choice of basis can rotate away.

    Where it makes no difference is worth knowing, because it is the common
    case and it is not luck: at a symmetry point the little group acts
    irreducibly on the multiplet, so by Schur's lemma any invariant operator
    restricted to it is a multiple of the identity and every basis is already
    a channel basis. Graphene's Dirac pair at K is exactly that case.
    """
    overlap = np.asarray(overlap)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    nst = eigenvalues.shape[0]
    rotation = np.eye(nst, dtype=complex)
    order = np.argsort(eigenvalues, kind="stable")
    start = 0
    while start < nst:
        stop = start + 1
        while stop < nst and eigenvalues[order[stop]] - eigenvalues[order[start]] < tol:
            stop += 1
        block = order[start:stop]
        if block.size > 1:
            sub = overlap[np.ix_(block, block)]
            _, vectors = np.linalg.eigh(0.5 * (sub + sub.conj().T))
            rotation[np.ix_(block, block)] = vectors
        start = stop
    return rotation


def transmission_at_k(amplitude, overlap, weights, coherent=True,
                      eigenvalues=None, tol=DEGENERACY_TOL):
    """One k-point's contribution, before its k-weight: shape (npoints,).

    `amplitude` is (nspinor, nst, npoints). A spinor's two components are two
    amplitude VECTORS, not one: the tip here is not spin-selective (the
    substrate is), so what tunnels in is the whole spinor and tracing the
    Landauer expression over the tip's spin gives sum_s a_s^dagger-form terms
    -- a SUM of two quadratic forms, not one form on a doubled vector. The two
    components leave through the same substrate but enter through independent
    tip channels.
    """
    amplitude = np.asarray(amplitude)
    overlap = np.asarray(overlap)
    weights = np.asarray(weights)
    nspinor, nst, npoints = amplitude.shape
    if overlap.shape != (nst, nst):
        raise ValueError(
            f"the overlap is {overlap.shape} and the amplitudes want {(nst, nst)}"
        )
    if weights.shape != (nst,):
        raise ValueError(f"state weights {weights.shape} do not match {nst} states")
    a = amplitude * weights[None, :, None]
    if coherent:
        # sum_nm a_n a*_m S[n,m] -- the conjugation on the SECOND factor,
        # because G(r, r') carries psi conjugated in the exit variable
        sa = np.einsum("nm,smp->snp", overlap, a.conj(), optimize=True)
        return np.real(np.einsum("snp,snp->p", a, sa, optimize=True))
    if eigenvalues is not None:
        u = channel_basis(overlap, eigenvalues, tol)
        a = np.einsum("ni,snp->sip", u, a, optimize=True)
        overlap = u.conj().T @ overlap @ u
    diagonal = np.real(np.diag(overlap))
    return np.einsum("n,snp->p", diagonal, np.real(a.conj() * a), optimize=True)


def compute_transmission(data, energies=None, broadening=0.005, stype=3,
                         exit_region="plane", incoherent=True,
                         degeneracy_tol=DEGENERACY_TOL):
    """The vertical transmission map, from `parse_transport`'s output.

    `energies` are absolute (Hartree); the default is the Fermi energy.
    `exit_region="cell"` replaces every S_k by the identity, which is the
    Tersoff-Hamann limit: T becomes sum_kn w_k |psi_kn(r)|^2 delta(E - eps),
    the tunnelling density of states at the tip, and so the SAME number
    elkpy's task-9003 STM image gives through an entirely separate Fortran
    path (rhomagv). It is the validation route that shares no line of code
    with the assembly it checks.

    Returns a dict: "transmission" (nE, npoints), "incoherent", "interference"
    and the diagnostics.
    """
    if exit_region not in ("plane", "cell"):
        raise ValueError(
            f"unknown exit_region {exit_region!r}: use 'plane' (the substrate) "
            "or 'cell' (the Tersoff-Hamann diagnostic)"
        )
    if energies is None:
        energies = [data["efermi"]]
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    occmax = 2.0 if data["nspinor"] == 1 else 1.0
    npoints = data["npoints"]

    total = np.zeros((energies.size, npoints))
    total_inc = np.zeros_like(total) if incoherent else None
    least, offdiag, channels, hermiticity = np.inf, [], [], 0.0
    for ik, (amp, ovl, eig) in enumerate(
        zip(data["amplitudes"], data["overlaps"], data["eigenvalues"])
    ):
        if eig.size == 0:
            continue
        if exit_region == "cell":
            ovl = np.eye(eig.size, dtype=complex)
        hermiticity = max(hermiticity, float(np.abs(ovl - ovl.conj().T).max()))
        hermitian = 0.5 * (ovl + ovl.conj().T)
        spectrum = np.linalg.eigvalsh(hermitian)
        least = min(least, float(spectrum.min()))
        # how many independent ways through the substrate there are: the
        # participation ratio of S_k's spectrum. Close to one in the vacuum,
        # where every band's evanescent tail has nearly the same shape on the
        # plane and differs only by a coefficient -- which is exactly why the
        # interference here is the answer rather than a correction
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
        wk = data["weights"][ik]
        for ie, energy in enumerate(energies):
            w = amplitude_weights(eig, energy, broadening, stype, occmax)
            total[ie] += wk * transmission_at_k(amp, ovl, w, True)
            if incoherent:
                total_inc[ie] += wk * transmission_at_k(
                    amp, ovl, w, False, eig, degeneracy_tol
                )
    return {
        "energies": energies,
        "transmission": total,
        "incoherent": total_inc,
        "interference": None if total_inc is None else total - total_inc,
        # 0.0, not None, when no k-point carried a state in the window: this
        # is compared against a tolerance by every caller
        "least_eigenvalue": 0.0 if least == np.inf else least,
        "hermiticity": hermiticity,
        "offdiagonal_weight": float(np.mean(offdiag)) if offdiag else 0.0,
        "channels": float(np.mean(channels)) if channels else 0.0,
    }


def green_function_transmission(tip_amplitude, exit_amplitude, area, weights,
                                kweight=1.0):
    """T at the tip points by evaluating the DEFINITION, with a quadrature.

    Builds G(r, r') = sum_n a_n(r) psi*_n(r') on the exit-plane sample points
    literally and integrates |G|^2 over the plane by the rectangle rule -- which
    is exact for this integrand once the sampling exceeds twice the largest
    in-plane G-vector index, the integrand being periodic and band-limited.

    This is what discriminates a_n a*_m S[n,m] from a*_n a_m S[n,m]. The two
    differ by a transpose of a Hermitian matrix, so the wrong one is real,
    non-negative, blind to a degenerate rotation, and exactly right in the
    Tersoff-Hamann limit where S is the identity: nothing but a literal check
    against the definition can tell them apart.

    `tip_amplitude` and `exit_amplitude` are (nspinor, nst, npts) samples of
    the same states at the tip points and at the exit-plane points.
    """
    tip = np.asarray(tip_amplitude) * np.asarray(weights)[None, :, None]
    exit_ = np.asarray(exit_amplitude)
    npx = exit_.shape[2]
    g = np.einsum("snp,snq->spq", tip, exit_.conj(), optimize=True)
    return kweight * (area / npx) * np.real(np.einsum("spq,spq->p", g, g.conj()))


def gram_matrix_by_quadrature(exit_amplitude, area, projector=None):
    """S_k[n, n'] = int_plane psi*_n P psi_n' by the same rectangle rule.

    The independent check on the closed-form G-parallel collapse
    src/elkpy_transport.f90 uses, which involves no quadrature at all.
    """
    psi = np.asarray(exit_amplitude)
    nspinor, nst, npx = psi.shape
    if projector is None:
        projector = np.eye(nspinor)
    projector = np.asarray(projector)
    s = np.zeros((nst, nst), dtype=complex)
    for a in range(nspinor):
        for b in range(nspinor):
            s += projector[a, b] * (psi[a].conj() @ psi[b].T)
    return (area / npx) * s
