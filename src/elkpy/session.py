"""EigenstateSession: an interactive, long-lived elk subprocess (task 9002)
for fast repeated eigenstate/overlap queries.

See docs/design.md #14 for why this is a persistent worker process (one
long-lived subprocess, stdin/stdout query loop) rather than an f2py
in-memory bridge -- in short, the dominant per-query cost is the
ground-state-dependent setup Elk must redo per subprocess, not process-spawn
overhead, and a worker process captures that same "stay warm" benefit
without needing a new -fPIC/shared-library build or converting Elk's
pervasive bare `stop` calls into catchable errors (which can't be done
additively). Calculation.eigenstate_session() is how one of these is
created.
"""

from collections import namedtuple

from .parsers.eigenstates import (
    parse_angular_momentum_response,
    parse_eigenstates_response,
    parse_momentum_response,
    parse_orbital_projection_response,
    parse_parity_response,
    parse_symlist_response,
    parse_symmetry_response,
    parse_overlap_response,
    parse_projection_response,
)
from .parsers.spin import compute_spin_operator
from .parsers.spin_hall import spin_current_operator as _spin_current_operator
from .parsers.symmetry import is_trim

READY_SENTINEL = "ELKPY_SESSION_READY"
END_SENTINEL = "ELKPY_SESSION_END"
ERROR_PREFIX = "ELKPY_SESSION_ERROR"

# l order OrbitalProjection.matrices' second axis uses -- s,p,d,f, matching
# elkpy_orbitalproj's pmat(:,:,0:3) and Elk's own lmaxdb=3 "band character"
# default (bandstr.f90 task 21).
ORBITAL_LABELS = ("s", "p", "d", "f")

Eigenstates = namedtuple("Eigenstates", ["k", "energies", "evecsv"])
AtomProjection = namedtuple("AtomProjection", ["k", "matrices"])
OrbitalProjection = namedtuple("OrbitalProjection", ["k", "matrices"])
SpinOperator = namedtuple("SpinOperator", ["k", "sx", "sy", "sz"])
Momentum = namedtuple("Momentum", ["k", "energies", "pmat", "evecsv"])
Parity = namedtuple("Parity", ["k", "energies", "pmat"])
SymmetryOperator = namedtuple("SymmetryOperator", ["k", "isym", "energies", "smat"])
AngularMomentum = namedtuple(
    "AngularMomentum", ["k", "lx", "ly", "lz", "lx_orbital", "ly_orbital", "lz_orbital"]
)


def _fmt(x):
    # Fortran-friendly decimal notation (always a decimal point or
    # exponent), full double precision.
    return f"{float(x):.17g}"


class EigenstateSession:
    """A live elk process (task 9002) parked in its interactive query loop.

    Use as a context manager to guarantee the process is asked to quit (and,
    failing that, killed) even if an exception is raised mid-session:

        with calc.eigenstate_session() as session:
            e1 = session.get_eigenstates((0, 0, 0))
            e2 = session.get_eigenstates((0.1, 0, 0))
            m = session.overlap((0, 0, 0), (0.1, 0, 0), ist0=1, ist1=4)

    Every query after the first pays only the diagonalisation cost, not the
    init0/init1/readstate/genvsig/linengy/genapwlofr/gensocfr setup Elk
    would otherwise redo per subprocess -- that setup happens once, before
    the session becomes ready. Prefer this over calling
    Calculation.get_eigenstates()/get_overlap() in a loop, which each open
    and close their own short-lived session.

    A query that reaches a Fortran `stop` deep in reused code (e.g. a
    pathological/near-singular overlap) still terminates the whole session --
    an accepted limitation (docs/design.md #14), detected here as an
    unexpected process exit and surfaced as a clear RuntimeError rather than
    a hang. Start a new session to continue after that.
    """

    def __init__(self, proc, workdir, nspinor=1):
        self._proc = proc
        self.workdir = workdir
        self._closed = False
        self._nspinor = nspinor
        self._wait_ready()

    def _send(self, line):
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"eigenstate session process exited unexpectedly (code "
                f"{self._proc.returncode}) in {self.workdir} -- likely a query that hit "
                f"an unrecoverable Fortran error (see docs/design.md #14); start a new "
                f"session to continue"
            )
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(
                f"eigenstate session process exited unexpectedly (code "
                f"{self._proc.poll()}) in {self.workdir} while sending a query -- likely a "
                f"prior query that hit an unrecoverable Fortran error (see "
                f"docs/design.md #14); start a new session to continue"
            ) from exc

    def _drain_fatal_error(self, first_line):
        """A line starting with Elk's own "Error(...)" convention means a
        `stop` is imminent (e.g. a pathological/near-singular query reaching
        a pre-existing `stop` deep in reused code -- docs/design.md #14).
        Reads until the process exits and raises with the full diagnostic,
        rather than letting the caller's sentinel-matching loop absorb error
        text as bogus numeric tokens or hit a bare, message-less EOF."""
        lines = [first_line]
        while True:
            raw = self._proc.stdout.readline()
            if raw == "":
                break
            lines.append(raw.rstrip("\n"))
        message = "\n".join(line for line in lines if line.strip())
        raise RuntimeError(
            f"eigenstate session process reported a fatal error in {self.workdir}:\n"
            f"{message}"
        )

    def _read_until_sentinel(self):
        tokens = []
        while True:
            raw = self._proc.stdout.readline()
            if raw == "":
                raise RuntimeError(
                    f"eigenstate session process ended without completing its response "
                    f"(exit code {self._proc.poll()}) in {self.workdir} -- see "
                    f"docs/design.md #14"
                )
            stripped = raw.strip()
            if stripped == END_SENTINEL:
                return tokens
            if stripped.startswith(ERROR_PREFIX):
                raise ValueError(stripped[len(ERROR_PREFIX) :].strip())
            if stripped.startswith("Error("):
                self._drain_fatal_error(stripped)
            tokens.extend(stripped.split())

    def _wait_ready(self):
        while True:
            raw = self._proc.stdout.readline()
            if raw == "":
                raise RuntimeError(
                    f"eigenstate session process exited before becoming ready (exit code "
                    f"{self._proc.poll()}) in {self.workdir}; see {self.workdir}/elk.out "
                    f"if present, else check the process's own stderr"
                )
            stripped = raw.strip()
            if stripped == READY_SENTINEL:
                return
            if stripped.startswith("Error("):
                self._drain_fatal_error(stripped)

    def get_eigenstates(self, k):
        """Second-variational energies (Hartree) and eigenvectors (evecsv,
        orthonormal spinor basis) at an arbitrary k-point (fractional
        lattice coordinates), via fresh on-the-fly diagonalisation -- same
        method get_berry_curvature_path() uses per point, not a read from a
        previously-diagonalised mesh.

        Returns an Eigenstates(k, energies, evecsv) namedtuple: energies
        shape (nstsv,) Hartree, evecsv shape (nstsv, nstsv) complex,
        evecsv[:, i] the i-th eigenvector's coefficients. See
        parsers.eigenstates.parse_eigenstates_response for what evecsv is
        (and is not) valid to compare against -- use overlap() for anything
        beyond inspecting a single query's own results.
        """
        k = tuple(float(x) for x in k)
        self._send(f"EIGENSTATES {_fmt(k[0])} {_fmt(k[1])} {_fmt(k[2])}")
        tokens = self._read_until_sentinel()
        energies, evecsv = parse_eigenstates_response(tokens)
        return Eigenstates(k=k, energies=energies, evecsv=evecsv)

    def overlap(self, k_a, k_b, ist0, ist1):
        """<psi_a(k_a)|psi_b(k_b)> for the contiguous, 1-indexed
        second-variational band window [ist0, ist1], via real-space
        wavefunction expansion + overlap integral at each k-point (the same
        genwfsvp-style expansion plus genolpq construction
        get_berry_curvature()/get_berry_curvature_path() already use) -- the
        only valid way to compare eigenstates across k-points, or across
        separate diagonalisations at the same k (see docs/design.md #14).

        Returns an (nst, nst) complex array, nst = ist1 - ist0 + 1;
        M[a, b] = <psi_{ist0+a}(k_a)|psi_{ist0+b}(k_b)>.
        """
        k_a = tuple(float(x) for x in k_a)
        k_b = tuple(float(x) for x in k_b)
        self._send(
            f"OVERLAP {_fmt(k_a[0])} {_fmt(k_a[1])} {_fmt(k_a[2])} "
            f"{_fmt(k_b[0])} {_fmt(k_b[1])} {_fmt(k_b[2])} {int(ist0)} {int(ist1)}"
        )
        tokens = self._read_until_sentinel()
        return parse_overlap_response(tokens)

    def atom_projection(self, k, ist0, ist1):
        """The atom-projection operator P_alpha, restricted to atom alpha's
        muffin-tin sphere, for EVERY atom alpha in the cell at once, in the
        contiguous 1-indexed second-variational band window [ist0, ist1] --
        via elkpy_atomproj (src/elkpy_eigenstates.f90, reusing wfmtsv/
        wr2cmt, the same muffin-tin machinery upstream gendmatk.f90 uses for
        the dos/bandstr tasks' atom/lm-projected DOS and band-character
        output). See docs/design.md #16 and docs/physics.tex Part V for the
        physics.

        Returns an AtomProjection(k, matrices) namedtuple: matrices shape
        (natmtot, nst, nst) complex, nst = ist1 - ist0 + 1, in Fortran's
        global 1-based atom order (species in order, then atoms within each
        species in order -- see get_forces()'s docstring for the same
        convention; Calculation.get_atom_projection() resolves a (species,
        index) pair to this array's index for you).

        matrices[a] is Hermitian and positive semi-definite:
        matrices[a][i, i] is state (ist0+i)'s fractional weight on atom a's
        muffin-tin sphere alone (meaningful on its own for a non-degenerate
        state); matrices[a][i, j] (i != j) mixes two different eigenstates
        of THIS one query's diagonalisation only. All natmtot matrices
        returned by one call share that same diagonalisation, so they may
        be validly combined with each other (e.g. summed and compared
        against the identity minus the interstitial remainder) -- but, as
        with evecsv generally (docs/design.md #14), do not combine a matrix
        from one atom_projection()/get_atom_projection() call with an
        eigenvector or matrix from a SEPARATE call, even at the same k:
        nothing here guarantees two independent diagonalisations picked the
        same internal basis.
        """
        k = tuple(float(x) for x in k)
        self._send(f"PROJECTION {_fmt(k[0])} {_fmt(k[1])} {_fmt(k[2])} {int(ist0)} {int(ist1)}")
        tokens = self._read_until_sentinel()
        return AtomProjection(k=k, matrices=parse_projection_response(tokens))

    def orbital_projection(self, k, ist0, ist1):
        """The l-resolved atom-projection operators P_{alpha,l} for l=0,1,2,3
        (s, p, d, f -- ORBITAL_LABELS), for EVERY atom alpha in the cell at
        once, in the contiguous 1-indexed second-variational band window
        [ist0, ist1] -- via elkpy_orbitalproj (src/elkpy_eigenstates.f90,
        reusing the same wfmtsv/wr2cmt muffin-tin machinery as
        atom_projection(), but resolved by l -- summed over m and spin only
        -- rather than summed over every (l, m) up to lmaxo). Note this
        projects onto an angular-momentum CHANNEL within the muffin-tin
        sphere (all radii, all principal quantum numbers of that l), not
        onto a specific atomic valence orbital's own radial shape -- l=1 on
        a transition-metal atom, say, includes semicore p states too, not
        only the outermost p shell. See docs/design.md #18 and
        docs/physics.tex Part VII for the physics.

        Returns an OrbitalProjection(k, matrices) namedtuple: matrices shape
        (natmtot, 4, nst, nst) complex, nst = ist1 - ist0 + 1, atom axis in
        Fortran's global 1-based atom order (same convention as
        atom_projection()/get_forces()), l axis 0..3 = s,p,d,f
        (ORBITAL_LABELS).

        matrices[a, l] is Hermitian and positive semi-definite; summed over
        l=0..3 it falls short of atom_projection()'s matrices[a] (which
        sums to Elk's own lmaxo, 6 by default) by the l=4,5,6 (g,h,i)
        weight -- not a defect, the same "not the whole atom" partiality
        atom_projection() itself has relative to the full identity. All
        4*natmtot matrices from one call share ONE diagonalisation and, per
        atom, one wfmtsv call -- see elkpy_orbitalproj's docstring for the
        precise gauge caveat (same spirit as atom_projection()'s, one level
        more specific: the four l channels of a single atom are exactly
        mutually consistent, not merely reproducibly so).
        """
        k = tuple(float(x) for x in k)
        self._send(f"ORBITAL {_fmt(k[0])} {_fmt(k[1])} {_fmt(k[2])} {int(ist0)} {int(ist1)}")
        tokens = self._read_until_sentinel()
        return OrbitalProjection(k=k, matrices=parse_orbital_projection_response(tokens))

    def angular_momentum(self, k, ist0, ist1):
        """The (orbital) angular momentum operators L_x, L_y, L_z, l-resolved
        for l=0,1,2,3 (s, p, d, f -- ORBITAL_LABELS) and restricted to atom
        alpha's muffin-tin sphere, for EVERY atom alpha in the cell at once,
        in the contiguous 1-indexed second-variational band window
        [ist0, ist1] -- via elkpy_angmomproj (src/elkpy_eigenstates.f90),
        the vector-operator sibling of orbital_projection(): where
        orbital_projection() reduces each l shell to a scalar weight,
        L_x/L_y/L_z mix m within that shell (the ladder-operator structure
        L_+-|l,m> propto |l,m+-1>), reusing upstream Elk's own lopzflm
        subroutine (unmodified -- the same one Elk's own on-site L.S trace,
        dmatls.f90, already uses) for the ladder-operator matrix elements.
        This is the ORBITAL angular momentum, not spin -- see
        spin_operator() for S_a. See docs/design.md #19 and
        docs/physics.tex Part VIII for the physics.

        Returns an AngularMomentum(k, lx, ly, lz, lx_orbital, ly_orbital,
        lz_orbital) namedtuple. lx_orbital/ly_orbital/lz_orbital have shape
        (natmtot, 4, nst, nst) complex (atom axis in Fortran's global
        1-based order, l axis 0..3 = s,p,d,f); lx/ly/lz are their l=0..3 sum,
        shape (natmtot, nst, nst) -- the headline per-atom operator, summed
        in Python (not a separate Fortran query), so -- unlike
        atom_projection()'s sum to Elk's own lmaxo (6 by default) -- this
        total only covers l=0..3, the same g/h/i-excluded partiality
        orbital_projection()'s own sum-of-four-l's already has relative to
        atom_projection().

        Each matrix is Hermitian, but -- unlike atom_projection()/
        orbital_projection() -- NOT positive semi-definite in general (an
        angular momentum expectation value can be negative). Two algebraic
        identities of the analytic, untruncated (2l+1)x(2l+1) operators,
        [L_x, L_y] = i*L_z and L_x^2 + L_y^2 + L_z^2 = l(l+1)*1, do NOT carry
        over to these nst x nst matrices as matrix products: both require a
        resolution of identity over every state, not just the requested
        band window, so evaluating them directly on lx/ly/lz here will not
        reproduce the analytic result -- expected truncation behaviour, not
        a bug (see tests/test_parsers_angular_momentum.py, which verifies
        the identities on the untruncated analytic matrices instead).
        """
        k = tuple(float(x) for x in k)
        self._send(f"ANGMOM {_fmt(k[0])} {_fmt(k[1])} {_fmt(k[2])} {int(ist0)} {int(ist1)}")
        tokens = self._read_until_sentinel()
        matrices = parse_angular_momentum_response(tokens)
        lx_orbital = matrices[:, :, 0]
        ly_orbital = matrices[:, :, 1]
        lz_orbital = matrices[:, :, 2]
        return AngularMomentum(
            k=k,
            lx=lx_orbital.sum(axis=1),
            ly=ly_orbital.sum(axis=1),
            lz=lz_orbital.sum(axis=1),
            lx_orbital=lx_orbital,
            ly_orbital=ly_orbital,
            lz_orbital=lz_orbital,
        )

    def momentum(self, k, ist0=None, ist1=None):
        """The momentum matrix elements p^a_nm = <psi_n|p_a|psi_m>,
        a = x, y, z, for every pair of second-variational states at an
        arbitrary k-point, together with the eigenvalues of that same
        diagonalisation -- via elkpy_momentum (src/elkpy_eigenstates.f90,
        calling upstream genpmatk unmodified, the same subroutine Elk's own
        task-120 PMAT.OUT export uses, but at an arbitrary k-point rather
        than a previously-diagonalised mesh point). See docs/design.md #22
        and docs/physics.tex Part XI for the physics.

        In Hartree atomic units (hbar = m_e = 1) and for Elk's LOCAL
        Kohn-Sham potential, p is numerically the velocity operator v, so
        this is equally the velocity matrix; genpmatk includes the
        (1/4c^2)[sigma x grad V_s] spin-orbit correction that keeps that
        identity true under spinorb=True.

        Returns a Momentum(k, energies, pmat, evecsv) namedtuple: energies
        shape (nstsv,) Hartree, pmat shape (3, nstsv, nstsv) complex (atomic
        units), pmat[a] the a-th CARTESIAN component (a = 0, 1, 2 for
        x, y, z -- note that parsers.optical's `directions` argument
        indexes these Cartesian axes, unlike get_berry_curvature()'s
        identically-named argument, which indexes reciprocal lattice
        axes), and evecsv shape (nstsv, nstsv) complex.

        `evecsv` is the second-variational eigenvector matrix of the SAME
        diagonalisation that produced `pmat` -- which is what makes it
        legal to multiply an evecsv-derived operator into pmat. The spin
        operators S_x, S_y, S_z are exactly that (parsers.spin builds them
        from evecsv's spin-up/spin-down row blocks and nothing else), so
        this is what the spin current operator J^z_a = (1/2){S_z, v_a} and
        hence the spin Berry curvature / spin Hall conductivity of
        parsers.spin_hall need. Using spin_operator()'s own evecsv instead
        would pair matrices from two independent diagonalisations, free to
        resolve a degenerate multiplet differently (docs/design.md #14) --
        a failure no Hermiticity or unitarity check detects. Prefer
        parsers.spin.compute_spin_operator(m.evecsv, ...) or the
        spin_current_operator() convenience below over a separate
        spin_operator() query when combining with pmat.

        `ist0`/`ist1` optionally restrict the returned arrays to a
        contiguous, 1-indexed band window; both default to None, meaning
        all nstsv states, which is what the Kubo-form sums in
        parsers.optical need (they run over states OUTSIDE the window of
        interest, so a pre-windowed pmat is the wrong object for them --
        pass the window to those functions instead, not here). Unlike
        every other query here the window is applied in Python, not
        Fortran: genpmatk's array is hard-dimensioned nstsv, so there is
        no windowed variant to ask for.

        pmat, energies and evecsv come from ONE diagonalisation, which is
        what makes it safe to use these energies as the denominators of a
        sum over these matrix elements -- do not substitute a separate
        get_eigenstates() call's energies (docs/design.md #14).

        The band window, when given, slices `energies` and `pmat` only;
        `evecsv` is always returned in full. Its ROW index is the
        first-variational spinor basis (i = p + (ispn-1)*nstfv), which a
        band window has no meaning for, and parsers.spin.compute_spin_
        operator() applies the window itself -- truncating the rows here
        would break its nstsv == 2*nstfv check.
        """
        k = tuple(float(x) for x in k)
        self._send(f"MOMENTUM {_fmt(k[0])} {_fmt(k[1])} {_fmt(k[2])}")
        tokens = self._read_until_sentinel()
        energies, evecsv, pmat = parse_momentum_response(tokens)
        if ist0 is not None or ist1 is not None:
            nstsv = len(energies)
            lo = 1 if ist0 is None else int(ist0)
            hi = nstsv if ist1 is None else int(ist1)
            if not (1 <= lo <= hi <= nstsv):
                raise ValueError(
                    f"invalid band window (ist0={lo}, ist1={hi}, nstsv={nstsv})"
                )
            energies = energies[lo - 1 : hi]
            pmat = pmat[:, lo - 1 : hi, lo - 1 : hi]
        return Momentum(k=k, energies=energies, pmat=pmat, evecsv=evecsv)

    def spin_current_operator(self, k, direction=1, spin="z"):
        """The spin current operator J^s_a = (1/2){S_s, v_a} at `k`, as an
        (nstsv, nstsv) Hermitian matrix, together with the energies and
        momentum matrix it was built from -- one MOMENTUM query and no
        other Fortran work.

        This exists so the one-diagonalisation rule above cannot be got
        wrong by accident: S_s and v_a here are guaranteed to share an
        eigenbasis, since both come from the single MOMENTUM response.
        See docs/design.md #24 and parsers.spin_hall.

        `direction` is a 1-based CARTESIAN axis (1, 2, 3 = x, y, z),
        matching parsers.optical's convention; `spin` is "x", "y" or "z".

        Returns (Momentum, j) -- the Momentum namedtuple (unwindowed) and
        the spin current matrix, so the caller can feed both straight to
        parsers.spin_hall.spin_berry_curvature().
        """
        if self._nspinor != 2:
            raise ValueError(
                f"the spin current operator requires a spin-polarized calculation "
                f"(spinpol=True or spinorb=True), got nspinor={self._nspinor} -- there "
                f"is no spin-up/spin-down block to build S_a from"
            )
        if spin not in ("x", "y", "z"):
            raise ValueError(f"spin must be 'x', 'y' or 'z', got {spin!r}")
        m = self.momentum(k)
        nstsv = m.evecsv.shape[0]
        nstfv = nstsv // self._nspinor
        # the whole point: this evecsv and this pmat are ONE diagonalisation
        s = compute_spin_operator(m.evecsv, nstfv, 1, nstsv)[f"s{spin}"]
        return m, _spin_current_operator(m.pmat, s, direction=direction)

    def parity(self, k, ist0, ist1):
        """The inversion (parity) operator P_mn = <psi_m|I|psi_n> over the
        contiguous, 1-indexed band window [ist0, ist1], plus all nstsv
        eigenvalues of the same diagonalisation -- via elkpy_parity
        (src/elkpy_eigenstates.f90, lifting the symmetry transformation of
        the first-variational coefficients from upstream getevecfv.f90 and
        reusing genwfsv/genolpq at q=0 for the overlap). See
        docs/design.md #23 and docs/physics.tex Part XII.

        Defined ONLY at a time-reversal-invariant momentum (every
        fractional component 0 or 1/2), where inversion maps k to itself
        modulo a reciprocal lattice vector; elsewhere it maps the state to
        -k and no such matrix exists. This is checked here before the query
        is sent (parsers.symmetry.is_trim), and independently in Fortran,
        which detects it as a G+k vector with no partner in the same basis.
        Requires a crystal with an inversion centre (Elk's `tsyminv`);
        without one the query raises ValueError rather than returning
        something meaningless.

        Returns a Parity(k, energies, pmat) namedtuple: energies shape
        (nstsv,) Hartree (ALL states, so the degeneracy structure around
        the window is visible), pmat shape (nst, nst) complex.

        Because [I, H] = 0, inversion preserves energy eigenspaces, so a
        window gapped from the rest of the spectrum is I-invariant: P is
        Hermitian with P^2 = 1 and eigenvalues exactly +-1. Note this is
        NOT spoiled by the band-window truncation, unlike the su(2) and
        Casimir identities of angular_momentum() (docs/design.md #19) --
        those need a resolution of identity over every state, whereas an
        invariant window supplies its own. Use
        parsers.symmetry.parity_eigenvalues() rather than reading pmat's
        diagonal: TRIM spectra are heavily degenerate and the diagonal
        entries are basis-dependent within a multiplet.
        """
        k = tuple(float(x) for x in k)
        if not is_trim(k):
            raise ValueError(
                f"parity is only defined at a time-reversal-invariant momentum "
                f"(every fractional component 0 or 1/2), got {k} -- inversion maps any "
                f"other k-point to -k, a different point"
            )
        self._send(
            f"PARITY {_fmt(k[0])} {_fmt(k[1])} {_fmt(k[2])} {int(ist0)} {int(ist1)}"
        )
        tokens = self._read_until_sentinel()
        energies, pmat = parse_parity_response(tokens)
        return Parity(k=k, energies=energies, pmat=pmat)

    def symmetries(self):
        """Every crystal symmetry Elk found, as a list of dicts
        {"isym", "rotation" (3x3 integer, LATTICE coordinates),
        "translation", "symmorphic"} -- see
        parsers.eigenstates.parse_symlist_response.

        Cheap: no diagonalisation, just Elk's already-computed space group.
        Use it to find the operation you want (e.g. the C_3 rotation) and
        which k-points it leaves invariant, then pass its `isym` to
        symmetry_operator(). Working in lattice coordinates means R.k == k
        is an exact integer test rather than a tolerance on a Cartesian
        rotation.
        """
        self._send("SYMLIST")
        tokens = self._read_until_sentinel()
        nspinor, ops = parse_symlist_response(tokens)
        self._symlist_nspinor = nspinor
        return ops

    def symmetry_operator(self, k, isym, ist0, ist1):
        """The matrix <psi_m|O_isym|psi_n> of crystal symmetry `isym` over
        the band window [ist0, ist1], plus that diagonalisation's
        eigenvalues -- the generalisation of parity() from inversion to any
        space-group element. See docs/design.md #28.

        `k` must be invariant under the operation (R.k == k modulo a
        reciprocal lattice vector); use symmetries() to find which
        operations fix a given k.

        Two restrictions, both enforced in Fortran rather than trusted here,
        because violating either returns a plausible but wrong matrix that
        unitarity and O^n = 1 would not catch:

          * nspinor must be 1. A spatial rotation on a spinor also needs the
            SU(2) spin rotation, which upstream's first-variational
            transformation never applies. Inversion escaped this (it acts
            trivially on spin); a rotation does not.
          * the operation must have zero translation. For a glide or screw,
            O^n is a k-dependent phase times unity rather than unity, so a
            caller binning eigenvalues into n-th roots of unity would
            misread it.

        Returns a SymmetryOperator(k, isym, energies, smat) namedtuple.
        """
        k = tuple(float(x) for x in k)
        self._send(
            f"SYMMETRY {int(isym)} {_fmt(k[0])} {_fmt(k[1])} {_fmt(k[2])} "
            f"{int(ist0)} {int(ist1)}"
        )
        tokens = self._read_until_sentinel()
        energies, smat = parse_symmetry_response(tokens)
        return SymmetryOperator(k=k, isym=int(isym), energies=energies, smat=smat)

    def spin_operator(self, k, ist0, ist1):
        """The spin operators S_x, S_y, S_z (hbar=1, so eigenvalues +-1/2
        for a pure spin state) as Hermitian nst x nst matrices in the
        contiguous 1-indexed band window [ist0, ist1], applicable to any
        wavefunction in that window -- the spin-space analogue of
        atom_projection(). See docs/design.md #17 and docs/physics.tex
        Part VI for the physics.

        Unlike atom_projection(), this issues NO new query to the Fortran
        session: evecsv already returned by an EIGENSTATES query fully
        determines S_a (the second-variational spinor basis makes S_a
        block-diagonal in the first-variational spatial index -- see
        parsers.spin.compute_spin_operator's docstring), so this is a plain
        get_eigenstates(k) call plus Python-side linear algebra.

        Raises ValueError if this calculation is not spin-polarized
        (nspinor=1, i.e. neither spinpol=True nor spinorb=True was set) --
        there is no spin-up/spin-down block to build S_a from.

        Returns a SpinOperator(k, sx, sy, sz) namedtuple, each of sx/sy/sz
        an (nst, nst) complex Hermitian array, nst = ist1 - ist0 + 1.
        """
        if self._nspinor != 2:
            raise ValueError(
                f"spin operators require a spin-polarized calculation "
                f"(spinpol=True or spinorb=True), got nspinor={self._nspinor} -- there is "
                f"no spin-up/spin-down block to build sx/sy/sz from"
            )
        state = self.get_eigenstates(k)
        nstfv = state.evecsv.shape[0] // self._nspinor
        ops = compute_spin_operator(state.evecsv, nstfv, ist0, ist1)
        return SpinOperator(k=state.k, **ops)

    def close(self, timeout=30):
        """Ask the session to quit and wait for it to exit; idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._proc.wait(timeout=timeout)
            except Exception:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=timeout)
                except Exception:
                    self._proc.kill()
                    self._proc.wait()
        self._proc.stdin.close()
        self._proc.stdout.close()
        # give back the launcher's machine-wide concurrency slot (see
        # launcher._acquire_slot) -- a live session occupies a core for as
        # long as it is answering queries, so it holds one for its lifetime
        from .launcher import _release_slot

        _release_slot(getattr(self._proc, "_elkpy_slot", None))
        self._proc._elkpy_slot = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
