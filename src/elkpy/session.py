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
    parse_eigenstates_response,
    parse_overlap_response,
    parse_projection_response,
)
from .parsers.spin import compute_spin_operator

READY_SENTINEL = "ELKPY_SESSION_READY"
END_SENTINEL = "ELKPY_SESSION_END"
ERROR_PREFIX = "ELKPY_SESSION_ERROR"

Eigenstates = namedtuple("Eigenstates", ["k", "energies", "evecsv"])
AtomProjection = namedtuple("AtomProjection", ["k", "matrices"])
SpinOperator = namedtuple("SpinOperator", ["k", "sx", "sy", "sz"])


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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
