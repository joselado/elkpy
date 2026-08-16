"""Execution of the Elk binary.

See docs/design.md #6: this is deliberately a small, separate object so a
scheduler-backed launcher (SLURM/PBS) can be added later without reshaping
Calculation. LocalLauncher is the only implementation for now: a blocking
local subprocess call.
"""

import fcntl
import os
import subprocess
import time

from . import config

# --- machine-wide concurrency cap -------------------------------------------
# Every elk run in elkpy goes through this module, so a counting semaphore
# here bounds total CPU use across ALL concurrent elkpy processes -- including
# separate Python interpreters that know nothing about each other (e.g.
# several agents, or several terminals, driving the same machine).
#
# The launcher already pins OMP_NUM_THREADS=1 (below), so one elk process is
# one core and "N slots" means "N cores". OpenBLAS and MKL are pinned too:
# Elk links -lopenblas, whose threading is NOT governed by OMP_NUM_THREADS
# when it is built against pthreads rather than OpenMP, so without this a
# single elk could still fan a zgemm out across every core.
#
# Tunable via the environment, so a machine with cores to spare is not
# throttled by a default chosen elsewhere:
#   ELKPY_MAX_CONCURRENT  number of simultaneous elk processes (default 4)
#   ELKPY_SLOT_DIR        where the lock files live (default /tmp/elkpy_slots)
SLOT_DIR = os.environ.get("ELKPY_SLOT_DIR", "/tmp/elkpy_slots")
MAX_CONCURRENT = int(os.environ.get("ELKPY_MAX_CONCURRENT", "4"))


def _acquire_slot(poll=2.0):
    """Block until one of MAX_CONCURRENT slots is free; return the locked fd.

    The lock is an flock on a per-slot file, so it is released automatically
    if the holding process dies -- a crashed or killed run cannot leak a slot
    and starve the machine, which a counter file or a directory of PID files
    would both get wrong.

    MAX_CONCURRENT <= 0 disables the cap entirely (returns None).
    """
    if MAX_CONCURRENT <= 0:
        return None
    os.makedirs(SLOT_DIR, exist_ok=True)
    while True:
        for i in range(MAX_CONCURRENT):
            fd = os.open(os.path.join(SLOT_DIR, f"slot{i}"), os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except OSError:
                os.close(fd)
        time.sleep(poll)


def _release_slot(fd):
    """Release a slot acquired by _acquire_slot(); tolerant of None/double
    release, since callers free it on several different paths."""
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _thread_pinned_env(omp_threads):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(omp_threads)
    # see SLOT_DIR comment: OpenBLAS/MKL have their own thread controls
    env["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    env["MKL_NUM_THREADS"] = str(omp_threads)
    return env


class LocalLauncher:
    """Runs `elk` as a blocking local subprocess in a given directory."""

    def __init__(self, elk_binary=None, nprocs=1, omp_threads=1):
        self.elk_binary = config.resolve_elk_binary(elk_binary)
        if nprocs > 1:
            # build-config/make.inc builds with mpi_stub.f90 (no real MPI) --
            # `mpirun -np N` against that binary launches N independent
            # serial copies into the same directory, all writing the same
            # *.OUT files with no error. Fails loudly here instead of
            # silently producing garbage output.
            raise ValueError(
                "nprocs > 1 requires an MPI-enabled elk build; the default "
                "build-config/make.inc uses mpi_stub.f90 (serial). Build "
                "with a real MPI compiler/make.inc before requesting nprocs > 1."
            )
        self.nprocs = nprocs
        self.omp_threads = omp_threads

    def run(self, workdir, log_name="elk.out"):
        """Run elk in `workdir`, writing stdout/stderr to `log_name`.

        Returns the path to the log file. Raises RuntimeError if the process
        exits non-zero. Elk itself prints "Elk code stopped" and exits 0 even
        on some internal errors/non-convergence it detects, so callers should
        still check INFO.OUT for correctness (see parsers/info.py), not just
        the return code.
        """
        env = _thread_pinned_env(self.omp_threads)

        command = [str(self.elk_binary)]

        log_path = workdir / log_name
        slot = _acquire_slot()
        try:
            with open(log_path, "w") as log:
                result = subprocess.run(
                    command, cwd=str(workdir), stdout=log, stderr=subprocess.STDOUT, env=env
                )
        finally:
            _release_slot(slot)
        if result.returncode != 0:
            raise RuntimeError(
                f"elk exited with code {result.returncode} in {workdir}; see {log_path}"
            )
        return log_path

    def start_session(self, workdir):
        """Start elk in `workdir` as a non-blocking, interactive subprocess
        (stdin/stdout pipes), for tasks that stay alive across many queries
        instead of running once to completion -- currently only the
        eigenstate/overlap session (task 9002, see src/elkpy/session.py).

        Unlike run(), this does not block until the process exits, does not
        write a log file (stdout is a pipe the caller reads directly), and
        does not raise on a bad exit code -- the caller (EigenstateSession)
        owns the process lifecycle and must detect an unexpected exit itself
        (e.g. a query that hits a Fortran `stop` deep in reused code; see
        docs/design.md #14).

        Returns the `subprocess.Popen` object. It carries the concurrency
        slot this acquired as an `_elkpy_slot` attribute; EigenstateSession
        releases it on close(), since a live session holds a core for as long
        as it is answering queries.
        """
        env = _thread_pinned_env(self.omp_threads)

        command = [str(self.elk_binary)]

        slot = _acquire_slot()
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(workdir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )
        except BaseException:
            _release_slot(slot)
            raise
        proc._elkpy_slot = slot
        return proc
