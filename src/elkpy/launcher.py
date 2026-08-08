"""Execution of the Elk binary.

See docs/design.md #6: this is deliberately a small, separate object so a
scheduler-backed launcher (SLURM/PBS) can be added later without reshaping
Calculation. LocalLauncher is the only implementation for now: a blocking
local subprocess call.
"""

import subprocess

from . import config


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
        import os

        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(self.omp_threads)

        command = [str(self.elk_binary)]

        log_path = workdir / log_name
        with open(log_path, "w") as log:
            result = subprocess.run(
                command, cwd=str(workdir), stdout=log, stderr=subprocess.STDOUT, env=env
            )
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

        Returns the `subprocess.Popen` object.
        """
        import os

        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(self.omp_threads)

        command = [str(self.elk_binary)]

        return subprocess.Popen(
            command,
            cwd=str(workdir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
