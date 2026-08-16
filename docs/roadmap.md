# elkpy implementation roadmap

Status: Tiers 1-3 implemented and verified against a real compiled Elk binary
(see git log; `tests/test_calculation_si.py`, `tests/test_calculation_fe.py`).
Tier 4 was never "scheduled" but has largely happened anyway, one landed
direction at a time — see the "landed for real" notes in that section, which
now cover Berry curvature, atom projection and momentum matrix elements.
Tier 5 (MPI, schedulers, sweep helpers) remains planning only. Tier 6: CI has
landed (`.github/workflows/ci.yml`, including the patch-series check §8 asks
for); the README quickstart exists; packaging polish does not. Builds on the v0 slice (`Structure`,
`Calculation.get_energy/get_bands/get_dos`, `run_tasks` escape hatch — see
`docs/design.md` §10) and on `docs/design.md`'s object model. "Additional
implementations" here means broader coverage of Elk's own standard
DFT-workflow capabilities (relaxation, forces, phonons, ...) — confirmed with
the user, not novel physics built on Elk's export tasks; that's noted as a
possible later direction in Tier 4 but is not the near-term goal.

## Tier 1 — Usability primitives — DONE

All six items implemented in `src/elkpy/calculation.py`/`structure.py`,
verified against real Elk runs (`tests/test_calculation_si.py`,
`tests/test_calculation_fe.py`).

1. **Generic block passthrough** — `Calculation(..., extra_blocks={...})`,
   merged in `_add_base_blocks` and included in `_basis_signature()`.
2. **First-class non-convergence status** — `calc.converged` property
   (`None`/`True`/`False`); `raise_on_nonconvergence=True` (default) keeps the
   old raise-immediately behavior for single-calculation use, set `False` for
   sweep-style usage that wants to check `.converged` and move on instead.
3. **`run_tasks` subdirectory collision** — default label now folds in a
   short hash of `blocks`, not just the task numbers.
4. **Species override** — `Structure(..., species_files={"Si": "custom.in"})`.
5. **Per-atom `bfcmt`** — species entries accept either a bare position or a
   `(position, bfcmt)` pair.
6. **Symbolic k-path** — `get_bands(kpath="GXWLGK")` via ASE's
   `Cell.bandpath().special_points` (optional dependency); disconnected
   segments (a `,` break in the path string) raise rather than silently
   dropping part of the path — not implemented, use `vertices=` for that case.

**Fix found during Tier 3 implementation, not anticipated here**:
`_run_resumed`'s subdirectory is now wiped clean before every run, not just
created-if-missing. Reusing a subdirectory that held partial output from an
earlier killed/crashed run caused a real, reproduced failure once phonon
DFPT (Tier 3 #5) was implemented — task 205 treats existing `DYN`/`DVS` files
as "already done" and resumed from corrupt partial state instead of erroring.
Same fix applies to every `_run_resumed` caller (bands/dos/forces/effmass/
density/phonons), not just phonons.

## Tier 2 — `spec` module — DONE

`src/elkpy/spec.py`: `XC_CODES`, `TASKS`, `OUTPUT_FILES`, each entry
cross-checked against `vendor/elk/src/` (not just the manual). The
manual-scraping generator discussed as a stretch goal is still out of scope —
not revisited, `spec.py` stayed small enough to maintain by hand through this
round of additions.

## Tier 3 — More `get_*` methods (the "additional implementations") — DONE

1. **`get_forces()`** — DONE, but not as originally planned: forces from a
   plain ground-state run go to `INFO.OUT`'s "Forces :" section
   (`src/writeforces.f90`), not a separate `FORCES.OUT` — there's no such
   file for a non-relaxation run. Implemented as a resumed run (task 1) with
   `tforce` in its own subdirectory, parsing that subdirectory's `INFO.OUT`.
2. **`get_relaxed()`** — DONE. Resolved the flagged trap by choosing the
   documented-tradeoff option: the new `Calculation` reconverges its own
   ground state from atomic densities rather than reusing the relaxation
   run's `STATE.OUT` (see the docstring in `calculation.py` for why —
   avoiding a repeat of the same stale-cache bug class). Confirmed correct on
   Si: forces ~0 at the input geometry (already near-equilibrium), relaxed
   energy differs from the original by ~6e-8 Hartree.
3. **`get_effective_mass()`** — DONE as planned (task 25, `vklem` block,
   `EFFMASS.OUT`).
4. **Volumetric plot family** — partially done: `get_density()` (task 33)
   implemented with a genuinely reusable parser
   (`parsers/volumetric.py::parse_plot3d`, since all of density/potential/ELF
   3D plots share the exact same `plot3d`-family writer,
   `src/plot3d.f90`). **Now complete**: `get_potential()` (task 43) and
   `get_elf()` (task 53) landed later, sharing a `_plot3d_lines()` helper with
   `get_density()`. Task 43 needed one shape change — it writes BOTH
   `VCL3D.OUT` and `VXC3D.OUT` from a single run, so `get_potential()` takes a
   `component=` selector to keep `get_density()`'s `(points, values)` contract.
   Verified with a check sharper than a shape assertion: since `xc="PW"` is a
   LOCAL functional, `v_xc` must be a pointwise function of `n`, and measuring
   `v_xc / [-(3n/pi)^(1/3)]` across the cell gives 1.14-1.23 — the PW
   correlation part on top of Dirac exchange — tying `RHO3D.OUT` and
   `VXC3D.OUT` together through an analytic formula.
5. **Phonons** — DONE, and simpler than expected: wrapping the DFPT method
   (task 205) rather than the classical supercell method (task 200) makes
   phonon dispersion/DOS single-shot within one `elk` invocation (confirmed
   against `examples/phonons-superconductivity/*-DFPT`), so it fits the
   existing `_run_resumed` pattern after all — the anticipated "needs its own
   directory-management logic" was not needed. What *did* need fixing was the
   stale-subdirectory bug noted under Tier 1 above, which phonons surfaced
   because task 205 (unlike every other task wrapped so far) checks for prior
   output files to decide what's already done. The classical supercell method
   remains unwrapped (multi-run coordination across displacements/machines,
   out of scope), reachable via `run_tasks()`.

   Confirmed correct on Si (`ngridq=(2,2,2)`, 2 atoms): dispersion gives 6
   branches, 3 acoustic branches at Γ within 1e-11 of zero as physically
   required; DOS runs and parses cleanly. Also confirmed expensive even at
   this minimal grid — dispersion took ~11 min, DOS ~13 min, dominated by
   DFPT's per-perturbation-per-q-point cost rather than anything elkpy
   controls. The regression tests
   (`tests/test_calculation_si_phonons.py`) are consequently gated behind
   `ELKPY_RUN_SLOW_TESTS=1`, not run by default.

## Tier 4 — Possible later direction (not the current goal)

Noted for completeness, not scheduled: Python-side post-processing built on
Elk's export tasks without needing new Fortran (momentum matrix elements —
task 120 — for custom optical/response quantities; task 550 Wannier90 export
as a bridge to a tight-binding-style model, e.g. compatible with pyqula's
`Hamiltonian`; direct `STATE.OUT` reading for custom density/potential
analysis). Revisit if/when there's a specific target for it — don't build
speculative infrastructure for this ahead of a real use case.

**One such direction landed for real**: `get_berry_curvature()` (Wilson-loop /
Fukui-Hatsugai-Suzuki Berry curvature, task 9000) — see `docs/design.md` §13 /
`docs/physics.tex` (Part II). Task 550's own overlap machinery
(`genwfsvp`/`genolpq`) turned out to be reusable, but task 550 itself wasn't
(its neighbour-shell search calls the external Wannier90 library, only a stub
in this build) — needed a small new Fortran task
(`patches/0002-berry-curvature-wilson-loop.patch`) rather than being pure
Python after all, since the two Wilson-loop directions being exact `ngridk`
mesh generators let elkpy skip the shell search entirely.

**The one this tier named explicitly has now landed too**: momentum matrix elements —
`get_momentum_matrix()` / `EigenstateSession.momentum()` (task 9002's `MOMENTUM` query,
`patches/0007-momentum-matrix-elements.patch`), see `docs/design.md` §22 /
`docs/physics.tex` Part XI. Not, in the end, via task 120's `PMAT.OUT` file export as
sketched above: task 120 writes matrix elements only for the ground-state *mesh*
$k$-points, whereas everything built on this (§22's circular dichroism, the Kubo-form
quantum geometric tensor) wants an arbitrary $k$. Upstream's `genpmatk` — the subroutine
task 120 itself calls — turned out to be directly reusable unmodified once fed a fresh
on-the-fly diagonalisation instead of file-backed mesh eigenvectors, the same substitution
the Berry-curvature path task already makes, so this is a ~40-line additive subroutine
rather than a new file. Everything above the matrix elements is pure Python
(`parsers/optical.py`).

**A second landed for real**: `get_atom_projection()` (atom-projection
operators, task 9002's `PROJECTION` query) — see `docs/design.md` §16 /
`docs/physics.tex` (Part V). Elk's own `dos`/`bandstr` tasks already compute
atom-resolved wavefunction weight internally (`gendmatk.f90`/`wfmtsv.f90`),
but only ever the diagonal-in-state entry, and only reachable through a task
that discards it into an energy-broadened DOS curve or ties it to Elk's own
band-path machinery — not an on-demand off-diagonal operator at an arbitrary
$k$-point, so a small additive Fortran extension
(`patches/0004-atom-projection.patch`) was needed, reusing `wfmtsv` itself
unchanged.

## Tier 5 — Scale & execution

1. **MPI-enabled build variant.** `LocalLauncher` now correctly refuses
   `nprocs > 1` against the serial (`mpi_stub.f90`) build. Add a second
   `build-config/make.inc.mpi`-style config (or a build-script flag) plus
   the corresponding launcher support, when parallel runs are actually
   needed.
2. **Scheduler-backed launcher** (SLURM/PBS submission, poll, collect) —
   `docs/design.md` §6's seam is already in place (`LocalLauncher` is a
   small, swappable object); implement a second launcher class when needed.
3. **Sweep helpers.** A thin Python utility for looping `Calculation`
   construction over a parameter (volume, `U`, ...) into separate
   directories, replacing Elk's native `batch` mode per `docs/design.md`
   §5. Straightforward once Tier 1 item 2 (non-convergence status) exists,
   so a sweep can skip/flag non-converged points instead of crashing
   partway through.

## Tier 6 — Housekeeping

1. `README.md` quickstart -- DONE.
2. **CI -- DONE** (`.github/workflows/ci.yml`): four staged jobs, cheapest
   first. `unit-tests` runs the whole suite with `ELKPY_ELK_BIN` pointed at a
   nonexistent path, so the binary-free tests select themselves (a `-k`
   expression drifts and under-selects: it misses the three binary-free tests
   living inside `test_calculation_soc.py`), with a minimum-test-count guard
   because pytest exits 0 if everything skips. `patch-series` is §8's check.
   Two findings shaped it:

   - **A per-patch `--dry-run` against pristine `vendor/elk/` cannot work.**
     The series is cumulative -- 0004-0009 all edit a file 0003 *creates* --
     so 8 of 9 fail that way. Patches must be applied sequentially, each
     dry-run against the partially-patched tree.
   - **`patch` exits 0 on a FUZZY apply.** A reworded comment near a hunk
     gives `fuzz 1` and success, so an exit-code check alone would pass a
     silently mis-applied series. The job greps for fuzz explicitly.

   `build-elk` runs `scripts/build_elk.sh` (measured 3m15s locally for 410
   objects, so viable on a public runner). The binary is deliberately never
   cached or shared between jobs: `build-config/make.inc` uses `-march=native`
   and CI runner fleets are heterogeneous, so a cached binary risks `SIGILL`.
   `integration` is manual/scheduled only.
3. Packaging polish (dependency version pins, a changelog) once the API
   surface is less likely to change week to week. One real bug found and
   fixed meanwhile: a non-editable `pip install .` put the package in
   site-packages, where `config.py`'s repo-root inference resolves into the
   venv and every default path becomes nonsense; it now names the cause.
