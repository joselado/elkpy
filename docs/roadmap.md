# elkpy implementation roadmap

Status: planning document, ordered by what unblocks the most future work per
unit of effort. Builds on the v0 slice (`Structure`, `Calculation.get_energy
/get_bands/get_dos`, `run_tasks` escape hatch — see `docs/design.md` §10 and
commit history) and on `docs/design.md`'s object model. "Additional
implementations" here means broader coverage of Elk's own standard
DFT-workflow capabilities (relaxation, forces, phonons, ...) — confirmed with
the user, not novel physics built on Elk's export tasks; that's noted as a
possible later direction in Tier 4 but is not the near-term goal.

## Tier 1 — Usability primitives (do these before adding more `get_*` methods)

Every item here is either a known gap that blocks real calculations today, or
a known bug. Doing these first means Tier 3's new methods don't inherit the
same gaps.

1. **Generic block passthrough.** `Calculation.__init__` only exposes `xc`,
   `spinpol`, `rgkmax`, `ngridk`, `vkloff`, `sppath` — every other Elk block
   (`maxscl`, `nempty`, `stype`/`swidth`, `epspot`/`epsengy`, `mixtype`,
   `spinorb`, `autokpt`/`radkpt`, `molecule`, `dft+u`, ...) is unreachable.
   Hand-adding each as a named kwarg defeats the point of the generic
   `InputFile` writer. Add `extra_blocks={}` (or `**kwargs`), merged in
   `_add_base_blocks` and included in `_basis_signature()` so it correctly
   participates in ground-state cache invalidation. Small (~5 lines) and
   makes the package usable for real calculations immediately — e.g.
   `spinorb` (needed for the bundled Au spin-orbit example) is otherwise
   unreachable.
2. **First-class non-convergence status.** `ensure_ground_state()` currently
   raises `RuntimeError` on non-convergence. `docs/design.md` §7 wants this
   as an inspectable result state, not exception-only — add a
   `calc.converged` property / `calc.status()` so callers building sweeps
   (Tier 5) can handle a non-converged point without a try/except around
   every call.
3. **`run_tasks` subdirectory collision.** The auto-generated subdirectory
   name is derived only from the task numbers, not `blocks` — two
   `run_tasks([120], blocks=A)` / `run_tasks([120], blocks=B)` calls silently
   overwrite each other's output. Either require an explicit `label` when
   `blocks` is given, or fold a hash of `blocks` into the default label.
4. **Species override.** Species filenames are always `{symbol}.in` resolved
   against one `sppath` (defaults to `vendor/elk/species/`). Real use needs
   per-`Structure` or per-species override (custom species files, doped/
   fractional `spzn`, ...).
5. **Per-atom magnetic fields (`bfcmt`) and moments.** `_add_base_blocks`
   hardcodes `bfcmt = (0,0,0)` for every atom. Needed for any real
   `spinpol` workflow (antiferromagnetic setups, symmetry breaking) — see
   manual §5.2.
6. **Symbolic k-path resolution for `get_bands`.** Currently only raw
   fractional vertices are accepted; `docs/design.md`'s illustrative
   `kpath="GXWLGK"` example is aspirational, not implemented. Needs
   Bravais-lattice-aware special points — an optional `spglib`/ASE-`bandpath`
   dependency, not hand-rolled. Lower priority than 1-5 (raw vertices are a
   real, if less convenient, escape valve already).

## Tier 2 — `spec` module (small, scoped)

`docs/design.md` §7 calls for one module holding all version-coupled
knowledge. What actually exists inline right now is small: `XC_CODES` (9
entries) in `calculation.py`, task numbers scattered across
`calculation.py`/`run_tasks` call sites, and output filenames hardcoded in
each `parsers/*.py`. Consolidate these into `src/elkpy/spec.py` before Tier
3 adds more of the same pattern — cheap now, more valuable the more `get_*`
methods exist.

Explicitly **not** in scope yet: auto-generating the block schema from
`docs/elk_manual.txt`. That text is pdftotext output with wrapped
descriptions and mangled math — brittle for a payoff not needed yet, since
`InputFile` doesn't validate block names anyway. Revisit only if
hand-maintaining `spec.py` across an Elk version bump turns out to be
genuinely painful.

## Tier 3 — More `get_*` methods (the "additional implementations")

In rough priority order; each entry notes what's non-trivial about it, not
just what task number it wraps.

1. **`get_forces()`** — `tforce .true.` on an existing ground-state run
   (cheap: modifies the task-0 run, not a new task), parses `FORCES.OUT`.
   Do this before relaxation; relaxation needs it internally and it's
   useful standalone.
2. **`get_relaxed()`** — tasks 2/3, `GEOMETRY_OPT.OUT`, returns a new
   `Structure` + `Calculation` per `docs/design.md` §3's "not in-place"
   policy. **Concrete trap to resolve before implementing, not after**: the
   relaxed `Structure` has different atomic positions, so the new
   `Calculation`'s `_basis_signature()` will legitimately differ from the
   old one — meaning a naive implementation throws away the just-relaxed
   density and reconverges from atomic densities. Decide explicitly: seed
   the new directory with the relaxation run's final `STATE.OUT` and record
   its signature as pre-validated (faster, correct — density from the
   relaxed geometry's last step is a good starting guess), or accept the
   reconverge and document why. This is the same class of STATE.OUT-identity
   bug fixed in the current codebase for `get_dos`/`get_bands` — name the
   policy in the code, don't leave it implicit.
3. **`get_effective_mass()`** — task 25, single k-point, cheap, no new
   structural complexity.
4. **Volumetric plot family** (density/potential/ELF, tasks 31-91) — these
   share `plot1d`/`plot2d`/`plot3d` block conventions and output format
   closely enough to warrant one general helper rather than one-off methods
   per task family.
5. **Phonons** (`get_phonon_dos()`/`get_phonon_dispersion()`, tasks 200s) —
   the most structurally different addition: phonon tasks accumulate `DYN`
   files across many partial/restarted runs per q-point rather than
   producing one shot in one invocation (`docs/design.md` §4's
   "task-specific file dependencies" warning). This does **not** fit the
   existing `_run_resumed` one-fixed-subdirectory-per-name pattern —
   budget for its own directory-management logic (per-q-point
   subdirectories, restart bookkeeping) rather than trying to force-fit it.

## Tier 4 — Possible later direction (not the current goal)

Noted for completeness, not scheduled: Python-side post-processing built on
Elk's export tasks without needing new Fortran (momentum matrix elements —
task 120 — for custom optical/response quantities; task 550 Wannier90 export
as a bridge to a tight-binding-style model, e.g. compatible with pyqula's
`Hamiltonian`; direct `STATE.OUT` reading for custom density/potential
analysis). Revisit if/when there's a specific target for it — don't build
speculative infrastructure for this ahead of a real use case.

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

1. `README.md` quickstart (none exists yet).
2. CI: run `scripts/build_elk.sh` + `pytest` on push. Add the "patch series
   still applies cleanly" check from `docs/design.md` §8 once `patches/`
   has real content (currently empty — nothing to check yet).
3. Packaging polish (dependency version pins, a changelog) once the API
   surface is less likely to change week to week.
