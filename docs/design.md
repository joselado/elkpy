# elkpy architecture strategy

Status: design proposal, no implementation yet. This document is the reference for
the intended shape of the package; update it as decisions change or reality
diverges from the plan.

## 1. Goal

A Python interface to Elk (vendored at `vendor/elk/`, currently 11.0.2) that makes
running DFT calculations and building additional analysis on top of Elk's output
easy, with the ergonomics of `pyqula` (the author's tight-binding package): a
central object, results reachable through `get_*` methods, real results in a
handful of lines. Where new physics genuinely needs new Fortran, that's fine —
the constraint is that such changes stay **isolated**, so a future Elk version
bump stays cheap, not that Fortran is off the table.

Elk is not a library: it is a standalone binary driven entirely by an `elk.in` text
file in the working directory (plus per-element species files), executing a
sequential `tasks` list of integer codes (task 0 = ground-state SCF, 1 = resume
from `STATE.OUT`, 2/3 = geometry optimisation, 10 = DOS, 20-23 = band structure,
100s = Fermi surface, 200s = phonons, 320/330 = TDDFT, 550 = Wannier90 export,
600s = GW, ... over 100 codes in total; see `docs/elk_manual.pdf` §5.127). It writes
uppercase `*.OUT` files into that same directory. elkpy wraps this process model.

## 2. Relationship to existing tools

- **ASE** (`ase.calculators.elk.ELK`) already wraps Elk, but only as a
  `FileIOCalculator` implementing `energy`/`forces` — i.e. it only ever runs task 0.
  It does not model tasks, chaining, or any of Elk's non-ground-state capabilities
  (bands, DOS, phonons, GW, TDDFT, BSE, Wannier export). elkpy exists to cover that
  surface.
- ASE also ships `ase.io.elk`, with an `elk.in` writer (`write_elk_in`) and parsers
  for `INFO.OUT`, `EIGVAL.OUT`, and k-points. Treat these as an **optional**
  dependency to reuse rather than reimplement — but do not make elkpy an ASE
  subclass; ASE's calculator model is too narrow for Elk's task surface. A
  `Structure.from_ase(atoms)` / `.to_ase()` conversion is worth having for
  interop, independent of whether the parsers are reused.
- elkpy is self-contained by default (matching pyqula's convention); ASE is a
  soft dependency for interop and I/O reuse, never a hard one.

## 3. Object model: a pyqula-style `Calculation`

Mirror pyqula's `Geometry` → `Hamiltonian` shape: a `Structure` (lattice vectors,
species, positions — analogous to `Geometry`) produces a `Calculation` (analogous
to `Hamiltonian`), and nearly everything hangs off `Calculation` as a `get_*`
method:

```python
from elkpy import Structure

s = Structure.from_ase(bulk("Si"))
calc = s.get_calculation(xc="PBE", ngridk=(4, 4, 4))

e = calc.get_energy()                     # task 0/1, TOTENERGY.OUT
k, bands = calc.get_bands(kpath="GXWLGK") # task 20, BAND.OUT
energies, dos = calc.get_dos(ngridk=(8, 8, 8))  # task 10, TDOS.OUT, denser mesh
calc2 = calc.get_relaxed()                # task 2/3, GEOMETRY_OPT.OUT -> new Calculation
```

- `Calculation` owns one run directory and the global `elk.in` parameters given at
  construction (`xc`, `ngridk`, `spinpol`, `rgkmax`, ...). Each `get_*` method is a
  thin delegator into a matching module (`get_bands` → `bandstructure.py`,
  `get_dos` → `dos.py`, `get_phonon_dos` → `phonons.py`, ...), same as pyqula's
  `Hamiltonian.get_bands` → `bandstructure.get_bands`. That module owns the task
  list construction, any extra `elk.in` blocks the observable needs (`plot1d` for
  bands, `plot3d` for density plots, ...), and the output-file parsing.
- **`get_*` calls block and can be expensive** — unlike pyqula, where a `get_*`
  call is milliseconds. Document this plainly; don't let the familiar naming imply
  the familiar cost. A `Calculation` should expose a way to check status
  (converged / running / not yet run) without re-triggering work.
- Each `get_*` call ensures a valid ground state exists in the calculation's
  directory before running its own task, reusing `STATE.OUT` when possible instead
  of always re-converging from atomic densities — see §4 for exactly what
  "valid"/reusable means, which is more permissive than "identical parameters."
- **Geometry optimisation returns a new object**, `get_relaxed()` → a new
  `Calculation` built from `GEOMETRY_OPT.OUT`, rather than mutating `calc` in
  place. Relaxed positions are a distinct, provenance-worthy result — the same
  copy-before-mutate spirit as pyqula's `.copy()` convention, applied to an object
  that owns a directory instead of an in-memory matrix.
- **Escape hatch for the long tail.** Elk's task surface is roughly 10x a typical
  tight-binding observable surface; don't attempt a `get_*` method for all 100+
  tasks. Give first-class `get_*` methods to the workhorse ones (energy, bands,
  dos, relaxation, phonon dispersion/DOS — roughly tasks 0/1, 2/3, 10, 20-23,
  200s), and cover everything else (GW, BSE, TDDFT, ELNES, Mössbauer, tensor
  moments, ...) through a generic method that takes raw task numbers and extra
  blocks:

  ```python
  result = calc.run_tasks([120], blocks={"kstlist": [[1, 1]]})
  ```

  Promote a task from the escape hatch to a named `get_*` method once it earns a
  module (parsing, defaults, a common enough usage pattern).

## 4. Chaining semantics and `STATE.OUT` reuse

Elk's own native chaining is a single `elk.in` with multiple tasks in its `tasks`
list (e.g. `[0, 20]`) — one invocation, one set of parameters applied to every
task in the list. A `get_*` call that needs different parameters than what the
ground state already converged under (most commonly: a denser `ngridk` for DOS or
bands than was used for SCF — a completely standard workflow, not an edge case)
requires a **second invocation** of `elk` in the same directory: a fresh `elk.in`
with `tasks: [1, <task>]`, resuming from the existing `STATE.OUT`.

Whether that resume is *valid* depends on what changed, and not everything that
changed matters equally:

- **Reused freely across `get_*` calls** — parameters that only affect how the
  converged density/potential is post-processed, not the density/potential
  itself: `ngridk`, `kstlist`, `plot1d`/`plot3d` and similar plotting blocks,
  energy windows. Task 1 is *designed* for exactly this: reconverging (or
  one-shot diagonalising) the existing density on a new sampling.
- **Invalidates the cached ground state** — parameters that change what the
  converged density/potential actually represents: `xctype`, `spinpol`, `avec`,
  `atoms`/positions/species, `rgkmax`, `lmaxapw`, `gmaxvr`, and similar
  basis/structure/functional-defining settings. A `get_*` call whose requested
  parameters differ here must trigger a fresh ground-state run (task 0), not a
  silent resume.
- `STATE.OUT` is also tied to the Elk **binary** (version, build/precision flags),
  not just `elk.in` content.

Implementation: `Calculation` keeps a manifest (e.g. `.elkpy_manifest.json` in its
directory) recording the basis/structure/functional-defining parameters and the
Elk binary identity behind the current `STATE.OUT`. Each `get_*` call diffs its
own requested parameters against that manifest: sampling-only differences reuse
`STATE.OUT` via a fresh task-1 invocation; any basis/structure/functional
difference forces a fresh task-0 run first. Silently serving a stale resume is
worse than an avoidable rerun, so an ambiguous case should re-run, not guess.

### Task-specific file dependencies

Some tasks depend on files written by a *specific* prior task, not merely "a
converged ground state" — tasks 120/130/135 need eigenvector files from a
specific prior step; phonon tasks (200s) accumulate `DYN` files across many
partial/restarted runs rather than producing one shot; tasks 2/3 write
`GEOMETRY_OPT.OUT` incrementally across optimisation steps. This is exactly why
coverage is added `get_*`-method-by-method with documented prerequisites (§3),
rather than generated generically from the task list.

## 5. Run directories and identity

- One `Calculation` = one directory. Use a **human-readable directory name**
  (user-supplied label, or derived from formula + key parameters) — DFT users
  debug by reading `elk.in` in place, and hash-named directories work against
  that. The manifest (§4) inside the directory carries the identity hash used for
  reuse/invalidation checks; the directory name itself doesn't need to encode it.
- elkpy never writes into `vendor/elk/`. Species files default to
  `vendor/elk/species/` (read-only *data*, not code — fine to depend on) via
  `sppath`, overridable per `Structure`.
- Do not replicate pyqula's "return objects and also drop conventionally-named
  `.OUT` files into cwd" duality. Elk already writes everything into the run
  directory on its own; `get_*` methods should return parsed Python objects,
  with the directory available on `calc` for anyone who wants the raw files —
  never additionally duplicate files into the caller's cwd. At the scale of a
  50-structure convergence study, cwd-dropped output files are actively hostile,
  not a convenience.
- Do not wrap Elk's built-in `batch .true.` parameter-sweep mode. An external
  Python loop constructing one `Calculation` per point composes better with the
  rest of this design and avoids `sed`-scraping `VARIABLES.OUT` the way the batch
  examples do natively.

## 6. Execution / launcher

Running `elk` is a subprocess call, potentially through MPI (`mpirun -np N`) with
OpenMP thread count as an orthogonal setting — and on HPC this normally goes
through a scheduler (SLURM/PBS), not a direct blocking subprocess call from
Python. Even if the first implementation only supports a local blocking subprocess
launch, keep a pluggable launcher seam from the start (a small interface separating
"prepare the run directory" from "execute it" from "collect results") so a
scheduler-backed launcher can be added later without reshaping the object model.
Resource spec (process/thread counts, launcher command template) should be a
per-`Calculation` setting, not a global — 50 concurrent Python-driven runs each
spawning unconfigured MPI would oversubscribe a shared machine.

## 7. Input/output layer

- **Generic `elk.in` block reader/writer**, not per-block hardcoded parsing —
  the manual's block format (name line + values, `!` comments, blank-line
  termination) is uniform enough that a general implementation stays correct as
  upstream adds new blocks, without elkpy code changes.
- **One `spec` module holding all version-coupled knowledge**: the task-code
  registry (name → integer, data not logic), block schema (name/type/default),
  and output filenames. An Elk version bump should mean editing this one place.
  `docs/elk_manual.txt` (plain-text export of the manual, already in this repo)
  has section 5 in a regular name/description/type/default table format —
  worth evaluating whether the block schema can be generated from it
  mechanically rather than transcribed by hand, to make version bumps closer to
  "regenerate" than "rewrite."
- **Prefer structured output files over scraping `INFO.OUT`.** `INFO.OUT` is a
  human-readable log whose exact text is more likely to drift across Elk
  versions. Prefer `TOTENERGY.OUT`, `EIGVAL.OUT`, `BAND.OUT`, and similar
  fixed-format numeric files. `VARIABLES.OUT` (enabled per-run via the `wrtvars`
  input flag, independent of `tasks`) appends structured records — name, array
  dimensions, a type code, then values — for many internal quantities during a
  run, and is a good parse target *when a parser genuinely needs it*. Do not
  turn `wrtvars` on unconditionally by default; the file can get large and most
  calculations don't need it.
- **Non-convergence is a first-class result state**, not an
  exception-or-silently-wrong-data coin flip. There is no separate
  `WARNING.OUT` file in Elk 11.0.2 (verified against `vendor/elk/src/`) --
  `Warning(...)` messages go to stdout (captured in the run's own log file,
  e.g. `elk.out`), not `INFO.OUT`. Convergence itself is readable from two
  literal strings `INFO.OUT` writes (`src/gndstate.f90`): "Convergence
  targets achieved" on success, "Reached self-consistent loops maximum" if
  `maxscl` was hit (see `parsers/info.py`). `get_energy()`/status should
  expose this rather than trusting output that may be from a non-converged
  run.

## 8. Build, Fortran changes, and isolation

Fortran is a legitimate implementation choice for new capability, not a last
resort — the constraint is that changes to Elk stay isolated from the vendored
tree, so upgrading `vendor/elk/` to a new upstream release stays cheap. Concretely:

- `vendor/elk/` itself always stays byte-for-byte what was downloaded from
  upstream — never edited directly, including `make.inc`.
- All building happens from a separate `build/` directory (out of the vendored
  tree, gitignored): copy `vendor/elk/` there, apply elkpy's changes, build.
  elkpy's own `make.inc` lives outside `vendor/` (e.g. `build-config/make.inc`)
  and is dropped into the `build/` copy at build time — this alone fixes the
  make.inc problem without needing it to be a "patch."
- **New Fortran capability should be additive where possible**: new modules/
  subroutines in new `.f90` files, added to the `build/` copy rather than edited
  into existing upstream files. This covers most new physics (a new task that
  computes something from existing data structures, a new post-processing
  routine).
- **Hooking a new file into Elk's existing control flow is sometimes
  unavoidable** — e.g. adding a `case` to the task dispatch in `elk.f90`, or
  registering a new input variable in `readinput.f90`. When it is, keep the edit
  to the smallest possible footprint (ideally a single added line calling out to
  the new file, clearly marked, e.g. `! elkpy: <name>`), and track every such
  edit as one hunk in a maintained patch series under `patches/`, applied to the
  `build/` copy at build time — never committed as a direct change to
  `vendor/elk/`.
- Reserve a block of task numbers and an input-block name prefix for elkpy's own
  Fortran additions (e.g. task numbers in a clearly-unused high range, block
  names with an `elkpy` or similar prefix) to minimize collision risk if a future
  upstream version happens to reuse the same numbers/names for something else.
- A CI check that the patch series still applies cleanly against the pinned
  vendored version gives early warning when a version bump breaks a patch,
  instead of a silent divergence — this is the actual cost center of this whole
  approach, and it's cheap to catch automatically.
- Prefer Elk's existing **export tasks** first when they're sufficient — task 120
  (momentum matrix elements), 130 (`⟨Ψ_{ik+q}|e^{iq·r}|Ψ_jk⟩`), 135 (plane-wave
  wavefunctions), 550 (Wannier90 export), 640 (density matrix/natural orbitals),
  and `STATE.OUT` itself cover a lot of post-processing-style new physics in pure
  Python with zero Fortran risk. Reach for a Fortran addition when the needed
  quantity genuinely isn't exposed by anything Elk already computes and writes
  out — not as a blanket rule to avoid Fortran.

## 9. Units

Elk is atomic units throughout (Bohr, Hartree; manual ch. 3). elkpy's public
values should be atomic units end-to-end, with explicit, named conversion helpers
for common needs (eV, Å) rather than silently converting inside getters — a wrapper
that quietly changes units between what it stores and what it returns is a classic
source of hard-to-catch bugs.

## 10. First implementation slice

The concrete v0 target, mirroring pyqula's "real result reachable in a handful of
lines" bar:

```python
from elkpy import Structure

s = Structure.from_ase(bulk("Si"))
calc = s.get_calculation(xc="PBE", ngridk=(4, 4, 4))
e = calc.get_energy()
k, bands = calc.get_bands(kpath="GXWLGK")
energies, dos = calc.get_dos()
```

covering: `Structure` (+ `from_ase`), the generic `elk.in` block writer, the local
subprocess launcher, `get_energy()` (tasks 0/1, `TOTENERGY.OUT`/`INFO.OUT`
parsing, convergence status), `get_bands()` (task 20, `plot1d` block construction
from a k-path string, `BAND.OUT` parsing), and `get_dos()` (task 10, `TDOS.OUT`
parsing). Everything else in this document (escape hatch, `get_relaxed()`,
phonons, the `spec` module, the build/patch mechanism, launcher pluggability) can
follow incrementally once this slice is real and tested against actual `elk`
runs.

## 11. Explicitly deferred / non-goals

- Wrapping all 100+ Elk tasks as first-class `get_*` methods.
- Wrapping Elk's built-in `batch` mode.
- Making elkpy an ASE `Calculator` subclass (ASE interop stays at the
  `Structure`/parser level, not inheritance).
- A scheduler-backed launcher (SLURM/PBS) — design the seam now (§6), implement
  later.
