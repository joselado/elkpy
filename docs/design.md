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
- Checking that the patch series still applies cleanly against the pinned
  vendored version gives early warning when a version bump breaks a patch,
  instead of a silent divergence — this is the actual cost center of this whole
  approach. It used to run in CI; with CI removed it is a manual step (apply
  `patches/*.patch` in order to a scratch copy, and grep the output for `fuzz`,
  since `patch` exits 0 on a fuzzy apply).
- Prefer Elk's existing **export tasks** first when they're sufficient — task 120
  (momentum matrix elements), 130 (`⟨Ψ_{ik+q}|e^{iq·r}|Ψ_jk⟩`), 135 (plane-wave
  wavefunctions), 550 (Wannier90 export), 640 (density matrix/natural orbitals),
  and `STATE.OUT` itself cover a lot of post-processing-style new physics in pure
  Python with zero Fortran risk. Reach for a Fortran addition when the needed
  quantity genuinely isn't exposed by anything Elk already computes and writes
  out — not as a blanket rule to avoid Fortran.

### Toolchain resolution: the same tree on a workstation and on a cluster

`build-config/make.inc` holds a workstation's answer — `gfortran`, `-lopenblas
-lfftw3 -lfftw3f`, `-march=native`. None of those three survive an HPC cluster
unchanged, so `build_elk.sh` link-tests them and repairs what does not work,
rather than hardcoding a second configuration. Aalto's Triton was the concrete
case; the three failures it exposed are generic:

1. **`-llapack` does not exist.** OpenBLAS bundles a complete LAPACK, and
   Spack-built OpenBLAS installations ship no separate reference `liblapack` at
   all. Asking for it is a hard `cannot find -llapack` even though every symbol
   Elk needs (`zheevx`, `zhegvx`, `dsyevd`, …) is present in `libopenblas.so`.
   So `-llapack` is gone from the default: it was redundant everywhere it worked
   and fatal where it did not.

2. **Nothing is on the link path until the environment modules are loaded.**
   When the default link line fails and an environment-module system is present,
   `build_elk.sh` loads `${ELKPY_MODULES:-openblas fftw}` and retries. One trap
   worth recording: `module` is a *shell function*, so `module load x | tee` runs
   it in a subshell and silently discards every environment change it makes —
   the load appears to succeed and nothing is loaded.

3. **`-march=native` on the login node is wrong for the compute nodes.** On
   Triton it resolves to `skylake-avx512` on the login node, while the default
   batch partition is Broadwell and others are Haswell or Zen3 — so the build
   succeeds and the first job dies with `SIGILL`. When environment modules are
   detected (taken as "this is a cluster"), the default becomes
   `-march=haswell -mtune=generic`: the baseline Aalto's own module tree targets,
   supported by every current partition. The performance cost is small because
   Elk's hot loops are inside BLAS, and OpenBLAS dispatches on the actual CPU at
   runtime regardless of how Elk itself was compiled.

A fourth problem is invisible at build time and appears on the first *run*:
a binary linked against module-provided libraries cannot find them in a batch
job that did not load the same modules (`libopenblas.so.0: cannot open shared
object file`). `build_elk.sh` therefore appends `-Wl,-rpath,<dir>` for every
directory in `LIBRARY_PATH` — which is exactly what module files manipulate, and
which carries more than the numerics: with a module-provided compiler it also
points at that compiler's own `libgfortran`/`libgomp`, which the system
compiler's copy need not satisfy.

Three environment variables override detection, each link-tested:

| Variable | Meaning |
| --- | --- |
| `ELKPY_F90_LIB` | Link line, verbatim. A failure is **fatal** — an explicit request is never silently replaced by a guess. |
| `ELKPY_MARCH` | `-march`/`-mtune` flags; a bare name such as `znver3` is read as `-march=znver3`. |
| `ELKPY_MODULES` | Modules to load if the default link line fails (default `openblas fftw`). |

Whatever is resolved is **appended** to the copied `build/elk/make.inc` rather
than substituted into it — `make` takes the last assignment, so that file reads
as "the defaults, then what this machine needed" and is a self-contained record
of what was actually built.

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

## 12. Per-species spin-orbit coupling scaling

The first real entry in the §8 patch series. Elk's second-variational scheme adds a
scalar-relativistic spin-orbit term to the Hamiltonian inside each muffin-tin,

$$\hat H_{\rm soc}(r) = f_{\rm soc}(r)\,\hat{\mathbf L}\cdot\boldsymbol\sigma, \qquad
f_{\rm soc}(r) = \frac{1}{(2Mc)^2}\,\frac1r\,\frac{\partial V_s}{\partial r}, \qquad
M(r) = 1 + \frac{1}{2c^2}\bigl(E - V_s(r)\bigr)\Big|_{E=0},$$

with $V_s$ the spherical part of the Kohn-Sham potential and $c$ the speed of light
(`solsc` in Elk's atomic units) — the Koelling-Harmon (1977) approximation to the
spin-orbit term of the Dirac equation, computed per-atom in `gensocfr.f90` (see the
docstring there, verified against `vendor/elk/src/gensocfr.f90`). Elk exposes a single
global multiplicative knob on top of this, `socscf` (default 1.0): the coefficient is
literally `cso = y00*socscf/(4*solsc**2)`, the same scalar for every atom in the cell.
The manual is explicit that this knob is phenomenological, not first-principles — it
exists "to enhance the effect of spin-orbit coupling in order to accurately determine
the magnetic anisotropy energy (MAE)", i.e. to compensate for whatever the
scalar-relativistic/second-variational treatment gets wrong relative to a full
four-component Dirac solve or experiment, on a per-material, fitted basis.

That compensation is not uniform across species. $f_{\rm soc}(r)$ is dominated by the
steep near-nuclear gradient of $V_s$, and the resulting SOC strength grows strongly
with atomic number (heuristically $\sim Z^4$ near the nucleus, the same scaling behind
atomic fine-structure splitting) — so a global `socscf` fitted to correct a heavy
species's MAE (e.g. a 5d transition metal) gets silently applied to every light species
in the same cell (e.g. O, N ligands) where no such correction was intended or
justified. `soc_scale={"Fe": 1.5}` generalizes `socscf` from one number to a per-species
override, so the fitted correction can be scoped to the species it was actually fitted
for.

The Fortran side changes nothing about $f_{\rm soc}(r)$ itself — `gensocfr.f90` already
loops per-atom (`do ias=1,natmtot`) to evaluate it, so
`patches/0001-per-species-soc-scale.patch` only moves the *scale* lookup inside that
existing loop: a new `socscfsp(maxspecies)` array (`modmain.f90`, sentinel `< 0` meaning
"not overridden, fall back to `socscf`") populated from a new `elkpy_socscale` input
block (`readinput.f90`), read per-species inside the loop that already computes
`cso`/`dvr` for that atom. Verified against a real compiled binary: reproduces the
global-`socscf` result exactly when a single species is present, and scales
independently per species in a two-species cell (`tests/test_calculation_soc.py`).

**How to use in code**:

```python
from elkpy import Calculation

calc = Calculation(structure, spinpol=True, spinorb=True, soc_scale={"Fe": 1.5})
e = calc.get_energy()
```

`soc_scale` requires `spinorb=True` (raises otherwise — a scale on a disabled term has
no effect), keys must be species present in `structure`, and values must be `>= 0`.
Species omitted from `soc_scale` keep Elk's default global `socscf` (1.0 unless
overridden separately via `extra_blocks`).

## 13. Berry curvature via the Wilson-loop (Fukui-Hatsugai-Suzuki) method

The second entry in the §8 patch series: `Calculation.get_berry_curvature(ist0, ist1,
directions=(1, 2))`, a new task (9000, reserved high range per §8) computing the
discretized Berry curvature of a contiguous band window via the Wilson-loop method of
Fukui, Hatsugai and Suzuki (FHS; J. Phys. Soc. Jpn. **74**, 1674 (2005),
arXiv:cond-mat/0503172) — the same lattice-gauge-theory construction used for Berry
curvature in tight-binding codes (e.g. `pyqula`'s `berry_curvature`, cross-checked for
convention per this project's development-practices policy). Full physics writeup
(Berry connection/curvature/Chern number, the FHS link-variable/plaquette-flux
construction, admissibility): `docs/physics.tex` (Part II).

For a mesh point $k_\ell$ and mesh directions $\hat\mu=1,2$, FHS build a gauge-invariant
plaquette flux entirely from wavefunction overlaps — never a bare derivative of a
numerically-arbitrary phase — via link variables $U_\mu(k_\ell) = \det M^{(\mu)}(k_\ell)
/ |\det M^{(\mu)}(k_\ell)|$, $M^{(\mu)}_{ab}(k_\ell) = \langle\psi_a(k_\ell)|
\psi_b(k_\ell+\hat\mu)\rangle$ restricted to the requested band window, and the Chern
number is the mesh sum of the resulting plaquette flux. The needed overlap matrices are
exactly what Elk's Wannier90 export (task 550, `writew90mmn.f90`) already computes via
`genwfsvp`/`genolpq` — but task 550 is unusable here without a change: which neighbour
$k+b$ to use is decided by `wannier_setup`, a call into the external Wannier90 library
that is only a stub (`w90_stub.f90`) unless Elk is linked against libwannier90, not part
of this project's build. elkpy sidesteps that dependency rather than adding it: since
the two Wilson-loop directions are exactly two of the three `ngridk` mesh generators,
the neighbour of $k_\ell$ in direction $\mu$ is simply $k_\ell+\hat\mu$ directly (a
mesh step of $1/\texttt{ngridk}(\mu)$), with no neighbour-shell search needed —
`patches/0002-berry-curvature-wilson-loop.patch` adds one new file,
`elkpy_berry.f90` (`elkpy_berrycurv`, reusing `genwfsvp`/`genolpq` exactly as
`writew90mmn.f90` does, against this directly-constructed neighbour instead), plus the
usual minimal-footprint hooks: one `elk.f90` dispatch line, one new `elkpy_berry` input
block (`readinput.f90`) for the two directions and the band window, and four new
`modmain.f90` integers. Requires `reducek=0` (set automatically by
`get_berry_curvature()`) so eigenvectors are available on the full, non-reduced
`ngridk` mesh the loop walks — with symmetry reduction on they only exist on the
irreducible wedge. All Wilson-loop arithmetic (link variables, plaquette flux, Chern
number, the admissibility diagnostic) is deliberately done in Python
(`parsers/berry.py`), not Fortran, so it's unit-testable against synthetic overlap
matrices with no Elk run at all (`tests/test_berry_gauge_invariance.py`).

Verified: gauge invariance of the flux/Chern number under a random synthetic gauge
transform of parsed overlap matrices (exact to floating-point precision, independent of
whether the overlaps are physical) — note this does *not* pin the overall sign of the
result: replacing every overlap matrix with its complex conjugate flips the sign of
every flux value but leaves gauge invariance exactly intact (the phase-cancellation
algebra is identical either way), so a conjugation-sign bug would pass this test
undetected. The sign of the Python-side arithmetic (`parsers/berry.py`) is instead
pinned directly against FHS eq. 8 with hand-constructed overlap matrices of known
target phase (`test_flux_sign_matches_fhs_eq8_*`).

**Sign convention** (§22): elkpy reports the Berry phase
$\gamma=-\mathrm{Im}\ln\prod_j\langle u_j|u_{j+1}\rangle$, i.e. minus the argument of
FHS eq. 8's link product — the King-Smith–Vanderbilt/Resta negation, which puts
curvature and Chern numbers in the standard $\mathbf A=i\langle u|\nabla_{\mathbf k}u\rangle$,
$\Omega=\nabla\times\mathbf A$ convention (Xiao, Chang & Niu, RMP 82, 1959 (2010)). It is
applied in exactly one place, `parsers.berry._berry_phase()`, which every consumer routes
through. That negation was **missing until §22's independent Kubo-form route exposed it**;
see §22 for the derivation, the three ways it was confirmed, and why every check here had
been blind to it. The Fortran overlap convention (`M(a,b)=conjg(oq(b,a))` in
`elkpy_berry.f90`), originally pinned only by derivation from `genolpq.f90`'s documented
BLAS (`zgemv`) semantics, is now **confirmed at runtime**: the same factor of $-1$ appeared
in pure Python and end-to-end through the Fortran, which localizes the error entirely in the
Python step and leaves the Fortran convention validated. Also verified against a real compiled binary:
bulk Si's 4 valence bands (a trivial insulator) give a Chern number of $\sim 10^{-19}$
(floating-point zero) on every slice of a $4\times4\times4$ mesh, with the
admissibility diagnostic comfortably inside FHS's validity regime
(`tests/test_calculation_berry.py`).

**How to use in code**:

```python
from elkpy import Structure

s = Structure.from_ase(bulk("Si"))
calc = s.get_calculation(xc="PW", ngridk=(4, 4, 4))
result = calc.get_berry_curvature(1, 4, directions=(1, 2))
result["chern_number"]  # one Chern number per slice along the free (3rd) direction
result["max_flux"]      # admissibility diagnostic; keep well under pi
```

### Path mode: Berry curvature at arbitrary k-points (task 9001)

`get_berry_curvature()` only ever evaluates on a periodic mesh covering the whole
Brillouin zone — the only way to get a genuine Chern number, but it means every
Wilson-loop corner has to be an actual mesh point (via `genwfsvp`'s file-backed
`getevecfv`/`getevecsv`, which only knows about previously-diagonalised k-points), so
querying curvature along an arbitrary band-structure-style path (e.g. Γ-K-M-K′-Γ) means
either interpolating a mesh or aligning `ngridk` to every point of interest by hand —
raised directly by a user wanting exactly that. `get_berry_curvature_path(kpoints, ist0,
ist1, directions=(1, 2), dk=0.005)` (task 9001) adds the complementary mode: the same
single-plaquette Wilson loop `pyqula`'s own `berry_curvature(h, k, dk=...)` uses — four
corners $k_0\pm\hat\mu\,\texttt{dk}$ around each requested point, independently
evaluated, no periodic mesh, no `ngridk`/`reducek` constraint at all. The trade-off is
exactly what a single small loop can't give you: no Chern number (that needs a closed
cover of the whole zone), and no automatic gap check across the mesh (no eigenvalues are
exported in this mode — see below).

Making this possible needed one genuinely new piece of machinery, not just a smaller
version of task 9000: `genwfsvp` can only expand a wavefunction from an eigenvector
*already computed and stored* for a specific k-point (`getevecfv`/`getevecsv` read from
`EVECFV.OUT`/`EVECSV.OUT`), so an arbitrary loop corner needs a **fresh
diagonalisation** from the converged potential instead. `elkpy_wfcorner`
(`elkpy_berry.f90`) does exactly that — the same on-the-fly-diagonalisation pattern
`src/bandstr.f90` already uses for a band-structure path (`readstate` → `genvsig` →
`linengy` → `genapwlofr` → `gensocfr`, then `eveqnfv`/`eveqnsv` in place of the file
read), reusing `genwfsvp`'s own on-the-fly `gengkvec`/`gensfacgp`/`match` G+k-vector
setup rather than the mesh's `ngk`/`igkig` bookkeeping arrays. Because each corner is a
one-off diagonalisation from the converged density/potential, the ground state's own
`ngridk` is irrelevant to this mode too — a Γ-only ground state is not a stopgap here,
it's sufficient, since task 9001 never touches whatever mesh the ground state happened
to converge on. `elkpy_berrycurv_path` builds the four corners per requested point
(`elkpy_berry_path` input block: two directions, `dk`, the band window, then an explicit
k-point list), reuses `genolpq` for each of the four cyclic edges exactly as task 9000
does, and writes the edge overlap matrices to `ELKPY_BERRY_PATH.OUT` —
`parsers.berry.compute_berry_curvature_path` does the arithmetic (a plain product of the
four edge link variables around the loop, algebraically identical to FHS eq. 8's
`U1(k)U2(k+1)U1(k+2)^{-1}U2(k)^{-1}` form — verified by direct derivation, not just
assumed — since each "backward" edge's link variable is exactly the complex-conjugate,
i.e. inverse, of the corresponding "forward" one), normalised by the loop's actual
Cartesian area (accounting for a non-orthogonal reciprocal lattice, not a bare
`dk*dk`-style normalisation).

`dk` is the accuracy knob here, and it has a floor unlike `pyqula`'s tight-binding
version: each corner is an independent LAPW diagonalisation, and the overlap between two
corners separated by a very small `dk` is dominated by basis-truncation noise once `dk`
gets small enough, so there's no universally-correct default — check stability by
evaluating one point of interest at a few `dk` values and looking for a plateau before
trusting a full path (`get_berry_curvature_path` doesn't do this automatically).

Verified: gauge invariance and the FHS-eq.-8 sign convention on synthetic single-loop
data, and separately that the loop's area normalisation is correct for a non-orthogonal
(hexagonal-like) reciprocal lattice, not just an orthogonal one
(`tests/test_berry_gauge_invariance.py`); against a real compiled binary, that path mode
and mesh mode agree (to ordinary numerical-noise tolerance, not bit-for-bit, since one
reads a converged mesh eigenvector and the other diagonalises fresh) when both are asked
to evaluate literally the same four k-points on bulk Si
(`tests/test_calculation_berry.py::test_path_and_mesh_conventions_agree`) — this is the
end-to-end Fortran-to-Python sign-chain cross-check the mesh-only verification above
notes was still missing, though it only catches a discrepancy *between* the two modes,
not a sign error shared by both. Also exercised on monolayer h-BN (broken sublattice
inversion symmetry, so K and K′ are physically inequivalent, unlike Si): the occupied
manifold's curvature vanishes at Γ and M and is exactly antisymmetric between K and K′
($\Omega(\mathrm K)=-8.568$, $\Omega(\mathrm K')=+8.568$ Bohr$^2$, agreeing to 0.01%; signs in the standard convention of §22 — this measurement predates that fix, which negated both) —
the sign flip time-reversal symmetry requires for a non-magnetic crystal,
$\Omega(-k)=-\Omega(k)$, and a much sharper discriminating check than Si's Chern number
(which is $0=-0$ either way). This also surfaced two real usage pitfalls worth noting
for anyone reaching for this method: (1) the occupied-band count should come from Elk's
own `EIGVAL.OUT` occupation numbers, not be assumed from a total (core + valence)
electron count — core electrons aren't among the valence bands `nstsv` indexes at all;
(2) a single band's curvature can still diverge approaching a k-point where it is
degenerate with a *neighbouring occupied* band (not just the first unoccupied one) even
when the requested window's own boundary is safely gapped — h-BN's bands 3 and 4 are
degenerate exactly at Γ, so band 4 alone diverges there while the full occupied window
(bands 1–4 together) does not, since the non-Abelian construction only needs the window
gapped from *outside* itself.

**How to use in code**:

```python
result = calc.get_berry_curvature_path(
    [(0.0, 0.0, 0.0), (1 / 3, 1 / 3, 0.0)],  # Gamma, K -- fractional coordinates
    1, 4, directions=(1, 2), dk=0.005,
)
result[0]["curvature"]  # Bohr^2, one dict per requested k-point
```

`ist0`/`ist1` must be a contiguous, 1-indexed band window (e.g. the occupied bands)
that stays gapped from the rest of the spectrum at every mesh k-point — checked
automatically from exported boundary eigenvalues, raising `ValueError` otherwise.

In place of `kpoints=`, a symbolic path can be passed as `kpath="GKMG"` (pyqula/ASE
style, resolved via `_kpath_to_points()`, the same ASE-special-points machinery
`get_bands()`/`get_phonon_dispersion()`'s `kpath=` already uses), discretized into
`npoints` points along the path; each returned point then also carries a `"distance"`
entry (cumulative Cartesian distance along the path) for plotting against the same
x-axis convention as `get_bands()`. Unlike `get_bands()`/`get_phonon_dispersion()`,
whose `kpath=` must be a single connected segment (they hand vertices to Elk's own
`plot1d` task, which interpolates one continuous line and so cannot jump), a
disconnected `,` path (e.g. `"GKM,K'G"`, to continue past M into a specific K′ zone
image rather than the nearest image `plot1d`-style interpolation would pick) is fully
supported here — task 9001 evaluates every point's Wilson loop independently via fresh
diagonalisation, so there's no interpolation across the break to get wrong.

## 14. Interactive eigenstate/overlap session

`Calculation.eigenstate_session()` (task 9002, `patches/0003-eigenstate-session.patch`,
`src/elkpy_eigenstates.f90`) gives direct access to Elk's eigenstates — second-variational
energies and eigenvectors at an arbitrary k-point, and overlaps between eigenstates —
alongside two one-off convenience wrappers, `get_eigenstates(k)` and
`get_overlap(k_a, k_b, ist0, ist1)`.

**Same-k vs. cross-k overlaps.** `evecfv`, Elk's first-variational coefficients, live in
a k-specific, non-orthogonal APW+lo basis — a raw dot product between two `evecfv`
arrays isn't a valid overlap without the basis's own overlap (S) matrix. `evecsv`, the
second-variational (spinor) coefficients, *is* built from an already-orthonormalized
first-variational basis, so `evecsv^H @ evecsv = I` for any single diagonalisation — but
that orthonormality is trivial (eigenvectors of one Hermitian matrix are automatically
orthogonal to each other) and basis-specific: the first-variational basis itself is
k-dependent (different G+k vectors at different k), so `evecsv` from two *different*
diagonalisations — whether at different k, or a second diagonalisation at the same k —
cannot be compared by a raw dot product either. `get_eigenstates()`/
`EigenstateSession.get_eigenstates()` therefore only returns one diagonalisation's own
energies and `evecsv`, documented as valid only for inspecting that one result (e.g.
degeneracies, spin character), not for computing overlaps directly. Any physically
meaningful overlap between independently-obtained eigenstates goes through
`get_overlap()`/`EigenstateSession.overlap()` instead, which reuses the same real-space
expansion (`elkpy_wfcorner`) plus overlap integral (`genolpq`) construction
`get_berry_curvature()`/`get_berry_curvature_path()` (§13) already use — the only route
that's valid regardless of whether the two k-points coincide.

**Why a persistent worker process, not an f2py in-memory bridge.** The initial ask was to
make the overlap operation fast by transferring data through memory via f2py. Two facts
ruled that out:

- The dominant per-query cost isn't process-spawn overhead — it's the
  ground-state-dependent setup (`init0`→`init1`→`readstate`→`genvsig`→`linengy`→
  `genapwlofr`→`gensocfr`) that must run once before any diagonalisation is valid. A
  mechanism that amortizes that setup across many queries captures essentially the whole
  realistic speedup, whether or not it also eliminates process-spawn cost.
- Making a persistent f2py bridge robust would mean converting every Fortran `stop`
  reachable from the new entry points (Elk's error handling is bare `stop` throughout —
  no catchable error path) into a real error return. That can't be done additively: it
  touches upstream files across most of `vendor/elk/src/`, directly conflicting with
  this project's core vendoring constraint (§8: "changes must stay isolated... prefer
  additive new files"). It would also need a new `-fPIC`/shared-library build path that
  doesn't exist today (`vendor/elk/src/Makefile` only links a single `program elk`
  executable) and raises OpenMP thread-pool reentrancy questions the project has never
  needed to reason about.

Instead, `eigenstate_session()` starts one long-lived `elk` subprocess (task 9002) that
does the setup once, prints a sentinel line (`ELKPY_SESSION_READY`), then loops reading
one query per line from standard input — `EIGENSTATES k1 k2 k3` or
`OVERLAP k1a k2a k3a k1b k2b k3b ist0 ist1` — writing each response back to standard
output, until a `QUIT` command. This captures the same "stay warm" benefit as an f2py
bridge purely via subprocess + pipes: no new build machinery, no `stop`-to-exception
conversion, and no OpenMP reentrancy concerns, since — unlike f2py calling into Fortran
repeatedly from Python's own process — this is one Fortran executable's own loop calling
OpenMP-parallel routines repeatedly, exactly the pattern its SCF loop already relies on.
`EIGENSTATES` and `OVERLAP` both reuse `elkpy_wfcorner` (§13, extended with optional
`evalsv_out`/`evecsv_out` arguments so `EIGENSTATES` queries can retrieve the
diagonalisation's own eigenvalues/eigenvectors, not just its real-space expansion) — the
same fresh, on-the-fly diagonalisation `bandstr.f90` and task 9001 use, so an
`eigenstate_session()` k-point needs no `ngridk` alignment either.

One easy-to-miss implementation detail: gfortran fully buffers standard output once it
isn't a tty — true here, since Python pipes it — so every response (including the
initial ready sentinel) ends with an explicit `flush(6)`; without it, the elkpy side
would block forever on output sitting in an unflushed buffer.

**Accepted limitation.** Malformed input or an invalid band window is new code's own to
handle gracefully (`ELKPY_SESSION_ERROR <message>`, loop continues) — but a query that
reaches a pre-existing `stop` inside reused code (e.g. a pathological/near-singular
`genolpq` call) still kills the whole session, the same way it would already kill a
one-shot task 9001 subprocess today; here it costs a warmed-up session rather than one
query. `EigenstateSession` recognizes Elk's own `Error(...)` diagnostic convention as a
signal that a `stop` is imminent and raises a `RuntimeError` carrying that diagnostic text
(rather than letting the sentinel-matching loop misparse it as numeric tokens or fall
through to a message-less end-of-file error), and separately detects the process having
exited unexpectedly on the next query. Neither attempts to make the session resilient to
this — the caller starts a new session to continue. A worker that hangs rather than exits
(e.g. a deadlock inside a BLAS call) is not covered: there is no read timeout, only
EOF/sentinel detection — out of scope for the same reason full `stop`-to-exception
conversion was rejected in the first place (§ above).

**How to use in code**:

```python
from elkpy import Structure

s = Structure.from_ase(bulk("Si"))
calc = s.get_calculation(xc="PW", ngridk=(4, 4, 4))

with calc.eigenstate_session() as session:
    state = session.get_eigenstates((0.1, 0.2, 0.05))
    state.energies  # (nstsv,) Hartree
    state.evecsv    # (nstsv, nstsv) complex, evecsv[:, i] the i-th eigenvector

    m = session.overlap((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), ist0=1, ist1=4)
    # m[a, b] = <psi_{1+a}(k_a)|psi_{1+b}(k_b)>

# one-off convenience wrappers, each opening/closing their own session --
# prefer eigenstate_session() directly when issuing more than one query
e = calc.get_eigenstates((0.0, 0.0, 0.0))
m = calc.get_overlap((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), 1, 4)
```

Verified against a real compiled binary: `evecsv^H @ evecsv = I` at an arbitrary
k-point; `overlap(k, k, ...)` is the identity matrix, to a tolerance (~1e-3) set by the
genwfsvp/genolpq real-space expansion's own inherent truncation error (angular-momentum
and interstitial G-vector cutoffs), not machine precision — observed at this order even
for a single, non-degenerate band at a generic k-point, so it's a property of the
numerical scheme, not a bug; a session issuing the same queries twice returns identical
results (the session actually stays usable across many queries, the point of this
feature); and — cross-checking two independent Fortran code paths, the same style as
§13's path/mesh agreement test — `overlap(Γ, Γ+e₁, ...)` from this task's fresh
diagonalisation matches the corresponding mesh-neighbour overlap task 9000 already
exports for the same pair, checked for band 1 specifically
(`tests/test_calculation_eigenstates.py`) — not the full occupied window, since bands
2-4 are triply degenerate at Γ (§13) and two *independent* diagonalisations are free to
pick different, equally valid bases within that degenerate subspace; comparing raw
overlap-matrix elements of a degenerate window between two independent diagonalisations
isn't meaningful (confirmed empirically: bands 1-2 agree to the same ~1e-3 floor between
the two methods, bands 3-4 do not, consistent with an arbitrary within-subspace unitary
mixing rather than a convention bug). Only gauge-invariant quantities built from a whole
degenerate window (e.g. §13's FHS flux) are safe to compare that way.

## 15. Quantum geometry: the quantum metric, alongside Berry curvature

`Calculation.get_quantum_geometry(kpoints, ist0, ist1, directions=(1, 2), dk=0.005,
kpath=None, npoints=100)` computes the full quantum geometric tensor $Q_{ab} = g_{ab} -
\tfrac{i}{2}F_{ab}$ of a band window — Berry curvature $F_{ab}$ (§13) *and* the quantum
metric $g_{ab}$ (Fubini-Study/Provost-Vallee metric) it had been missing — at an
arbitrary, explicit list of k-points, the same interface shape as
`get_berry_curvature_path()`. Full physics writeup (the quantum metric, why it's a
distinct observable from curvature, the Marzari-Vanderbilt/Resta discretization, the
Löwdin-normalization fix and its derivation): `docs/physics.tex` Part IV.

**No new Fortran task.** Every overlap this needs — including each loop corner's own
self-overlap $\langle\psi(\mathbf k)|\psi(\mathbf k)\rangle$, the one new ingredient
curvature-alone never needed — is already exposed by the task 9002 interactive session
(§14): `EigenstateSession.overlap(k_a, k_b, ist0, ist1)` with `k_a=k_b` for a
self-overlap, or `k_a != k_b` for a cross overlap. `get_quantum_geometry()` opens one
`eigenstate_session()` and, per requested point, walks a $3\times3$ grid of corners
centered at that point — $\mathbf k$, $\mathbf k\pm\mathbf v_1$, $\mathbf k\pm\mathbf v_2$,
$\mathbf k\pm\mathbf v_1\pm\mathbf v_2$ ($\mathbf v_\mu = \texttt{dk}\times
\mathbf b_{\mu}$, $\mathbf b_\mu$ a Cartesian reciprocal lattice vector computed directly
from `self.structure.avec` by `_reciprocal_vectors()` — the exact formula Elk's own
`src/reciplat.f90` uses, kept in Python rather than read back from an Elk-written file,
unlike `parsers.berry`'s `bvec`) — issuing 19 overlap queries (9 self-overlaps, 8 cross
overlaps from $\mathbf k$, plus 2 more cross overlaps for curvature's own forward
sub-loop; see below for why the metric needs the full centered grid while curvature only
needs its original forward quadrant) and handing them to
`parsers.quantum_geometry.compute_quantum_geometry()`, pure Python, unit-testable without
an Elk run (`tests/test_quantum_geometry_gauge_invariance.py`) the same way
`parsers.berry` is. This was an explicit design choice over adding a new task analogous
to 9001 (a dedicated `elkpy_quantum_geometry.f90` exporting the same matrices in one
subprocess launch instead of 19 round-trips over an already-open pipe) — see
docs/physics.tex Part IV for why the metric's dominant error source is much cheaper to
fix in Python than in Fortran, which is the actual reason no new Fortran was needed here,
not merely that it
was avoidable.

**Why the metric needs an extra step curvature never did.** `parsers.berry`'s Wilson-loop
curvature is built from $\arg(\det M)$ around a *closed* loop — Elk's `genolpq` overlap
carries a real-space truncation floor of order $10^{-3}$ (§14: `overlap(k,k,...)` is the
identity only to that tolerance), but a common-mode modulus deficiency in every link
variable cancels exactly in that closed-loop phase product, which is why §13 never needed
to worry about it. The quantum metric is built directly from $|M|^2$
(`quantum_distance()` = $J - \mathrm{Re}\,\mathrm{Tr}[MM^\dagger]$, $J$ = band-window
size) and is *not* protected the same way: an overlap deficient by a relative factor
$(1-\epsilon)$ contributes $\sim 2\epsilon J$ to `quantum_distance()`, a constant offset
independent of the true metric, that *grows* relative to the true $O(\texttt{dk}^2)$
metric signal as `dk` shrinks — confirmed on a real compiled binary (bulk Si, generic
k-point): raw (unnormalized) $g_{11}$ at `dk` = 0.05, 0.02, 0.01, 0.005, 0.002 goes
40.5, 48.2, 52.5, 62.9, 131.8 (diverging), while the fixed (Löwdin-normalized) value
goes 40.5, 47.6, 49.7, 50.5, 51.0 (converging) — the concrete, measured version of the
$1/\texttt{dk}^2$ blowup this section's fix removes
(`tests/test_calculation_quantum_geometry.py::test_normalization_prevents_truncation_divergence`).
`parsers.quantum_geometry._normalize_overlap()` applies the standard Löwdin symmetric
normalization, $M \to S_a^{-1/2} M S_b^{-1/2}$ using each corner's own self-overlap $S$,
before computing anything from it — exact by construction (it forces the normalized
self-overlap to be the identity at every corner, not just approximately), and provably
inert for curvature (§ above; $S^{-1/2}$ is Hermitian positive-definite, so
$\det(S^{-1/2})$ is real and positive and cannot shift $\arg(\det M)$), so it's applied
uniformly rather than only where it matters.

**The metric now uses a centered stencil, not a forward one — and why that fixed more
than just the convergence order.** The metric was originally built from a plain forward
difference: $g_{11} = D(\mathbf v_1)/\texttt{dk}_1^2$ and, via the polarization identity,
$g_{12} = [D(\mathbf v_1+\mathbf v_2) - D(\mathbf v_1) - D(\mathbf v_2)] /
(2\,\texttt{dk}_1\texttt{dk}_2)$, using only the forward quadrant of corners ($\mathbf k$,
$\mathbf k+\mathbf v_1$, $\mathbf k+\mathbf v_2$, $\mathbf k+\mathbf v_1+\mathbf v_2$) the
curvature loop already needed. $D(\mathbf v) = g_{ab}v^av^b + O(|\mathbf v|^3)$
(the quadratic-form expansion, `docs/physics.tex` Part IV) has a generically nonzero
cubic correction, so this forward estimate carried an $O(\texttt{dk})$ error — small for
$g_{11}/g_{22}$ in practice (h-BN's K/K′ diagonal agreed to $<1\%$ at `dk=0.01`) but large
for $g_{12}$, built from a *difference* of three comparable-magnitude
`quantum_distance()` values that amplifies it: measured on h-BN's K/K′ valleys, the K-vs-K′
gap in $g_{12}$ shrank by very close to a factor of 2 per `dk`-halving (1.83, 0.97, 0.49 at
`dk` = 0.02, 0.01, 0.005) — the textbook signature of an $O(\texttt{dk})$ error going to
zero, not of two values that already agree.

`parsers.quantum_geometry.compute_quantum_geometry()` now walks a full $3\times3$ grid of
corners centered at $\mathbf k$ (9 self-overlaps, 8 cross overlaps from $\mathbf k$, plus
curvature's own 2 forward-loop edges — 19 queries total, up from 9) and uses the standard
centered stencil for the metric — $g_{11} = [D(\mathbf v_1)+D(-\mathbf v_1)] /
(2\,\texttt{dk}_1^2)$, and $g_{12}$ from the four diagonal corners
$\mathbf k \pm\mathbf v_1\pm\mathbf v_2$ — the mixed-partial analogue of a centered
numerical derivative, which cancels the cubic correction exactly and leaves an
$O(\texttt{dk}^2)$ error instead (derivation, and the same claim confirmed on noise-free
synthetic data with the O(dk) vs O(dk^2) trend directly visible:
`docs/physics.tex` Part IV, `tests/test_quantum_geometry_gauge_invariance.py::
test_offdiagonal_metric_centered_stencil_converges_quadratically`). Curvature's own Wilson
loop is left exactly as it was — a forward (non-centered) sub-loop — since it was already
$O(\texttt{dk}^2)$ accurate and centering it would cost more corners for no gain.

That fix turned out to do more than improve the convergence *order*: for this
time-reversal-symmetric, non-spin-orbit structure, the K/K′ metric agreement is now
*exact up to floating-point roundoff* (measured relative differences
$\sim10^{-13}$–$10^{-12}$ across `dk` = 0.02, 0.01, 0.005 — not a shrinking-but-nonzero
gap, so there's no `dk`-convergence trend left to observe in real Elk output any more;
`tests/test_calculation_quantum_geometry.py::
test_hbn_metric_offdiagonal_k_kprime_symmetry_is_near_exact`). The reason: a
time-reversal-symmetric, non-spin-orbit Hamiltonian gives $\psi(-\mathbf k) =
\psi(\mathbf k)^*$, hence $D_{-\mathbf k}(\mathbf v) = D_{\mathbf k}(-\mathbf v)$
(derivation in `docs/physics.tex` Part IV), and every term of the centered stencil is manifestly
invariant under negating $\mathbf v_1,\mathbf v_2$ simultaneously — so $g_{ab}$ evaluated
at $-\mathbf k$ is *literally the same arithmetic expression* as at $\mathbf k$, not just a
closely converged one. Curvature doesn't get this same exactness: its sub-loop is anchored
at $\mathbf k$ (not centered), so the same conjugation maps it to a loop in the
diagonally-opposite quadrant rather than the identical one, leaving only the ordinary
$O(\texttt{dk}^2)$ discretization agreement it already had (K/K′ curvature matches to
$<1\%$, not machine precision). This exactness is conditional on
$\psi(-\mathbf k)=\psi(\mathbf k)^*$ — it does **not** hold under `spinorb=True`
(§17/§19's spin/orbital-locking checks), where $H(-\mathbf k)\ne H(\mathbf k)^*$ in this
simple form; only the ordinary $O(\texttt{dk}^2)$ stencil accuracy is expected there.

**How to use in code**:

```python
result = calc.get_quantum_geometry(
    [(0.0, 0.0, 0.0), (1 / 3, 1 / 3, 0.0)],  # Gamma, K -- fractional coordinates
    1, 4, directions=(1, 2), dk=0.01,
)
result[1]["g"]                # (2,2) real array [[g11,g12],[g12,g22]], Bohr^2
result[1]["berry_curvature"]  # Bohr^2, identical convention to get_berry_curvature_path
result[1]["Q"]                # (2,2) complex array, Q = g - (i/2)*berry_curvature*[[0,1],[-1,0]]
```

Same `kpoints=`/`kpath=`/`ist0`/`ist1`/`directions`/`dk`/`npoints` conventions as
`get_berry_curvature_path()` — including its `kpath=` disconnected-`,`-path support (each
point independently evaluated) and the caveat that `ist0`/`ist1` isn't gap-checked here
either (no eigenvalues exported by an overlap-only query).

Verified against a real compiled binary: bulk Si's `dk` divergence/convergence contrast
above; that curvature from `get_quantum_geometry()` matches `get_berry_curvature_path()`
(task 9001) on literally identical loop corners (anchoring this method's non-centered
loop at `get_berry_curvature_path()`'s corner 1, with double the step size, visits the
same four points in the same cyclic order — the same reasoning as §13's own path/mesh
agreement test), confirming this Python-level loop construction didn't introduce a
sign/corner-order bug independent of the already-verified FHS arithmetic it reuses; and,
on monolayer h-BN (same structure as §13's K/K′ verification), the occupied manifold's
quantum metric is positive semi-definite at Γ, K, M, K′, its diagonal AND off-diagonal
are K/K′-symmetric to $10^{-8}$ relative at `dk=0.01` (as time-reversal requires — see
above for why this is now essentially exact, not just small), curvature vanishes at the
time-reversal-invariant points Γ and M and is K/K′-antisymmetric (reproducing §13's own
finding via an entirely independent Python code path — `EigenstateSession.overlap()`
rather than task 9001's dedicated Fortran corners); and, at all four points, $\det g \ge
(F_{12}/2)^2$ — not a loose plausibility band but an exact theorem ($Q_{ab}$ is positive
semi-definite as a 2x2 Hermitian matrix in the direction indices, being a sum over band
indices of Gram-matrix-like $\langle\cdot|Q|\cdot\rangle$ terms, and $\det Q\ge0$ for a
PSD 2x2 Hermitian matrix is exactly this inequality) tying the metric's absolute scale to
the already-trusted curvature value at the same point without dividing by a curvature
that's near zero at Γ/M — holds with comfortable margin at K/K′ ($\det g \approx 2\times$
the bound) and trivially at Γ/M
(`tests/test_calculation_quantum_geometry.py::test_hbn_gkm_valley_quantum_geometry`).

## 16. Atom-projection operators

`Calculation.get_atom_projection(k, ist0, ist1)` (task 9002's `PROJECTION` query,
`patches/0004-atom-projection.patch`, `elkpy_atomproj` in `src/elkpy_eigenstates.f90`)
computes, for every atom $\alpha$ in the cell at once, the atom-projection operator
$P_\alpha$ — the muffin-tin-sphere restriction of the identity — as an
`nst`$\times$`nst` Hermitian matrix in the second-variational eigenbasis of one fresh
diagonalisation at $k$, `nst = ist1 - ist0 + 1`. Physics writeup (the operator, why it's
Hermitian PSD, the exact-partition identity $\sum_\alpha P_\alpha + P_\text{interstitial}
= \mathbb 1$, why it's not gauge-comparable across separate diagonalisations): §14's
discussion applies unchanged (same physical basis), spelled out fully in
`docs/physics.tex` Part V.

**Reused, not new, machinery.** Elk's own upstream code already computes exactly this
kind of quantity twice over — `dos.f90`'s atom/lm-resolved partial DOS (`PDOS_Sss_Aaaaa.OUT`)
and `bandstr.f90`'s task 21-24 band character (`BAND_Sss_Aaaaa.OUT`) both call
`gendmatk.f90`, which itself calls `wfmtsv.f90` to expand a batch of second-variational
states into per-atom, $(\ell m,\sigma)$-resolved muffin-tin coefficients, then radially
integrates $|\psi_{\ell m\sigma}(r)|^2$ against the quadrature weight `wr2cmt` — but only
ever the *diagonal* in state index (one state's own occupation-matrix entry), and only
ever reachable through a task that either periodic-mesh-integrates the result into an
energy-binned DOS curve or ties it to Elk's own band-path machinery, neither an on-demand
query at an arbitrary $k$-point returning the raw per-state number. `elkpy_atomproj`
duplicates `elkpy_diagonalize`'s fresh on-the-fly diagonalisation (the same small
duplication `elkpy_wfcorner` already has, kept a separate copy per this file's existing
convention rather than a shared helper with optional arguments — see
`elkpy_diagonalize`'s own docstring), calls the same `wfmtsv` for the requested atom, and
then generalises `gendmatk`'s per-state diagonal reduction to a full $i,j$ off-diagonal
one — a single `zgemm` per spin channel over the flattened muffin-tin index, weighted by
the same `wr2cmt` quadrature Elk already trusts — summed over every $(\ell,m)$ rather
than resolved by it, since only the total atomic weight is wanted here, not an
$\ell$-resolved breakdown. No new radial function, matching coefficient, or numerical
method was needed; only a different reduction of an already-computed array.

**Why all atoms in one query, not one atom per query.** `PROJECTION k1 k2 k3 ist0 ist1`
returns `natmtot` matrices from a single diagonalisation, looping `elkpy_atomproj` over
every atom internally rather than letting the caller request one atom per round-trip.
This isn't just convenience: as with `evecsv` generally (§14), two matrices from
*separate* diagonalisations — even nominally the same $k$ — are not guaranteed to share
an internal basis, so a meaningful cross-atom identity like $\sum_\alpha P_\alpha \le
\mathbb 1$ can only be checked when every $P_\alpha$ being summed came from the exact
same query. Since every atom's `wfmtsv` call reuses the same `apwalm`/`evecfv`/`evecsv`
from that one diagonalisation, looping atoms server-side costs one extra `wfmtsv` call
per atom, not another diagonalisation.

**How to use in code**:

```python
proj = calc.get_atom_projection((0.1, 0.2, 0.05), ist0=1, ist1=4)
proj.matrices.shape         # (natmtot, nst, nst) complex
proj.matrices[a][i, i].real # state (ist0+i)'s fractional weight on atom a's muffin tin
n = calc.structure.atom_index("N")  # (species, index) -> this array's atom axis
proj.matrices[n]

# kpoints=/kpath= (same convention as get_berry_curvature_path()/get_quantum_geometry(),
# §13/§15): opens ONE eigenstate_session() and reuses it across every point instead of
# re-paying the ground-state-dependent setup cost per point -- returns a list of dicts
# ({"k", "matrices", ...} plus "distance" when kpath= is used) rather than the single
# AtomProjection namedtuple k= returns.
path = calc.get_atom_projection(kpath="GXW", ist0=1, ist1=4, npoints=50)
path[0]["matrices"], path[0]["distance"]
```

Verified against a real compiled binary: every returned matrix is Hermitian and positive
semi-definite (a direct consequence of `wr2cmt` being a positive quadrature weight, not
external data to check against); summing every atom's matrix and subtracting from the
identity is still Hermitian PSD (the interstitial remainder can't be negative), with each
atom's own diagonal weight a substantial, physically reasonable fraction of the cell for
bulk Si (not near zero, not exceeding 1); diamond Si's two atoms — related by inversion
through the bond midpoint, which sends $k\to-k$ and therefore fixes Γ — have identical
weight on band 1 (non-degenerate; bands 2-4 are degenerate at Γ, the same caveat §13/§14
already document) at Γ; a diagonal entry matches an entirely independent Fortran code
path — upstream `bandstr.f90` task 21's own `BAND_Sss_Aaaaa.OUT` "total atomic character"
column, which calls the same `gendmatk`/`wfmtsv` machinery via a completely separate call
site (Elk's own path diagonalisation, not `elkpy_atomproj`'s fresh one) and reduction (an
explicit per-`l` loop, not this feature's `zgemm`) — to 5 decimal places once task 21's
default `lmaxdb`=3 is raised to match the ground state's own `lmaxo`=6, catching e.g. a
packing/stride bug the Hermitian/PSD and sum-below-identity checks alone would pass
silently (they're satisfied by any consistent normalization, even an undercounting one);
a spin-polarized run (`spinpol=True`, `nspinor`=2) is still Hermitian PSD and
sum-below-identity, exercising the `do ispn=1,nspinor` accumulation the unpolarized
fixture above never runs twice (SOC/`soc_scale` remains untested for this feature); and on
monolayer h-BN, at $K=(1/3,1/3,0)$ the occupied valence-top ($\pi$) band is N-dominated
and the unoccupied conduction-bottom ($\pi^*$) band is B-dominated — the more
electronegative N pulling the bonding state's weight toward itself, the standard
qualitative picture for h-BN's band character, and a sharp sign-of-the-effect prediction
rather than a plausibility band, the same spirit as §13's K/K′ curvature antisymmetry check;
and `kpoints=` batching is purely a plumbing change, not a numerical one -- a
multi-point call reproduces the same matrices as the equivalent sequence of one-point
calls, and `kpath=` resolves through the same `_kpath_to_points()` every other `kpath=`
consumer uses (`tests/test_calculation_atom_projection.py`).

## 17. Spin operators ($S_x$, $S_y$, $S_z$) applicable to wavefunctions

`Calculation.get_spin_operator(k, ist0, ist1)` / `EigenstateSession.spin_operator(k, ist0,
ist1)` compute the spin operators $S_x$, $S_y$, $S_z$ (units $\hbar=1$, so eigenvalues
$\pm\tfrac12$ for a pure spin state) as `nst`$\times$`nst` Hermitian matrices in the
second-variational eigenbasis of one diagonalisation at $k$ — the spin-space analogue of
§16's atom-projection operators, and, like §15's quantum metric, needing **no new Fortran
at all**: every number this consumes (`evecsv`) is already returned by the existing
`EIGENSTATES` query (task 9002, §14), so `src/elkpy/parsers/spin.py` is pure NumPy.

**Why evecsv alone is enough.** Elk's second-variational scheme builds the full spinor
Hilbert space as a *product* basis: the same `nstfv` first-variational (spin-independent,
scalar-relativistic) spatial states $\{\varphi_p\}$ span both spin channels, so
$\{|\varphi_p\rangle\otimes|{\uparrow}\rangle,\,|\varphi_p\rangle\otimes|{\downarrow}\rangle\}_{p=1}^{n_{\rm fv}}$
is an orthonormal basis of the $2n_{\rm fv}$-dimensional spinor space (`eveqnsv.f90`'s own
row layout: `evecsv` row `i = p + (ispn-1)*nstfv`, `ispn` $\in\{1,2\}$, confirmed directly
against `eveqnsv.f90` and `init1.f90`'s `nstsv=nstfv*nspinor`). A spin operator
$S_a=\mathbb 1_{\rm spatial}\otimes\tfrac12\sigma_a$ is therefore block-diagonal in $p$ in
this basis — no radial integral, muffin-tin partition, or real-space expansion is needed
at all, unlike every other elkpy quantity built from `wfmt`/`wfir` — and its matrix
elements in the second-variational eigenbasis reduce to plain inner products between
`evecsv`'s spin-up and spin-down row blocks (`parsers.spin.compute_spin_operator`; full
derivation in `docs/physics.tex` Part VI). This is why `EigenstateSession.spin_operator()`
issues no new session query: it calls `get_eigenstates(k)` (already implemented) and does
the rest in Python.

**Requires spin polarization.** `nstsv=nstfv*nspinor`, and $S_a$'s block split needs
`nspinor=2` — i.e. `Calculation(spinpol=True)` or `spinorb=True` (which forces `spinpol`
internally, per Elk's own `init0.f90`, the same forcing rule `eigenstate_session()` already
relies on when computing `nspinor` to pass to `EigenstateSession`). `spin_operator()` raises
`ValueError` immediately for a `nspinor=1` calculation rather than silently returning a
zero or meaningless operator — checked before any Fortran query, so it costs nothing beyond
opening the session.

**How to use in code**:

```python
wse2 = Structure(WSE2_AVEC, WSE2_SPECIES).get_calculation(
    "wse2", xc="PW", ngridk=(3, 3, 1), rgkmax=7.0, spinorb=True
)
wse2.get_energy()

with wse2.eigenstate_session() as session:
    ops = session.spin_operator((1 / 3, 1 / 3, 0), ist0=ist1, ist1=ist1)  # valence-band top at K
    ops.sz[0, 0].real  # <S_z> for that state, in [-1/2, 1/2]

# calc.get_spin_operator(kpath=..., ist0=, ist1=) does the above session-reuse loop for
# you (same kpoints=/kpath= convention as §16's get_atom_projection()), returning a list
# of dicts instead of the single SpinOperator namedtuple k= returns.
path = wse2.get_spin_operator(kpath="GKM", ist0=ist1, ist1=ist1, npoints=50)
path[0]["sz"], path[0]["distance"]
```

**Which row block is actually "up"?** Eqs. above assume `evecsv` row block 1 (rows
`0:nstfv`) is physically spin-up, `ispn=1` in `eveqnsv.f90`'s own convention — read directly
from the Fortran (§ above), but a *derivation from reading code* is not the same as a
runtime check, and this one has a specific failure mode that none of Hermiticity, the
$\mathfrak{su}(2)$ algebra, or K/K′ antisymmetry can catch: a global relabelling of the two
blocks (swap "up" and "down" everywhere) flips the sign of every one of those checks
*identically*, so they stay satisfied either way. Closing this needs an independent
Fortran code path that does not go through `evecsv`-block arithmetic at all — see below.

Verified against a real compiled binary: `sx`/`sy`/`sz` are Hermitian at a generic $k$-point
(immediate from `compute_spin_operator`'s Gram-matrix-like construction, checked directly
rather than assumed); `spin_operator()` raises `ValueError` for a non-spin-polarized
calculation; and, on monolayer WSe2 with spin-orbit coupling (`spinorb=True`) — broken
inversion symmetry (unlike bulk 2H stacking, a monolayer TMD has no inversion center) plus
strong SOC locks the valence-band-top spin to the valley index, so $S_z(K)=-S_z(K')$ (Xiao,
Liu, Feng, Xu \& Yao, *Coupled Spin and Valley Physics in Monolayers of MoS₂ and Other
Group-VI Dichalcogenides*, Phys. Rev. Lett. 108, 196802 (2012)) — the RELATIVE sign is that
published sign-of-the-effect prediction, the same spirit as §13's K/K′ curvature
antisymmetry and §16's N/B atom-character checks; the ABSOLUTE sign ($S_z(K)$ specifically
negative) is a regression pin, not itself a physics prediction — it depends on this
structure's own conventions (chalcogen z-ordering, lattice-vector handedness) and on the
row-block labelling discussed above. Measured: $S_z(K)\approx-0.4997$,
$S_z(K')\approx+0.4997$ — nearly maximally spin-polarized, consistent with WSe2's unusually
strong ($\gtrsim$400 meV) valence-band SOC splitting (`tests/test_calculation_spin.py`).

**Closing the row-block question**, on collinear ferromagnetic Fe (`spinpol=True`, no SOC,
no noncollinear magnetism): `eveqnsv.f90` takes its block-diagonalization branch in this
case (zeroing the off-diagonal spin blocks and diagonalizing the two remaining blocks
separately), so bands `1..nstfv` are EXACTLY pure spin-up and `nstfv+1..nstsv` EXACTLY pure
spin-down BY CONSTRUCTION — $S_z=\pm0.5$ to machine precision (no `genolpq`
real-space-truncation floor involved, since this feature never expands a real-space
wavefunction at all) — a strong structural check, but still not by itself proof of which
physical spin is which. That comes from upstream `bandstr.f90` task 23 ("spin character of
band"), an entirely separate code path (`gendmatk`/`wfmtsv`, no `evecsv`-block arithmetic)
whose own log message names its output columns "spin-up and spin-down characters" for
`ispn=1,2` — i.e. Elk's own code independently labels `ispn=1` "spin-up". Checked at the
same k-point via a 2-point `plot1d` path (same first-vertex-equals-query-k argument as
§16's task 21 cross-check): band 1 (this feature's own "spin-up" state) is the one
`bandstr.f90` also reports as spin-up-dominated, and band `nstfv+1` is spin-down-dominated
(`tests/test_calculation_spin.py`). This is the same "internally consistent but possibly
mislabelled" risk §16 already flagged for atom-projection weight (only an external
reference computing the same number a different way closes it) — here applied to a sign
rather than a magnitude.

The synthetic-data unit tests (`tests/test_parsers_spin.py`) independently pin the
arithmetic itself, ahead of and without needing a real diagonalisation: Hermiticity and the
$\mathfrak{su}(2)$ commutation relation $[S_x,S_y]=iS_z$ (cyclic) hold for *any* unitary
change of basis (a random unitary `evecsv`, not just the identity), since $S_a$ in the
eigenbasis is exactly a similarity transform $V^\dagger(\mathbb 1\otimes\tfrac12\sigma_a)V$
of the fixed physical operator; the spin-$\tfrac12$ Casimir
$S_x^2+S_y^2+S_z^2=\tfrac34\mathbb 1$ holds for the same reason; and definite-$\sigma_a$
pure states (built directly from the up/down block split, no diagonalisation involved)
reproduce the expected $\pm\tfrac12$ diagonal expectation values and vanishing cross-axis
ones.

## 18. Orbital-character (s, p, d, f) projection operators

`Calculation.get_orbital_projection(k, ist0, ist1)` (task 9002's `ORBITAL` query,
`patches/0005-orbital-projection.patch`, `elkpy_orbitalproj` in
`src/elkpy_eigenstates.f90`) computes, for every atom $\alpha$ in the cell and every
angular-momentum channel $\ell=0,1,2,3$ (s, p, d, f — `elkpy.session.ORBITAL_LABELS`) at
once, the $\ell$-resolved atom-projection operator

$$ (P_{\alpha,\ell})_{ij} = \sum_{m=-\ell}^{\ell}\sum_\sigma \int_0^{R_\alpha}
 \psi_{i,\ell m\sigma}^*(r)\,\psi_{j,\ell m\sigma}(r)\,r^2\,dr, $$

an `nst`$\times$`nst` Hermitian matrix in the second-variational eigenbasis of one fresh
diagonalisation at $k$, `nst = ist1 - ist0 + 1`. This is §16's atom-projection operator
$P_\alpha=\sum_{\ell=0}^{\text{lmaxo}}P_{\alpha,\ell}$ (summed over every $(\ell,m)$ up to
Elk's own `lmaxo`, 6 by default) resolved by $\ell$ instead — summed over $m$ and spin
only — the orbital-character analogue of atom-projection, and the same physical
construction §16 already describes (Hermitian PSD weighted Gram matrix, not
gauge-comparable across separate diagonalisations); full derivation in
`docs/physics.tex` Part VII.

**What this is not.** $P_{\alpha,\ell}$ projects onto an angular-momentum *channel*
inside atom $\alpha$'s muffin-tin sphere — every radial shell and every principal
quantum number sharing that $\ell$, not a specific atomic orbital's own radial shape.
"s/p/d/f *for each element*" in the everyday chemistry sense (a transition metal's
outermost $d$ shell, say, as opposed to its filled semicore $d$ states) would need a
free-atom reference radial function to separate those out — a different, not-yet-built
object. `get_orbital_projection()` answers "how much of this Bloch state's weight on
atom $\alpha$ has $\ell=1$ symmetry", not "how much sits in atom $\alpha$'s valence $p$
shell specifically" — the two coincide whenever a cell has no semicore states of that
$\ell$ close in energy (true for the light $sp$-bonded systems `tests/test_calculation_orbital_projection.py`
checks below), but are not the same question in general.

**Reused, not new, machinery — one level more so than §16.** `elkpy_orbitalproj` is
`elkpy_atomproj` with the $m$-sum restricted per call: one fresh diagonalisation and one
`wfmtsv` call per atom (not per $\ell$), then four separate reductions of the SAME
`wfmt` array — a masked, re-zeroed quadrature weight `wgt` (nonzero only for the lm
sub-range $\ell^2{+}1,\dots,(\ell{+}1)^2$ within each radial shell) feeding one `zgemm`
per spin channel per $\ell$, mirroring `gendmatk.f90`'s own per-$\ell$ loop (used by
upstream `dos`/`bandstr` task 21's $\ell$-resolved output) rather than its per-$(\ell,m)$
one. Two correctness details this masking makes load-bearing, both silent under
Hermiticity/PSD checks alone (a real, non-negative `wgt` is Hermitian-PSD-preserving
regardless of which lm range it is nonzero on): `wgt` must be explicitly zeroed before
the masked fill (unlike `elkpy_atomproj`, which writes every entry, so needed no such
zeroing), and the inner (muffin-tin-interior) region only contributes when
$\ell\le$`lmaxi` — `lmaxi` defaults to 1, so d and f get no inner-region contribution at
all — mirroring `gendmatk.f90`'s own `if (l <= lmaxi)` guard exactly; omitting either
would silently read stale/uninitialised memory or the wrong radial shell's coefficients
while staying perfectly Hermitian and PSD.

**Why one call returns all four $\ell$ per atom.** Unlike `PROJECTION`'s
"all-atoms-in-one-query" argument (§16 — atoms compared against each other need to share
a diagonalisation, so the query loops atoms server-side), `ORBITAL`'s reason for
batching per atom is sharper: the four $\ell$ matrices of ONE atom share not just one
diagonalisation but one `wfmtsv` call, so they are exactly mutually consistent (not
merely reproducibly so, the weaker guarantee §16 already flags for combining separate
`PROJECTION`/`ORBITAL` calls, or different atoms' matrices from one `ORBITAL` call,
across which only LAPACK determinism is relied on). This is what makes
"$\sum_{\ell=0}^3 P_{\alpha,\ell} \le P_\alpha$" (from a separate `get_atom_projection()`
call) a clean, checkable inequality rather than a coincidence of reproducibility alone.

**How to use in code**:

```python
from elkpy.session import ORBITAL_LABELS  # ("s", "p", "d", "f")

orb = calc.get_orbital_projection((0.1, 0.2, 0.05), ist0=1, ist1=4)
orb.matrices.shape                    # (natmtot, 4, nst, nst) complex
l = ORBITAL_LABELS.index("p")
orb.matrices[a, l][i, i].real         # state (ist0+i)'s p-channel weight on atom a
n = calc.structure.atom_index("N")
orb.matrices[n]                       # (4, nst, nst) -- all four l channels for N

# calc.get_orbital_projection(kpoints=..., ist0=, ist1=) reuses one session across every
# point (§16's kpoints=/kpath= convention), returning a list of dicts.
path = calc.get_orbital_projection(kpoints=[(0, 0, 0), (1 / 3, 1 / 3, 0)], ist0=1, ist1=4)
path[1]["matrices"][n]
```

Verified against a real compiled binary: every $(\alpha,\ell)$ matrix is Hermitian and
positive semi-definite; summing s+p+d+f and subtracting from `get_atom_projection()`'s
own total for the same atom is still Hermitian PSD (the g/h/i remainder can't be
negative); a diagonal entry matches an entirely independent Fortran code path —
upstream `bandstr.f90` task 21's own $\ell$-resolved `BAND_Sss_Aaaaa.OUT` columns (4
onward), calling the same `gendmatk`/`wfmtsv` machinery via a completely separate call
site and reduction — to 5 decimal places, and here with NO `lmaxdb` override needed at
all (unlike §16's atom-projection cross-check, which needed `lmaxdb=6` to match
`lmaxo`): task 21's own default, `lmaxdb=3`, is exactly s,p,d,f, the same four channels
this feature returns, so the comparison is exact at Elk's own default rather than only
after raising a cutoff; a spin-polarized run (`spinpol=True`, `nspinor=2`) still matches
that same task-21 cross-check, not merely Hermitian/PSD — closing a gap §16's own suite
leaves open, where the spin-polarized fixture checks Hermitian/PSD only, with no
external reference exercising the `do ispn=1,nspinor` accumulation; and on monolayer
h-BN, at $K=(1/3,1/3,0)$, nitrogen's occupied valence-top ($\pi$) band is
p-channel-dominated ($p\approx0.522$, s/d/f $\approx0$) while a much deeper
(bonding $\sigma$-type) valence band on the SAME atom is s-channel-dominated instead
($s\approx0.534$, p/d/f small) — the dominant channel flips between the two bands of one
atom, a sharp sign-of-the-effect prediction (not a plausibility band), the same spirit
as §16's N/B atom-character check and §13's K/K′ curvature antisymmetry
(`tests/test_calculation_orbital_projection.py`).

## 19. Atomic angular momentum operators ($L_x$, $L_y$, $L_z$) applicable to wavefunctions

`Calculation.get_angular_momentum(k, ist0, ist1)` (task 9002's `ANGMOM` query,
`patches/0006-angular-momentum.patch`, `elkpy_angmomproj` in `src/elkpy_eigenstates.f90`)
computes the (orbital) angular momentum operators $L_x,L_y,L_z$, restricted to atom
$\alpha$'s muffin-tin sphere and resolved by $\ell=0,1,2,3$ (s, p, d, f —
`elkpy.session.ORBITAL_LABELS`), for every atom $\alpha$ in the cell at once — the
vector-operator sibling of §18's $P_{\alpha,\ell}$: where $P_{\alpha,\ell}$ *projects*
onto an $\ell$ channel, $(L_a)_{\alpha,\ell}$ *applies the angular momentum operator
within it*, mixing $m$ instead of summing over it. Full derivation in
`docs/physics.tex` Part VIII.

**Reusing upstream Elk, not deriving a new operator.** The muffin-tin wavefunction
expansion `wfmtsv` returns (§16) is already in the complex spherical-harmonic basis
Elk's own APW matching step (`match.f90`, via `genylmv`) builds — the exact basis in
which the angular momentum ladder operators have their standard, simple matrix form.
Rather than deriving that matrix independently, `elkpy_angmomproj` calls upstream
`lopzflm.f90` (unmodified) — the same subroutine Elk's own on-site $\hat{\bf L}\cdot
\hat{\bf S}$ density-matrix trace, `dmatls.f90`, already uses for its own orbital-moment
output — to apply $L_x,L_y,L_z$ to each radial shell's $(\ell,m)$ coefficient vector
($L$ acts purely angularly, so each shell transforms independently), using the
identities `lopzflm`'s own docstring states:
$$ (L_x+iL_y)Y_{\ell m}=\sqrt{(\ell-m)(\ell+m+1)}\,Y_{\ell,m+1},\quad
   (L_x-iL_y)Y_{\ell m}=\sqrt{(\ell+m)(\ell-m+1)}\,Y_{\ell,m-1},\quad
   L_zY_{\ell m}=mY_{\ell m}. $$
The result is then contracted against the original (unweighted) `wfmt` as the bra,
`wr2cmt`-weighted and $\ell$-masked exactly as §18's `zgemm` reduction already is —
mirroring `dmatls.f90`'s own pattern (`lopzflm` applied to one index of a density-
matrix-like object, then a `wr2cmt`-weighted lm-sum) but generalised from a trace over
a single diagonal density matrix to the full `nst`$\times$`nst` bra-ket matrix §16/§18
already build, and from an $\ell$-summed total to an $\ell$-resolved breakdown.
Consequently no angular-momentum matrix element is derived independently in this
codebase — the only new arithmetic is the `wr2cmt`-weighted, $\ell$-masked contraction
already established by §18, applied to `lopzflm`'s output instead of to `wfmt` itself.

**Hermitian, not positive semi-definite; two identities that do not survive truncation.**
$(L_a)_{\alpha,\ell}$ is Hermitian by the same weighted-Gram-type argument as §16/§18
($L_x,L_y,L_z$ are each Hermitian on the $|\ell,m\rangle$ basis, and `wr2cmt` is real and
$m$-independent within one $\ell$ shell) — but, unlike $P_{\alpha,\ell}$, is NOT positive
semi-definite in general: an angular momentum expectation value can be negative. Two
standard identities of the *analytic*, untruncated $(2\ell{+}1)\times(2\ell{+}1)$
operators, $[L_x,L_y]=iL_z$ (and cyclic) and $L_x^2+L_y^2+L_z^2=\ell(\ell{+}1)\mathbb{1}$,
do **not** carry over to the returned `nst`$\times$`nst` matrices as matrix products:
both require a resolution of identity over every state of the full Hilbert space, not
just the requested band window (`Ψ P L_a P L_b P Ψ ≠ Ψ P L_a L_b P Ψ` when $P$, the
band-window projector, is incomplete) — the same truncation gap this project has
already flagged for $[S_x,S_y]$ if it were checked on a restricted window, made concrete
here because it *is* checked, on the untruncated analytic matrices only
(`tests/test_parsers_angular_momentum.py`, via `elkpy.parsers.angular_momentum`, a pure
-Python transcription of `lopzflm.f90`'s formula used only for this pin, never in the
production Fortran path).

**The one bug class Hermiticity cannot catch.** In the ascending-$m$ complex-harmonic
basis, $L_z$ is real diagonal, $L_x$ is real symmetric tridiagonal, and $L_y$ is
*purely imaginary* off-diagonal. Swapping which index of a bra-ket pair an operator
acts on sends a matrix to its transpose, i.e. $L_a\to L_a^{\!\top}=\overline{L_a}$ for
Hermitian $L_a$ — a no-op for the real $L_x,L_z$ but a sign flip for the purely
imaginary $L_y$ — and Hermiticity cannot see it ($\overline{L_a}$ is exactly as
Hermitian as $L_a$), the same "$\mathrm{conj}(M)$ is exactly as gauge-invariant as $M$"
blind spot §13 already documents for Berry curvature's sign. The su(2) commutator is
the discriminator (it pins $L_y$'s sign against $L_z$'s), so
`tests/test_parsers_angular_momentum.py` includes a regression pin that flips $L_y$'s
sign by hand and confirms the commutator identity — which does hold exactly on the
untruncated analytic matrices — breaks.

**How to use in code**:

```python
from elkpy.session import ORBITAL_LABELS  # ("s", "p", "d", "f")

lm = calc.get_angular_momentum((0.1, 0.2, 0.05), ist0=1, ist1=4)
lm.lz.shape                           # (natmtot, nst, nst) complex -- l=0..3 total
l = ORBITAL_LABELS.index("d")
w = calc.structure.atom_index("W")
lm.lz_orbital[w, l][0, 0].real        # <Lz> of state ist0 on W's d channel

# calc.get_angular_momentum(kpath=..., ist0=, ist1=) reuses one session across every
# point (§16's kpoints=/kpath= convention), returning a list of dicts.
path = calc.get_angular_momentum(kpath="GKM", ist0=1, ist1=4, npoints=50)
path[0]["lz_orbital"][w, l]
```

Verified against a real compiled binary: every returned matrix (per atom, per $\ell$,
and the Python-side $\ell=0..3$ total) is Hermitian; the $\ell=0$ (s) channel is
identically zero for $L_x,L_y,L_z$ (a one-dimensional, $m=0$-only space); and on
monolayer WSe$_2$ with `spinorb=True` — the same structure §17's spin-valley-locking
check uses — $\langle L_z\rangle$ restricted to W's d channel, on the valence-band-top
state, is large and of opposite sign at $K$ vs. $K'$ (Xiao, Liu, Feng, Xu & Yao, *Coupled
Spin and Valley Physics in Monolayers of MoS$_2$ and Other Group-VI Dichalcogenides*,
PRL 108, 196802 (2012) — the same paper §17 already cites for $S_z(K)=-S_z(K')$, whose
$\mathbf{k}\cdot\mathbf{p}$ model additionally identifies the valence-band Bloch state as
predominantly $d_{x^2-y^2}\mp id_{xy}=Y_2^{\mp2}$, i.e. a *pure* $m=\mp2$ state within the
d-channel at $K/K'$). Measured: $L_z^{(d)}(K)\approx-1.1380$, $L_z^{(d)}(K')\approx
+1.1380$, and W's own d-weight $P_{W,d}(K)$ (§18) is $0.56900$ — to 5 decimal places,
$|L_z^{(d)}(K)|=2\times P_{W,d}(K)$ exactly, sharper than "large and sign-flipped": since
$L_z$'s operator norm on $\ell=2$ is $2$, this equality is only possible for a pure
$m=\pm2$ eigenstate, not a mixture of $\ell=2$ $m$-values — confirming the cited
$\mathbf{k}\cdot\mathbf{p}$ model's orbital character quantitatively, and tying this
feature's absolute scale to $P_{W,d}$, itself already cross-checked against upstream
`bandstr.f90` in §18 (`tests/test_calculation_angular_momentum.py`).

## 20. The $Z_2$ topological invariant via Wannier-charge-center pumping

`Calculation.get_z2_invariant(ist0, ist1, loop_direction=1, pump_direction=2, nkx=, nt=)`
computes the $Z_2$ invariant $\nu\in\{0,1\}$ of a 2D time-reversal-invariant insulator's
occupied band window, via Wannier-charge-center (WCC) pumping: Yu, Qi, Bernevig, Fang &
Dai's non-Abelian-Berry-connection formulation of the invariant (PRB 84, 075119 (2011),
arXiv:1101.2011), combined with Soluyanov & Vanderbilt's "largest gap" crossing-counting
method for robustly extracting $\nu$ from the computed WCC trajectories (PRB 83, 235401
(2011), arXiv:1102.5600). Full derivation in `docs/physics.tex` Part IX.

**No new Fortran — reusing task 9000's already-trusted mesh export.** Unlike every
other Fortran-patch-series entry above, this feature needs no new patch at all. The
non-Abelian Wilson loop $D(k_2) = U(F_0)U(F_1)\cdots U(F_{N-1})$ (product of
SVD-unitarized nearest-neighbour overlap matrices around a closed loop in one
reciprocal-lattice direction $k_1$, at fixed pumping value $k_2$) is built from exactly
the same mesh-neighbour overlaps §13's `get_berry_curvature()` already exports via task
9000 (`elkpy_berry.f90`) and reads with `parsers.berry.parse_berry_overlaps()` — just
read here as a full multi-band matrix rather than collapsed to a single `det`-phase link
variable. Crucially, this reuses the Brillouin-zone-boundary periodic gauge closure
§13's Chern-number arithmetic already depends on for its integers to come out clean —
not a new, separately-trusted assumption. An earlier design considered driving this from
the arbitrary-k `eigenstate_session()` (task 9002, §14) instead — closing the loop by
literally querying `overlap(k_last, (1,0,0))` past the last mesh point — but this would
require independently re-establishing that Elk's arbitrary-k diagonalization reproduces
the exact periodic-gauge wavefunction at $k+G$, an assumption never previously exercised
by any existing test; task 9000's mesh export sidesteps this entirely by reusing
machinery whose periodicity handling is already empirically validated (§13's exact-zero
Chern number on trivial Si).

**Pure Python arithmetic, physically-grounded synthetic validation.** All WCC/$Z_2$
arithmetic (link unitarization, the Wilson loop product, the largest-gap reference
curve, the crossing-count parity) lives in `parsers/wilson.py`, independently
unit-tested against synthetic overlap matrices with no Elk run
(`tests/test_wilson_gauge_invariance.py`) — gauge invariance of the WCC angles under a
random per-$k$ unitary transform, an exact single-band phase pin, and, for the harder
question of whether the *crossing-count* logic is actually correct (gauge invariance
alone can't rule out a crossing-counter that's simply always wrong), a cross-check
against a completely independent, already-trusted code path: a time-reversal-symmetric
two-copy Qi-Wu-Zhang lattice model (spin-up and its complex-conjugate spin-down partner,
giving exactly opposite Chern numbers by construction), where $Z_2$ equals the
single-spin-sector Chern number mod 2 (Kane & Mele, PRL 95, 146802 (2005), whenever
$S_z$ is conserved) — checked directly against `parsers.berry`'s own plaquette-flux
Chern number (§13) computed on the identical wavefunctions, not merely hand-derived by
constructing a trajectory and guessing its expected $\nu$ (an earlier attempt at that —
hand-built smooth WCC trajectories meant to look like a textbook "partner exchange" —
gave $\nu=0$ unexpectedly for what was intended as the topological case, not because the
implementation was wrong but because an exactly mirror-symmetric ($a(s)=-b(s)$) 2-band
trajectory is a degenerate edge case where the largest-gap reference tracks the pair's
own motion rather than acting as a fixed-enough reference to register a crossing;
resolved by validating against the independently-checkable QWZ model instead of trying
to hand-craft and hand-verify a synthetic trajectory's expected answer).

**Additivity across independently-gapped band groups.** `ist0`/`ist1` is checked gapped
at every mesh point via `berry.check_gap()` (the same guard `get_berry_curvature()`
uses), and is chosen, in `tests/test_calculation_z2.py`, to span *every* occupied
valence band (the standard ab initio convention, same EIGVAL.OUT-occupation-derived
count as §16/§17's hexagonal-slab fixtures) rather than hand-isolating just the
topologically relevant low-energy complex. This is safe because $Z_2$ is additive mod 2
across independently-gapped band groups: a deep, symmetry-generic valence manifold
(e.g. graphene's $\sigma$-bonding complex) is essentially always $Z_2$-trivial on its
own, so folding it into a topologically nontrivial low-energy complex (graphene's
$\pi/\pi^*$ manifold, once gapped by spin-orbit coupling) changes $\nu$ by $0\bmod 2$ —
i.e. doesn't change it — provided the two groups stay mutually gapped everywhere on the
sampled mesh, which `check_gap()` verifies directly rather than assuming.

**Choosing `soc_scale` for a numerically resolvable gap, not just a nonzero one.** A
first attempt at `soc_scale={"C": 100.0}` (the value initially requested for this
feature) passed `check_gap()` (a real, ~15 meV gap at $K$) but gave $\nu=0$ — the
*wrong* answer, traced not to a bug in `get_z2_invariant()`/`parsers/wilson.py` but to
mesh aliasing: `check_gap()` only checks the eigenvalue gap at the mesh points actually
sampled, and says nothing about how narrow, in $k$, the Dirac-point anticrossing region
is. Diagnosed directly (not just suspected) via a cheap scan with the already-open
`eigenstate_session()` along $k_x$ through $K$ at fixed `soc_scale=100`: the occupied-
window overlap `session.overlap(K, K+dk, 1, 8)` has two singular values that stay well
below 1 (as low as ~0.71) even at `dk` an order of magnitude finer than any practical
mesh spacing (`nkx` in the thousands) — i.e. the occupied $\pi$ state's character
rotates almost completely between neighbouring mesh points at `nkx=30`'s spacing
($1/30\approx0.033$, vs.\ the anticrossing's actual width of order $10^{-3}$ in
fractional coordinates), so the Wilson loop's link unitarization at that link is
essentially an arbitrary choice — the topological signal is aliased away, not absent.
Raising `soc_scale` to 3000 (a 30x further increase — the anticrossing-width scaling
$\Delta k\sim E_{\rm gap}/(2\hbar v)$ means resolving the same relative width at a fixed
mesh spacing needs a proportionally larger gap) opens a $\sim$1.4 eV gap at $K$ whose
overlap singular values
stay close to 1 (>0.97) at a practical mesh spacing — confirmed with the same cheap
scan before committing to a full mesh run. This is a numerics-only knob, not a change of
physics: Kane & Mele's QSH prediction holds for *any* nonzero intrinsic coupling, so
`soc_scale=100`'s $\nu=1$ is exactly as real as `soc_scale=3000`'s — it is simply below
what a practically-sized `nkx` can resolve, the same way an under-sampled Chern-number
mesh (§13's `max_flux` diagnostic) can silently miss a real Berry-curvature feature
without any single point's own check failing.

Verified against a real compiled binary: on monolayer graphene (`spinorb=True`,
`soc_scale={"C": 3000.0}`, chosen via the resolvability diagnostic above rather than the
original 100x) the occupied $\pi$ band stays gapped from $\pi^*$ by $\sim$1.4 eV at
$K$ — the Brillouin-zone minimum, well above `check_gap()`'s default tolerance and
still well below graphene's own $\sigma$-$\pi$ separation (occupied bandwidth
$\sim$19 eV), confirming the scaled SOC term (§12) is acting on the right states without
reorganizing the $\sigma$ manifold — and `get_z2_invariant()` on the full occupied
valence manifold gives $\nu=1$, Kane & Mele's own prediction for graphene with enhanced
intrinsic spin-orbit coupling (`tests/test_calculation_z2.py`).

**A second, independent physical test with no artificial SOC scaling at all.**
Freestanding monolayer bismuth ("bismuthene": a buckled honeycomb lattice, 2 atoms per
cell vertically offset — the same structural motif as buckled silicene/germanene, space
group P-3m1) was predicted a QSH insulator by Murakami (PRL 97, 236805 (2006),
arXiv:cond-mat/0607001), driven by bismuth's own large *atomic* spin-orbit coupling —
unlike graphene, no `soc_scale` enhancement is needed to see a numerically convenient
gap. Structure (`a`=4.34 Å, buckling=1.73 Å, gap 0.555 eV without SOC / 0.500 eV with
SOC) from Cheng, Liu, Tan, Zhang, Wei, Lv, Shi & Tang, "Thermoelectric Properties of a
Monolayer Bismuth", *J. Phys. Chem. C* 118, 904 (2014), confirmed independently in
Freitas, Rivelino, de Brito Mota, de Castilho, Kakanakova-Georgieva & Gueorguiev,
"Topological Insulating Phases in Two-Dimensional Bismuth-Containing Single Layers
Preserved by Hydrogenation", *J. Phys. Chem. C* 119, 23599 (2015), Table 1 ("in good
agreement with the work of Cheng et al."). Scanning `get_eigenstates()` across
Γ-K-M with the real compiled binary confirms a genuinely different band-inversion
mechanism from graphene's Dirac-point-at-K picture: the gap minimum (~0.6 eV) sits at
Γ (an s-p-orbital inversion, HgTe/CdTe-style), monotonically increasing to >2 eV at K
and M — checked directly rather than assumed, since `get_z2_invariant()`'s default
mesh (`loop_direction`/`pump_direction`=(1,2)) always includes Γ at mesh index (0,0)
regardless of `nkx`, unlike K (which needed `nkx`/`nky_full` multiples of 3 for
graphene's own test to land on it exactly). `get_z2_invariant()` on the full occupied
valence manifold (30 bands) gives $\nu=1$ — Murakami's own prediction — confirming the
method on a second, structurally and mechanistically distinct QSH system
(`tests/test_calculation_z2.py`).

## 21. The 3D strong/weak $Z_2$ classification via the six time-reversal-invariant planes

`Calculation.get_z2_invariant_3d(ist0, ist1, nkx=, nt=)` computes the full 3D
classification $(\nu_0;\nu_1\nu_2\nu_3)$ of a 3D time-reversal-invariant insulator's
occupied band window — Fu, Kane & Mele, PRL 98, 106803 (2007), arXiv:cond-mat/0607699.
$\nu_0=1$ (a strong topological insulator) is the physically robust classification: an
odd number of protected Dirac surface states on any termination. Full derivation in
`docs/physics.tex` Part X.

**Six 2D problems, no new machinery beyond §20 itself.** The 3D Brillouin zone's 8 TRIM
split into 6 time-reversal-invariant (TRI) *planes* ($k_i=0$ or $k_i=\pi$ for each
reciprocal direction $i=1,2,3$); each plane, with its other two directions free, is
itself a genuine 2D time-reversal-invariant system (the fixed component satisfies
$-k_i\equiv k_i$), so §20's WCC-pumping method applies inside it unmodified. FKM's
parity-product definition (their eqs. 2-3) factors exactly into a statement about these
six per-plane 2D invariants $z(k_i{=}0,\pi)$: $\nu_0 = z(k_i{=}0)\oplus z(k_i{=}\pi)$ for
*any* axis $i$ (an algebraic identity — the three axis choices must agree, since they
are three different ways of splitting the same 8-number product into two groups of 4),
and $\nu_i = z(k_i{=}\pi)$ (the $\pi$-plane specifically, not the $0$-plane — FKM's own
convention). $(\nu_1,\nu_2,\nu_3)$ are basis-dependent (FKM: they combine into a
reciprocal-lattice vector $\mathbf G_\nu=\sum_i\nu_i\mathbf b_i$); $\nu_0$ is not.

The only new plumbing needed: `get_z2_invariant()` gained a `plane_offset` parameter
(default 0.0, so existing 2D callers are unaffected) selecting the fractional coordinate
of the one direction that's neither `loop_direction` nor `pump_direction`, via a
one-point k-mesh offset (`vkloff`) in that direction only — threaded through
`_run_resumed`/`_add_base_blocks`, which gained their own optional `vkloff` override for
this (same sampling-only-parameter reasoning as `ngridk`, §4). `loop_direction`/
`pump_direction`'s own `self.vkloff` components must be exactly 0 (checked, raising
`ValueError` otherwise) — a nonzero offset there would silently shift the pumping
direction's two sampled endpoints off the true TRI momenta, giving a wrong 2D invariant
with no error, the same class of silent-wrong-answer already hit once for mesh aliasing
(§20). `get_z2_invariant_3d()` calls `get_z2_invariant()` six times (axis $i$ fixed at
0 and 0.5, cyclically using the other two directions as that call's loop/pump pair) and
combines the six 0/1 results via a new pure-Python function,
`parsers.wilson.combine_3d_invariants()` — unit-tested on synthetic data
(`tests/test_wilson_gauge_invariance.py`), including that a $\nu_0$-axis disagreement
raises rather than silently picking an answer.

**The test example: cesium on a dimerized diamond lattice — Fu & Kane's own toy
model, not a proxy for it.** Rather than picking an arbitrary real material and hoping
its published Z2 classification carries over, this feature was exercised against
the minimal lattice model Fu & Kane use to *introduce* $(\nu_0;\nu_1\nu_2\nu_3)$ in the
first place (PRB 76, 045302 (2007), arXiv:cond-mat/0611341, their eq. 4 and §IV.3,
confirmed directly against the arXiv HTML source, not just summarized): the diamond
structure (space group $Fd\bar3m$ — the same structure as this project's own Si tests,
§1), with the second basis atom displaced along the cubic body diagonal [111] by a
small $\delta$ — from the ideal $(0.25,0.25,0.25)$ to
$(0.25{-}\delta,0.25{-}\delta,0.25{-}\delta)$ — shortening exactly one of the four
tetrahedral bonds per atom while lengthening the other three. Unlike a [001] tetragonal
strain of the same lattice (see the corrected §α-Sn discussion below), this [111]
distortion reduces $Fd\bar3m$ to $R\bar3m$ (#166) — *symmorphic* (only a 3-fold rotation
about [111] and inversion survive), so none of the nonsymmorphic zone-boundary sticking
below applies. FKM's own stated sign convention, quoted directly: "When the 111
distorted bond is stronger than the other three bonds, so that the system is dimerized,
the system is a strong topological insulator ... When the 111 bond is weaker than the
other three, so that the system is layered, it is a weak topological insulator." A
shorter bond is $\delta>0$ in this parametrization (atom 2 moved toward atom 1) — the
sign used here.

Cesium (a single 6s valence electron) was chosen — per this project's standing rule to
ask Fable about material/structure choices (CLAUDE.md's "Development practices"
section) — because its low-energy physics sits close to the single-s-orbital-per-site
picture FKM's tight-binding Hamiltonian assumes; this is not a real crystal phase of
cesium (the $a=16$ Bohr lattice constant used is chosen only to keep muffin-tin spheres
non-overlapping). Real single-band SOC is expected to be extremely weak (no orbital
angular momentum on a pure s state), so `soc_scale={"Cs": 3000.0}` enhances it to a
numerically convenient scale — the same reasoning as §20's graphene test, not a claim
about real cesium. Verified against a real compiled binary: a comfortable, generically
large gap everywhere sampled on a $\Gamma$-L-X-$\Gamma$ path (minimum 0.136 eV, at L —
nowhere near §20's graphene-aliasing danger zone), and
`get_z2_invariant_3d(1, ist1, nkx=12, nt=7)` gave $\nu_0=1$ — a strong topological
insulator, apparently matching FKM's $\delta t_1>0$ prediction. **That result has since
been retracted; see §23.** The Fu-Kane parity indicator (exact, no mesh) gives
$(0;000)$ for this structure, robustly across six independently-gapped band windows, and
refining the WCC mesh on a disputed plane gives $z=1,0,1,0$ for
$(n_{kx},n_t)=(12,7),(18,9),(24,13),(32,17)$ — an oscillation, not a converged value, so
the original $n_{kx}=12$ sample carried no information. The structure is topologically
trivial, and the apparent agreement with FKM was coincidental: a hypothetical Cs diamond
lattice with SOC scaled 3000$\times$ does not realize the phase of FKM's single-orbital
tight-binding model. What survives unchanged is the *implementation* check —
`combine_3d_invariants()`'s algebraic consistency across the three axis splits held, and
the WCC method still agrees with the parity route in 2D (graphene, §23) — so what was
under-converged is the six-plane 3D sweep at a practical mesh, not the 2D machinery
(`tests/test_calculation_z2_3d.py`, `tests/test_calculation_parity.py`).
$(\nu_1,\nu_2,\nu_3)=(0,0,0)$ here, not FKM's own $(1,1,1)$ — expected, not a
discrepancy, since $(\nu_1,\nu_2,\nu_3)$ are basis-dependent (§21's own combination
formula) and FKM's Hamiltonian is written in a different primitive-lattice-vector
convention than this project's `Structure`.

**A corrected dead end: strained $\alpha$-Sn's pinned "gap" was inconclusive, not proof
of gaplessness.** Freestanding gray tin (diamond structure, same space group as Si) is a
zero-gap semimetal with inverted band ordering; a first attempt applied a *uniaxial
[001] tetragonal* lattice strain (matching a specific epitaxial-growth geometry from
Huang & Liu, PRB 95, 201101(R) (2017), not FKM's own [111] model above) and found the
gap pinning to $\sim10^{-6}$ eV at the strained zone boundary, identically, across four
strains tried (`c/a` = 0.95, 0.99, 1.01, 1.05, both signs) — a value that doesn't move
with strain looks like an exact symmetry constraint, not a numerically-small-but-real
gap. This project's history initially over-concluded from that: [001] strain reduces
$Fd\bar3m$ to $I4_1/amd$ (#141), a *nonsymmorphic* space group, and nonsymmorphic space
groups are indeed known to enforce band sticking along zone-boundary lines at any
strain magnitude of the same symmetry-preserving type — but, per Fable (consulted
directly, per this project's standing rule above, rather than left as a plausible-looking
guess): that sticking groups bands into stuck quartets, it does not by itself forbid a
gap at the material's actual filling. Watanabe, Po, Zaletel & Vishwanath (PRL 117,
096404 (2016), arXiv:1603.05646) show $I4_1/amd$ still admits genuine band insulators at
fillings that are multiples of 4 — and published DFT (Huang & Liu 2017 above) reports a
genuinely gapped 3D TI for compressive [001]-strained $\alpha$-Sn. The measured
$\sim10^{-6}$ eV almost certainly reflects a splitting measured *inside* one such
symmetry-stuck quartet (or the nearly-flat semicore manifold), not the true
valence-conduction gap — the original band window was derived the same
occupation-counting way used throughout this project, but which specific bands that
lands on, relative to where the stuck quartets sit, was never checked. The correct
record is: **this probe was inconclusive, not a disproof** — [111] distortion (the
verified cesium model above) sidesteps the question entirely by staying in the
symmorphic $R\bar3m$, where no such sticking exists to complicate the diagnosis. See
Hirschmann, Leonhardt, Kilic, Fabini & Schnyder, Phys. Rev. Materials 5, 054202 (2021),
arXiv:2102.04134, Table II, for the specific $I4_1/amd$ zone-boundary sticking
classification.

**An honestly-recorded inconclusive attempt: bulk Bi$_2$Se$_3$.** Rhombohedral,
$R\bar3m$ (#166), the material that made 3D topological insulators a major experimental
subfield (Zhang, Liu, Qi, Dai, Fang & Zhang, Nature Physics 5, 438 (2009),
arXiv:0812.1622). Structure sourced from a real deposited crystal structure
(Crystallography Open Database entry 9011965, digitizing Nakajima's original
diffraction refinement, J. Phys. Chem. Solids 24, 479 (1963)) rather than hand-converted
from reported hexagonal Wyckoff $z$-parameters (an earlier hand-conversion attempt, using
a wrong transformation matrix, gave a self-contradictory $\sim11$ Å "bond" where
$\sim3$ Å was expected, despite starting from correct literature $z$-parameters — the
transformation was the bug, not the numbers; see CLAUDE.md's standing rule on this) —
independently bond-length-verified before any DFT (3.075 Å / 2.851 Å Bi-Se distances,
7.365 Å quintuple-layer span, 2.579 Å van-der-Waals gap, all matching the known physical
picture). Ground state converged with a robust, correctly-sized gap (0.258 eV at
$\Gamma$, the Brillouin-zone minimum sampled along a $\Gamma$-Z-F-L-$\Gamma$ path,
comfortably above §20's graphene-aliasing danger zone). But
`get_z2_invariant_3d(1, 78, nkx=8, nt=5)` gave $\nu_0=0$ on all six planes — the *wrong*
answer relative to the well-established literature result $(1;000)$. Diagnosed as far as
a single re-run reasonably allows: the saved mesh data shows $\sim3600$
near-machine-precision internal degeneracies deep in the semicore manifold (bands
1-50ish, likely flat Bi 5d/Se-derived states) — but re-running one plane with a narrower,
cleanly-isolated window (bands 61-78, gapped by 4.9 eV below and 0.28 eV above, no
internal degeneracies below meV scale) gave the *same* $\nu_0=0$ on that plane, ruling
out semicore-window contamination as the cause. Whether this is a genuine
mesh-convergence problem (`nkx=8`/`nt=5` too coarse for Bi$_2$Se$_3$'s more intricate,
all-electron band structure — a materially denser mesh would cost several times the
$\sim45$ minutes the six-plane sweep already took) or something else was not resolved,
and is left here as an open, explicitly documented question rather than silently
dropped or chased further without the compute budget to do so properly. No test asserts
a result for this structure.

**Resolved (§23): Bi$_2$Se$_3$ is $(1;000)$ after all, and the $\nu_0=0$ above was an
under-converged crossing count.** The parity indicator settles it exactly — 8
diagonalisations, no mesh — and gives $\delta(\Gamma)=-1$ with $+1$ at all seven other
TRIM, i.e. $\nu_0=1$, $\nu=(0,0,0)$, the accepted literature answer
(`tests/test_calculation_bi2se3_parity.py`). Four things make this decisive rather than
merely a second opinion:

- **It is the same ground state.** Direct gap at $\Gamma$ of 0.2575 eV against the 0.258 eV
  recorded above, so the difference is purely the method, not the structure, the SCF or the
  setup.
- **The delta pattern reproduces the known mechanism, not just the known parity.** Exactly
  one odd TRIM, and it is the zone centre — the single $\Gamma$ band inversion of Zhang,
  Liu, Qi, Dai, Fang & Zhang, Nature Physics 5, 438 (2009). An odd $\delta$ at a
  $k_i=\pi$ TRIM would have given nonzero weak indices and pointed at cell orientation
  instead.
- **It rules out the band window**, which was this section's own leading suspicion. The
  narrow bands-61-78 window tried above — the one that still gave the wrong WCC answer — is
  among seven windows tested, all of which give $(1;000)$. Three of the seven return all
  eight $\delta_i$ negated with $\nu_0$ unchanged, §23's even-TRIM sign immunity showing up
  on a real material.
- **It needs no `soc_scale`.** Unlike the cesium structure, this is a real material with its
  own atomic spin-orbit coupling and a robust 0.26 eV gap.

Cost: 1-3 minutes per window, against the ~45 minutes the WCC sweep took to return the
wrong integer. The structure is now a checked-in fixture (COD 9011965 via
`spglib.standardize_cell(to_primitive=True)` — a library transformation, not a hand
derivation, per this project's standing rule), gated behind `ELKPY_RUN_SLOW_TESTS=1`.

## 22. Momentum/velocity matrix elements, optical selection rules, and Kubo-form quantum geometry

*Physics writeup: `docs/physics.tex` Part XI.*

`Calculation.get_momentum_matrix(k)` / `EigenstateSession.momentum(k)` (task 9002's
`MOMENTUM` query, `patches/0007-momentum-matrix-elements.patch`, `elkpy_momentum` in
`src/elkpy_eigenstates.f90`) return the momentum matrix elements

$$ p^a_{nm}(\mathbf k)=\langle\psi_{n\mathbf k}|\Big(-i\nabla+\tfrac{1}{4c^2}\big[\vec\sigma\times\nabla V_s(\mathbf r)\big]\Big)_a|\psi_{m\mathbf k}\rangle,\qquad a=x,y,z, $$

for every pair of second-variational states at an arbitrary $k$-point, together with the
eigenvalues of that same diagonalisation. In Hartree atomic units ($\hbar=m_e=1$) and for
Elk's **local** Kohn-Sham potential, $\hat{\mathbf v}=\hat{\mathbf p}$ numerically, so this
is equally the velocity operator; the $(1/4c^2)[\vec\sigma\times\nabla V_s]$ term is the
spin-orbit correction that keeps that identity true under `spinorb=True` (Rathgen &
Katsnelson, *Physica Scripta* **T109**, 170 (2004), cited in `genpmatk.f90`'s own header).

This is the missing *primitive* rather than a single new observable. §16–§19 built a family
of operators at arbitrary $k$ (atom, $\ell$-channel, spin, orbital angular momentum), and
§13/§15 built a family of *geometric* quantities from finite-difference wavefunction
overlaps. The velocity operator is what connects them: it turns the geometric quantities
into sum-over-states expressions that need no $k$-derivative at all, and it is the matrix
element every optical response is built from.

**Fortran side: almost nothing new.** `genpmatk.f90` is used unmodified — the same
subroutine Elk's own task-120 `PMAT.OUT` export (`putpmat.f90`) calls. `putpmat` reads
previously-diagonalised *mesh* eigenvectors via `getevecfv`/`getevecsv`, so it only ever
knows mesh points; `elkpy_momentum` substitutes a fresh on-the-fly diagonalisation plus a
`genwfsv` expansion, the same substitution patch 0002 already makes for the arbitrary-$k$
Berry-curvature path task (§13). Two `genwfsv` flags differ from `elkpy_wfcorner`'s and are
load-bearing, not stylistic:

- `tsh=.true.` — muffin-tin part in spherical **harmonics**, not spherical coordinates,
  because `genpmatk` applies `gradzfmt` to it, which expects $(\ell,m)$ coefficients.
- `tgp=.true.` — interstitial part as $G+p$ coefficients, because `genpmatk` takes the
  gradient in reciprocal space itself. With `tgp=.true.` the `ngridg_`/`igfft_` arguments
  are unused inside `wfirsv.f90`, so this file's coarse-grid `ngdgc`/`igfc` is equivalent to
  `putpmat`'s fine-grid `ngridg`/`igfft`; `genpmatk`'s own interstitial FFTs are on the
  coarse grid either way.

`genpmatk`'s `pmat` argument is hard-dimensioned `nstsv`, so there is **no band-window
variant to expose** — all states are always computed and any windowing happens in Python.
That suits the intended use: the Kubo sums below run over states *outside* the window of
interest, so a windowed export would be the wrong object. The eigenvalues travel with the
matrix elements for the same reason §16's `PROJECTION` returns every atom at once: those
sums' energy denominators must come from the same diagonalisation as the matrix elements,
and pairing this response with a separate `EIGENSTATES` query would reintroduce exactly the
degenerate-subspace basis ambiguity §14 warns about.

**Index convention, and why it is not a guess.** `pmat[a][n, m]` $=\langle\psi_n|p_a|\psi_m\rangle$
— the conjugated (bra) state on the row. `genpmatk.f90`'s header defines
$P_{ij}=\int\Psi_i^*(-i\nabla+\ldots)\Psi_j$, and its accumulation is `zgemv('C', ...)` into
`pmat(1,jst,i)`, placing the conjugated factor on the first index. This matters because a
transposed `pmat` would flip *both* the Kubo Berry curvature and the circular polarization
below, so no consistency check between the two could catch it.

**Hermiticity is not evidence here.** Unlike §16/§18/§19's reductions, `genpmatk` enforces
Hermiticity by construction (upper triangle computed, lower set by conjugation, diagonal
forced real), so asserting it would say nothing about this export path. The checks with
teeth are the two below.

### Optical selection rules: valley-selective circular dichroism

With $a,b$ the two in-plane Cartesian axes, the circular-basis interband matrix elements
$P_\pm=p^a_{cv}\pm i\,p^b_{cv}$ give the degree of circular polarization

$$ \eta(\mathbf k)=\frac{|P_+|^2-|P_-|^2}{|P_+|^2+|P_-|^2}, $$

the normalized difference in absorption strength between left- and right-circularly
polarized light ($\eta=+1$: couples only to $\sigma^+$; $\eta=-1$: only to $\sigma^-$). In a
gapped honeycomb lattice the three-fold rotation symmetry at the zone corner forces
$|\eta|=1$ there, with opposite sign at the two inequivalent valleys — valley-selective
circular dichroism (Yao, Xiao & Niu, PRB **77**, 235406 (2008); Xiao, Liu, Feng, Xu & Yao,
PRL **108**, 196802 (2012) — the same paper §17/§19 already cite for $S_z(K)=-S_z(K')$ and
$L_z(K)=-L_z(K')$; Cao *et al.*, Nat. Commun. **3**, 887 (2012) for the measurement).

`parsers.optical.circular_polarization` accepts a single 1-based band index, an inclusive
`(lo, hi)` range, or an iterable, and **sums $|P_\pm|^2$ over all $(c,v)$ pairs** when given
a group — the correct handling for a degenerate manifold, where no individual pair's matrix
element is separately meaningful. The converse is a real trap: lumping two *non-degenerate*
bands together (e.g. both partners of a spin-orbit-split valence-band top) averages their
separate selectivities and dilutes $\eta$, which is a physically different question from the
band-edge transition's own selectivity.

### Kubo-form quantum geometry, and an independent check on §13/§15

Second-order perturbation theory in $\mathbf k$ replaces the derivative
$|\partial_a u_n\rangle$ by a sum over the other states,
$\langle u_m|\partial_a u_n\rangle=\langle m|v_a|n\rangle/(\varepsilon_n-\varepsilon_m)$ for
$m\neq n$, converting the quantum geometric tensor into velocity matrix elements alone — no
$k$-derivative, no finite difference, no $dk$. Writing

$$ T_{ab}=\sum_{n\in W}\sum_{m\notin W}\frac{\langle n|v_a|m\rangle\langle m|v_b|n\rangle}{(\varepsilon_n-\varepsilon_m)^2}, $$

$$ g_{ab}=\mathrm{Re}\,T_{ab},\qquad F_{ab}=-2\,\mathrm{Im}\,T_{ab},\qquad Q_{ab}=g_{ab}-\tfrac i2 F_{ab}. $$

Pairs with both states inside the window $W$ do not appear: the window's geometric tensor is
a property of the projector onto $W$, blind to how $W$ is internally resolved — the same
multi-band (non-Abelian trace) quantity §13's Wilson loop and §15's overlap stencil compute.
`parsers.optical.kubo_quantum_geometry` returns the same `{"g", "berry_curvature", "Q"}`
keys and shapes `parsers.quantum_geometry.compute_quantum_geometry` does, so the two are
directly comparable. Both are in Bohr² (with $p$ in atomic units and $\varepsilon$ in
Hartree).

**Truncation.** The $m$-sum runs only over states Elk computed, i.e. up to `nstsv`, which
`nempty` sets; Elk's default leaves few empty states, so raise it
(`Calculation(extra_blocks={"nempty": [...]})`) and confirm stability between two values.
Core states are not in `nstsv` at all and are omitted entirely, suppressed by
$1/(\Delta\varepsilon)^2$ but not zero. For the metric's *diagonal* components every term is
$|v^a_{nm}|^2/(\Delta\varepsilon)^2\geq0$, so under-convergence there is a clean
underestimate; that monotonicity does **not** extend to $g_{ab}$ ($a\neq b$) or to $F_{ab}$,
whose terms carry either sign. Agreement with §13/§15 is therefore a few-percent claim, not
a machine-precision one.

A degeneracy guard raises `ValueError` if any window/outside pair is closer in energy than
`degeneracy_tol`: the window would not be a well-separated band group, its projector would be
discontinuous in $k$, and the $1/(\Delta\varepsilon)^2$ weight would be dominated by an
arbitrarily-split near-degenerate pair. Fails loud rather than clipping, matching
`parsers.quantum_geometry._hermitian_inv_sqrt`.

### The Berry-curvature sign convention, fixed by this cross-check

Adding an independent route to the Berry curvature exposed a sign error in §13's Python
arithmetic, which this change corrects: **elkpy now uses the standard convention
$\mathbf A=i\langle u|\nabla_{\mathbf k}u\rangle$, $\Omega=\nabla\times\mathbf A$ (Xiao,
Chang & Niu, RMP 82, 1959 (2010)) everywhere** — Wilson-loop curvature (§13), quantum-geometry
curvature (§15), Chern numbers, and the Kubo form here.

**What was wrong.** With link variables
$U(\mathbf k\to\mathbf k')=\langle u(\mathbf k)|u(\mathbf k')\rangle/|\cdot|$, expanding
$\langle u|u+\boldsymbol\delta\cdot\nabla u\rangle=1+\boldsymbol\delta\cdot\langle u|\nabla u\rangle=e^{-i\mathbf A\cdot\boldsymbol\delta}$
makes the product around a closed counterclockwise loop $e^{-i\oint\mathbf A\cdot d\mathbf l}$,
so its *argument* is minus the Berry phase. The standard discrete Berry phase carries that
explicit negation, $\gamma=-\mathrm{Im}\ln\prod_j\langle u_j|u_{j+1}\rangle$
(King-Smith–Vanderbilt/Resta, PRB 47, 1651(R) (1993)); `parsers.berry`'s `flux/area` step
omitted it, so `curvature` was $-\Omega$ and `chern_number` carried a flipped sign.

**How it was found and confirmed**, three independent ways:

1. *Synthetic, analytic.* On a massive Dirac model, where $\mathbf p=\partial H/\partial\mathbf k$
   is exact, the Kubo curvature matches direct numerical differentiation of
   $\Omega=-2\,\mathrm{Im}\langle\partial_x u|\partial_y u\rangle$ to 10 significant figures,
   while `parsers.berry`'s link arithmetic on the identical eigenvectors gave exactly minus that.
2. *End-to-end against a real binary.* On monolayer h-BN's occupied manifold the old
   `get_berry_curvature_path()` gave $+8.101$ at $K$ where the Kubo sum gives $-8.194$ — the same
   factor of $-1$, so the discrepancy was entirely in that Python step, which **confirmed the
   Fortran `moverlap`/`genolpq` conjugation convention** (previously resting only on a `zgemv`
   BLAS-semantics derivation with no runtime test).
3. *An analytic pin that had itself been mis-calibrated.* §15's Provost–Vallée spin-1/2 test
   asserted $F_{\theta\phi}=+\tfrac12\sin\theta$. Deriving it properly:
   $A_\phi=i\langle u|\partial_\phi u\rangle=-\sin^2(\theta/2)$, $A_\theta=0$, so
   $\Omega_{\theta\phi}=\partial_\theta A_\phi=-\tfrac12\sin\theta$, which integrates over
   the sphere to $-2\pi$ — the spin-$\tfrac12$ monopole charge, Berry phase $=-\tfrac12\times$
   solid angle. The old test value had been calibrated to the code rather than to the paper; it
   now matches the derivation.

**Why it survived until now.** Every existing check was sign-blind: bulk Si's Chern number is
$0=-0$; the h-BN benchmark is a *relative* $K/K'$ antisymmetry; $Z_2$ is a crossing-count parity;
§15's $\det g\geq(F_{12}/2)^2$ is even in $F$; and `test_berry_gauge_invariance.py`'s sign pins
pinned the FHS eq. 8 *arithmetic*, not its identification with $\Omega$. Nothing previously
reported was wrong except the overall sign of curvature and Chern numbers.

**Scope of the fix.** One place: `parsers.berry._berry_phase()`, which every consumer (mesh
curvature, path curvature, Chern number, `parsers.quantum_geometry`) now routes through, so the
convention is set once and cannot diverge again. $Z_2$ (§20/§21) is untouched — `parsers.wilson`
builds its own Wilson loop from `parse_berry_overlaps()`'s raw overlaps and never consumes
`flux`, and a crossing-count parity is invariant under reflecting the WCC curves anyway.

**Relationship to pyqula.** §13's path mode was written to follow
[`pyqula`](https://github.com/joselado/pyqula)'s `berry_curvature`, and pyqula's arithmetic
carries the same omission: `topologytk/overlap.py`'s `uij(wf1, wf2)[i,j] = <wf1_i|wf2_j>` is the
same $M(a,b)$ convention, and `topology.py` builds the identical counterclockwise link product
then takes `arctan2(Im det, Re det) / (4 dk^2)` with no negation. So elkpy's curvature and Chern
numbers now differ in sign from pyqula's. That is a deliberate choice of the published convention
over cross-project agreement — worth propagating to pyqula rather than reverting here.

### How to use in code

```python
hbn = Structure(HBN_AVEC, HBN_SPECIES).get_calculation(
    "hbn", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
    extra_blocks={"nempty": [12]},  # states for the Kubo sums to run over
)
hbn.get_energy()

from elkpy.parsers import optical

with hbn.eigenstate_session() as session:
    m = session.momentum((1 / 3, 1 / 3, 0))        # energies + p^a_nm at K

    # valley-selective circular dichroism of the band-edge transition
    optical.circular_polarization(m.pmat, valence=ist1, conduction=ist1 + 1)["eta"]

    # the same Berry curvature get_berry_curvature_path() computes, by a
    # completely different route -- same sign convention, directly comparable
    optical.kubo_berry_curvature(m.energies, m.pmat, ist0, ist1, directions=(1, 2))
    optical.kubo_quantum_metric(m.energies, m.pmat, ist0, ist1, directions=(1, 2))

    # band velocity = dE_n/dk exactly (Hellmann-Feynman, local potential)
    optical.band_velocity(m.pmat, ist=ist1)

# calc.get_momentum_matrix(kpath=..., npoints=...) does the session-reuse loop for you
# (same convention as §16's get_atom_projection()), returning a list of dicts.
path = hbn.get_momentum_matrix(kpath="GKM", npoints=60)
```

Note that `directions` in `parsers.optical` indexes **Cartesian** axes ($1,2,3=x,y,z$),
because that is what `genpmatk`'s three components are — unlike the identically-named
argument of `get_berry_curvature()`/`get_quantum_geometry()`, which indexes **reciprocal
lattice** axes. The two coincide for the `(1, 2)` default on a $c$-axis-out-of-plane 2D cell
(graphene/h-BN/WSe₂, where $\mathbf b_1,\mathbf b_2$ span $xy$, so both mean "the $z$
component"), not in general.

### Verification against a real compiled binary

`tests/test_calculation_momentum.py`, on monolayer h-BN:

- **Band velocity against the eigenvalue slope.** The Hellmann-Feynman identity
  $\mathbf v_n=\partial\varepsilon_n/\partial\mathbf k$ is exact for a local potential, and
  the right-hand side comes from eigenvalues alone — no matrix-element machinery — so this
  exercises the whole export path (matching, `genwfsv` expansion, `genpmatk` reduction, token
  protocol, parser packing) against a genuinely separate code path. Stepping in *fractional*
  coordinates differentiates along a reciprocal lattice vector, so the slope is
  $\mathbf v\cdot\mathbf b_i$, not $v_i$ — one dot product, and it exercises all three
  Cartesian components at once.
- **Valley-selective circular dichroism.** $\eta(K)=-\eta(K')$ with $|\eta|>0.98$ — measured
  $\eta(K)=-1.000000$, $\eta(K')=+1.000000$ to six figures, the $C_3$-enforced perfect
  selectivity. The relative sign is the published sign-of-the-effect prediction; the absolute
  sign depends on this structure's own conventions (which sublattice carries B vs N, lattice
  handedness), as with §17's $S_z(K)$.
- **Kubo curvature against the Wilson loop**, at $K$ and $K'$, agreeing to 5% in both
  magnitude and sign. This is the check that exposed the missing negation above: it
  originally failed with a clean factor of $-1$.
- **`nempty` stability**, `nempty=12` vs `nempty=20`, so the truncation is checked rather
  than assumed.
- **Kubo metric against `get_quantum_geometry()`'s** finite-difference stencil at $K$ (looser
  tolerance: the Kubo metric truncates at `nstsv`, the finite-difference one carries
  $O(dk^2)$ error plus `genolpq`'s Löwdin-corrected truncation floor).

`tests/test_parsers_optical.py` pins the arithmetic on synthetic data, ahead of any Elk run,
using the massive Dirac model $H_\tau=v(\tau k_x\sigma_x+k_y\sigma_y)+\Delta\sigma_z$ where
$\mathbf p=\partial H/\partial\mathbf k$ is exact: $\eta=\pm1$ at the two valleys and
decaying away from them; the Kubo curvature against direct differentiation (the sign pin) and
against the closed form $\pm1/(2\Delta^2)$ at $k=0$ (the magnitude pin), plus the two-band sum
rule $\sum_n\Omega_n=0$; the metric against its closed form $\mathbb 1/(4\Delta^2)$, its
positive-semi-definiteness, and $\det g\geq(F/2)^2$ saturating in this two-band case; and both
guard conditions (a window not separated from the outside, a window covering every state).

## 23. Parity eigenvalues at the TRIM, and the Fu-Kane symmetry-indicator $Z_2$

*Physics writeup: `docs/physics.tex` Part XII.*

`Calculation.get_parity(k, ist0, ist1)` / `EigenstateSession.parity()` (task 9002's
`PARITY` query, `patches/0008-inversion-parity-operator.patch`, `elkpy_parity` in
`src/elkpy_eigenstates.f90`) return the inversion operator

$$P_{mn}=\langle\psi_m|\hat I|\psi_n\rangle$$

over a band window at one $k$-point, together with all `nstsv` eigenvalues of the same
diagonalisation. `Calculation.get_fu_kane_invariant(ist0, ist1, dimension=)` drives it
over the time-reversal-invariant momenta and returns the $Z_2$ classification.

**The physics** (Fu & Kane, *Topological insulators with inversion symmetry*, PRB **76**,
045302 (2007)): for a crystal with an inversion centre, the $Z_2$ invariants of a
time-reversal-invariant insulator are fixed entirely by parity eigenvalues at the 8 (3D)
or 4 (2D) TRIM, with **no Brillouin-zone integration at all**:

$$\delta_i=\prod_{m=1}^{N}\xi_{2m}(\Gamma_i),\qquad (-1)^{\nu_0}=\prod_{i=1}^{8}\delta_i,\qquad (-1)^{\nu_k}=\prod_{\Gamma_i:\,k_i=\pi}\delta_i$$

with $\xi=\pm1$ the parity eigenvalues and the product in $\delta_i$ running over **one
member of each Kramers pair**. This is the cheap counterpart to §20/§21's
Wannier-charge-center pumping: 8 diagonalisations instead of a mesh sweep, at the cost of
requiring the inversion symmetry to actually be present. The two are complementary, not
redundant — the WCC method is general, this one is exact.

**Fortran: almost nothing new.** The transformation of first-variational coefficients
under a crystal symmetry is lifted from upstream `getevecfv.f90`, which does exactly this
to reconstruct wavefunctions at symmetry-related $k$-points — code exercised by every
symmetry-reduced (`reducek=1`) Elk run. Elk makes it markedly simpler than expected:
`findsymcrys.f90` moves inversion to be **crystal symmetry element 2 with a zero
translation** whenever `tsyminv` is true (it clears `tsyminv` otherwise, having already
shifted the basis to put the inversion centre at the origin), so upstream's
non-zero-translation branch drops out. The local-orbital Bloch phases are kept — those are
$e^{-2\pi i\mathbf k\cdot\tau}$ terms at atomic positions, not the (zero) translation
phase — and `rotzflm` handles the improper rotation, computing $\det R<0$ and applying the
$(-1)^\ell$ factor via `ylmrot`.

The overlap itself needs no new machinery either. Because $\hat I\mathbf k\equiv\mathbf k$
at a TRIM, the rotated coefficients live in the **same** LAPW basis, so
`genwfsv` + `genolpq` at $q=0$ (the session's existing `OVERLAP` path, §14) computes
$\langle\psi_m|\hat I\psi_n\rangle$ directly — no basis-overlap matrix required. At $q=0$
the muffin-tin phase factor is identically unity and is set directly, rather than through
`genylmv` on a null vector.

Inversion is its own inverse, so upstream's active/passive direction convention is
immaterial here. **That will not be true for a $C_n$ generalization**, where the direction
must be pinned explicitly first.

**Why $P^2=\mathbb 1$ survives the band-window truncation.** Since $[\hat I,\hat H]=0$,
inversion preserves energy eigenspaces, so a window gapped from the rest of the spectrum
is $\hat I$-invariant and $P$ restricted to it is Hermitian with eigenvalues exactly
$\pm1$. This is unlike §19's $\mathfrak{su}(2)$/Casimir identities, which a band window
genuinely does spoil: those need a resolution of identity over *every* state, whereas an
invariant window supplies its own.

**Two traps, both real and both now guarded.** First, the parity eigenvalues are **not**
$P$'s diagonal entries: TRIM spectra are heavily degenerate and the diagonalisation
returns an arbitrary basis within each multiplet, so `parity_eigenvalues()` diagonalises.
Second, with `nspinor=2` Kramers partners share a parity eigenvalue, so the product over
*all* occupied states is identically $+1$ and carries no information whatsoever — the
pairing in $\delta_i$ is the entire content of the formula, implemented as
$\delta=(-1)^{N_-/2}$ with $N_-$ the number of $-1$ eigenvalues.

**Sign immunity.** A global sign error in $P$ would flip every $\xi$ at once, but every
invariant here is a product over an *even* number of TRIM (8 for $\nu_0$, 4 for each
$\nu_k$ and for the 2D $\nu$), so it cancels identically. This is not hypothetical — it
was observed directly: different band windows on the cesium structure below returned all
eight $\delta_i$ flipped, with $\nu_0$ unchanged. So no absolute per-band sign pin is
chased; the checks with teeth are structural ($P$ Hermitian, $P^2=\mathbb 1$, eigenvalues
$\pm1$, even Kramers counts) plus agreement with §20.

### Verification, and a retraction it forced

`tests/test_calculation_parity.py`:

- **Graphene** (`soc_scale=3000`, the §20 fixture): $\nu=1$ from 4 $k$-points, matching
  `get_z2_invariant()`'s WCC answer for the same system. The two share no arithmetic, and
  the parity route additionally exercises a Fortran path nothing else in elkpy touches.
- **Error paths**: h-BN (different species on the two sublattices, so no inversion centre)
  is refused by the Fortran side with the session left alive; a non-TRIM $k$-point is
  refused in Python before the query is sent, and independently in Fortran as a rotated
  $G+k$ vector with no partner in the same basis.
- **Structure**: $P$ Hermitian, $P^2=\mathbb 1$, eigenvalues $\pm1$, even Kramers counts.

**The [111]-dimerized diamond ("cesium") structure of §21 is topologically trivial,
$(0;000)$ — retracting §21's reported $\nu_0=1$.** The evidence:

| | result |
|---|---|
| Parity indicator (exact, no mesh), 6 gapped windows | $(0;000)$ every time |
| WCC on the disputed plane, $(n_{kx},n_t)=(12,7),(18,9),(24,13),(32,17)$ | $z=1,0,1,0$ |
| Direct gap on that plane ($13\times13$ scan) | $\geq0.19$ eV, smooth |

The disagreement localizes cleanly: all three $k_i=0$ planes differ (WCC 1, parity 0),
all three $k_i=\pi$ planes agree (both 0). The WCC number on the disputed plane
**oscillates rather than converging**, so §21's single $n_{kx}=12$ sample carried no
information. The plane is robustly gapped, so this is *not* §20's graphene mesh-aliasing
failure mode (a $10^{-3}$-wide anticrossing) — the WCC crossing count is simply
under-resolved at practical meshes for this system. The parity result, by contrast, is
mesh-free and was checked to be independent of the band window across six
independently-gapped choices (including one dropping a semicore block below a 69 eV gap,
exercising $Z_2$'s additivity mod 2 over gapped groups).

The physical reading: a hypothetical Cs diamond lattice with SOC scaled $3000\times$ does
not realize the phase of FKM's *single-orbital tight-binding* model, and §21's apparent
agreement was an unconverged number landing on the hoped-for answer.

**Which WCC calculations to distrust — a narrower statement than "3D sweeps".** A mesh
ladder on two real 2D systems shows the method is not generally fragile:

| system | occupied bands | min gap | WCC at $(n_{kx},n_t)=(8,5),(12,7),(18,9),(24,13)$ |
|---|---|---|---|
| graphene, `soc_scale=3000` | 6 | several eV | 1, 1, 1, 1 |
| bismuthene | 30 | ~0.5 eV | 1, 1, 1, 1 |
| cesium, one plane | 38 | 0.19 eV | 1, 0, 1, 0 |
| Bi$_2$Se$_3$, six-plane sweep | 78 | 0.26 eV | wrong ($\nu_0=0$ vs the true 1) |

Zero wobble across a 9$\times$ k-point range on both 2D systems, so no general convergence
warning on WCC pumping is warranted. Note the failures are *not* separated from the
successes by dimensionality — cesium's failure is on a single plane, which is a 2D
computation. What the two failures share is a **large occupied manifold and a sub-0.3 eV
gap**; the two successes have either few bands or a comfortable gap. The honest reading is
that the largest-gap crossing count becomes under-resolved when many Wannier bands are
packed together, not that 3D is broken.

The practical recommendation, therefore, is narrow: **report each plane at two meshes and
treat disagreement as "not converged"**, rather than trusting a single sweep — and where
the crystal is centrosymmetric, prefer the parity indicator, which is exact.

The WCC implementation itself is not impugned: its algebraic axis-split consistency held,
and it agrees with the parity route on both graphene and bismuthene.

This also bears on §21's other open item. Bulk Bi$_2$Se$_3$ gave $\nu_0=0$ against a
literature $(1;000)$, with mesh convergence explicitly untested because it was too
expensive; the cesium oscillation is direct evidence that the 3D WCC sweep is
under-converged at that scale, which is now the leading explanation there too. Bi$_2$Se$_3$
is inversion-symmetric, so the parity indicator can settle it in minutes — not done here
only because its structure is not a checked-in fixture (§21 records that no test asserts
it) and would need re-sourcing from COD 9011965 first.

**The individual $\delta_i$ depend on where the inversion centre sits; $\nu$ does not.**
This looks exactly like a bug the first time it is seen, and is not one. On bismuthene the
three $C_3$-related M points came out $-1,+1,-1$ — not equal, despite being related by a
symmetry of the crystal. The cause is that Elk chooses the inversion centre itself
(`findsymcrys.f90` shifts the basis to put it at the origin) and here picked the Bi-Bi
**bond midpoint**, which is displaced from the $C_3$ axis by $t=(0,\tfrac12,0)$. A centre
offset by $t$ multiplies $\delta(\mathbf k)$ by $\big(e^{2\pi i\,\mathbf k\cdot2t}\big)^{N}$
with $N$ the number of occupied Kramers pairs — so the asymmetry appears only for **odd**
$N$ (here $N=15$). Transporting the centre onto the axis restores
$\delta(\Gamma)=-1$ with all three $\delta(M)=+1$: $C_3$-symmetric, one odd TRIM at
$\Gamma$, matching the $\Gamma$-centred s-p band inversion bismuthene is documented to
have. `SYMCRYS.OUT` confirms both $C_3$ elements carry non-lattice translations.

Crucially the invariant is unaffected — both assignments give $\nu=1$, since the phases
enter an even number of TRIM products. So `deltas` should be read as *a* valid set for
Elk's chosen origin, not as a canonical per-TRIM fingerprint, and only the combination
is physical. `tests/test_calculation_parity.py` deliberately does not assert $C_3$
uniformity of the deltas for this reason.

**A third guard, found the hard way.** A window boundary sitting *inside* a band group
passes every check above — Hermitian, $\pm1$ eigenvalues, even Kramers counts — while
describing no topological group at all. On this structure `ist0=19` did exactly that and
returned a confident $\nu_0=1$. `parsers.symmetry.check_window_gap()` now rejects it, in
the same spirit as `parsers.berry.check_gap()`.

### How to use in code

```python
calc = Structure(AVEC, SPECIES).get_calculation(
    "run", xc="PW", ngridk=(4, 4, 4), rgkmax=7.0, spinorb=True
)
calc.get_energy()

# the whole classification, from 8 k-points
result = calc.get_fu_kane_invariant(1, ist1, dimension=3)
result["nu0"], result["nu"], result["deltas"]

# or one TRIM's operator directly
from elkpy.parsers import symmetry
p = calc.get_parity(k=(0.5, 0.5, 0.5), ist0=1, ist1=ist1)
symmetry.parity_eigenvalues(p.pmat)   # +-1, NOT p.pmat's diagonal
symmetry.trim_delta(p.pmat)           # that TRIM's delta
```

Requires `spinorb=True`: without spin-orbit coupling, spin-SU(2) forces the two spin
sectors to opposite Chern numbers and $Z_2$ is trivially 0, so the invariant carries no
information. `dimension=2` uses the 4 TRIM of the first two reciprocal directions, holding
the third (vacuum/stacking) at 0.

## 24. Optical absorption: Elk's dielectric function, and the circular-polarization-resolved spectrum

*Physics writeup: `docs/physics.tex` Part XIII.*

Two related additions, one wrapping upstream Elk and one going beyond it.

### 24.1 `get_dielectric_function()` -- wrapping task 121

`Calculation.get_dielectric_function(components=, wplot=, nwplot=, swidth=,
intraband=, ngridk=)` runs tasks 120 then 121 (`src/writepmat.f90`,
`src/dielectric.f90`) and parses `EPSILON_ij.OUT`/`SIGMA_ij.OUT`
(`parsers/dielectric.py`). Task 121 is the independent-particle
(random-phase, no local fields, no excitons) dielectric tensor, evaluated
from the Kubo-Greenwood formula of *Physica Scripta* **T109**, 170 (2004) --
the same reference `genpmatk.f90` cites for the momentum matrix elements
(§22):

$$
\sigma_{ij}(\omega)=\frac{i}{N_k\Omega}\sum_{\mathbf k}\sum_{n,m}
\frac{f_n\left(1-f_m/f_{\max}\right)}{\varepsilon_{m}-\varepsilon_{n}}
\left[\frac{p^i_{nm}\,\overline{p^j_{nm}}}{\omega-\varepsilon_{mn}+i\varsigma}
+\frac{\overline{p^i_{nm}\overline{p^j_{nm}}}}{\omega+\varepsilon_{mn}+i\varsigma}\right],
$$

$$
\epsilon_{ij}(\omega)=\delta_{ij}+\frac{4\pi i\,\sigma_{ij}(\omega)}{\omega+i\varsigma},
$$

with $\varepsilon_{mn}=\varepsilon_m-\varepsilon_n$, $\Omega$ the unit-cell
volume, $N_k$ the number of **non-reduced** $k$-points, $f_n$ the occupation
numbers, $f_{\max}$ = `occmax` (2 without spin polarization, 1 when
`nspinor`=2) and $\varsigma$ = `swidth`, whose reciprocal is the relaxation
time. `getpmat` supplies $p^i_{nm}$ at each non-reduced point by rotating
task 120's reduced-mesh `PMAT.OUT` with the crystal symmetry that maps it
there.

Two version-coupled details, both taken from the Fortran rather than the
manual and both load-bearing:

- **Task 121 needs task 120 first.** `dielectric.f90` calls `getpmat`, which
  reads `PMAT.OUT` from disk and `stop`s if it is absent or has a different
  `nstsv`. Pairing the two tasks in one `_run_resumed()` call is why this is
  a named method rather than a bare `run_tasks([121])`.
- **The output filenames carry indices** (`EPSILON_11.OUT`, one file per
  `optcomp` entry), so `spec.py` grew an `OUTPUT_FILE_TEMPLATES` dict beside
  the fixed-name `OUTPUT_FILES`. The format inside is the two-block layout
  `moke.f90` also writes: `nwplot` `(omega, value)` pairs for the real part,
  a blank line, then the same for the imaginary part. The energy grid is
  $\omega_i = \omega_1 + (\omega_2-\omega_1)(i-1)/N_\omega$ with
  $\omega_1=\max(\texttt{wplot(1)},0)$ -- non-negative, and **excluding** the
  upper endpoint.

### 24.2 `get_circular_absorption()` -- the polarization-resolved spectrum

Stock Elk gives $\sigma_{xx}$, $\sigma_{xy}$ and the rest of the Cartesian
tensor, but never $\sigma_\pm$. The circular channels are what carry the
valley physics: with the circular-basis interband matrix elements of §22,

$$P_\pm(\mathbf k)=p^x_{cv}(\mathbf k)\pm i\,p^y_{cv}(\mathbf k),$$

the absorption resolved by photon helicity is

$$
\operatorname{Im}\epsilon_\pm(\omega)=\frac{4\pi^2}{\Omega\,\omega^2}
\sum_{\mathbf k}W_{\mathbf k}\sum_{v,c}f_v\!\left(1-\frac{f_c}{f_{\max}}\right)
\tfrac12\left|P_\pm(\mathbf k)\right|^2
\delta\!\left(\omega-(\varepsilon_c-\varepsilon_v)\right),
$$

which is exactly the $\varsigma\to0$ limit of §24.1's $\operatorname{Im}
\epsilon_{ii}$ with the linear intensity $|p^i_{cv}|^2$ replaced by the
circular one. `Calculation.get_circular_absorption()` evaluates it from
elkpy's own arbitrary-$k$ momentum matrix elements (§22's task-9002
`MOMENTUM` query) over the full non-reduced mesh, with all arithmetic in
`parsers/optical.circular_absorption()`; **no new Fortran at all**, and one
`eigenstate_session()` for the whole sweep.

Three things make this the right shape:

- **The mesh must not be symmetry-reduced.** $|P_\pm|^2$ is *not* invariant
  under the operations that fold the mesh -- that non-invariance *is* the
  dichroism -- so the usual reduced-mesh-with-weights sum, correct for the
  linear components, is wrong here. `Calculation._kmesh()` reproduces Elk's
  own non-reduced grid, $\mathbf k=(\mathbf i+\texttt{vkloff})/\texttt{ngridk}$.
- **Occupations come from Elk's own `EIGVAL.OUT`** (`parsers/eigval.py`), the
  same `occsv` array `dielectric.f90` reads, and are required to be
  $k$-independent -- an independent-particle interband spectrum is a gapped
  system's object, and `occupations_if_uniform()` raises on a metal rather
  than silently mis-counting a Fermi surface. This also sidesteps §13's
  standing pitfall of inferring a band count from a valence-electron count.
- **The $\pm$ labelling follows `circular_polarization()`** (§22), i.e.
  $|P_+|^2$ is the $\sigma^+$ intensity. The relative sign between valleys is
  the published physics; the absolute handedness is a
  structure-convention pin, exactly as for $S_z(K)$ (§17) and $\eta(K)$ (§22).

### 24.3 The two lineshapes, and why the naive one is not enough

The $\delta$-function formula above is the textbook independent-particle
spectrum, but it is only the $\varsigma\to0$ limit of what task 121 actually
evaluates. Putting the real intensity $z=|e\cdot p_{cv}|^2$ into §24.1 and
taking $\operatorname{Im}\epsilon=4\pi\operatorname{Re}[\sigma/(\omega+i\varsigma)]$
gives, per transition, exactly

$$
\frac{4\pi\varsigma\,z}{\Omega\,\Delta\,(\omega^2+\varsigma^2)}
\left[\frac{2\omega-\Delta}{(\omega-\Delta)^2+\varsigma^2}
+\frac{2\omega+\Delta}{(\omega+\Delta)^2+\varsigma^2}\right],
\qquad \Delta=\varepsilon_c-\varepsilon_v,
$$

whereas the $\delta$-form with a Lorentzian of width $\varsigma$ gives
$(4\pi^2 z/\Omega\omega^2)\cdot(\varsigma/\pi)/[(\omega-\Delta)^2+\varsigma^2]$.
The two agree at resonance, $\omega=\Delta$, and their ratio elsewhere is

$$\frac{2\omega-\Delta}{\Delta}\cdot\frac{\omega^2}{\omega^2+\varsigma^2},$$

which is **first order** in $(\omega-\Delta)/\Delta$: a few percent across one
linewidth, not a small correction, and *not* a defect of either code. Two
further consequences worth stating, because both were measured rather than
anticipated:

- The $\delta$-form has a spurious $1/\omega^2$ blow-up at small $\omega$,
  where a Lorentzian's fat tail is multiplied by a diverging prefactor. On
  h-BN this produced a fake "peak" of $\sim2\times10^3$ at
  $\omega=10^{-3}\,$Ha, dwarfing the real one at $0.236\,$Ha.
- The exact form's two terms cancel *exactly* at $\omega=0$ (the resonant one
  alone even turns negative below $\omega=\Delta/2$), so keeping only the
  resonant term is a real error that no peak-region comparison would catch.

`broadening="elk"` (the default) therefore evaluates the exact finite-$\varsigma$
response; `"lorentzian"`/`"gaussian"` give the textbook $\delta$-form and are
kept because that *is* the formula the physics is usually quoted in, and
because the two agree on the integrated oscillator strength
$\int\omega^2\operatorname{Im}\epsilon\,d\omega$ at any $\varsigma$.

### 24.4 Verification

The cross-check is an algebraic identity, not a symmetry assumption:

$$|P_+|^2+|P_-|^2=2\left(|p^x_{cv}|^2+|p^y_{cv}|^2\right)
\;\Longrightarrow\;
\operatorname{Im}\epsilon_++\operatorname{Im}\epsilon_-
=\operatorname{Im}\epsilon_{xx}+\operatorname{Im}\epsilon_{yy},$$

so a task-121 run asked for the $(1,1)$ and $(2,2)$ components on the same
mesh, `nempty` and `swidth` reproduces the polarization-summed elkpy
spectrum. Nothing is shared between the two routes but the ground state:
Elk reads reduced-mesh momentum matrix elements off `PMAT.OUT` and rotates
them, sums in Fortran over `nkptnr`, and forms $\epsilon$ from $\sigma$;
elkpy re-diagonalises at every non-reduced point through the task-9002
session and does everything else in NumPy.

Against a real compiled binary, monolayer h-BN (the §22 slab; $6\times6\times1$
mesh, `nempty`=12, `swidth`=0.005 Ha, `rgkmax`=7):

- **The polarization-summed spectrum reproduces task 121 essentially exactly.**
  Over the 297 grid points where $\operatorname{Im}\epsilon$ exceeds 2% of its
  peak, the largest relative deviation is $9.3\times10^{-4}$ and the median is
  $2.0\times10^{-5}$; at the peak, 15.414871 (elkpy) against 15.414867 (Elk).
  The integrated oscillator strength
  $\int\omega^2\operatorname{Im}\epsilon\,d\omega$ agrees to $8\times10^{-5}$
  relative. The residual is at the level expected from the two subdirectories'
  independent task-1 SCF continuations and from `getpmat`'s symmetry rotation
  (Elk's own $\operatorname{Im}\epsilon_{xx}$ vs $\operatorname{Im}\epsilon_{yy}$,
  which exact symmetry makes equal, differ by $1\times10^{-4}$ of the peak).
- **The zone-integrated circular channels are equal to machine precision**:
  $\max|\operatorname{Im}\epsilon_+-\operatorname{Im}\epsilon_-|=5.6\times10^{-13}$
  against a peak of 15.4, i.e. $4\times10^{-14}$ relative -- the time-reversal
  statement above, and confirmation that nothing in the $\pm$ bookkeeping is
  biased.
- **Restricting the sum to one valley gives perfect circular selectivity**:
  at the absorption edge ($\omega=0.167$ Ha $=4.5$ eV) $\eta(K)=-0.9999$ and
  $\eta(K')=+0.9999$, with $\operatorname{Im}\epsilon_-=122.6$ against
  $\operatorname{Im}\epsilon_+=0.0044$ at $K$. This is §22's $\eta(K)=-1$
  band-edge result reappearing as a *spectrum*, and it is the output stock Elk
  cannot produce.
- **The $\delta$-lineshape's cost is measured, not assumed**: the same sum with
  `broadening="lorentzian"` at this $\varsigma$ deviates from task 121 by up to
  13.6% (median 2.2%) across the peak region, plus the $1/\omega^2$ artifact
  below the edge (a spurious $2\times10^3$ "peak" at $\omega=10^{-3}$ Ha).
  With `"gaussian"` it is worse still (median 25% in the peak region), as
  expected: Elk's broadening is Lorentzian by construction.

`tests/test_parsers_absorption.py` pins the arithmetic ahead of any Elk run,
on the massive Dirac model of §22: the $\delta$-form against its closed form
(prefactor, $1/\omega^2$, occupation weight and Lorentzian normalization at
once), the `"elk"` form against `dielectric.f90`'s expression transcribed
directly, the $\omega=0$ cancellation, the convergence of the two lineshapes
as $\varsigma$ falls, the circular-vs-linear identity above against an
independently written linear sum, $\eta$ at the peak against the already
trusted `circular_polarization()`, the $k\to-k$ cancellation and its recovery
under a mask, and oscillator-strength conservation for both the Lorentzian
and the Gaussian.

### 24.5 What this is not

The independent-particle spectrum has no electron-hole interaction, so it
misses the excitonic physics that dominates real h-BN optics: the measured
optical gap sits well below the calculated absorption onset, and the
oscillator strength is redistributed into a bound exciton peak. (Elk's own
`dielectric_bse.f90` is the route to that; not wrapped here.) The
Kohn-Sham gap itself is also LDA-underestimated. In addition, for a slab the
dielectric function is a **supercell** quantity -- $\epsilon-1$ scales like
$1/L_z$ with the vacuum thickness, so absolute values are not the 2D
material's own response. None of this affects the comparison above, where both
sides use the same cell, but all of it affects reading the numbers as h-BN's
optics.

### How to use in code

```python
hbn = Structure(HBN_AVEC, HBN_SPECIES).get_calculation(
    "hbn", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
    extra_blocks={"nempty": [12]},   # both routes must see the same nstsv
)
hbn.get_energy()

# Elk's own dielectric tensor (tasks 120 + 121)
elk = hbn.get_dielectric_function(components=((1, 1), (2, 2)), swidth=0.005)
w, eps_xx = elk["energies"], elk["epsilon"][(1, 1)]

# elkpy's circular-resolved spectrum over the same non-reduced mesh
out = hbn.get_circular_absorption(swidth=0.005)
out["eps2_total"]                      # == Im eps_xx + Im eps_yy
out["eps2_plus"], out["eps2_minus"]    # equal once summed over the zone

# the valley physics: restrict the k-sum to one valley. The returned dict
# carries everything the sum needs, so re-weighting costs no Elk time.
from elkpy.parsers import optical
v = hbn.get_circular_absorption(kpoints=[(1/3, 1/3, 0), (-1/3, -1/3, 0)],
                                swidth=0.005)
one = optical.circular_absorption(
    v["kdata"], v["omega"], v["occupations"], v["volume"],
    occmax=v["occmax"], swidth=0.005, weights=[1.0, 0.0],
)
one["eta"]     # ~ -1 across the K band edge, +1 for the K' mask
```

One caveat on `swidth` worth carrying into the docs: it is written into the
task-121 subdirectory's `elk.in`, so it also governs the smearing of the
task-1 SCF continuation that runs there, while `get_circular_absorption()`'s
own session uses whatever the `Calculation` was built with. For a gapped
system that is immaterial (both give integer occupations and identical
eigenvalues, which is why the agreement above is at the $10^{-5}$ level); for
a metal the two sides would be smeared differently and should be aligned via
`extra_blocks={"swidth": [...]}` on the `Calculation` itself. Elk's own MOKE
example raises `swidth` after the ground state for exactly this reason.

## 25. The effective-mass tensor from the k·p sum rule

*Physics writeup: `docs/physics.tex` Part XIV.*

## The formula

In the Bloch Hamiltonian obtained by pulling the plane-wave factor out of the
wavefunction,

$$ H(\mathbf k)=e^{-i\mathbf k\cdot\mathbf r}He^{i\mathbf k\cdot\mathbf r}
=\tfrac12(\mathbf p+\mathbf k)^2+V_s(\mathbf r) $$

(Hartree atomic units, $\hbar=m_e=1$), the $\mathbf k$-dependence is explicit and
elementary:

$$ \frac{\partial H}{\partial k_a}=p_a+k_a=v_a,\qquad
\frac{\partial^2H}{\partial k_a\partial k_b}=\delta_{ab}. $$

Ordinary non-degenerate second-order perturbation theory in $\mathbf k$ then gives
the band curvature exactly, with no derivative of anything numerical:

$$ \boxed{\;\Big(\frac{1}{m^*}\Big)^{ab}_n=\frac{\partial^2\varepsilon_n}{\partial k_a\partial k_b}
=\delta_{ab}+2\sum_{m\neq n}\frac{\mathrm{Re}\big[p^a_{nm}p^b_{mn}\big]}{\varepsilon_n-\varepsilon_m}\;} $$

with $p^a_{nm}=\langle\psi_{n\mathbf k}|(-i\nabla+\text{SOC})_a|\psi_{m\mathbf k}\rangle$
the momentum matrix elements §22 exports (for a local Kohn–Sham potential these are
equally the velocity matrix elements). The mass tensor $m^*_{ab}$ is the matrix
inverse; `"inverse_mass"` is the primitive here, and `"mass"` is `None` when that
inverse does not exist — a direction of vanishing curvature is an infinite mass, not
an error.

Each symbol: $n$ the band whose mass is asked for, $m$ every other state at the same
$\mathbf k$, $\varepsilon$ the Kohn–Sham eigenvalues (Hartree), $a,b$ Cartesian axes.

**Reading the formula physically.** The $\delta_{ab}$ is the bare free-electron term:
an electron with no interband coupling at all has $m^*=m_e$. Every deviation from unit
mass is interband repulsion — states *below* band $n$ ($\varepsilon_m<\varepsilon_n$,
positive denominator) push the curvature up, states *above* push it down, each weighted
by how strongly the velocity operator connects them. That decomposition — *which*
coupling to *which* band produces the mass — is real physics a finite-difference
curvature cannot deliver, and `decompose=True` returns it term by term.

**Why `free_electron_term=False` exists.** The $\delta_{ab}$ comes from
$\tfrac12(\mathbf p+\mathbf k)^2$, i.e. from the fact that the crystal Hamiltonian's
basis is the complete Hilbert space. A model Hamiltonian written directly in a finite
band basis — a $\mathbf k\cdot\mathbf p$ or tight-binding $H(\mathbf k)$ — has
$\partial^2H/\partial k_a\partial k_b=0$ instead, and the sum over the model's own bands
is the whole answer. The flag is what makes the massive-Dirac unit tests an exact,
analytic pin rather than an approximate one.

## The Thomas–Reiche–Kuhn f-sum, stated honestly

`parsers.optical.oscillator_strength_sum()` returns

$$ f^{ab}_n=\sum_{m\neq n}\frac{2\,\mathrm{Re}\big[p^a_{nm}p^b_{mn}\big]}{\varepsilon_m-\varepsilon_n}, $$

the usual velocity-form oscillator strengths of the transitions out of band $n$. This is
the *same arithmetic* as the mass with the denominator reversed, so identically, at every
$\mathbf k$-point,

$$ f^{ab}_n=\delta_{ab}-\Big(\frac{1}{m^*}\Big)^{ab}_n . $$

"One unit of oscillator strength per electron" is therefore **not** a pointwise statement
in a crystal — read as one, it would assert that every band is flat. The true statement is
the Brillouin-zone average: for a filled band the zone average of
$\partial^2\varepsilon_n/\partial k_a\partial k_b$ vanishes (it is the second derivative of
a periodic function integrated over a period), hence

$$ \big\langle f^{ab}_n\big\rangle_{\rm BZ}=\delta_{ab}, $$

which is the f-sum rule behind the optical conductivity's spectral weight,
$\int\sigma_1(\omega)\,d\omega=\pi n/2$. Its practical role here is as a truncation
diagnostic with an exactly known target.

## Convergence: the first power of $\Delta\varepsilon$ is the whole story

The denominator appears to the **first** power, unlike §22's Kubo geometric sums
$T_{ab}\sim1/(\Delta\varepsilon)^2$. High-lying intermediate states are suppressed only as
$1/\Delta\varepsilon$, so the sum converges markedly more slowly in `nempty`, and three
distinct kinds of missing state matter:

1. everything above `nempty` (fixable, and the thing the sweep below measures);
2. the core states, which are not among the `nstsv` valence states at all — they lie
   *below* band $n$, so their omission biases the mass in the **opposite** direction from
   (1), which is what makes the observed overshoot self-consistently attributable to (1);
3. the high continuum the finite (`rgkmax`), energy-linearized LAPW basis cannot represent
   even in principle.

**Sign logic, which is what makes an under-converged number readable.** Every state removed
by truncation lies *above* band $n$, so each omitted diagonal term
$-2|p^a_{nm}|^2/|\Delta\varepsilon|$ is negative: a truncated $(1/m^*)^{aa}$ is an
**overestimate**, and falls monotonically as `nempty` rises. That monotonicity does not
extend to off-diagonal components, whose terms carry either sign.

`nstates=N` truncates the $m$-sum from Python. That is *exactly* equivalent to having run
Elk with the corresponding smaller `nempty` — verified directly on bulk Si, where the
`nempty=8` run's own full sum and the `nempty=40` run truncated to the same 21 states agree
to $10^{-11}$ (their eigenvalues agree to $10^{-11}$ too), which is what makes a whole
convergence sweep affordable out of a single ground state.

## Verification against a real compiled binary (bulk Si)

`tests/test_calculation_effective_mass.py`; Si, `ngridk=(4,4,4)`, `rgkmax=7.0`,
`nempty=40` (85 states), compared against `get_effective_mass()` (Elk task 25,
`src/effmass.f90`), which fits a polynomial to *eigenvalues* on a 27-point mesh of
Cartesian displacements `deltaem=0.025` around the point and differentiates it. Task 25's
"matrix of eigenvalue derivatives" is the inverse-mass tensor; its "effective mass tensor"
is that matrix inverted.

**Γ, band 1** (the non-degenerate $\Gamma_1$ $s$-bonding band, isolated by 12 eV):

| states retained | $(1/m^*)^{xx}$ | deviation from task 25 | ${\rm Tr}\,f/3$ |
|---|---|---|---|
| 4 (occupied only) | 1.0000 | +16.1% | 0.000 |
| 8 | 0.9276 | +7.7% | 0.217 |
| 20 | 0.8984 | +4.3% | 0.305 |
| 40 | 0.8914 | +3.5% | 0.326 |
| 60 | 0.8900 | +3.4% | 0.330 |
| 85 (all) | 0.8876 | +3.1% | 0.337 |
| task 25 (finite difference) | 0.8611 | — | 0.139 |

Monotone from above, exactly as the sign argument requires, and still descending at 85
states: the sum rule recovers $0.1124/0.1389\approx81\%$ of the interband repulsion that
lowers the curvature below the free-electron 1. The residual 19% is the missing continuum —
the highest computed state sits at only 3.7 Ha above the band.

**A generic, low-symmetry k-point** $(0.2,0.15,0.1)$, band 1, where the tensor is not forced
isotropic and the off-diagonal components are a real test:

| | $xx$ | $yy$ | $zz$ | $xy$ | $xz$ | $yz$ |
|---|---|---|---|---|---|---|
| sum rule (85 states) | 0.8675 | 0.8656 | 0.8642 | −0.0199 | −0.0073 | −0.0050 |
| task 25 | 0.8369 | 0.8360 | 0.8352 | −0.0208 | −0.0076 | −0.0053 |

3.5–3.7% on the diagonal (same convergence trend: 15.1% → 3.6%), and 4–6% on the
off-diagonals, which are two orders of magnitude smaller and carry the correct sign and
magnitude.

**Where the two routes disagree more, and whose fault it is.** For bands with a close
neighbour (min gap $\sim0.02$ Ha) at the generic point, differences reach a few tenths of an
atomic unit (e.g. band 5, $xx$: −1.14 vs −1.36). Part of that is task 25's, not the sum
rule's: its polynomial fit samples over $\pm0.025$ Bohr$^{-1}$, comparable to the scale on
which those bands curve near an avoided crossing, so the quadratic fit is itself
questionable there. Neither route is ground truth in that regime.

**The decomposition, and a selection rule that could not be a numerical accident.** At Γ,
diamond Si's states have definite parity about the bond centre and $\mathbf p$ is odd, so a
transition between two even states is forbidden. Band 1 ($\Gamma_1$, even) accordingly gets
*exactly nothing* from the even valence triplet $\Gamma_{25'}$ (bands 2–4): those
contributions are $\sim10^{-30}$, i.e. zero at machine precision, not merely small. Its
entire mass comes from the odd conduction triplet $\Gamma_{15}$ (bands 5–7, $-0.0241$ each,
together 64% of the total deviation from 1) and its higher analogues at 1.18 Ha. This is the
textbook selection rule behind Si's optical spectrum ($\Gamma_{25'}\to\Gamma_{15}$ allowed,
$\Gamma_1\to\Gamma_{25'}$ forbidden), read straight off the mass decomposition.

**The f-sum rule, Brillouin-zone averaged.** Summing $f_n$ over Si's four occupied bands on a
generic (unshifted-symmetry-free) $\mathbf k$-mesh and averaging:

| mesh | $\langle{\rm Tr}f/3\rangle$ per occupied band, 85 states | at 40 states |
|---|---|---|
| $4^3=64$ points | 1.103 | 1.096 |
| $6^3=216$ points | 0.971 | 0.964 |
| $8^3=512$ points | 0.932 | 0.924 |

against the exact value 1. The two error sources are separable and pull in opposite
directions. Truncation makes $f$ too *small* (equivalently $\langle1/m^*\rangle_{\rm BZ}$ too
*large*, since every omitted term is negative pointwise): at fixed mesh, adding states raises
$f$ monotonically (0.888 → 0.932 going from 12 to 85 states on the $8^3$ mesh), and the
remaining deficit of $\sim7\%$ is the missing continuum, the same physics as the $19\%$ at Γ
above but averaged over a whole band manifold rather than one state. Mesh discretization of
the average is the other error and is *not* small at $4^3$ — it is what pushes that row above
1, i.e. to the physically impossible side, and it is why $\langle1/m^*\rangle_{\rm BZ}$ comes
out $-0.103$ there instead of the required non-negative value. From $6^3$ on the sign is at
least correct ($+0.029$, $+0.069$), but the truncation deficit is mesh-independent by
construction while those two differ by $2.4\times$, so mesh error still cancels roughly half
the truncation bias at $6^3$; only at $8^3$ is the residual mostly truncation, and it is still
drifting (successive differences $-0.132$, $-0.039$, extrapolating to $\approx0.92$, a deficit
of $\approx8\%$). Treat this as a consistency check at the several-percent level, not a sharp
one.

**Synthetic pins, ahead of any Elk run** (`tests/test_parsers_optical.py`): on the massive
Dirac model $H_\tau=v(\tau k_x\sigma_x+k_y\sigma_y)+\Delta\sigma_z$, where
$\mathbf p=\partial H/\partial\mathbf k$ is exact and the two-band basis is complete, the sum
rule reproduces the analytic Hessian
$\partial^2\varepsilon_\pm/\partial k_a\partial k_b=\pm(v^2\delta_{ab}/E-v^4k_ak_b/E^3)$,
$E=\sqrt{v^2k^2+\Delta^2}$, to machine precision at every $k$ tried and for both bands
(with `free_electron_term=False`, per the argument above) — the same
eigenvalue-derivative-vs-matrix-element comparison the Si test makes, here with no truncation
to blur it. Plus: symmetry of the tensor, the two-band cancellation
$(1/m^*)_1+(1/m^*)_2=0$, the decomposition summing to the total, `nstates` truncation
matching a genuinely shorter input, the degeneracy guard, and $f^{ab}=\delta_{ab}-(1/m^*)^{ab}$
together with its closed form $f_{xx}=v^2/\Delta$ at the Dirac point.

## Guard

The sum rule above is *non-degenerate* perturbation theory. If band `ist` is within
`degeneracy_tol` (default $10^{-4}$ Ha) of any other retained state, `ValueError` is raised
rather than dividing by an arbitrary splitting: within a degenerate multiplet a single band's
curvature is not defined at all (the partners' dispersions cross), and Elk's own
finite-difference task 25 is equally meaningless there. Si's three-fold $\Gamma_{25'}$ valence
top is the canonical case, and is asserted to raise.

## How to use in code

```python
si = Structure(SI_AVEC, SI_SPECIES).get_calculation(
    "si", xc="PW", ngridk=(4, 4, 4), rgkmax=7.0,
    extra_blocks={"nempty": [40]},      # the k.p sum's slow 1/dE tail needs these
)
si.get_energy()

# one-off wrapper: momentum matrix elements at k, then the sum rule
kp = si.get_effective_mass_sum_rule((0.0, 0.0, 0.0), ist=1)
kp["inverse_mass"]     # (3,3) d^2 eps/dk_a dk_b, atomic units
kp["mass"]             # its inverse, or None if a direction is flat

# the independent route Elk already provides, for comparison
fd = si.get_effective_mass((0.0, 0.0, 0.0))
fd[0]["derivative_tensor"]   # <- compare against "inverse_mass"
fd[0]["tensor"]              # <- compare against "mass" (that matrix inverted)

# convergence sweep and the interband decomposition, from ONE run
from elkpy.parsers import optical

with si.eigenstate_session() as session:
    m = session.momentum((0.0, 0.0, 0.0))

for n in (8, 20, 40, len(m.energies)):
    optical.effective_mass_tensor(m.energies, m.pmat, 1, nstates=n)["inverse_mass"]

per_band = optical.effective_mass_tensor(
    m.energies, m.pmat, 1, decompose=True
)["contributions"]          # (nstates, 3, 3): which coupling makes the mass

optical.oscillator_strength_sum(m.energies, m.pmat, 1)   # TRK f-sum tensor
```

## 26. Spin Berry curvature and the intrinsic spin Hall conductivity

*Physics writeup: `docs/physics.tex` Part XV.*

## What was blocking this, and why the fix is one array

Everything this needs already existed, in two halves that could not legally
be multiplied together:

- the **velocity matrix elements** $v_a=p_a$ of §22's `MOMENTUM` query,
  which returned `energies` and `pmat`;
- the **spin operators** $S_x,S_y,S_z$ of §17, built in pure Python from
  `evecsv`'s spin-up/spin-down row blocks ($i=p+(\mathrm{ispn}-1)\,$`nstfv`),
  obtained from the `EIGENSTATES` query.

Those are two *separate* diagonalisations at the same $k$. §14 spells out
why that matters: a degenerate multiplet's eigenvectors are only defined up
to a unitary rotation within the multiplet, and two independent
diagonalisations are free to — and empirically do — pick different ones. A
product $S_z v_a$ formed across that boundary is meaningless, and, crucially,
**nothing cheap detects it**: $S_z$ stays Hermitian with eigenvalues in
$[-\tfrac12,\tfrac12]$, `evecsv` stays unitary, `pmat` stays Hermitian, and
the resulting curvature is a plausible finite number. The failure is silent.

Patch 0009 is therefore small on purpose: `elkpy_momentum` gains one
`intent(out)` argument, `evecsv_out(nstsv,nstsv)`, written where the routine
previously wrote a local array it then discarded; and the session's
`MOMENTUM` case prints that array in the same `do b; do a` column-major
block the `EIGENSTATES` response already uses. No new upstream subroutine is
called, no new task number, no extra computation — a `MOMENTUM` response is
now literally an `EIGENSTATES` response with the three momentum components
appended. `Momentum` gains an `evecsv` field; `EigenstateSession.momentum()`
returns it unwindowed even when `ist0`/`ist1` slice `energies`/`pmat`, since
its *row* index is the first-variational spinor basis, which a band window
has no meaning for (and truncating it would break
`compute_spin_operator()`'s `nstsv == 2*nstfv` check).

`EigenstateSession.spin_current_operator(k, direction, spin)` exists so the
pairing cannot be got wrong by accident: it issues one `MOMENTUM` query and
builds $S_s$ from *that response's own* `evecsv`.

## The physics

The **conventional spin current operator** is the symmetrized product

$$ J^s_a \;=\; \tfrac12\{S_s,v_a\} \;=\; \tfrac12\left(S_s v_a + v_a S_s\right), $$

with $S_s$ the spin operator for projection $s\in\{x,y,z\}$ and $v_a$ the
velocity operator along Cartesian axis $a$. The symmetrization is not
cosmetic: $S_s v_a$ alone is not Hermitian once $[S_s,v_a]\neq0$, which is
exactly the spin-orbit-coupled case of interest. It is also load-bearing for
the discretization below — the exact cancellation of intra-window terms
needs *both* operators Hermitian.

Its Kubo linear response to an electric field along $b$ gives the **spin
Berry curvature** of an occupied band window $W$,

$$ \Omega^{s}_{ab}(\mathbf k) \;=\; -2\,\mathrm{Im}\sum_{n\in W}\sum_{m\notin W}
   \frac{\langle n|J^s_a|m\rangle\,\langle m|v_b|n\rangle}{(\varepsilon_n-\varepsilon_m)^2}, $$

and the **intrinsic spin Hall conductivity** is its Brillouin-zone integral,

$$ \sigma^{s}_{ab} \;=\; \int_{\mathrm{BZ}}\frac{d^d k}{(2\pi)^d}\;\Omega^{s}_{ab}(\mathbf k). $$

Structurally this is §22's Kubo quantum geometric tensor with the first
velocity factor replaced by the spin current, so it reuses that module's
arithmetic verbatim (`parsers.optical.kubo_sum`, generalized in this change
from "a `pmat` plus two axis indices" to "two Hermitian operator matrices").
It therefore inherits §22's sign convention unchanged —
$\mathbf A=i\langle u|\nabla_{\mathbf k}u\rangle$, $\Omega=\nabla\times\mathbf A$
(Xiao, Chang & Niu, RMP **82**, 1959 (2010)) — and setting $S\to\mathbb 1$
recovers the ordinary charge Berry curvature exactly, which is the cheapest
available pin on that shared sign and on the absence of a stray factor in
the anticommutator.

**Why only the window sum is exposed.** The textbook per-band $\Omega^s_n$
sums over *all* $m\neq n$, including states inside $W$. For the
window-summed quantity those intra-window terms cancel **exactly**, for any
Hermitian $J$ and $v$: writing $X=\langle n|J|m\rangle$ and
$Y=\langle m|v|n\rangle$, the reversed pair contributes
$\langle m|J|n\rangle\langle n|v|m\rangle=X^*Y^*=(XY)^*$ over the identical
denominator, so the two imaginary parts cancel. Hence
$\sum_{n\in W}\sum_{m\neq n} = \sum_{n\in W}\sum_{m\notin W}$, and the sum
never touches an intra-window energy denominator — whereas a per-band
$\Omega^s_n$ diverges on any degenerate occupied pair. The remaining
degeneracy guard is §22's unchanged: a window not separated from the states
outside it raises `ValueError` rather than being clipped.

**The $\tfrac12$, stated once because it is exactly a factor of 2.** elkpy's
$S_a$ (§17) has eigenvalues $\pm\tfrac12$, i.e. $\hbar=1$ and
$S=\vec\sigma/2$. So for a system with conserved $S_z$, where the bands
decouple into two sectors and $J^z_a=s_\sigma v_a$ within each,

$$ \Omega^{s} \;=\; \sum_\sigma s_\sigma\,\Omega^{\sigma} \;=\; \tfrac12\left(\Omega^{\uparrow}-\Omega^{\downarrow}\right), $$

*half* the bare difference. Papers quoting SHC in units of $\hbar/2e$ work
with the $\pm1$-normalized spin, so multiply by 2 to compare.

**Caveat, and it is a real one.** $\tfrac12\{S_z,v\}$ is the *conventional*
spin current, which is not conserved once spin-orbit coupling breaks spin
conservation; a "proper" definition adds a torque-dipole term. elkpy
computes the conventional one — what essentially all first-principles SHC
numbers in the literature use — and names it for what it is rather than
quietly implying conservation.

## Verification

**Synthetic** (`tests/test_parsers_spin_hall.py`, no Elk run), on §22's
massive Dirac model $H_\tau=v(\tau k_x\sigma_x+k_y\sigma_y)+\Delta\sigma_z$
where $\mathbf p=\partial H/\partial\mathbf k$ is exact. Two Dirac sectors
are glued into one four-state system with conserved $S_z$ (states sorted by
energy, `pmat` and $S_z$ permuted together so the window really is "both
valence bands"), and the result is checked against
`parsers.optical.kubo_berry_curvature` — an independent, already
sign-pinned code path — on each sector alone:

- $\Omega^s=\sum_\sigma s_\sigma\Omega^\sigma$ exactly (to $10^{-10}$), over
  all four valley combinations. Two of those combinations are the ones with
  teeth, and neither alone suffices:
  - **the time-reversal pair** ($\tau_\uparrow=+1$, $\tau_\downarrow=-1$):
    charge curvature cancels to zero while the spin curvature is maximal —
    a $J$-vs-$v$ swap would return 0 here instead of a large number;
  - **two copies of one valley**: charge curvature doubles while the spin
    curvature cancels — the opposite pattern, catching a dropped or
    mis-signed $S$ factor that the first case's coincidence hides.
- $S\to\mathbb 1$ reproduces the ordinary curvature exactly.
- $J$ is Hermitian for non-commuting random Hermitian $S$ and $v$ (with the
  unsymmetrized product asserted *not* Hermitian, so the test isn't vacuous).
- the window sum equals an explicit brute-force sum over all $m\neq n$ on a
  random Hermitian model — a direct check of the cancellation argument
  above, not a restatement of it.

**Against a real compiled binary** (`tests/test_calculation_spin_hall.py`),
on the graphene + `soc_scale={"C": 3000}` fixture §20 already uses. One
practical note first, because it bit this test: band windows here are
**explicit and gap-checked**, not read off occupation numbers. Scaling
carbon's SOC by 3000 reorders the band structure so drastically that the
cell's 8 valence electrons (confirmed in `INFO.OUT`) do not fill a fixed 8
states at every $k$ — Elk's own occupancies show 6 occupied at the first
$k$-point — so `sum(occ > 0.5)` at one $k$-point, the idiom §13/§22's tests
use on genuine insulators, does not define a band group here. Every
quantity in this section is a property of a *gapped group's projector*, so
the tests assert the boundary gap of each window they use. (This is worth
carrying over to §20's own use of the same fixture.)

Measured, at both valleys, for the two gapped groups used:

| window | boundary gap | $\Omega^{\text{charge}}$ | $\Omega^{s}(K)$ | $\Omega^{s}(K')$ |
|---|---|---|---|---|
| [1,4] | 1.25 eV | $\sim2\times10^{-4}$ | $-78.4035$ | $-78.4016$ |
| [1,6] | 10.4 eV | $<10^{-5}$ | $-0.31936$ | $-0.31936$ |

The $K/K'$ agreement is to $\sim2\times10^{-5}$ relative, across windows
whose spin curvatures differ by more than two orders of magnitude, while
the charge curvature is numerical noise about zero in both. The absolute
sign of $\Omega^s$ is a regression pin, not a prediction — it depends on
this structure's own conventions and on which `evecsv` row block is
physically "up", exactly as §17 documents for $S_z(K)$.

Specifically:

- the `MOMENTUM` response's new `evecsv` is unitary to machine precision
  (not the $\sim10^{-3}$ `genolpq` truncation floor overlaps carry — these
  are eigenvectors of one Hermitian problem);
- $S_z$ built from it has the same *spectrum* over the window as
  $S_z$ from an `EIGENSTATES` query at the same $k$, and the two responses'
  energies agree exactly. The comparison is of the spectrum, not of matrix
  elements: graphene is Kramers degenerate at *every* $k$ (inversion +
  time reversal), so an elementwise comparison would fail with nothing
  wrong — which is precisely the ambiguity patch 0009 exists to remove;
- the physics check: graphene's inversion **and** time-reversal symmetry
  together force the charge Berry curvature to vanish pointwise
  ($\Omega(-\mathbf k)=+\Omega(\mathbf k)$ from inversion,
  $-\Omega(\mathbf k)$ from time reversal), while the spin Berry curvature
  is under no such constraint. $\Omega^s$ is **even** under each symmetry
  separately: $J^z_a$ picks up sign flips from *both* $S_z$ and $v$ where
  $v$ alone picks up one, so the two flips cancel. Hence
  $\Omega^s(K)=+\Omega^s(K')$ — the *opposite* relative sign to the $K/K'$
  antisymmetry §13's Berry curvature, §17's $S_z$, §19's $L_z$ and §22's
  circular dichroism all assert, and the reason a spin Hall response is
  allowed in a time-reversal-symmetric crystal where an anomalous Hall
  response is not. A large, valley-**symmetric** spin curvature sitting on
  a vanishing, valley-antisymmetric charge one is the defining signature of
  a quantum spin Hall system. Note what the test would show if the two
  operator factors were swapped or $S$ silently dropped: the charge
  curvature, i.e. zero — so the *magnitude* is as diagnostic as the sign
  pattern here.

Not asserted: a converged $\sigma^s_{xy}$ against a literature value. The
spin Berry curvature is sharply peaked near the gapped Dirac points, so the
BZ integral converges far more slowly than a total energy on the same mesh —
the same resolution problem §20 documents for the $Z_2$ mesh. The
conductivity helper is exercised end-to-end on a coarse mesh for shape and
units only, and says so.

## How to use in code

```python
from elkpy.parsers import optical, spin_hall
from elkpy.parsers.spin import compute_spin_operator

calc = Structure(GRAPHENE_AVEC, GRAPHENE_SPECIES).get_calculation(
    "graphene", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
    spinorb=True, soc_scale={"C": 3000.0},
    extra_blocks={"nempty": [12]},   # states for the Kubo sums
)
calc.get_energy()

ist0, ist1 = 1, 4   # a gapped band group -- CHECK the boundary gap, see above

with calc.eigenstate_session() as session:
    # one MOMENTUM query: energies, pmat AND evecsv from ONE diagonalisation
    m = session.momentum((1 / 3, 1 / 3, 0))
    nstsv = m.evecsv.shape[0]
    sz = compute_spin_operator(m.evecsv, nstsv // 2, 1, nstsv)["sz"]

    # spin Berry curvature of the occupied window (Bohr^2), sigma^{s_z}_xy
    spin_hall.spin_berry_curvature(m.energies, m.pmat, sz, ist0, ist1,
                                   directions=(1, 2))

    # the charge curvature of the same window, for comparison -- zero here,
    # forced by graphene's inversion x time-reversal symmetry
    optical.kubo_berry_curvature(m.energies, m.pmat, ist0, ist1)

    # or let the session pair S_z and v for you (same single query):
    m, j = session.spin_current_operator((1 / 3, 1 / 3, 0), direction=1, spin="z")

# fold a mesh of curvatures into the conductivity (atomic units)
sigma = spin_hall.spin_hall_conductivity(curvatures, cell_volume=volume)
```

`directions` here indexes **Cartesian** axes ($1,2,3=x,y,z$), the same as
`parsers.optical` and for the same reason (`genpmatk`'s components) — not
the reciprocal-lattice convention of `get_berry_curvature()`.

## References

- Guo, Yao & Niu, *Ab initio calculation of the intrinsic spin Hall effect
  in semiconductors*, PRL **94**, 226601 (2005), arXiv:cond-mat/0505146 —
  the ancestor Kubo formula both papers below derive from, and the one whose
  $\omega\to0$ limit fixes the $e\hbar/V_c$ prefactor used here.
- Yao & Fang, *Sign Changes of Intrinsic Spin Hall Effect in Semiconductors
  and Simple Metals*, PRL **95**, 156601 (2005), arXiv:cond-mat/0502351 —
  prints the curvature with $-2\,\mathrm{Im}$, the form elkpy's own
  convention coincides with, and states the sign in words.
- Guo, Murakami, Chen & **Nagaosa**, *Intrinsic spin Hall effect in platinum
  metal*, PRL **100**, 096401 (2008), arXiv:0705.0409 — prints the same
  expression with $+2\,\mathrm{Im}$; both papers nevertheless report a
  positive SHC of comparable size for hole-doped GaAs, so the printed
  difference is absorbed in the charge vertex, not physical. (Note the
  fourth author is Nagaosa, not Niu — an easy misattribution given the
  ancestor paper above.)
- Shi, Zhang, Xiao & Niu, *Proper definition of spin current in spin-orbit
  coupled systems*, PRL **96**, 076604 (2006), arXiv:cond-mat/0503505 — the
  conserved-spin-current objection to $\tfrac12\{S_z,v\}$, not implemented
  here.
- Kane & Mele, PRL **95**, 226801 (2005), arXiv:cond-mat/0411737 (the
  current-level reduction $\mathbf J_s=(\hbar/2e)(\mathbf J_\uparrow-\mathbf
  J_\downarrow)$) and PRL **95**, 146802 (2005), arXiv:cond-mat/0506581 (the
  invariant-level one, already cited by §20) — the quantum spin Hall effect
  in graphene, the fixture's own prediction.
- Sinova, Valenzuela, Wunderlich, Back & Jungwirth, *Spin Hall effects*, RMP
  **87**, 1213 (2015), arXiv:1411.3249 — review covering both the Kubo
  conventions and the proper-current caveat.
- Xiao, Chang & Niu, RMP **82**, 1959 (2010) — the Berry-phase sign
  convention elkpy uses throughout (§22).

## 27. Named wrappers for Elk's potential, ELF and MOKE tasks

Routine wrapping of upstream capability rather than new physics, so this is
recorded compactly; the MOKE verification is the part worth reading.

Doc prose for merging into `docs/design.md` (a new numbered section, or an
extension of the volumetric/plot3d discussion), `docs/physics.tex` (the MOKE
part below is written as a `\part{}` draft), `README.md`'s `FUNCTIONALITIES`
list and `CLAUDE.md`'s project status. Written by the agent that added
`get_potential()`, `get_elf()` and `get_moke()`; not part of the published
docs.

**Still outstanding when merging:** a notebook per new capability (README
`FUNCTIONALITIES` bullets link to one), per the README/notebook style rules.
A single "volumetric quantities" notebook (density, Kohn-Sham potential,
ELF, plotted as 2D slices through the Si bond) plus a MOKE spectrum notebook
(Kerr rotation and ellipticity vs photon energy for the two magnetization
directions) would cover all three.

Also update `docs/roadmap.md` Tier 3 item 4, which currently says potential
(43) and ELF (53) "are one `_run_resumed` call away using the same parser but
don't have named `get_*` methods yet" -- they now do. And `CLAUDE.md`'s
"Not implemented" list, which mentions "named `get_*` methods for
potential/ELF volumetric plots (reachable via `run_tasks()` +
`parsers.volumetric`)".

## 1. Kohn-Sham potential and ELF as 3D plots (tasks 43, 53)

Tasks 33 (density, already wrapped as `get_density()`), 43 (potential) and
53 (ELF) all end in the same writer, `vendor/elk/src/plot3d.f90`, and take
the same `plot3d` input block -- a parallelepiped given as an origin plus
three corner vectors in lattice coordinates, and a grid size. So the
Python side needs no new parser at all: `parsers/volumetric.py::parse_plot3d`
already reads the format, and the three methods now share a
`Calculation._plot3d_lines()` helper that builds the block.

Task numbers and filenames verified in `vendor/elk/src/elk.f90`'s dispatch
(`case(31,32,33) -> rhoplot`, `case(41,42,43) -> potplot`,
`case(51,52,53) -> elfplot`) and in the `open(50, file=...)` statements of
the subroutines themselves, not from the manual.

### Potential (task 43)

`potplot.f90`'s `case(43)` branch writes **two** files from a single run:

* `VCL3D.OUT` -- the electrostatic (Coulomb) potential `v_C`, i.e. the
  nuclear plus Hartree terms, from `vclmt`/`vclir`.
* `VXC3D.OUT` -- the exchange-correlation potential
  `v_xc = delta E_xc / delta n`, from `vxcmt`/`vxcir`.

Their sum is the Kohn-Sham effective potential entering
`(-1/2 grad^2 + v_C + v_xc) psi_i = eps_i psi_i`. This is the one place the
new methods do not map one-to-one onto `get_density()`'s shape: a single
task writes two distinct fields. `get_potential(..., component=)` selects
which of the two is parsed ("coulomb" by default, or "xc") and keeps
`get_density()`'s `(points, values)` return contract; asking for the other
component re-runs the task, which is cheap (no SCF beyond the resumed
ground state, just `readstate` + `plot3d`) and was judged preferable to a
method whose return arity changes with an argument.

### ELF (task 53)

`elfplot.f90` writes `ELF3D.OUT`. The (spin-averaged) electron localization
function is

    f_ELF(r) = 1 / (1 + [D(r)/D0(r)]^2)

with

    D(r)  = (1/2) ( tau(r) - (1/4)|grad n(r)|^2 / n(r) )
    D0(r) = (3/5) (6 pi^2)^(2/3) (n(r)/2)^(5/3)
    tau(r) = sum_i |grad psi_i(r)|^2

(the docstring of `elfplot.f90` itself; Becke and Edgecombe, J. Chem. Phys.
92, 5397 (1990); the reference Elk cites is Burnus, Marques and Gross, PRA
71, 010501 (2005)). `D` is the excess of the local kinetic energy density
over its von Weizsaecker (single-orbital) value, i.e. the Pauli-principle
contribution; `D0` is the same quantity for the homogeneous electron gas at
the local density. `f_ELF` is therefore dimensionless and bounded to [0, 1]
by construction: 1 means an electron pair is perfectly localized (a
covalent bond, a lone pair, a closed shell), 1/2 reproduces the homogeneous
electron gas, and 0 marks the delocalized limit.

Caveat worth repeating from Elk's own example (`examples/ELF/BN`): the ELF
depends on density gradients and is not continuous across the muffin-tin
boundaries at default cut-offs, so a plot can show a visible sphere-boundary
seam that is a basis-set artefact rather than physics. Raising `rgkmax`,
`gmaxvr`, `lmaxo`, `lmaxapw` (via `extra_blocks`) or `highq=.true.` smooths it.

### How to use in code

```python
from elkpy.structure import Structure

si = Structure([(5.13, 5.13, 0.0), (5.13, 0.0, 5.13), (0.0, 5.13, 5.13)],
               {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]})
calc = si.get_calculation("si", xc="PW", ngridk=(4, 4, 4))

# Cartesian grid points (Bohr) and the field sampled on them
points, rho = calc.get_density(grid=(20, 20, 20))
points, v_c = calc.get_potential(grid=(20, 20, 20))                  # VCL3D.OUT
points, v_xc = calc.get_potential(grid=(20, 20, 20), component="xc") # VXC3D.OUT
points, elf = calc.get_elf(grid=(20, 20, 20))

# a plane through the Si-Si bond instead of the whole cell: origin + two
# in-plane vectors + a degenerate third, in lattice coordinates
box = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)]
points, elf_slice = calc.get_elf(box=box, grid=(60, 60, 1))
```

### What was verified against a real binary

Bulk Si, `ngridk=(2,2,2)`, 4x4x4 plot grid (`tests/test_calculation_si.py`):

* `get_potential()` returns both components on the same grid as
  `get_density()`, and they are genuinely different fields (catching a
  filename/spec mix-up that returned the same file twice).
* **Sharp cross-check, no Elk-internal reference needed:** `xc="PW"` is a
  *local* density functional, so `v_xc(r)` must be a pointwise function of
  `n(r)` alone. Measured against the analytic Dirac exchange potential
  `v_x = -(3n/pi)^(1/3)` on the identical grid, `v_xc/v_x` lies in
  1.14-1.23 across this cell's density range (0.006-0.085 e/Bohr^3) -- the
  extra 14-23% being the Perdew-Wang correlation potential. This ties two
  separate Elk runs' output files (task 33's `RHO3D.OUT`, task 43's
  `VXC3D.OUT`) to an analytic formula; a shifted column, a wrong file or a
  unit error breaks it immediately, whereas a shape/sign check would not.
* ELF lies in [0, 1] everywhere (a hard bound of the formula, not a
  tolerance), and reaches 0.94 in the Si-Si bonding region -- a covalent
  crystal must depart strongly from the homogeneous-gas value of 1/2
  somewhere in the cell.

Note for anyone reading absolute values near a nucleus: the plot3d output
is an interpolation onto the requested Cartesian grid, and a grid point
landing exactly on an atom does *not* show the divergent `-Z/r` Coulomb
potential or the enormous core density (measured on the Si cell: `RHO3D`
peaks at 0.085 e/Bohr^3, in the bond, not at the nucleus). Treat these
plots as valence/bonding-scale visualizations.

## 2. Magneto-optic Kerr effect (task 122)

`get_moke()` wraps task 122 (`vendor/elk/src/moke.f90`, dispatched at
`case(122)` in `elk.f90`), returning `(energies, kerr)` -- the photon-energy
grid in Hartree and the **complex Kerr angle in degrees**, whose real part
is the Kerr rotation `theta_K` and whose imaginary part is the Kerr
ellipticity `eta_K`. Output file `KERR.OUT`, format read off `moke.f90`'s
own write statements: two blank-line-separated blocks of `(2G18.10)` pairs,
real part then imaginary part, on a shared energy grid -- parsed by the new
`parsers/moke.py`.

### The physics

Linearly polarized light reflected from a magnetized surface comes back
elliptically polarized, with its major axis rotated. The effect is
first-order in the magnetization and vanishes without spin-orbit coupling:
it needs the *off-diagonal* conductivity `sigma_xy`, which is odd under time
reversal, and only SOC ties the (time-reversal-odd) spin magnetization to
the orbital motion the light actually couples to.

`moke.f90` calls `dielectric` internally with `optcomp` fixed to the 11 and
12 components, obtaining `sigma_xx` and `sigma_xy` from the Kubo formula
implemented in `dielectric.f90` (Physica Scripta T109, 170 (2004)), and then
forms the standard polar-Kerr expression

    theta_K + i eta_K = - sigma_xy / ( sigma_xx sqrt(1 + 4 pi i sigma_xx / omega) )

(atomic units; `moke.f90` writes both parts multiplied by 180/pi, i.e. in
degrees). The square-root factor is the refractive-index denominator of the
Fresnel reflection coefficients for the two circular polarizations. Elk
returns exactly zero at `omega = 0`, where the expression is singular.

Task 122 needs the momentum matrix elements on disk (`dielectric` reads
`PMAT.OUT` via `getpmat`), so `get_moke()` runs task 120
(`writepmat.f90`) first in the same directory. That pairing is precisely
what makes this worth a named method rather than a bare `run_tasks()` call.

Note the interaction with `dielectric` (task 121, wrapped separately): task
122 *also* writes `SIGMA_11.OUT`, `SIGMA_12.OUT`, `EPSILON_11.OUT` and
`EPSILON_12.OUT` as a side effect, since it calls the same subroutine. Both
wrappers run in their own wiped `_run_resumed()` subdirectory, so they
cannot overwrite each other's output; but a caller who wants both the
conductivity tensor and the Kerr angle pays for the response function twice.

### Smearing and convergence

`swidth` (Hartree) sets the smearing whose reciprocal is the relaxation time
entering the response function. Elk's own example
(`examples/TDDFT-optics/Ni-MOKE`) raises it *after* the ground-state run to
smooth the spectrum, warning that a large smearing during the SCF cycle
suppresses the moment -- which is exactly what passing `swidth=` to
`get_moke()` does, since it runs resumed from an already-converged
`STATE.OUT` and never feeds `swidth` back into the ground state.

The Kerr angle is a Brillouin-zone integral over interband transitions and
converges slowly in `ngridk` (Elk's Ni example uses 32x32x32). At a coarse
mesh the spectrum is dominated by individual transition spikes and the
near-singular `sigma_xx` denominator, which inflates the peak values by
orders of magnitude: measured on fcc Ni, `ngridk=(4,4,4)` with default
smearing gives peaks of tens of degrees, while `ngridk=(10,10,10)` with
`swidth=0.01` brings them down to the ~0.01-0.1 degree scale of a real Ni
Kerr spectrum. Treat a coarse-mesh result as a symmetry/sign probe, not a
spectrum.

### How to use in code

```python
from elkpy.structure import Structure

# vendor/elk/examples/TDDFT-optics/Ni-MOKE: ferromagnetic fcc Ni
ni = Structure([(1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)],
               {"Ni": [(0.0, 0.0, 0.0)]}, scale=3.33)
calc = ni.get_calculation(
    "ni", xc="PW", spinpol=True, spinorb=True, ngridk=(8, 8, 8),
    extra_blocks={"bfieldc": [(0.0, 0.0, 0.01)]},  # magnetize along +z
)

# energies in Hartree, kerr complex in degrees
energies, kerr = calc.get_moke(wplot=(0.0, 0.5), nwplot=500,
                               swidth=0.01, ngridk=(16, 16, 16))
theta_K, eta_K = kerr.real, kerr.imag
```

`get_moke()` raises `ValueError` before launching anything if
`spinorb=False` or `spinpol=False`: `sigma_xy` is then identically zero by
symmetry, so the run could only ever return a flat zero.

### What was verified against a real binary

`tests/test_calculation_moke.py`, fcc Ni (Elk's own MOKE example structure)
at `ngridk=(4,4,4)`, ~50 s for the whole file:

* The guards fire without running Elk (no SOC / no magnetization).
* Shapes/dtype, the energy grid (starts at 0 -- `dielectric.f90` clips a
  negative `wplot(1)` to zero), and `kerr[0] == 0` exactly.
* **Sign of the effect:** the Kerr angle is odd under reversal of the
  magnetization. Two *independent* ground states (`bfieldc` along +z and
  -z), each with its own momentum matrix elements and conductivity tensor,
  sharing no arithmetic, give spectra that agree in magnitude to 0.6% and
  cancel to 0.9% of the peak when added:
  `max|kerr(+M) + kerr(-M)| = 0.0088 max|kerr(+M)|`. This is the check that
  distinguishes a genuine Kerr response from anything even under time
  reversal (e.g. `|sigma_xy|`), which a magnitude-only assertion would pass
  just as happily.

One convention rests on a source read rather than a runtime test: which of
`KERR.OUT`'s two blocks is the real part. `moke.f90` writes `dble(kerr)`
first and `aimag(kerr)` second, and `parsers/moke.py` follows that order --
but every check above (oddness under M reversal, `kerr[0] == 0`, magnitude
agreement) is invariant under swapping the two, so none of them would catch
a flip. Same class of source-derived convention as the Fortran conjugation
sign in the Berry-curvature work. A future denser-mesh comparison against a
published Ni Kerr spectrum would pin it empirically.

Not verified: the absolute spectrum against experiment or published DFT --
that needs a k-mesh well beyond what these tests can afford, and the coarse
mesh inflates peak magnitudes as described above.

## 28. Rotation-eigenvalue symmetry indicators

*Physics writeup: not yet written — pre-existing gap. (This section originally
promised `docs/physics.tex` Part XVI, but no such part was ever added; Part XVI
is now §29's exchange writeup.)*

The general symmetry operator at a fixed $k$-point, and the
Benalcazar-Li-Hughes corner-charge indices built on it — the same
"skip the intermediate model" idea as §13/§20/§23, applied to the pipeline
DFT → `irvsp`/`vasp2trace` → Bilbao tables.

**The primitive** (`patches/0010`, `elkpy_symop`, session queries `SYMLIST`
and `SYMMETRY`): $S_{mn}=\langle\psi_m|\hat O_{\rm isym}|\psi_n\rangle$ for
any space-group element that fixes $\mathbf k$, generalizing §23's inversion.
`SYMLIST` exposes Elk's own space group (integer lattice rotations, so
$R\mathbf k\equiv\mathbf k$ is an exact integer test, not a Cartesian
tolerance). Two restrictions are enforced in Fortran, not merely documented,
because violating either returns a plausible matrix that unitarity and
$\hat O^n=\mathbb 1$ would both pass:

- **`nspinor=1` only.** A spatial rotation on a spinor also needs the SU(2)
  spin rotation, which upstream's first-variational transformation never
  applies. Inversion escaped this by acting trivially on spin; a rotation
  does not. BLH's indices are spinless anyway.
- **zero translation only.** For a glide or screw, $\hat O^n$ is a
  $k$-dependent phase times unity, so a caller binning eigenvalues into
  $n$-th roots of unity would silently misread it.

**The indices** (Benalcazar, Li & Hughes, PRB 99, 245151 (2019),
arXiv:1809.02142, Eqs. 3, 4, 14; general framework: Po, Vishwanath &
Watanabe, Nat. Commun. 8, 50 (2017), arXiv:1703.00911). With
$\Pi_p^{(n)}=e^{2\pi i(p-1)/n}$ and $\#\Pi_p^{(n)}$ the number of occupied
bands carrying that eigenvalue,

$$[\Pi_p^{(n)}]=\#\Pi_p^{(n)}-\#\Gamma_p^{(n)},\qquad Q^{(3)}_{\rm corner}=\tfrac e3[K_2^{(3)}]\ \mathrm{mod}\ e$$

with the C₂/C₄/C₆ index sets and charge formulas in
`parsers/indicators.py`. A nonzero $Q$ signals an obstructed atomic limit:
occupied Wannier centers sitting at a Wyckoff position other than the
rotation centre.

### Two practical findings, both load-bearing

**`tshift=False` is mandatory.** Elk relocates the origin by default, and
when a crystal has inversion it puts the *inversion centre* there. For a
honeycomb that is the bond midpoint — a different point from the C₃ axis —
so every rotation acquires a fractional translation and becomes
non-symmorphic in Elk's setting, at which point `elkpy_symop` refuses them
all. Measured directly on graphene: with the default shift, 8 of 24
operations are symmorphic and no C₃ survives; with
`extra_blocks={"tshift": [False]}` all 24 are symmorphic and C₃ is
available. Since $Q$ is defined *relative to a chosen rotation centre*, this
is not merely a technicality — it is how the centre gets chosen.

**Graphene cannot be used for the K-point indices.** Without spin-orbit
coupling its valence and conduction bands touch at K (gap measured
$4.3\times10^{-7}$ Ha — the Dirac point), so the occupied manifold is not a
gapped group there and the indices are undefined.
`parsers.symmetry.check_window_gap` refuses it, correctly. h-BN, gapped at K
by the B/N sublattice asymmetry, is the usable C₃ test case.

### Status: the operator is verified, the corner charge is not

Verified against a real compiled binary on h-BN (`tshift=False`, hexagon
centre at the origin, 4 occupied bands): all 12 crystal symmetries
symmorphic; the C₃ operator unitary and $\hat O^3=\mathbb 1$ to
$\sim10^{-3}$ (the `genolpq` truncation floor of §14); eigenvalue counts
summing to the band count at every point; and $[\Gamma_p]=0$ by
construction. Measured counts: $\Gamma\,[2,1,1]$, $K\,[1,1,2]$,
$K'\,[1,2,1]$.

**What is not settled is the convention, and therefore the charge.** K and
K′ are time-reversal partners, so their rotation eigenvalues are complex
conjugates — which exchanges the $p=2$ and $p=3$ bins. Since
$Q^{(3)}=\frac e3[K_2^{(3)}]$ reads only one bin, the answer depends on how
the generator ($R$ versus $R^{-1}$) is paired with the corner ($K$ versus
$K'$): the same data gives $Q=0$ at K and $Q=e/3$ at K′.

This is the direction-convention ambiguity §23 was able to defer, because
inversion is its own inverse. Two natural discriminators were tried and
**both fail**:

- $D(9)^2=D(3)$ holds under *either* convention, since $C_3$ is abelian and
  inversion-of-argument is then still a homomorphism.
- The $C_{2z}=\hat I\cdot\sigma_h$ identity against §23's parity cannot help,
  because order-2 operations are their own inverse — precisely why the
  ambiguity did not arise before.

So no corner charge is asserted here, and no test asserts one. The physics
expectation for h-BN is an obstructed atomic limit (the occupied $\pi$ band
is N-centred, at a Wyckoff position away from the hexagon centre), which
favours $e/3$ over $0$ — but "the expected answer is nonzero, and one of our
two candidate conventions gives a nonzero number" is not evidence, and this
project has been bitten by exactly that reasoning before (§21's cesium).

**The concrete way to settle it**, not done here: use §19's $L_z$ operator.
A state's $C_n$ eigenvalue is $e^{-2\pi i m/n}$ with $m$ its angular momentum
about the axis, so an independently measured $\langle L_z\rangle$ fixes the
sense of rotation and hence the generator. That is cleanest for an atom
sitting *on* the rotation axis, so it needs a test structure with one —
h-BN has none. Alternatively, a system with an independently known nonzero
corner charge would pin the pairing directly.

### How to use in code

```python
calc = Structure(AVEC, SPECIES).get_calculation(
    "run", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
    extra_blocks={"tshift": [False]},   # keep YOUR origin: it is the rotation centre
)
calc.get_energy()

from elkpy.parsers import indicators, symmetry

with calc.eigenstate_session() as session:
    ops = session.symmetries()                    # no diagonalisation
    c3 = indicators.find_rotation(ops, 3)
    counts = {}
    for name, k in [("Gamma", (0, 0, 0)), ("K", (1/3, 1/3, 0))]:
        assert indicators.fixes_kpoint(c3, k)
        r = session.symmetry_operator(k, c3["isym"], ist0, ist1)
        symmetry.check_window_gap(r.energies, ist0, ist1)
        counts[name] = indicators.eigenvalue_counts(r.smat, 3)

chi = indicators.relative_indices(counts["K"], counts["Gamma"])
indicators.corner_charge({"K": chi}, 3)   # read the convention caveat above
```

## 29. Anisotropic exchange constants by four-state energy mapping

*Physics writeup: `docs/physics.tex` Part XVI.*

The full 3×3 exchange tensor $J_{ij}^{\alpha\beta}$ of a magnetic pair —
isotropic Heisenberg exchange, symmetric anisotropy (the Kitaev $K$ and
$\Gamma$ terms), and the antisymmetric Dzyaloshinskii-Moriya vector — from
the total energies of constrained non-collinear DFT states. Elk itself has
no exchange-parameter capability at all (nothing in `docs/elk_manual.txt`,
no task), so this is new physics on top of the interface rather than
wrapping.

**Method**: Šabani, Bacaksiz & Milošević, PRB 102, 014457 (2020),
arXiv:2002.10861 — the generic four-state method, which generalizes Xiang,
Kan, Wei, Whangbo & Gong, PRB 84, 224429 (2011), arXiv:1106.5549 from an
isotropic/DM Hamiltonian to a general tensor. Convention here is theirs:

$$H=\sum_{i<j}\mathbf S_i\cdot\mathbf J_{ij}\cdot\mathbf S_j+\sum_i\mathbf S_i\cdot\mathbf A_{ii}\cdot\mathbf S_i$$

with classical spin vectors of magnitude $|\mathbf S_i|=S$, each pair counted
once, and a plus sign — so $J<0$ is ferromagnetic. Every one of the nine
components comes from the same formula,

$$J_{ij}^{\alpha\beta}=\frac{E_1+E_4-E_2-E_3}{4S^2m_{ij}}$$

where the four states put $\mathbf S_i$ along $\pm\hat\alpha$ and
$\mathbf S_j$ along $\pm\hat\beta$ in the sign combinations
$(+{+},+{-},-{+},-{-})$, and **every other magnetic atom points along the
third axis, identically in all four states**.

### Why the other terms cancel

Only the target term carries the sign pattern $(+,-,-,+)$ that
$E_1+E_4-E_2-E_3$ selects. Cross terms between one member of the pair and a
spectator are linear in a single flipped spin, so they carry $(+,+,-,-)$ or
$(+,-,+,-)$ and are annihilated. Spectator-spectator terms never move at all
and appear identically in all four energies. And single-ion anisotropy is
*quadratic* in spin, hence invariant under $\mathbf S\to-\mathbf S$, so it
too appears identically — which is why the exchange formula needs no
knowledge of the SIA, and equally why the SIA needs its own configurations
and its own formulas (`single_ion_offdiagonal`, `single_ion_difference`;
note the different sign pattern *and* the different denominator $2S^2$).
Only *differences* of the diagonal SIA are extractable: with a classical spin
of fixed length, $(S^x)^2+(S^y)^2+(S^z)^2=S^2$ makes any common part of the
diagonal an additive constant, invisible to every energy difference.

`tests/test_parsers_exchange.py` tests this numerically rather than trusting
it — a model with a known tensor on the target pair *plus* single-ion
anisotropy on every site, couplings from the pair to the spectators, and
couplings among the spectators returns the target tensor exactly.

### Two published traps

**Xiang's DM formula carries an extra sign** that is correct only for a
Hamiltonian already assumed antisymmetric, not for a general tensor. Šabani
et al. document this and show it manufactures a spurious DM vector on
monolayer CrI₃, where an inversion centre at the bond midpoint forbids one.
Here nothing is special-cased: all nine components use the identical formula
and $\mathbf D$ is extracted afterwards from the antisymmetric part,
$D_x=\tfrac12(J^{yz}-J^{zy})$ and cyclic (the same convention KKR uses,
Mankovsky & Ebert arXiv:2206.09969 Eq. 55). This makes **$\mathbf D=0$ on an
inversion-symmetric bond a genuine null test** of the whole pipeline rather
than something built in.

**The multiplicity $m_{ij}$** — how many bonds the pair actually contributes
in the supercell, *including periodic images* — is implicitly 1 in
Xiang/Šabani and is only 1 if the supercell is large enough. Flipping site
$j$ flips all of its images, so the energy difference measures
$\sum_{\mathbf T}J(i,j+\mathbf T)$. `parsers.exchange.bond_multiplicity`
computes it, and `check_supercell` raises when an image sits inside the
retained interaction range at a *different* distance — at which point the
measured difference mixes distinct neighbour shells and no choice of
multiplicity can unmix them (criterion from arXiv:2512.08471).

### Why this needs no new Fortran

Elk already constrains per-atom moment directions natively. `fsmtype=-2`
plus a per-atom `mommtfix` block adds a Lagrange-style field updated each SCF
cycle (`vendor/elk/src/bfieldfsm.f90`), and for the *negative* variant that
field is projected perpendicular to the target direction (`r3vo`) — so the
magnitude relaxes freely and, once the moment is on target, the field does no
work on it. Measured directly: the constraining field's component along each
target comes back as ~10⁻¹⁸.

More importantly, **the reported total energy is the constrained-DFT energy
functional with no constraining-field contribution**. `energy.f90:226` forms
the kinetic energy as `engykn = evalsum - engyvcl - engyvxc - sm`, where
`sm = ∫ m·B_s` is built from the *total* effective field `bsmt` — into which
`addbfsm.f90` has already folded the constraining field (and `bfcmt`). Its
work is therefore removed exactly, for any `fsmtype`. This is what makes
energy mapping legitimate here, and it is a property of the source rather
than an assumption; it also means `fsmtype=+2` (fixing the full moment
vector, magnitude included) is an energetically clean fallback if direction-
only constraint proves unstable.

### Three settings that are load-bearing, not cosmetic

- **`nosym` / `reducek=0`.** The four configurations deliberately break the
  crystal symmetry, and `checkfsm.f90` hard-`stop`s if `mommtfix` is not
  invariant under the symmetry group `findsym.f90` found — which inspects
  `bfcmt0` but *not* `mommtfix`. Beyond dodging that stop, an identical k-set
  across all four states is what lets their systematic errors cancel in the
  difference; symmetry-reduced meshes would differ between configurations and
  leave a residue in exactly the quantity being extracted.
- **`epsengy`.** Elk's default is 1e-4 Ha ≈ 2.7 meV, larger than the entire
  signal. The formula sums four total energies, so the noise floor is about
  four times the per-run convergence. The literature standard is 1e-5 eV
  (Šabani et al.; Hou et al. drive their constraint penalty below the same);
  elkpy defaults to 1e-8 Ha.
- **`bfcmt` seeding.** The constraining field is perpendicular to the target,
  so it can rotate an existing moment but cannot create one — a configuration
  started from zero magnetisation has nothing to constrain. Seeding `bfcmt`
  along each target direction both starts the SCF near the intended state and
  keeps Elk's own symmetry analysis consistent with it (`findsym.f90` reads
  `bfcmt0`). Elk's sign convention was pinned empirically rather than assumed:
  a single bcc Fe atom with `bfcmt = (0,0,+4)` converges to a moment of
  **−2.75 μB**, i.e. the moment is *antiparallel* to `bfcmt`, so
  `seeded_structure()` applies the seed with a minus sign.

### The check that the method actually rests on

`fsmtype=-2` constrains direction only, so the magnitude is free to collapse
— and a collapsed moment is held by nothing. A configuration that silently
drifted off its target still produces a perfectly plausible total energy, so
no downstream consistency check can catch it. `exchange.check_constraint()`
therefore re-reads each finished run's muffin-tin moments out of `INFO.OUT`
and raises if any constrained atom is off target by more than a tolerance.
This is not defensive programming: it was added because the first
perpendicular two-site probe attempted here (bcc Fe, moments 90° apart) ran
60 SCF cycles without ever reaching its targets, the moments collapsing to
~0.16 μB instead.

Relatedly, a configuration that fails to converge invalidates the whole
component and raises, rather than being dropped — unlike a least-squares fit
over many configurations, four-state depends on complete convergence of all
four states (a failure mode arXiv:2512.08471 calls out explicitly).

### Cost

Nine components × four states = **36 constrained SCF runs per pair**.
`components="symmetric"` costs 24 and is valid only where an inversion
centre forces $\mathbf D=0$; computing all nine on at least one bond is what
turns that into a test. The work is embarrassingly parallel over
configurations and each Elk process is serial, so `workers=` threads the
sweep rather than a single run (`ELKPY_MAX_CONCURRENT` still bounds actual
concurrency). Each configuration is a separate `Calculation` in its own
subdirectory, so the ground-state manifest cache applies per configuration
and a re-run after a crash reuses whatever already converged.

### Known limitation: pseudospin systems

Elk's `mommtfix` constrains the **spin** moment only. For a $j_{\rm eff}=1/2$
spin-orbit Mott insulator such as α-RuCl₃ this is not sufficient: the moment
is 2/3 orbital and 1/3 spin, and Hou, Xiang & Gong (PRB 96, 054410 (2017),
arXiv:1612.00761) measure that spin and orbital directions "seriously deviate
from each other" unless the orbital moment is constrained too, adding a
penalty functional
$E_{\rm constr}=\lambda\sum_t[\mathbf L^t-\hat{\mathbf L}^{t,0}(\hat{\mathbf L}^{t,0}\cdot\mathbf L^t)]^2$
with $\lambda=0.2$ eV μ$_B^{-2}$.

Elk is unusually well shaped for adding this, should it be needed:
`eveqnsv.f90:96-105,160-168` already implements $\hat H_{Bo}=\tfrac{1}{2c}\mathbf B\cdot\hat{\mathbf L}$
(`bforb`) inside a loop that is already per-atom, applying $\hat L_{x,y,z}$
via `lopzflmn` — the same `lopz*` family patch 0006 reuses for §19. Only the
field source is hard-wired to the *global* `bfieldc`. A patch would mirror
`bfieldfsm.f90`'s feedback loop against
$\langle\mathbf L\rangle_{ias}=\mathrm{Tr}[\mathbf{dmatmt}_{ias}\hat L_a]$
(available every SCF cycle whenever DFT+U is on, `gndstate.f90:192`) — plus,
critically, its own total-energy correction: unlike the spin constraint this
term enters the second-variational Hamiltonian rather than `bsmt`, so
`energy.f90`'s `sm` subtraction does **not** remove it. **Not implemented**;
the diagnostic that would justify it is a spin-only-constrained run with
$\langle\mathbf L\rangle$ measured per atom via §19's `get_angular_momentum()`.

### Verification status

**The Python arithmetic is verified; the end-to-end DFT extraction is NOT yet.**
Stated explicitly because every other capability in this document was
validated against a real compiled binary before being written up.

Verified (`tests/test_parsers_exchange.py`, 16 synthetic pins, no Elk run):
the round trip recovers a known tensor exactly for all nine components; the
cancellation claim holds numerically against a model that adds single-ion
anisotropy on every site, pair-to-spectator couplings and spectator-spectator
couplings; multiplicity divides out; Xiang's uncorrected formula is shown to
manufacture a spurious DM vector on a symmetric tensor where ours gives zero;
the DM extraction matches its cross-product definition; the Kitaev
parametrisation and frame rotation round-trip; the SIA formulas recover a
known anisotropy; and `check_supercell` rejects a contaminated pair.

Verified against a real binary, but only as isolated pieces rather than a
completed tensor:

- **The Phase-0 constraint probe is encouraging but PARTIAL.** A perpendicular
  two-site configuration on NiO (atom 0 along $+x$, its J₂ partner along $+z$,
  all six spectators along $+y$) had every Ni moment on target to ~1° with a
  healthy 1.80 μB — the right size for Ni(II), $S=1$ — but the run was
  interrupted after ~4 SCF cycles and never reached `epsengy`, so this shows
  the constraint *locking on quickly*, not that it holds through convergence.
  Encouraging because the comparison case failed outright: an earlier bcc Fe
  probe at 90° ran 60 cycles with the moments collapsing to ~0.16 μB instead,
  which is why `check_constraint()` exists. That production path — the check
  applied at convergence — has itself never fired on a completed run.
- **The `bfcmt` sign convention** was measured, not assumed: a single bcc Fe
  atom with `bfcmt = (0,0,+4)` converges to a moment of −2.75 μB.
- **The supercell bookkeeping** is checked through the real code path
  (`test_supercell_is_large_enough_for_the_j2_pair`, no binary needed): NiO's
  J₂ pair in a 16-atom cell has multiplicity 6 at 4.170 Å.

**Still to run**: the full nine-component sweep, i.e.
`ELKPY_RUN_SLOW_TESTS=1 python3 -m pytest tests/test_calculation_exchange_nio.py`.
Its three assertions are the ones with real teeth, and none of them has been
exercised yet:

1. $J_2$ antiferromagnetic and of order 20 meV — the literature benchmark.
2. **The tensor exactly isotropic with SOC off.** Convention-free: nothing
   ties spin to the lattice, so the energy is invariant under a global spin
   rotation. Whatever deviation appears *is* the method's numerical noise
   floor, measured rather than assumed.
3. $\mathbf D=0$ on the J₂ bond, whose midpoint is the bridging oxygen and
   hence an inversion centre.

Measured cost, which is why it has not been run: ~150 s per SCF loop for the
16-atom cell at `ngridk=(2,2,2)` with `reducek=0` (8 k-points), roughly 40
loops per configuration, 36 configurations — of order 60 CPU-hours. It
parallelises cleanly over configurations, and the per-configuration manifest
cache means an interrupted sweep resumes rather than restarting.

## How to use in code

```python
from ase.build import bulk
from elkpy.structure import Structure

# a supercell large enough that the target pair is not aliased onto a
# different neighbour shell by periodic images
structure = Structure.from_ase(bulk("NiO", "rocksalt", a=4.17) * (2, 2, 2))
calc = structure.get_calculation(
    "nio", xc="PW", spinpol=True, ngridk=(2, 2, 2),
    extra_blocks={"dft+u": [(1, 1), (1, 2, 0.2205, 0.0367)]},   # FLL, U/J on Ni d (Ha)
)

result = calc.get_exchange_tensor(
    i=0, j=4,              # 0-based global atom indices, Elk's own ordering
    magnetic="Ni",         # every magnetic atom: spectators must be held too
    spin=1.0,              # Ni(II) is S = 1 -- sets the convention, not just a scale
    components="all",      # 36 runs; "symmetric" costs 24 where D = 0 by symmetry
    workers=8,
)
result["tensor"]        # 3x3, meV
result["isotropic"]     # Heisenberg J
result["dm"]            # Dzyaloshinskii-Moriya vector (0 on an inversion-symmetric bond)
result["symmetric"]     # traceless symmetric anisotropy

# for a honeycomb Kitaev magnet, rotate into the cubic octahedral frame
from elkpy.parsers import exchange
cubic = exchange.rotate_tensor(result["tensor"], exchange.honeycomb_cubic_frame())
exchange.kitaev_parameters(cubic)     # J, K, Gamma, Gamma', and a residual
```

## 30. Spin-polarised STM images (Tersoff-Hamann)

`Calculation.get_spin_stm()` / `get_spin_stm_3d()` (elkpy tasks 9003/9004,
`patches/0011-spin-polarized-stm.patch`, `src/elkpy_stm.f90`) simulate what a
spin-polarised scanning tunneling microscope measures: the vacuum local
density of states above a magnetic surface, projected onto the tip's own
magnetisation direction — an arbitrary Cartesian $\hat{\mathbf e}_T$, e.g.
$(1,1,1)$.

### The physics

Tersoff and Hamann (PRL **50**, 1998 (1983); PRB **31**, 805 (1985)) model
the tip as an $s$-wave orbital, which reduces the tunnel current to the
sample's local density of states at the tip position $\mathbf r$ and the
energy set by the bias. Wortmann, Heinze, Kurz, Bihlmayer and Blügel (PRL
**86**, 4132 (2001)) extended this to a magnetic tip, giving

$$
\frac{dI}{dV}(\mathbf r) \;\propto\; n_T\Big[\,n(\mathbf r, E_F + eV)
  \;+\; P_T\,\mathbf m(\mathbf r, E_F + eV)\cdot\hat{\mathbf e}_T\,\Big],
\qquad P_T = \frac{|\mathbf m_T|}{n_T},
$$

where $n$ and $\mathbf m$ are the sample's *energy-resolved* charge and
magnetisation densities

$$
n(\mathbf r, E) = \sum_{n\mathbf k} \delta(E-\varepsilon_{n\mathbf k})\,
   \psi^\dagger_{n\mathbf k}(\mathbf r)\,\psi_{n\mathbf k}(\mathbf r),
\qquad
\mathbf m(\mathbf r, E) = \sum_{n\mathbf k} \delta(E-\varepsilon_{n\mathbf k})\,
   \psi^\dagger_{n\mathbf k}(\mathbf r)\,\boldsymbol\sigma\,\psi_{n\mathbf k}(\mathbf r),
$$

and $n_T$, $\mathbf m_T$ the tip's own (structureless, in this
approximation) density of states and magnetisation. What the approximation
neglects is everything about the tip beyond those two numbers: its orbital
character, its atomic structure, and any tip-sample interaction. What it
captures is the part that matters for atomic-scale magnetic contrast — the
spin-dependent projection, and the exponential vacuum decay that makes an
STM a surface probe at all.

Note $\mathbf m = n_\uparrow - n_\downarrow$ in Elk's convention (the
expectation of $\boldsymbol\sigma$, not of $\mathbf S = \tfrac12
\boldsymbol\sigma$ — a factor of 2 relative to the spin operators of §17), so
$|\mathbf m \cdot \hat{\mathbf e}| \le n$ pointwise and the local spin
polarisation the tip sees is bounded by 1.

The physical content of the projection is that a *chemically identical* set
of atoms can be magnetically inequivalent. A conventional STM integrates over
spin and sees only the chemical lattice; the spin-polarised image carries the
magnetic periodicity, which is generally larger — and, since each 2D Fourier
component of the vacuum LDOS decays as $\exp\!\big(-2z\sqrt{\kappa^2 +
|\mathbf G|^2/4}\,\big)$ with $\kappa=\sqrt{2m\Phi}/\hbar$, the
longer-wavelength magnetic superstructure decays *more slowly* than the
atomic corrugation. Retracting the tip therefore enhances the magnetic
contrast relative to the chemical one (Heinze *et al.*, Science **288**, 1805
(2000)).

### Why no new physics machinery was needed, only a new field to plot

Elk already computes $\mathbf m(\mathbf r, E)$ and throws it away. Upstream
task 162 (`src/wfplot.f90`) makes an STM image by replacing the
second-variational occupation numbers with an energy-selecting weight and
calling `rhomagv`, which fills the *global* `rhomt`/`rhoir` **and**
`magmt`/`magir`, symmetrises both (`symrf`/`symrvf`) and converts both to the
fine radial/interstitial grids. Task 162 then plots the charge density alone.

So `elkpy_stm.f90` is upstream's own routine with the magnetisation kept: the
same occupation replacement, the same `rhomagv` call, and then the linear
combination

$$
f_1 = n,\qquad
f_2 = \mathbf m \cdot \hat{\mathbf e}_T,\qquad
f_3 = n + P_T\,\mathbf m\cdot\hat{\mathbf e}_T
$$

formed pointwise in the same muffin-tin/interstitial representation (the
combination is linear, so it commutes with the spherical-harmonic expansion
and needs no re-expansion), handed to Elk's generic `plot2d`/`plot3d`
writers. $f_1$ and $f_2$ are the complete information — any other tip
polarisation is a recombination of them, not a re-run.

Two representation details:

- **Collinear vs non-collinear vs spin-orbit.** `ndmag` is 1 for a collinear
  run and 3 for a non-collinear one (`init0.f90`: 3 if any `bfcmt` seed has an
  $x$/$y$ component, or if `spinorb` is set — SOC is non-collinear in
  general). With `ndmag = 1` only the $z$-component exists, so an in-plane tip
  direction returns identically zero — the routine prints a note saying so
  rather than failing, since that *is* the correct answer for a collinear
  magnet. **Trap**: `init0.f90` applies `if (cmagz) ndmag=1` *after* the
  `spinorb` line, so setting `cmagz` (a common speed knob for a collinear
  ferromagnet with SOC) overrides it and silently collapses the magnetisation
  back to one component. It defaults to `.false.`.

  SOC needs nothing special from this task and is verified: it enters the
  second-variational Hamiltonian (`eveqnsv`), so `evalsv`/`evecsv` already
  carry it and `rhomagv` is unchanged downstream. Re-running the worked
  example with `spinorb` gives the same Néel state (3.742 $\mu_B$, 120°
  apart), the same exact $0:+1:-1$ sublattice ratios, $|\mathbf m\cdot
  \hat{\mathbf e}|\le n$ everywhere, and a `FERMIDOS.OUT` agreement of
  $1.3\times10^{-5}$ — the same normalisation accuracy as without SOC. Note
  that structure's $\hat z$ null *survives* SOC ($m_z = 0$ exactly on all
  three atoms), but for a different reason than without it: easy-plane
  anisotropy plus the $C_3$/mirror symmetry forbids out-of-plane canting.
  A lower-symmetry magnet with SOC would generally cant, and the null with it.
- **Energy window.** Two modes: a smeared $\delta$-function at $E_F + eV$
  (`elkpy_stmint = .false.`, the differential-conductance map, weight
  $\delta_w(E_F + eV - \varepsilon)$, units of states per Hartree per
  Bohr$^3$), or the smeared window between $E_F$ and $E_F + eV$
  (`elkpy_stmint = .true.`, constant-current/topograph mode, weight
  $|\Theta_w(E_F + eV - \varepsilon) - \Theta_w(E_F - \varepsilon)|$, units of
  states per Bohr$^3$ — note *no* $1/w$ factor, this is a state count, not a
  density of states). `swidth` sets $w$; the ground-state default of
  $10^{-3}$ Ha is a sharper energy selection than a practical k-mesh can
  resolve, so the STM call raises it.

### An upstream bug found on the way

`rhomagv` passes `wkpt(ik)` to `rhomagk` as a separate argument, and
`rhomagk` multiplies the occupation by it — so `occsv` must be a *pure*
occupancy, exactly as `occupy.f90` stores it. Upstream `wfplot.f90`'s
task-162 branch sets `occsv = occmax*wkpt(ik)*sdelta(...)/swidth`, folding
the weight in a second time; its image is therefore weighted by
$w_{\mathbf k}^2$. The same file's task-61/62/63 branch makes this visible:
it sets `occsv = 1/wkpt(ik)` precisely to cancel the factor `rhomagv` will
apply. With a uniform weight this is a harmless overall scale, but with
`reducek /= 0` the weights differ between k-points and the image is genuinely
distorted. `elkpy_stm.f90` omits the extra factor, which is what lets its
cell integral be checked against `FERMIDOS.OUT` (see below); the
elkpy-vs-upstream image comparison is consequently a proportionality, not an
equality.

### Verification

Against a real compiled binary, on a freestanding Cr monolayer with the
triangular lattice of Cr/Ag(111) — Elk's own
`examples/magnetism/Cr-monolayer` with the vacuum opened up — which orders in
a coplanar 120° Néel state (Kurz, Förster, Nordström, Bihlmayer and Blügel,
PRB **69**, 024415 (2004)). This is the system SP-STM was originally proposed
for (Wortmann *et al.* above) and has since been simulated in detail (Palotás,
Hofer and Szunyogh, PRB **84**, 174428 (2011)). Its three chemically
identical atoms with moments 120° apart make the checks exact rather than
plausibility bands (`tests/test_calculation_spin_stm.py`):

- The spin-summed LDOS is **identical** above all three atoms (measured
  spread $< 10^{-6}$ relative) — a conventional STM sees a 1×1 lattice — while
  the projection onto a tip along $\hat x$ gives the ratios $0 : +1 : -1$ over
  the three sublattices, fixed by the moment directions alone. Rotating the
  tip to $\hat y$ changes them to $-1 : +\tfrac12 : +\tfrac12$: a different
  sublattice goes dark. That tip-direction dependence is how a non-collinear
  structure is identified experimentally (Gao, Wulfhekel and Kirschner, PRL
  **101**, 267205 (2008)).
- The projection vanishes at the hollow site equidistant from all three
  sublattices, for any tip direction — the three moments sum to zero there.
- A tip along $\hat z$ is an **exact null** (measured $5\times10^{-7}$ of the
  charge LDOS): without spin-orbit coupling the seed's $m_z = 0$ subspace is
  invariant under the Kohn-Sham flow, so the Néel state stays coplanar in
  $xy$. This is the check that the Cartesian component asked for is the one
  that comes back — not, say, $|\mathbf m|$.
- A $(1,1,1)$ tip reproduces $(\mathbf m\!\cdot\!\hat x + \mathbf
  m\!\cdot\!\hat y + \mathbf m\!\cdot\!\hat z)/\sqrt3$ from three separate Elk
  runs to $2.5\times10^{-7}$ relative — the off-axis case is exactly linear,
  and the normalisation of $\hat{\mathbf e}_T$ is the same every time.
- $|\mathbf m \cdot \hat{\mathbf e}| \le n$ pointwise, which a factor-of-2
  convention slip between $\boldsymbol\sigma$ and $\mathbf S$ would break.
- **Absolute normalisation** — the one thing every check above, being a ratio
  or a null, leaves free — against a genuinely independent Elk code path: the
  unit-cell integral of the plotted spin-summed LDOS at zero bias *is* the
  density of states at $E_F$, which `src/occupy.f90` computes as a sum over
  eigenvalues (no charge density, no plotting machinery at all) and writes to
  `FERMIDOS.OUT`. Measured agreement of order $10^{-5}$ relative ($1.2\times10^{-5}$ on the
  worked example, $3.8\times10^{-5}$ on a coarser cell) — the same order as the
  density representation's own total-charge error. This is the
  check that caught the `wkpt` double counting described above, as an exact
  factor of 2.
- Agreement with upstream task 162's own `STM2D.OUT` on the identical plane,
  on an unreduced mesh (`reducek = 0`, so every weight is $1/N_{\mathbf k}$):
  the pointwise ratio is constant to $10^{-6}$ across the grid *and* equals
  $N_{\mathbf k}$ exactly — not merely a proportionality, but a quantitative
  confirmation that the whole difference between the two routines is the one
  extra $w_{\mathbf k}$ described above.

### How to use in code

```python
from elkpy.structure import Structure

a, c = 5.50836, 24.0
avec = [(1.5 * a, 0.866 * a, 0), (1.5 * a, -0.866 * a, 0), (0, 0, c)]
# three Cr atoms with bfcmt seeds 120 degrees apart in the xy-plane
species = {"Cr": [((0, 0, 0), (0.0, 0.1, 0.0)),
                  ((1 / 3, 1 / 3, 0), (-0.0866, -0.05, 0.0)),
                  ((2 / 3, 2 / 3, 0), (0.0866, -0.05, 0.0))]}

calc = Structure(avec, species).get_calculation(
    "cr", xc="PW", spinpol=True, ngridk=(6, 6, 1),
    extra_blocks={"nempty": [8], "reducebf": [0.5]},   # seeds off once converged
)

r = calc.get_spin_stm(
    direction=(1, 1, 1),   # tip magnetisation, CARTESIAN
    height=0.25,           # tip plane, fraction of the c axis
    grid=(60, 60),
    polarization=1.0,      # P_T; only the "image" field depends on it
    bias=0.0,              # sampling energy relative to E_F, Hartree
    swidth=0.005,          # width of the energy selection
)
r["ldos_grid"]        # (n2, n1) conventional STM image
r["spin_ldos_grid"]   # (n2, n1) magnetic contrast, m . e_T
r["image_grid"]       # (n2, n1) n + P_T m . e_T -- what the tip measures
r["dos"]              # cell integral of the first field = DOS at E_F

# the vacuum decay, for choosing a tip height
points, values = calc.get_spin_stm_3d(direction=(1, 1, 1), grid=(20, 20, 40))
```

The same calculation runs directly from an `elk.in`, with no Python at all —
see `examples/spin-stm/`:

```
tasks
  0
  9003

elkpy_stmdir
  1.0  1.0  1.0

elkpy_stmpol
  1.0

plot2d
  0.0  0.0  0.25
  1.0  0.0  0.25
  0.0  1.0  0.25
  60  60
```


## 31. Vertical tunnelling transport: a point tip to a substrate plane

`Calculation.get_vertical_transport()` (elkpy task 9005,
`patches/0012-vertical-transport.patch`, `src/elkpy_transport.f90`) computes
what gets *through* a two-dimensional material, rather than what a scanning
tunneling microscope sees above it (§30). An electron enters at a point
$\mathbf r$ above the sheet — the STM tip — and leaves into an infinite,
featureless metallic plane below it — the substrate the material sits on.

### The physics

In the tunnelling regime the Landauer-Büttiker transmission is
$T=\mathrm{Tr}[\Gamma_{\rm t}G^r\Gamma_{\rm s}G^a]$, and a **point** tip makes
$\Gamma_{\rm t}$ rank one, which collapses the trace exactly to an integral of
the nonlocal Green's function over the exit region:

$$
T(\mathbf r;E)\;\propto\;\int_{\rm plane}\big|G(\mathbf r,\mathbf r';E)\big|^2
  \,d^2r',
\qquad
G(\mathbf r,\mathbf r';E)=\sum_{n\mathbf k}
  \frac{\psi_{n\mathbf k}(\mathbf r)\,\psi^*_{n\mathbf k}(\mathbf r')}
       {E-\varepsilon_{n\mathbf k}+i\eta}.
$$

The band sum happens **before** the modulus is taken, and that is the whole
point: different bands are different routes through the material and they add
as amplitudes, not as probabilities. $\eta$ is the energy width the leads let
states through in — the leads' own coupling, not a numerical smearing. The two
couplings are unfixed prefactors, so what this delivers is the map, its
contrast and its bias dependence, not an absolute conductance.

**The substrate is an infinite, featureless plane**, invariant under every
lateral lattice translation. That is a physical claim, and it is what makes
the calculation cheap: the exit integral over the *whole* plane,

$$
\int_{\rm all\ cells} d^2r'\,\psi^*_{n\mathbf k}(\mathbf r')
  \psi_{n'\mathbf k'}(\mathbf r')
\;\sim\;\delta(\mathbf k'_\parallel-\mathbf k_\parallel+\mathbf G_\parallel),
$$

forces $\mathbf k'=\mathbf k$ on a Brillouin-zone mesh. **The k-sum is
therefore incoherent and the interference is between bands at the same
$\mathbf k$.** What survives is one Hermitian matrix per k-point — the bands'
Gram matrix restricted to the plane — and one quadratic form:

$$
S_{\mathbf k}[n,n']=\int_{\rm plane\ in\ one\ cell}
  \psi^*_{n\mathbf k}(\mathbf r')\,\hat P_{\rm s}\,
  \psi_{n'\mathbf k}(\mathbf r')\,d^2r',
\qquad
T(\mathbf r;E)=\sum_{\mathbf k}w_{\mathbf k}\sum_{nm}
  a_n(\mathbf r)\,a^*_m(\mathbf r)\,S_{\mathbf k}[n,m].
$$

A *finite* contact patch is a different physical regime and is not this.

Three things follow from the word "Gram" and all three are load-bearing. The
transmission is **non-negative by construction** rather than by luck. Widening
the exit region to the whole cell makes $S_{\mathbf k}$ the identity by
orthonormality, so the quantity becomes the Tersoff-Hamann tunnelling density
of states — the validation route below. And a quadratic form is **blind to a
rotation inside a degenerate multiplet**, so §14's warning about `evecsv`
bases is satisfied by construction rather than by handling degeneracies.

$\hat P_{\rm s}=1+P_{\rm s}\hat{\mathbf n}\cdot\boldsymbol\sigma$ is a
*magnetic* substrate's spin acceptance — the spin-space sibling of §30's
magnetic tip, but sitting **inside** the overlap integral, between two
different bands, where a density has already been squared. The normalisation
is §30's own (not the $(1+P\hat{\mathbf n}\cdot\boldsymbol\sigma)/2$ a
Landauer coupling matrix is usually written with) so that
$\mathrm{tr}[\hat P_{\rm s}\rho]=n+P_{\rm s}\mathbf m\cdot\hat{\mathbf n}$
matches `get_spin_stm()` exactly and $P_{\rm s}=0$ gives the plain spin-summed
Gram matrix with no factor.

### Two conventions taken from a measurement, not chosen

Both come from the sister plane-wave code (`defumat`), which measured them;
they are restated here rather than rediscovered.

**The amplitude is the on-shell one.** Writing $G$ with its literal energy
denominator is the exact Landauer expression, and a sum over states **cannot
evaluate it**: the denominator has a long tail, so the states far from $E$
carry the barrier's evanescent decay, and that decay is built entirely out of
cancellation between them. Measured on a cell small enough to diagonalise
completely, the running sum for one $G(\mathbf r_{\rm tip},\mathbf r_{\rm
exit})$ wanders over more than an order of magnitude and only lands when the
basis is *complete*, at a cancellation ratio of 349. There is no band count at
which a truncated version is right. What converges is the **modulus** of that
denominator with its phase held fixed, which is the weak-coupling limit of the
same expression — Bardeen's golden rule, the sample visited on shell:

$$
a_{n\mathbf k}(\mathbf r)=\psi_{n\mathbf k}(\mathbf r)\,
  \sqrt{\mathrm{occmax}\;\delta_\eta(E-\varepsilon_{n\mathbf k})/\eta}.
$$

It is *exactly* the resolvent's modulus for a Lorentzian
($|1/(E-\varepsilon+i\eta)|=\sqrt{\pi\delta_\eta/\eta}$), so all that is
dropped is the arctangent the phase sweeps across a resonance; and the
interference it keeps is between bands **degenerate at the tip energy**, which
is the interference a tunnelling experiment lets happen and which no local
density of states can represent. What it drops is direct tunnelling through
the barrier without going on shell in the sample — which is what a
weak-coupling tunnelling geometry is defined by not having.

**The conjugation sits on the exit variable.** $G(\mathbf r,\mathbf
r')=\sum_n a_n(\mathbf r)\psi^*_n(\mathbf r')$ carries $\psi$ conjugated in
$\mathbf r'$, so the plane integral of $|G|^2$ is $\sum_{nm}a_na^*_mS[n,m]$
and **not** $\sum_{nm}a^*_na_mS[n,m]$. The two differ by a transpose of a
Hermitian matrix, so the wrong one is real, non-negative, blind to a
degenerate rotation, and *exactly* right in the Tersoff-Hamann limit where $S$
is the identity — it is wrong only where $S$ has an off-diagonal, which is to
say only where the interference this exists for actually lives. Nothing but a
literal check against the definition separates them, which is what
`parsers.transport.green_function_transmission()` is for.

### Why this fits Elk cleanly

Both planes sit in the vacuum, i.e. in the **interstitial**, where Elk's
wavefunction is an exact plane-wave sum. The normalisation is `wfirsv.f90`'s
own — it applies $\Omega^{-1/2}$ only on its real-space branch, and
`genolpq.f90` confirms it by scaling its interstitial contribution back up by
$\sqrt\Omega$ — so with $c_{\mathbf G}$ the $G+p$-space coefficients,

$$
\psi_{n\mathbf k}(\mathbf r)=\Omega^{-1/2}\sum_{\mathbf G}c_{\mathbf G}
  e^{i(\mathbf k+\mathbf G)\cdot\mathbf r},
\qquad
S_{\mathbf k}[n,n']=\frac{A}{\Omega}\sum_{\mathbf G_\parallel}
  b^*_n(\mathbf G_\parallel)\,b_{n'}(\mathbf G_\parallel),
$$

with $b_n(\mathbf G_\parallel)=\sum_{G_3}c_n(\mathbf G_\parallel,G_3)
e^{2\pi iG_3s_3}$ and $A=|\mathbf a_i\times\mathbf a_j|$ the surface cell
area. **It is computed exactly, with no quadrature and no convergence
parameter**: one gather and one matrix product per k-point. A plane that is
*not* spanned by two lattice vectors has no such relation, which is why the
exit plane is specified as an axis plus a height rather than as a general
parallelogram — a two-dimensional material's substrate is never tilted.

**The k-mesh is the task's own, and that is better than reusing the ground
state's.** `elkpy_transport_wf` diagonalises fresh at each generated k-point
(`eveqnfv`/`eveqnsv`, the route `bandstr.f90` takes for an arbitrary path,
already used by §13's task 9001 and §14's session), so the transport mesh is
independent of `ngridk` *and of `reducek`* — no SCF re-run, no symmetry
restriction, and no risk of the failure mode a reduced wedge would carry here:
the symmetry reduction uses $z$-flipping operations, and the Gram matrix at
$C_2'\mathbf k$ is the overlap on the *mirror-image* plane, not
$S_{\mathbf k}$. It differs from `elkpy_wfcorner` in expanding only the
interstitial part (`wfirsv` with `tgp=.true.`, no muffin-tin part at all) —
not an optimisation for its own sake, since the muffin-tin expansion would be
both unused and, at `natmtot` atoms times `nstsv` states, the dominant cost.

Only states inside `elkpy_transport_window` are exported, which is what keeps
the file small; the on-shell weight kills the rest.

### The traps, all guarded in Fortran rather than documented

- **A plane cutting a muffin-tin sphere.** Both formulas use the interstitial
  plane-wave representation, which is not the wavefunction inside a sphere and
  would return a perfectly plausible number there. Checked exactly for the exit
  plane (every atom's perpendicular distance, including the plane's periodic
  images along the offset axis — lateral images all lie *in* the plane, which
  is the point of an exit plane spanned by two lattice vectors) and against
  every atom's nearest images for the tip points. Note the radius that matters
  is the one `checkmt` ends up with, not the one in the species file: on the
  graphene fixture it shrinks carbon from 1.80 to 1.32 Bohr.
- **`tshift` is mandatory-false**, the same trap as §28's rotation
  indicators. Elk relocates the origin by default — for a crystal with
  inversion, onto the inversion centre, which for a honeycomb is the bond
  midpoint — while both plotting planes stay in the input frame. Measured
  directly: with the default the graphene sheet moves from $z=0.5$ to $z=0$
  and both planes end up on the same side of it. Enforced in Python
  (`extra_blocks={"tshift": [False]}`), and named in the Fortran error message
  when the geometry check fires.
- **The material must lie between the planes.** A cell is periodic, so "above"
  and "below" are only meaningful relative to where the atoms are. A refusal,
  not a warning — except for the one deliberate case where the tip plane
  *coincides* with the exit plane, which is the self-check configuration
  below.
- **`rgkmax` bounds how far into the vacuum this is meaningful.** An
  all-electron LAPW basis has a far lower plane-wave cutoff than a
  pseudopotential code, and both planes live in the tail that cutoff
  represents. $S_{\mathbf k}$ stays Hermitian and positive semi-definite while
  becoming numerically meaningless, so this one is a documented limitation,
  not a guard: keep both planes reasonably close to the slab.
- **A mesh without K gives identically zero, not something small**, for
  graphene — the Gaussian/Fermi-Dirac weight underflows when no state is
  within a few $\eta$ of the tip energy. The transport k-grid must be a
  multiple of three.

### Four traps that are NOT in the Fortran, found by driving this in the field

`docs/field_report_nibr2.md` is a report from a real 45-atom NiBr2 monolayer
spin spiral (15 Ni per magnetic period, in-plane helix, `spinorb`) put through
tasks 9003 and 9005. The headline is a result, not a bug: the two tasks
**agreed with each other to 0.2% on every well-sampled harmonic** — `rhomagv`
in Fortran against the Python Gram-matrix contraction, two code paths sharing
nothing, on a system an order of magnitude past anything in `tests/`. The four
items below are the edges it found, and all four are about the *Python* API
surface, which is why none of them could be a Fortran guard.

- **`compute_transmission(energies=)` is ABSOLUTE; `get_vertical_transport(energies=)`
  is relative to $E_F$.** Same parameter name, adjacent layers, opposite
  conventions — the wrapper does `absolute = [efermi + e for e in energies]`
  and the parser does not. Dropping to the parser is the *documented* way to
  drive this by hand (a bias sweep costs one contraction per energy, so the
  parser layer is where a $dI/dV$ map gets built), and the mismatch failed
  **silently**: `amplitude_weights`' smeared delta has exponential tails, so
  every exported state still picks up a small non-zero weight and what comes
  back is a plausible small map at the wrong energy rather than an error. On
  the NiBr2 run $E_F = -0.162$ Ha, so a $+0.10$ Ha bias is an absolute
  $-0.062$ Ha, while the same number passed unshifted lands $0.16$ Ha from
  where it belonged — on a window $0.38$ Ha wide ($[-0.322, +0.058]$
  absolute), $0.04$ Ha past the top. It cost a full re-run, and was caught only
  because the `exit_region="cell"` cross-check came out at $1.4$ instead of
  $\sim10^{-16}$.

  Fixed by `parsers.transport._check_energy_window()`, which refuses an energy
  outside $[E_F + w_1, E_F + w_2]$ and names the shift in the message. A
  **raise**, not a warning: `src/elkpy_transport.f90` selected only the states
  in that band (its `e0`/`e1`), so outside it there is nothing to compute
  from and no degraded answer to fall back on. The exported `window` is
  written back *relative* to $E_F$, exactly as it was passed in, which is what
  makes the bound checkable at all. The argument was deliberately **not**
  renamed: `energies=` fans out to the tests, the notebook, the example and
  this document, and the check alone closes the silent failure.

  The report's own guard added a $3\eta$ margin inside those bounds, on the
  grounds that the smeared delta is truncated at the export edge and the
  weight there is wrong rather than absent. That is right for a caller and
  wrong for a library — it would refuse energies that genuinely have states
  — so the bound here is the hard one. It does not arise through
  `get_vertical_transport()` anyway, whose default `nsigma=8` pads the export
  by eight broadenings either side of every requested energy.

- **`amplitude_weights(..., occmax=)` must not guess.** `occmax` is Elk's own
  spin-degeneracy factor (`src/occupy.f90`): 2 for `nspinor=1`, 1 for
  `nspinor=2`. `compute_transmission` derives it from `data["nspinor"]`
  correctly, but the standalone function cannot see `nspinor` and used to
  default to 2.0 — a silent factor of two in any map built by calling it
  directly, for every spin-orbit run. Since the whole design point of §31 is
  that the arithmetic is Python and reusable, calling it directly is the
  expected thing to do. It is now required: `occmax=None` is a sentinel that
  raises with the rule in the message. The signature keeps its shape so every
  existing positional call still works.

- **`ramdisk` and eigenvector reuse.** Elk 11 defaults `ramdisk .true.`
  (`readinput.f90:412`), so eigenvectors live in memory and a run writes no
  `EVEC*.OUT` at all. Task 9003 needs them: `rhomagv` calls
  `getevecfv`/`getevecsv` from store and does **not** re-diagonalise, which is
  exactly what makes a bias sweep cheap. elkpy's own path is unaffected —
  `_run_resumed()` prepends task 1 in the *same* elk process as 9003, so the
  RAM disk that task 1 fills is the one 9003 reads. It bites the hand-driven
  workflow: a ground state converged in a separate `elk` invocation leaves
  nothing on disk for a later 9003 to read, and the fix is `ramdisk .false.`
  in the run that produces the eigenvectors. Task 9005 is immune either way,
  since `elkpy_transport_wf` diagonalises fresh at its own k-mesh.

- **A version boundary in the eigenvector files, not elkpy's to fix.** Elk
  11.0.2 has `nstfv = nint(chgval/2) + nempty + 1` (`init1.f90:319`) where
  10.2.4 had no `+1`, so **10.2.4 eigenvector files are rejected** by
  `getevalsv`/`getevecfv` ("differing nstsv") while `readstate` accepts the
  10.2.4 *density* with only a version warning — the density crosses the
  boundary and the eigenvectors do not. Note also that the `nempty` block is
  not the number it looks like: `init1.f90:316` sets
  `nempty = nint(nempty0*natmtot)`, so on a 45-atom cell `nempty 180` asks for
  8100 empty states and `nstfv` is then clamped to the matrix size (`nstsv`
  came out at 10768, not 1110). The clean workaround for both is task 1 with
  `maxscl 1` and `ramdisk .false.`, regenerating eigenvectors from the old
  density.

### Transverse sampling can alias the atomic lattice into a low harmonic

A `plot2d` map over a supercell invites harmonic analysis — both
`examples/spin-stm/` and `examples/vertical-transport/` do exactly that, and
so did the NiBr2 run — and the transverse sample count is not free.

Average an $n_1 \times n_2$ map over $\mathbf a_2$ to get a profile along
$\mathbf a_1$, and the average kills every Fourier component
$\mathbf G = m\mathbf b_1 + p\mathbf b_2$ except those with
$p \equiv 0 \pmod{n_2}$. That is a *sampling* statement, not a physical one:
components with $p \ne 0$ that survive it are aliases, and they land at
harmonic $m$ of the profile as if they were the real thing. On the NiBr2
$15\times1$ supercell both sublattices satisfy $p \equiv 2m \pmod{15}$, so
atomic weight leaks into $x$-harmonic $m$ whenever
$2m \equiv 0 \pmod{\gcd(15, n_2)}$; with $n_2 = 6$ the $(m,p) = (3,6)$
component landed on the $3q$ spiral harmonic and read $3.4\times10^{-3}$ where
the true value is $2.9\times10^{-6}$ — three orders of magnitude of pure
artifact.

**The rule: pick $n_2$ sharing the supercell's own periodicity.** And a single
line cut is not a cheaper substitute for the average: it shows every $(m,p)$
at harmonic $m$ with no suppression at all, which on that run read a 66%
charge modulation at $q$ where the correctly averaged value is
$1.4\times10^{-4}$.

### Verification

Neither Elk nor any widely used plane-wave code computes this quantity —
QE's `PWCOND` is a Landauer transmission but between two semi-infinite
*crystalline leads*, with no point contact and therefore no map — so there is
no reference output anywhere and the validation closes inside the package.

- **The Tersoff-Hamann limit, exact rather than approximate.** With
  `exit_region="cell"` every $S_{\mathbf k}$ is the identity and $T$ becomes
  $\sum_{n\mathbf k}w_{\mathbf k}|\psi_{n\mathbf k}(\mathbf r)|^2
  \delta(E-\varepsilon)$ — the same number `get_spin_stm()` (§30) reaches
  through `rhomagv` and Elk's own plotting machinery, a code path sharing no
  line with this one. Measured on graphene: **1.6e-5 of the peak, with no
  factor between them**. That single number pins `occmax`, the k-weights, the
  spinor sum, the $\Omega^{-1/2}$ normalisation, the tip sampler and the
  transcription of Elk's `sdelta` at once — and it is the only check a wrong
  overall normalisation could not hide in, every other one here being a ratio
  or a null. The residual is the density representation's own error (the STM
  image interpolates `rhoir` off the coarse FFT grid where this evaluates the
  plane-wave sum exactly), the same order §30 measures for its own
  `FERMIDOS.OUT` check.
- **The closed form against a quadrature, on real coefficients.** Setting the
  tip plane equal to the exit plane makes the exported tip amplitudes a
  sampling of the exit plane itself, so $S_{\mathbf k}$ can be checked against
  a literal rectangle rule (exact for this band-limited periodic integrand
  once the sampling exceeds twice the largest in-plane index): **7e-15
  relative**, and Hermitian to 1e-18 with a least eigenvalue of $+2\times
  10^{-12}$.
- **The conjugation**, on the same run: the fast contraction against a literal
  $\int|G|^2$ agrees to **8e-15**, while the transposed convention differs by
  2e-6 — small, real, non-negative and wrong.
- **The physics, as an exact symmetry statement.** Fable's sharpening of the
  argument is what makes this a theorem rather than a correlation: the group
  that matters is not the little group of K but its subgroup of
  **$z$-preserving** elements, since anything flipping $z$ maps the substrate
  plane onto the tip side and is broken by the geometry. For **monolayer
  graphene** that subgroup is $C_{3v}$, whose mirror $\sigma_v$ exchanges the
  two sublattices *without* flipping $z$; the Dirac doublet is still the 2D
  irrep $E$, so by Schur's lemma $S_K\propto\mathbb 1$. Measured: eigenvalues
  $0.05242094$ and $0.05242103$, **equal to 8.5e-7 relative**, with exactly
  2.000000 open channels, zero weight off the diagonal, and a correlation with
  the tunnelling image of $>0.999$. (`defumat`, a plane-wave pseudopotential
  code, measures $\mathrm{diag}(0.05405086,0.05405084)$ for the same object —
  a cross-code agreement in the *value*, not only the structure.)
- For the **AB (Bernal) bilayer** the pair at K is degenerate for a symmetry
  reason too, but the only element exchanging its members is an in-plane
  $C_2'$, which flips $z$. The intact subgroup is $C_3$ alone, which is
  abelian, so Schur gives nothing: the two states are the non-dimer sites of
  the two *different* layers, and the substrate sees them through very
  different amounts of carbon. Measured, with the same 2.5 Bohr standoff on
  both sides so the tip weights stay comparable: $S_K$ eigenvalues
  $3.33\times10^{-5}$ and $4.36\times10^{-2}$, a **ratio of 1311** against the
  monolayer's 1.0000002. Half of $S_{\mathbf k}$ then sits off its diagonal
  (0.50 against the monolayer's 0), the correlation with the tunnelling image
  falls to 0.81, and the incoherent map — every substrate channel tunnelling
  on its own — sits 74x above the coherent one. Nothing in a Tersoff-Hamann
  image can express that.
- **Synthetic pins** (`tests/test_parsers_transport.py`, no Elk run): the
  quadrature/closed-form agreement, the conjugation, the no-factor
  Tersoff-Hamann limit, the invariance of the coherent map under a rotation
  inside a degenerate multiplet *and* the non-invariance of the naive
  incoherent diagonal until `channel_basis()` fixes it, Schur's lemma as
  arithmetic, the spin partition ($+\hat{\mathbf n}$ plus $-\hat{\mathbf n}$
  is twice unpolarised), the refusal of a Methfessel-Paxton delta (it goes
  negative on its wings and an amplitude is its square root), and the two
  index orders the Fortran writes in.

**One deliberate asymmetry with §30.** `examples/spin-stm/` needs no Python at
all; `examples/vertical-transport/` needs a dozen lines of it, because the
final contraction lives in `parsers/transport.py`. That is the same split as
§13's Berry curvature (Fortran exports overlaps, Python does the Wilson loop)
and is a choice, not an omission: implementing the weighting and the
contraction in Fortran as well would put one formula in two languages, and it
is what makes an energy sweep free — neither the wavefunctions nor
$S_{\mathbf k}$ depend on $E$, only the per-state weight does.

**Not implemented, and stated rather than silently dropped**: the literal
resolvent method (measured non-convergent above — it is not exposed at all,
only described); a finite contact patch; a tilted exit plane; plane-to-plane
geometry; and any absolute conductance.

### How to use in code

```python
from elkpy.structure import Structure

a, c = 4.647926, 20.0
avec = [(a, 0, 0), (-a / 2, a * 3**0.5 / 2, 0), (0, 0, c)]
calc = Structure(avec, {"C": [(0, 0, 0.5), (1/3, 2/3, 0.5)]}).get_calculation(
    "graphene", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
    extra_blocks={"tshift": [False], "nempty": [10]},   # tshift: mandatory
)

r = calc.get_vertical_transport(
    exit_height=0.38,      # the substrate plane, below the sheet
    height=0.62,           # the tip plane, above it (plot2d, as get_spin_stm)
    grid=(24, 24),
    kgrid=(6, 6, 1),       # the task's own mesh; a multiple of 3 contains K
    broadening=0.005,      # eta, the leads' energy window (Hartree)
    energies=[0.0],        # bias relative to E_F; a sweep costs almost nothing
)
r["transmission_grid"][0]   # (n2, n1) -- what gets through
r["interference"]           # coherent minus incoherent: what no local map has
r["channels"]               # open substrate channels (participation ratio)

# the same thing with the substrate widened to the whole cell, which IS the
# Tersoff-Hamann image -- the cross-check, and a useful comparison map
local = calc.get_vertical_transport(
    exit_height=0.38, height=0.62, grid=(24, 24), kgrid=(6, 6, 1),
    broadening=0.005, exit_region="cell", incoherent=False)

# a magnetic substrate: it accepts 1 + P n.sigma (needs spinpol/spinorb)
spin = calc.get_vertical_transport(
    exit_height=0.38, height=0.62, kgrid=(6, 6, 1),
    substrate_direction=(0, 0, 1), substrate_polarization=1.0)
```

The Elk run itself is drivable from a plain `elk.in` — task 9005 plus the
`elkpy_transport_exit` / `elkpy_transport_window` / `elkpy_transport_kgrid` /
`elkpy_transport_koffset` / `elkpy_transport_sdir` / `elkpy_transport_spol`
blocks and the usual `plot2d` — see `examples/vertical-transport/`.

## 32. The full Elk task surface: six task-family mixins and the input table

Everything above §12 is elkpy's own physics — capabilities Elk does not have.
This section is the opposite kind of work: it makes the physics Elk *already*
has reachable by name. Before it, `Calculation` wrapped about twenty of the
task codes `vendor/elk/src/elk.f90` dispatches on, and everything else went
through `run_tasks()` — which could always reach every code, but only by
handing the caller the job of knowing the task number, the prerequisite chain,
the input blocks and the output layout. That is exactly the knowledge this
project keeps insisting belongs in `spec.py` and in a docstring rather than in
a user's head. 143 of the 146 live dispatch codes now sit behind a named
method (the three exceptions are at the end of this section).

The work is organised as one mixin module per task family under
`src/elkpy/tasks/`, composed in `tasks/__init__.py`'s `ALL_MIXINS` and
unpacked by `class Calculation(*ALL_MIXINS)`. No mixin holds state, defines
`__init__`, or imports `..calculation`, so the composition is documentation
rather than semantics; `Calculation`'s own methods come first in the MRO, so
nothing a mixin defines can shadow the existing surface.

### The input surface (`params.py`)

Elk's behaviour is set almost entirely by `elk.in`, and this project has
repeatedly found that one input parameter separates a physical result from a
plausible-looking artefact. §29's four-state exchange mapping is unusable at
Elk's default `epsengy` of 1e-4 Ha (~2.7 meV), which exceeds the whole
anisotropy signal, and needs `nosym`/`reducek=0` so the four symmetry-broken
configurations share one k-set and their systematic errors cancel. §28's
rotation-eigenvalue indicators and §31's vertical transport both require
`tshift=False`, because Elk otherwise relocates the origin onto the inversion
centre — the bond midpoint of a honeycomb, not the $C_3$ axis — which silently
makes every rotation non-symmorphic and, for transport, moves the sheet out
from between the two plotting planes. Until now those settings reached Elk
through an untyped `extra_blocks` dict, where a misspelt name or a wrong arity
surfaced only as `Error(readinput)` inside a subprocess.

`params.py` records all 315 `case(...)` branches of `readinput.f90` as data:
the Fortran variable each sets and that variable's declared type and module,
the value's shape (scalar, fixed vector, padded vector, optional derivative
tail, fixed multi-line block, blank-terminated list, count-prefixed list,
verbatim text), the default, the range checks the branch performs before
`stop`ping, and a one-line description from the manual where it documents the
block and from the comment above the Fortran declaration otherwise. On top sit
a validator that rejects a bad name, type, arity or out-of-range value in
Python; a renderer emitting exact `elk.in` text; and a browsable surface
(`describe`, `search`, 27 physics categories), re-exported from `elkpy`
directly. Coverage is asserted rather than assumed — the test re-parses
`readinput.f90` and fails in *both* directions, so a version bump reports
exactly which blocks appeared or vanished, in the same spirit as `spec.py`.

### Deformation, fields, and the nucleus

Elk evaluates no analytic stress tensor. It constructs a symmetry-adapted
orthonormal basis of strain tensors $\{e_k\}$ (`genstrain.f90` — $e_1$ is the
isotropic $A/\|A\|_F$, each further candidate symmetrised over the point
group, orthogonalised against its predecessors, kept only if a finite norm
survives), deforms $A \to A + \delta\,e_k$, and differences converged total
energies, $\sigma_k = [E(\delta)-E(0)]/\delta$. `get_strain_tensors()` exposes
that basis, so the component index of every strain-differentiated quantity has
a meaning; `nstrain` is a property of the symmetry alone (exactly 1 for a
cubic crystal). Because $e_1$ is isotropic, $V(t) = V_0(1+t/\|A\|_F)^3$ and
the hydrostatic pressure $P = -\sigma_1\|A\|_F/(3V_0)$, which `get_stress()`
returns alongside the raw derivatives. This is the missing half of
`get_relaxed()`: task 2/3 relaxes positions at fixed cell; the stress says
whether the cell itself is at equilibrium. **A stress is basis-limited, not
physics, until `rgkmax` is converged** — the basis is `rgkmax` times a
muffin-tin radius that does not change with the cell, so expanding at fixed
`rgkmax` improves the basis and lowers the energy spuriously (Pulay stress);
measured $+4$ GPa at Si's experimental lattice constant, the wrong sign for
converged LDA.

The same finite-difference machinery with the King-Smith–Vanderbilt Berry-phase
polarisation (`polar.f90`) in place of the energy gives the piezoelectric
tensor $d_{ki}=\partial P_i/\partial t$ (forbidden by inversion, so a
centrosymmetric crystal is a genuine null test) and, differencing in the
applied field, the linear magnetoelectric tensor
$\alpha_{ji}=\partial P_i/\partial B_j$ — odd under both time reversal and
inversion and identically zero without spin-orbit coupling, which elkpy now
enforces rather than documents. Two all-electron nuclear-site observables
follow, both properties of muffin-tin components a pseudopotential code cannot
reach without reconstructing the core: the electric field gradient
$V_{ij}=\partial^2 V'_C/\partial r_i\partial r_j$, which sets the nuclear
quadrupole coupling measured by NQR/NMR and vanishes identically at a cubic
site; and the Mössbauer contact density $\rho(0)$ with the Fermi contact
hyperfine field $B=\tfrac{8\pi}{3}\mu_B\langle m(0)\rangle$ (Blügel et al.,
PRB 35, 3271 (1987)), which fix the isomer shift and hyperfine splitting. The
diffraction observables themselves are $F(\mathbf H)=\int\rho\,e^{i\mathbf
H\cdot\mathbf r}$ and its magnetic sibling, printed in the crystallographic
convention so that $F(0)$ is exactly the electron count — a hard normalisation
check. Beyond the local-density ground state, Hartree-Fock replaces $v_{xc}$
by the nonlocal Fock operator (or mixes them, giving hybrids); RDMFT replaces
the density by the one-body density matrix, making natural-orbital occupations
variational and so reaching the fractional occupations static correlation
demands; and the coupled tensor moments $w^{kpr}_t$ (Bultmark, Cricchio,
Granas & Nordström, PRB 80, 035121 (2009)) re-expand a DFT+U density matrix,
their low ranks being the shell occupation, spin moment and orbital moment and
their higher ranks the multipoles that distinguish competing orbital orderings.

### The plotting triples, band character, and the Fermi surface

Elk's real-space output is organised as triples whose last task digit selects
dimensionality — x1 a line, x2 a plane, x3 a parallelepiped — all routed
through `plot1d`/`plot2d`/`plot3d`. elkpy previously exposed only the 3D
member of the density, potential and ELF triples. Alongside those it now plots
the magnetisation $\mathbf m(\mathbf r)$ — the vector part of the spin density
matrix, hence *twice* §17's spin expectation value — together with its
conjugate field $\mathbf B_{xc}=\delta E_{xc}/\delta\mathbf m$, that field's
divergence (the unphysical magnetic-monopole density a local spin-density
functional carries, which `nosource` drives to zero), the local torque
$\mathbf m\times\mathbf B_{xc}$ that vanishes identically for a collinear
state and so maps where non-collinearity actually lives, the electrostatic
field $\mathbf E=-\nabla v_C$, the paramagnetic current $\mathbf j_p$, the
meta-GGA potential $W_{xc}=\delta E_{xc}/\delta\tau$, and the static screened
density. A single state's $|\psi_{n\mathbf k}(\mathbf r)|^2$ comes the way
`wfplot.f90` does it — zero every occupancy but one and reuse the charge
machinery with symmetrisation off; replacing that occupancy with a smeared
delta at $E_F$ gives upstream's spin-summed Tersoff-Hamann image, the
counterpart of §30.

Two genuinely new capabilities arrive here. The **Fermi surface**, in all five
of Elk's representations: $\prod_n(\varepsilon_n(\mathbf k)-E_F)$ over the
crossing bands, whose zero isosurface is the surface; the same eigenvalues per
band so each sheet draws separately; a smeared delta at $E_F$, summed or per
band; and the XCrySDen `.bxsf` grid the rest of the community's tooling reads.
Alongside it the **nesting function**
$N(\mathbf q)=\frac{\Omega_{BZ}}{N_k}\sum_{\mathbf k}
[\sum_n\delta(E_F-\varepsilon_{n\mathbf k})]
[\sum_m\delta(E_F-\varepsilon_{m\mathbf k+\mathbf q})]$,
whose peaks locate the parallel Fermi-surface patches that precondition a
charge- or spin-density-wave instability and a phonon Kohn anomaly, and which
is identically zero for an insulator. And the **character decomposition** of a
band structure or DOS — resolved by $\ell$, by $(\ell,m)$ (optionally rotated
into the irreps of the site symmetry group), by spin along an arbitrary
quantisation axis, or by the local moment. All are built from `gendmatk`'s
muffin-tin density matrix, the same `wfmtsv`/`wr2cmt` expansion §§16, 18, 19
turn into the atom, $\ell$-channel and angular-momentum operators, so elkpy
now has an independent Fortran code path for those. Because the weight is a
muffin-tin integral the atomic characters sum to less than one, and `dos.f90`
subtracts each channel from the running total as it goes, so `IDOS.OUT` holds
the *interstitial remainder* rather than a second total — making
IDOS + $\sum$PDOS = TDOS an exact identity (measured 8.6e-9 on a scale of
23.5) that pins the block ordering, the spin-sign convention and the meaning
of the interstitial file at once.

**One Fortran finding governs the whole Fermi-surface family**: `init1.f90`
lines 130-134 *replace* `ngridk` with `np3d` for tasks 100-104, and (except
for 102) the k-point box with the `plot3d` box. The plotting grid therefore
sets the k-mesh, and the calculation's own `ngridk` is ignored — so these
methods take `grid`, not `ngridk`, and that argument fixes the cost.

### Optical response beyond one Kubo sum

Where §22's `get_dielectric_function()` gives the independent-particle
spectrum, this family covers what lies beyond it. The second-order
susceptibility $\chi^{abc}(-2\omega;\omega,\omega)$ follows Sipe & Ghahramani,
PRB 48, 11705 (1993) and Hughes & Sipe, PRB 53, 10751 (1996): from
$r_{nm}=p_{nm}/i(\varepsilon_m-\varepsilon_n)$ it accumulates the interband
$\chi_{II}$, the intraband-modulation $\eta_{II}$ and the intraband
$\tfrac{i}{2\omega}\sigma_{II}$ terms separately and as their sum. Both
$\omega$ and $2\omega$ denominators appear, which is what makes second-harmonic
generation sensitive to states a linear spectrum cannot see; $\chi^{(2)}$
vanishes identically in a centrosymmetric crystal, so Si is a null test.

Two complementary routes to the electron-hole interaction are wrapped. The
**Bethe-Salpeter** chain diagonalises
$H_{vc\mathbf k,v'c'\mathbf k'}=(\varepsilon_{c\mathbf k}-\varepsilon_{v\mathbf
k})\delta+2K^x-K^d$ over the pair basis, with $K^d$ screened by the inverse RPA
dielectric matrix $\epsilon^{-1}(\mathbf G,\mathbf G';\mathbf q,0)$; an
eigenvalue below the independent-particle gap is a bound exciton. **Linear-
response TDDFT** instead solves the Dyson equation in the full $\mathbf
G,\mathbf G'$ basis, so it carries the local-field corrections task 121
discards; at $\mathbf q=0$ the head is a $3\times3$ matrix whose re-inversion
gives the measurable macroscopic tensor, together with Faraday rotation, a
TDDFT Kerr angle and magnetic linear dichroism. Its spin-polarised sibling
returns the full $4\times4$ charge/magnetisation response $\chi_{ij}$ whose
transverse component $\chi_{+-}$ has the **magnon energies** as its poles —
the dynamical counterpart of §29's static exchange constants.

The **real-time** branch is a protocol rather than a task:
$\mathbf A(t)=\sum_i\mathbf A_0^i e^{-(t-t_0^i)^2/2\sigma_i^2}
\sin[\omega_i(t-t_0^i)+\phi_i+r_c^i t^2/2]$
is built first (with its power density and the spectrum of $\mathbf
E=-\tfrac1c\,d\mathbf A/dt$), the Kohn-Sham orbitals are propagated under it
recording $\mathbf J(t)$, and Ohm's law is inverted in frequency space,
$\epsilon_{ij}(\omega)=\delta_{ij}+4\pi i J_i(\omega)/[(\omega+is)E_j(\omega)]$,
to extract the dielectric tensor of the actual propagation — local fields, the
kernel in use and any non-linearity the pulse excited included. Two
wavefunction-level primitives round it out: $\langle i,\mathbf k+\mathbf
q|e^{i\mathbf q\cdot\mathbf r}|j,\mathbf k\rangle$, the plane-wave matrix
elements every density response is assembled from; and the LAPW states
re-expanded in a pure plane-wave basis, whose per-state norm
$\sum_H|c_H|^2\to1$ measures directly how well a plane-wave representation
captures an all-electron wavefunction (0.982-0.994 at `hkmax=3`).

### Lattice dynamics, coupling, and superconductivity

Within the harmonic approximation the lattice is the dynamical matrix
$D_{\kappa a,\kappa'b}(\mathbf q)=(M_\kappa M_{\kappa'})^{-1/2}\sum_{\mathbf
R}e^{i\mathbf q\cdot\mathbf R}\,\partial^2E/\partial u_{\kappa a}(0)\partial
u_{\kappa'b}(\mathbf R)$, built by DFPT (205) or by finite differences of
Hellmann-Feynman forces in a supercell (200 — **the classical method this
project previously listed as not implemented**, and the only route for a
magnetic cell, since `phonon.f90` hard-stops on `spinpol`). `get_phonon_modes()`
exposes frequencies *and* eigenvectors at arbitrary $\mathbf q$, the complement
of `get_phonon_dispersion()`, which discards the eigenvectors; note `dynev.f90`
diagonalises $D/\sqrt{M_iM_j}$, so a physical displacement is
$e_i/\sqrt{M_i}$, and an unstable mode returns as a **negative** frequency,
Elk storing $\mathrm{sign}(\sqrt{|\omega^2|},\omega^2)$.

The Born effective charge $Z^*_{\kappa,ab}=\Omega\,\partial P_a/\partial
u_{\kappa b}=\partial F_{\kappa a}/\partial E_b$ comes by the same Berry phase,
with core and nuclear charge already folded into the diagonal; its
finite-frequency generalisation comes from the current following a small static
vector potential, propagated in real time. Born charges plus $\varepsilon_\infty$
are exactly what the non-analytic term
$D^{\rm NA}\propto(\mathbf q\cdot Z^*_\kappa)_a(\mathbf q\cdot
Z^*_{\kappa'})_b/(\mathbf q\cdot\varepsilon_\infty\cdot\mathbf q)$ needs, so
`get_phonon_dispersion_loto()` chains tasks 120, 121, 208, 205 and 220 to give a
dispersion carrying the LO-TO splitting at $\Gamma$. On the electronic side the
vertex $g^{\mathbf q\nu}_{mn}(\mathbf k)=(2\omega_{\mathbf q\nu})^{-1/2}
\langle\psi_{m\mathbf k+\mathbf q}|\partial V_{\rm KS}/\partial u_{\mathbf
q\nu}|\psi_{n\mathbf k}\rangle$ gives the Allen linewidth $\gamma_{\mathbf
q\nu}$, the mode coupling $\lambda_{\mathbf q\nu}=\gamma/\pi N(\varepsilon_F)
\omega^2$, its interpolation along a q-path, the Eliashberg function
$\alpha^2F(\omega)=[2\pi N(\varepsilon_F)]^{-1}\sum_{\mathbf q\nu}
(\gamma/\omega)\delta(\omega-\omega_{\mathbf q\nu})$ with
$\lambda=2\int\alpha^2F/\omega\,d\omega$ and the Allen-Dynes $T_c$ (PRB 12, 905
(1975)), a full isotropic Matsubara-axis solution for $\Delta(i\omega_n)$ and
$Z(i\omega_n)$, and — skipping the isotropic approximation entirely — the
self-consistent coupled electron-phonon Bogoliubov problem (C.-Yu Wang et al.,
PRB 105, 174509 (2022)).

Because `dyntask`/`bectask` treat an existing DYN/BEC file as "already done"
and silently skip it — the hazard §4 records for task 205 — every method
needing dynamical matrices recomputes them in its own wiped subdirectory.
`get_superconductivity()` exists so the whole chain (205, 210, 220, 240, 245,
250, 260) runs **once** off a single set of dynamical matrices.

### Magnetism, GW, Wannier export, ultra-long-range

The **magnetic anisotropy energy** $\Delta E=\max_i E(\hat m_i)-\min_i E(\hat
m_i)$ is computed by brute force, one complete ground state per direction —
and `mae.f90` does not rotate the moment at all but rotates the *lattice* by
the inverse rotation while holding $\mathbf m$ along $+z$ with `cmagz`, which
keeps the collinear machinery valid and avoids re-deriving the symmetry group
for a tilted moment. It is also exactly the quantity §12's per-species
`soc_scale` is normally fitted against. The **exchange-correlation torque**
$\boldsymbol\tau=\int\mathbf m\times\mathbf B_{xc}\,d^3r$ measures how badly
the local approximations violate the zero-torque theorem (an exact functional's
$E_{xc}$ is invariant under a global spin rotation, so $\boldsymbol\tau$ must
vanish). **Spin spirals** come by two exactly complementary routes, and Elk's
naming is misleading: `spiralsc` (350-352) is the *supercell* method, while the
*supercell-free* generalised Bloch theorem is not a task code at all but the
`spinsprl`/`vqlss` input pair on an ordinary ground state — a translation
combined with a spin rotation about $z$ is a symmetry, so the two spinor
components carry Bloch vectors $\mathbf k\mp\mathbf q/2$ and an incommensurate
spiral fits in the chemical unit cell. That theorem holds only without
spin-orbit coupling, which is precisely what the supercell route buys back.
Sweeping the Bloch route gives $E(\mathbf q)-E(0)=-[J(\mathbf q)-J(0)]$, the
reciprocal-space counterpart of §29's real-space tensor.

**GW** (600-640) is Elk's finite-temperature $G_0W_0$ evaluated entirely on
the imaginary axis: $\Sigma$ is built from $W=\epsilon^{-1}v$ at Matsubara
frequencies $\omega_j=(2j+1)\pi/\beta$, whose spacing is set by `tempk` and
extent by `wmaxgw` — neither a physical temperature, just a way to keep the
frequency count tractable. Working there makes everything smooth, which is why
the real-axis $A=-\tfrac1\pi\mathrm{Im\,Tr}\,G$ arrives later and by analytic
continuation. The prerequisite chain was established from the Fortran, not the
manual: 610, 630 and 640 all read `GWSEFM.OUT` back, so re-finding the
interacting Fermi energy is *not* the cheap post-processing its one-number
output suggests; 620 is the exception, recomputing the entire inverse
dielectric matrix at every path point. **Wannier90 export** is wrapped
honestly: `.win` and `.eig` are complete, but `.amn`/`.mmn`/`.spn` need the
Wannier90 library for the neighbour-shell $\mathbf b$-vectors and this build
links `w90_stub.f90`, which aborts — the very gap `patches/0002` fills by
reimplementing the overlap export without the shell search (§13). Finally the
**ultra-long-range** family targets order incommensurate with the unit cell
without a supercell, by keeping fast and slow degrees of freedom in different
representations: orbitals stay in the unit-cell LAPW basis at k-points shifted
by a few ultracell $\boldsymbol\kappa$-points, while density, magnetisation and
potential acquire a slow dependence expanded in a handful of ultracell
reciprocal vectors, $n(\mathbf r,\mathbf R)=\sum_{\mathbf Q}n_{\mathbf
Q}(\mathbf r)e^{i\mathbf Q\cdot\mathbf R}$, so cost scales with the number of
$\mathbf Q$-points rather than the ultracell's atom count.

### Run shapes, and what is not covered

`_run_resumed()` prefixes task 1 and `run_tasks(resume=False)` prefixes task 0.
Both are **wrong** for the tasks that drive their own sequences of ground
states — 380, 390, 420/421 and 440 set `trdstate=.false.` or override
`tshift`/`ngridk`/`maxscl` before their own runs, so a prefixed ground state
would be discarded or computed under different settings. Those go through a
`_run_standalone()` helper: same wiped-subdirectory discipline, no task prefix.
Three families additionally need the *opposite* of a wipe, and each solved it
separately: molecular-dynamics restart (`restart=True`, since `readtimes`
reads `TIMESTEP.OUT` back), the GW/ULR dependents (`_run_dependent`, running in
place in the producing run's directory and replaying its blocks from a JSON
sidecar), and `continue_tddft_evolution()` (rewriting `elk.in` in place,
because the `_TD.OUT` eigenvectors and APPEND-mode observables must survive).
Three mechanisms for one idea is a known non-uniformity, recorded here rather
than refactored now.

Of the 148 codes `elk.f90` dispatches on, **2 are upstream no-ops** — 670 and
680 dispatch to commented-out calls — leaving 146 live, of which **143 are
emitted by a named method (97.9%)**. The three that are not:

- **task 2** (`geomopt` from atomic densities). The capability is wrapped:
  `get_relaxed()` emits task 3, the same subroutine reading `STATE.OUT`, which
  is what `_run_resumed` always supplies. Only the non-resume variant is
  unreachable, and it would be wrong there.
- **task 201** (`phononsc` resume) and **task 271** (`gndsteph` resume). Both
  resume from files a *previous* run of the same task left in the directory,
  which the wiped-subdirectory invariant cannot supply. Supporting them
  honestly needs a non-wiping run mode; tasks 200 and 270 are wrapped, so only
  the restarts are missing.

Coverage is measured against the dispatch, not claimed: a code counts only
when a named method actually places it in a task list it runs. `run_tasks()`
could always reach all 146, which is exactly what this section improves on.

## 33. The LAPW export: Elk's own first-variational eigenproblem, on demand

This is the one entry in the patch series that adds no physics. It exports
quantities Elk already computes so that the JAX port (`docs/jax_port.md`,
Workstream B) has a reference for them, and it exists because the port
otherwise had none: **nothing in `vendor/elk/src/` writes `apwalm`**, checked
by grep across the whole tree. Patch **0013** adds a `LAPW` query to the task
9002 session (`elkpy_lapwexport`, `EigenstateSession.lapw_problem(k)`), the
same pattern as 0004-0010 — no new file, no `elk.f90` dispatch arm, no
`Makefile` edit. There is no `physics.tex` part and no notebook, deliberately:
the formalism is already §14's, the LAPW generalised eigenproblem
$(H-\varepsilon O)v=0$, and this only hands its ingredients out.

At one arbitrary $k$-point it writes: the $\mathbf{G+k}$ set (`igpig`, `vgpl`,
`vgpc`, `gpc`); `apwalm`; the small derivative matrices $D_{ij}=d^{i-1}u_j/dr^{i-1}|_R$
that `match` inverts, per $(\ell,\alpha)$; the last `npapw` radial points of
each $u_{j\ell}$ and of the mesh they sit on, which is exactly what `polynm`
fits; $H$ and $O$; their interstitial contributions **on their own**; and
`evalfv`/`evecfv`. Elk's own `atposc`, `avec`, `omega`, `rmt`, `apword` travel
with it, because the caller must not regenerate any of them (see the traps
below).

### Three details that are load-bearing, not cosmetic

**Only the upper triangles of $H$ and $O$ are written**, because that is all
Elk fills. `olpistl`/`hmlistl` run `do i=1,j`, and every muffin-tin
contribution added afterwards (`olpaa`, `olpalo`, `olplolo` and the Hamiltonian
counterparts) does the same. The lower triangle of the allocated array is never
assigned, so exporting it would ship uninitialised memory. `parsers.eigenstates`
Hermitises. Getting the *order* wrong is the failure this actually hit: the
Fortran walks the triangle column by column, `numpy.triu_indices` walks it row
by row, and the first version used the latter — which does not return a subtly
wrong number, it returns a non-Hermitian matrix that `scipy.linalg.eigh`
refuses outright. Pinned by `test_upper_triangle_reconstruction_is_hermitian`.

**`tefvr` is forced to `.false.` while $H$ and $O$ are built**, then restored.
When a crystal has an inversion centre `symmetry.f90` leaves `tefvr` true, and
`olpaa`/`hmlaa` then accumulate through `rzmctmu` rather than `zmctmu`. That
routine's `dgemv('T',2*l,j,...,c(1,j),2)` strides by two over the complex array
as reals, so it adds **only the real part** of the muffin-tin APW-APW block —
correct for the real symmetric solver `eveqnfvr` that consumes it, and silently
wrong as an exported matrix. It stays Hermitian and positive definite either
way, so nothing short of comparing eigenvalues catches it. Bulk silicon has an
inversion centre, so this is the default case, not an exotic one.

**The derivative matrices are recomputed, not captured.** `zgesv` overwrites
its coefficient matrix in place, so `match` no longer holds $D$ when it
returns. The four lines that build it are copied from `match.f90` — the same
kind of copy patch 0008 makes of `getevecfv.f90`'s symmetry transformation, and
the same maintenance cost on an upstream bump. Note `match` skips that
construction entirely in its `omax == 1` fast path, dividing by
`apwfr(nr,1,1,l,ias)` directly; the general construction reduces to exactly
that at `ord = 1`.

### What it settles for the port

**Phase 0c's forward half, closed against Elk itself.** `src/elkjax/lapw.py`'s
transcription of `match` had only its own defining equation to check against,
and `docs/jax_port_phase0.md` records why that is necessary but not sufficient:
dropping `genylmv`'s $4\pi(-i)^\ell$ prefactor multiplies each $\ell$ block by a
constant and still passes the `dmatch` identity at 7e-16, because a constant
commutes with $\partial/\partial\mathbf r_\alpha$. Element-wise agreement with
Elk's array is blind to none of that. Measured on bulk Si at a generic
$k$-point: **1.9e-15** relative in `match`'s `omax == 1` branch and **8.0e-13**
in the general linear-solve branch (reached through a species file the test
generates with `apword = 2`, since every species file Elk ships sets
`apword = 1` and so never reaches it).

**$\kappa(O)$ for a real LAPW overlap**, which the study's §8(b) tolerance
$\epsilon\,\kappa(S)\,\lVert H\rVert$ depends on and which had only ever been
measured on synthetic matrices. The numbers are in `docs/jax_port_phase0.md`;
the finding that matters is that the cheap Cholesky-diagonal estimate §8(b)
proposes is low by between 300x and 40,000x on real data, against 140x on the
synthetic case — and worse, it is nearly *constant* while the truth varies by
two orders, so it carries no signal at all. It is a *lower* bound, which is the
dangerous direction for a tolerance meant to bound from above.

The response is plain text over the session's pipe and is $O(n_{\rm mat}^2)$
tokens, so its cost tracks the matrix rather than the physics: 0.2 s at
$n_{\rm mat}=161$ (bulk Si, `rgkmax=7`), 17 s and ~220 MB of text at
$n_{\rm mat}=1402$ (monolayer h-BN with 30 Bohr of vacuum). That is the right
trade for a reference at a handful of $k$-points and the wrong one for a bulk
export; if the port ever needs the latter, the fix is a binary side file, not a
wider pipe.

### Traps for a caller, all of them measured

- **Use Elk's `atposc`, never the input positions.** `tshift` is on by default
  and moves the origin onto the inversion centre; for diamond silicon the two
  atoms come back at $\pm(3.8475,3.8475,3.8475)$ rather than $(0,0,0)$ and
  $(1/4,1/4,1/4)$. The structure factor uses Elk's frame. Same trap as §28 and
  §31, in a third guise.
- **Use Elk's `vgpc`/`gpc`, never a regenerated $\mathbf{G+k}$ set.** An
  element-wise comparison of `apwalm` needs identical ordering, and
  `gengkvec`'s ordering is not something to rediscover. Checking that a
  regenerated set matches *as a set* is a separate question.
- `rmt` is not the species file's value: `checkmt` shrinks it (2.1964 against
  the file's 2.2 for silicon here). The export carries the value actually used.

### Verification

`tests/test_calculation_lapw_export.py`, 6 tests run twice (`apword` 1 and 2).
The one that validates the export as a whole is
`test_exported_matrices_reproduce_elks_own_eigenvalues`: `scipy.linalg.eigh` on
the parsed, Hermitised $H$ and $O$ returns Elk's own `evalfv` to **1.7e-15**
(2.7e-15 at `apword=2`). That is not a tautology — `evalfv` comes from
`eveqnfv` through Elk's configured path, *before* the `tefvr` override — and it
pins the column-major convention, the triangle fill and the whole assembly at
once. Beside it: the APW-APW block of $O$ minus its interstitial part equals
$\sum_{\ell m,i_o}\overline{A_{i}}A_{j}$ (that is `olpaa`'s `zmctmu` written
out) to 3.6e-15, tying `apwalm` to $O$ through Elk's own assembly so a packing
error would have to be shared by both to survive; $D$ against an independent
`numpy` polynomial fit to the exported radial tails, 6.1e-14, which checks the
tail alignment Phase 1 will build $D$ from; the $\mathbf{G+k}$ set against an
independent enumeration of every integer triple with
$|\mathbf{G+k}|<{\tt rgkmax}/\min({\tt rmt})$, which agrees exactly and is what
`elkjax.lapw.gkvectors` assumes; the exported `evecfv` against the exported $H$
and $O$ ($HV=OV\varepsilon$ to 1.5e-15, $V^\dagger OV=\mathbb 1$ to 5.8e-15,
and the occupied-subspace projector $VV^\dagger O$ idempotent to 1.3e-14 —
which is the gauge-invariant object Phase 1's forward criterion is stated on,
and which, since `evecfv` comes from `eveqnfvr` and never sees the complex
matrices, is a second and independent confirmation that the `tefvr` override
exports the right ones); and the two-atom separation recovered from `atposc`
modulo a lattice vector. `parsers.eigenstates` has its own token-level round
trip needing no binary.

Run on monolayer h-BN as well (`elkjax.phase0b_overlap`), the export
self-check holds at 3.4e-14 on a **two-species** cell — the only one in this
work — which exercises `idxis`'s indexing into `rmt`/`nrmt`/`apword` and
`apword`'s own per-$\ell$ variation. Bulk silicon cannot.

### Patch 0014: the radial integrals, and what they unlock

Patch 0013 exports the *coefficients*; it does not export what they are
contracted with. Assembling $H$ and $O$ needs, in addition, the muffin-tin
radial integrals `oalo`, `ololo` (`olprad.f90`) and `haa`, `hloa`, `hlolo`
(`hmlrad.f90`), the local-orbital bookkeeping `nlorb`/`lorbl`/`idxlo`
(`genidxlo.f90`), and the complex Gaunt array `gntyry`. Patch **0014** appends
all of them to the same `LAPW` response, behind a header of their own so a
caller that knows only 0013's fields still parses correctly up to that point.
No new subroutine, no new call site — the smallest possible extension.

**The interstitial ingredients are deliberately not written.** `cfunig` and
`vsig` would be the remaining inputs of `olpistl`/`hmlistl`, but 0013 already
writes `hi` and `oi`, which are those contributions *assembled*; and `vsig` is
built from the interstitial Kohn-Sham potential, which is an output of a later
phase of the port than the one this serves. There is nothing to compare a
half-built interstitial block against.

`gntyry` dominates the response — $l_{\max}^{\rm o}=6$, $l_{\max}^{\rm
apw}=8$ gives $49\times81\times81$ complex numbers — and it is
$k$-independent, so writing it per query is waste. It is written anyway rather
than regenerated by the caller, for the reason the whole export exists: its
convention is $\langle Y_{\ell_1m_1}|R_{\ell_2m_2}|Y_{\ell_3m_3}\rangle$,
with a **real** harmonic in the middle slot (the muffin-tin potential is stored
in a real basis, the wavefunctions in a complex one), and that is exactly the
kind of thing a re-derivation gets subtly wrong while still producing a
Hermitian matrix. If the response size ever matters, the fix is a separate
one-shot query, not a regeneration.

**One trap this export carries out of `hmlrad.f90`, and it is not 0013's.**
`hlolo`'s $\ell_2=0$ element is

$$\int u^{\rm lo}_i(r)\,\bigl(\hat H u^{\rm lo}_j(r)\bigr)\,r^2\,dr,$$

built with no symmetrisation over $(i,j)$ and no kinetic surface term. Contrast
`haa`, whose $\ell_2=0$ counterpart is explicitly averaged over the two
orderings, carries the surface term, and has its transpose explicitly assigned
(`haa(lm2,io,l1,jo,l3) = haa(lm2,jo,l3,io,l1)`). `hmllolo` consequently runs
`do jlo; do ilo = 1, jlo` with an `if (i > j) cycle` and evaluates each pair in
**one order only**, Hermitising the rest — and because `genidxlo` numbers
columns in increasing $(i_{\rm lo}, \ell m)$, that set is exactly the matrix
upper triangle. A consumer that uses both halves of the array gets a plausible,
Hermitian, wrong matrix. Measured on the nitrogen of monolayer h-BN, which
carries two $\ell=0$ local orbitals: the two orderings differ by
$1.3\times10^{-2}$ Ha, which reaches the assembled $H$ as $3.7\times10^{-3}$ Ha
and `evalfv` as $4.3\times10^{-7}$ Ha. Bulk silicon cannot see it — its local
orbitals are one s and one p, so no pair shares an $\ell$.

### Patch 0015: the potential behind the radial integrals

Patch 0014 exports the radial integrals; patch **0015** exports what *they*
are built from, which is the last thing standing between the port and a
spectrum that is a function of the Kohn-Sham potential rather than of a set of
imported numbers. It appends the muffin-tin potential `vsmt`, the radial mesh
`rlmt` and its quadrature weights `wr2mt`, the APW and local-orbital
linearisation energies (`apwe`, `lorbe`) with their derivative orders
(`apwdm`, `lorbdm`, `deapw`, `delorb`) and energy ordering (`idxelo`), and
`apwfr`, `apwdfr` and `lofr` **in full** — patch 0013 wrote only the last
`npapw` points of `apwfr`, which is all `match` needs.

Two of those choices are deliberate. `wr2mt` is exported rather than rebuilt
from the mesh: it is a Simpson-like weight array from `wsplint`, not $r^2dr$,
and a caller that rebuilt it would agree with Elk to the quadrature's own
truncation error instead of to roundoff. And `vsmt` is written **exactly as
Elk packs it** — $l_{\max}^{\rm i}$ harmonics per radial point over the inner
region, $l_{\max}^{\rm o}$ over the outer one, radial-point-slowest — because
that packing, and not the values, is what a transcription of `hmlrad` gets
wrong; unpacking it in Fortran would hide the very thing being checked.

**The patch also calls `genapwlofr` before building anything, and that is
load-bearing.** `gndstate.f90` calls `genapwlofr` at the *top* of an SCF
iteration and `potks`/`mixerifc` at the *bottom*, so when the loop exits,
`apwfr`, `lofr` and the five radial integrals belong to the potential of the
**previous** iteration while `vsmt` is the current one. The export was
therefore internally inconsistent by exactly one mixing step. It went
unnoticed for as long as the potential was not exported — nothing else in the
response depends on it — and the moment it is, the discrepancy is
indistinguishable from a transcription bug. Measured on bulk Si before the
fix: every radial integral that does *not* touch the potential (the
$\ell_2=0$ elements, which are $\langle u|\hat Hu\rangle$) agreed to
$2.6\times10^{-16}$ relative, while every one that does was off by
$3\times10^{-10}$. That split is what identified the cause; a single
aggregate number would have looked like a subtle indexing error.

Regenerating leaves the exported arrays consistent with each other by
construction, and `evalfv` is still computed by `eveqnfv` through Elk's own
configured path afterwards, so diagonalising the exported $H$, $O$ and
recovering it remains a real check. `linengy` is deliberately *not* called:
the linearisation energies are exported as they stand and held fixed, which is
also what Elk's own forces assume. The side effect is that a `LAPW` query
updates those global arrays for the rest of the session — idempotent, and what
the next SCF iteration would have done anyway.

One knock-on worth recording, because it is a general lesson about what to
pin. The regeneration moves Elk's own matrices by $\sim3\times10^{-10}$
relative, and that retuned a constant in the smearing suite by 13x: the naive
`eigh` rule's error at graphene's $K$ went from
$2.8\times10^{-10},2.8\times10^{-9},2.8\times10^{-8}$ to
$2.2\times10^{-11},2.2\times10^{-10},3.1\times10^{-9}$ across the three
smearing widths, while the Dirac splitting itself moved only in its eighth
digit ($3.3712177\times10^{-7}\to3.3712180\times10^{-7}$ Ha) and the safe
rule stayed on its $3$–$6\times10^{-14}$ floor. Measured both ways, with the
call switched off and on, rather than inferred. The mechanism is that the
naive rule's error is set by $\lVert A-A^\dagger\rVert/\delta\lambda$ with
$A=v^\dagger\delta Hv$, and inside a pair split at the assembly's own roundoff
the eigensolver's choice of basis is free to rotate. So the separation between
the two rules is a mechanism, not a number: the test now asserts growth in the
smearing width and a widening separation instead of a fixed factor.

### Patches 0016-0018: the ground state, the Poisson inputs, and the symmetrisation operator

Patch **0016** adds a `GROUNDSTATE` query — the converged density and
potentials on the grids Elk holds them on, taking **no k-point**, which is why
it is a query of its own rather than more fields on `LAPW`. Muffin tin, in
Elk's own packing: `rhomt`, `vclmt`, `vxcmt`, `exmt`, `ecmt`, the radial mesh
and weights, and the four spherical-harmonic transform matrices
(`rbshti`/`rfshti`/`rbshto`/`rfshto`). Interstitial: `rhoir`, `vclir`,
`vxcir`, `exir`, `ecir`, `vsir`, `cfunir`, plus `cfunig`, `vsig` and the
reciprocal-lattice set. Two contents are what a transcription must reproduce
rather than what it might expect — `rhomt` includes the core density and
`vclmt` includes the nuclear $-Z/r$ — and one is a bound rather than a value:
`vsig` is allocated to `ngvc`, not `ngvec`, because `genvsig` fills it from
the coarse grid, so writing `ngvec` of them reads past the end of the array.

Patch **0017** adds four more fields, and the interesting part is what it does
*not* add. The Weinert Poisson solve (`potcoul` → `genzvclmt` → `zpotclmt`,
plus `zpotcoul`) consumes eleven arrays; seven of them are rebuilt in
`src/elkjax/poisson.py` instead. $r^\ell$ and $R^\ell$ are the mesh; $4\pi/G^2$
is `gc`; and $Y_{\ell m}(\hat G)$, $e^{i\mathbf G\cdot\mathbf r_\alpha}$ and
$j_\ell(GR)$ are `genylmv`, `gensfacgp` and `sbessel`, all three already
transcribed in `elkjax.lapw` and already pinned against Elk element-wise by
patch 0013's own checks. Exporting `ylmg` alone would be about 38 MB of text
to avoid reusing code that is verified.

Patch 0017 also carries `spzn` and `energy.f90`'s own thirteen converged
scalars — `evalsum` through `engytot`. Those are a *reference*, not an
ingredient, and they are exported for a specific reason: a total-energy
transcription checked against `INFO.OUT` is limited to its print width, and a
total that agrees to $10^{-8}$ says nothing about which convention is right.
Term by term at full precision, it does (`docs/jax_port_phase2.md` §2f).

What is left of the ingredients is the four that cannot be rebuilt: `wprmt` (`wsplint`'s
cumulative spline weights — **not** `wr2mt`, and no closed form worth
retyping), `vcln` (the nuclear potential, which `potcoul` adds to the $l=0$
channel *before* `zpotcoul` reads the sphere-boundary multipoles, so a
transcription that omits it gets every $q_{\ell m}$ wrong), `npsd`/`lnpsd`, and
`atposc`, which no other query carries.

The measurements are in `docs/jax_port_phase2.md` §2e: `vclir` to 1.6e-15
relative and `vclmt` to 4e-20 ($\ell=0$) and 7e-14 ($\ell>0$) on two
structures, with the monopole identity $\sqrt{4\pi}q_{00}=N_{\rm MT}-Z$
recovering the nuclear charges as exact integers from an independent code
path.

Patch **0018** makes the same call one step further. `potxc.f90` symmetrises
the muffin-tin $v_{xc}$ and not its energy densities, so a pointwise
transcription of the functional reproduces `exmt`/`ecmt` exactly and misses
`vxcmt` by $1.2\times10^{-4}$ relative. Closing that needs `symrfmt`, and
transcribing `symrfmt` means transcribing `rotrflm` — `roteuler`'s Euler-angle
extraction, a real-harmonic Wigner-$D$ construction, and the improper-rotation
$(-1)^\ell$ branch — **whose only consumer inside Elk is `symrfmt` itself**, so
a re-derivation would have no independent check except agreement with the thing
it replaces. It would also drag in `ieqatom`, `tfeqat` and the *inverse*
lattice rotation of the rotate-into-equivalent-atoms loop.

So `elkpy_gsexport` calls upstream `symrfmt` on basis vectors and writes back
the linear operator: one $l_{\max}^{\rm o}$-square matrix per ordered atom
pair, which is the whole of it, a rotation being diagonal in the radial index
and not mixing $\ell$. None of Elk's symmetry bookkeeping is transcribed, so
none of it can be got wrong on this side; what is tested is the operator's
application. Applying it takes the `vxcmt` gap to $6.4\times10^{-14}$
(§2g).


### What the port does with it

`src/elkjax/hamiltonian.py` transcribes `olpfv`/`hmlfv`'s muffin-tin half and
`tests/test_calculation_lapw_assembly.py` compares its six blocks against
Elk's own, **separately**, so a failure names one upstream routine rather than
"$H$ is wrong". Every block agrees to machine precision on bulk Si at
`apword` 1 and 2 and on monolayer h-BN, and diagonalising the assembled pair
returns Elk's `evalfv` to 9e-15 Ha. The measurements, the three fixtures and
what each one alone can catch are in `docs/jax_port_phase1.md`.


### Patch 0024: the same exports at iteration zero

Everything above describes a **converged** calculation. `eigenstate_session()`
copies `STATE.OUT` into a fresh subdirectory and runs task 1 before task 9002,
so `readstate` fixes the potential, the density and the eigenvectors at Elk's
own answer. That was the right default while the port was checking
transcriptions against a reference — but it makes a self-consistent loop
untestable in the only way that matters. Elk's converged potential is a fixed
point of the port's map (`docs/jax_port_phase3.md` §3b), so a loop started
there does nothing, and the alternative was to perturb it by hand: a start
still built out of the answer.

Patch **0024** adds task **9006** (`src/elkpy_initstate.f90`,
`Calculation.initial_state_session()`), which is `gndstate.f90`'s *other*
branch. Where task 1 takes `trdstate=.true.` and calls `readstate`, this takes
`trdstate=.false.`:

```
init0; init1; rhoinit; maginit; potks(.true.); genvsig
```

and then the top of `gndstate`'s first self-consistent iteration:

```
gencore; linengy; genapwlofr; gensocfr; genevfsv; occupy
```

and stops. No `rhomag`, no `potks` on a new density, no `mixerifc`. Every array
the 9002 queries read then holds its iteration-zero value: the density is the
superposition of free atomic densities `rhoinit` builds, the potential is that
density's Kohn-Sham potential, and — because no mixing has happened — the two
halves of the potential are on the *same* side of `mixerifc` for once, so the
"Elk mixes in the middle of its own iteration" trap does not apply here.

Nothing in the new file is new physics: every line is an upstream call, in
`gndstate`'s own order, with no arguments. What it costs on an upstream bump is
that the sequence is *mirrored* rather than called — `gndstate`'s
initialisation is not a subroutine — so re-checking it means diffing that
branch and the head of its SCF loop. That is the same exposure patch 0011
already carries, and it is the reason the file documents the sequence it
mirrors rather than just executing it.

The alternative was transcribing `init0`/`init1`/`rhoinit` into Python. That is
not a smaller job: `rhoinit` superposes the free-atom densities that
`allatoms` → `atom.f90` produces, i.e. a full radial Dirac solver for every
species, plus the whole of Elk's grid, symmetry and species bookkeeping. None
of it is a functional of the density — it is identical at every iteration and
for every potential — so none of it is on the path a gradient would take.
Exporting it is the cheap and correct move; §8's "prefer Elk's own export
routes" is exactly this case.

**One extra export travels with the patch**, appended to `elkpy_lapwexport`:
`autolinengy` and the per-orbital `apwve`/`lorbve` flags. The port freezes the
linearisation energies across its loop, and whether that is exact or an
approximation is decided entirely by those flags — `linengy` calls `findband`
only where they are true, and otherwise leaves `apwe`/`lorbe` at the species
file's own `apwe0`/`lorbe0` for the whole run. `apwe` alone cannot say which
happened: a searched energy and a default one are the same kind of number. Elk
ships every stock species file with the flags false and `autolinengy` off, so
for those the freeze is exact and `elkjax.driver.check_linearisation_frozen()`
passes; where it is not, that function raises rather than returning a plausible
total energy.

# 34. `STATE.OUT`: the binary layout and the conventions inside it

elkpy does not read `STATE.OUT` — `ensure_ground_state()` only checks that it
exists, and every task that needs the density hands the file back to Elk's own
`readstate`. This section exists anyway, because the *conventions* it records are
what a reader outside elkpy needs, and every one of them has a plausible wrong
answer that produces a smooth, believable, incorrect density. Everything below is
read off `vendor/elk/src/` at Elk 11.0.2, with the file and line, and three
committed fixtures make it executable rather than remembered:

| | | closes |
|---|---|---|
| `tests/fixtures/h_sc` | simple cubic H, one atom | the baseline: no core at all, so all-electron $=$ valence |
| `tests/fixtures/c_diamond` | diamond, two C | a frozen core inside `rhomt`; the **atom-inner** half of the `ias` order; `tshift` not a no-op; GGA (`xcgrad = 1`) |
| `tests/fixtures/sic_zb` | 3C-SiC, one C + one Si | `natmtot` $\ne$ `natoms(1)`; `nrmt(is)` $\ne$ `nrmtmax`; the **species-outer** half |

`c_diamond` and `sic_zb` each carry a second state file as well —
`STATE_INIT.OUT`, the same calculation stopped at the top of Elk's first
iteration; see below. `tests/test_state_fixture.py` asserts all of it.

Written for the sibling project reconstructing $\rho(\mathbf{r})$ from a converged
Elk state and seeding a plane-wave SCF with it, but it is the standing reference.

## The unit is a run directory, not a file

`STATE.OUT` carries no lattice vectors, no atomic positions and no species names.
It carries the radial meshes, the grid sizes and the functions. So `avec` and
`atposl` have to come from `GEOMETRY.OUT` alongside it —
`parsers.geometry.parse_last_geometry()` reads that file (it is `writegeom.f90`
output, same block syntax as `elk.in`, which `inputfile.read_blocks()` handles
generically).

Not from `elk.in`. Elk's `tshift` defaults to `.true.` and moves the origin onto
the inversion centre; `GEOMETRY.OUT` is written after that shift and `elk.in`'s
`atoms` block is not. This is the same trap as §28 and §31 one layer down.

Measured, on `c_diamond`. Diamond's inversion centre is the bond midpoint at
$(\tfrac18,\tfrac18,\tfrac18)$, so with the default the same input gives atoms
at $\pm(\tfrac38,\tfrac38,\tfrac38)$ — not $(0,0,0)$ and
$(\tfrac14,\tfrac14,\tfrac14)$, and both of them moved. Elk then reports
`Crystal has inversion symmetry` and switches to the real symmetric eigensolver;
pinning the frame costs that and is worth it. In `h_sc` the same line is a no-op,
which is exactly why one fixture was not enough.

$r_{\rm MT}$ needs neither file. `genrmesh.f90:56-59` builds the radial mesh as
$r_i = r_{\rm min}\exp\!\big[(i-1)\log(r_{\rm MT}/r_{\rm min})/(n_r-1)\big]$, so
$r_{\rm sp}(n_{r{\rm MT}}) = r_{\rm MT}$ up to the round-off of that exp/log round
trip (1.4000000000000004 on the fixture) — and it is the value *after* `autormt`
has adjusted it, which the species file's own `rmt` is not.

## Header records

In `writestate.f90` order: `version` (3 int32 — the same tuple as
`spec.ELK_VERSION`), `spinpol`, `nspecies`, `lmmaxo`, `nrmtmax`, `nrcmtmax`; then
per species `natoms`, `nrmt`, `rsp(1:nrmt)`, `nrcmt`, `rcmt(1:nrcmt)`; then
`ngridg`, `ngvec`, `ndmag`, `nspinor`, `fsmtype`, `ftmtype`, `dftu`, `lmmaxdm`,
`xcgrad`, `efermi`, `dlefe`.

`efermi` being in there (`writestate.f90:49`) means the Fermi level needs no
`INFO.OUT` and no `EFERMI.OUT`. `readstate.f90:126` gates it on version $\ge$ 9.6
and falls back to `readefm` below that; `dlefe` on $\ge$ 10.7.

`nrmti` and `lmmaxi` are **not** written, and are not needed — see the muffin-tin
packing below.

## The muffin-tin functions

`rhomt` is the real-spherical-harmonic expansion of the density inside each
sphere,

$$\rho(\mathbf{r}) = \sum_{lm} \rho_{lm}(r)\, R_{lm}(\hat{\mathbf{r}}),$$

and the array holds $\rho_{lm}(r)$ itself, **not** $r^2\rho_{lm}(r)$. The
$r^2$ lives in the integration weight: `charge.f90:30` integrates with `wr2mt`,
which `genrmesh.f90:66-68` builds as spline weights and *then* multiplies by
$r^2$; `rfmtint.f90`'s own header states it returns
$4\pi Y_{00}\int f_{00}(r)r^2\,dr$.

The real spherical harmonics are Elk's `genrlmv.f90` convention,

$$R_{lm} = \begin{cases}\sqrt{2}\,{\rm Re}\,Y_{lm} & m>0\\ \sqrt{2}\,{\rm Im}\,Y_{lm} & m<0\\ {\rm Re}\,Y_{l0} & m=0\end{cases}$$

with Condon-Shortley inside $Y_{lm}$, packed at $j = l(l+1)+m+1$ with $m$ ascending
from $-l$ to $+l$ and $l$ running contiguously. At the sphere centre only $l=0$
survives, so $\rho = \rho_{00}\,y_{00}$ with $y_{00} = 1/\sqrt{4\pi}$.

**The file is unpacked.** Internally Elk stores only `lmmaxi` components on the
inner part of the muffin tin ($r$ below `fracinr`$\cdot r_{\rm MT}$) and `lmmaxo`
outside it. `writestate.f90:58` calls `rfmtpack(.false., ...)`, whose else-branch
zeros the $lm >$ `lmmaxi` entries for $ir \le$ `nri` (`rfmtpack.f90`), so what
reaches the file is a plain `(lmmaxo, nrmtmax, natmtot)` block whose inner region
is genuinely zero beyond `lmmaxi`. That is the "unpacked to maintain backward
compatibility" comment at `writestate.f90:52`, and it is why the reader needs
neither `nrmti` nor `lmmaxi`.

Atom index `ias` runs species-outer, atom-inner (`init0.f90:78-91`) — the same
order `parsers.info.parse_charges()` returns its per-atom charges in.

Getting that order wrong is not always detectable. In `c_diamond` the two
carbons sit at the ends of a bond whose midpoint is an inversion centre, so
their $\rho_{lm}$ differ by $(-1)^l$ — measured on the file, $l = 0, 4, 6$
agree to 1e-14 and $l = 3$ is exactly opposite, with $l = 1, 2, 5$ zero by the
Td site symmetry. The $l = 0$ nucleus check therefore **cannot** see a swapped
`ias` there; only the odd-$l$ channels can. `sic_zb`'s two columns share no
symmetry relation at all (2095 e/Bohr³ at the Si nucleus against 130 at the C),
so a swap is visible in $l=0$ alone.

The whole chain has one cheap end-to-end check, and `tests/test_state_fixture.py`
runs it. `rfpts` clamps $r$ up to $r_{\rm sp}(1)$ at the nucleus, and its `poly4`
window there starts at $ir_0=1$, so `RHO3D.OUT`'s value at the origin is *exactly*
$\rho_{00}(r_1)\,y_{00}$ read out of `STATE.OUT`. On the fixture both are
0.2859303012, which pins the record layout, the $lm$-fastest reshape, the $y_{00}$
factor and $\rho$-not-$r^2\rho$ in one number. (That number is also physics: the
density at a hydrogen nucleus is finite — the 1s wavefunction has a cusp, not a
pole — and it sits just under the free-atom $1/\pi = 0.3183$, the cell being
compressed at $a = 3$ Bohr.)

## The interstitial function

`rhoir` is the smooth density on the `ngridg` FFT grid, extended over the whole
cell including the inside of the muffin tins. It is **raw**: not multiplied by the
characteristic function, not by $\Omega$, not by anything. `charge.f90:33`
multiplies `cfunir` in explicitly at integration time,
$\text{chgir} = (\Omega/N_{\rm grid})\sum_i \rho_{\rm ir}(i)\,\Theta(i)$.

## Magnetisation

`ndmag = 1` stores $m_z$ at index **1**, not 3. `ndmag = 3` is Cartesian
$(m_x, m_y, m_z)$. From `rhomagk.f90`'s contained `rmk1`/`rmk2`, with $\psi_\uparrow$
and $\psi_\downarrow$ the two spinor components,

$$m_x = 2\,{\rm Re}(\psi_\uparrow^*\psi_\downarrow),\quad m_y = 2\,{\rm Im}(\psi_\uparrow^*\psi_\downarrow),\quad m_z = |\psi_\uparrow|^2 - |\psi_\downarrow|^2,$$

which is $\mathrm{Tr}[\rho\,\boldsymbol{\sigma}]$ in the standard convention — no
hidden sign on the $y$ component, which was the plausible wrong answer. It is a
spin density (up minus down), so it points along the majority spin, opposite the
magnetic moment vector.

## Reconstruction: what Elk itself does

Tasks 31/32/33 all end in `plot3d.f90` $\to$ `rfpts.f90`, which is the reference
implementation of "evaluate the density at an arbitrary point":

1. `findmtpt.f90` folds the lattice point into $[0,1)$ with `r3frac`, then tests
   all 27 periodic images against a **sharp** $r^2 < r_{\rm MT}^2$; first match
   wins. Skip the images and a point near a cell face falls into the interstitial
   when it should not.
2. Inside a sphere: a 4-point Lagrange interpolation (`poly4`, contained in
   `rfpts.f90`) on the log mesh, over a window $ir_0 = ir-2$ clamped at both ends,
   with $r$ clamped up to $r_{\rm sp}(1)$ at the nucleus. A cubic spline instead
   leaves an interpolation-level residual against `RHO3D.OUT` that is not a bug.
3. Outside: FFT `rhoir` to $G$-space and sum $\sum_{\mathbf{G}}\tilde\rho(\mathbf{G})
   e^{i\mathbf{G}\cdot\mathbf{r}}$ over the first **`ngvec`** vectors, not all
   `ngtot` — the FFT-box corners outside the $|G| <$ `gmaxvr` sphere are dropped.

`cfunir` appears nowhere in this. It is needed only to reproduce `chgir`
(`charge.f90`) and `momir` (`moment.f90`), it is not in `STATE.OUT`, and it does
not need to be: `gencfun.f90` gives it in closed form,

$$\tilde\Theta_i(G) = \frac{4\pi R_i^3}{\Omega}\frac{j_1(GR_i)}{GR_i}\ (0<G\le G_{\max}),\qquad \tilde\Theta_i(0) = \frac{4\pi R_i^3}{3\Omega},$$
$$\tilde\Theta(\mathbf{G}) = \delta_{\mathbf{G},0} - \sum_{ij} e^{-i\mathbf{G}\cdot\mathbf{r}_{ij}}\,\tilde\Theta_i(G),$$

from $r_{\rm MT}$, $\Omega$, the atomic positions and the $G$ list — all of which
are already to hand.

The consequence is the one that costs real time if it is missed: **a sharp-step
reconstruction must not be validated against Elk's printed `chgir`.** They are
different quantities.

That has since been measured cleanly, by the sibling project, on `h_sc`. Building
`cfunir` from the closed form above reproduces the printed `chgir` **exactly**
(0.38742380044 against 0.3874238004), which isolates the remaining difference:
the sharp in-or-out characteristic function and Elk's Fourier-truncated one
differ by 2.7644e-3, or **0.71% of the interstitial charge**. The earlier
0.38466 quoted here was the same effect seen through a $12^3$ grid staircasing
the sphere; with `cfunir` in closed form the staircase is gone and what is left
is the Gibbs difference alone.

Two further numbers from the same reader, both worth having: pointwise against
`RHO3D.OUT` at all 4096 points it agrees to 4.52e-11 absolute, which is that
file's own `G18.10` print floor rather than interpolation error — so
transcribing `poly4` (rather than substituting a spline) is what makes the
caution above moot. And reintegrating `chgmt` with Elk's own `wsplint` weights
gives 0.61257620 against the printed 0.6125761996; the 9e-6 residual quoted
below is Simpson, exactly as stated.

That 4.52e-11 is a property of `h_sc`, not of the method, and the same reader on
`sic_zb` separates three different floors that are easy to run together. Over the
same 4096 points the residuals there are 2.4e-10 relative in the median, 7.2e-9
relative at worst, and 3.5e-7 absolute at worst — three numbers, three causes, and
only the middle one is about the reconstruction's inputs at all.

- **The median is the printed value.** Ten significant digits is 5e-11 to 5e-10 of
  relative rounding, so 2.4e-10 is the file's own floor and nothing else. `h_sc`
  sits on the same floor: 7.3e-11 relative in the median, 4.5e-11 absolute.
  The two fixtures agree here, and a reader that reaches this is done.
- **The worst relative point is the printed *coordinates*.** `plot3d.f90` writes
  the coordinates in the same `(7G18.10)` as the value, so a coordinate of order a
  few bohr carries up to 5e-10 bohr of rounding. The 7.2e-9 points are not the ones
  inside `nri`, where the coarse `lmmaxi` expansion is the natural suspect; they sit
  at $r \approx 0.36$ bohr from a carbon — the first grid point off it — where
  $|\nabla\rho| \approx 20$ e/bohr$^4$. Two rounded columns move the density by
  1.5e-8 there on their own, against 1.48e-8 observed. `h_sc` has no such tail at
  all, by arithmetic: $a = 3$ bohr, so every grid coordinate is a multiple of
  $3/16 = 0.1875$ and prints exactly in ten digits, while `sic_zb`'s
  $4.119225/16 = 0.2574515625$ needs eleven from the fifth multiple on.
- **The worst absolute point is the printed value again, at the silicon nucleus.**
  It is line 1094 of the fixture, Cartesian $(2.0596125)^3$ — lattice $(1/4,1/4,1/4)$,
  an exact grid point whose coordinates also print exactly. `rfpts` clamps $r$ there,
  so it is not an interpolation either. $\rho = 2094.517773$ in ten digits has a
  print floor of 5e-7 by itself, and the observed 3.5e-7 is 1.7e-10 relative — the
  median floor, seen through a density four orders of magnitude larger.

So: tightening a pointwise comparison past the median needs more digits in the
file, not a better transcription; expect the SiC tail on any cell whose lattice
constant is not short in decimal; and check the printed coordinates before
suspecting the interpolation.

## `rhonorm`, and which printed charge is safe to check against

Which of Elk's printed charges describes the density in `STATE.OUT` is decided by
one routine that is easy to miss. `rhomag.f90` calls `charge`, then `rhonorm`
(`trhonorm` is on by default). `rhonorm.f90` adds a uniform constant to `rhoir`
and to the $l=0$ channel of every `rhomt` so the total charge comes out right —
the muffin-tin density is built on a $(\theta,\phi)$ grid and transformed to
spherical harmonics, and that loses a little charge. It then updates `chgmt` and
`chgmttot`, and sets $\text{chgir} = \text{chgtot} - \text{chgmttot}$. It does
**not** update `chgcalc`.

So on the fixture:

- `chgmt` = 0.6125761996 is **post-shift**, and describes the `rhomt` the file
  holds. Reintegrating the $l=0$ channel of `STATE.OUT` recovers it to 9e-6, which
  is Simpson-on-the-log-mesh against Elk's spline weights and nothing else (2.3e-5
  on `c_diamond`, 5e-5 on `sic_zb` — same quadrature, larger integrand). This is
  the integrated check a reader can trust, and with Elk's own `wsplint` weights it
  closes to the printed digits.
- `chgmt` + `chgir` = 1 exactly, by construction.
- `total calculated charge` = 1.000739542 and `error` = 7.4e-4 are **pre-shift**.
  That number is what `rhonorm` corrected, *not* a floor under a reintegration
  check — an easy thing to get backwards, and it would send someone hunting a 1e-3
  discrepancy that is not in their code.
- `chgir` = 0.3874238004 is post-shift, hence a residual defined to close the sum,
  and before that a smooth-`cfunir` integral. Neither is the sharp-boundary sum,
  which measures 0.38466 here.

`tests/test_state_fixture.py` asserts all four.

## Traps in the binary layout

- **One Fortran record holds two arrays.** `write(100) rfmt, rhoir` is a single
  sequential record with both concatenated; likewise `(rvfmt, magir)`,
  `(rvfmt, bxcir)`, `(rvfcmt, bsir)`. A one-array-per-record reader desyncs at the
  first density record and it looks like a byte-order problem.
- **`bsmt`/`bsir` use the COARSE radial mesh** — `writestate.f90` packs them with
  `nrcmt`/`nrcmti` into `(lmmaxo, nrcmtmax, natmtot, ndmag)` while every other
  muffin-tin array is `nrmtmax`.
- **`natmtot` is the sum of `natoms` over all species**, not `natoms(1)`. A reader
  that writes the latter reads a fraction of the muffin-tin block and then desyncs
  on `rhoir`. `sic_zb` is the only fixture that can fail this.
- **Padding is uninitialised, not zero.** `rfmt` is dimensioned to `nrmtmax`; for a
  species with `nrmt` $<$ `nrmtmax` the rows past `nrmt(is)` hold leftover buffer
  contents. Always slice `[:, :nrmt(is), ias]`.

  What makes this bite is that `nrmt` is set by **periodic-table row**, not by
  radius: the species files carry 200 for row 1, 300 for row 2, 400 for row 3, and
  `init0.f90:363` rounds each down to `nrmt - mod(nrmt-1, lradstp)` — so 197, 297,
  397. `checkmt`/`autormt` moves `rmt`, and never touches `nrmt`. **Two species from
  the same row have identical `nrmt` and cannot show this at all**, which is why the
  fixture is SiC (C 297, Si 397, `nrmtmax` 397) and not, say, BN. Measured on
  `sic_zb`: carbon's rows 298-397 hold values of order 1e-3 — sixteen orders above
  the 1e-19 floor of a symmetry-forbidden channel, and 2% of the real $l>0$ density
  at the sphere boundary. Nothing about them looks like roundoff.
- **Units.** Only the potentials and fields are energies: `vclmt`/`vclir`,
  `vxc*`, `vs*`, `bxc*`, `bs*`, `efermi`, `dlefe` are Hartree. `rhomt`/`rhoir` are
  $e/\text{Bohr}^3$ and `magmt`/`magir` are per volume — no conversion.
- **The build sets the byte layout.** `build-config/make.inc` is gfortran, so
  records are 4-byte length markers front and back and `spinpol` is a 4-byte
  logical; `scipy.io.FortranFile` reads it directly. `tests/test_spec.py` asserts
  that layout on the fixture. An Intel-built `STATE.OUT` is not generally
  byte-compatible.
- **The frozen core is in `rhomt` and cannot be removed.** `rhocore.f90` adds it,
  and `STATE.OUT` holds only the sum. Hydrogen is the case where this does not
  bite, which is why the fixture is hydrogen. Anything that needs a valence-only
  density from a heavier element has to get the core to cancel — e.g. by
  differencing two Elk states rather than splicing one.

## The initial state (task 9006), and which side of `mixerifc` each array sits on

`patches/0024-initial-state.patch` adds task 9006, which runs `gndstate`'s own
`trdstate = .false.` initialisation and the top of its first iteration and then
stops:

```
init0; init1; rhoinit; maginit; potks(.true.); genvsig
gencore; linengy; genapwlofr; gensocfr; genevfsv; occupy; writestate
```

`c_diamond` and `sic_zb` each carry the resulting file as `STATE_INIT.OUT`
(Elk writes both states to the same name, so `regenerate.sh` runs task 9006 in
a scratch subdirectory first and moves the file out). Two things about it must
not be mis-stated:

- **It is not "after one SCF iteration".** `rhomag` never runs. What the file
  holds is `rhoinit`'s superposition of **free atomic densities**
  (`rhoinit.f90:115-122` adds `rhosp` into the $l=0$ channel), and `potks` of
  exactly that.
- **It is therefore the one internally consistent pair Elk writes.**
  `mixerifc` is never called, so `vsmt`/`vsir` is the Kohn-Sham potential of the
  `rhomt`/`rhoir` in the same file.

The converged `STATE.OUT` is *not* that, and this is the standing
`mixerifc` trap one more time, now located exactly. In `gndstate.f90`:

| written by | line | relative to the mix |
|---|---|---|
| `evalsv`, `occsv`, `efermi` | `occupy`, :166 | before |
| `rhomt`, `rhoir`, `magmt`, `magir` | `rhomag`, :189 | **before** |
| `vsmt`, `vsir`, `vclmt`, `vxcmt`, `bxc*` | `potks`, :209, then **mixed** at :211 | **after** |
| the file itself | `writestate`, :279 (per `nwrite`) and :338 (after the loop) | after |

`mixrho` is `.false.` by default (`readinput.f90:110`), so `vmixer => vsbs`
(`init0.f90:699`) and it is the **potential** that is mixed, not the density.
So in a converged `STATE.OUT` the density and the potential are one mixing step
apart — bounded by `epspot` at convergence, but not zero, and not a
transcription bug when it shows up.

The other thing `STATE_INIT.OUT` is useful for is measuring what a frozen core
actually costs, and here it says: **the core does not cancel in
$\rho_{\rm SCF}-\rho_{\rm init}$.** The initial file's core is `rhosp`'s
free-atom core; the converged file's is `gencore`'s core in the crystal
potential. On `c_diamond` they differ by 1.67 e/Bohr³ out of 460 at the
innermost mesh points (0.36%), and by 1.9e-3 electrons inside $r<0.2$ Bohr. A
difference of the two states is valence change **plus** core relaxation.

## How to use in code

```python
from elkpy import spec
from elkpy.parsers import geometry, info

spec.ELK_VERSION                 # (11, 0, 2) -- STATE.OUT's first record too

# lattice vectors (Bohr) and atoms (lattice coordinates), from the run directory
avec, species = geometry.parse_last_geometry("run/GEOMETRY.OUT")

# the converged charges; per-atom entries are in STATE.OUT's own ias order
charges = info.parse_charges("run/INFO.OUT")
charges["muffin_tin"]            # chgmt per atom -- a clean integrated check
charges["interstitial"]          # chgir -- NOT comparable to a sharp-step sum
```

The pointwise ground truth comes from Elk's own reconstruction. For an
elkpy-managed run that is `get_density()`, which converges the ground state and
then runs task 33 on it:

```python
from elkpy.structure import Structure

structure = Structure([(3.0, 0, 0), (0, 3.0, 0), (0, 0, 3.0)],
                      {"H": [(0.0, 0.0, 0.0)]})
calc = structure.get_calculation(workdir, ngridk=(4, 4, 4),
                                 extra_blocks={"tshift": [False]})
points, density = calc.get_density(grid=(16, 16, 16))   # task 33 -> RHO3D.OUT
```

`points` is Cartesian Bohr and `density` is $e/\text{Bohr}^3$, one value per grid
point, so a reader can be compared where it actually has to be right rather than
only through an integral.

`get_density()` runs task 33 in its own wiped subdirectory, so it does not leave
`RHO3D.OUT` beside the `STATE.OUT` it came from. When the two are wanted as one
consistent set — which is what a reader needs — put `tasks 0` and `33` in a single
`elk.in` instead. That is what all three fixtures are, already run and committed,
and `parsers.volumetric.parse_plot3d()` reads their `RHO3D.OUT`. Each has a
`regenerate.sh` that rebuilds it from the committed `elk.in` and prunes back to the
committed file list, and a `README.md` carrying every number the run produces.

# 35. The momentum-resolved tunnelling Fermi surface: a planar tip

`Calculation.get_tunnelling_fermi_surface()` (elkpy task 9007,
`patches/0025-tunnelling-fermi-surface.patch`,
`src/elkpy_fermitunnel.f90`) computes the Fermi surface **as a tunnel
junction samples it**, rather than as the band structure alone defines it.

§31 puts a *point* tip at $\mathbf r$ above a two-dimensional material and
gives a real-space map $T(\mathbf r;E)$. Replace the point by an infinite
**plane** at height $z_{\rm t}$ and the same Landauer-Büttiker trace becomes a
double plane integral,

$$
T(E)=\int_{\rm tip}d^2r\int_{\rm exit}d^2r'\,
      \big|G(\mathbf r,\mathbf r';E)\big|^2 .
$$

Both planes are invariant under every lateral lattice translation, so the
lateral momentum is conserved going **in** as well as going out — the tip's
own translational invariance is the new ingredient — and the $\mathbf
k$-sum is incoherent from both sides. What is left is one number per
$\mathbf k$-point,

$$
T(E)=\sum_{\mathbf k}w_{\mathbf k}\,W(\mathbf k;E),
\qquad
W(\mathbf k;E)=\mathrm{Tr}\big[D\,G^{\rm t}_{\mathbf k}\,D\,S_{\mathbf k}\big],
\qquad D=\mathrm{diag}(g_n),
$$

with $g_n=\sqrt{\mathrm{occmax}\,\delta_\eta(E-\varepsilon_{n\mathbf k})}$ the
same on-shell amplitude §31 uses, and the two Gram matrices

$$
G^{\rm t}_{\mathbf k}[n,m]=\int_{\rm tip}\psi^*_{n\mathbf k}\hat Q_{\rm t}
  \psi_{m\mathbf k}\,d^2r,
\qquad
S_{\mathbf k}[n,m]=\int_{\rm exit}\psi^*_{n\mathbf k}\hat P_{\rm s}
  \psi_{m\mathbf k}\,d^2r' .
$$

### Why it is a k-decomposition of §31, not a new object

$T(\mathbf r;E)$ integrated over one tip cell **is** $\sum_{\mathbf
k}w_{\mathbf k}W(\mathbf k;E)$, exactly:

$$
\int_{\rm cell}T(\mathbf r;E)\,d^2r
=\sum_{\mathbf k}w_{\mathbf k}\sum_{nm}g_ng_m
 \Big[\int_{\rm tip}\psi_{n}\psi^*_{m}\Big]S_{\mathbf k}[n,m]
=\sum_{\mathbf k}w_{\mathbf k}\,
 \mathrm{Tr}\big[D\,G^{\rm t}_{\mathbf k}\,D\,S_{\mathbf k}\big].
$$

That is the derivation *and* a test on real data: nothing here is a new
approximation on top of §31, only a different way of resolving the same
transmission.

### The contraction is a trace of a product, not an elementwise sum

The Fortran exports **both** planes in the same convention, conjugate on the
first index. The tip integral that appears in $|G|^2$ is therefore
$\int_{\rm tip}\psi_n\psi^*_m=G^{\rm t}[m,n]$ by hermiticity, and

$$
W=\sum_{nm}g_ng_m\,G^{\rm t}[m,n]\,S[n,m]
 =\mathrm{Tr}[D\,G^{\rm t}D\,S].
$$

The tempting elementwise form $\sum_{nm}g_ng_m G^{\rm t}[n,m]S[n,m]$ is
$\mathrm{Tr}[D\,G^{\rm t}D\,S^{\mathsf T}]$. It is real, non-negative, blind
to a degenerate rotation, and exactly right whenever $S$ is diagonal — so it
is wrong only where the interference this exists for lives. This is the same
transpose trap §31 documents one level down, on the exit variable, and it is
pinned the same way: `parsers.fermitunnel.quadrature_weight()` evaluates the
double integral literally, and
`tests/test_parsers_fermitunnel.py::test_contraction_is_the_double_integral`
samples **two distinct planes** (with one plane the two forms coincide and
the test would prove nothing) and asserts the wrong form disagrees.

### Two limits, free from the same export

- $G^{\rm t}=S=\mathbb 1$ gives $W=\mathrm{occmax}\sum_n\delta_\eta(E-
  \varepsilon_{n\mathbf k})$ — the **plain Fermi surface**, the same quantity
  Elk's own task 103 writes through a completely separate code path.
- $S=\mathbb 1$ alone gives $W=\sum_n g_n^2\,G^{\rm t}[n,n]$ — the Fermi
  surface weighted by how much of each state survives out at the tip plane: a
  planar Tersoff-Hamann image, in momentum space.

`compute_fermi_weight()` returns the plain surface as `"bare"` on **every**
call, on the same mesh, so the ratio — which pocket the junction actually
sees — costs nothing and needs no second run.

### What it is for

A Bloch state's vacuum tail decays as $e^{-\kappa z}$ with
$\kappa=\sqrt{2(V_0-E)+|\mathbf k+\mathbf G|^2}$, so a Fermi-surface sheet at
large in-plane momentum is exponentially invisible to a tunnel junction while
one at the zone centre is not. The plain Fermi surface and the one a junction
sees can therefore look nothing alike. Monolayer 1H-NbSe2 is the worked case
(`notebooks/21_tunnelling_fermi_surface.ipynb`): its $\Gamma$ and K pockets
carry nearly the same density of states, and the junction sees essentially
only $\Gamma$.

### Implementation

Both planes are specified as an **axis plus a fractional height**, never as a
general parallelogram, because that is what makes each Gram matrix exact. In
the interstitial $\psi=\Omega^{-1/2}\sum_{\mathbf G}c_{\mathbf
G}e^{i(\mathbf k+\mathbf G)\cdot\mathbf r}$, and on a plane spanned by two
lattice vectors the $\mathbf k$-dependent phases cancel between the two
wavefunctions, leaving an orthogonality relation between the in-plane
$\mathbf G$ indices:

$$
S[n,m]=\frac{A}{\Omega}\sum_{\mathbf G_\parallel}
  b^*_n(\mathbf G_\parallel)\,\hat P\,b_m(\mathbf G_\parallel),
\qquad
b_n(\mathbf G_\parallel)=\sum_{G_3}c_n(\mathbf G_\parallel,G_3)
  e^{2\pi iG_3 s}.
$$

Nothing is sampled and nothing converges. `elkpy_plane_gram` in
`src/elkpy_fermitunnel.f90` does this; it is a deliberate **copy** of the
block `elkpy_transport.f90` runs inline for its exit plane, rather than a
shared routine, so that the already-verified task 9005 is not edited — and
`tests/` pins the two against each other on a real run so the duplicate
cannot drift.

The $k$-points are generated and diagonalised fresh by the task (via
`elkpy_transport_wf`, patch 0012), so the mesh is independent of `ngridk` and
of `reducek`. That independence is not a convenience here: a symmetry-reduced
mesh does not cover the Brillouin zone, so it cannot draw a Fermi surface.
Measured cost on monolayer NbSe2 (3 atoms, 18 Å cell, `rgkmax=7`): ~70 s of
`init0`/`init1`/`linengy`/`genapwlofr` setup, then **0.17 s per k-point**, so
a $54\times54$ mesh is under ten minutes.

Same requirements as §31, same reasons: `tshift=False` (Elk otherwise
relocates the origin while both plane heights stay in your frame), both planes
in the interstitial, and the material between them. All three are checked in
the Fortran, which refuses rather than returning a plausible number.

### Three observables, one export: DOS, Tersoff-Hamann, and the real dI/dV

The same export answers a question that has nothing to do with $\mathbf k$
resolution: **how far is a measured dI/dV from the density of states?** Three
contractions, all returned on every call:

| returned as | formula | bands add as |
|---|---|---|
| `"bare"` | $N(E)=\sum_{n\mathbf k}w_{\mathbf k}\delta_\eta(E-\varepsilon_{n\mathbf k})$ | — |
| `"tersoff_hamann"` | $n(E;z_{\rm t})=\sum_{n\mathbf k}w_{\mathbf k}\delta_\eta\int_{\rm tip}\!|\psi_{n\mathbf k}|^2$ | probabilities |
| `"weight"` | $\mathrm{Tr}[D\,G^{\rm t}_{\mathbf k}\,D\,S_{\mathbf k}]$ | amplitudes |

with `"total_bare"`, `"total_tersoff_hamann"` and `"total"` their
$\mathbf k$-integrals. `"bare"` and `"tersoff_hamann"` are fixed references:
they ignore `tip_region`/`exit_region`, so asking for `weight` without the tip
weighting cannot silently redefine the curve it is being compared against.

Sweeping `energies` turns the three into three **spectra** at essentially no
cost — neither the wavefunctions nor the Gram matrices depend on the energy,
only $D$ does. The middle row is what makes the comparison honest: without it,
"dI/dV differs from the DOS" conflates two separate effects, the vacuum
weighting (row 1 → row 2) and the interference (row 2 → row 3).

`"total"` and `"total_tersoff_hamann"` are both checked against §31's
independently implemented point-tip map integrated over the tip plane —
a real-space grid contracted pixel by pixel, versus a closed-form G-sphere
collapse contracted as one trace per $\mathbf k$. On NbSe2 they agree to
**1e-14**.

**Measured on monolayer NbSe2** ($36\times36$ mesh, $\eta=0.004$ Ha, both
planes 3.5 Å out, 101 energies over $\pm1.36$ eV): the DOS has exactly one
peak in that range, at $-0.08$ eV. The dI/dV has a *dip* near there
($-0.22$ eV) and two large peaks, at $-0.76$ and $+0.52$ eV, where the DOS has
no structure at all. At $+0.52$ eV the dI/dV is $12.8\times$ its value at
$E_F$ while the DOS is $0.36\times$ — the junction does not merely attenuate
the DOS, it inverts the trend. The transfer function rises by $5.3\times$
across the window through one vacuum gap and $22\times$ through two, because
$\kappa=\sqrt{2(V_0-E)+|\mathbf k_\parallel|^2}$ falls as $E$ rises, so no
constant divisor recovers $N(E)$ from a measured spectrum.

This is the same conclusion the STM-theory literature reaches for graphene
(Wehling *et al.*, PRL **101**, 216803 (2008), who write the middle row as
$dI/dU\sim|\Psi_\Gamma|^2N_\Gamma+|\Psi_K|^2N_K$ and measure
$|\Psi_\Gamma/\Psi_K|^2\propto e^{1.7\,\text{\AA}^{-1}z}$), and for planar
2D-2D junctions (Li, Nie, Cho & Feenstra, J. Electron. Mater. **46**, 1378
(2017), whose geometry is exactly this task's). **Caveat worth carrying:**
everything here is the *elastic* channel. Wehling's point is that in graphene
the elastic K channel is so suppressed that an inelastic, phonon-mediated one
takes over — so a small computed dI/dV at some energy is a lower bound on what
is measured, not a prediction of invisibility.

### The one thing that will silently mislead you: the plane-wave floor

`rgkmax` sets how far into the vacuum the interstitial plane-wave sum stays
meaningful, and the **large-$\mathbf k$ pocket is what falls below that floor
first** — precisely the pocket whose suppression is the result. Measured on
NbSe2 at `rgkmax=7`, tip-only weighting, $12\times12$ mesh:

| tip distance from the outer Se | $W(\Gamma)$ | $W(\mathrm K)$ | $\Gamma/\mathrm K$ |
|---|---|---|---|
| 1.5 Å | 6.19e-2 | 7.30e-3 | 8.5 |
| 2.5 Å | 1.03e-2 | 3.60e-4 | 28.7 |
| 3.5 Å | 1.40e-3 | 1.54e-5 | 91.2 |
| 4.5 Å | 1.92e-4 | 1.47e-6 | 131 |
| 5.5 Å | 3.12e-5 | 1.15e-6 | 27.1 |
| 6.5 Å | 4.66e-6 | 1.01e-6 | 4.6 |

$\Gamma$ decays as a clean exponential across the whole range
($\kappa_\Gamma\approx0.95$ Å$^{-1}$). K decays as a clean, *steeper*
exponential ($\kappa_{\rm K}\approx1.54$ Å$^{-1}$) only out to ~3.5 Å and
then **flattens onto ~1e-6**, after which the apparent contrast *falls*. The
flattening is height-**independent** — the last three rows differ by 15% over
2 Å, which would need $\kappa\approx0.07$ Å$^{-1}$, i.e. a state within
0.01 eV of the vacuum level — so it is the basis running out, not a state.
The physical answer is the 3.5 Å row; the 6.5 Å row is numerical noise dressed
as a result. **Sweep the tip height and check both pockets are still straight
lines on a log plot before quoting a ratio.** Raising `rgkmax` pushes the floor
down and the usable range out.

The positive form of the same test — showing a run *is* clean — is to do it at
two distances and check the physics closes on itself. With both planes at 2.5
and then 3.5 Å, the $\Gamma$ and K pocket sums give
$\kappa_\Gamma=1.14$ Å$^{-1}$ and $\kappa_{\rm K}=1.50$ Å$^{-1}$; those two
numbers alone then predict how much the $\Gamma$/K contrast should grow
between the runs, $e^{2\Delta\kappa\Delta z}=4.17$, against a measured
$25.32/6.08=4.17$. A floor on either pocket would break that.

One trap in the diagnostic itself, which looks exactly like a floor and is
not: dividing $W(\mathbf k)$ by the plain surface gives a ratio that stops
falling above $|\mathbf k|\approx0.66$ Å$^{-1}$ — at **both** distances, and
with the two curves separated by exactly the expected exponential in $z$. A
basis floor would move with distance. What saturates is the ratio, because a
band several $\eta$ off the sampled energy contributes almost nothing to the
denominator and can still dominate the numerator through a far better vacuum
tail. Take pocket sums of $W$ itself, not ratios, when the question is how
much current a sheet carries.
