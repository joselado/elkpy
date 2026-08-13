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
target phase (`test_flux_sign_matches_fhs_eq8_*`); the Fortran overlap convention
(`M(a,b)=conjg(oq(b,a))` in `elkpy_berry.f90`) is pinned by direct derivation from
`genolpq.f90`'s documented BLAS (`zgemv`) semantics, cross-checked twice, rather than
verified at runtime — no test exercises the real Fortran-to-Python sign chain
end-to-end, since Si's Chern number is 0 regardless of sign ($0=-0$) and no other
topological reference case is used here. Also verified against a real compiled binary:
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
($\Omega(\mathrm K)=8.568$, $\Omega(\mathrm K')=-8.568$ Bohr$^{-2}$, agreeing to 0.01%) —
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
result[0]["curvature"]  # Bohr^-2, one dict per requested k-point
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
result[1]["berry_curvature"]  # Bohr^-2, identical convention to get_berry_curvature_path
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
