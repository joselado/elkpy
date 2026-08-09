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
`eigenstate_session()` and, per requested point, walks a small rectangular loop anchored
at that point — corners $\mathbf k$, $\mathbf k+\mathbf v_1$, $\mathbf k+\mathbf
v_1+\mathbf v_2$, $\mathbf k+\mathbf v_2$ ($\mathbf v_\mu = \texttt{dk}\times
\mathbf b_{\mu}$, $\mathbf b_\mu$ a Cartesian reciprocal lattice vector computed directly
from `self.structure.avec` by `_reciprocal_vectors()` — the exact formula Elk's own
`src/reciplat.f90` uses, kept in Python rather than read back from an Elk-written file,
unlike `parsers.berry`'s `bvec`) — issuing 9 overlap queries (4 self-overlaps, 5 cross
overlaps) and handing them to `parsers.quantum_geometry.compute_quantum_geometry()`,
pure Python, unit-testable without an Elk run
(`tests/test_quantum_geometry_gauge_invariance.py`) the same way `parsers.berry` is.
This was an explicit design choice over adding a new task analogous to 9001 (a
dedicated `elkpy_quantum_geometry.f90` exporting the same 9 matrices in one subprocess
launch instead of 9 round-trips over an already-open pipe) — see docs/physics.tex Part
IV for why the metric's dominant error source is much cheaper to fix in Python than in
Fortran, which is the actual reason no new Fortran was needed here, not merely that it
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

**A genuine remaining discretization subtlety, not a bug**: the metric's off-diagonal
component $g_{12}$ converges markedly more slowly with `dk` than $g_{11}/g_{22}$ or the
curvature — it's built from a *difference* of three comparable-magnitude
`quantum_distance()` values via the polarization identity ($g_{12} = [D(\mathbf
v_1+\mathbf v_2) - D(\mathbf v_1) - D(\mathbf v_2)] / (2\,\texttt{dk}_1\texttt{dk}_2)$),
so its own $O(\texttt{dk})$ discretization error is amplified relative to its
$O(\texttt{dk}^2)$ signal in a way the diagonal terms' direct evaluation isn't. Observed
directly on monolayer h-BN's K/K′ valleys (structure and occupied-window convention as
in §13's path-mode verification): $g_{11}(K)$/$g_{11}(K')$ already agree to $<0.1\%$ at
`dk=0.01`, matching the tight discretization curvature already achieves at that `dk`
(§13), while $g_{12}(K)$/$g_{12}(K')$ — which time-reversal symmetry ($g_{ab}(-\mathbf
k)=g_{ab}(\mathbf k)$, the same argument §13 uses for curvature's sign, applied to the
metric instead) requires to agree in the `dk`$\,\to 0$ limit — only visibly converge
toward each other as `dk` is refined (`tests/test_calculation_quantum_geometry.py::
test_hbn_metric_offdiagonal_parity_improves_with_smaller_dk`), rather than already
agreeing tightly at a single practical `dk`. Anyone reaching for $g_{12}$ specifically
(not just $g_{11}$/$g_{22}$/curvature) should check its own `dk`-convergence before
trusting a single value, the same discipline `get_berry_curvature_path()`'s own
docstring already asks for regarding curvature.

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
quantum metric is positive semi-definite at Γ, K, M, K′, its diagonal is K/K′-symmetric
to $<1\%$ at `dk=0.01` (as time-reversal requires), curvature vanishes at the
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
