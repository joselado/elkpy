# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Roadmap Tiers 1-3 from `docs/roadmap.md` are implemented: `Structure` → `Calculation` with
`get_energy()`, `get_bands()`, `get_dos()`, `get_forces()`, `get_relaxed()`, `get_effective_mass()`,
`get_density()`, `get_phonon_dos()`/`get_phonon_dispersion()`, and a `run_tasks()` escape hatch, plus a
`spec.py` module for version-coupled task/xctype/filename knowledge. Verified end-to-end against a real
compiled Elk binary on bulk Si/Fe (`tests/test_calculation_si.py`, `tests/test_calculation_fe.py`,
`tests/test_calculation_si_phonons.py`). Also present: `vendor/elk/` (vendored Elk 11.0.2, unmodified,
tracked in git), `docs/elk_manual.pdf`/`.txt` (official manual, plain-text version for grepping), and
`docs/design.md` + `docs/roadmap.md` (architecture strategy and forward plan — read before adding code).
Not implemented: symbolic k-path's disconnected-segment support (`,` breaks), the classical supercell
phonon method (task 200, DFPT/task 205 only), scheduler-backed launchers, MPI, named `get_*` methods for
potential/ELF volumetric plots (reachable via `run_tasks()` + `parsers.volumetric`). Check `src/elkpy/`
directly rather than assuming the docs describe current code; update both as they diverge.

Also implemented, as the first real entry in the Fortran patch series described below:
`Calculation(spinorb=, soc_scale={"Fe": 1.5, ...})` — per-species scaling of the spin-orbit coupling
term, on top of upstream Elk's single global `socscf` scalar. `patches/0001-per-species-soc-scale.patch`
adds a `socscfsp(maxspecies)` array (`modmain.f90`) and an `elkpy_socscale` input block (`readinput.f90`)
overriding `socscf` per species inside `gensocfr.f90`'s existing per-atom loop — the smallest possible
hook, verified against a real compiled binary to reproduce the global-`socscf` result exactly for a
single-species cell and to scale independently per species in a two-species cell
(`tests/test_calculation_soc.py`). Physics writeup (Koelling-Harmon SOC term, why a global scale is
the wrong shape for multi-species cells): `docs/design.md` §12 and `docs/physics.tex` (Part I).

Also implemented, as the second entry in the Fortran patch series:
`Calculation.get_berry_curvature(ist0, ist1, directions=(1, 2))` — Berry curvature and Chern number via
the Wilson-loop (Fukui-Hatsugai-Suzuki) method. `patches/0002-berry-curvature-wilson-loop.patch` adds a
new task (9000) and `elkpy_berry.f90`, reusing Elk's own `genwfsvp`/`genolpq` (the machinery behind
Wannier90 export, task 550) to export mesh-neighbour wavefunction overlap matrices without needing
task 550's own neighbour-shell search (which calls the external Wannier90 library, unavailable in this
build) — the two loop directions are exact `ngridk` mesh generators, so the neighbour is constructed
directly. All Wilson-loop arithmetic (link variables, plaquette flux, Chern number, admissibility) is
done in Python (`parsers/berry.py`), independently unit-tested against synthetic overlap matrices, both
for gauge invariance and, separately, for sign -- gauge invariance alone can't catch a conjugation-sign
flip, since `conj(M)` is exactly as gauge-invariant as `M`, so the Python arithmetic's sign is pinned
directly against FHS eq. 8 with hand-built matrices of known phase (`tests/test_berry_gauge_invariance.py`).
The Fortran conjugation convention itself rests on a `zgemv` BLAS-semantics derivation, not a runtime
test. Against a real compiled binary, bulk Si's trivial valence manifold gives a Chern number of
floating-point zero (`tests/test_calculation_berry.py`) -- which, being zero either way, would not itself
catch a sign error.

Also implemented: `Calculation.get_berry_curvature_path(kpoints, ist0, ist1, directions=(1, 2), dk=)`
(task 9001) -- Berry curvature at an arbitrary, explicit list of k-points (e.g. a band-structure-style
path), via one small Wilson loop per point (`pyqula`'s `berry_curvature(h, k, dk=...)` convention)
rather than a periodic mesh. Also accepts `kpath="GKMG"` (pyqula/ASE-style symbolic path, same
convention as `get_bands()`/`get_phonon_dispersion()`'s `kpath=`, resolved via `_kpath_to_points()`)
instead of `kpoints=`, discretized into `npoints` points with each returned point additionally carrying
a `"distance"` entry for plotting; unlike `get_bands()`/`get_phonon_dispersion()` (which hand vertices to
Elk's own `plot1d` task, so a disconnected `,` path can't be interpolated), `get_berry_curvature_path()`
evaluates every point independently, so a disconnected `,` path (e.g. to jump straight to a specific K′
zone image) is fully supported. Needed one genuinely new piece of Fortran, not just a smaller task 9000:
`elkpy_wfcorner` expands a wavefunction at an arbitrary k-point via fresh on-the-fly diagonalisation
(`eveqnfv`/`eveqnsv`, the same pattern `src/bandstr.f90` uses for a band-structure path) rather than
`genwfsvp`'s file-backed `getevecfv`/`getevecsv`, which only knows previously-diagonalised mesh points --
so this mode needs no `ngridk` alignment and no `reducek=0` at all, and a Γ-only ground state is
sufficient (not a stopgap), since task 9001 never touches the ground state's own mesh. Trade-off: no
Chern number (needs a closed cover of the zone) and no automatic gap check (no eigenvalues exported in
this mode). Verified against a real compiled binary two ways: mesh mode and path mode agree (to ordinary
numerical tolerance) when asked to evaluate the literal same four k-points on bulk Si
(`tests/test_calculation_berry.py::test_path_and_mesh_conventions_agree`); and on monolayer h-BN (broken
sublattice inversion symmetry, unlike Si), the occupied manifold's curvature is exactly antisymmetric
between the two physically inequivalent valleys K/K′ (8.568 vs -8.568 Bohr⁻², agreeing to 0.01%) --
required by time-reversal symmetry and a much sharper check than Si's Chern number ($0=-0$ either way).
That h-BN run also surfaced two real pitfalls worth remembering for any future band-window usage: the
occupied-band count should come from Elk's own `EIGVAL.OUT` occupation numbers, not assumed from a total
(core + valence) electron count (core electrons aren't among the valence bands `nstsv` indexes at all);
and a single band's curvature can diverge near a point where it's degenerate with a *neighbouring
occupied* band, not just the first unoccupied one, even when the requested window's own outer boundary
stays gapped -- the fix is windowing the full degenerate group together, not picking a different single
band. Physics writeup (both modes, the FHS link-variable/plaquette-flux construction, and exactly what
is/isn't verified): `docs/design.md` §13 and `docs/physics.tex` (Part II).

## Architecture

- `src/elkpy/structure.py` — `Structure`: lattice vectors (`avec`, Bohr) + species, each atom either a
  bare `(x,y,z)` position or a `(position, bfcmt)` pair (per-atom magnetic field, manual sec. 5.2).
  `species_files` optionally overrides the `{symbol}.in` species-filename convention. `from_ase()`/
  `to_ase()` convert Angstrom/Cartesian ASE `Atoms` (optional dependency).
- `src/elkpy/calculation.py` — `Calculation`: owns one run directory and the ground-state-defining
  parameters (`xc`, `spinpol`, `spinorb`, `soc_scale`, `rgkmax`, `ngridk`, `extra_blocks` for anything
  else — e.g. `maxscl`). `soc_scale={"Fe": 1.5}` requires `spinorb=True` and needs the
  `patches/0001-per-species-soc-scale.patch` Fortran extension applied (i.e. a binary built via
  `scripts/build_elk.sh` after the patch was added — see git log for when). `get_*` methods block and
  can be expensive — real Elk subprocesses, not in-memory work.
  `ensure_ground_state()` reuses a prior task-0 run only if a JSON manifest (`.elkpy_manifest.json`)
  shows the basis/structure/functional-defining parameters (including `extra_blocks`) and the Elk
  binary identity are unchanged; sampling parameters (e.g. a denser `ngridk` passed to `get_dos()`) are
  free to differ. Every other `get_*` runs via `_run_resumed()` in its own **wiped-clean** subdirectory,
  never in `self.workdir` — this isn't just tidiness, it's load-bearing correctness: some tasks (phonon
  DFPT, task 205) treat the mere presence of prior output files as "already done" and resume from them,
  so a stale subdirectory from an earlier killed/crashed run silently corrupts the next result instead
  of erroring (hit this for real while implementing phonons — see git log). `converged` property +
  `raise_on_nonconvergence=` control whether non-convergence raises or must be checked explicitly.
- `src/elkpy/spec.py` — version-coupled knowledge (task codes, `xctype` codes, output filenames) as
  data, each entry cross-checked against `vendor/elk/src/` (not just the manual) — an Elk version bump
  should mean editing this one file.
- `src/elkpy/inputfile.py` — generic `elk.in` block writer (block name + value lines) and `read_blocks()`
  reader (used to parse `GEOMETRY_OPT.OUT`, which Elk writes in the same block syntax).
- `src/elkpy/launcher.py` — `LocalLauncher`: blocking local subprocess execution; refuses `nprocs > 1`
  since `build-config/make.inc` builds serial (`mpi_stub.f90`) — that combination would silently launch
  N racing copies into one directory, not parallelize.
- `src/elkpy/parsers/` — one small module per output file family (`info`, `totenergy`, `band` — reused
  for phonon dispersion, since `PHDISP.OUT` shares `BAND.OUT`'s exact layout — `dos`, reused for phonon
  DOS, `forces`, `geometry`, `effmass`, `volumetric`), each verified against real Elk output, not
  assumed from the manual. `berry` is the exception to "just a parser": it also does all of the
  Wilson-loop/Chern-number arithmetic in Python (`compute_berry_curvature()`), deliberately kept out of
  Fortran so it's unit-testable against synthetic overlap matrices (`tests/test_berry_gauge_invariance.py`)
  without an Elk run.
- `src/elkpy/config.py` — locates the built `elk` binary (`build/elk/src/elk` by default, override via
  `ELKPY_ELK_BIN`) and the species directory (`vendor/elk/species/` by default).

## Project purpose

A Python interface to Elk, an all-electron full-potential linearized augmented-plane-wave (LAPW)
density-functional-theory (DFT) code written in Fortran. On top of the interface, this project adds
extra functionality that Elk itself does not provide.

## Development practices

When implementing new physics (a new Fortran capability, a new formula, a new numerical scheme —
not routine wrapping of an existing Elk task), checking arXiv for the relevant method/paper is
encouraged where it fits the task, to ground the implementation in the actual published formalism
(e.g. matching sign/normalization conventions, confirming which approximation a term corresponds
to) rather than guessing from the code alone.

Whenever a new formalism is added or an existing one is modified (new physics, a changed formula, a
different numerical scheme — not routine wrapping of an existing Elk task), update the documentation
describing it in both forms: the Markdown docs (`docs/design.md`/`docs/roadmap.md` or wherever the
capability is described) and the corresponding LaTeX writeup, added as a new `\part{}` (with a `\label`)
inside the single shared file `docs/physics.tex` — not a new `.tex` file per addition. Both must be
physics-focused, not just an API description: state the relevant formula(e), define each symbol, and
explain the physical meaning/approximation being made (what it captures, what it neglects, how it
relates to the underlying published method) — not merely "this function computes X". Each writeup must
also include a "how to use in code" part showing the actual elkpy call(s) (e.g. `Calculation(...)`, the
relevant `get_*()`) that exercise the formalism, so the physics and the API surface stay tied together.
Keep both in sync with the code in the same change — don't defer either to a follow-up.

## Core constraint: isolate changes to vendored Elk source, don't avoid Fortran

Elk's own Fortran source will be vendored into this repository (not treated as an installed system
dependency), so that the Python interface has a fixed, buildable copy to target. Because upstream Elk
is expected to be swapped for a newer release in the future, **changes must stay isolated** — Fortran
itself is a fine implementation choice for new capability; what's constrained is how it touches the
vendored tree:

- `vendor/elk/` always stays byte-for-byte what was downloaded from upstream — never edited directly,
  including `make.inc`. All building and any Fortran changes happen from a separate out-of-tree copy
  (see `docs/design.md` §8).
- Prefer Elk's existing export tasks (matrix elements, wavefunction/Wannier90 export, `STATE.OUT`
  post-processing — see `docs/design.md` §8) for new physics when they're sufficient; this is cheaper
  and carries zero Fortran risk, not a mandate to avoid Fortran altogether.
- When new Fortran is genuinely needed, prefer additive new files (new modules/subroutines) over
  editing existing upstream files. When hooking into existing control flow is unavoidable (e.g. the
  task dispatch in `elk.f90`), keep the edit to the smallest possible footprint, clearly marked, and
  track it as one hunk in a maintained patch series applied to the build copy — never committed as a
  direct change to `vendor/elk/`.
- New functionality should live in Python or in clearly separated new Fortran files rather than being
  folded into Elk's existing modules, so the patch series stays small and easy to re-evaluate against a
  new upstream version.

## Commands

- Build Elk out-of-tree (copies `vendor/elk/` to `build/elk/`, applies `patches/*.patch` if any, drops
  in `build-config/make.inc`, builds — never touches `vendor/elk/`):
  `./scripts/build_elk.sh`. Must be run before any test/example that actually invokes Elk.
  Serial by design — `make -j` races on an implicit ordering dependency in Elk's own `src/Makefile`
  (stub files like `mpi_stub.f90`/`libxcifc_stub.f90` must compile before the modules that `use` them,
  and that isn't expressed as an explicit prerequisite upstream); this is a pre-existing upstream
  issue, not something to fix by editing `vendor/elk/`.
- Install elkpy (editable): `python3 -m pip install -e .`
- Run tests: `python3 -m pytest tests/`. Run a single test: `python3 -m pytest tests/test_calculation_si.py::test_get_bands`.
  `tests/test_calculation_*.py` are integration suites that run the real `elk` binary on bulk Si/Fe —
  they self-skip if `build/elk/src/elk` doesn't exist yet, so run `./scripts/build_elk.sh` first to
  actually exercise them. `tests/test_structure.py`'s ASE round-trip test self-skips if `ase` isn't
  installed (`pip install -e .[ase]`).
- `tests/test_calculation_si_phonons.py` (DFPT phonon dispersion/DOS) is skipped by default even with
  the binary built — confirmed ~11-13 minutes per test on a minimal 2-atom, `ngridq=(2,2,2)` grid, cost
  dominated by DFPT's per-perturbation-per-q-point work, not anything elkpy controls. Set
  `ELKPY_RUN_SLOW_TESTS=1` to actually run them.
- `build-config/make.inc` targets GNU Fortran + OpenBLAS/LAPACK + FFTW, serial (no MPI); edit it (not
  `vendor/elk/make.inc`) to change compiler/library configuration.
