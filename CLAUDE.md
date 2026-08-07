# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

An early but working v0 slice exists, per `docs/design.md` §10: `Structure` → `Calculation` with
`get_energy()`, `get_bands()`, `get_dos()`, and a `run_tasks()` escape hatch, verified end-to-end
against a real compiled Elk binary on bulk Si (`tests/test_calculation_si.py`). Also present:
`vendor/elk/` (the vendored Elk 11.0.2 source, unmodified, tracked in git), `docs/elk_manual.pdf` /
`docs/elk_manual.txt` (the official manual, plain-text version pdftotext-extracted for grepping), and
`docs/design.md` (the full architecture strategy — read it before adding code; it covers the object
model, `STATE.OUT` reuse semantics, the build/patch mechanism, and the Fortran-isolation policy). Much
of the design (geometry relaxation, phonons, the `spec` module, scheduler-backed launchers, symbolic
k-path resolution) is not implemented yet — check `src/elkpy/` directly rather than assuming
`docs/design.md` describes current code; update both as they diverge.

## Architecture

- `src/elkpy/structure.py` — `Structure`: lattice vectors (`avec`, Bohr) + species with fractional
  positions. `from_ase()`/`to_ase()` convert Angstrom/Cartesian ASE `Atoms` (optional dependency).
- `src/elkpy/calculation.py` — `Calculation`: owns one run directory and the ground-state-defining
  parameters (`xc`, `spinpol`, `rgkmax`, `ngridk`, ...). `get_*` methods block and can be
  expensive — they run real Elk subprocesses, not in-memory work. `ensure_ground_state()` reuses a
  prior task-0 run only if a JSON manifest (`.elkpy_manifest.json` in the run directory) shows the
  basis/structure/functional-defining parameters and the Elk binary identity are unchanged; sampling
  parameters (e.g. a denser `ngridk` passed to `get_dos()`) are free to differ and just trigger a
  task-1 resume, not a full reconverge.
- `src/elkpy/inputfile.py` — generic `elk.in` block writer (block name + value lines), not hardcoded
  per block.
- `src/elkpy/launcher.py` — `LocalLauncher`: blocking local subprocess execution of the `elk` binary.
- `src/elkpy/parsers/` — one small module per output file (`info.py` for SCF convergence via the
  literal strings Elk writes to `INFO.OUT`, `totenergy.py`, `band.py`, `dos.py`), each verified against
  real output rather than assumed from the manual alone.
- `src/elkpy/config.py` — locates the built `elk` binary (`build/elk/src/elk` by default, override via
  `ELKPY_ELK_BIN`) and the species directory (`vendor/elk/species/` by default).

## Project purpose

A Python interface to Elk, an all-electron full-potential linearized augmented-plane-wave (LAPW)
density-functional-theory (DFT) code written in Fortran. On top of the interface, this project adds
extra functionality that Elk itself does not provide.

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
  `tests/test_calculation_si.py` is an integration suite that runs the real `elk` binary on bulk Si —
  it self-skips if `build/elk/src/elk` doesn't exist yet, so run `./scripts/build_elk.sh` first to
  actually exercise it. `tests/test_structure.py`'s ASE round-trip test self-skips if `ase` isn't
  installed (`pip install -e .[ase]`).
- `build-config/make.inc` targets GNU Fortran + OpenBLAS/LAPACK + FFTW, serial (no MPI); edit it (not
  `vendor/elk/make.inc`) to change compiler/library configuration.
