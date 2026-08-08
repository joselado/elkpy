# elkpy

A Python interface to [Elk](https://elk.sourceforge.io/), an all-electron
full-potential linearized augmented-plane-wave (FP-LAPW) density-functional-theory
(DFT) code written in Fortran. elkpy wraps Elk's `elk.in`/task-number workflow in a
small, `pyqula`-style object model (`Structure`, `Calculation`), and adds physics Elk
itself does not provide — currently per-species spin-orbit coupling scaling, Berry
curvature/Chern numbers via a Wilson-loop method, and a fast interactive
eigenstate/overlap query session — as new Fortran extensions plus Python arithmetic
on top.

A `Calculation.get_*()` call runs a real Elk subprocess (seconds to minutes, not an
in-memory operation) and returns already-parsed NumPy arrays instead of raw Elk
output files.

```python
from elkpy.structure import Structure

avec = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
species = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}

calc = Structure(avec, species).get_calculation("run/si", xc="PW", ngridk=(4, 4, 4))
calc.get_energy()                          # Hartree
distances, energies = calc.get_bands(kpath="GXWLGK")
```

## What's implemented

- **Core workflow**: `Structure` (lattice + species, ASE round-trip via `from_ase`/
  `to_ase`) → `Calculation.get_energy()` / `get_bands()` / `get_dos()` /
  `get_forces()` / `get_relaxed()` / `get_effective_mass()` / `get_density()` /
  `get_phonon_dos()` / `get_phonon_dispersion()`, plus a `run_tasks()` escape hatch
  for any Elk task not wrapped as a named method.
- **Per-species spin-orbit coupling scaling** — `Calculation(spinorb=True,
  soc_scale={"Fe": 1.5})`, on top of upstream Elk's single global `socscf` scalar.
- **Berry curvature / Chern numbers** — `get_berry_curvature()` (periodic-mesh
  Wilson-loop/Fukui-Hatsugai-Suzuki method, gives a Chern number) and
  `get_berry_curvature_path()` (the same construction at an arbitrary, explicit list
  of k-points via fresh on-the-fly diagonalisation — no periodic mesh required).
- **Interactive eigenstate/overlap session** — `Calculation.eigenstate_session()`, a
  long-lived Elk subprocess for fast repeated eigenstate/overlap queries at arbitrary
  k-points, plus one-off wrappers `get_eigenstates()`/`get_overlap()`.

Each of these is verified end-to-end against a real compiled Elk binary — see
`tests/`. Physics behind each new capability (formulas, symbols, what's captured vs.
neglected) is written up in `docs/physics.tex`; the current implementation status and
what's *not* yet implemented is tracked in `docs/roadmap.md`. `docs/design.md` covers
the architecture strategy in depth.

## Repository layout

- `src/elkpy/` — the Python package (`structure.py`, `calculation.py`, `session.py`,
  `spec.py`, `inputfile.py`, `launcher.py`, `config.py`, `parsers/`).
- `vendor/elk/` — vendored Elk 11.0.2 source, unmodified, tracked in git.
- `patches/` — a small, additive patch series applied to an out-of-tree build copy of
  `vendor/elk/` (never to `vendor/elk/` itself) for the new Fortran extensions above.
- `docs/` — `design.md` (architecture), `roadmap.md` (status/plan), `physics.tex`
  (physics writeups, one `\part` per new formalism), `elk_manual.pdf`/`.txt`.
- `notebooks/` — one Jupyter notebook per feature area, demonstrating the Python API
  against a real Elk run (see below).
- `tests/` — unit tests (no Elk needed) plus integration suites that run the real
  `elk` binary on bulk Si/Fe/h-BN; self-skip if the binary isn't built.

## Getting started

```bash
# 1. Build Elk out-of-tree (copies vendor/elk/ -> build/elk/, applies patches/*.patch,
#    drops in build-config/make.inc, builds -- never touches vendor/elk/)
./scripts/build_elk.sh

# 2. Install elkpy (editable)
python3 -m pip install -e .        # add .[ase] for Structure.from_ase()/to_ase()

# 3. Run the tests
python3 -m pytest tests/
```

`build-config/make.inc` targets GNU Fortran + OpenBLAS/LAPACK + FFTW, serial (no
MPI); edit it (not `vendor/elk/make.inc`) to change compiler/library configuration.
`tests/test_calculation_si_phonons.py` (DFPT phonons) is skipped by default — it
takes ~11-13 minutes per test — set `ELKPY_RUN_SLOW_TESTS=1` to run it.

## Notebooks

Each notebook is runnable end-to-end against a real compiled `elk` binary (just
`./scripts/build_elk.sh` first) and is checked in with its actual output cells, not
just code:

| Notebook | Feature |
| --- | --- |
| [`01_getting_started.ipynb`](notebooks/01_getting_started.ipynb) | `Structure`/`Calculation`, `get_energy()`, `get_bands()`, `get_dos()` |
| [`02_relaxation_forces_and_properties.ipynb`](notebooks/02_relaxation_forces_and_properties.ipynb) | `get_forces()`, `get_relaxed()`, `get_effective_mass()`, `get_density()`, `run_tasks()` |
| [`03_phonon_dispersion_and_dos.ipynb`](notebooks/03_phonon_dispersion_and_dos.ipynb) | `get_phonon_dispersion()`, `get_phonon_dos()` (DFPT; slow, see notebook) |
| [`04_per_species_soc_scaling.ipynb`](notebooks/04_per_species_soc_scaling.ipynb) | `Calculation(spinorb=True, soc_scale={...})` |
| [`05_berry_curvature.ipynb`](notebooks/05_berry_curvature.ipynb) | `get_berry_curvature()` (Si Chern number), `get_berry_curvature_path()` (monolayer h-BN K/K' valleys) |
| [`06_eigenstate_session.ipynb`](notebooks/06_eigenstate_session.ipynb) | `eigenstate_session()`, `get_eigenstates()`, `get_overlap()` |

New notebooks should be added here alongside any new `get_*()`/`Calculation`
capability, following this same table.

## Project purpose and status

Roadmap Tiers 1-3 (core DFT workflow) are implemented and verified; Tiers 4-6 remain
planning-only (see `docs/roadmap.md`). Not implemented: symbolic k-path's
disconnected-segment support, the classical supercell phonon method, scheduler-backed
launchers, MPI, and named `get_*` methods for potential/ELF volumetric plots
(reachable via `run_tasks()` + `parsers.volumetric` today). `src/elkpy/` is the source
of truth — check it directly rather than assuming the docs describe current behavior.
