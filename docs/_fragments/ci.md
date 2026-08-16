# Continuous integration (roadmap Tier 6, item 2)

*Draft fragment for `docs/design.md` (new section) and the "Commands" part of
`CLAUDE.md`. Not yet merged.*

`.github/workflows/ci.yml` implements roadmap Tier 6 item 2: `pytest` and
`scripts/build_elk.sh` on push, plus the "patch series still applies cleanly"
check §8 asks for. Four jobs, staged cheapest-first so a failure surfaces in
seconds rather than after a multi-minute Fortran build:

| job | cost | trigger | what it protects |
|---|---|---|---|
| `unit-tests` | ~10 s | every push/PR | the pure-Python arithmetic (parsers, Wilson loops, gauge invariance, `Structure`/`InputFile`) |
| `patch-series` | ~10 s | every push/PR | §8's isolation contract — `patches/` still applies to `vendor/elk/` |
| `build-elk` | ~5-15 min | every push/PR, after both above | the eight patches still *compile* against the vendored tree |
| `integration` | ~tens of min | `workflow_dispatch` + weekly cron only | end-to-end behaviour against a real binary |

## 1. `unit-tests` — run everything, with no binary present

The job installs the package and runs the **whole** `tests/` tree with
`ELKPY_ELK_BIN` pinned to a path that cannot exist. Every
`tests/test_calculation_*.py` module then self-skips through its existing
`config.default_elk_binary().is_file()` guard, and what remains is exactly the
binary-free set.

This is deliberately *not* a name-based selection such as
`pytest -k "parsers or gauge_invariance or inputfile or structure"`. Measured on
the tree as it stands, that expression selects 109 of 214 tests and **misses
three binary-free tests that live inside an integration module** —
`test_calculation_soc.py`'s `test_soc_scale_requires_spinorb`,
`test_soc_scale_rejects_unknown_species`, `test_soc_scale_rejects_negative`,
which are constructor-validation tests deliberately kept next to the SOC physics
they validate. A filename filter would silently drop them, and would need
maintaining every time a new module mixes the two kinds of test. Letting the
existing skip guards do the selection has no such drift.

The cost of that design is that `pytest` exits 0 when *everything* skips, so a
broken import guard or a stray module-level `skipif` would look green. A final
step parses the JUnit XML and fails if fewer than 100 tests actually ran
(currently 131 without `ase`, 132 with). It is a floor, not an exact count —
raise it only if the number ever legitimately drops.

Matrix: Python 3.9 (the `requires-python` floor — worth having because numpy
resolves to a different major there, `numpy >= 2.1` requiring 3.10+) and 3.13.
`ase` is installed only on 3.13: current `ase` requires Python >= 3.10, so on 3.9
pip would fall back to an old `ase` (and its old scipy/matplotlib) for the sake
of exactly one test — `test_structure.py`'s `from_ase`/`to_ase` round trip is the
only binary-free `importorskip("ase")` in the suite.

## 2. `patch-series` — the §8 check, and why `--dry-run` alone cannot do it

The obvious implementation — dry-run each patch against a pristine `vendor/elk/`
— is wrong for this series, and measurably so: **7 of the 8 patches fail that
way**. The series is cumulative, not independent. 0002 re-touches `elk.f90`,
`modmain.f90` and `readinput.f90` after 0001 has already changed them, and
0004-0008 all edit `src/elkpy_eigenstates.f90`, a file that does not exist until
0003 creates it.

So the job copies `vendor/elk/` to a scratch directory and applies each patch
**for real**, in glob order, exactly as `scripts/build_elk.sh` does — minus the
compile. `--batch` keeps `patch` non-interactive; without it, a patch that cannot
locate its target prompts on stdin for a filename and would hang the runner.

Two failure modes are checked, not one:

- **Hunk failure** — a hard `patch` error, surfaced as a GitHub `::error`
  annotation pointing at the offending `patches/*.patch` and at
  `patches/README.md`'s per-patch upstream-file checklist.
- **Fuzz** — `patch` exits 0 when it places a hunk after ignoring mismatched
  outer context lines, printing only `Hunk #1 succeeded at 32 with fuzz 1`. That
  is precisely the silent-divergence case this check exists to catch, so fuzz is
  treated as a failure. The series applies with **zero fuzz and zero offsets**
  today, so the tripwire is achievable, not aspirational.

Both were verified by deliberately breaking a copy: a reworded declaration in
`gensocfr.f90` (the file 0001 patches) produces a hard failure, and a reworded
*comment* two lines above the same hunk produces fuzz 1 — the second of which
would otherwise have gone unnoticed.

A third, trivial step checks that every `patches/*.patch` has a row in
`patches/README.md`. That table is the documented checklist for an upstream bump,
so a patch added without a row is exactly what goes stale first.

Repository size is not a concern: `vendor/elk/` is 11 MB across 1612 files, fully
tracked in git, and the whole `.git` is 16 MB, so a default `actions/checkout`
plus the copy costs seconds.

## 3. `build-elk` — viable on a public runner

**Assessment: yes, comfortably.** A full serial build of the patched tree —
410 objects from ~75k lines of Fortran, plus the `eos` and `spacegroup`
sub-binaries that `make all` also builds — takes about **3 min 15 s** on the
development workstation (measured from object-file timestamps of an actual
`scripts/build_elk.sh` run, 00:42:11 to 00:45:22). A 2-4 core GitHub-hosted
runner has slower single-thread performance but not dramatically so; 5-15 minutes
is the realistic band, against a 6-hour per-job limit. `timeout-minutes: 60` is a
runaway guard, not an expectation.

Dependencies are exactly what `build-config/make.inc` links:
`gfortran libopenblas-dev liblapack-dev libfftw3-dev` (the make.inc names both
`-lopenblas` and `-llapack`; `libfftw3-dev` supplies both `-lfftw3` and the
single-precision `-lfftw3f`; `-fopenmp` ships with gfortran).

The serial `make` is **not** to be "optimised". Elk's own `src/Makefile` has
implicit ordering dependencies (`mpi_stub.f90` before `modmpi.f90`,
`libxcifc_stub.f90` before `modxcifc.f90`/`moddftu.f90`) that are not expressed
as prerequisites, so `make -j` races nondeterministically. This is upstream's
issue and is not fixable by editing `vendor/elk/`.

Two build-job details worth keeping:

- **The binary is not uploaded as an artifact.** `build-config/make.inc` compiles
  with `-march=native`, and GitHub's hosted runner fleet is heterogeneous, so a
  binary built in one job can `SIGILL` on the next job's runner. Anything needing
  the binary must build it in its own job — which is why `integration` rebuilds
  rather than downloading.
- **The smoke test greps stdout, not the exit code.** Run with no `elk.in`
  present, Elk prints its startup banner, then
  `Error(readinput): error opening elk.in` — and **exits 0**. The exit status
  proves nothing; `grep -q "Elk code version"` proves the binary loaded and ran.

## 4. `integration` — recommended policy

Not on push. The `tests/test_calculation_*.py` suites run real ground states and
cost minutes each; the DFPT phonon tests are 11-13 minutes *per test* even on the
workstation and are already gated behind `ELKPY_RUN_SLOW_TESTS=1`, which stays
unset on hosted runners.

The job therefore runs only on `workflow_dispatch` and a weekly cron, builds Elk
in its own job (see `-march=native` above), and runs a deliberately narrow slice:
`tests/test_calculation_si.py`, the bulk-Si ground-state suite. The heavier
physics suites (berry, z2, z2_3d, quantum_geometry, momentum, parity, spin — and
the h-BN, WSe2 and bismuthene fixtures they build) should be added only after
real runner timings have been observed, one at a time.

## 5. Packaging findings

`pip install -e ".[ase,test]"` into a clean venv works: the only hard dependency
is numpy, `ase` and `pytest` are correctly optional extras, and the codebase
carries no type annotations or 3.10+ syntax, so the `requires-python = ">=3.9"`
floor is honest.

**But the package only functions as an editable / in-tree install.**
`src/elkpy/config.py` derives `repo_root()` from `Path(__file__).parents[2]`, so
after a plain `pip install .` it resolves to the interpreter's `lib/python3.x`
directory: `default_species_path()` points at a nonexistent
`.../lib/python3.14/vendor/elk/species` and `default_elk_binary()` at
`.../lib/python3.14/build/elk/src/elk`. Verified directly in a throwaway venv.
Nothing raises at import — it fails later, at the first calculation, with a
confusing path. This is defensible for now (elkpy is inseparable from the
vendored Elk tree it builds and patches), but it is worth either documenting
explicitly or making `config` fail loudly when `repo_root()` has no `vendor/elk`
in it. CI uses `-e` throughout.

## 6. Deliberately left out

- **Build caching** (`actions/cache` on `build/elk`). Elk's object files depend
  on `vendor/elk/`, `patches/` and `build-config/make.inc`, so the cache key is
  easy to state, but a stale-cache false pass would undermine the very check the
  build job exists for. Worth revisiting only if build time becomes a real
  bottleneck.
- **MPI / parallel builds.** `build-config/make.inc` is serial by design
  (`mpi_stub.f90`), and `LocalLauncher` refuses `nprocs > 1` for that reason.
- **Notebook execution.** Every checked-in notebook needs a real binary and
  minutes of DFT; they are validated by being committed with their real output
  cells, not by CI.
