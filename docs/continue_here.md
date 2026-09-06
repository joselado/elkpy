# Continue here

Working state as of 2026-09-06, so this can be picked up cold. Workstream A landed
on `elk-full-coverage` and is now on `master`; Workstream B continues on `jax-port`.

```
b15c0b4  Add the Elk-to-JAX port design study            docs/jax_port.md
f04eabd  Record the code-review findings as a work list  docs/review_findings.md
27afcca  Extend the Python interface to the whole of Elk 64 files, +19,828
51bab45  (master) Add vertical tunnelling transport, patch 0012
```

`elk-full-coverage` has since been merged; `master` is at `39de3e4`. Workstream B
continues on branch `jax-port` (this section 3, plus `docs/jax_port_phase0.md`).

---

## 1. Environment — read this before running anything

**The Elk binary does not build with the checked-in defaults on this machine.**
`build-config/make.inc`'s link line (`-lopenblas -lfftw3 -lfftw3f`) fails: there
is no OpenBLAS, and FFTW ships only runtime `.so.3` files with no `.so` dev
symlinks. `build_elk.sh`'s fallback also fails, because there is no environment
module system to load from. The line that does work here, via the documented
override:

```bash
ELKPY_F90_LIB="-llapack -lblas /usr/lib/x86_64-linux-gnu/libfftw3.so.3 /usr/lib/x86_64-linux-gnu/libfftw3f.so.3" ./build_elk.sh
```

That builds and runs (verified). Note it links **reference** BLAS, not OpenBLAS,
so every Elk run is substantially slower than this project's timings assume.
`build/elk/make.inc` records what was actually used. `build-config/make.inc` was
deliberately NOT edited — the override is environment-only, so the repo still
carries the workstation defaults.

**CPU budget.** Work in this session was pinned to four cores. `launcher.py`
already enforces exactly that for anything going through `Calculation`
(`omp_threads=1` per process at `launcher.py:83`, times an flock semaphore of
`ELKPY_MAX_CONCURRENT=4` in `/tmp/elkpy_slots`). The gap is a binary run by hand
from a shell, which bypasses both and would take all 12 cores, so
`.claude/settings.json` pins the environment:

```json
{"env": {"ELKPY_MAX_CONCURRENT": "4", "OMP_NUM_THREADS": "1",
         "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}}
```

`.claude/` is gitignored, so that file is **not in the repo** and a fresh clone
will not have it. Recreate it if the four-core limit still applies; delete it if
it does not.

**elkpy is not pip-installed in the interpreter used here.** Every command below
needs `PYTHONPATH=src`, or run `python3 -m pip install -e .` once (which drops
`src/elkpy.egg-info` into the working tree — it is gitignored).

---

## 2. Workstream A — the full-coverage wrapper

### What landed (`27afcca`)

115 `get_*` methods on `Calculation` (was ~40), covering **143 of 146 live task
codes** in `elk.f90`'s dispatch; a typed table of all ~330 `elk.in` input blocks
(`src/elkpy/params.py`) with a validator, renderer and discovery API; five task
mixins under `src/elkpy/tasks/`; 22 new parsers; `spec.py` grown to 152 TASKS /
190 OUTPUT_FILES / 22 templates.

Coverage is counted honestly: a code counts only when a named method places it
in a task list it runs. Tasks 670/680 dispatch to commented-out upstream calls
and are in neither numerator nor denominator. The three misses (2, 201, 271) are
resume variants that read files a previous run left behind.

### Verification status — the thing to keep in mind

**Most of it has never been executed.** Task orderings, input blocks and output
layouts were transcribed by reading `vendor/elk/src/*.f90`, and the parsers were
unit-tested only against fixtures written from that same reading — so a fixture
built from a misread `write` statement passes its own test perfectly. Each
method is labelled `binary-verified` / `format-derived` / `untested` in its
docstring and in `docs/design.md` §32.

```bash
PYTHONPATH=src python3 -m pytest tests/ -q -k "not calculation_"   # 438 pass, ~1 s
```

**Never executed, and worth running first** (they need the binary and take real
time; run them one at a time under the four-core cap):

```bash
PYTHONPATH=src python3 -m pytest tests/test_calculation_spectra.py -q             # 34 tests
PYTHONPATH=src python3 -m pytest tests/test_calculation_optics.py -q              # 16
PYTHONPATH=src python3 -m pytest tests/test_calculation_magnetism_manybody.py -q  # 19
```

Several assertions in those files were flagged by their own authors as likely to
need adjustment on first run — see the last bullet of `docs/design.md` §32.

### The 17 review findings

`docs/review_findings.md` has all of them with the `vendor/elk/src/` line that
proves each, what the verifier re-read, and a suggested fix. Suggested order:

1. **The three high findings are one bug.** `get_gw_self_energy(reuse_epsinv=True)`
   (task 601), `get_ulr_ground_state(from_state=True)` (task 701) and
   `get_anomalous_entropy`'s chain all select a task whose purpose is to read a
   file that `_run_resumed`'s unconditional `shutil.rmtree` (`calculation.py:280`)
   has just deleted. The first two are dead under **any** label. Fix by routing
   through `magnetism_manybody._run_dependent()` (the non-wiping mode already
   written for this) and, for the third, using task **241** not 240 plus
   `lmaxi>=2` — `tasks/phonons.py:1180` already gets that chain right.
2. **`phonons.py:630`** — the `couplings` from `LAMBDAQ.OUT` are exactly **half**
   the `lambda` returned beside them in the same dict. Settled from the source:
   `writelambda.f90` divides by `pi*fermidos` (total, both-spin) where
   `alpha2f.f90` uses `fermidos/2`. Costs an order of magnitude in T_c.
3. **`groundstate.py:269`** — `get_stress()` pressure is wrong by `scale^2`
   (105x for a standard Si cell) because `readinput.f90:2275` scales `avec`
   before any physics.
4. Then the rest, medium before low.

### Also open, not from the review

`src/elkpy/inputfile.py:15` — `_format_value` renders floats with a fixed
ten-decimal format, so anything below 1e-10 becomes the literal string of ten
zeros (`epsband`'s own Elk default is 1e-12), and it quotes every `str`, which
corrupts the verbatim lines of `notes`/`xlwin`/`wann_bands`. **Two independent
shims** (`params.FortranReal`, `magnetism_manybody._RawToken`) work around this
rather than one fix. Findings 13 and 15 are both downstream of it. The float
branch is an unambiguous bug; the string branch is subtler than it looks, since
species filenames genuinely need the quotes — so a `Verbatim` marker is probably
the right design, just not two of them.

Note the fix touches a file every one of the 64 new files depends on, which is
why it was left alone during a parallel merge.

---

## 3. Workstream B — the JAX port study

`docs/jax_port.md` (1,623 lines). Verdict: **a research project justified by
differentiability, not by the GPU** — SIRIUS already does FP-LAPW on CUDA/ROCm
and was built with Elk as its reference, Elk's hot spots are already near-peak
BLAS-3, and all-electron cannot leave FP64. That premise comes from the
prior-art agent and **has not been independently verified**; check it before
relying on the verdict, since the verdict rests on it.

Phase 0 (§6) is designed to kill the project rather than start it. Its two real
questions were: does reverse-mode implicit differentiation through the SCF fixed
point work at all (DFTK shipped forward-only, and `lax.custom_root` with an
iterative `tangent_solve` measurably raises `NotImplementedError`), and does the
safe-K projector rule survive a degeneracy. **The second is answered — it does,
and it is needed** (`docs/jax_port_phase0.md`); the first is untouched.

### The measurement disagreement — SETTLED (Phase 0b)

§8b claimed the occupied-subspace projector "still returns garbage under JAX's default
VJP" whenever the window is gapped; the check recorded here disagreed for the hard
integer window with the multiplet fully enclosed. **§8b was right.** Full numbers and
reproduction in `docs/jax_port_phase0.md`; code in `src/elkjax/`, assertions in
`tests/test_jax_projector.py`.

What settled it was the reference this document asked for — the closed-form
Daleckii-Krein derivative rather than finite differences. Over 3 assemblies x 21
Hermitian directions on the disputed spectrum, worst relative error: naive AD
`1.1e+1` forward and `3.4e+0` reverse, safe-K rule `2.9e-14`, central FD `3.7e-8`.

Two things the earlier check got backwards, both worth remembering:

- **Finite differences were reliable here**, not noisy. `Tr[P M]` is a smooth function
  of `H` whenever the *window boundary* is gapped, however degenerate the interior, so
  central FD is stable across three step sizes. The third assembly's `1.3e-2` was
  therefore AD error, not FD noise.
- **The direction was the whole story.** `e00` — one real diagonal entry — is nearly
  benign in reverse mode (`<1e-7`) and already wrong at `1.3e-2` in *forward* mode on
  the same matrix. Testing one direction in one mode is what produced the false pass.
  For a scalar-in scalar-out function, forward and reverse disagreeing is by itself the
  proof; that check costs nothing and should be in every AD test from here on.

A finding neither document had: **which failure mode appears is the eigensolver's
choice.** LAPACK and XLA split the same engineered pair differently, and at n=1000 XLA
returns it bitwise equal where LAPACK gives 1.5e-14 — so the identical code gives finite
garbage at n=400 and `NaN` at n=1000. §10 item 1's "split by cause" is refined
accordingly.

### Still open in Phase 0

- **0a / 0a′** — reverse-mode implicit differentiation through the SCF fixed point, and
  then `jax.hessian` through it. Not started. Note the rule in `elkjax.projector` is
  **first-order only**: its JVP body calls `jnp.linalg.eigh`, so a second derivative
  falls back on JAX's default eigenvector rule and the hazard returns. Use the analytic
  reference for 0a too — its stated kill criterion is "agreement with central FD", and
  §8(b)'s own measurements show FD cannot serve near the engineered degeneracy.
- **0c** — `jax.jvp(match)` vs `dmatch.f90`. Not started; needs no SCF.
- **0d** — `vmap(eigh)` vs `lax.map` at n=1000: **requires a GPU this machine does not
  have**. Deferred rather than faked on CPU.
- **0e** — compile time and peak memory at production shapes. Do it ahead-of-time
  (`elkjax.memory.compiled_cost`), never by executing: H+S over 100 k-points at n=3000
  is 26.8 GiB and this box has ~28 GiB. See CLAUDE.md's "JAX port" section for the full
  memory/CPU rules, including that `OMP_NUM_THREADS` does **not** govern XLA (measured:
  40 threads under `OMP_NUM_THREADS=1`; use `taskset`).
- **κ(S) for a real LAPW overlap has still never been measured**, and §8b's cheap
  Cholesky-diagonal estimate underestimates a synthetic κ=1e6 by 140x — the dangerous
  direction, since the tolerance is meant to be an upper bound.

## 4. Decisions waiting on you

- Merge `elk-full-coverage` into `master`, or keep reviewing on the branch.
- Fix `inputfile.py` at the source and collapse the two shims, or leave the shims.
- Run the three never-executed integration suites, which will likely surface
  assertion adjustments rather than passing clean.
- Whether the JAX port is worth Phase 0 at all, given that its own verdict says
  the GPU motivation is largely answered by SIRIUS.
