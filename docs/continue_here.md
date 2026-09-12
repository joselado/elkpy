# Continue here

Working state as of 2026-09-12, written to be picked up cold. **Re-checked on 2026-09-12 without a rebuild**: all **26** patches (0025 and 0026 are new since this header was last written; **0026 is a fix to UPSTREAM Elk**, the only one in the series) apply to a fresh copy of `vendor/elk/` in order with `patch -p1` exiting 0 and **zero** `fuzz` lines, and the binary-free suite is **622 passed / 451 skipped / 0 failed** (`ELKPY_ELK_BIN=/nonexistent/elk PYTHONPATH=src python3 -m pytest tests/ -q`). Neither `./build_elk.sh` nor any `tests/test_calculation_*.py` suite was re-run that day, so the line below is the last word on those. Last verified with a build (2026-09-07): fast suite **525 passed / 12 skipped**, `./build_elk.sh` completes, and all 50 `tests/test_calculation_*.py` suites run. **Both of the two failures that line used to carry are DIAGNOSED AND FIXED (2026-09-12), and in both the Fortran was right and the wrapper's own description was wrong** — `docs/review_findings.md` findings 18 and 19, `docs/design.md` §32. The plane-wave norm exceeded 1 because `genwfpw` does its muffin-tin Fourier transform on the COARSE radial mesh (`lradstp`, default 4), an error that *grows* with the cut-off (+7.7e-5, +3.2e-4, +1.7e-3 at `gmaxvr` 12/16/20) while `lradstp=1` converges to 1 from below as Bessel requires; on the way, `hkmax=` turned out to be inert for task 135 (`init4.f90:24` overwrites it with `0.5*gmaxvr`), so the cut-off is `gmaxvr`. The ELNES cross-section is identically zero at `q=0` because `genexpmat.f90:30-38` returns the IDENTITY there, which zeroes `elnes.f90`'s occupation weight for every fully occupied and every empty state — correct physics, so `q` is now required and `q=0` refused. Running the rest of `tests/test_calculation_spectra.py` — which had never been run either — then found **two more, one of them upstream Elk's**: task 22 aborts in malloc because `bandstr.f90:40` declares `elm` `real(4)` where `genlmirep` writes `real(8)` (patch **0026**, findings 20-21), and $\mathbf j_p$ on Elk's reduced k-set keeps a residue of 3e-2 to 6e-2 that does **not** shrink with the mesh, because the cancellation is time reversal's and time reversal is not in `nsymcrys` (`get_paramagnetic_current` now forces `reducek=0`, where the same number is 2e-14). `tests/test_calculation_spectra.py` is now **38 passed / 0 failed** and `tests/test_calculation_optics.py` **11 passed / 7 skipped**, both against a binary rebuilt with all 26 patches.

(Historical, and the reason the command below is worth keeping now that GitHub CI
has been removed from the repository.) CI's `unit-tests` job had been red on every push since at least
2026-09-07, and not for an infrastructure reason: constructing a `Calculation`
resolved the `elk` binary eagerly, so the 28 deliberately binary-free tests in
`tests/test_tasks_spectra.py` and the mixin half of
`tests/test_calculation_params.py` — both of which say "no Elk run" in their own
docstrings — errored on any machine without a build. `Calculation.launcher` is
now a lazy property. Run the binary-free suite locally with
`ELKPY_ELK_BIN=/nonexistent/elk python3 -m pytest tests/ -q`: **622 passed, 451
skipped, 0 failed** (2026-09-12).

Run the integration suites **one file per `pytest` process**. In a single process the whole set eventually dies inside `jaxlib`: `elkpy` launches Elk with `os.fork()` and JAX is multithreaded, which the interpreter warns about on every call.

**Status in one paragraph.** Both workstreams are on `master` and `master` is pushed.
*Workstream A* — the full-coverage Elk wrapper — is complete and untouched since the
session that landed it; its open items are §2's. *Workstream B* — the JAX port — has
Phase 0 closed (it did not kill the project; the only open item is 0d's timing, which
needs a GPU), Phase 1 closed except for three named items, Phase 2 closed as a set of
forward checks (§§2a-2k, with §2k's electrostatic functional derivative still open), and
**Phase 3's forward criterion met, from a cold start**: the Kohn-Sham loop closes and
converges to Elk's own total energy (3.0e-8 Ha) and Fermi level (1.4e-9 Ha) from a start
0.30 away in potential norm — and §3c now runs bulk Si **from its `elk.in` alone**
(`elkjax.driver.run()`, patch 0024's task 9006), converging from `rhoinit`'s atomic
superposition in 40 iterations to 3.6e-4 Ha and 4.3e-5 Ha, all of which is the frozen
core: swapping Elk's converged `rhocr`/`engykncr` in gives 3.8e-8 Ha and 4.5e-9 Ha.
Phase 3's three *gradient* criteria have not been started. The port's own premise
is demonstrated rather than argued: §2b transcribes only PBE's *energy* densities and lets
`jax.grad` supply the functional derivative Elk gets from Perdew's hand-derived
expression, and the two agree.

**Where to start.** §3's "Start here next session" is the ranked list, and the three
Phase 2 items below it are ordered by cost. In short:

1. **~~A fixed-point iteration.~~ DONE, forward** (§§3a-3b, `src/elkjax/scf.py`,
   patch 0023). `occupy.f90` was the last piece of Fortran between the two
   half-steps; it reproduces Elk's `efermi`/`occsv` **bitwise**. With it the
   loop closes: Elk's converged potential is a fixed point of the map to
   1.8e-15 relative in the muffin tin (1.0e-9 in the interstitial, where the
   start mixes Elk's mixed `vsmt` with its unmixed `vsir`), and a start 0.30
   away converges geometrically to Elk's own
   `engytot` (3.0e-8 Ha) and `efermi` (1.4e-9 Ha), with `|v - v*|` tracking the
   residual all the way down. §2f's two imported scalars (`evalsum`, `engyts`)
   are now computed; only the CORE half of `evalsum` and `engynn` are imported.

   **~~The `epsocc` skip.~~ DONE**: it is a zeroed weight now
   (`density.skip_below_epsocc`), and with `elkjax.response` replacing JAX's
   own `eigh` rule the step is traced and its `jvp` matches a central
   difference (6.6e-9 in the interstitial half; JAX's own rule gives 1.9e-3
   and does not move with the step size).

   **~~Starting from an `elk.in` rather than from Elk's answer.~~ DONE**
   (§3c, patch 0024, `src/elkjax/driver.py`). Task 9006 is `gndstate`'s
   `trdstate=.false.` branch plus the top of its first iteration, so the
   exports describe iteration zero. It also fixed a formula, not a tolerance:
   what may be frozen is $T_{\rm core}$ (`engykncr`), not the core eigenvalue
   sum — worth 2.0 Ha, and invisible to every earlier test because they all
   started at the potential `evalsumcr` was written at.

   **The next step is Gradient A** (linear vs Anderson, the inter-mixer
   difference falling linearly with `epspot`), which needs no reference value
   and is the sharpest available. The forward blocker that stood in front of
   it is gone. After that, `gencore` in the loop closes the last 3.6e-4 Ha.
2. **~~`symrfmt`~~ DONE** (§2g, patch 0018) — the operator is *exported* rather than
   transcribed, so Elk's Euler-angle/Wigner-$D$ construction and its atom bookkeeping
   are not re-derived at all. Applying it takes the pointwise `vxcmt` gap from 5.3e-3
   to 6.4e-14.
3. **~~The Phase 3 boundary.~~ MOVED.** `evalsum` and `engyts` are computed from
   this port's own occupations (§3b); `engynn` is a lattice constant and stays
   imported, as does the CORE half of `evalsum` — patch 0023 exports it, an
   input at fixed potential exactly as `rhocr` is. What is still absent is
   `eveqnsv`: a spin-polarised or spin-orbit ground state is **refused**
   (`scf.check_scalar`) rather than treated as first-variational, so magnetism
   is the next real boundary.
4. **The two Phase 1 leftovers**: the position derivative $d\varepsilon/d\mathbf R$
   (needs moving radial integrals, so it waits on Phase 2); and smeared occupations at
   *second* order, which needs a Chebyshev expansion of the Fermi function —
   `sign_projector` is hard-window only. The `lax.scan` item is done (§1m).

**Five rules this port has paid for, in the order they will bite again.** Each is a
measurement in `docs/jax_port_phase{0,1,2}.md`, not a maxim.

* **A green gradient test does not validate a transcription.** AD and finite differences
  differentiate the *same* function, truncated the same way, so they agree on a wrong
  one. Dropping `genylmv`'s $4\pi(-i)^l$ prefactor still passes the exact `dmatch`
  identity to 7e-16. Every derivative check needs a forward check beside it, and the
  strongest available without Elk is the quantity's own defining equation. Four
  instances so far.
* **Always compare forward mode against reverse mode.** For a scalar-in, scalar-out
  function they are the same number, so disagreement is proof on its own and costs
  nothing. A single real diagonal direction in reverse mode made the *naive* projector
  rule look correct; the same direction in forward mode is wrong at 1.3e-2.
* **Elk discretises the exact continuum derivative; AD returns the exact derivative of
  the discretised energy.** These differ, and the difference is not an error in either.
  §2b's 2.4e-5 PBE residual tracks the reduced gradient $s$, exactly as that reading
  predicts. This will recur through all of Phase 3.
* **When a field-valued quantity disagrees, decompose it in the basis the code stores it
  in before ruling anything out.** §2d spent six decisive-looking eliminations on a
  scalar residual; the $l$ decomposition identified the cause in one run, by showing Elk
  holding exact zeros in three channels where the transcription held $10^{-3}$.
* **An export is only as consistent as the point in the SCF loop it is taken at.** The
  radial functions belong to the *previous* iteration's potential unless `genapwlofr` is
  called first (patch 0015), worth 3e-10 in `haa`; `vxcmt` is symmetrised and
  `exmt`/`ecmt` are not (§2d), worth 1.2e-4. Both were found by splitting an aggregate
  comparison, never by tightening it.

**Traps that cost real time, kept as a list.** `vsig` is allocated to `ngvc`, not
`ngvec` — writing `ngvec` of them reads past the end of the array. `rhomt` includes the
core density. `potks` trims `vxcir` and nothing else. `tshift=False` is mandatory
wherever a rotation centre or a plotting plane matters (§28, §31). `lax.map`/`lax.scan`
over the k-axis is the memory default; `vmap(eigh)` materialises every k-point at once.
`jax_enable_x64` must be set before the first array exists. Wrap every JAX invocation in
`taskset` — the thread pins in `.claude/settings.json` do not govern XLA.

Recent commits, newest first:

```
4b13175  Carry the lifted symtype=0 restriction into the index documents
ce50d36  Lift the symtype=0 restriction: symrfir, and a boundary that was wrong
f7ca80f  Record the final verification state
6f5ea27  Carry the differentiability result and its open half into the index
94a3f08  Sharpen the open electrostatic-derivative question
c81ee1d  Differentiate Phase 2, and record the half that does not close
2fc5d0e  Record the final verification state and this session's commits
3a63abc  Untangle the Phase 2 index table
1102ccf  Record the second-structure check and what it found
86d5392  Check the density chain on a second structure, and fix what it found
2c75bde  Run the density chain to Elk's converged arrays
14a495f  Carry the closed loop into the index documents
cb335b6  Close the loop: potential to eigenvectors to density
f67750e  Check that the Kohn-Sham potential composes, pointwise
e90b2f4  Carry the completed density chain into the index documents
54b31f7  Close the density chain: rhomagsh and rfmtctof
32a557a  Carry the completed valence density into the index documents
6684a5e  Complete the valence density: the muffin-tin half too
89c77c0  Write up the interstitial density across the index documents
ff2b19e  Build the interstitial valence density from the eigenvectors
308f673  Record the session's commits and the verification state
8a8044e  Export symrfmt's operator, closing 2d's remaining consequence
be38b85  Carry the total energy into the two index documents
6b837bd  Assemble the total energy, and correct a prediction 2d got wrong
9f31ccb  Write up the Poisson solve across the four documents
1389745  Transcribe Elk's Weinert Poisson solve
19de80c  Retire the scan item from the three places that still list it as open
264c1ec  Scan the Newton-Schulz tape instead of unrolling it
```

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

**JAX is installed (0.7.1) but there is no CUDA jaxlib**, so `jax.devices()` is
`[CpuDevice(id=0)]` and Phase 0d's timing question cannot be answered here. Two rules
that are not obvious and were measured, not assumed: the `OMP_NUM_THREADS=1` above does
**not** govern XLA's CPU backend — one 1200x1200 `jnp` matmul spawns 40 threads under it
— so wrap every JAX command in `taskset -c 0-3`; and the port's production shape is
26.8 GiB of H and S against ~28 GiB available, so it is compiled ahead-of-time and never
executed. `docs/jax_port_status.md` has the full set.

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

1. **~~The three high findings are one bug.~~ DONE (2026-09-12).**
   `magnetism_manybody._run_reusing()` is the non-wiping dispatch the module
   header always said it wanted; tasks 601 and 701 go through it, and
   `get_anomalous_entropy` now runs `(205, 241, 270)` with the phonon
   family's own `lmaxi>=2` blocks. Fixing them turned up a fourth claim of
   the same kind: task 601 cannot be used "at a different `wmaxgw`/`tempk`
   with the same screening" either, because both set the Matsubara count and
   `getcfgq` stops on a differing record dimension. Details in
   `docs/review_findings.md`; 14 binary-free tests in
   `tests/test_tasks_restart_chains.py`. The description below is what was
   wrong, kept because it is the clearest statement of the trap.

   `get_gw_self_energy(reuse_epsinv=True)`
   (task 601), `get_ulr_ground_state(from_state=True)` (task 701) and
   `get_anomalous_entropy`'s chain all select a task whose purpose is to read a
   file that `_run_resumed`'s unconditional `shutil.rmtree` (`calculation.py:280`)
   has just deleted. The first two are dead under **any** label. Fix by routing
   through `magnetism_manybody._run_dependent()` (the non-wiping mode already
   written for this) and, for the third, using task **241** not 240 plus
   `lmaxi>=2` — `tasks/phonons.py:1180` already gets that chain right.
2. **~~The six medium findings.~~ DONE (2026-09-12).** All of 4-9, with the
   details in `docs/review_findings.md`. Two changed a NUMBER rather than a
   docstring: `get_stress()`'s pressure was out by `scale**2` (measured 105.3
   for Si, exactly 10.26²), and `get_electron_phonon_coupling`'s `couplings`
   were half the `lambda` in the same dict. Tests:
   `tests/test_tasks_medium_findings.py` (12, no binary) and
   `tests/test_calculation_medium_findings.py` (2, ~13 s) — the second was
   confirmed to FAIL without each fix before being kept. Suites after them:
   spectra **38 passed**, optics **11 passed / 7 skipped**, groundstate
   **12 passed / 7 skipped**, phonons **15 passed / 2 skipped**, and the
   binary-free set **526 passed**. The descriptions
   below are what was wrong, kept for the reading.

   **`phonons.py:630`** — the `couplings` from `LAMBDAQ.OUT` are exactly **half**
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

## 3. Workstream B — the JAX port

`docs/jax_port.md` (1,623 lines) is the design study; `docs/jax_port_phase0.md`,
`docs/jax_port_phase1.md` and `docs/jax_port_phase2.md` are the running logs of what
each phase actually measured. Phase 0's is the file to read first for the verdict;
Phase 1's is where most of the work is.

Study verdict: **a research project justified by differentiability, not by the GPU** —
SIRIUS already does FP-LAPW on CUDA/ROCm and was built with Elk as its reference, Elk's
hot spots are already near-peak BLAS-3, and all-electron cannot leave FP64. That premise
came from the study's prior-art agent and **has still not been independently verified**;
check it before relying on the verdict, since the verdict rests on it.

### Phase 0 is closed, and it did not kill the project

| item | result |
|---|---|
| **0b** safe-K projector rule | needed, and works: 2.9e-14 against the closed form where naive AD is wrong by 11x |
| **0a** reverse-mode implicit diff through the SCF fixed point | works: ≤1e-14 against a dense IFT reference on four spectra including exactly degenerate ones |
| **0a′** second order | works, but only with an eigensolver-free projector (`sign_projector`); the `eigh`-based rule gives `NaN` |
| **0c** `jax.jvp(match)` vs `dmatch.f90` | exact to 4e-16, forward and reverse |
| **0e** compile cost | flat in the shapes; superlinear (≈1.85) in HLO op count |
| **0d** `vmap` vs `lax.map` | memory settled (0.41 GiB vs 40.2 GiB); **timing needs a GPU** |

**The honest qualification, and it is not small: nothing in Phase 0 has touched an LAPW
Hamiltonian.** 0a/0a′ pass on a toy with an exactly degenerate spectrum, 0b's overlaps
are synthetic with a prescribed κ(S), and 0c's radial derivative matrices are inputs
rather than Elk's own `apwfr`. **0b is now closed against real matrices** by Phase 1f
(§1f of `docs/jax_port_phase1.md`); 0a/0a′ are not.

### Three findings worth carrying into Phase 1

None of these is in the study, and each cost a wrong answer to find.

1. **A green gradient test does not validate a transcription.** Measured twice on 0c:
   dropping `genylmv`'s 4π(-i)^l prefactor, and handing `match` a transposed derivative
   matrix, both leave the exact `dmatch` identity passing at 3e-16 while the
   coefficients are wrong by O(1) — a constant factor and a basis change both commute
   with d/dr. Every AD check needs a forward check beside it, and the strongest one
   available without Elk is the quantity's *own defining equation*, not a comparison of
   its pieces. The same shape appears in 0a: a perturbation that **respects** the
   symmetry protecting a degeneracy hides the projector bug completely (1.5e-14
   symmetric vs 4.7e-1 symmetry-broken, same Hamiltonian), so test along general
   directions and break the symmetry. And always compare forward mode against reverse —
   for a scalar-in scalar-out function they are the same number, so disagreement is
   proof on its own and costs nothing.
2. **Never unroll a Pulay-type mixer.** Unrolled Anderson reaches a forward value good
   to 1.8e-13 while its gradient is wrong by 1e17 to 1e32 relative, across five decades
   of the mixer's internal ridge; unrolled linear mixing converges normally. Elk's
   default `mixtype=3` is a Broyden scheme of the same shape. Note also that "the
   implicit gradient agrees between mixers" proves nothing — the `custom_vjp` backward
   pass only ever sees (θ, v*).
3. **`scan` repeated structure; unroll only what must be.** Compile time does not care
   about tensor size (0.46 s at both 200×4 and 3000×100) but is superlinear in HLO op
   count; a `lax.scan` over 4x more radial points costs nothing, while unrolled
   Gram-Schmidt over 8→128 columns costs 0.44 s→32.4 s. And `vmap` over the k-axis is
   not a benchmark question: 40.2 GiB does not fit on a 40 GB device.

### Start here next session

The open items below are a mix of done and outstanding; this is the ranked entry
point. Items 1-8 are Phase 1 and all are closed or deliberately withdrawn; 9-11 are
Phase 2, which is now under way (`docs/jax_port_phase2.md`). The Phase 1 leftovers
named in item 12 need no Elk run and are the cheapest work available.

1. **~~Wire the Cholesky-reduced eigensolve to the safe-$K$ projector rule.~~ DONE**
   (`docs/jax_port_phase1.md` §1f). The rule is needed and it works on Elk's own
   matrices: at bulk Si's $\Gamma_{25'}$ triplet the naive route is wrong by 6.7e-2 in
   forward mode and returns `NaN` in reverse, against 4.1e-12 for the safe one; at a
   generic $k$ both agree to 8e-14, which is what makes it a measurement of the rule
   rather than of the fixture. The refusal is `occupied_window`, and the projector
   reproduces Elk's own occupied subspace to 6e-14.

2. **~~Remove the two poles in `elkjax.lapw.match`.~~ DONE** (§1g). The $k$-tangent of
   the assembly was `NaN` at $\Gamma$ — and at **every $k_z=0$ point of a slab cell**,
   since $\mathbf G=(0,0,\pm2\pi/c)$ is then in the basis — while the value there was
   exact. Two independent causes, both at a basis function with $\mathbf G+\mathbf k$ on
   the $z$-axis: $Y_{\ell m}(\hat v)$ has no derivative where the direction is undefined,
   and $\lvert\mathbf G+\mathbf k\rvert$ is $\sqrt\cdot$ at zero. Fixing only the first
   leaves the second, which is invisible until it is. `match` now regroups the product as
   a regular solid harmonic (`solid_harmonics`, a polynomial in the Cartesian components)
   times $j_\ell^{(i_o)}(x)x^{i_o-\ell}$ (`spherical_bessel_scaled`, a function of $x^2$
   alone), and forms neither $\hat g$ nor $\lvert g\rvert$ — which is why `gkc` is no
   longer an argument of `match`. `spherical_harmonics`/`spherical_bessel` are untouched
   and still what item 0c checks. Forward values are unchanged: Elk's `apwalm`
   element-wise at generic $k$ **and now at $\Gamma$**, the `dmatch` identity, and all
   six assembly blocks are all still green. $dP/dk$ at $\Gamma$ through the multiplet
   goes from `NaN` to 1.4e-14.

   One caution: the *naive* projector's $k$-derivative at $\Gamma$ is **still** `NaN`.
   The two fixes are independent, and only together give a projector derivative at a
   high-symmetry point.

3. **~~The negative test at an exact degeneracy.~~ DONE** (§1h), and it corrected the
   study's own fixture suggestion. "h-BN at $\Gamma$" cannot work and neither can Si at
   $\Gamma$: at a time-reversal-invariant momentum every branch is even in $\mathbf k$, so
   the sorted branches never exchange and AD and central FD both correctly return zero.
   Degeneracy is not enough — the branches must cross LINEARLY. Graphene at $K$ (2 atoms,
   `rgkmax=6`, under two minutes with the ground state) gives AD $\pm0.184$, central FD
   $\pm0.0009$ and one-sided $\mp0.376$ on the Dirac pair, agreeing only in the trace.

4. **~~Smeared occupations, on a real metallic LAPW matrix.~~ DONE** (§1i,
   `elkjax/phase1_smearing.py`, `tests/test_calculation_lapw_smearing.py`). The premise
   was half right and the correction matters: smearing is what makes the branch's
   *value* nonzero, but whether it **fires** is set by the assembly's roundoff, not by
   the physics — and graphene, the metal, does not fire (Dirac pair split
   $3.4\times10^{-7}$ Ha, $5\times10^4\times$ tol) while **gapped silicon does**
   ($1.1\times10^{-15}$ Ha inside $\Gamma_{25'}$). So both fixtures were needed, and each
   shows only one of two distinct failures: on graphene the quotient is exact to
   $10^{-13}$ and JAX's eigenvector rule fails *proportionally in $w$* (3.9e-9, 3.9e-8,
   2.8e-7 over two decades); on Si that rule returns `NaN` outright while the quotient is
   wrong by up to 3.7e-3 — and at Elk's default `swidth` it is **exactly zero** against a
   true kernel of $-3.2\times10^{-2}$, the two occupations being bitwise equal. The
   oracle is analytic: `reference.fermi_divided_difference_kernel`, the logistic
   difference quotient in a form with no subtraction in it, so it arbitrates between the
   two branches rather than being a third opinion. The self-consistent Fermi level came
   with it — `mu` is now a differentiable primal of `smeared_projector`, so
   `fixed_number_projector` is just the composition; §8b's $d\mu$ rule is tested,
   gauge-invariant at a multiplet, and **dominant rather than corrective** (dropping it
   is wrong by 69x at a half-filled level). Its denominator is a physical singularity,
   and `check_fermi_level_determined` refuses a gapped system at small width.
   Two things left behind: §8b's k-point weights $w_j$ cancel at a single $k$ and so are
   still untested (a zone-summed Fermi level is Phase 2), and the safe rule's floor on Si
   is $2\times10^{-9}$ rather than $10^{-13}$ because the *other* pair sits $67\times$
   ABOVE the tolerance and therefore takes the cancellation-prone quotient — the
   tolerance is a cliff. **That has since been fixed**: the closed form is now inside
   `smeared_projector`'s JVP (`projector.fermi_kernel`), `tol` is *inert* for the smeared
   case — asserted as equality across fourteen decades of it, not as a plateau — and Si's
   floor is 7.4e-12 / 3.1e-12 / 4.8e-14 across the three widths. What remains is not the
   kernel: it is LAPACK and XLA disagreeing about the eigenvalues by 2.1e-14 Ha,
   amplified by the kernel's $1/w$ relative sensitivity; rerunning the same closed form
   on XLA's own decomposition gives 7.9e-14 / 5.5e-15 / 2.6e-15
   (`phase1_smearing.eigensolver_floor`). Knock-on for the suite:
   `reference.fermi_divided_difference_kernel` and `projector.fermi_kernel` now share a
   formula, so the arbiter where it matters is central FD (legitimate, $P=f(H)$ is
   smooth) and `direct_quotient_projector`, which stays literal.

5. **~~Second derivatives on a real LAPW matrix.~~ DONE** (§1j,
   `elkjax/phase1_secondorder.py`, `tests/test_calculation_lapw_secondorder.py`). At
   bulk Si's $\Gamma$ with the $\Gamma_{25'}$ triplet inside the window, `grad(grad)`
   through `sign_projector` agrees with a central difference of the safe rule's own
   first derivative to $3\times10^{-11}$–$2\times10^{-10}$, while **both** `eigh`-based
   routes — the naive one *and* the safe-$K$ rule — return `NaN`, which is the latter's
   own docstring warning measured rather than argued. The control is in the same ground
   state: at a generic $k$ all three agree to $10^{-10}$, so the failure is the
   multiplet and not the order. Differentiated twice in $k$ through the whole assembly
   as well, where refining the FD step gives 1.44e-4, 1.29e-5, 1.44e-6, 1.29e-7 —
   textbook $O(h^2)$, i.e. FD converging **onto** AD. Cost: the Newton-Schulz count is
   set by the top of the basis (17.9 Ha) and not the 0.35 Ha valence manifold, so the
   ratio is 190 and the count 13; 10 steps is not converged, 20 is, to $3\times10^{-14}$
   against both the safe rule and Elk's own subspace, in 42 ms at $n=177$. Left behind:
   `sign_projector` is hard-window only (smeared at second order needs a Chebyshev
   expansion of the Fermi function). The unrolled iteration is **since fixed** (§1m):
   `lax.scan` is the default, worth 230x the instructions and 142x the compile time at
   80 steps and second order — but it saves the graph, not the memory (10%), and it is
   not bitwise, XLA reassociating inside each fused region.

6. **~~The radial integrals, and the spectrum as a function of the potential.~~ DONE**
   (§1k, patch 0015, `elkjax/radial.py`, `elkjax/radial_functions.py`,
   `elkjax/phase1_potential.py`). The plan put this in Phase 2 on the grounds that it
   needs the muffin-tin potential; it does, but a converged potential is an *input* that
   can be exported and held fixed exactly as `STATE.OUT` already is. Everything is
   machine-precision against Elk element-wise: `oalo`/`ololo` 2.4e-16,
   `haa`/`hloa`/`hlolo` 2.0e-16, `apwfr`/`apwdfr` 1.3e-14, `lofr` 7e-15.

   **The finding to carry forward is the channel split.** Frozen-basis vs full AD on
   bulk Si: a purely SPHERICAL perturbation gives frozen-basis derivative **exactly
   zero** and 100% basis response; a purely NON-SPHERICAL one gives 7.0e-16 basis
   response. Both are structural. `hmlrad`'s $\ell_2=0$ element is
   $\langle u|\hat Hu\rangle$ — `genapwfr` has already applied $\hat H$, the radial
   functions being that operator's own solutions — so the spherical potential never
   appears in a radial integral and reaches $H$ only through the basis; and
   `genapwfr`/`genlofr` integrate in the spherical part alone, so the non-spherical
   potential cannot move the basis. **Consequence for Phase 2**: a chain that produced
   a perfectly correct $\delta v_s$ and fed it to a frozen LAPW basis would return
   ZERO for the spherical channel while passing the study's pointwise $v_{xc}$ check.

   **And one trap that no gradient check can catch, hit here for real.** A perturbed
   potential reaches $H$ and $O$ twice: inside the radial integrals, and through the
   matrix $D$ of radial derivatives at $R_{\rm MT}$ that `match` inverts for `apwalm`.
   The first version rebuilt `apwfr` and left `apwalm` exported — a basis frozen at the
   sphere boundary and moving inside. AD and FD then differentiate the same truncated
   function and agree to 4e-10, the closed form is unaffected, and both structural zeros
   survive; only a FORWARD check sees it (`radial_functions.derivative_matrices` against
   the exported `dmat`). Phase 0's "a green gradient test does not validate a
   transcription", verbatim. It was worth 21% of the full derivative.

   Two more things worth remembering. The export was internally inconsistent by one
   mixing step until 0015 called `genapwlofr` (`gndstate` mixes the potential AFTER
   building the radial functions) — found by splitting the comparison into
   potential-free and potential-carrying integrals, which is a much better diagnostic
   than a single aggregate. And the map from the potential to the radial integrals is
   AFFINE, not linear: its constant part is that same $\ell_2=0$ block, and carrying it
   into $\delta H$ flips the sign of the closed-form reference rather than merely
   degrading it. (Filling that slice with the POTENTIAL integral instead is what turns
   Elk's matrix perturbation into the Hellmann-Feynman term — same slice, three uses,
   and the wrong one gives a plausible number every time.)

   **"Full minus frozen" is NOT the basis relaxation**, and reading it that way was an
   error corrected in a later commit. It is the Hellmann-Feynman term Elk's $\ell_2=0$
   bookkeeping hides PLUS the relaxation. `phase1_potential.hellmann_feynman` computes
   $\sum_n\langle\psi_n|\delta V|\psi_n\rangle$ explicitly, and the genuine relaxation
   depends on the SHAPE of the perturbation by a factor of 100: **29%** of the derivative
   for white noise down to the nuclear cusp, **0.30%** for a smooth valence-region bump —
   roughly what an SCF update does. Quote the second, not the first.

7. **~~The adversarial `soc_scale` sweep.~~ WITHDRAWN as written** — `soc_scale`
   cannot move the first-variational spectrum at all. `socfr` enters only
   `eveqnsv`; it appears zero times in `hmlfv`/`olpfv`/`hmlaa`/`hmlalo`/`hmllolo`/
   `olpaa`/`olpalo`/`olplolo`/`eveqnfv`/`hmlrad`/`olprad` (grep-verified). The
   first-variational Dirac point is therefore exactly degenerate at every scale and
   the sweep is "refuse always", not a threshold crossing. Its actual content — a
   refusal that is *required* to fire at a stated threshold — is delivered in §1f by
   cutting Si's $\Gamma_{25'}$ triplet instead, with no extra ground state.
   Reinstating a continuous sweep needs the second-variational step.

8. **~~The position derivative.~~ HALF DONE** (§1l, `elkjax/phase1_position.py`). The
   frozen-potential half — the rigid-muffin-tin picture, where positions enter only
   through `match`'s structure factor — is done and pinned by an exact identity rather
   than a finite difference: **rigid translation of every atom cannot move the
   spectrum**, since the matrix transforms by a diagonal unitary
   $U=\mathrm{diag}(e^{i(\mathbf G_i+\mathbf k)\cdot\boldsymbol\delta})$. Measured
   1.8e-15 Ha on Si and 3.8e-15 on h-BN. $\tilde\Theta$ itself is BUILT
   (`hamiltonian.characteristic_function_matrix`, closed-form geometry — element-wise
   against Elk's own $O^{\rm I}$ to 1e-16, which also closes the overlap half of item
   1c), so the only response supplied by hand is the interstitial Kohn-Sham potential's,
   $\tilde v(\mathbf G)\to\tilde v(\mathbf G)e^{-i\mathbf G\cdot\boldsymbol
   \delta}$. Without it: 2.6e-5 Ha and 1.4e-3 Ha. Note building $\tilde\Theta$ does
   NOT monotonically shrink the residual — on h-BN it grows, the two omissions having
   partly cancelled — so a smaller residual is not evidence of a more correct assembly.
   It does move the single-atom derivative by 7.5% (Si) and 35% (h-BN).

   **The finding: the FORWARD form of that null is sharper than the gradient form.**
   With the interstitial response left out, the error is $O(\delta^2)$ on Si (ratios
   4.01, 4.00 per halving) and $O(\delta)$ on h-BN (1.87, 1.79) — so the wrong assembly
   satisfies the *gradient* null identically on silicon and only the finite-shift
   comparison separates them on both. That is the third distinct instance in this port
   of "a green gradient test does not validate a transcription" (after 0c's
   $4\pi(-i)^\ell$ and §1k's frozen `apwalm`), and every time the check with teeth was
   forward. Treat it as a rule for Phase 2, not three anecdotes.

   **Still missing for a real force**: the muffin-tin and interstitial Kohn-Sham
   potentials' own response to the displacement, which is Phase 2 and nothing else. What IS unlocked is the Phase 4 isolation that compares a quantity
   with and without `stop_gradient` on `apwalm`, since the moving half now exists.

9. **~~The exchange-correlation functional, and the cell integrals.~~ DONE**
   (§§2a-2c, patch 0016, `elkjax/xc.py`, `elkjax/grid.py`, `elkjax/integrate.py`).
   A `GROUNDSTATE` query exports the converged density and potentials on Elk's own
   grids, taking no k-point. LDA (`xc_pwca`) is pinned four ways — Dirac exchange
   1e-15, the exact exchange spin scaling 1e-14 (the only check that exercises the
   $\zeta$ machinery), the Gell-Mann–Brueckner high-density limit, and `jax.grad`
   against Elk's hand-coded $v_{xc}$ at machine precision — and reproduces Elk's own
   `vxcir` to **4.4e-16**, but only once `potks`'s `trimrfg` low-pass is reproduced;
   without it 2.5e-5, which looks like a bad transcription and is not one. The cell
   integral gives the electron count to 1.1e-14 against the study's 1e-8 criterion,
   and $E_x$/$E_c$ match `INFO.OUT` to 1e-9.

   **§2b is the port's first demonstration of its own premise.** Only PBE's ENERGY
   densities are transcribed; `jax.grad` supplies the functional derivative Elk gets
   from Perdew's hand-derived expression, which needs $\nabla^2\rho$ and
   $(\nabla\rho)\cdot(\nabla|\nabla\rho|)$ as extra inputs that nothing here computes.
   They agree to 2.4e-5 median, and **the residual is not an error in either**: it
   tracks the reduced gradient $s$ (5.8e-6 in the lowest quarter, 1.25e-5 in the
   highest), because Elk discretises the exact continuum functional derivative while
   AD returns the exact derivative of the discretised energy. Carry that reading into
   Phase 3.

10. **~~The muffin-tin angular transform.~~ DONE, and it found a real property of
    Elk** (§2d). `rbsht`/`rfsht` are transcribed and are mutual inverses to 2.7e-12;
    `exmt`/`ecmt` from the angular-grid density are exact (1.8e-15, 2.8e-16). `vxcmt`
    misses by 5.3e-3 on a scale of 45 **because `potxc.f90:55-58` calls `symrfmt` on
    the potential and the field and not on the energy densities** — so in a muffin tin
    Elk's $v_{xc}=\hat S\,v_{xc}[\rho]$ while $\varepsilon_{xc}=\varepsilon_{xc}[\rho]$.
    On a `symtype=0` ground state the same code gives 1.4e-14.

    Two consequences. Reproducing Elk's SCF on a symmetric cell needs `symlatc`,
    `lsplsymc`, `ieqatom`, `isymlat` exported and `rotrfmt` transcribed — one more
    patch, not a research problem — or a `symtype=0` run. And inside a symmetric
    muffin tin Elk's own $v_{xc}$ is **not** the functional derivative of its own
    $E_{xc}$, so a force or total-energy check better than $\sim10^{-4}$ relative
    there would be evidence of a mistake rather than of success.

    The method note is worth more than the result: six individually decisive
    eliminations on a scalar residual all missed the cause, and the $l$ decomposition
    found it in one run. Decompose a field-valued disagreement in the basis the code
    stores it in *first*.

11. **~~The Weinert Poisson solve.~~ DONE** (§2e, patch 0017, `elkjax/poisson.py`).
    `vclir` to 1.6e-15 relative and `vclmt` to 4e-20 ($l=0$, which carries the
    nucleus and is of order $10^7$) and 7e-14 ($l>0$), on bulk Si and monolayer
    h-BN. One function per step of `potcoul.f90`: `rtozfmt`, `zpotclmt`'s exact
    radial solution, the nuclear term, `zpotcoul`'s pseudocharge and boundary
    matching.

    Patch 0017 exports only `wprmt`, `vcln`, `npsd`/`lnpsd` and `atposc` —
    everything else is rebuilt from what 0016 already carries, including `ylmg`,
    `sfacg` and `jlgrmt` from `elkjax.lapw`, which patch 0013 already pinned
    element-wise. Exporting `ylmg` alone would be 38 MB of text.

    **Two checks owe Elk's `vclmt` nothing.** The monopole identity
    $\sqrt{4\pi}q_{00}=N_{\rm MT}-Z$ recovers $Z=14.000000,5.000000,7.000000$ with
    $N_{\rm MT}$ from `elkjax.integrate` — exact integers out of splines, Bessel
    functions and an FFT. And two mutation tests remove one thing Elk does each
    (the nuclear term *before* the multipoles are read; the outer region's own
    spline weights for $l>l_{\max}^{\rm i}$) and assert the answer moves; both
    mutants are smooth, of the right order and wrong.

    **Three details worth carrying.** `genylmv`'s $4\pi(-i)^l$ prefactor makes
    `ylmg[:,0]` the real constant $4\pi y_{00}$, so `zpotcoul`'s three $l=0$
    special cases *are* the uniform expression — writing them out separately
    would be a second transcription of one line. `vcln` is the $(0,0)$
    coefficient, $\sqrt{4\pi}$ times the potential. And Elk leaves the FFT array's
    slots beyond `ngvec` (7799 of 21952) holding the raw density transform;
    zeroing them is a correction, not a transcription, and the 8e-15 agreement
    says Elk's version is what the round trip needs.

    Nothing here is differentiated, deliberately: Poisson is linear in the
    density, so an AD-versus-FD check would confirm that JAX can differentiate a
    linear map.

12. **~~The total energy at fixed input potential.~~ DONE** (§2f,
    `elkjax/energy.py`). Every density-functional term of `energy.f90` matches
    Elk's own exported scalars to $<10^{-13}$ relative on bulk Si and monolayer
    h-BN, asserted term by term rather than through the total — `engykn` is $+579$
    against `engyen`'s $-1219$, so an error of $10^{-3}$ in either would leave
    `engytot` looking fine at $10^{-6}$. `evalsum`, `engyts` and `engynn` are
    imported (the second-variational step, a zone sum, and the lattice), and the
    module says so; this is the density-functional half, not a transcription of
    `energy.f90`.

    Patch 0017 also exports Elk's thirteen converged scalars, which is what makes
    the comparison term-by-term at full precision instead of `INFO.OUT`'s print
    width — a total that agrees to $10^{-8}$ says nothing about which convention
    is right.

    **§2d's prediction was wrong, and that is the finding.** §2d predicted that
    $E_{v_{xc}}$ would inherit the symmetrisation gap at $10^{-4}$; it agrees at
    $2.7\times10^{-16}$, because $\hat S$ is a group average — an orthogonal
    projection — and $\rho$ is already in its range, so the leak lives entirely in
    harmonics $\rho$ does not have. Measured: potentials differing by
    $5.3\times10^{-3}$ pointwise, overlap with $\rho$ at $10^{-16}$ relative. What
    survives of §2d is that an SCF iteration compares potentials *pointwise* and
    still needs $\hat S$. **A prediction derived from a verified finding is not
    itself verified** — write predictions where a later test will run into them.

13. **~~`symrfmt`.~~ DONE** (§2g, patch 0018, `elkjax/symmetry.py`). §2d's remaining
    consequence, closed: applying the operator takes the pointwise `vxcmt` gap from
    $5.3\times10^{-3}$ to $6.4\times10^{-14}$ on both structures.

    **The design choice is the content.** `rotrflm`'s Euler-angle and Wigner-$D$
    construction has no consumer inside Elk but `symrfmt` itself, so a Python
    re-derivation would have no independent check except agreement with what it
    replaces — and `ieqatom`, `tfeqat` and the *inverse* lattice rotation in the
    rotate-into-equivalent loop would have to come with it. Patch 0018 calls
    `symrfmt` on basis vectors and exports the resulting linear operator instead,
    so none of that is transcribed and none of it can be got wrong here. Same call
    as `wprmt` in 0017, but stronger: there the alternative had a defining equation
    to check against, here it does not.

    One measurement worth keeping: **idempotence is exact ($10^{-16}$) on a cubic
    lattice and only $1.2\times10^{-11}$ on a hexagonal one**, growing with $l$.
    That is Elk's own `roteuler`, whose inverse trigonometry is exact when the
    Cartesian `symlatc` entries are $0$ and $\pm1$. It bounds how idempotent
    `symrfmt` can be, not the operator's accuracy in use.

14. **~~The valence density from the eigenvectors.~~ DONE, both regions** (§2h,
    patch 0019, `elkjax/density.py`). The step that closes the loop's circle —
    every other Phase 2 section goes from a density to an energy, this goes back.
    9.0e-16 and 7.7e-16 in the muffin tins, 7.5e-16 in the interstitial.

    **The reference is what made the muffin tin one routine instead of five.**
    Patch 0019 loops Elk's own `rhomagk` over the k-set into a *local* array, so
    the comparison is against the density before `rhomagsh`, `symrf`, `rfmtctof`
    and `rhocore` — none of which is transcribed. Patch 0018's design again.

    **Two traps, one of them hit.** `evecfv` has $n_{\rm mat}=n_{gk}+n_{\rm
    lotot}$ coefficients and the first version exported only $n_{gk}$: the
    interstitial stayed EXACT (local orbitals vanish there) while the muffin tin
    was smooth, positive, correctly scaled and 100% wrong. And `wfmtsv`'s outer
    region restarts its radial stride one step *past* the inner boundary rather
    than continuing it.

    **Patch 0015's finding recurred**, and the consequence was sharper: the
    reference was built from the previous iteration's radial functions, so the
    query's answer depended on whether `LAPW` had been asked for first — 1.2e-10
    against 9e-16. Fixed with `genapwlofr` in the Fortran, not documented around,
    and the test asks for `DENSITYK` first so it cannot return.

    **The post-processing is done too** (patch 0020): `rhomagsh` at 8.7e-16 and
    `rfmtctof` at 1.0e-15, each against its own exported intermediate rather than
    through their composition. `rfmtctof` is exported as a MATRIX — its spline
    weights come from `wspline` and depend only on the mesh — with **two** per
    species, since it interpolates the whole radial range below $l_{\max}^{\rm i}$
    and the outer region alone above it; using the wrong one reads the inner
    zeros as data, which is smooth, finite and wrong. **With `symtype=0` those
    three stages are the whole of `rhomagv`.**

    **Two of the three results are scope statements, and both are asserted.** On
    a symmetry-reduced mesh it is **16% off**, because `rhomagv` calls `symrf`
    afterwards and this does not — the same shape as §2g's `symrfmt`, in the
    interstitial, where the operator is `symrfir`; closing it would be the same
    kind of patch as 0018. And the residual on the unreduced mesh is `rhonorm`'s
    *uniform* shift, identified by measurement rather than by reading the source:
    switching `trhonorm` off takes it from 2.82e-05 to 2.8e-18.

    The comparison adds no error of its own — `rfirctof` zero-pads, so the fine
    density carries no content beyond the coarse cutoff and `coarsen` is
    lossless, a property asserted on Elk's own `rhoir` rather than assumed.

15. **~~The Kohn-Sham potential, composed.~~ DONE** (§2i). $v_{\rm cl}+\hat S
    v_{xc}$ from three separate modules against Elk's own, POINTWISE: <1e-14 in
    the muffin tin, <1e-13 against `vsir`. Only the second is independent — in
    the muffin tin `vsmt` is `vclmt+vxcmt` by construction, while `vsir` is
    formed inside `potks` *after* `trimrfg` has been applied to `vxcir` and not
    to `vclir`. The mutation test is what makes the tolerance mean something:
    trimming the Coulomb term too is smooth, of the right magnitude, integrates
    correctly against $\rho$, and is wrong by only $10^{-12}$.

16. **~~The loop closed.~~ DONE** (§2j). `density_from_potential` goes potential
    → $H,O$ at every $k$ → eigensolve → density with **nothing in the path
    reading an eigenvector**, agreeing at 5e-11. The missing link was building
    the interstitial blocks from `vsig`/`cfunig` in $G$-space rather than
    recovering $\tilde v_s$ as a matrix in one $k$-point's basis; against Elk's
    own matrices at $\Gamma$, $H$ to 2.7e-15 and $O$ to 5.6e-16.

    **The 5e-11 is measured, not excused**: it is Elk's own two exports of
    `evecfv` disagreeing by 8.5e-9, `elkpy_lapwexport` diagonalising fresh after
    `genapwlofr` while `elkpy_denskexport` reads the stored ones. On the
    gauge-invariant occupied projector at $\Gamma$: this solve vs the fresh
    `evecfv` 1.6e-14; Elk's own $H,O$ re-diagonalised vs its stored `evecfv`
    1.0e-14; this assembly vs Elk's 2.2e-14; **stored vs fresh 3.7e-11**. Patch
    0015's finding for the third time.

17. **~~The density to Elk's converged arrays.~~ DONE** (§2h tail, patch 0021).
    `rhomagk` → `rhomagsh` → `rfmtctof`/`rfirctof` → `rhocore` → `rhonorm`
    against Elk's own `rhomt`/`rhoir`: 3.2e-13 / 1.6e-13 and 5.8e-12. `rhocr` is
    exported, not solved for — the core states are a functional of the
    potential, so at fixed potential the core density is an input like `vsmt`.
    Checked on **two structures**, and the second one found a real bug: `lorbl`
    is a ragged per-species list (boron has 2 local orbitals, nitrogen 3) that
    `np.asarray` only tolerates when there is one species.

18. **~~Phase 2, differentiated.~~ DONE for the XC half; the electrostatic half
    is OPEN** (§2k, `tests/test_calculation_functional_derivative.py`). The
    port's justification is differentiability and every Phase 2 section until
    this one checked a *value*.

    It found a defect first: `elkjax.integrate.cell_inner_product` called
    `np.asarray` on its muffin-tin argument and **could not be differentiated at
    all**, silently, since §2c — every consumer had passed concrete arrays.
    Fixed, with a unit test that differentiates the cell integral and checks the
    gradient *is* the quadrature weight.

    $\delta E_{xc}/\delta\rho=v_{xc}[\rho]$ then holds to **5.7e-17**, with AD
    running through `xc_pwca` *and* the quadrature — a different statement from
    §2a's pointwise check. The 1.25e-5 gap to Elk's `vxcir` is entirely
    `trimrfg`, asserted as an equality with `grid.trim`.

    **The electrostatic half does not close** and is pinned two-sidedly so a
    later fix fails the test rather than quietly passing it: 0.10 Ha absolute,
    which is 33% of $v_{\rm cl}$'s own range but only 4% of the $v_H$ and
    $v_{\rm nuc}$ that nearly cancel to make it. Four candidates are named in
    §2k; the one to test first is whether the discretised Coulomb kernel's
    reciprocity is only as accurate as the pseudocharge construction, which
    would make this a property of the Weinert method rather than a bug.

19. **~~`symrfir`, and the `symtype=0` restriction.~~ LIFTED** (patch 0022).
    `symrfir` acts in $G$-space as a permutation plus a phase, so the whole
    operator is two small arrays; exported for §2g's reason. On Elk's **default**
    mesh (3 k-points, 48 operations) against the converged arrays, the
    interstitial goes 0.165 → **1.1e-15** and the muffin tin 8.0e-6 →
    **1.9e-13**. `converged_density` detects `nsymcrys > 1` rather than being
    asked.

    **One detail nothing structural catches**: `rhomag` calls `symrf` *before*
    `rfmtctof`, so the array is on the coarse mesh and §2g's operator needs the
    **coarse** boundary `nrcmti`. Passing the fine `nrmti` treats every coarse
    point as interior — still a rotation, still a smooth positive density, wrong
    at 8e-6 where the correct one is 1.9e-13.

    §2h's own claim needed correcting with it: its reference is written *before*
    `symrf`, so the un-symmetrised zone sum matched it exactly even on a reduced
    mesh. The 16% was never the k-sum, only the missing post-processing.

20. **The Phase 1 leftovers**, neither of which needs an Elk run. (`lax.scan` over the
    Newton-Schulz tape is **done**, §1m.) Smeared occupations at **second**
    order, which needs a Chebyshev expansion of the Fermi function — `sign_projector`
    is hard-window only, and §1i removed the tolerance from the smeared *first*
    derivative, not the `eigh` from its JVP. And §8(b)'s k-point weights, which cancel
    at a single $k$ and stay untested until a zone-summed Fermi level exists.

One caution carried from this session for whatever comes next: the AD-vs-`genpmatk`
comparison agreed to 0.2-1.4%, which is a **physics** agreement, not a correctness
oracle. Keep a finite difference of the *same* code path beside every gradient check,
as the control that separates an AD bug from a real basis effect.

### What is left, and what each needs

- **~~Patch 0013 — the one Fortran job.~~ DONE** (`a8f45cc`, `docs/design.md` §33). A
  `LAPW` query on the task-9002 session writes `apwalm`, the derivative matrices `D`,
  the radial-function tails, `H`/`O` with their interstitial parts separated, and Elk's
  own `evalfv`/`evecfv`. `elkjax.lapw.match` reproduces Elk's array to **1.9e-15** in
  the `omax==1` branch and **8.0e-13** in the general solve branch — the latter reached
  through a generated `apword=2` species file, since every species file Elk ships sets
  `apword=1` and never enters it. Three traps are in §33 rather than folklore: `tefvr`
  must be forced false or the muffin-tin APW-APW block loses its imaginary part
  silently; only the upper triangles are filled by Elk; `D` must be recomputed because
  `zgesv` destroys it. `apwfr`'s normalisation is no longer an unverified assumption,
  but `D` is still an **input** to `elkjax.lapw` rather than built from `genapwfr` —
  that construction is Phase 1.
- **~~Phase 1's first build step, `hmlfv`/`olpfv`.~~ DONE.** `src/elkjax/hamiltonian.py`
  builds the muffin-tin half of $H$ and $O$; all six blocks agree with Elk's own to
  machine precision on bulk Si (`apword` 1 and 2) and monolayer h-BN, and the assembled
  pair reproduces `evalfv` to 9e-15 Ha. Patch **0014** appends the radial integrals and
  the Gaunt array to 0013's export. Numbers, fixtures and what each one alone catches:
  `docs/jax_port_phase1.md`. Three things to carry forward from it. The
  **interstitial blocks are taken from the export, not built** — $H^{\rm I}$ needs
  $V_s$, which is Phase 2 — so the next forward step is the radial integrals
  (`genapwfr`/`genlofr`/`hmlrad`/`olprad`), which also need Phase 2's muffin-tin
  potential; that makes the **Cholesky-reduced `eigh`** the only Phase 1 forward item
  reachable without Phase 2. `hlolo` is **not symmetric** under exchanging its two
  local orbitals, so only the half Elk evaluates may be used — invisible on silicon,
  1.3e-2 Ha on h-BN's nitrogen. And any element-wise comparison against Elk's $H$ has a
  ~1e-12 floor from `hmlaa`/`hmlalo`'s own `zaxpy` guard, measured, not assumed.
- **~~Nothing assembled has been differentiated.~~ The $k$-derivative is done**
  (`eigenproblem_at`, `first_variational_eigenvalues`). $V_s$ is recovered as a matrix
  from the two exported interstitial blocks, so the whole spectrum is a differentiable
  function of $k$ with no new Fortran; AD agrees with central FD of the same function to
  1e-9. **The finding to carry forward**: against `genpmatk` the two agree only to
  0.2-1.4%, and that gap is FLAT in `rgkmax` while shrinking 4x with `apword` — it is
  the muffin-tin linearisation, not the plane-wave cutoff, because Hellmann-Feynman
  needs a $k$-independent basis and LAPW's is not one. So `genpmatk` is not a
  machine-precision oracle for a band velocity (this is why §22's own test needs
  `rel=2e-2`), and the remaining gradient criteria must FD the same code path.
- **~~Nothing here is wired to `projector.py`'s safe-$K$ rule or a per-run
  $\kappa(O)$.~~ DONE** (§1f). Two findings from doing it. A symmetry-required
  degeneracy comes out of Elk's assembly split by anywhere between $10^{-15}$ and
  $10^{-5}$ Ha — Si's $\Gamma_{25'}$ triplet splits *unevenly*, 5.1e-15 for one pair and
  3.53e-9 for the other, the second being 68x **above** the tolerance — so the refusal
  detects unresolvability, not symmetry, and is necessary but not sufficient; window the
  whole degenerate group as `docs/design.md` §13 already does for Berry curvature. And
  tightening `epspot` 1e-6 → 1e-9 leaves that 3.53e-9 identical to twelve digits, so it
  is not SCF convergence — source still open.
- **The POSITION derivative is still open.** The radial integrals are no longer imported
  (§1k), so half of the old obstacle is gone; what remains is that moving an atom moves
  the *potential* inside its sphere, and where that comes from is Phase 2. What IS
  available now is the frozen-potential position derivative (the `apwalm` structure
  factor alone) and, with it, the isolation `docs/jax_port.md` §Phase 4 asks for — the
  same force with and without `stop_gradient` on `apwalm` — because the other half of
  that comparison now exists.
- **0d's timing needs a GPU instance.** Do not fake it on CPU; the study's own 1.03x CPU
  number settles nothing. The memory half is already answered and points the other way.
  **This is now the only open Phase 0 item.**
- **~~κ(S) for a real LAPW overlap has never been measured.~~ DONE**
  (`docs/jax_port_phase0.md` §0b(ii); rerun with
  `python3 -m elkjax.phase0b_overlap <workdir>`). κ(O) ≈ 5e3 at the standard
  `rgkmax=7`, giving a tolerance of ~1.6e-11 Ha on Si and 6.5e-11 Ha on h-BN. Three
  results that change how it should be used. It is set by the **cutoff, not the matrix
  size** — at rgkmax=8, Si's 227x227 overlap and h-BN's 2118x2118 have the same κ to
  within the spread across k-points, while raising `rgkmax` 7→8→9 takes Si from 5e3 to
  2.6e4 to 2.1e5 and h-BN 7→8 from 5.0e3 to 3.5e4. So the tolerance must be recomputed
  per run, and 0b(ii)'s "repeat at n~1000" varies the wrong axis. §8b's cheap Cholesky-diagonal
  estimate is not merely a 140x-low lower bound, it is **uninformative**: it moves 8.05
  → 9.25 across a 74-fold range of κ, so the "low by" factor grows to 39,000x purely
  because the truth grew. And the tolerance should use ‖L⁻¹HL⁻ᴴ‖, not ‖H‖ — 3x larger
  here. **Knock-on for Phase 1**: the study's adversarial `soc_scale` sweep
  3000 → 3 stops three to six orders of magnitude short of the gap at which its own
  refusal criterion is meant to fire, so it has to be extended below `soc_scale=1`
  (§0b(ii) point 5 — the spread is the unknown curvature of the gap in the scale).
- **~~The self-consistent Fermi level.~~ DONE** (§1i). `projector.fermi_level` is a
  `custom_jvp` whose primal is a bisection (never differentiated) and whose tangent is
  §8b's closed form. Tested against FD of a re-solved μ on synthetic spectra and on both
  LAPW fixtures. Two findings: the rule is gauge-invariant at a multiplet even though it
  uses `A_jj`, because inside a degenerate group f' is constant and the sum is a trace;
  and its denominator vanishing is *physics*, not numerics — in a gap nothing responds,
  μ is undetermined, and the honest answer is the refusal
  `check_fermi_level_determined` now gives. Still open: the k-point weights, which
  cancel at a single k.
- **`sign_projector` covers hard windows only.** Smeared occupations at second order
  would need a Chebyshev expansion of the Fermi function. Separate work, and the one
  thing §1i and §1j between them do NOT reach: §1i removed the tolerance from the
  smeared FIRST derivative, not the `eigh` from its JVP.

### The code, and how to run it

`src/elkjax/` is a **sibling package** to `elkpy`, not a submodule: Phase 0 is
explicitly "no Elk code" and elkpy's fast suite must not acquire a `jax` dependency.

```
memory.py      RLIMIT_AS cap, AOT compile cost, the production-shape arithmetic
reference.py   closed-form eigenproblem derivatives in NumPy — the reference for 0a/0b
projector.py   the safe-K custom_jvp rule, and sign_projector (eigensolver-free)
fixedpoint.py  custom_vjp + GMRES on the transposed operator; the mixers
scftoy.py      a Kohn-Sham-shaped fixed point with an engineered degeneracy
lapw.py        match, gengkvec, gensfacgp, genylmv, sbessel
phase0a/b/c/e  the experiment drivers, one per item
phase0b_overlap.py  item 0b(ii): kappa(O) on real Elk overlaps. The one module here
               that imports elkpy and needs no JAX at all
hamiltonian.py Phase 1a: olpfv/hmlfv, the muffin-tin half of H and O; the
               k-dependent assembly; the Cholesky reduction, the per-run
               tolerance, the occupied projector and its refusal
phase1_projector.py  item 1f: the rule wired to the eigensolve, on real matrices
phase1_smearing.py   item 1i: smeared occupations and the self-consistent Fermi
               level, on real matrices -- the first configuration in which the
               kernel's near-degenerate branch is not vacuous
phase1_secondorder.py  item 1j: second derivatives via sign_projector, where both
               eigh-based routes return NaN at a real multiplet
xc.py          items 2a/2b: xc_pwca and PBE.  Energy densities only for
               PBE -- the potential is what jax.grad is for.  The rho -> 0
               guard is written so the GRADIENT survives
grid.py        the interstitial FFT grid: G-vectors at each slot, spectral
               gradient and Laplacian, and trimrfg's |G| <= 2 kmax low-pass.
               Also rbsht/rfsht, the muffin-tin angular transform (item 2d)
integrate.py   item 2c: rfint/rfinp -- the cell integral and inner product
radial.py      item 1k: hmlrad/olprad -- the muffin-tin radial integrals, from
               the potential.  The vsmt packing is the load-bearing part
poisson.py     item 2c: potcoul -- rtozfmt, zpotclmt's exact radial
               solution, the nuclear term, and zpotcoul's pseudocharge and
               boundary matching.  Everything Elk needs that is NOT exported
               is rebuilt here from lapw.py
symmetry.py    symrfmt, applied.  The OPERATOR is exported (patch 0018), not
               transcribed -- rotrflm has no consumer but symrfmt itself
energy.py      item 2f: energy.f90's density-functional terms.  evalsum,
               engyts and engynn are imported and it says so
phase1_scan.py item 1m: the Newton-Schulz tape unrolled vs lax.scan, in
               compile time and HLO instruction count
radial_functions.py  item 1k: rschrodint/genapwfr/genlofr -- the radial
               Schrodinger equation on Elk's own mesh, with Elk's own
               predictor-corrector (transcribed, not improved)
phase1_potential.py  item 1k: the derivative in the potential, split into a
               frozen-basis branch with a closed-form oracle and a full one
phase1_position.py   item 1l: the frozen-potential position derivative, pinned
               by the rigid-translation sum rule
```

```bash
PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase0b   # and phase0a, phase0c, phase0e
ELKPY_RUN_SLOW_TESTS=1 PYTHONPATH=src taskset -c 0-3 python3 -m pytest \
    tests/test_jax_projector.py tests/test_jax_fixedpoint.py \
    tests/test_jax_compile_cost.py tests/test_jax_lapw.py -q      # ~6 min
# needs the elk binary too -- the Phase 1 suites, ~20 min with ground states cached
PYTHONPATH=src taskset -c 0-3 python3 -m pytest \
    tests/test_calculation_lapw_assembly.py \
    tests/test_calculation_lapw_projector.py \
    tests/test_calculation_lapw_smearing.py \
    tests/test_calculation_lapw_secondorder.py \
    tests/test_calculation_lapw_radial.py \
    tests/test_calculation_lapw_radial_functions.py \
    tests/test_calculation_lapw_potential.py \
    tests/test_calculation_lapw_position.py \
    tests/test_calculation_xc.py tests/test_calculation_poisson.py \
    tests/test_calculation_integrate.py \
    tests/test_calculation_muffin_tin_xc.py \
    tests/test_calculation_poisson_solve.py \
    tests/test_calculation_energy.py \
    tests/test_calculation_symmetrise.py -q
```

**`taskset` is not decoration.** `.claude/settings.json`'s `OMP_NUM_THREADS=1` does not
govern XLA's CPU backend — measured, one 1200x1200 `jnp` matmul spawns 40 threads under
it, and `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` changes nothing. The full memory
and CPU rules are in `docs/jax_port_status.md`; the short version is that the
production shape (26.8 GiB of H and S) is never allocated, only compiled.

---

## 4. Decisions waiting on you

**Workstream B**

- ~~Start patch 0013.~~ Done and committed — see §3. It was taken as the obvious next
  step rather than put to you, since it was the single item all three open bullets
  shared. The maintenance commitment is real and is now recorded in `patches/README.md`.
- ~~Merge `jax-port` into `master`.~~ Done, fast-forward, at your instruction, and
  **pushed**: `origin/master` is current. Both feature branches can be deleted.
- Phase 1 at all, or the §9.2 hybrid. **Started** — 1a is done, at the user's
  instruction to continue the port, which settled this for that session but not in
  general. The scope question below is unchanged. Phase 1 does
  not commit to Phase 3: everything built so far reads a converged `STATE.OUT`.
- Phase 0 removed the technical objections; the
  scope question it does not answer is whether the targets that need `dv*/dθ` — phonons,
  Born charges, elastic constants, response functions, ML-XC training, reverse-mode
  inverse design — are the goal. §9.2 puts the hybrid at 13-15 weeks and it delivers
  everything that needs no SCF derivative. **This is the one open Workstream B decision.**

Three things Phase 1 should not rediscover — the first two are now done, the third is
still ahead. `match` was already checked against Elk element-wise, so the first build
step was `hmlfv`/`olpfv`, and that is done too. The G+k set, `atposc` and `rmt` must be
taken from the export rather than regenerated (`gengkvec`'s ordering, `tshift`'s origin
shift and `checkmt`'s radius shrink are three separate ways to get a correct-looking
transcription that cannot be compared element-wise) — `hamiltonian.py` takes every one
of them from the export and never regenerates any. ~~And the projector tolerance must be
computed per run from a real κ, since it is a cutoff property and §8(b)'s cheap estimate
is useless; that bites at the Cholesky-reduced `eigh`, which is the next forward step.~~
Done — `hamiltonian.projector_tolerance` measures it per run from a dense `eigvalsh` of
$O$ and the reduced norm (§1f).

**Workstream A** (unchanged from the previous session)

- The 17 review findings in `docs/review_findings.md`, three of which are one bug.
- Fix `inputfile.py:15` at the source and collapse the two shims, or leave the shims.
- Run the three never-executed integration suites (`spectra`, `optics`,
  `magnetism_manybody`), which will likely surface assertion adjustments.
- ~~`docs/field_report_nibr2.md` — 7 items from a real 45-atom NiBr2 spin-spiral run
  through tasks 9003/9005, none triaged against the code yet.~~ **Triaged; five of the
  seven are closed.** Per-item verdicts are inline in the report, the substance is in
  `docs/design.md` §31, and the summary is in `docs/status.md` §31. Fixed with tests:
  item 1 (`compute_transmission` now refuses an energy outside the exported window,
  which is what the absolute-vs-relative mismatch used to sail through silently) and
  item 2 (`amplitude_weights`' `occmax` is required, no 2.0 default). Fixed without a
  test, since it needs a real threaded run to exercise: item 4 — `OMP_STACKSIZE` via
  `setdefault`, *plus* `RLIMIT_STACK` raised soft-to-hard in a `preexec_fn`, which is
  the `ulimit -s` half the report did not name and the only half that matters at the
  launcher's default `omp_threads=1`. Documented: items 5 (`ramdisk`; elkpy's own path
  is immune) and 7 (`plot2d` transverse aliasing, now a section in both example
  READMEs). **Three things are left, all decisions rather than fixes:**

  - **Item 3 — promote `spin_ldos()`/`tip_image()` into `parsers/transport.py`?** The
    reporter's validated code is staged at `docs/field_report_nibr2_spin_ldos.py` and
    its self-test passes here with no Elk run. The case for it: `get_spin_stm()` costs
    one Elk run per bias *and* per tip direction, each re-reading 7.3 GB of eigenvectors
    on a cell that size, while one 9005 export gives the whole $dI/dV(x,E)$ map at every
    tip direction, because nothing in the export depends on energy. The cost: a new
    public API plus a `physics.tex` part. Not taken unilaterally.
  - **Item 6 — `adopt_ground_state(workdir)`?** Pointing a `Calculation` at someone
    else's converged `STATE.OUT` currently means hand-forging `.elkpy_manifest.json`,
    which is strictly worse than a method that could re-read the adopted `elk.in` and
    *check* the basis signature. Recommendation if taken: adopt by **copying**
    `STATE.OUT` into `self.workdir` and marking the manifest `adopted`, never by
    pointing at a foreign directory — that leaves the wiped-subdirectory invariant
    untouched.
  - **The NiBr2 fixture.** 9003 and 9005 agreed to 0.2% on that system, a stronger
    end-to-end check of §31 than anything in `tests/`. Vendoring it needs a hard cut of
    a 1.85 GB export onto a transverse count sharing the supercell's period, keeping
    *both* sides (the 9003 maps are 1.4 MB each), and re-establishing the 0.2% on the
    cut — a naive decimation would enshrine the aliasing artifact instead of the check.
    Data at `/scratch/work/ladovj1/calculations/NiBr2_elk_stm/`; ask before it is
    cleaned up.
