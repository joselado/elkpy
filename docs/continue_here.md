# Continue here

Working state as of 2026-09-06, so this can be picked up cold. **Both workstreams are
now on `master`**: Workstream A landed via `elk-full-coverage`, and Workstream B's
`jax-port` was fast-forwarded in earlier. Nothing is pushed — `origin` is still at the
pre-merge `master`, and is now several commits behind. The `jax-port` branch still
exists and points at an older commit; deleting it is safe.

**Phase 0 is closed and Phase 1 has started.** `hmlfv`/`olpfv` are transcribed and
checked against Elk element-wise (`docs/jax_port_phase1.md`, patch 0014); the only open
Phase 0 item is still 0d's timing, which needs a GPU.

```
NEW      Assemble the LAPW Hamiltonian and overlap in JAX      src/elkjax/hamiltonian.py
NEW      (docs)                                                patches/0014, docs/jax_port_phase1.md
2ae2528  Record the merge in the continuation document
4be2933  Correct the soc_scale shortfall to a range
096d2eb  Close the kappa table with h-BN at the higher cutoff
e559ac7  Record what a LAPW query costs, and what apword=2 pins
81c8021  Check the exported eigenvectors against the exported eigenproblem
35aec88  Measure kappa(O) on real overlaps, retire 8(b)'s estimate   elkjax/phase0b_overlap.py
a8f45cc  Export Elk's LAPW eigenproblem, close 0c against it         patches/0013
c299c14  Bring the continuation document up to date
7a88920  Close a hole in 0c's forward check, pin the harmonic's pole
794175f  Add the Phase 0 verdict and repair two stale status rows
fe03924  Settle Phase 0c: the LAPW matching coefficients             src/elkjax/lapw.py
01bf937  Correct two mechanism attributions, guard sign_projector
adcb2d0  Settle Phase 0e: compile flat in shapes, ~n^1.85 in ops     src/elkjax/phase0e.py
410be6a  Unblock Phase 0a-prime with an eigensolver-free projector
14782ad  Settle Phase 0a: reverse mode survives the eigensolve       src/elkjax/fixedpoint.py
15d910d  Settle Phase 0b: the safe-K projector rule is needed        src/elkjax/projector.py
39de3e4  Add a continuation document                                 docs/continue_here.md
```

**Phase 0 of the JAX port is closed and it did not kill the project** — §3 has the
results, `docs/jax_port_phase0.md` the numbers. Since the last session, patch **0013**
has closed the two items §3 listed as outstanding: 0c's forward coefficients now agree
with Elk's own `apwalm` element-wise, and $\kappa(O)$ has been measured on real LAPW
overlaps. **The only Phase 0 item still open is 0d's timing, which needs a GPU.**
Workstream A is untouched since the previous session; its open items are still §2's.

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
executed. CLAUDE.md's "JAX port" section has the full set.

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

## 3. Workstream B — the JAX port

`docs/jax_port.md` (1,623 lines) is the design study; `docs/jax_port_phase0.md` is the
running log of what Phase 0 actually measured, and is the file to read first.

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
rather than Elk's own `apwfr`.

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
- **The POSITION derivative is still open**, and is harder than the $k$ one: moving an
  atom moves the muffin-tin potential and hence the radial integrals, which are imported.
  So is the adversarial `soc_scale` sweep, and the negative test at a degeneracy —
  nothing here is wired to `projector.py`'s safe-$K$ rule or a per-run $\kappa(O)$.
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
- **The self-consistent Fermi level.** 0a covers smearing at *fixed* μ. Fixed electron
  number adds a second constraint whose rule §8b gives in closed form,
  `dmu/deps_i = w_i f'_i / sum_j w_j f'_j`. Untested.
- **`sign_projector` covers hard windows only.** Smeared occupations at second order
  would need a Chebyshev expansion of the Fermi function. Separate work.

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
hamiltonian.py Phase 1a: olpfv/hmlfv, the muffin-tin half of H and O
```

```bash
PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase0b   # and phase0a, phase0c, phase0e
ELKPY_RUN_SLOW_TESTS=1 PYTHONPATH=src taskset -c 0-3 python3 -m pytest \
    tests/test_jax_projector.py tests/test_jax_fixedpoint.py \
    tests/test_jax_compile_cost.py tests/test_jax_lapw.py -q      # 38 tests, ~6 min
```

**`taskset` is not decoration.** `.claude/settings.json`'s `OMP_NUM_THREADS=1` does not
govern XLA's CPU backend — measured, one 1200x1200 `jnp` matmul spawns 40 threads under
it, and `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` changes nothing. The full memory
and CPU rules are in CLAUDE.md's "JAX port" section; the short version is that the
production shape (26.8 GiB of H and S) is never allocated, only compiled.

---

## 4. Decisions waiting on you

**Workstream B**

- ~~Start patch 0013.~~ Done and committed — see §3. It was taken as the obvious next
  step rather than put to you, since it was the single item all three open bullets
  shared. The maintenance commitment is real and is now recorded in `patches/README.md`.
- ~~Merge `jax-port` into `master`.~~ Done, fast-forward, at your instruction. **Not
  pushed** — that is still open.
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
of them from the export and never regenerates any. And the projector tolerance must be
computed per run from a real κ, since it is a cutoff property and §8(b)'s cheap estimate
is useless; that bites at the Cholesky-reduced `eigh`, which is the next forward step.

**Workstream A** (unchanged from the previous session)

- The 17 review findings in `docs/review_findings.md`, three of which are one bug.
- Fix `inputfile.py:15` at the source and collapse the two shims, or leave the shims.
- Run the three never-executed integration suites (`spectra`, `optics`,
  `magnetism_manybody`), which will likely surface assertion adjustments.
