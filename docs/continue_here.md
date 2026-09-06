# Continue here

Working state as of 2026-09-06, so this can be picked up cold. Everything below
is on branch `elk-full-coverage`; `master` is untouched at `51bab45`.

```
b15c0b4  Add the Elk-to-JAX port design study            docs/jax_port.md
f04eabd  Record the code-review findings as a work list  docs/review_findings.md
27afcca  Extend the Python interface to the whole of Elk 64 files, +19,828
51bab45  (master) Add vertical tunnelling transport, patch 0012
```

Nothing is merged to `master`. To land it: `git checkout master && git merge --ff-only elk-full-coverage`.

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
questions: does reverse-mode implicit differentiation through the SCF fixed
point work at all (DFTK shipped forward-only, and `lax.custom_root` with an
iterative `tangent_solve` measurably raises `NotImplementedError`), and does the
safe-K projector rule survive a degeneracy. Also worth doing early and cheaply:
benchmark `vmap(eigh)` vs `lax.map` on a **GPU** — measured only 1.03x on CPU
here, and the GPU number is unknown.

### An unresolved measurement disagreement — settle this before trusting §8b

§8b claims the occupied-subspace projector "still returns garbage under JAX's
default VJP" whenever the window is gapped. An independent check disagrees for
the **hard integer window with the multiplet fully enclosed**. Reproduction
(JAX 0.7.1, CPU, x64 enabled):

```python
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
n = 6
evals = jnp.array([-2.0, 1.0, 1.0, 3.0, 4.0, 5.0])   # degenerate pair at 1,2
M = jnp.diag(jnp.arange(n).astype(float)).astype(jnp.complex128)

def assemble(seed):                    # the REALISTIC route: U diag(e) U^H
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
    U, _ = jnp.linalg.qr(jnp.array(A))
    return (U * evals) @ U.conj().T

def hard(H):                           # integer occupations, multiplet enclosed
    w, v = jnp.linalg.eigh(H)
    P = v[:, :3] @ v[:, :3].conj().T
    return jnp.real(jnp.trace(P @ M))

def smeared(H, mu=2.0, w_s=0.3):       # Fermi-Dirac, the metallic DFT case
    w, v = jnp.linalg.eigh(H)
    f = jax.nn.sigmoid(-(w - mu) / w_s)
    return jnp.real(jnp.trace(((v * f) @ v.conj().T) @ M))

def fd(fn, H, h=1e-6):
    D = jnp.zeros((n, n), dtype=jnp.complex128).at[0, 0].set(1.0)
    return float((fn(H + h*D) - fn(H - h*D)) / (2*h))

for name, fn in [("hard", hard), ("smeared", smeared)]:
    for s in range(3):
        H = assemble(s)
        ad, ref = float(jnp.real(jax.grad(fn)(H)[0, 0])), fd(fn, H)
        print(name, s, ad, ref, abs(ad-ref)/abs(ref))
```

Measured: hard window agrees with central FD to `1.5e-9` and `7.4e-10` on two of
three assemblies (the third shows `1.3e-2`, which is FD noise near a degeneracy,
not AD error). Smeared occupations show `2.1e-2` to `5.0e-2`, consistent with
§8b's own `2.08e-2`.

Reading: the divergent terms for a fully-enclosed multiplet are exact negatives
and cancel bitwise, so the hard-window case is safe; with smearing, `f_i - f_j`
is a nonzero rounding-level number and the `1/(λ_i-λ_j)` amplifies it. If that
holds, the insulator mitigation is the one this project already uses for Berry
curvature (CLAUDE.md §13): **window the whole degenerate group together**.

**A 6x6 toy cannot settle it, because finite differences are themselves
unreliable near a degeneracy** — visible in the data above. Settling it needs a
reference that is not FD: an analytically differentiable model (a 2x2 or 4x4
k·p Hamiltonian with a closed-form projector derivative), or complex-step
differentiation, at realistic matrix size with a Cholesky-reduced overlap.

The headline hazard, by contrast, **is** confirmed independently: a diagonal test
matrix gives `NaN`, while the same spectrum assembled from `U diag(e) U^H` splits
by ~1e-15 and returns a **finite** gradient of order 1e14 whose sign flips
between assemblies (+7.9e12, -1.1e12, -1.6e13 measured). Unit tests use the
first shape; real Hamiltonians are the second.

---

## 4. Decisions waiting on you

- Merge `elk-full-coverage` into `master`, or keep reviewing on the branch.
- Fix `inputfile.py` at the source and collapse the two shims, or leave the shims.
- Run the three never-executed integration suites, which will likely surface
  assertion adjustments rather than passing clean.
- Whether the JAX port is worth Phase 0 at all, given that its own verdict says
  the GPU motivation is largely answered by SIRIUS.
