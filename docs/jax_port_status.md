# JAX port (Workstream B): running status, phase tables, and the discipline it runs under

This was the "JAX port" section of `CLAUDE.md` until it outgrew a file that is loaded
into every session. `CLAUDE.md` now carries a short summary plus the hard rules that
must not be violated even by someone who never opens this file; everything else is here.

**Read the "Memory and CPU discipline" section below before running any JAX in this
repository.** It is not background: the study's own Phase 0e asks for production shapes
that this machine cannot hold, and an unguarded run drives a 39 GB workstation into swap.

Companion documents: `docs/jax_port.md` is the design study, `docs/continue_here.md` §3
the cold-start summary, and `docs/jax_port_phase{0,1,2,3}.md` the running logs with every
measured tolerance.

**Where the port is.** Phase 0 closed, Phase 1 closed except for three named items,
Phase 2 closed as a set of forward checks (§§2a-2k, one open question in §2k), and
**Phase 3 has its forward criterion, now from a cold start**: the Kohn-Sham loop closes,
Elk's converged potential is a fixed point of it — 1.8e-15 relative in the muffin tin,
1.0e-9 in the interstitial, where the start mixes Elk's mixed `vsmt` with its unmixed
`vsir` — and §3c runs bulk Si **from the `elk.in` alone**: patch 0024's task 9006 stops
Elk at the top of its own first iteration, and the JAX loop converges from `rhoinit`'s
atomic superposition in 40 iterations to within **3.6e-4 Ha** of Elk's total energy and
**4.3e-5 Ha** of its Fermi level. All of that gap is the frozen core: swapping Elk's
converged `rhocr`/`engykncr` in, and changing nothing else, gives 3.8e-8 Ha and
4.5e-9 Ha (`docs/jax_port_phase3.md`). The loop is **forward only** — the step is
traceable and its `jvp` is checked, but nothing has been differentiated through the
converged fixed point, and Phase 3's three gradient signatures have not been started.

---


`docs/jax_port.md` (1,623 lines) is the design study, `docs/continue_here.md` §3 the cold-start
summary, `docs/jax_port_phase0.md` the running log of what Phase 0 measured,
`docs/jax_port_phase1.md` the same for Phase 1 (through §1m), and
`docs/jax_port_phase2.md` for Phase 2 (§§2a-2k) and `docs/jax_port_phase3.md` for
Phase 3 (§§3a-3b). §2a is the LDA
exchange-correlation functional: `src/elkjax/xc.py` transcribes `xc_pwca.f90`, with
`jax.grad` reproducing Elk's hand-coded $v_{xc}$ at machine precision against the study's
stated $10^{-10}$, exchange exact against Dirac and its spin scaling, correlation
anchored on the Gell-Mann–Brueckner high-density limit, Elk's own $v_{xc}$ on a real grid
at 3e-5 median — limited by a nonlinear functional not commuting with either of Elk's
representations of a real-space function, not by the transcription — and the study's
named `NaN` hazard turned into an assertion: Elk's `rho < 1e-20` guard written as one
`jnp.where` gives the correct value and a `NaN` gradient at exactly zero density.
Patch **0016** adds a `GROUNDSTATE` query — the density and potentials on Elk's own
grids, $k$-independent — which makes that comparison **exact**: Elk evaluates the
functional pointwise on the FFT grid, so `vxcir` is literally the transcription applied
to `rhoir`, to 4.4e-16, once `potks`'s own `trimrfg` low-pass at $|G|>2k_{\max}$ is
reproduced with it (`elkjax.grid.trim`; without it the same comparison stops at 2.5e-5
and looks like a mediocre transcription). §2b is **PBE**, and it is the port's first
real demonstration of its own premise: only the ENERGY densities are transcribed —
exact against Elk's `exir`/`ecir` at 4e-16, which unlike `vxcir` are not trimmed — and
`jax.grad` of the discretised energy supplies the functional derivative
$-\nabla\cdot(\partial(\rho\varepsilon)/\partial\nabla\rho)$ that Elk gets from
Perdew's hand-derived expression needing $\nabla^2\rho$ and
$(\nabla\rho)\cdot(\nabla|\nabla\rho|)$ as extra inputs. **Nothing here computes a
Laplacian.** It agrees with Elk to 2.4e-5 median — not machine precision, and the reason
matters for Phase 3: **Elk discretises the exact continuum functional derivative; AD
returns the exact derivative of the discretised energy**, and those differ because
$|\nabla\rho|$ is not band-limited even when $\nabla\rho$ is. The prediction is
asserted rather than described — the residual tracks the reduced gradient $s$ (5.8e-6 in
its lowest quarter, 1.25e-5 in its highest) — and for scale the gradient terms are 11% of
$v_{xc}$, so the disagreement is ~1% of what AD reproduces from nothing. §2c is the cell
integral and inner product (`rfint`/`rfinp`): $\int\rho$ gives the electron count to
**1.1e-14** against the study's 1e-8 criterion (note `rhomt` INCLUDES the core density —
assuming valence misses by 20 electrons, not by a tolerance), and $E_x$/$E_c$ match Elk's
INFO.OUT to its print width). §2d transcribes the muffin-tin angular transform
(`rbsht`/`rfsht`) and found a real property of Elk with it: `potxc.f90:55-58` calls
`symrfmt` on `vxcmt` and `bxcmt` and **not** on `exmt`/`ecmt`, so inside a muffin tin
$v_{xc}=\hat S\,v_{xc}[\rho]$ while $\varepsilon_{xc}=\varepsilon_{xc}[\rho]$ — the
projection is not the identity even on an already-symmetric $\rho$, because $v_{xc}[\rho]$
is not band-limited when $\rho$ is and the SHT round trip leaks weight into the forbidden
harmonics (Elk holds 1e-20 at Si's $l=1,2,5$; the pointwise potential holds 1e-3). On a
`symtype=0` ground state the same code reproduces `vxcmt` to 1.4e-14. Two consequences:
reproducing Elk's SCF on a symmetric cell needs `symlatc`/`lsplsymc`/`ieqatom` exported and
`rotrfmt` transcribed (one more patch, not a research problem), or a `symtype=0` run; and
**Elk's own $v_{xc}$ is not the functional derivative of its own $E_{xc}$ there**, so a
force or total-energy check better than $\sim10^{-4}$ relative would be evidence of a
mistake rather than of success. §2e is the **Weinert Poisson solve** (patch **0017**,
`src/elkjax/poisson.py`), which closes the last ingredient no export supplies as a
function of the density: `vclir` to 1.6e-15 relative and `vclmt` to 4e-20 ($l=0$, of order
$10^7$ since it carries the nucleus) and 7e-14 ($l>0$), on bulk Si and monolayer h-BN.
Patch 0017 exports only `wprmt`, `vcln`, `npsd`/`lnpsd` and `atposc`; $r^l$, $R^l$ and
$4\pi/G^2$ come from the mesh and `gc`, and `ylmg`/`sfacg`/`jlgrmt` are `genylmv`/
`gensfacgp`/`sbessel`, already pinned element-wise by patch 0013 — exporting `ylmg` alone
would be 38 MB of text. Two checks owe Elk's `vclmt` nothing: the monopole identity
$\sqrt{4\pi}q_{00}=N_{\rm MT}-Z$ recovers $Z=14.000000,5.000000,7.000000$ with $N_{\rm MT}$
from `elkjax.integrate`, and two mutation tests remove one thing Elk does each (the nuclear
term *before* the multipoles are read; the outer region's own spline weights for
$l>l_{\max}^{\rm i}$) and assert the answer moves — both mutants smooth, of the right order
and wrong. Nothing there is differentiated, deliberately: Poisson is linear in $\rho$, so
its linearisation is itself. §2f assembles the **total energy** (`src/elkjax/energy.py`),
which is the first thing in Phase 2 to use more than one of its own pieces at once — none
of the checks above says the functional, the quadrature and the Poisson solve are
consistent WITH EACH OTHER. Every density-functional term of `energy.f90` matches Elk's
own exported scalars to $<10^{-13}$ relative on two structures, asserted term by term
rather than through the total (`engykn` is $+579$ against `engyen`'s $-1219$, so an error
of $10^{-3}$ in either would leave `engytot` looking fine at $10^{-6}$). `evalsum`,
`engyts` and `engynn` are **imported** — they need the second-variational step, a zone sum
and the lattice — so this is the density-functional half, not a transcription of
`energy.f90`. Patch 0017 exports Elk's thirteen converged scalars precisely so the
comparison is term-by-term at full precision rather than at `INFO.OUT`'s print width.
**And it found §2d's prediction to be wrong**: §2d predicted $E_{v_{xc}}$ would inherit the
symmetrisation gap at $10^{-4}$, and it agrees at $2.7\times10^{-16}$, because $\hat S$ is
a group average — an orthogonal projection — and $\rho$ is already in its range, so
$\langle\rho,\hat Sv\rangle=\langle\rho,v\rangle$ identically and the leak lives entirely
in harmonics $\rho$ does not have (measured: 5.3e-3 pointwise, 1e-16 relative against
$\rho$). What survives of §2d is that an SCF iteration compares potentials *pointwise* and
still needs $\hat S$. **A prediction derived from a verified finding is not itself
verified.** §2g closes §2d's remaining consequence with patch **0018**, and the design
choice is the content: `symrfmt`'s operator is **exported rather than transcribed**,
because `rotrflm`'s Euler-angle and Wigner-$D$ construction has no consumer inside Elk
but `symrfmt` itself — so a re-derivation would have no independent check except
agreement with what it replaces, and `ieqatom`/`tfeqat`/the inverse lattice rotation
would have to come with it. `elkpy_gsexport` calls `symrfmt` on basis vectors, giving one
$l_{\max}^{\rm o}$-square matrix per ordered atom pair (a rotation is diagonal in the
radial index and does not mix $l$, so the inner region uses its top-left block), and
applying it takes the pointwise `vxcmt` gap from 5.3e-3 to **6.4e-14**. One measurement
worth keeping: the operator is idempotent to 1e-16 on a CUBIC lattice and only 1.2e-11 on
a hexagonal one, growing with $l$ — Elk's own `roteuler`, whose inverse trigonometry is
exact when the Cartesian `symlatc` entries are $0$ and $\pm1$ and is not otherwise. That
bounds how idempotent `symrfmt` can be, not its accuracy in use. §2h turns the chain
into a circle with patch **0019**: every other Phase 2 section goes from a density to an
energy, and `elkjax.density` goes back, reproducing Elk's valence density
from `evecfv` in BOTH regions to **9e-16** (9.0e-16 and 7.7e-16 in the muffin tins,
7.5e-16 in the interstitial) on an unreduced mesh — exact, since in the interstitial
an LAPW state is a plain plane-wave sum and the only truncation is the basis's own. Two of
its three results are scope statements and both are asserted rather than written down: on a
symmetry-REDUCED mesh it is **16% off**, because `rhomagv` calls `symrf` afterwards and this
does not (§2g's story again, in the interstitial, where the operator is `symrfir`); and the
residual on the unreduced mesh is `rhonorm`'s UNIFORM shift, identified by measurement —
switching `trhonorm` off takes it from 2.82e-05 to 2.8e-18. The muffin-tin half needed only ONE routine
(`wfmtsv`) rather than five, because patch 0019 loops Elk's own `rhomagk` into a LOCAL
array and exports THAT — the density before `rhomagsh`, `symrf`, `rfmtctof` and
`rhocore`, none of which is therefore transcribed (patch 0018's design again). Two traps,
one hit: `evecfv` carries $n_{\rm mat}=n_{gk}+n_{\rm lotot}$ coefficients and exporting
only $n_{gk}$ left the interstitial EXACT (local orbitals vanish there) while the muffin
tin came out smooth, positive, correctly scaled and 100% wrong; and `wfmtsv`'s outer
region restarts its radial stride one step PAST the inner boundary rather than continuing
it. Patch 0015's finding also recurred with a sharper consequence — the reference was
built from the previous iteration's radial functions, so the query's answer depended on
whether `LAPW` had been asked for first (1.2e-10 against 9e-16); fixed with `genapwlofr`
in the Fortran rather than documented around, since **a query whose answer depends on
which query ran before it is a trap**. Patch **0020** then closes the chain: `rhomagsh` (back to
harmonics) at 8.7e-16 and `rfmtctof` (coarse radial mesh to fine) at 1.0e-15, each
against its own exported intermediate rather than through their composition. `rfmtctof`
is exported as a MATRIX for the same reason `symrfmt` is — `rfinterp`'s spline weights
come from `wspline` and depend only on the mesh — with TWO per species, since it
interpolates the whole radial range below $l_{\max}^{\rm i}$ and the outer region alone
above it; using the wrong one reads the inner region's zeros as data, which is smooth,
finite and wrong. **With `symtype=0` those three stages are the whole of `rhomagv`**, so
the chain from first-variational eigenvectors to the valence density on the fine mesh is
closed and exact. Patch **0022** then lifts the `symtype=0`
restriction: `symrfir` acts in $G$-space as a permutation plus a phase, so the whole
operator is two small arrays, and on Elk's DEFAULT mesh (3 k-points, 48 operations) the
density against the converged arrays goes 0.165 → **1.1e-15** (interstitial) and 8.0e-6 →
**1.9e-13** (muffin tin). One detail nothing structural catches: `rhomag` calls `symrf`
BEFORE `rfmtctof`, so §2g's operator needs the COARSE boundary `nrcmti` — passing the
fine `nrmti` treats every coarse point as interior, and the result is still a rotation,
still a smooth positive density, and wrong at 8e-6. Still not done: `rhocore` and the
core states (an input at fixed potential, so an export would do), `rhonorm`'s one
constant is done, and the magnetic branches. §§2i-2j then put the pieces together: the Kohn-Sham
potential COMPOSES pointwise ($v_{\rm cl}+\hat Sv_{xc}$ from three separate modules,
<1e-14 in the muffin tin and <1e-13 against Elk's own `vsir`, which `potks` forms after
trimming `vxcir` and not `vclir` — a mutation test pins that trimming the Coulomb term
too is smooth, correctly-integrating and wrong by only $10^{-12}$); and
`density_from_potential` closes the loop, going potential → $H,O$ at every $k$ →
eigensolve → density with **nothing in the path reading an eigenvector**, at 5e-11. The
missing link was building the interstitial blocks from `vsig`/`cfunig` in $G$-space
instead of recovering $\tilde v_s$ as a matrix in one $k$-point's own basis; against
Elk's own matrices at $\Gamma$, $H$ agrees to 2.7e-15 and $O$ to 5.6e-16. **The 5e-11 is
measured, not excused**: it is Elk's own two exports of `evecfv` disagreeing by 8.5e-9 —
`elkpy_lapwexport` diagonalises fresh after `genapwlofr` while `elkpy_denskexport` reads
the stored ones — and on the gauge-invariant occupied projector this solve matches the
fresh `evecfv` at 1.6e-14 while stored-vs-fresh is 3.7e-11. Patch 0015's finding for the
third time. What is NOT yet done is iterating: that needs `rhocore`, `rhonorm`, a
zone-summed Fermi level and `symrfir`, none a research problem, and the open question is
whether the iteration is stable. **§2k finally differentiates Phase 2** — every section
before it checks a *value* — and found a defect doing so:
`elkjax.integrate.cell_inner_product` called `np.asarray` on its muffin-tin argument and
could not be traced at all, silently, since §2c. Fixed. Then
$\delta E_{xc}/\delta\rho=v_{xc}[\rho]$ holds to **5.7e-17** with AD running through both
`xc_pwca` and the quadrature, and the 1.25e-5 gap to Elk's `vxcir` is entirely `trimrfg`
— asserted as an EQUALITY with `grid.trim`, not an order-of-magnitude coincidence. **The
electrostatic half does not close**: 0.10 Ha absolute, which is 33% of $v_{\rm cl}$'s own
range and 4% of the $v_H$ (2.17) and $v_{\rm nuc}$ (2.48) that nearly cancel to make it
(0.31) — quoting either number alone misleads. Pinned two-sidedly so a later fix fails
the test instead of passing it quietly; the candidate to test first is whether the
discretised Coulomb kernel's reciprocity is only as accurate as the pseudocharge
construction, making it a property of the Weinert method rather than a bug. Verdict, in one line: **a research project justified by
differentiability, not by the GPU** — SIRIUS already does FP-LAPW on CUDA/ROCm with Elk as its
reference, and Elk's hot spots are already near-peak BLAS-3. Nothing about the port is a plan of
record; **Phase 0 (§6 of the study) is designed to kill it, not to start it**, and that is what
is being worked on.

Code lives in `src/elkjax/` — a **sibling package** to `elkpy`, deliberately not `elkpy.jax`.
Two reasons, both load-bearing: Phase 0 is explicitly "no Elk code", and elkpy's fast unit tests
must not acquire a `jax` dependency at all (measured: 0.2 s of import time, plus a 40-thread XLA
pool on the first array operation — see below). Install with
`python3 -m pip install -e .[jax]`. Tests are `tests/test_jax_*.py` and self-skip when `jax` is
unimportable, the same pattern `tests/test_structure.py` uses for ASE. They add ~22 s to the
default suite (each Phase 0a test converges a real fixed point), so
`-k "not calculation_ and not jax"` still gets elkpy's own 440 in ~1 s; the heavier sweeps are
behind `ELKPY_RUN_SLOW_TESTS=1` and take ~3 min. The one test needing BOTH jax and the Elk
binary is `tests/test_calculation_lapw_assembly.py` — named `calculation_`, not `jax_`, so
the fast-suite filter above already excludes it; it converges three ground states, ~30 s.

### Memory and CPU discipline — read before running any JAX in this repository

This box: 12 cores, 39 GB RAM (~28 GB available, ~4 GB of swap already in use), **no CUDA
jaxlib** (`jax.devices()` is `[CpuDevice(id=0)]`). The rules below exist because the study's own
Phase 0e asks for production shapes that this machine cannot hold.

- **Never allocate production shapes here.** At the study's own production figures
  ($n_{\rm mat}\approx3000$, $n_{\bf k}\approx100$, complex128) a single k-point's $H$ or $S$ is
  $3000^2\times16$ B $=144$ MB, so $H+S$ over the k-set is **26.8 GiB before the
  eigenvectors** (another 13.4 GiB) and before any eigensolver workspace, against ~28 GiB
  available — it does not fit, and swapping a 39 GB box is how a workstation is lost for an hour. Execute at $n\le1500$, $n_{\bf k}\le4$ and measure the scaling exponent instead.
- **For Phase 0e, do not execute at all — lower and compile.**
  `jax.jit(step).lower(*jax.ShapeDtypeStruct(...)).compile()` gives both numbers the item asks
  for without allocating a byte of the shapes: wall time for compile, and
  `.memory_analysis()` (verified present in JAX 0.7.1) for
  `temp_size_in_bytes`/`argument_size_in_bytes`/`output_size_in_bytes`. Extrapolation from an
  executed small case is a fallback, not the method.
- **Cap every JAX script and test** with `elkjax.memory.limit_address_space()` (a
  `resource.setrlimit(RLIMIT_AS, ...)` wrapper, default 16 GB) so a runaway allocation raises
  `MemoryError`/`XlaRuntimeError` immediately instead of driving the machine into swap. Verified
  to leave a CPU `eigh` at $n=800$ untouched while turning a 25 TB allocation into a prompt
  error. `tests/test_jax_projector.py` calls it at module level; do the same in anything new.
- **`lax.map`/`lax.scan` over the k-axis is the memory default; `vmap(eigh)` is opt-in.** `vmap`
  materialises every k-point's matrix simultaneously — exactly the 26.8 GiB above. Phase 0d is the
  design fork that would justify `vmap`, and it **requires a GPU this machine does not have**;
  it is deferred, not answered. A CPU ratio (1.03x, measured in the study) does not settle it.
- **`.claude/settings.json`'s `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` pins do NOT govern XLA.**
  Measured here: a single 1200x1200 `jnp` matmul under `OMP_NUM_THREADS=1` spawns **40 threads**;
  `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` changes nothing (still 40). What works is
  affinity: `taskset -c 0-3 python3 ...` gives 16 threads confined to 4 cores. **Wrap every JAX
  invocation in `taskset` while the four-core budget stands** — `launcher.py`'s flock semaphore
  only covers `elk` subprocesses and sees none of this.
- **`jax_enable_x64` must be set before the first array exists**, and doubles every figure above.
  `import elkjax` does it (`src/elkjax/__init__.py` calls `jax.config.update`), so import that
  first and never set it mid-module; `JAX_ENABLE_X64=1` in the environment works too. All of
  this work is float64/complex128: an all-electron spectrum spans ~2500 Ha, so float32 is not an
  option (study §1).

### Phase 0 — prove or kill (study §6)

| Item | What it settles | Status |
|---|---|---|
| 0b | safe-$K$ projector rule: does the $(f_i-f_j)/(\lambda_i-\lambda_j)$ `custom_jvp` fix the reassembly jitter, and what happens in the padding block | **done, and now closed against REAL matrices** by Phase 1f — the rule is exercised on Elk's own Cholesky-reduced $\tilde H$ at a symmetry multiplet, with $\kappa(O)\approx5\times10^3$ measured per run rather than prescribed |
| 0a | reverse-mode implicit diff (`custom_vjp` + GMRES) through an SCF fixed point whose matvec passes through `eigh` at a multiplet | **done — it works**, ≤1e-14 against a dense IFT reference on four spectra including an exactly degenerate one; the naive rule fails on the same machinery |
| 0a′ | the same at second order — decides the full port over §9.2's hybrid | **done — it works**, via `projector.sign_projector` (matrix sign by Newton-Schulz: no eigensolve, so differentiable to any order); `grad(grad)` through the fixed point agrees with central FD to 1.2e-9 where the `eigh`-based rule gives `NaN`. Use `grad(grad)`, never `jax.hessian` — a `custom_vjp` cannot be forward-differentiated |
| 0c | `jax.jvp(match)` against `dmatch.f90`'s analytic $d(\texttt{apwalm})/dr$ | **done — exact to 7e-16** in both modes (`src/elkjax/lapw.py` transcribes `match`, `gengkvec`, `gensfacgp`, `genylmv`, `sbessel`), **and the forward half is now closed against Elk's own `apwalm`** by patch 0013 (§33): 1.9e-15 in `match`'s `omax==1` division branch and 8.0e-13 in its general linear-solve branch, the latter reachable only through a generated `apword=2` species file since every species file Elk ships sets `apword=1` |
| 0d | `vmap(eigh)` vs `lax.map` at $n=1000$ on a real GPU | **timing deferred (no GPU); memory settled by 0e** — at production shapes a `lax.scan` accumulator holds 0.411 GiB of temporaries and `vmap` holds 40.2 GiB, so `vmap` over the k-axis does not fit on a 40 GB device whatever the timing says |
| 0e | `jit` compile time and peak memory for one traced SCF step at production shapes | **done — `docs/jax_port_phase0.md`.** Compile time is FLAT in the shapes (0.46 s at both $(200,4)$ and $(3000,100)$) and **superlinear (exponent ≈1.85) in HLO op count** — isolated with the corrector, which is linear in its pass count, since Gram-Schmidt's own op count is quadratic in `n_lo` — while a `lax.scan` over 4x more radial points costs nothing. Design rule: `scan` repeated structure, unroll only what must be. Differentiating the step adds only ~1.2x |

### Phase 1 — one k-point, one species, no SCF (study §6)

**`hmlfv`/`olpfv` are done, forward.** `src/elkjax/hamiltonian.py` transcribes the
muffin-tin half of the first-variational LAPW eigenproblem —
$O^{\rm MT}=A^\dagger A$ and $H^{\rm MT}=A^\dagger ZA$ with
$Z=\sum_{\ell_2m_2}\langle Y_{\ell_1m_1}|R_{\ell_2m_2}|Y_{\ell_3m_3}\rangle\,
h_{\ell_2m_2}$ — and each of its **six blocks is compared separately** against Elk's
own (`tests/test_calculation_lapw_assembly.py`), so a failure names one upstream
routine rather than "$H$ is wrong". Machine precision on bulk Si at `apword` 1 and 2 and
on monolayer h-BN; the assembled pair reproduces Elk's `evalfv` to 9e-15 Ha. Patch
**0014** supplies what 0013 did not: the radial integrals `oalo`/`ololo`/`haa`/`hloa`/
`hlolo`, the local-orbital bookkeeping and the complex Gaunt array `gntyry`. The
**interstitial blocks are taken from the export, not built** — $H^{\rm I}$ needs the
interstitial Kohn-Sham potential $V_s$, which is Phase 2 — so `cfunig`/`vsig` are
deliberately not exported.

**The trap, and only one of the three fixtures can see it.** `hmlrad.f90` builds
`hlolo`'s $\ell_2=0$ element as the unsymmetrised
$\int u^{\rm lo}_i(\hat H u^{\rm lo}_j)r^2dr$ — no averaging over the two orderings and
no kinetic surface term, unlike `haa`, whose counterpart is explicitly averaged and whose
transpose is explicitly assigned. `hmllolo` therefore evaluates each local-orbital pair
in ONE order and Hermitises the rest, and a consumer must do the same. Using both halves
gives a Hermitian, positive-definite, plausible, **wrong** matrix: measured on h-BN's
nitrogen (two $\ell=0$ local orbitals), the orderings differ by 1.3e-2 Ha, reaching $H$
as 3.7e-3 Ha and `evalfv` as 4.3e-7 Ha. Silicon cannot see it — one s and one p, no
repeated $\ell$ — so the h-BN premise is **asserted** by its own test rather than
assumed. `apword=2` is equally load-bearing: at `apword=1` the APW-order axes of `haa`
and `hloa` are length 1, so an $i_o\leftrightarrow\ell$ swap is a no-op rather than a
detected error. Mutation-tested one error at a time.

**The residual is Elk's own guard, measured not assumed.** `hmlaa`/`hmlalo` skip any
Gaunt-contracted $z_1$ below 1e-12 (the `zaxpy` guard); reproducing it takes h-BN's
`hmlalo` from 3.3e-14 to 3.0e-17. The dense version here is the more accurate of the
two, so the guard is documented rather than copied — and any element-wise comparison
against Elk has a ~1e-12 floor because of it.

**The assembly is now a function of $k$, and it has been differentiated.** Two
observations remove the need for any new Fortran: $O$'s interstitial block IS the
characteristic function and is $k$-independent, and $H$'s is that plus an explicit
kinetic term — so $V_s$, the one Phase 2 ingredient involved, is recovered as a MATRIX
by subtracting the two exported blocks elementwise, with no Fourier index mapping.
`apwalm` comes from `elkjax.lapw.match`, closed with a Cholesky-reduced `eigvalsh`.
Back at the exported $k$ it reproduces Elk's $H$, $O$ and `evalfv` (9.8e-15 / 2.7e-15 /
5.0e-15 on Si), and `jax.grad` agrees with central FD of the same function to 1e-9.

**And it found that $d\varepsilon/dk \neq \langle p\rangle$ in a finite LAPW basis.**
Against `genpmatk` (§22's `MOMENTUM`, an independent Fortran path) the two agree in sign
and to 0.2-1.4%, NOT to machine precision — and the gap is **flat in `rgkmax`** (2.336e-3,
2.339e-3, 2.340e-3 at 7, 8, 9 for band 0) while the eigenvalue converges, so it is not
the plane-wave cutoff. It is the **muffin-tin linearisation**: `apword` 1→2 (augmenting
with $\dot u$ as well as $u$) cuts it up to 4x, most for the high band where it was
largest. Hellmann-Feynman needs a $k$-INDEPENDENT basis and LAPW's is not one, so AD
returns $v^\dagger(\partial_kH-\varepsilon\,\partial_kO)v$ while `genpmatk` returns
$\langle\psi|-i\nabla|\psi\rangle$. **Two consequences**: `genpmatk` is not a
machine-precision reference for a band velocity — which is the mechanism behind
`tests/test_calculation_momentum.py`'s own Hellmann-Feynman check needing `rel=2e-2` —
and any remaining gradient criterion must finite-difference the *same* code path. The
test asserts the direction of the effect rather than a tolerance, and requires the two
NOT to agree exactly so its premise cannot lapse silently. Separately: `apword=2`'s
matrices inherit `match`'s ill-conditioned general branch (1.3e-12 vs 9.8e-15) but the
SPECTRUM does not (3.8e-15 either way) — a near-null-space rotation inside the APW order
space, which eigenvalues are blind to.

**The eigensolve is now wired to the safe-$K$ projector rule, and 0b is closed against
real matrices** (§1f, `src/elkjax/phase1_projector.py`,
`tests/test_calculation_lapw_projector.py`). `hamiltonian.py` gained `cholesky_reduce`,
`projector_tolerance` ($\epsilon\,\kappa(O)\,\lVert\tilde H\rVert$, measured per run
from a dense `eigvalsh` — §8b's Cholesky estimate is uninformative), `occupied_window`
(the host-side refusal) and `occupied_projector`. Bulk Si at $\Gamma$ supplies the
disputed configuration by SYMMETRY rather than by construction: the $\Gamma_{25'}$
triplet inside a window whose boundary is open by 0.093 Ha. Measured there, the naive
route is wrong by 3.9e-1 in forward mode and returns `NaN` in reverse, against 2.2e-11
for the safe one; at a generic $k$ on the same ground state both agree to 2e-13, which
is what makes it a measurement of the rule rather than of the fixture. The naive error
tracks $1/\delta\lambda$ across three fixtures spanning ten decades of splitting
(3.8e-13 at 2.6e-2 Ha, 1.1e-10 at 1.9e-5 Ha, total failure at 5.1e-15 Ha). The
projector reproduces Elk's own occupied subspace — as $YY^\dagger$ with
$Y=L^\dagger C$, a projector comparison because `evecfv` is arbitrary inside the
triplet — to 5e-14, six orders inside the study's own $10^{-8}$ criterion.

**Three findings from doing it.** (i) The tolerance is a *resolution* floor, not a
symmetry statement: Elk's matrices split the $\Gamma_{25'}$ triplet **unevenly**,
5.1e-15 Ha for one pair and 3.53e-9 Ha for the other, the second 68x ABOVE the
5.21e-11 Ha tolerance, so cutting the triplet at `nocc=3` is refused while cutting the
SAME triplet at `nocc=2` is accepted and returns a finite derivative of a subspace that
is not physically separable. The refusal is necessary, not sufficient — window the whole
degenerate group, as §13 already does for Berry curvature. Tightening `epspot` 1e-6 →
1e-9 leaves that 3.53e-9 identical to twelve digits, so it is not SCF convergence;
source open. (ii) **`soc_scale` cannot move the first-variational spectrum at all** —
`socfr` enters only `eveqnsv`, with zero occurrences in `hmlfv`/`olpfv`/`hmlaa`/
`hmlalo`/`hmllolo`/`olpaa`/`olpalo`/`olplolo`/`eveqnfv`/`hmlrad`/`olprad`
(grep-verified) — so the study's adversarial sweep is "refuse always", not a threshold
crossing, and is withdrawn as written; its content is delivered by cutting a real
multiplet instead. (iii) **The $k$-tangent is `NaN` at $\Gamma$** while the value there
is exact — see the next paragraph.

**That third finding was a real blocker, and it is fixed (§1g).** At any basis function
with $\mathbf G+\mathbf k$ on the $z$-axis, $Y_{\ell m}(\hat v)$ has no derivative (the
direction is undefined) and $\lvert\mathbf G+\mathbf k\rvert$ is $\sqrt\cdot$ at zero —
**two independent poles, and fixing only the first leaves the second, which is invisible
until it is**. $\mathbf G=0$ is in every basis, so this was every reciprocal-lattice
point, and in a slab cell $\mathbf G=(0,0,\pm2\pi/c)$ is too, so it was the entire
$k_z=0$ plane — all of a 2D material's physics, the $K$ point included — which combined
with (i) left the safe-$K$ rule and the $k$-derivative usable in DISJOINT places, since
multiplets live at high-symmetry points. The product is smooth even though its factors
are not, so `match` now regroups it as a **regular solid harmonic** $r^\ell Y_{\ell m}$
(`elkjax.lapw.solid_harmonics` — `spherical_harmonics`' own recursion with
$\cos\theta\to z$, $\sin\theta e^{i\phi}\to x+iy$, $\beta\to\beta r^2$, hence a
polynomial in the Cartesian components) times $j_\ell^{(i_o)}(x)x^{i_o-\ell}$
(`spherical_bessel_scaled` — even in $x$, hence a function of
$x^2=R^2(\mathbf G+\mathbf k)\cdot(\mathbf G+\mathbf k)$ with its own small-$x$ series),
and forms neither $\hat g$ nor $\lvert g\rvert$; `gkc` is therefore **no longer an
argument of `match`**, since leaving it in the signature would let a call site
reintroduce the second pole. `spherical_harmonics`/`spherical_bessel` are untouched and
remain what item 0c checks. The identity is exact, so forward values are unchanged:
Elk's `apwalm` element-wise at generic $k$ **and now at $\Gamma$** (a real check, since
Elk handles $\mathbf G+\mathbf k=0$ its own way), the `dmatch` identity, and all six
assembly blocks are still green; $dP/dk$ at $\Gamma$ through the multiplet goes from
`NaN` to 1.4e-14. The new route is also *more* accurate near the origin than the old
one, which loses 100% at $\ell=6,i_o=2,x=10^{-6}$ by forming $j_6''\sim10^{-27}$ and
dividing by $x^4$. **The two fixes are independent**: with the poles gone, the *naive*
projector's $k$-derivative at $\Gamma$ is still `NaN`, which its own test asserts so
that "we fixed the pole" cannot be mistaken for "the $k$-derivative is fine".

**The study's negative test is done too (§1h), and it corrected the study's own fixture
suggestion.** The criterion is that AD, central FD and one-sided FD of an INDIVIDUAL
eigenvalue must *disagree* at an exact degeneracy while the multiplet trace agrees. The
study names "h-BN at $\Gamma$", and that cannot work — nor can Si at $\Gamma$: at any
time-reversal-invariant momentum every branch is EVEN in $\mathbf k$, so the sorted
branches never exchange between $+t$ and $-t$ and AD and central FD both correctly return
zero (measured $10^{-17}$–$10^{-11}$ on Si's $\Gamma_{25'}$ triplet). Degeneracy is not
enough; the branches must cross LINEARLY. On graphene at $K$ (2 atoms, `rgkmax=6`, under
two minutes including the ground state) the Dirac pair gives AD $\pm0.184$ (a
basis-dependent number that means nothing on its own — it is the diagonal of
$v^\dagger\delta Hv$ in whichever basis the eigensolver picked, so only its *disagreement*
with the other two is the result), central FD $\pm0.0009$ (the branch average, since the
branches exchange) and one-sided $\mp0.376$ (the extreme branch), while their trace agrees
across all three to
$5\times10^{-7}$ of that scale — with the $\sigma$ doublet at the same $K$, equally
degenerate but not linearly split, as the in-fixture control where AD and central FD do
agree. So `first_variational_eigenvalues` is safe for a trace and unsafe for an
individual band inside a multiplet, asserted rather than documented.

**Smeared occupations are done too (§1i), and they corrected their own premise.** A hard
integer window cannot exercise the kernel's near-degenerate branch at all — both branches
of a same-side pair are identically zero — so §1f's `tol` plateau said nothing about it;
Fermi-Dirac occupations make the branch value $f'$, and the threshold becomes
load-bearing. What was *expected* was that a metal is where it bites. What was measured is
that **whether the branch fires is set by the assembly's roundoff, not by the physics**:
graphene at $K$ — the metal, with the Fermi level exactly on the Dirac degeneracy
(confirmed by the single-$k$ $\mu$ reproducing Elk's own zone-integrated `EFERMI.OUT` to
$3.4\times10^{-9}$ Ha) — is split $3.4\times10^{-7}$ Ha and does **not** fire, while gapped
Si at $\Gamma$ is split $1.1\times10^{-15}$ Ha inside $\Gamma_{25'}$ and does. Both were
needed, because each shows only one of two distinct failures: on graphene the direct
quotient is exact to $10^{-13}$ and it is JAX's own eigenvector rule that fails,
**proportionally in the smearing width** (3.9e-9, 3.9e-8, 2.8e-7 across $w=10^{-3}$ to
$10^{-1}$, since its absolute error is set by the splitting while the quantity it is
measured against is $f'=1/4w$ — broader smearing is not gentler); on Si that rule returns
`NaN` outright while the quotient survives but is wrong by up to 3.7e-3, and at Elk's
default `swidth` is **exactly zero** against a true kernel of $-3.2\times10^{-2}$, the two
occupations being bitwise equal. The oracle for all of this is analytic, not either
candidate: `reference.fermi_divided_difference_kernel` writes the logistic difference
quotient as $-\frac1{4w}\,\mathrm{sinhc}(z_{ij})/(\cosh u_i\cosh u_j)$, which contains no
subtraction and so arbitrates between the branches. The **self-consistent Fermi level**
came with it: `mu` is now a differentiable primal of `smeared_projector`, so
`fixed_number_projector` is just the composition with `projector.fermi_level` (a
`custom_jvp` over a never-differentiated bisection), and study §8(b)'s
$d\mu=\sum_jf'_jA_{jj}/\sum_jf'_j$ is tested for the first time — gauge-invariant at a
multiplet (inside a degenerate group $f'$ is constant, so the sum is a trace), and
**dominant rather than corrective**: dropping it is wrong by 69x at a half-filled level.
Its denominator vanishing is physics, not numerics — in a gap nothing responds and $\mu$ is
undetermined, which `check_fermi_level_determined` refuses. Two limits left: §8(b)'s
k-point weights cancel at a single $k$ and remain untested (a zone-summed Fermi level is
Phase 2), and the safe rule's floor on Si is $2\times10^{-9}$ rather than $10^{-13}$
because the *other* pair sits $67\times$ ABOVE the tolerance and therefore takes the
cancellation-prone quotient — the tolerance is a cliff, and beating that needs the stable
kernel inside the JVP rather than a better threshold.

**The tolerance was then removed rather than tuned, and second derivatives followed
(§1i second pass, §1j).** The stable form of the kernel is *analytically exact at every
splitting*, so it belongs inside the JVP rather than beside it: `projector.fermi_kernel`
now carries it and `smeared_projector` uses it, which takes Si's floor from
$2\times10^{-9}$ to $7.4\times10^{-12}$/$3.1\times10^{-12}$/$4.8\times10^{-14}$ across the
three widths and makes `tol` **inert** for smeared occupations — asserted as *equality*
across fourteen decades of it, which is a stronger and different statement from study
§8(b)'s "flat over two decades". (`tol` stays load-bearing for `hard_window_projector`,
whose straddling-pair `NaN` is a claim about a derivative that does not exist.) What is
left is **not the kernel**: the residual shrinks as the smearing *widens*, which is
backwards, and chasing that showed LAPACK and XLA disagreeing about the eigenvalues by
$2.1\times10^{-14}$ Ha, amplified by the kernel's $f''/f'\sim1/w$ relative sensitivity —
rerunning the identical closed form on XLA's own decomposition gives 7.9e-14 / 5.5e-15 /
2.6e-15, and the difference between the two references *is* the residual to two digits
(`phase1_smearing.eigensolver_floor`). Knock-on worth remembering: the reference and the
implementation now share a *formula*, so the arbiter where it matters is central FD
(legitimate here, since $P=f(H)$ is smooth) and `direct_quotient_projector`, which keeps
the literal quotient. **Second order (§1j)**: the safe-$K$ rule is first-order by
construction — its JVP body calls `eigh` — and on bulk Si at $\Gamma$ with the
$\Gamma_{25'}$ triplet enclosed, `grad(grad)` through it returns `NaN`, as does the naive
route, while `sign_projector` (matrix sign by Newton-Schulz, no eigensolve) returns a
finite value agreeing with a central difference of the safe rule's own first derivative to
$3\times10^{-11}$–$2\times10^{-10}$. The control is in the same ground state: at a generic
$k$ all three agree to $10^{-10}$, so the failure is the multiplet, not the order.
Differentiated twice in $k$ through the whole assembly too, where refining the FD step
gives 1.44e-4, 1.29e-5, 1.44e-6, 1.29e-7 — textbook $O(h^2)$, i.e. FD converging *onto*
AD. Cost, which is the half a toy could not supply: the Newton-Schulz count is set by the
top of the basis (17.9 Ha) and not by the 0.35 Ha valence manifold, so the ratio is 190
and the predicted count 13 — 10 steps is not converged (error 1.1, not a projector at
all), 20 reaches $3\times10^{-14}$ against both the safe rule and Elk's own occupied
subspace, in 42 ms at $n=177$. Use `grad(grad)`, never `jax.hessian`.

**The radial integrals are no longer inputs (§1k), and the chain closes without any
Phase 2 ingredient.** The plan put this in Phase 2 because it needs the muffin-tin
potential — it does, but a converged potential is an *input* that can be exported and
held fixed exactly as `STATE.OUT` already is. Patch **0015** exports `vsmt` in Elk's own
packing, the radial mesh `rlmt` with its `wr2mt` quadrature weights, the linearisation
energies, and `apwfr`/`apwdfr`/`lofr` in full; `src/elkjax/radial.py` transcribes
`hmlrad`/`olprad` and `src/elkjax/radial_functions.py` transcribes
`rschrodint`/`genapwfr`/`genlofr`, so **vsmt → apwfr/lofr → radial integrals → H, O →
evalfv** is closed and differentiable. Element-wise against Elk on the three fixtures:
`oalo`/`ololo` 2.4e-16, `haa`/`hloa`/`hlolo` 2.0e-16, `apwfr`/`apwdfr` 1.3e-14, `lofr`
7e-15. Elk's predictor-corrector is transcribed rather than replaced — a Runge-Kutta step
would converge to the same continuum solution and disagree at the mesh's own truncation
error, four decades above what is checked; its first three points are unrolled (their
stencil windows overlap the $r\to0$ boundary values, and the current iterate sits INSIDE
the four-point window it is integrated over) and `lax.scan` starts at the fourth.

**The finding, and it is structural: the two channels of the potential are exactly
complementary.** Frozen-basis vs full AD on bulk Si over the occupied window — a purely
SPHERICAL perturbation gives a frozen-basis derivative of **exactly zero**; a purely
NON-SPHERICAL one moves the basis not at all (7.0e-16). `hmlrad`'s $\ell_2=0$ element is
$\langle u|\hat Hu\rangle$ and `genapwfr` has already applied $\hat H$ — the radial
functions ARE that operator's solutions, so the radial equation has **eliminated** the
explicit $\int u\,v_{\rm sph}\,u$ integral, which is the LAPW construction itself; and
`genapwfr`/`genlofr` integrate in the spherical part alone, so the non-spherical potential
cannot move the basis. **So "full minus frozen" is NOT the basis relaxation** — it is the
Hellmann-Feynman term Elk's bookkeeping hides *plus* the relaxation, and conflating them
was a real error corrected here. `phase1_potential.hellmann_feynman` computes
$\sum_n\langle\psi_n|\delta V|\psi_n\rangle$ explicitly (the same integrals with the
$\ell_2=0$ slice filled by the potential integral rather than zeroed), and the honest
decomposition is HF + relaxation. **The relaxation depends on the SHAPE of the
perturbation by a factor of 100**: 29% of the derivative for white noise reaching the
nuclear cusp, where a basis at fixed linearisation energy cannot follow it, and **0.30%**
for a smooth valence-region bump — roughly what an SCF update does. Quoting the first
alone misrepresents the method. The Phase 2 warning that survives is narrower and
sharper: a chain producing a perfectly correct $\delta v_s$ and feeding it to a basis
frozen in Elk's own $\ell_2=0$ sense returns **zero** for the spherical channel while
passing the study's pointwise $v_{xc}$ check. **The relaxation itself has two routes**, and
the first version of this measurement missed one: a perturbed potential reaches $H$ and
$O$ both inside the radial integrals and through the matrix $D$ of radial derivatives at
$R_{\rm MT}$ that `match` inverts, so rebuilding `apwfr` without rebuilding $D$ (and hence
`apwalm`) freezes the basis at the sphere boundary while its interior moves. Nothing in
the gradient checks could see it — AD and FD then differentiate the same truncated
function and agree to 4e-10, and both structural zeros survive — which is Phase 0's own
"a green gradient test does not validate a transcription" recurring verbatim; the check
that exposes it is forward, `elkjax.radial_functions.derivative_matrices` against the
exported `dmat`. It was worth 21% of the full derivative. The frozen branch is pinned by
a closed form, not
by finite differences (at fixed basis the overlap does not respond, so first-order
perturbation theory collapses to $\sum_n c_n^\dagger\delta Hc_n$ with Elk's own
`evecfv`): 1.2e-15. Getting that reference right needs one non-obvious fact — **the map
from the potential to the radial integrals is AFFINE, not linear**, its constant part
being that same $\ell_2=0$ block, and carrying it into $\delta H$ flips the sign
(-2.27e-1 against a true +2.63e-2) rather than merely degrading it. AD vs central FD of
the same function: 3.9e-10 at $h=10^{-4}$, degrading as $h$ shrinks — the $1/h$ roundoff
signature, not a wrong gradient.

**Patch 0015 also had to fix an inconsistency in the export itself.** `gndstate` calls
`genapwlofr` at the top of an SCF iteration and `potks`/`mixerifc` at the bottom, so on
exit the radial functions and integrals belong to the PREVIOUS iteration's potential;
0015 calls `genapwlofr` before exporting. Found by splitting the comparison into
potential-free and potential-carrying integrals (2.6e-16 vs 3e-10) — an aggregate number
would have read as an indexing bug. Knock-on: that regeneration moves Elk's own matrices
by ~3e-10 and retuned one over-fitted constant in the smearing suite by 13x, while the
Dirac splitting moved only in its eighth digit; measured with the call off and on rather
than inferred, and the test now asserts the mechanism rather than a fixed factor.

**The position derivative is half done (§1l)**, and the half that exists is pinned by an
exact identity rather than a finite difference. At frozen potential (rigid muffin tin)
positions enter only through `match`'s structure factor, and **a rigid translation of
every atom cannot move the spectrum** — the matrix transforms by the diagonal unitary
$U=\mathrm{diag}(e^{i(\mathbf G_i+\mathbf k)\cdot\boldsymbol\delta})$, the muffin-tin
blocks getting that right on their own. $\tilde\Theta$ is BUILT too — closed-form
geometry (`hamiltonian.characteristic_function_matrix`, `gencfun`+`genffacgp`), matching
Elk's own $O^{\rm I}$ element-wise to 1e-16 and closing the overlap half of item 1c — so
the only response supplied by hand is the interstitial Kohn-Sham potential's,
$\tilde v(\mathbf G)\to\tilde v(\mathbf G)e^{-i\mathbf G\cdot\boldsymbol\delta}$.
Measured 1.8e-15 Ha (Si) and 3.8e-15 (h-BN) with it, 2.6e-5 and 1.4e-3 without. Building
$\tilde\Theta$ does NOT monotonically shrink the residual (on h-BN it grows, the two
omissions having partly cancelled), so a smaller residual is not evidence of a better
assembly; it does move the single-atom derivative by 7.5%/35%. **The FORWARD form of that null is sharper than the
gradient form**: without the interstitial response the error is $O(\delta^2)$ on Si and
$O(\delta)$ on h-BN, so the wrong assembly satisfies the *gradient* null identically on
silicon. Third distinct instance in this port of "a green gradient test does not validate
a transcription" (after 0c's $4\pi(-i)^\ell$ and §1k's frozen `apwalm`) — every time the
check with teeth was forward. **It is NOT a force**: the muffin-tin and interstitial
Kohn-Sham potentials' own response to the displacement is Phase 2, and is now the only
thing missing. Also open: smeared occupations **at second order**, which `sign_projector` does not cover
(it is hard-window only; that needs a Chebyshev expansion of the Fermi function, and §1i
removed the tolerance from the smeared first derivative, not the `eigh` from its JVP); and
the unrolled Newton-Schulz tape, which is **now fixed** (§1m): `lax.scan` is the default,
worth 230x the HLO instructions and 142x the compile time at 80 steps and second order
(0.21 s against 30.0 s), with a count flat in the tape. Two qualifications — the
differentiation order multiplies the ratio rather than adding to it, so the order this
routine exists for is the one that pays most; and `scan` saves the GRAPH, not the memory
(10%, since its backward pass stores one residual per iteration exactly as the unrolled
tape does). It is also not bitwise (6e-16): calling the same function the same number of
times does not fix the arithmetic, XLA fusing the two shapes into different regions.

**$\kappa(O)$ for a real LAPW overlap is measured, and the cheap estimate is
useless.** Patch 0013 (§33) supplies real $H$ and $O$; `python3 -m elkjax.phase0b_overlap`
reproduces the table in `docs/jax_port_phase0.md` §0b(ii). At the standard `rgkmax=7`,
$\kappa(O)\approx5\times10^3$, so the safe-$K$ tolerance
$\epsilon\,\kappa\,\lVert\tilde H\rVert$ is $\approx1.6\times10^{-11}$ Ha on Si and
$6.5\times10^{-11}$ Ha on h-BN. Three things to carry forward. It is set by the **cutoff,
not the matrix size** — at `rgkmax=8`, Si's $227\times227$ overlap and h-BN's
$2118\times2118$ agree to within the spread across $k$-points, while `rgkmax`
$7\to8\to9$ takes Si from $5\times10^3$ to $2.6\times10^4$ to $2.1\times10^5$ (h-BN
$7\to8$: $5.0\times10^3\to3.5\times10^4$), so the tolerance must be recomputed per run
rather than hard-coded. The study's §8b Cholesky-diagonal estimate is worse
than "a lower bound, 140x low": it is **uninformative**, moving only 8.05→9.25 across a
74-fold range of $\kappa$, so it must not be used to set a threshold. And the norm in
that formula should be $\lVert L^{-1}HL^{-\dagger}\rVert$, not $\lVert H\rVert$ — 3x
larger here. Knock-on: the study's Phase 1 adversarial `soc_scale` sweep 3000→3 stops
three to six orders of magnitude above the gap at which its own refusal criterion is
meant to fire (the spread is the unknown curvature of the gap in the scale), so it has to
be extended below `soc_scale=1`.

**Two mixers, one caveat about Elk's own.** Unrolling the SCF instead of differentiating
it implicitly is not merely inaccurate: measured, unrolled *Anderson* reaches a forward
value good to 1.8e-13 while its gradient is wrong by $10^{17}$–$10^{32}$ relative, across
a five-decade sweep of the mixer's internal ridge, where unrolled *linear* mixing
converges normally. Elk's default `mixtype=3` is a Broyden scheme of the same shape. The
implicit route is indifferent by construction — which also means "implicit agrees between
mixers" proves nothing on its own, since its backward pass only ever sees $(\theta,v^*)$.

**Two traps that a badly chosen test walks straight past**, both measured here rather
than reasoned about. A *direction* that is a single real diagonal entry makes the naive
projector rule look correct in reverse mode (§0b), and a *perturbation that respects the
symmetry protecting a degeneracy* makes it look correct everywhere — error 1.5e-14
symmetric versus 4.7e-1 symmetry-broken on the same Hamiltonian (§0a), because neither
the perturbation nor the observable then has a matrix element between the partners.
Test along general directions, and break the symmetry; and always compare forward-mode
against reverse-mode, which for a scalar-in scalar-out function must agree exactly.

**A green gradient test does not validate a transcription.** Measured on 0c: dropping
`genylmv`'s $4\pi(-i)^l$ prefactor multiplies each $\ell$ block by a fixed complex
number, and the result still passes the exact `dmatch` identity to 7e-16 — a constant
factor commutes with $\partial/\partial\mathbf r_\alpha$. Every AD check here needs a
forward check beside it, and the strongest available without Elk is the quantity's own
defining equation (for `match`, the matching condition $DA=b$ rebuilt independently),
not a comparison of its pieces.

**Use an analytic reference, not finite differences, wherever a degeneracy is in play.** The
study's own §8(b) measures FD failing at a multiplet — central FD of the *sorted* spectrum
returns the branch average, so it cannot detect a wrong individual-eigenvalue gradient at all,
and near a degeneracy it is noise. For the occupied projector the closed form is available and
costs nothing:
$dP=\sum_{i\in W,\,j\notin W}\big(|i\rangle\langle i|\,dH\,|j\rangle\langle j| + \text{h.c.}\big)/(\lambda_i-\lambda_j)$,
gauge-invariant, exact, and valid at any $n$ — that is the reference `docs/continue_here.md` §3
says is missing, and it is what decides whether the custom rule is needed for a hard integer
window at all. **Measured, and the answer is yes**: over 3 assemblies x 21 Hermitian directions
on the disputed spectrum the naive route is wrong by $1.1\times10^{1}$ (forward) and
$3.4\times10^{0}$ (reverse) relative, against $2.9\times10^{-14}$ with the rule, while central
FD agrees with the closed form to $3.7\times10^{-8}$ — so FD was *reliable* here and the earlier
check's one-in-three disagreement was AD error, not FD noise. It saw agreement because it probed
a single real diagonal direction in reverse mode only; that same direction in forward mode was
already wrong at $1.3\times10^{-2}$. **That reverse-mode agreement does not reproduce across
builds and was never supposed to** — on a rebuilt binary the naive rule is wrong in both modes
there (1.6e-2 forward, 3.9e-2 reverse), for the reason the next paragraph gives: which mode looks
right is the eigensolver's arbitrary split of a roundoff-degenerate pair. The test now asserts
only what does not depend on the build. **Always check forward against reverse** — for a
scalar-in, scalar-out function they are the same number, so disagreement is proof on its own and
costs nothing. Also measured: which failure mode appears is the *eigensolver's* choice, since
LAPACK and XLA split the same engineered pair differently and XLA returns it bitwise equal at
$n=1000$ (finite garbage at $n=400$, `NaN` at $n=1000$, same code). The mechanism, for the
record: the two divergent terms are exact negatives and would cancel bitwise *if*
$A=v^\dagger\,\delta H\,v$ were bitwise Hermitian, and JAX's `_eigh_jvp_rule` forms it with no
symmetrisation — so $\|A-A^\dagger\|/\delta\lambda\approx0.2$–$0.5$ survives, which is the
size of the observed failure.

### Phase 3 — the SCF loop with implicit differentiation (study §6)

| item | what it settles | status |
|---|---|---|
| the occupations and the zone-summed Fermi level (`occupy.f90`) | whether study §8(b)'s rule works with the k-point weights that a single k-point makes cancel | **done, patch 0023** — Elk's `efermi` and `occsv` reproduced **bitwise** on bulk Si and fcc Al, on Elk's own `evalsv`, i.e. with the assembly out of the path. $d\mu$ is a `custom_jvp` over a bisection that is never differentiated; forward against reverse against central FD in all three differentiable arguments on a real metal. Replacing Elk's reduced-mesh weights with uniform ones moves $\mu$, asserted, so the weights cannot silently stop mattering |
| **Forward:** a converged ground state reproducing Elk's total energy and Fermi level | whether the composition of §2i and §2j is *stable*, which no Phase 2 check asked | **done** — Elk's converged $v^*$ is a fixed point of the map to $1.8\times10^{-15}$ relative in the muffin tin and $1.0\times10^{-9}$ in the interstitial (the two halves differ by four orders of magnitude in norm, so one bound on the packed vector says nothing about the second); from a start $0.30$ away in potential norm, linear mixing at $\beta=0.4$ converges geometrically (~0.62/iteration) with $\lVert v-v^*\rVert$ tracking $\lVert F(v)-v\rVert$ all the way down, reaching `engytot` within **3.0e-8 Ha** and $\mu$ within **1.4e-9 Ha** — inside the study's own 1e-6 and 1e-8. The iteration count is deliberately NOT compared, as the study itself withdraws that criterion |
| **Forward, from cold:** a ground state from the `elk.in` alone | whether the loop is a *calculation* or only a map with a hand-made starting point | **done, patch 0024** (§3c) — task 9006 stops Elk at the top of its own first iteration, so the exports describe iteration zero, and `elkjax.driver.run()` converges bulk Si from `rhoinit`'s atomic superposition ($\mu=0.1249$ Ha against the converged $0.2140$) in 40 iterations of linear mixing at $\beta=0.4$: `engytot` within **3.6e-4 Ha**, $\mu$ within **4.3e-5 Ha**. **All of that is the frozen core** — the same run with Elk's converged `rhocr`/`engykncr` swapped in gives 3.8e-8 Ha and 4.5e-9 Ha. It also corrected a formula: what may be frozen is $T_{\rm core}$ (`engykncr`), not the core eigenvalue sum, worth **2.0 Ha** and invisible to every test that starts at Elk's answer. The remaining work to close 3.6e-4 Ha is `gencore` in the loop |
| **Gradient A** (inter-mixer difference scaling with `epspot`), **B** ($d\mu/d\varepsilon$ on bcc Fe), **C** (the tolerance plateau) | whether implicit differentiation is actually wired up | **not started**, but the forward blocker is gone: `rhomagk`'s `epsocc` skip is a zeroed weight now (`density.skip_below_epsocc`) and `elkjax.response` replaced JAX's own `eigh` rule, so `scf.step` traces and its `jvp` matches a central difference to 6.6e-9 in the interstitial half (JAX's rule: 1.9e-3, flat in the step size). Nothing has yet been differentiated *through* the converged fixed point |

**Two facts from Phase 3 worth carrying even if the log is never opened.**

- **`genvsig` transforms $v_s\Theta$, not $v_s$** (`rfirftoc.f90`'s first line), while
  `rfirctof` — the map the *density* uses in the other direction — has no such factor.
  So `density.coarsen` is `rfirctof`'s inverse and is **not** `rfirftoc`; using it is
  wrong by a factor of 12, and the symptom is not a slightly wrong potential but an
  empty density, because the spectrum drops below `e0min` and §3a's gate zeroes every
  occupancy.
- **Elk mixes in the middle of its own iteration, and this is now the fourth array pair
  caught on opposite sides of that line.** `init0.f90` makes the mixer's target
  `vsbs` = [`vsmt`, `vsirc`] — the *coarse* interstitial potential — while `vsir` is a
  separate array nothing mixes, so the export carries a `vsir` one un-mixed step ahead
  of the `vsig` built from it: 2.6e-8 at Elk's default `epspot`, 2.0e-9 at 1e-8. The
  previous three are §1k's `haa` (which patch 0015's `genapwlofr` call fixed), §2h's
  muffin-tin density and §2j's two `evecfv` exports. **When two Elk arrays disagree at the size of the last
  mixing step, that is what it is** — check where each is written in `gndstate.f90`
  before looking for a transcription bug.

`docs/continue_here.md` is current as of §3b: both workstreams are on `master`, and its
§3 marks patches 0013/0014/0015, the κ(S) measurement, the projector rule at a real
multiplet, the `match` pole removal, the negative test, the smeared occupations, second
derivatives, the radial integrals/potential derivative and the frozen-potential position
derivative all done. The `ELKPY_F90_LIB` override it documents is still what builds Elk
here.


