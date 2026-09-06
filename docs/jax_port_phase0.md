# Phase 0 of the Elk-to-JAX port: measurements

Running log for the "prove or kill" phase of `docs/jax_port.md` §6. Each item records
what was run, the numbers it produced, and what it settles. Reproduce with

```bash
PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase0b     # the projector rule
PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase0a     # the SCF fixed point
PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase0e     # compile cost, AOT only
ELKPY_RUN_SLOW_TESTS=1 PYTHONPATH=src taskset -c 0-3 python3 -m pytest \
    tests/test_jax_projector.py tests/test_jax_fixedpoint.py \
    tests/test_jax_compile_cost.py -q
```

`taskset` is not decoration: `.claude/settings.json`'s `OMP_NUM_THREADS=1` does **not**
govern XLA's CPU backend (measured — one 1200x1200 `jnp` matmul spawns 40 threads under
it, and `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` changes nothing). Peak RSS is
1.7 GB for either driver; nothing here goes near the memory rules in CLAUDE.md.

| Item | Status |
|---|---|
| 0b — safe-$K$ projector rule | **settled at synthetic $S$, below.** The rule is necessary and it works; item 0b(ii)'s *real* Cholesky-reduced LAPW overlap is still Phase 1's first measurement |
| 0a — reverse-mode implicit diff through the SCF fixed point | **settled, below.** It works, and it needs 0b's rule |
| 0a′ — the same at second order | **settled, below.** Blocked with an `eigh`-based projector; **works** with an eigensolver-free one |
| 0c — `jax.jvp(match)` vs `dmatch.f90` | not started |
| 0d — `vmap(eigh)` vs `lax.map` on a GPU | deferred: no GPU on this machine |
| 0e — compile time and peak memory at production shapes | AOT-only (`elkjax.memory.compiled_cost`) |

---

## 0a. Reverse-mode implicit differentiation through the SCF fixed point

### What was at stake

Study §8(a): the ground state is $v^*=F(v^*;\theta)$, and reverse mode solves the
transposed system $(\mathbb 1-K_{\rm Hxc}\chi_0)^{\mathsf T}\lambda=\partial L/\partial v$
once. The risk is not implicit differentiation in general — a smooth scalar fixed point
with no eigensolve is already known to work and settles nothing. It is that
**$\chi_0$ *is* the derivative of the occupied projector**, so the adjoint matvec is one
Kohn-Sham JVP, transposed: it evaluates `eigh`'s derivative at every multiplet, on every
GMRES iteration. Blockers B1 and B2 are coupled, and 0a is whether the coupling survives.

### The model, and why its degeneracy is not a tuned one

`elkjax.scftoy` is a Kohn-Sham-shaped fixed point with nothing else in it:
$H(v;\theta)=\mathcal L[h_0+\mathrm{diag}(v)]+\theta W$, density $\rho_a=\sum_{\rm copies}[P]_{aa}$,
$F=K_{\rm Hxc}\rho$. The kernel is scaled so that $\rho(K\chi_0)\approx0.84$ — the
self-consistency is doing real work, not supplying a perturbative correction.

The degeneracy is engineered by **doubling**, $\mathcal L[h]=Q(\mathbb 1_2\otimes h)Q^\dagger$,
so every level is exactly two-fold. That is what spin degeneracy in an `nspinor=1` code
is, and unlike tuning two levels to cross it survives the SCF moving $v$. Four spectra
are run: none, doubled and rotated, doubled and block-diagonal, and doubled with a
**symmetry-breaking** $W$ that couples the partners — the last being what a displacement
or a strain lowering a crystal symmetry does, i.e. the forces-and-phonons case.
(The rotated and block-diagonal rows were meant to separate a roundoff-split degeneracy
from a bitwise one. They do not: XLA's `eigh` returns splittings of 1.9e-16 at $m=6$ and
3.9e-16 at $m=12$ but **exactly zero** at $m=8$, which is the size used in the table
below — the backend-dependence §0b measures directly, showing up again.)

The reference is a dense implicit-function-theorem solve in NumPy from the closed-form
projector derivative: no autodiff, LU instead of GMRES. Central FD of the converged
fixed point is reported as a third opinion.

### Result: 0a passes

| spectrum | $dL/d\theta$ (AD) | dense IFT reference | central FD | rel | GMRES residual |
|---|---|---|---|---|---|
| no degeneracy | $-0.0532754795$ | $-0.0532754795$ | $-0.0532754794$ | 1.8e-15 | 3.4e-16 |
| doubled, rotated | $-0.0073965740$ | $-0.0073965740$ | $-0.0073965740$ | 6.0e-15 | 5.7e-16 |
| doubled, bitwise | $-0.0073965740$ | $-0.0073965740$ | $-0.0073965740$ | 1.0e-14 | 1.5e-15 |
| doubled, symmetry-broken | $-0.5369976409$ | $-0.5369976409$ | $-0.5369976494$ | 4.5e-15 | 4.4e-15 |

The same holds for the band energy $\sum_{i\in\rm occ}\varepsilon_i$ (worst 8.2e-15).
And the implicit part is not decoration: against the frozen-potential gradient
$\partial L/\partial\theta|_{v}$ the symmetry-broken band energy is $+0.7145$ against
$+0.1738$, and in the plain doubled case it is $-0.0321$ against $+0.0451$ — **opposite
sign**. A test that forgot the implicit solve would not quietly lose a few percent.

### The control, and a trap it exposed

The identical machinery with the naive projector instead of the rule fails on every
degenerate spectrum — `NaN` at the size where XLA returns a bitwise pair, 47% relative
where it does not. This is what ties 0a's success to 0b: nothing else changes.

**But a symmetry-respecting perturbation hides the bug completely.** With $W$ and the
observable $M$ both commuting with the doubling symmetry they have no matrix element
between the degenerate partners, so the naive rule's erroneous off-diagonal block is
suppressed *twice* and its error drops to 1.5e-14 — a clean pass. The same Hamiltonian
with a symmetry-breaking $W$ gives 4.7e-1. A Phase 1 test suite that perturbs only along
symmetric directions would therefore validate a broken implementation, which is the same
shape of mistake as 0b's single-diagonal-direction check. Pinned as
`test_a_symmetry_respecting_perturbation_hides_the_bug`.

### The three properties that need no reference value

- **Mixer independence — and what it is worth.** Linear mixing and Anderson reach points
  3.7e-13 apart and implicit gradients 2.5e-13 apart. But that agreement is structural,
  not evidence: the `custom_vjp` backward pass is handed only $(\theta, v^*)$, so it
  *cannot* see the mixer. It pins $v^*$. The test with teeth is study §8(a)'s own —
  compare two **unrolled** mixers — and there the effect is far larger than §8(a)'s
  measured 1e-3. Unrolled linear mixing converges normally (relative gradient error
  3.8e-5 at 160 steps, 7.7e-13 at 320). Unrolled **Anderson** reaches a forward value
  good to 1.8e-13 while its gradient is wrong by $10^{17}$ to $10^{32}$ relative, and
  that survives a five-decade sweep of the mixer's internal ridge, so it is not a
  regularisation artefact: the extrapolation is self-correcting forwards and expanding
  in its linearisation, and reverse mode multiplies a few hundred of those Jacobians.
  **This is the objection that matters for Elk**, whose default `mixtype=3` is a Broyden
  scheme built on the same shape of `dgetrf`/`dgetri` history solve. Stated as measured
  with this Anderson implementation; a QR-based or restarted variant might differ, but
  the implicit route is indifferent by construction.
- **Exactness only *at* a fixed point.** The gradient error against the fully converged
  answer tracks the SCF residual linearly — 2.3e-4, 2.4e-6, 2.3e-8, 2.4e-10, 2.0e-12 at
  residuals 8.9e-5 … 9.0e-13 — while agreeing with the IFT solve *at whatever point was
  reached* to 2e-15 throughout. Gradient work needs tighter convergence than a
  forward-only run, which is the same reason §29 drops `epsengy` to $10^{-8}$.
- **Iteration-count independence.** Unrolling the mixer instead reaches the same number
  eventually (2.3e-13 at 400 steps) — §8(a) is right that unrolling is not *inaccurate* —
  but it needs the steps, and it does not even approach monotonically: 1.2e-1 at 20
  steps, 1.0e+0 at 50, 2.0e-2 at 100, 5.1e-7 at 200. The implicit route is $O(1)$ in
  iterations and carries no tape.

Fermi-Dirac smearing at fixed $\mu$ agrees to 2.0e-16, with $\rho(K\chi_0)=1.49$ — so
GMRES inverted a Jacobian the plain iteration is not contractive under. **The
self-consistent Fermi level is not covered**: fixing $\mu$ is grand-canonical, and fixed
electron number adds a second constraint with its own closed-form rule
($d\mu/d\varepsilon_i=w_if'_i/\sum_jw_jf'_j$, §8b), untested here.

## 0a′. Second order

**It works — but not through an `eigh`-based projector.** The blocker turned out to be
one localised piece of machinery, and replacing it answers the item.

### Where it is blocked, and where it is not

All on the symmetry-broken fixed point unless the column says otherwise; FD at
$h=10^{-5}$.

| route | no degeneracy | doubled, `eigh` rule | doubled, sign projector |
|---|---|---|---|
| `grad(grad)` (reverse-over-reverse) | $-0.030860631613$ | `NaN` | $-0.878932465536$ |
| `jacrev(jacrev)` | $-0.030860631613$ | `NaN` | $-0.878932465536$ |
| `jax.hessian`, `jacfwd(grad)` | `TypeError` | `TypeError` | `TypeError` |
| central FD of $dL/d\theta$ | $-0.030860631609$ | $-0.878932466358$ | $-0.878932466580$ |

1. **Reverse-over-reverse through the `custom_vjp` fixed point works** — matching FD of
   the first derivative to 3.6e-7 (FD truncation at $h=10^{-3}$). The implicit-diff
   machinery is twice differentiable; the fixed point is not the problem.
2. **What failed is the safe-$K$ rule's own second derivative.** Its JVP body calls
   `jnp.linalg.eigh`, so differentiating it again falls back on JAX's default
   eigenvector rule and meets the multiplet — `NaN` at the size where the spectrum comes
   back bitwise degenerate, and a wrong finite number (3.3% at $m=6$) where it does not.
   The second derivative demonstrably *exists*: FD returns a finite, convergent number.
3. **`jax.hessian` raising `TypeError` is a JAX limitation, not a result.** It is
   `jacfwd(jacrev)`, and a `custom_vjp` cannot be forward-differentiated at all — the
   same shape of trap as the study's warning about `lax.custom_root` raising
   `NotImplementedError`. Use `grad(grad)`; reporting the `TypeError` as a kill would
   be wrong.

### The fix: a projector with no eigensolve in it

For a hard window with a gapped boundary,

$$
P = \tfrac12\big(\mathbb 1 - \mathrm{sign}(H-\mu\mathbb 1)\big),
$$

with $\mu$ anywhere in the gap and the matrix sign from Newton-Schulz iteration
$X\leftarrow\tfrac12(3X-X^3)$ started at $X_0=(H-\mu)/\|H-\mu\|_2$
(`elkjax.projector.sign_projector`). This is a chain of matrix products: no
eigendecomposition, therefore no gauge to be arbitrary and no $1/(\lambda_i-\lambda_j)$
anywhere, and JAX differentiates it natively to any order.

Two shortcuts in it are **exact rather than approximations**: $P$ is locally constant in
$\mu$ while the gap stays open, so taking $\mu$ from a `stop_gradient`-ed spectrum loses
nothing (and keeps the ill-posed derivative of an individual eigenvalue out of the graph);
and $\mathrm{sign}(A/s)=\mathrm{sign}(A)$, so the normalisation is `stop_gradient`-ed too.

Measured, on the same exactly-degenerate, symmetry-broken fixed point:

- $\|P_{\rm sign}-P_{\rm eigh}\|=2.0\times10^{-15}$, converged in 10 Newton-Schulz steps
  at this gap-to-norm ratio.
- Its **first** derivative equals the safe-$K$ rule's to 1e-10 — which makes it a *third*,
  eigensolver-free code path confirming §0b, not a restatement of it. The naive rule
  disagrees with both by 31% on the same matrices.
- Its **second** derivative through the implicit fixed point is $-0.878932465536$ against
  central FD's $-0.878932466580$, a relative agreement of 1.2e-9 where the `eigh`-based
  rule gives `NaN`. On the smaller instance the FD step was swept: relative error
  5.5e-4, 5.5e-6, 5.5e-8, 5.6e-10 for $h=10^{-2}\ldots10^{-5}$ — textbook $O(h^2)$, i.e.
  FD converging onto the AD value rather than the reverse.
- §0b's padding trap, $H_{\rm pad}=E_{\rm big}\mathbb 1$, goes through unchanged: the
  first derivative matches the closed form exactly and the second is finite, where the
  naive rule returns `NaN`. The pad block is far from $\mu$ and the sign function does
  not care that it is degenerate.

### What this costs, and what it does not cover

The Newton-Schulz loop is unrolled — roughly $\log(\|H\|/\Delta)/\log(3/2)$ iterations
for a gap $\Delta$, so an all-electron spectrum spanning 2500 Ha with a 1 eV gap would
need order 30 and each is three $n^3$ matmuls. That is a real cost against one
diagonalisation, and it is exactly the unrolled tape the study warns about at production
shapes; whether it is affordable is Phase 0e's question, not this one. **Hard windows
only**: smeared occupations would need a Chebyshev expansion of the Fermi function
instead, which is separate work.

So study §6's "the item that actually decides the full port over the hybrid" is answered
in the affirmative — second derivatives through the SCF fixed point are available — with
the qualification that they need a projector built for the purpose, and that this has
been shown on a toy, not on an LAPW Hamiltonian.

---

## 0b. The safe-$K$ projector rule

### What was in dispute

Study §8(b) says the occupied projector $P=\sum_i f_i|i\rangle\langle i|$ "still returns
garbage under JAX's default VJP" whenever the window is gapped, because `eigh`'s rule
divides by $\lambda_i-\lambda_j$ before the $f_i-f_j$ numerator can cancel. A later
check, recorded in `docs/continue_here.md` §3, disagreed for the **hard integer window
with the multiplet fully enclosed** — the two divergent terms are exact negatives there,
so they were expected to cancel bitwise — and measured agreement with central finite
differences of $1.5\times10^{-9}$ and $7.4\times10^{-10}$ on two of three assemblies of
the same spectrum. The third showed $1.3\times10^{-2}$, attributed to FD noise. That
document says what would settle it: a reference that is not finite differences.

### The reference

The closed form, not FD. For any spectral function of a Hermitian $H$ the
Daleckii-Krein formula gives the directional derivative exactly,

$$
dP = V\big(K\circ V^\dagger\,\delta H\,V\big)V^\dagger,
\qquad
K_{ij}=\frac{f_i-f_j}{\lambda_i-\lambda_j},
\qquad
K_{ii}=f'(\lambda_i),
$$

which for a hard window collapses to a sum over occupied-empty pairs only. It is
gauge-invariant, exact, and valid at any $n$ (`elkjax.reference`). Everything is
measured as a directional derivative $d/dt$ along a Hermitian $\delta H$ at real $t$,
which removes the Wirtinger-convention question that `jax.grad` over a complex array
otherwise raises.

### Result: the original claim stands; the caveat does not

Measured on the disputed case itself — the spectrum $(-2,1,1,3,4,5)$ with $n_{\rm occ}=3$,
so the degenerate pair is strictly inside the window, assembled as $U\,\mathrm{diag}\,U^\dagger$
so the pair splits at $1.1\times10^{-15}$ rather than bitwise — over 3 assemblies and
21 directions each, worst relative error:

| route | worst relative error |
|---|---|
| central FD, $h=10^{-5}$ | $3.7\times10^{-8}$ |
| naive AD, forward mode | $1.1\times10^{1}$ |
| naive AD, reverse mode | $3.4\times10^{0}$ |
| safe-$K$ rule, either mode | $2.9\times10^{-14}$ |

**Finite differences are reliable here**, and that is the fact the earlier check needed.
$\mathrm{Tr}[PM]$ is a smooth function of $H$ whenever the *window boundary* is gapped,
however degenerate the interior is, so central FD agrees with the closed form to
$4\times10^{-8}$ and is stable across $h=10^{-4},10^{-5},10^{-6}$. The third assembly's
$1.3\times10^{-2}$ was therefore **AD error, not FD noise**.

**Why the earlier check saw agreement.** It used a single direction — one real diagonal
entry, $\delta H = e_{00}$ — in reverse mode only. That direction is nearly benign:
measured naive reverse-mode error $<10^{-7}$ relative on two assemblies. The same
direction in *forward* mode on the same matrix is already wrong at $1.3\times10^{-2}$,
and a general Hermitian direction breaks reverse mode too. Forward and reverse
disagreeing with each other is itself proof of failure — for a scalar-in, scalar-out
function they are the same number. Both facts are pinned in
`tests/test_jax_projector.py::test_the_benign_direction_that_misled_the_earlier_check`.

### Why the two terms do not cancel

The belief behind the caveat was that the divergent contributions are exact negatives
and cancel bitwise. They *are* exact negatives — and they cancel only if
$A = v^\dagger\,\delta H\,v$ is **bitwise** Hermitian, which it is not. JAX's
`_eigh_jvp_rule` forms `vdag_adot_v` as `dot(dot(_H(v), a_dot), v)` with no
symmetrisation (read: `jax/_src/lax/linalg.py`), and `Fmat` is $1/(\lambda_j-\lambda_i)$
off-diagonal, so what survives for a same-window pair is

$$
\frac{A_{ij}-\overline{A_{ji}}}{\lambda_i-\lambda_j},
$$

a rounding error over a rounding error. Measured on the disputed case:
$\|A-A^\dagger\|_F = 2.0$–$2.9\times10^{-16}$ against a splitting of
$4.4\times10^{-16}$–$1.8\times10^{-15}$, i.e. a ratio of $0.16$–$0.45$ — which is the
size of the failure actually observed. Pinned in
`tests/test_jax_projector.py::test_why_the_terms_do_not_cancel_bitwise`.

This also explains the direction dependence: $e_{00}$ is a real, rank-one, diagonal
perturbation, for which that asymmetry is anomalously small.

### How the failure scales, and which face it wears

The naive error grows as the enclosed pair closes ($n=400$, hard window, relative to the
closed form):

| pair splitting | naive fwd | naive rev | safe-$K$ |
|---|---|---|---|
| $10^{-4}$ | $1.6\times10^{-12}$ | $5.9\times10^{-13}$ | $2.3\times10^{-13}$ |
| $10^{-8}$ | $7.3\times10^{-8}$ | $3.3\times10^{-9}$ | $5.0\times10^{-13}$ |
| $10^{-12}$ | $6.0\times10^{-5}$ | $4.0\times10^{-5}$ | $8.5\times10^{-14}$ |
| $1.5\times10^{-14}$ | $1.0\times10^{-1}$ | $1.7\times10^{-2}$ | $1.6\times10^{-12}$ |
| $4.9\times10^{-15}$ (requested 0) | $1.7\times10^{-1}$ | $2.3\times10^{-1}$ | $1.1\times10^{-12}$ |

**Whether the naive route returns a wrong number or a `NaN` is the eigensolver's
choice, not the physics'.** The same matrix reduced through a Cholesky factor gives
different splittings from LAPACK and from XLA, and at $n=1000$ XLA returns the pair
*bitwise* equal where LAPACK splits it at $1.5\times10^{-14}$ — JAX's rule then forms
$1/0=\infty$ and multiplies it by an exactly-zero numerator:

| $n$ | splitting (LAPACK) | splitting (XLA) | naive reverse | safe-$K$ |
|---|---|---|---|---|
| 200 | $1.6\times10^{-14}$ | $6.0\times10^{-15}$ | $6.7\times10^{-3}$ | $6.3\times10^{-13}$ |
| 400 | $2.7\times10^{-14}$ | $6.7\times10^{-16}$ | $2.6\times10^{-2}$ | $3.5\times10^{-12}$ |
| 600 | $1.0\times10^{-14}$ | $5.3\times10^{-15}$ | $2.3\times10^{-3}$ | $3.4\times10^{-15}$ |
| 800 | $2.0\times10^{-14}$ | $2.7\times10^{-15}$ | $5.7\times10^{-2}$ | $1.5\times10^{-12}$ |
| 1000 | $1.5\times10^{-14}$ | $0$ | `NaN` | $1.1\times10^{-11}$ |

So §8(b)'s "garbage" and §10 item 1's "`NaN`" are one bug wearing whichever face the
backend's rounding picks, and neither is reproducible across backends on one matrix.
That is worse than a consistently wrong answer: a Phase 1 developer would see a test
pass at $n=400$ and a `NaN` at $n=1000$.

### The other three sub-items

- **Reassembly jitter** (perturb $H$ at $10^{-16}$; the answer moves by $10^{-16}$):
  relative spread of the gradient is $6.4\times10^{-1}$ naive, $6.4\times10^{-15}$ with
  the rule, against $3.3\times10^{-15}$ for the closed form. Study §8(b) reported
  159% → $4.3\times10^{-15}$; same conclusion, same order.
- **The padding block.** $H_{\rm pad}=E_{\rm big}\mathbb 1$ with a *non-degenerate*
  physical block: naive reverse mode returns `NaN`, the rule returns the right number.
  Confirms §8(b)'s warning; §3.2's obvious padding choice is a day-one trap.
- **Smeared occupations** (Fermi-Dirac, $\mu=2.0$, $w=0.3$): naive is wrong by up to
  $3.3\times10^{0}$ relative, the rule agrees to $4.6\times10^{-14}$. This is the case
  §8(b) and the later check already agreed on.

### The tolerance

The `jnp.where` threshold must come from the eigenvalue backward error
$\epsilon\,\kappa(S)\,\|H\|$ of the **generalized** problem (§8b). Two findings:

1. **For a hard window the tolerance is inert away from the window boundary.** Both
   branches of $K$ are identically zero for a same-side pair ($f_i-f_j=0$ exactly, and
   $f'=0$), so a hard-window `tol` sweep is flat for a trivial reason and settles
   nothing. The plateau test is only meaningful with smearing, where $f_i-f_j$ is a
   genuine small number. Run there ($n=400$, $\kappa(S)=10^6$), the gradient is flat to
   $2.4\times10^{-12}$ over **eight decades** of `tol`, degrading only at the bottom
   ($8.3\times10^{-8}$ when `tol` drops below the pair's own $8.9\times10^{-12}$
   splitting). Study §8(b) asks for two decades; there are eight.
2. **§8(b)'s cheap $\kappa(S)$ estimate from the Cholesky diagonal is a provable
   *lower bound*, not an approximation.** Each $L_{ii}^2$ is a Schur-complement pivot,
   $\det S_{1:i}/\det S_{1:i-1}$, and therefore lies in
   $[\lambda_{\min}(S),\lambda_{\max}(S)]$ — so
   $(\max\mathrm{diag}\,L/\min\mathrm{diag}\,L)^2 \le \kappa_2(S)$ always, with no
   bound on how loose it gets. Measured instance: a synthetic $S$ with
   $\kappa_2(S)=10^6$ by construction returns $7.2\times10^{3}$, 140x low. Since the
   tolerance is meant to be an *upper* bound on the indistinguishable splitting, that is
   the dangerous direction, and the estimate needs a safety factor rather than trust.
   `elkjax.phase0b` uses $10^3\times$ throughout — a placeholder, not a derived number.
   Whether any of it matters is unknown, because $\kappa(S)$ for Elk's APW+lo overlap
   has still never been measured; that is Phase 1's first job.

Where a window boundary itself is degenerate there is no differentiable occupied
subspace at all, and `elkjax.projector` returns `NaN` deliberately rather than a
plausible number. The standing mitigation is the one this project already applies to
Berry curvature (CLAUDE.md §13): window the whole degenerate group together.

### What is NOT settled by this

- **The rule is first-order only.** Its JVP body calls `jnp.linalg.eigh`, so
  differentiating it a second time falls back on JAX's default eigenvector rule and the
  hazard returns. Measured in 0a′ above; the answer there is to use `sign_projector`
  instead when a second derivative is wanted, at the cost of an unrolled Newton-Schulz
  loop and hard windows only.
- **$\mu$ is held fixed** in the smeared projector. The self-consistent Fermi level has
  its own closed-form rule ($d\mu/d\varepsilon_i = w_if'_i/\sum_j w_jf'_j$, §8b) and
  belongs to Phase 0a, where the electron-number constraint enters.
- **No LAPW matrix has been near this.** Every $S$ here is a synthetic SPD matrix with a
  prescribed condition number. The measurements pin the *arithmetic*, not Elk.

---

## 0e. Compile time and peak memory for one traced SCF step

Study §6 item 0e wants both numbers at production shapes ($n_{\rm mat}\approx3000$,
$n_{\bf k}\approx100$), and warns that §8(d)'s reassuring toy-scale figures
($n\le400$, 4 k-points) must not be extrapolated, because §3 proposes unrolled
constructs — an 8-pass corrector, unrolled Gram-Schmidt, a scan over ~700 radial
points — inside a step that also contains per-k eigensolves.

**Nothing was executed.** `jax.jit(f).lower(*avals).compile()` builds the executable from
abstract shapes, so the 26.8 GiB this machine cannot hold was never allocated; peak RSS
for the whole sweep is 1.4 GB. The shapes, dtype, loop structure and op-count-inflating
constructs are the real ones; the *arithmetic* inside them is a stand-in, since `match`,
Weinert and XC are Phase 1 and 2 work. These are therefore a lower bound on the real
step's cost — the useful direction for a "is this a wall?" question — and the bound is
loose in exactly the direction §5's own design pushes, since §5 puts
`apwfr`/`lofr`/`socfr` *inside* the fixed point, so a real step also carries the radial
ODE and the density/potential pass. One factor can be measured rather than guessed:
compiling `jax.grad` of the step, the unit a Phase 3 gradient actually builds, costs
0.45 s against 0.36 s undifferentiated at $(800,4)$, 1.67 s against 1.57 s with 32 local
orbitals, and 0.51 s at the full $(3000,100)$ with temporaries unchanged at 0.411 GiB.
Differentiation is a ~1.2x factor here, not a 20x one.

### Compile time is flat in the shapes

| $n_{\rm mat}$ | $n_{\bf k}$ | k axis | compile (s) | arguments (GiB) | temporaries (GiB) |
|---|---|---|---|---|---|
| 200 | 4 | scan | 0.49 | 0.005 | 0.002 |
| 3000 | 4 | scan | 0.46 | 1.073 | 0.411 |
| 3000 | 25 | scan | 0.46 | 6.706 | 0.411 |
| 3000 | 100 | scan | 0.46 | 26.822 | 0.411 |
| 3000 | 100 | `lax.map` | 0.50 | 26.822 | 0.413 |
| 3000 | 100 | `vmap` | 0.51 | 26.822 | **40.233** |

A 15,000-fold increase in problem size changes compile time by less than 30%, which is
within the run-to-run scatter. **Hazard K is not a wall from problem size**: XLA compiles
an HLO graph whose op count is independent of the tensor extents.

The `arguments` column is an independent confirmation of the figure CLAUDE.md quotes:
$H$ and $S$ over 100 k-points at $n=3000$ really are 26.8 GiB, against ~28 GiB available.

### The k-axis, which settles the memory half of item 0d

Item 0d's *timing* question needs a GPU this machine does not have. Its memory question
does not, and the answer is not close: at the production shape a `lax.scan` accumulator
holds **0.411 GiB** of temporaries and `vmap` holds **40.2 GiB** — 98x, and above this
machine's entire RAM. The `scan` figure is *exactly* independent of $n_{\bf k}$;
`lax.map` sits between the two, at 0.413 GiB, because it still stacks the per-k
**outputs** ($n_{\bf k}\times n_{\rm mat}$, not $n_{\bf k}\times n_{\rm mat}^2$) — small,
but not nothing, and avoidable by carrying an accumulator instead.

So `vmap` over the k-axis is not a style preference to be settled by a benchmark. On a
40 GB device it does not fit, whatever the GPU timing turns out to be.

### Where the cost actually is: unrolled op count

At one fixed shape ($n_{\rm mat}=800$, $n_{\bf k}=4$), varying only how much is unrolled:

| construct | compile (s) | temporaries (GiB) |
|---|---|---|
| no unrolled Gram-Schmidt | 0.19 | 0.029 |
| baseline (8 local orbitals, 8-pass corrector) | 0.44 | 0.029 |
| 32 local orbitals | 1.84 | 0.029 |
| 64 local orbitals | 6.65 | 0.029 |
| 128 local orbitals | 32.38 | 0.029 |
| 64-pass corrector | 1.28 | 0.029 |
| 256-pass corrector | 16.47 | 0.029 |
| `lax.scan` over 2800 radial points instead of 700 | 0.37 | 0.029 |

Two factors are being conflated if this is read as one exponent, and the corrector is
what separates them. Modified Gram-Schmidt over $n_{\rm lo}$ columns emits
$O(n_{\rm lo}^2)$ inner bodies by construction — 36 at 8 columns, 8256 at 128 — so its
apparent exponent 2.07 in $n_{\rm lo}$ is mostly the *op count itself* growing
quadratically. The corrector is linear in its pass count, and it gives 1.28 s → 16.47 s
for 64 → 256 passes: **compile time is superlinear, exponent ≈1.85, in HLO op count**.
Neither shows up in the memory column, so op count cannot be traded against buffers.
Meanwhile quadrupling a `lax.scan`'s trip count costs **nothing**, because a scan is a
loop in the HLO rather than a tape.

The design rule follows directly, and it is the useful output of this item: **`scan`
every repeated structure and unroll only what genuinely must be**. Elk's `nlotot` runs to
the low hundreds for a heavy cell, so §3.2's unrolled Gram-Schmidt over the local-orbital
block is precisely the construct that would put a step into the tens of seconds per
compile — per shape, and a different `ngridk` or `rgkmax` is a different shape.
