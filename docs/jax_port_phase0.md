# Phase 0 of the Elk-to-JAX port: measurements

Running log for the "prove or kill" phase of `docs/jax_port.md` §6. Each item records
what was run, the numbers it produced, and what it settles. Reproduce with

```bash
PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase0b
ELKPY_RUN_SLOW_TESTS=1 PYTHONPATH=src taskset -c 0-3 python3 -m pytest tests/test_jax_projector.py -q
```

`taskset` is not decoration: `.claude/settings.json`'s `OMP_NUM_THREADS=1` does **not**
govern XLA's CPU backend (measured — one 1200x1200 `jnp` matmul spawns 40 threads under
it, and `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` changes nothing). Peak RSS for the
whole suite is 1.7 GB.

| Item | Status |
|---|---|
| 0b — safe-$K$ projector rule | **settled at synthetic $S$, below.** The rule is necessary and it works; item 0b(ii)'s *real* Cholesky-reduced LAPW overlap is still Phase 1's first measurement |
| 0a — reverse-mode implicit diff through the SCF fixed point | not started |
| 0a′ — the same at second order | not started; note the rule below is first-order only |
| 0c — `jax.jvp(match)` vs `dmatch.f90` | not started |
| 0d — `vmap(eigh)` vs `lax.map` on a GPU | deferred: no GPU on this machine |
| 0e — compile time and peak memory at production shapes | AOT-only (`elkjax.memory.compiled_cost`) |

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
  hazard returns. Phase 0a′ needs the rule made recursive, or forward-over-forward.
- **$\mu$ is held fixed** in the smeared projector. The self-consistent Fermi level has
  its own closed-form rule ($d\mu/d\varepsilon_i = w_if'_i/\sum_j w_jf'_j$, §8b) and
  belongs to Phase 0a, where the electron-number constraint enters.
- **No LAPW matrix has been near this.** Every $S$ here is a synthetic SPD matrix with a
  prescribed condition number. The measurements pin the *arithmetic*, not Elk.
