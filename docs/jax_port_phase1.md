# Phase 1 of the Elk-to-JAX port: measurements

Running log of what Phase 1 has actually measured, in the shape
`docs/jax_port_phase0.md` established: what was at stake, what was run, and
what is *not* settled by the result. `docs/jax_port.md` §6 is the plan;
Phase 0's log is the sibling file, and its three carried-forward findings
(a green gradient test does not validate a transcription; never unroll a
Pulay-type mixer; `scan` repeated structure) apply here unchanged.

Phase 1 as written is "one k-point, one species, no SCF": build
`match` → `hmlfv`/`olpfv` → Cholesky-reduced `eigh`, reading a converged
`STATE.OUT` as a fixed input. `match` was closed in Phase 0c and against
Elk's own `apwalm` by patch 0013, so the first build step is the assembly.

| item | result |
|---|---|
| **1a** `olpfv`/`hmlfv`, muffin-tin blocks, forward | **done — machine precision on three fixtures**, six blocks compared separately; assembled pair reproduces Elk's `evalfv` to 9e-15 Ha |
| **1b** the radial integrals from `genapwfr`/`genlofr`/`hmlrad`/`olprad` | not started; needs the muffin-tin potential, i.e. Phase 2 |
| **1c** the interstitial blocks from `gencfun`/`genvsig` | not started; `vsig` needs the interstitial Kohn-Sham potential, i.e. Phase 2 |
| **1a′** the assembly as a differentiable function of $k$ | **done — and it found that $d\varepsilon/dk \neq \langle p\rangle$ in a finite LAPW basis** |
| **1d** gradient vs finite differences on displaced h-BN | not started |
| **1e** the adversarial `soc_scale` sweep, and the required refusal | **withdrawn as written — `soc_scale` cannot move the first-variational spectrum at all** (§1f); the refusal itself is done, by cutting a real multiplet |
| **1f** the Cholesky-reduced eigensolve wired to the safe-$K$ projector rule, on real matrices | **done — the rule is needed and it works**: 3.9e-1 (forward) / `NaN` (reverse) naive against 2.2e-11 safe, at a multiplet Elk's own matrices supply by symmetry |

---

## 1a. The Hamiltonian and overlap, assembled

### What was at stake

Everything downstream of the eigenproblem is built on $H$ and $O$, and the
usual way this goes wrong is not a visibly broken matrix. Both are Hermitian
and positive definite by construction from almost any consistent-looking
index convention, so a transposed axis, a conjugation on the wrong factor or a
misread Gaunt slot produces a matrix that diagonalises happily and gives
eigenvalues that are merely *different*. There is no internal check with teeth;
the reference has to be Elk's own array.

Patch 0013 supplied $H$, $O$ and `apwalm`. It did not supply what `apwalm` is
contracted with, so patch **0014** appends `oalo`, `ololo`, `haa`, `hloa`,
`hlolo`, the local-orbital bookkeeping and `gntyry` (`docs/design.md` §33).

### What is built, and what is taken

Both matrices split into an interstitial part and one muffin-tin part per
sphere. **Only the muffin-tin halves are built.** The interstitial blocks

$$O^{\rm I}_{ij} = \tilde\Theta(\mathbf G_i - \mathbf G_j), \qquad
H^{\rm I}_{ij} = V_s(\mathbf G_i - \mathbf G_j)
 + \tfrac12(\mathbf G_i + \mathbf k)\cdot(\mathbf G_j + \mathbf k)\,
   \tilde\Theta(\mathbf G_i - \mathbf G_j)$$

are taken from the export. That is not a shortcut deferred for time: $V_s$ is
the *interstitial Kohn-Sham potential*, an output of Phase 2, so $H^{\rm I}$
could not be built here whatever the effort. $\tilde\Theta$ alone is
closed-form (`gencfun`) but building half a block buys nothing.

The muffin-tin part of a sphere is, with $A$ the matching coefficients
flattened into Elk's own `do l; do lm; do io` order,

$$O^{\rm MT} = A^\dagger A, \qquad H^{\rm MT} = A^\dagger Z A, \qquad
Z_{(\ell_1m_1,o),(\ell_3m_3,o')} = \sum_{\ell_2m_2}
 \langle Y_{\ell_1m_1}|R_{\ell_2m_2}|Y_{\ell_3m_3}\rangle\,
 h^{\alpha}_{\ell_2m_2,\,o'\ell_3,\,o\ell_1}.$$

The overlap carries no radial integral because Elk normalises the APW radial
functions on the sphere. The angular selection rule ($\ell_1+\ell_2+\ell_3$
even, and the triangle condition) is **not** imposed by hand: `gntyry` is
exactly zero otherwise, which is what Elk's own `do l2 = l0, lmaxo, 2` stride
exploits, so a dense contraction reproduces it.

Local orbitals add `nlotot` columns whose matching coefficients are trivial —
a local orbital is confined to one sphere — so their blocks are the radial
integrals themselves, contracted with the same Gaunt array for $H$.

### Result: machine precision, six blocks at a time

Six blocks are compared **separately** rather than only in total, so a failure
names one upstream routine instead of "$H$ is wrong somewhere". Generic
k-point $(0.1, 0.2, 0.05)$; `max|diff|` against Elk's own matrices with the
exported interstitial contribution subtracted off:

| block | bulk Si, `apword=1` | bulk Si, `apword=2` | monolayer h-BN |
|---|---|---|---|
| `olpaa` APW-APW | 6.7e-16 | 8.3e-16 | 1.9e-16 |
| `olpalo` APW-lo | 0 | 0 | 0 |
| `olplolo` lo-lo | 0 | 0 | 0 |
| `hmlaa` APW-APW | 6.7e-16 | 1.8e-15 | 6.5e-15 |
| `hmlalo` APW-lo | 2.0e-17 | 5.6e-17 | 3.3e-14 |
| `hmllolo` lo-lo | 0 | 0 | 0 |
| **`evalfv` after `eigh`** | **2.4e-15 Ha** | **3.1e-15 Ha** | **9.1e-15 Ha** |

The exact zeros are not vacuous: every reference block is asserted to have a
magnitude above $10^{-3}$, and the overlap's local-orbital blocks are single
multiplications of exported doubles, so bitwise agreement is what should
happen. The `evalfv` row is the whole-object check the block-by-block one
cannot be — `evalfv` comes from `eveqnfv` through Elk's own configured path,
*before* the export's `tefvr` override — so it pins the assembly, the
local-orbital column numbering and the interstitial padding together.

### The three fixtures, and what each alone can catch

- **Bulk Si, `apword=1`.** The stock case: every species file Elk ships sets
  `apword = 1`.
- **Bulk Si, `apword=2`.** The only way to exercise the APW *order* axes of
  `haa` and `hloa`, which are length 1 otherwise — so at `apword=1` an
  $i_o \leftrightarrow \ell$ axis swap is not merely undetected, it is a
  no-op. Same generated species file the 0013 export test uses.
- **Monolayer h-BN.** Two species, hence `idxis`'s indexing into
  `rmt`/`apword`/`nlorb`; and, decisively, a nitrogen carrying **two $\ell=0$
  local orbitals**.

Mutation-tested, one deliberate error at a time: dropping the local-orbital
ordering restriction below fails **h-BN alone**, as claimed; transposing $Z$
and conjugating the wrong factor in `hmlalo` each fail every fixture.

### The trap only h-BN can see

`hmlrad.f90` builds `hlolo`'s $\ell_2=0$ element as

$$\int u^{\rm lo}_i(r)\,\bigl(\hat H u^{\rm lo}_j(r)\bigr)\,r^2\,dr,$$

with **no symmetrisation** over $(i,j)$ and no kinetic surface term — unlike
`haa`, whose $\ell_2=0$ counterpart is explicitly averaged over the two
orderings, carries the surface term, and has its transpose explicitly
assigned. `hmllolo` therefore runs `do jlo; do ilo = 1, jlo` with an
`if (i > j) cycle`, evaluating each local-orbital pair in **one order only**
and Hermitising the rest; and because `genidxlo` numbers columns in increasing
$(i_{\rm lo},\ell m)$, that set is exactly the matrix upper triangle.

Using both halves of the exported array gives a Hermitian, positive-definite,
plausible, wrong matrix. Measured on h-BN's nitrogen: the two orderings differ
by $1.3\times10^{-2}$ Ha, reaching the assembled $H$ as $3.7\times10^{-3}$ Ha
and `evalfv` as $4.3\times10^{-7}$ Ha. Silicon's local orbitals are one s and
one p, so no pair shares an $\ell$ and the bug is invisible there — which is
what a two-fixture suite would have shipped.

This is a third instance of Phase 0's standing finding, in a new guise: the
check that passes is not the check with teeth. There, a symmetric perturbation
hid a projector bug and a constant prefactor survived an exact derivative
identity. Here, a fixture without a repeated $\ell$ hides an asymmetry.
Accordingly the h-BN premise is **asserted rather than assumed**: one test
fails if no species has a repeated $\ell$, and another if `hlolo`'s $\ell_2=0$
block is symmetric to better than $10^{-4}$, so the fixture cannot quietly
stop testing what it exists to test if Elk's shipped `N.in` changes.

### The residual is Elk's guard, not the transcription

h-BN's `hmlalo` sits at 3.3e-14 rather than 1e-16. `hmlaa` and `hmlalo` skip
any Gaunt-contracted $z_1$ with $|{\rm Re}\,z_1| + |{\rm Im}\,z_1| \le 10^{-12}$
(the `zaxpy` guard), which the dense contraction here keeps. Reproducing the
guard takes `hmlalo` from 3.3e-14 to **3.0e-17** and `hmlaa` from 6.5e-15 to
1.6e-15, so the attribution is measured rather than assumed. 64% of the
$z_1$ elements fall below it, almost all of them structural Gaunt zeros.

The guard is documented, not copied: the dense version is the more accurate of
the two, and a port should not inherit a numerical shortcut whose only purpose
is to skip a `zaxpy`. The consequence is a floor of $\sim10^{-12}$ on any
element-wise comparison against Elk, and the test's tolerance says so.

### What is NOT settled by this

- **The radial integrals are inputs, not outputs.** `haa`, `hloa`, `hlolo`,
  `oalo` and `ololo` are taken from Elk. Building them needs `genapwfr`,
  `genlofr`, `hmlrad` and `olprad`, and through `vsmt` the muffin-tin
  potential — Phase 2. The same was true of $D$ after Phase 0c, and is the
  standing shape of the port: each phase closes the layer above the one it
  still imports.
- **Nothing has been differentiated.** This is forward agreement only.
  Phase 0's first carried-forward finding applies in the other direction too:
  a green forward check does not validate a gradient. `apwalm` already
  differentiates exactly (0c), and $Z$ is a fixed contraction, so the
  position-derivative path is short — but it is untested, and Phase 1's
  gradient criteria (displaced h-BN against central finite differences at
  three step sizes; the adversarial `soc_scale` sweep with a *required*
  refusal; the negative test at an exact degeneracy) are all still open.
- **One k-point, and no `eigh` of the port's own.** The eigenvalues above come
  from `scipy.linalg.eigh` on the assembled pair, not from the Cholesky
  reduction Phase 1 is meant to build, and not from a differentiable one.
  Phase 0b's safe-$K$ projector rule and the $\kappa(O)\approx5\times10^3$
  tolerance it needs are what that step has to be wired to.


---

## 1a′. The spectrum as a function of $k$, and the first derivative

### What this needed, and why it needed no new Fortran

Everything in 1a is frozen at one $k$: `apwalm` comes from the export. Making
it a *function* of $k$ needs `match` — already transcribed (0c) and already
checked against Elk's own array element-wise (patch 0013) — plus the two
interstitial blocks, which are where $V_s$ lives and $V_s$ belongs to Phase 2.

Two observations remove that obstacle without exporting anything new.
$O^{\rm I}_{ij}=\tilde\Theta(\mathbf G_i-\mathbf G_j)$ does not depend on $k$
at all; and
$H^{\rm I}_{ij}=V_s(\mathbf G_i-\mathbf G_j)+\tfrac12(\mathbf G_i+\mathbf
k)\cdot(\mathbf G_j+\mathbf k)\,\tilde\Theta(\mathbf G_i-\mathbf G_j)$ differs
from it by an explicit, closed-form kinetic term. So

$$V_s(\mathbf G_i-\mathbf G_j) = H^{\rm I}_{ij}(\mathbf k_0)
 - \tfrac12(\mathbf G_i+\mathbf k_0)\cdot(\mathbf G_j+\mathbf k_0)\,
   O^{\rm I}_{ij}$$

is recoverable as a **matrix**, elementwise in $(i,j)$, with no need to know
which pair maps to which G-vector difference. Held fixed, it is exactly right
for varying $k$ at a fixed ground state — which is what a band velocity is.
The radial integrals and `gntyry` are $k$-independent too, so the only
remaining $k$-dependence is `match` and that kinetic term. Closed with a
Cholesky-reduced `eigvalsh`, the reduction whose $\kappa(O)$ 0b(ii) measured.

The G-vector *set* is held fixed. It is a property of the cutoff and changes
discontinuously with $k$, which is irrelevant to a derivative at a point but
does mean the function is only valid near the exported $k$.

### Forward, and the AD control

Evaluated back at the exported $k$, everything rebuilt reproduces Elk's own:

| | Si `apword=1` | Si `apword=2` | h-BN |
|---|---|---|---|
| $H$ | 9.8e-15 | 1.3e-12 | 3.3e-14 |
| $O$ | 2.7e-15 | 6.2e-13 | 2.2e-15 |
| `evalfv` | 5.0e-15 | 3.8e-15 | 5.9e-15 |

`apword=2`'s *matrices* inherit `match`'s general branch, whose $2\times2$
solve has rows $u(R)$ and $u'(R)$ differing by orders of magnitude — 0c
measured 8.0e-13 there against 1.9e-15 in the division branch. The
**spectrum does not inherit it** (3.8e-15 either way): the error is a
near-null-space rotation inside the APW order space, which eigenvalues are
blind to.

`jax.grad` of an individual eigenvalue with respect to Cartesian $\mathbf k$
agrees with central differences **of the same function** to 1e-9 (step
$10^{-5}$, i.e. at the FD truncation floor). That control has no physics in
it and is the point: it separates an AD bug from the real effect below, which
Phase 0's first standing finding says one must never assume.

### The finding: $d\varepsilon/dk$ is not $\langle p\rangle$ in a finite LAPW basis

Against an entirely independent Fortran path — `genpmatk`, elkpy's §22
`MOMENTUM` query, which shares no code with `hmlfv`/`olpfv` — the two agree in
sign and magnitude but **not to machine precision**. Bulk Si at
$(0.1,0.2,0.05)$, relative difference of the full Cartesian vector:

| band | `apword=1` | `apword=2` |
|---|---|---|
| 0 | 2.34e-3 | 2.04e-3 |
| 3 | 2.20e-3 | 1.04e-3 |
| 7 | 1.41e-2 | 3.41e-3 |

**It is not the plane-wave cutoff.** Sweeping `rgkmax` $7\to8\to9$ (ngp
$153\to219\to315$) leaves band 0's gap at 2.336e-3, 2.339e-3, 2.340e-3 — flat
to three digits — while the eigenvalue itself converges
($-0.222006\to-0.221692\to-0.221627$ Ha). The basis is demonstrably improving
and the disagreement does not care.

**It is the muffin-tin linearisation.** Raising `apword` from 1 to 2 —
augmenting with $\dot u = \partial u/\partial E$ as well as $u$, i.e. the
textbook LAPW basis rather than plain APW — cuts it by up to 4x, and cuts it
most for band 7, where it was largest and where the eigenvalue is furthest
from the linearisation energy. That is the right pattern for the right reason.

**The mechanism.** Hellmann-Feynman gives $d\varepsilon_n/d\mathbf k =
\langle\psi_n|\partial\hat H/\partial\mathbf k|\psi_n\rangle$ for a complete,
$k$-**independent** basis. The LAPW basis is neither. $H$ and $O$ both depend
on $k$ — through the plane waves and, in the spheres, through `match` — so
what `jax.grad` returns is the variational derivative

$$\frac{d\varepsilon_n}{d\mathbf k} = v^\dagger\!\left(
 \frac{\partial H}{\partial\mathbf k} - \varepsilon_n
 \frac{\partial O}{\partial\mathbf k}\right)\!v ,
 \qquad v^\dagger O v = 1,$$

including the $\partial O/\partial\mathbf k$ term, while `genpmatk` returns
$\langle\psi_n|-i\nabla|\psi_n\rangle$. The two coincide only in the complete-
basis limit, and the limit that matters here is the radial one, which
`rgkmax` does not touch.

**Two practical consequences.** For the port: `genpmatk` is not a
machine-precision reference for a $k$-derivative, so Phase 1's remaining
gradient criteria must finite-difference the *same* code path (as the study
already specifies for the position derivative) and keep an independent
reference for what it is — an agreement-in-physics check at the percent level,
not a correctness oracle. For elkpy itself: this is the mechanism behind
`tests/test_calculation_momentum.py`'s own Hellmann-Feynman check needing
`rel=2e-2`, and anyone reading `get_momentum_matrix`'s diagonal as a band
velocity is making an approximation of that size which does **not** improve
with the plane-wave cutoff.

The test
(`test_k_derivative_and_elks_momentum_differ_by_muffin_tin_incompleteness`)
asserts the direction of the effect rather than a tolerance, and additionally
requires the two **not** to agree to better than 1e-4 — so if a future change
made them coincide, the test fails rather than silently passing a bound whose
premise had evaporated.

### What is NOT settled by this

- **This is the $k$-derivative, not the position derivative.** Phase 1's
  stated gradient criterion is $d\varepsilon_j/d\mathbf R$ on displaced h-BN,
  and that is harder: moving an atom moves the muffin-tin potential and hence
  the radial integrals, which are imported here. What is closed is that AD
  runs end to end through `match`, the assembly, the Cholesky and the
  eigensolve, and returns the right number.
- **No degeneracy has been touched** *(closed by §1f)*. The fixture here is a
  generic $k$ with every low band separated by $>10^{-3}$ Ha, asserted in the
  test; §1f runs the same machinery at $\Gamma$, where a real multiplet sits
  inside the occupied window.
- **`eigvalsh`, not the safe-$K$ rule** *(closed by §1f)*. The Cholesky
  reduction landed here unwired to `projector.py`'s custom rule and to a
  per-run $\kappa(O)$; §1f is that wiring, and measures that at a multiplet
  the unwired pipeline fails exactly the way 0b describes.
- **The $k$-derivative is only available away from the $z$-axis** — found by
  §1f, and it is a limitation of *this* section's `eigenproblem_at`, not of
  the projector: see §1f's second finding.

---

## 1f. The eigensolve wired to the safe-$K$ projector rule

### What was at stake

`projector.py`'s safe-$K$ rule was written for Phase 0b and has worked since,
but it had **only ever been exercised on a synthetic overlap with a prescribed
$\kappa(S)$** — `docs/continue_here.md` §3 calls that the biggest hole left in
Phase 0. §1a′'s `first_variational_eigenvalues`, meanwhile, closes with a plain
`eigvalsh`, so at a multiplet it fails exactly the way 0b describes. The two had
never met.

What a real ground state supplies that no synthetic pair can is a **real
multiplet with a real tolerance around it**. Diamond silicon at $\Gamma$ carries
the three-fold $\Gamma_{25'}$ valence level *inside* its four occupied
first-variational bands, with the boundary to the conduction band open by
0.093 Ha — which is the disputed configuration of experiment 0b-A, arrived at by
symmetry instead of by engineering a spectrum. A generic $k$ on the same ground
state is the control: no degeneracy anywhere, so both AD routes must agree there
or the comparison is measuring the fixture rather than the rule.

### What was built

`elkjax.hamiltonian` gained the last step of the chain the study's Phase 1 lists
(`match` → `hmlfv`/`olpfv` → Cholesky-reduced `eigh`):

- `cholesky_reduce(h, o)` → $(\tilde H, L)$ with $O=LL^\dagger$,
  $\tilde H=L^{-1}HL^{-\dagger}$ — factored out of §1a′'s eigenvalue routine so
  one expression serves both.
- `projector_tolerance(reduced, o)` = $\epsilon\,\kappa(O)\,\lVert\tilde
  H\rVert_2$, **measured on this run**. Three details are conclusions of
  0b(ii) rather than preferences: $\kappa(O)$ comes from a dense `eigvalsh`
  because §8(b)'s Cholesky-diagonal estimate is uninformative on real overlaps;
  the norm is the *reduced* one; and it is recomputed per run because
  $\kappa(O)$ is a cutoff property.
- `occupied_window(export, kc, nocc)` — the host-side **refusal**, returning
  $(\mathrm{tol}, \mathrm{gap})$ and raising when the boundary gap is at or
  below the tolerance. The study asks for "an explicit refusal (not a returned
  number)"; the in-graph `NaN` of the safe-$K$ rule is a safety net, and a
  `NaN` is still a returned non-number.
- `occupied_projector(export, kc, nocc, tol)` — the projector itself,
  differentiable in $k$, through `hard_window_projector`. `tol` is required
  rather than defaulted, so a caller cannot differentiate through an unchecked
  boundary while believing a threshold was applied.
- `elk_occupied_projector(export, nocc)` — the same subspace from Elk's own
  `evecfv`, as $YY^\dagger$ with $Y=L^\dagger C$, orthonormal because Elk
  normalises $C^\dagger OC=\mathbb 1$.

Driver `python3 -m elkjax.phase1_projector`; test
`tests/test_calculation_lapw_projector.py` (11 tests, 37 s with the ground
state cached).

### Result

Three fixtures, all at `rgkmax=7`: bulk Si at $\Gamma$ (multiplet enclosed), the
same ground state at $k=(0.1,0.2,0.05)$ (control), and monolayer h-BN at
$\Gamma$.

| | Si $\Gamma$ | Si generic | h-BN $\Gamma$ |
|---|---|---|---|
| $n_{\rm mat}$ / $n_{\rm occ}$ | 177 / 4 | 161 / 4 | 906 / 4 |
| $\kappa(O)$, $\lVert\tilde H\rVert$ | 1.31e4, 17.89 | 4.97e3, 14.25 | 4.91e3, 58.50 |
| tolerance $\epsilon\kappa\lVert\tilde H\rVert$ | 5.21e-11 Ha | 1.57e-11 Ha | 6.38e-11 Ha |
| tightest split inside the window | **5.11e-15** | 2.60e-2 | 1.89e-5 |
| boundary gap | 9.26e-2 | 1.05e-1 | 2.22e-1 |
| **A** $\lVert\tilde P_{\rm JAX}-\tilde P_{\rm Elk}\rVert_F$ | 5.0e-14 | 2.9e-14 | 9.0e-14 |
| **B** naive forward / reverse | **3.9e-1 / `NaN`** | 3.8e-13 / 2.3e-13 | 1.1e-10 / 7.0e-11 |
| **B** safe forward / reverse | 2.2e-11 / 2.2e-11 | 3.0e-13 / 2.3e-13 | 2.9e-13 / 2.9e-13 |
| **C** derivative in $k$ | `NaN` (see below) | 4.6e-14 | `NaN` |

**A** is the study's own Phase 1 forward criterion ($<10^{-8}$), met by six
orders, and it is a *projector* comparison because `evecfv` is arbitrary inside
the triplet — comparing columns would fail at $\Gamma$ for a gauge reason and
mean nothing. Elk's `evalfv` is reproduced to 4.5e-15 Ha at the same time.
**B** is the 0b-B experiment with Elk's $\tilde H$ in place of a synthetic one:
five random Hermitian directions, worst relative error against the
Daleckii–Krein closed form, with central FD carried alongside as the control
that separates "AD is broken" from "the test is broken" (it agrees to 7e-7, its
own truncation error). **C** is the whole pipeline differentiated in $k$, with
$d\tilde H/dk$ from a matrix-valued `jvp` fed to the same closed form —
legitimate because every step from `match` through the Cholesky is analytic in
$k$, so only the projector is not.

**The rule is needed, and the need is graded by the splitting.** The naive
route's error tracks $1/\delta\lambda$ across three fixtures spanning ten
decades of it: 3.8e-13 at a 2.6e-2 Ha gap, 1.1e-10 at 1.9e-5 Ha, and complete
failure at 5.1e-15 Ha. That is the mechanism Phase 0b identified — JAX's
`_eigh_jvp_rule` forms $A=v^\dagger\delta Hv$ without symmetrising, so the two
divergent terms fail to cancel by $\lVert A-A^\dagger\rVert$ — and it is
reproduced here on matrices nobody engineered. Which face the failure wears is
still the eigensolver's choice: forward mode returns 0.39 relative and reverse
mode returns `NaN`, on the same matrix in the same process.

The safe route's 2.2e-11 at $\Gamma$ is not machine precision and should not be
quoted as such. Inside a pair split at 5e-15 the individual eigenvectors are
ill-determined; the projector is not, but $K_{ij}$ *varies* across the split for
$j$ outside the window, so the ill-determined mixing does not cancel exactly.
2e-11 is that residual, and it is four orders below the naive route's best case
on the same matrix.

### Finding 1: a symmetry degeneracy is not a numerical one, and the tolerance cannot tell

$\epsilon\,\kappa(O)\,\lVert\tilde H\rVert$ is the eigenvalue backward error —
a statement about what the *arithmetic* can resolve. It is not a statement about
what the point group requires, and on real matrices the two are far apart in
both directions.

Elk's own $H$ and $O$ split the $\Gamma_{25'}$ triplet **unevenly**: two of the
three agree to 5.1e-15 Ha and the third sits 3.53e-9 Ha away — 68 times *above*
the 5.21e-11 Ha tolerance. h-BN's $\Gamma$ $\sigma$ pair is 1.89e-5 Ha apart,
$3\times10^5$ times above it. So symmetry-required degeneracies come out of a
converged all-electron assembly split by anywhere between $10^{-15}$ and
$10^{-5}$ Ha, with the linear-algebra resolution sitting in the middle of that
range.

The consequence is concrete and is pinned by a test. Cutting Si's triplet at
$n_{\rm occ}=3$ is refused; cutting the **same triplet** at $n_{\rm occ}=2$ is
accepted, and returns a perfectly finite derivative of a subspace that is not
physically separable at all. **The refusal is necessary but not sufficient**: a
caller who knows the point group must still window the whole degenerate group
together — the same mitigation `docs/design.md` §13 already applies to Berry
curvature. Tightening `epspot` from 1e-6 to 1e-9 leaves the 3.53e-9 splitting
identical to twelve digits, so it is not SCF convergence; its source (why one
pair is exact and the other is not) is left open.

A second consequence, for the adversarial sweep §1e was meant to be: any
$k\to K$ approach to a Dirac point will floor at this assembly noise, which is
one to four orders *above* the tolerance, so a sweep cannot reach the threshold
by closing a gap continuously. Cutting a multiplet reaches it directly, which is
what the test does.

### Finding 2: `soc_scale` cannot move the first-variational spectrum

§1e as written — sweep graphene's `soc_scale` 3000 → 3 and require a refusal
when the Dirac gap falls below the tolerance — **cannot be run on this
pipeline**, and not for want of effort. Spin-orbit coupling enters Elk only
through `eveqnsv`: `socfr` appears there and in nothing else in the
first-variational chain (`hmlfv`, `olpfv`, `hmlaa`, `hmlalo`, `hmllolo`,
`olpaa`, `olpalo`, `olplolo`, `eveqnfv`, `hmlrad`, `olprad` — zero occurrences,
grep-verified against `build/elk/src/`). Patch 0001 scales a term the matrices
built here never see.

So the first-variational Dirac point is exactly degenerate at *every*
`soc_scale`, and the sweep degenerates into "refuse always" rather than a
threshold crossing. The item is withdrawn as written. Its content — a required
refusal at a threshold, which is what makes "agreement or a clean refusal"
falsifiable — is delivered instead by cutting Si's $\Gamma_{25'}$ triplet, which
needs no extra ground state and no `soc_scale` at all. Reinstating the sweep
would mean building the second-variational step, i.e. a later phase.

### Finding 3: the $k$-derivative is unavailable at $\Gamma$, and it is a removable pole

Experiment **C** returns `NaN` at $\Gamma$ on both fixtures, while the *value*
there is exact (A and B are unaffected) and central FD of the same loss is
stable across three step sizes. The cause is in `elkjax.lapw.match`, at any
basis function whose $\mathbf G+\mathbf k$ lies on the $z$-axis, and there are
**two independent poles** there:

- $Y_{\ell m}(\hat v)$ has no derivative where the direction is undefined —
  already pinned in isolation by `spherical_harmonics`' own docstring and by
  `tests/test_jax_lapw.py`, but recorded there as a Phase 4 (stress) concern.
  It is a Phase 1 concern too.
- $\lvert\mathbf G+\mathbf k\rvert$ is $\sqrt{\cdot}$ at zero, whose tangent is
  `NaN` independently of the harmonics, and it feeds the Bessel argument and the
  $g^{i_o}$ factors.

The reach is much wider than "$\Gamma$". $\mathbf G=0$ is in every basis, so
every reciprocal-lattice point is affected; and in a slab cell $\mathbf
G=(0,0,\pm2\pi/c)$ is in the basis too, so the **entire $k_z=0$ plane** is —
which is all of a 2D material's physics, including the $K$ point this section
would otherwise have used. Combined with the multiplet result above, the safe-$K$
rule and the $k$-derivative are currently usable in disjoint places: multiplets
live at high-symmetry points, and that is where the tangent is `NaN`.

The underlying function is smooth: $j_\ell(gR)\,Y_{\ell m}(\hat g)$ is
$\propto g^\ell Y_{\ell m}(\hat g)$ near the origin, which is a *regular solid
harmonic* — a polynomial in the Cartesian components — times an even series in
$g^2$. So both poles are removable in the transcription. The fix is to run
`spherical_harmonics`' own recursion with $\cos\theta\to z$,
$\sin\theta\,e^{i\phi}\to x+iy$ and $\beta\to\beta r^2$ (which returns
$r^\ell Y_{\ell m}$ exactly), and to pair it with
$j_\ell^{(i_o)}(x)/x^{\ell-i_o}$ carrying its own small-$x$ series, so that
neither $\hat g$ nor $\lvert g\rvert$ is ever formed. That is the next step and
is not done here; the current behaviour is **pinned by a test** that asserts the
value is finite and the tangent is not, so the fix will announce itself by
flipping it.

### What is NOT settled by this

- **Smeared occupations.** Everything here is a hard integer window. The
  metallic case is `smeared_projector`, whose tolerance branch is genuinely
  live (a hard window's kernel is zero on both branches for a same-side pair),
  and Phase 0b's plateau sweep for it is still synthetic.
- **Second derivatives.** `hard_window_projector`'s JVP calls `eigh` itself, so
  a second derivative falls back on JAX's default rule. Phase 0a′'s
  `sign_projector` is the eigensolver-free route and has not been run on a real
  LAPW matrix — where its Newton–Schulz step count is set by the *deepest*
  state in the window, not by the valence bandwidth.
- **Nothing here is `jit`-ed or batched over $k$.** The driver's peak RSS is
  3.6 GB at $n_{\rm mat}=906$, from reverse-mode tapes through the full
  assembly at three k-directions. The production shape is not reachable this
  way and was never meant to be (Phase 0e).
