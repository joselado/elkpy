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
| **1b** the radial integrals from `genapwfr`/`genlofr`/`hmlrad`/`olprad` | **done, and without Phase 2** (§1k): patch 0015 exports the muffin-tin potential, so the chain $v_s^{\rm MT}\to$ radial functions $\to$ radial integrals $\to H,O\to\varepsilon_j$ closes; every stage machine-precision against Elk |
| **1c** the interstitial blocks from `gencfun`/`genvsig` | not started; `vsig` needs the interstitial Kohn-Sham potential, i.e. Phase 2 |
| **1a′** the assembly as a differentiable function of $k$ | **done — and it found that $d\varepsilon/dk \neq \langle p\rangle$ in a finite LAPW basis** |
| **1d** gradient vs finite differences on displaced h-BN | not started |
| **1e** the adversarial `soc_scale` sweep, and the required refusal | **withdrawn as written — `soc_scale` cannot move the first-variational spectrum at all** (§1f); the refusal itself is done, by cutting a real multiplet |
| **1f** the Cholesky-reduced eigensolve wired to the safe-$K$ projector rule, on real matrices | **done — the rule is needed and it works**: 6.7e-2 (forward) / `NaN` (reverse) naive against 4.1e-12 safe, at a multiplet Elk's own matrices supply by symmetry |
| **1g** the two removable poles in `match` | **done — the $k$-derivative now works at $\Gamma$**, and at every $k_z=0$ point of a slab cell, which is where every multiplet is |
| **1h** the negative test at an exact degeneracy (required to fail) | **done — on graphene at K**, and it corrected the study's own fixture suggestion: a time-reversal-invariant point cannot show the disagreement, however degenerate |
| **1i** smeared occupations and the self-consistent Fermi level | **done** — the closed-form kernel inside the JVP makes the tolerance INERT for smeared occupations; whether the near-degenerate branch fires is set by the assembly's roundoff, not by the physics |
| **1j** second derivatives at a real multiplet | **done** — only `sign_projector` survives; both `eigh`-based routes return `NaN`, and the control at a generic $k$ says the failure is the multiplet, not the order |
| **1k** the spectrum as a differentiable function of the muffin-tin potential | **done — and the two channels are exactly complementary**: the spherical part of $v_s$ enters ONLY through the basis (frozen-basis derivative exactly zero) and the non-spherical part ONLY through the integrals (basis response 7e-16) |

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
| tightest split inside the window | **4.58e-15** | 2.60e-2 | 1.89e-5 |
| boundary gap | 9.26e-2 | 1.05e-1 | 2.22e-1 |
| **A** $\lVert\tilde P_{\rm JAX}-\tilde P_{\rm Elk}\rVert_F$ | 5.8e-14 | 2.9e-14 | 8.3e-14 |
| **B** naive forward / reverse | **6.7e-2 / `NaN`** | 7.9e-14 / 7.5e-14 | 2.0e-10 / 1.3e-11 |
| **B** safe forward / reverse | 4.8e-12 / 4.1e-12 | 8.0e-14 / 7.8e-14 | 1.1e-13 / 1.1e-13 |
| **C** naive derivative in $k$ | **`NaN`** | 3.2e-13 | 2.8e-12 |
| **C** safe derivative in $k$ | 1.7e-12 | 2.6e-13 | 2.2e-12 |

(The numbers are from the current code, i.e. **after** §1g removed `match`'s poles.
Before it, the whole **C** row read `NaN` at both $\Gamma$ columns — that is how the
poles were found, and §1g has the before/after. §1g changes rounding only, so the other
rows moved in their last digits and nowhere else.)

**A** is the study's own Phase 1 forward criterion ($<10^{-8}$), met by six
orders, and it is a *projector* comparison because `evecfv` is arbitrary inside
the triplet — comparing columns would fail at $\Gamma$ for a gauge reason and
mean nothing. Elk's `evalfv` is reproduced to 5.2e-15 Ha at the same time.
**B** is the 0b-B experiment with Elk's $\tilde H$ in place of a synthetic one:
five random Hermitian directions, worst relative error against the
Daleckii–Krein closed form, with central FD carried alongside as the control
that separates "AD is broken" from "the test is broken" (it agrees to 3.3e-7,
its own truncation error). **C** is the whole pipeline differentiated in $k$, with
$d\tilde H/dk$ from a matrix-valued `jvp` fed to the same closed form —
legitimate because every step from `match` through the Cholesky is analytic in
$k$, so only the projector is not.

**The rule is needed, and the need is graded by the splitting.** The naive
route's error tracks $1/\delta\lambda$ across three fixtures spanning ten
decades of it: 7.9e-14 at a 2.6e-2 Ha gap, 2.0e-10 at 1.9e-5 Ha, and complete
failure at 4.6e-15 Ha. That is the mechanism Phase 0b identified — JAX's
`_eigh_jvp_rule` forms $A=v^\dagger\delta Hv$ without symmetrising, so the two
divergent terms fail to cancel by $\lVert A-A^\dagger\rVert$ — and it is
reproduced here on matrices nobody engineered. Which face the failure wears is
still the eigensolver's choice: forward mode returns 6.7e-2 relative and reverse
mode returns `NaN`, on the same matrix in the same process. Note too that the naive
route is fine at h-BN's $\Gamma$ pair and along $k$ there: a symmetry pair split at
1.9e-5 Ha is not a numerically degenerate one, which is the same distinction Finding 1
draws from the other side.

The safe route's 4e-12 at $\Gamma$ is not machine precision and should not be
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

Experiment **C** returned `NaN` at $\Gamma$ on both fixtures when it was first run,
while the *value* there was exact (A and B were unaffected) and central FD of the same
loss was stable across three step sizes. The cause is in `elkjax.lapw.match`, at any
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
$g^2$. So both poles are removable in the transcription. **§1g does that**, and
the `NaN` row of the table above is what it removes.

### What is NOT settled by this

- **Smeared occupations.** Everything here is a hard integer window. The
  metallic case is `smeared_projector`, whose tolerance branch is genuinely
  live (a hard window's kernel is zero on both branches for a same-side pair),
  and Phase 0b's plateau sweep for it is still synthetic. This is the next
  reachable item.
- **The negative test at a degeneracy** *(closed by §1h)*, which needed §1g
  first: the fixtures for it are all high-symmetry points.
- **Second derivatives.** `hard_window_projector`'s JVP calls `eigh` itself, so
  a second derivative falls back on JAX's default rule. Phase 0a′'s
  `sign_projector` is the eigensolver-free route and has not been run on a real
  LAPW matrix — where its Newton–Schulz step count is set by the *deepest*
  state in the window, not by the valence bandwidth.
- **Nothing here is `jit`-ed or batched over $k$.** The driver's peak RSS is
  3.6 GB at $n_{\rm mat}=906$, from reverse-mode tapes through the full
  assembly at three k-directions. The production shape is not reachable this
  way and was never meant to be (Phase 0e).

---

## 1g. Removing the two poles in `match`

### What was at stake

§1f's third finding, and it is worse than an inconvenience. The $k$-tangent of the
assembly was `NaN` at $\Gamma$ — and at every reciprocal-lattice point, and across the
whole $k_z=0$ plane of a slab cell — while the *value* there was exact. Multiplets live
at high-symmetry points, so the safe-$K$ projector rule §1f had just closed and the
$k$-derivative §1a′ had just built were usable in **disjoint** places. Neither was much
use to the other.

### The two poles, and why fixing one is not enough

Elk writes the matching coefficient as a harmonic of a *direction* times a Bessel
function of a *length*,

$$b_i \;\propto\; \overline{Y_{\ell m}(\widehat{\bf G+k})}\;
\lvert{\bf G+k}\rvert^{\,i-1} j_\ell^{(i-1)}\!\big(\lvert{\bf G+k}\rvert R_\alpha\big),$$

and **both factors are singular where the vector lies on the $z$-axis**:
$Y_{\ell m}(\hat v)$ because $\hat v$ is undefined there (for $m\neq0$ the azimuth has no
derivative, and $\sin\theta=\sqrt{1-\cos^2\theta}$ has an infinite one at the pole), and
$\lvert{\bf G+k}\rvert$ because $\sqrt{\cdot}$ has no derivative at zero. They are
independent, and the second is **invisible until the first is fixed** — which is exactly
the shape of bug this project keeps meeting, so it is recorded rather than quietly
handled.

### The regrouping

The *product* is not singular. Near the origin $j_\ell(x)\propto x^\ell$, so

$$\overline{Y_{\ell m}(\hat g)}\;g^{i_o}j_\ell^{(i_o)}(gR)
= \overline{S_{\ell m}({\bf G+k})}\;R^{\,\ell-i_o}\,P_{\ell,i_o}(x),
\qquad x^2 = R^2\,({\bf G+k})\!\cdot\!({\bf G+k}),$$

with two analytic factors:

- $S_{\ell m}=r^\ell Y_{\ell m}$, the **regular solid harmonic** — a polynomial in
  $v_x,v_y,v_z$. `solid_harmonics` is `spherical_harmonics`' own recursion with three
  substitutions, $\cos\theta\to v_z$, $\sin\theta\,e^{i\phi}\to v_x+iv_y$ and
  $\beta\to\beta r^2$, and the same $S_{\ell,-m}=(-1)^m\overline{S_{\ell m}}$.
- $P_{\ell,i_o}(x)=j_\ell^{(i_o)}(x)\,x^{i_o-\ell}$, which is **even** in $x$ and
  therefore a function of $x^2$ alone — and $x^2$ is a polynomial in the Cartesian
  components. `spherical_bessel_scaled` evaluates it from its own series
  $\sum_n c_n(\ell)\,(\ell+2n)_{i_o}\,x^{2n}$ below $x=0.1$ and from
  `spherical_bessel_derivative` divided by $x^{\ell-i_o}$ above it.

So `match` forms **neither $\hat g$ nor $\lvert g\rvert$**, which is why `gkc` is no
longer one of its arguments: leaving it in the signature would let a call site
reintroduce the second pole while the first stayed fixed. `spherical_harmonics` and
`spherical_bessel` are untouched and remain what item 0c checks; `solid_harmonics` is a
separate function rather than a refactor of the first, deliberately, so that the
element-wise `apwalm` comparison keeps testing the same code it always did.

The identity is exact, so this changes rounding and nothing else.

### Result

| check | before | after |
|---|---|---|
| `solid_harmonics` $/\,r^\ell$ vs `spherical_harmonics`, off-axis | — | 1.3e-15 |
| `spherical_bessel_scaled` vs the $\ell=0$ closed form, orders 0/1/2 | — | 2.7e-15 |
| the same across its own branch cut, orders 0/1/2 | — | 2e-15 / 2e-13 / 1.5e-11 |
| `jvp(match)` in $k$ at a basis with ${\bf G+k}=0$ | `NaN` | finite, and central FD to $<10^{-6}$ |
| element-wise vs Elk's `apwalm`, generic $k$ | machine precision | 3.7e-15 |
| element-wise vs Elk's `apwalm`, $\Gamma$ | *never checked* | 5.6e-16 |
| the `dmatch` identity, forward and reverse | 7e-16 | unchanged |
| all six assembly blocks vs Elk, three fixtures | machine precision | unchanged *(and cannot see this: they take `apwalm` from the export)* |
| **C: $dP/dk$ at $\Gamma$ through the multiplet** | **`NaN`** | **1.4e-14, 3.8e-15, 1.7e-12** |

The rows that actually exercise the new `match` end to end are **A** (5.8e-14, through
`eigenproblem_at`, which rebuilds `apwalm`) and **C**; the six-block comparison is listed
for completeness but takes `apwalm` from the export and could not have detected a change
here either way.

The branch-cut row is the one worth reading twice. The comparison had to be made where
*both* routes are accurate, because below $x\sim10^{-3}$ the recurrence-and-divide route
is the wrong one — measured, it is off by 100% at $\ell=6$, $i_o=2$, $x=10^{-6}$, where
it forms $j_6''\sim10^{-27}$ and divides by $x^4$. That is the entire reason the series
branch exists, and it means the new route is *more* accurate near the origin than the old
one was, not merely defined there. The growing error across orders (2e-15 → 1.5e-11) is
`jax.jacfwd`-through-Miller's, not the series'.

Both `apwalm` rows are at `apword=1` on the Si fixture; the export test's own
two-parameter fixture (APW orders 1 and 2) still passes its 1e-11 bound at both.
The $\Gamma$ row matters more than it looks: Elk handles ${\bf G+k}=0$ in its
own way (`sbessel` returns $j_\ell(0)=\delta_{\ell0}$, `genylmv` picks $+z$ at the
origin), so reproducing its array *there* is a real check on the solid-harmonic route
rather than a restatement of the generic-$k$ one. Phase 0's first carried-forward
finding is that a green gradient test does not validate a transcription, and this is the
forward half standing beside the finite tangent.

### What this does and does not unlock

It does not make the naive eigensolve work at $\Gamma$: with the poles gone, the *naive*
projector's $k$-derivative through the $\Gamma_{25'}$ triplet is still `NaN` in all three
directions tried, which is asserted by its own test so that "we fixed the pole" cannot be
mistaken for "the $k$-derivative is fine now". The two fixes are independent and both are
needed — §1f's rule for the eigensolve, §1g's regrouping for the assembly — and only
together do they give a projector derivative at a high-symmetry point.

Still untouched: `spherical_harmonics` is *still* singular on the $z$-axis, which is
correct for a `genylmv` transcription and is still pinned by its own test. Anything else
that consumes it directly — Phase 4's stress moves ${\bf G+p}$ itself — has the same
problem and the same fix available.

---

## 1h. The negative test, and the fixture the study names for it

### What was at stake

The study's Phase 1 list closes with a criterion that is *required to fail*: at a
$k$-point where two occupied bands are exactly degenerate, AD, central finite
differences and one-sided finite differences of an **individual** eigenvalue must
**disagree**, and the multiplet trace must agree across all three. Without it, §8(b)'s
degeneracy caveat is a comment rather than a signal — and this file's own
`first_variational_eigenvalues` carries a warning about exactly that while
`occupied_projector` does not.

It became reachable only after §1g: the natural fixtures are all high-symmetry points.

### The fixture the study names does not work, and neither does the obvious one

The study says "h-BN at $\Gamma$, or graphene at K with `soc_scale=0`". **The first is
wrong**, and so is bulk Si at $\Gamma$, for the same reason. At any time-reversal-invariant
momentum every branch is *even* in $\mathbf k$, so the sorted branches do **not** exchange
places between $+t$ and $-t$; AD and central FD then agree with each other and with the
true derivative, which is **zero**. Measured on Si's $\Gamma_{25'}$ triplet, along a
general direction: AD returns $10^{-17}$ to $10^{-14}$ and central FD $10^{-12}$ to
$10^{-11}$, for every band of the multiplet. The multiplet is every bit as degenerate as
graphene's — what is missing is a *linear* splitting.

One-sided FD is the only one that moves there, $-4.7\times10^{-4}$ at step $10^{-4}$, and
it moves for an entirely ordinary reason: $(\varepsilon(t)-\varepsilon(0))/t\approx
\tfrac12\varepsilon''t$ at a critical point. It shrinks by a factor of 10 per decade of
step ($-4.68\times10^{-4}$, $-5.00\times10^{-3}$, $-4.98\times10^{-2}$ at $10^{-4}$,
$10^{-3}$, $10^{-2}$), which is what separates truncation error from a degeneracy signal
— and asserting that is why this is a test rather than a comment. A version of this test
built on Si would otherwise have *passed*, on a disagreement that has nothing to do with
the multiplet.

### Graphene at K, where the branches cross linearly

2 atoms, 20 Bohr of vacuum, `rgkmax=6`, $6\times6\times1$ — $n_{\rm mat}=610$ and under
two minutes including the ground state, so it runs by default. Writing the two branches
as $\varepsilon_0\mp v\lvert t\rvert$, each route is wrong in its own way rather than
noisy: central FD of the **sorted** spectrum returns the branch average $0$, because the
branches exchange; one-sided FD returns the extreme branch $-v$; and AD returns the
diagonal of $v^\dagger\,\delta H\,v$ in whatever basis the eigensolver picked inside the
multiplet, which is neither.

| along $\delta k\propto(0.3,-0.5,0)$ | AD | central FD | one-sided FD |
|---|---|---|---|
| band 4 (lower Dirac branch) | $+0.18377$ | $+0.00086$ | $-0.37610$ |
| band 5 (upper Dirac branch) | $-0.18377$ | $-0.00087$ | $+0.37608$ |
| **their sum** | $-1.93\times10^{-7}$ | $-2.01\times10^{-7}$ | $-1.67\times10^{-5}$ |

(Band numbering is 1-based throughout this section; the pair is split by
$3.4\times10^{-7}$ Ha.) **The $0.184$ means nothing on its own** — it is the diagonal of
$v^\dagger\,\delta H\,v$ in whichever basis the eigensolver happened to pick inside the
multiplet, so it can be anywhere in $[-v,+v]$ and is not reproducible across
eigensolvers, the same caution 0b-C already attaches to naive-AD values. What is the
result is that it agrees with *neither* of the other two. The three individual
values differ by more than 10% of the scale from each other; the trace agrees to
$5\times10^{-7}$ of it between AD and central FD, and to $4\times10^{-5}$ one-sided, whose
own $O(\text{step})$ truncation is the residue. The fixture carries its own control: the
$\sigma$ doublet at K (bands 1-2, split $1.3\times10^{-9}$ Ha) is just as degenerate but
does not split linearly along this direction, and there AD and central FD **agree**
($-5.37\times10^{-6}$ vs $-5.28\times10^{-6}$).

### What this settles

`first_variational_eigenvalues` is safe for a **trace** and unsafe for an individual band
inside a multiplet, and that is now asserted rather than documented. It is also the sharpest
available statement of why `occupied_projector` exists: on the same fixture, in the same
code path, the projector's derivative is well defined where the individual eigenvalue's is
not. And the fixture correction is worth carrying: *degeneracy is not enough* — the
disagreement needs branches that cross, so a TRI point cannot show it however degenerate it
is.

## 1i. Smeared occupations, and the self-consistent Fermi level

`docs/continue_here.md` §3 ranked this next, and gave the reason §1f leaves it open:
**for a hard integer window the near-degenerate branch of the divided-difference kernel
is inert.** Both branches of a same-side pair are identically zero — $f_i-f_j=0$
exactly and $f'=0$ — so §1f's plateau in `tol` measures nothing about the branch, only
that switching it on does no harm. With Fermi-Dirac occupations (Elk's own `stype=3`,
`stheta_fd.f90`) the branch value is $f'(\bar\lambda)$, which at a half-filled level is
$-1/4w$ — large, not zero — so the threshold is load-bearing for the first time. The
same step makes the fixed-electron-number Fermi level reachable, whose rule study §8(b)
gives in closed form and which `continue_here.md` lists as untested.

Driver: `PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase1_smearing`. Tests:
`tests/test_calculation_lapw_smearing.py` (real matrices) and the Phase 1i block of
`tests/test_jax_projector.py` (synthetic, where the splitting can be dialled).

### The oracle: a difference quotient with no subtraction in it

Neither candidate is a reference for the other, so both were checked against a third
thing that is exact. For the logistic function the difference quotient has a closed
form containing no cancellation at all:

$$
K_{ij}=\frac{f_i-f_j}{\lambda_i-\lambda_j}
      =-\frac{1}{4w}\,\frac{\sinh(z_{ij})/z_{ij}}{\cosh u_i\,\cosh u_j},
\qquad u_i=\frac{\lambda_i-\mu}{2w},\quad z_{ij}=u_i-u_j,
$$

from $f_a-f_b=-2e^{(a+b)/2w}\sinh(z)f_af_b$ together with $f_ie^{u_i}=1/(2\cosh u_i)$.
It reduces to $f'=-f(1-f)/w$ on the diagonal exactly (checked to $10^{-14}$), and is the
*same analytic function* as the quotient — so it arbitrates between the two branches
instead of being a third opinion. `elkjax.reference.fermi_divided_difference_kernel`,
which uses this form for $|z|\le1$ and the plain quotient above it, where the two
occupations differ by an $O(1)$ fraction and $\cosh$ would overflow.

Note also that FD is a **legitimate** second check here, unlike §1h: $P=f(H)$ is a smooth
matrix function when $f$ is smooth, so the derivative exists at every degeneracy and
there is no branch exchange for a central difference to average over.

### Two fixtures, chosen for opposite reasons

**Graphene at $K$ is the physical one.** The two $\pi$ bands are degenerate there *and*
the Fermi level sits on them, so $f=\tfrac12$ exactly and $f'$ is at its maximum. That
is not an assumption: requiring $\sum_if_i=4$ at $K$ **alone** gives
$\mu=-0.0471925804$ Ha against Elk's own zone-integrated `EFERMI.OUT` of
$-0.0471925770$ Ha — agreement to $3.4\times10^{-9}$ Ha, because particle-hole symmetry
at the Dirac point puts the local answer on the global one. At Elk's default `swidth`
the occupancy is $4.0000017$ against an electron count of exactly 4.

**Silicon at $\Gamma$ is the numerical one**, the same matrix §1f used. Its
$\Gamma_{25'}$ triplet carries a pair split by $1.08\times10^{-15}$ Ha, which is
$2\times10^{-5}$ of that run's tolerance — the branch fires. (§1f measured
$5.1\times10^{-15}$ and $3.53\times10^{-9}$ on the same triplet; this run gets
$1.1\times10^{-15}$ and $3.5\times10^{-9}$. That the tighter of the two is not
reproducible to a factor of five *is* the point below.) The two fixtures together
say something worth stating on its own:

> A real LAPW multiplet's splitting is a property of the **assembly's roundoff**, not of
> the symmetry that requires it, and it spans at least eight decades: $1.1\times10^{-15}$
> and $3.5\times10^{-9}$ Ha inside Si's $\Gamma_{25'}$ triplet, $3.4\times10^{-7}$ Ha at
> graphene's Dirac point. Whether the tolerance branch fires is therefore not predictable from the physics
> and must be measured per run.

The kernel entries themselves, at Elk's default `swidth`, make the same point from the
other side. At graphene's Dirac pair the quotient and the branch **agree**
($-2.49999999\times10^{2}$ vs $-2.49999998\times10^{2}$, a relative difference of
$9.8\times10^{-13}$): the pair is degenerate *on the scale of $w$*, so the branch's value
is right even though it is not needed. At the two neighbouring pairs, split by $0.39$ and
$0.38$ Ha, the exact kernel is $-1.28$ and $-1.31$ while $f'$ is $-125$ — a hundredfold
error. That is the argument against a generous tolerance stated numerically: the branch is
correct only where the pair is unresolvable, and `tol` is what has to know the difference.

### What fails, and where

Three routes, all avoiding or not avoiding a different thing: **naive** is JAX's own
`eigh` rule; **quotient** is the safe-$K$ rule with the branch switched off (`tol=0`),
which avoids the eigenvector derivative but still forms the numerator's difference; and
**safe** is the rule at `tol` = `projector_tolerance`. Worst relative error over five
random Hermitian directions, against the exact kernel:

| fixture | $w$ (Ha) | naive fwd | naive rev | quotient | safe | central FD |
|---|---|---|---|---|---|---|
| graphene $K$ (split $3.4\times10^{-7}$) | $10^{-3}$ | 3.9e-9 | 6.1e-10 | 1.0e-13 | 1.0e-13 | 6.9e-7 |
| | $10^{-2}$ | 3.9e-8 | 6.1e-9 | 7.7e-14 | 7.7e-14 | 6.9e-7 |
| | $10^{-1}$ | 2.8e-7 | 5.6e-8 | 2.0e-10 | 2.0e-10 | 8.1e-7 |
| Si $\Gamma$ (split $1.1\times10^{-15}$) | $10^{-3}$ | **NaN** | **NaN** | 3.7e-3 | 2.0e-9 | 4.6e-7 |
| | $10^{-2}$ | **NaN** | **NaN** | 1.6e-5 | 4.4e-11 | 9.3e-8 |
| | $10^{-1}$ | **NaN** | **NaN** | 3.4e-4 | 8.9e-10 | 9.9e-7 |

**Three separate results are in that table.**

1. **The two failure modes are distinct, and each fixture shows only one.** On graphene
   the branch never fires and the quotient is exact to $10^{-13}$; what fails is the
   eigenvector rule. On Si the eigenvector rule fails *completely* — `NaN`, because the
   pair is split at the level where JAX's own $1/\delta\lambda$ is a ratio of rounding
   errors — while the quotient survives but is wrong by up to $3.7\times10^{-3}$. Only
   the safe rule is right on both. A single fixture would have credited the wrong
   mechanism.
2. **The naive rule's relative error grows LINEARLY in the smearing width** — 3.9e-9,
   3.9e-8, 2.8e-7 over two decades of $w$. Its absolute error is
   $\sim\epsilon\lVert A-A^\dagger\rVert/\delta\lambda$, set by the splitting; the
   quantity it is measured against is of order $f'=1/4w$, which *shrinks* as $w$ grows.
   Broader smearing is not gentler. The same holds for the quotient's own cancellation,
   $\sim\epsilon\,w/\delta\lambda$, which is why `tol` cannot be a fixed number: what it
   has to beat depends on the width as well as on the splitting. Pinned on synthetic
   spectra at a $10^{-15}$ splitting, where the quotient's error goes 8.9e-5, 8.0e-4,
   2.1e-2 across the same three widths.
3. **At Elk's default `swidth` the quotient on Si's pair is not merely inaccurate, it is
   zero.** $f_i$ and $f_j$ are bitwise equal at $10^{-15}$ splitting and $w=10^{-3}$, so
   the numerator is exactly $0$ where the true kernel is $-3.21\times10^{-2}$: a 100%
   error **in that entry**, and one that no plausibility check would catch, since $0$ is
   a perfectly ordinary kernel entry. The derivative it feeds is off by $3.7\times10^{-3}$
   rather than by 100%, because that entry is small beside the gap-straddling pair's
   $-10.8$ — which is exactly why the failure is easy to miss.

The `NaN` deserves a note. §1f's hard-window naive route returned `NaN` at Si's
$\Gamma$ too, but for a different reason — there the window boundary cut nothing and the
`NaN` came from the enclosed multiplet's $0/0$. Here the multiplet is at the *same*
place but the occupations are smooth, so the derivative genuinely exists and a correct
rule returns it; the `NaN` is purely JAX's own.

There is one more thing in the table that is not a success. **The safe rule on Si is
$2\times10^{-9}$, not $10^{-13}$**, and the reason is the *other* pair: $3.5\times10^{-9}$
Ha, which is $67\times$ the tolerance, so it takes the quotient branch and contributes its
own cancellation. The tolerance is a cliff and a pair just above it gets the
cancellation-prone route — the same "necessary but not sufficient" shape §1f found for
the refusal. Anything wanting better than $10^{-9}$ on a matrix like this needs the
stable form of the kernel in the JVP itself, not a threshold.

### Removing the tolerance rather than tuning it

The $2\times10^{-9}$ above is not a threshold that wants adjusting. The stable form of
the kernel quoted at the top of this section is **analytically exact at every
splitting**, so it can go inside the JVP and there is then nothing left for `tol` to
select. `projector.fermi_kernel` does that, and `smeared_projector` now uses it; the
near/far branch that remains is about `cosh` overflowing (a real LAPW spectrum reaches
$|u|\sim10^6$ at Elk's default width), not about resolving a degeneracy, and both
`jnp.where` arguments are made finite before the select so the discarded branch cannot
put a `NaN` into the gradient.

| | $w=10^{-3}$ | $10^{-2}$ | $10^{-1}$ |
|---|---|---|---|
| Si $\Gamma$, safe rule with the `tol` branch | 2.0e-9 | 4.4e-11 | 8.9e-10 |
| Si $\Gamma$, safe rule with the closed form | **7.4e-12** | **3.1e-12** | **4.8e-14** |
| graphene $K$, either | 1.0e-13 | 1.0e-13 | 2.0e-10 |
| Si $\Gamma$, fixed-$N$ (experiment E) | — | 7.6e-10 → **3.0e-13** | 2.0e-9 → **6.0e-13** |

`tol` is now **inert** for smeared occupations, and that is asserted as *equality*
across fourteen decades of it including zero — a stronger statement than study §8(b)'s
"flat over two decades", and a different one: a threshold removed rather than tuned to
be harmless. It remains load-bearing for `hard_window_projector`, whose straddling-pair
`NaN` refusal is a statement about a derivative that does not exist.

**And what is left is not the kernel.** The residual shrinks as the smearing *widens*,
which is backwards for every other effect in this section, so it was chased down:
`reference.dprojector_fermi` builds its own eigendecomposition with LAPACK while
`smeared_projector` uses XLA's, and on this matrix the two disagree by
$2.1\times10^{-14}$ Ha. The kernel's sensitivity to an eigenvalue is $f''\sim1/w^2$
against a value $f'\sim1/w$, so that disagreement enters the relative error divided by
$w$. Re-running the *same closed form* on XLA's own decomposition confirms it exactly:

| | AD vs LAPACK-built reference | AD vs XLA-built | the two references |
|---|---|---|---|
| $w=10^{-3}$ | 1.8e-12 | **7.9e-14** | 1.9e-12 |
| $w=10^{-2}$ | 3.1e-12 | **5.5e-15** | 3.1e-12 |
| $w=10^{-1}$ | 2.0e-14 | **2.6e-15** | 2.2e-14 |

The third column is the first, to two digits. So the rule is accurate to $\sim10^{-14}$
and what the driver reports is a *comparison* artifact — worth knowing before anyone
tunes anything against a $10^{-12}$ target. `elkjax.phase1_smearing.eigensolver_floor`
is that measurement, kept in the driver.

One consequence for the test suite, stated because it weakens an oracle:
`reference.fermi_divided_difference_kernel` and `projector.fermi_kernel` now share a
*formula*. They remain independent **implementations** (NumPy and XLA, and the table
above is exactly what that catches), but the arbiter where it matters is central finite
differences — legitimate here, since $P=f(H)$ is smooth — and
`direct_quotient_projector`, which keeps the literal quotient and is therefore still
formula-independent wherever it is accurate.

### The $k$-derivative, and the self-consistent Fermi level

The whole pipeline differentiated in $k$ with smeared occupations agrees with the exact
kernel fed by a matrix-valued `jvp` of $d\tilde H/dk$: $6\times10^{-16}$ to
$3\times10^{-10}$ on graphene, $4\times10^{-15}$ to $4\times10^{-12}$ on Si, against
`NaN` for the naive route on Si and $10^{-10}$ to $10^{-7}$ on graphene, with central FD as the
control at $10^{-7}$–$10^{-8}$ (its own step-truncation floor: the step must satisfy
$v_Fh\ll w$). Unlike §1f's hard window, where an enclosed multiplet contributes exactly
nothing, here the states at the Fermi level carry the **largest** kernel entries in the
matrix, so the $k$-tangent passes straight through them.

`fermi_level` is a `custom_jvp` whose primal is a bisection — never differentiated,
which is the Phase 0a lesson about unrolled solvers in miniature — and whose tangent is
study §8(b)'s closed form
$d\mu=\sum_jf'_jA_{jj}\big/\sum_jf'_j$, $A=V^\dagger\,\delta H\,V$. Two things about it:

* **It is gauge-invariant at a multiplet even though it uses $A_{jj}$.** Inside a
  degenerate group $f'$ is constant, so the sum over that group is $f'\,\mathrm{Tr}\,A$
  — invariant under the arbitrary unitary `eigh` picks. Tested by rotating the
  degenerate block explicitly and requiring $d\mu$ unchanged while the individual
  $A_{jj}$ move by $O(1)$.
* **Its denominator is a physical singularity.** $\sum_jf'_j\to0$ for a gapped system at
  small width: no state responds, the constraint stops determining $\mu$, and $d\mu$ is
  genuinely $0/0$. `check_fermi_level_determined` refuses there, and does so on Si at
  $w=10^{-3}$ (measured $\sum|f'|=4.4\times10^{-13}$) while accepting graphene
  ($5.0\times10^{2}$) — a real distinction between a metal and an insulator falling out
  of the derivative's existence, not a numerical guard.

`fixed_number_projector` is then nothing but the composition, and the chain rule supplies
$dP|_N=dP|_\mu-V\,\mathrm{diag}(f')\,V^\dagger d\mu$. **That term is not a correction.**
Against a reference that re-solves $\mu$ at every displaced matrix:

| fixture | $w$ (Ha) | AD fwd | AD rev | central FD | fixed-$\mu$, i.e. the term dropped |
|---|---|---|---|---|---|
| graphene $K$ | $10^{-3}$ | 1.8e-13 | 7.1e-14 | 2.5e-7 | **69** |
| graphene $K$ | $10^{-2}$ | 1.1e-12 | 1.1e-12 | 4.2e-7 | **78** |
| Si $\Gamma$ | $10^{-2}$ | 7.6e-10 | 7.6e-10 | 1.8e-7 | **10** |
| Si $\Gamma$ | $10^{-1}$ | 2.0e-9 | 2.0e-9 | 6.9e-7 | **132** |

At a level pinned to the Fermi energy, holding $\mu$ fixed while $H$ moves is wrong by
one to two orders of magnitude, not by a percent. And $d\mu$ itself agrees with a
finite difference of a re-solved $\mu$ to $5\times10^{-10}$ (Si, $w=10^{-1}$).

**The test is along Hermitian directions and deliberately not along $k$.** At $K$ the
$\pi$ pair's trace is stationary by symmetry, so $d\mu/dk=0$ and a $k$-direction test
would pass with the correction identically zero — Phase 0a's "a perturbation that
respects the symmetry protecting the degeneracy hides the bug" trap, in a new place.

### What this settles, and what it does not

Settled: the safe-$K$ rule works with smeared occupations on Elk's own matrices, in both
modes and at the end of the $k$-pipeline; the near-degenerate branch is load-bearing for
the first time and its value is right; and study §8(b)'s Fermi-level rule is correct,
gauge-invariant at a multiplet, and dominant rather than corrective at a half-filled
level. `continue_here.md` §3's framing — that "a metal is the only place the threshold is
actually load-bearing" — is **half right**: smearing is what makes the branch's *value*
nonzero, but whether it *fires* is set by the assembly's roundoff, and graphene, the
metal, does not fire while gapped silicon does.

Not settled: the $k$-point weights $w_j$ in §8(b)'s formula are untested, since a
single-$k$ constraint makes them cancel — a zone-summed Fermi level is Phase 2 work. The
kernel's own JVP still forms the quotient rather than the stable closed form, which is
what puts the floor at $2\times10^{-9}$ on Si. And second derivatives are untouched here:
`smeared_projector`'s JVP calls `eigh`, so a second derivative falls back on JAX's rule
exactly as §0a′ describes, and `sign_projector` covers hard windows only.


## 1j. Second derivatives, and the route that survives them

`docs/continue_here.md` §3 item 5, and the last Phase 1 item reachable without Phase 2.
§1f's safe-$K$ rule is **first order by construction**: its own JVP body calls
`jnp.linalg.eigh`, so a second derivative falls back on JAX's default eigenvector rule
and the hazard the rule exists to remove comes straight back.
`projector.sign_projector` is the eigensolver-free route — the projector as a matrix
sign function,

$$
P=\tfrac12\big(\mathbb 1-\mathrm{sign}(\tilde H-\mu)\big),
\qquad X\leftarrow\tfrac12(3X-X^3),\quad X_0=(\tilde H-\mu)/\lVert\tilde H-\mu\rVert_2,
$$

with no eigendecomposition in it, hence no gauge and no $1/(\lambda_i-\lambda_j)$
anywhere, which JAX differentiates to any order natively. Phase 0a′ showed that on a
toy; this is its first LAPW matrix.

Driver: `python3 -m elkjax.phase1_secondorder`. Tests:
`tests/test_calculation_lapw_secondorder.py`.

### The cost, which a toy could not have supplied

Newton-Schulz needs about $\log(\lVert\tilde H-\mu\rVert_2/\Delta)/\log(3/2)$ steps, and
the numerator is set by **the top of the basis, not by the valence bandwidth**. Measured
on bulk Si at $\Gamma$, `rgkmax=7`: the reduced spectrum runs $[-0.243,\,17.9]$ Ha
against a 0.35 Ha valence manifold and a 0.0926 Ha boundary gap, so the ratio is
$1.9\times10^{2}$ and the predicted count 12.9. Observed: **10 steps is not converged**
(error 1.1, i.e. not a projector at all), 20 steps reaches $3.4\times10^{-14}$ against
the safe-$K$ rule and $4.2\times10^{-14}$ against Elk's own occupied subspace, and 30, 40
and 60 do not improve on it. 42 ms at $n=177$ on this machine.

Note this is the *first-variational* matrix, which contains no core states at all. The
study's own "deepest state in the window" phrasing anticipates a second-variational or
all-electron block, where the numerator — and so the tape the study worries about —
would be larger still.

### Only one route survives, and the control says why

At $\Gamma$, with the $\Gamma_{25'}$ triplet inside the window, `grad(grad)` of the
scalar $\mathrm{Tr}[P M]$ along three Hermitian directions:

| | direction 1 | direction 2 | direction 3 |
|---|---|---|---|
| reference (central FD of the first derivative) | 0.00332328 | 0.00312484 | 0.00269235 |
| `sign_projector` | 1.6e-10 | 2.6e-11 | 8.8e-11 |
| `hard_window_projector` (the safe-$K$ rule) | **NaN** | **NaN** | **NaN** |
| naive | **NaN** | **NaN** | **NaN** |
| second difference of the loss (coarse control) | 8.8e-5 | 2.6e-6 | 1.0e-5 |

The reference is deliberately **not** a second difference of the loss, which loses four
digits; it is a central difference of the safe rule's *first* derivative, which §1f
validated against the closed form and which shares no code with the sign route.

**The control is in the same ground state.** At a generic $k$ with no degeneracy, all
three second derivatives agree with the reference to $4$–$9\times10^{-11}$. So the `NaN`
is not "second derivatives of a projector are hard", it is the enclosed multiplet — the
same statement §1f makes about the first derivative, extended one order. And the
safe-$K$ rule failing here is not a defect: it is its own docstring's warning,
now measured on a real matrix rather than argued.

### Through the assembly, and which side of the disagreement is wrong

Differentiating twice in $k$ — through `match`, the Gaunt contractions, the Cholesky
reduction and the sign iteration — gives $-2.4960647845$ in 12.8 s. The first
derivative through the two routes agrees to $2.4\times10^{-14}$, which is the entry
ticket. For the second, one FD step could only report a $10^{-6}$ disagreement without
saying whose. Refining it does say:

| central FD step $h$ | $10^{-3}$ | $3\times10^{-4}$ | $10^{-4}$ | $3\times10^{-5}$ |
|---|---|---|---|---|
| relative to the AD value | 1.44e-4 | 1.29e-5 | 1.44e-6 | 1.29e-7 |

Exactly $O(h^2)$ — a factor of 100 per decade of step — so this is the finite difference
converging **onto** the AD value, not AD converging onto anything. The same argument
Phase 0a′ used on the toy, now through the whole LAPW assembly.

### What this settles

`sign_projector` works on a real LAPW matrix, at a real symmetry multiplet, to second
order, at a cost of about 20 unrolled Newton-Schulz steps set by the top of the basis.
That closes the last Phase 1 item that needed no Phase 2 ingredient.

**It does not settle the §9.2 hybrid question**, and it would be easy to overclaim here.
The targets that decide that question — phonons, Born charges, response functions — need
second derivatives *through the SCF fixed point*, i.e. $dv^*/d\theta$ differentiated
again, which is Phase 0a′'s territory and has only ever been demonstrated on the toy.
What §1j supplies is the **projector** half of that composition, the piece 0a′ identified
as the blocker, now on a real LAPW matrix at a real multiplet. The fixed-point half at
anything like production shapes is still untested.

Two things it does not settle. `sign_projector` covers **hard windows only**; smeared
occupations at second order would need a Chebyshev expansion of the Fermi function,
which is separate work and is *not* what §1i built (§1i's closed form removes the
tolerance from the first derivative, not the `eigh` from the JVP). And the iteration is
unrolled, which at $n=177$ is free and at production shapes is exactly the tape Phase 0e
measured compile time against — `lax.scan` over the Newton-Schulz steps is the obvious
answer and has not been tried, since scanning a step whose body is two matmuls is the
textbook case for it.

---

## 1k. The potential: the radial functions, the radial integrals, and the derivative

### What was at stake

Through §1j, `hamiltonian.py` **imported** the five muffin-tin radial integrals
`haa`, `hloa`, `hlolo`, `oalo`, `ololo`. Everything Phase 1 measured — the
assembly, the $k$-derivative, the projector rule at a multiplet, smeared
occupations, second derivatives — was a statement about the machinery *above*
them. Below them sat Elk, and the plan (`docs/jax_port.md` §6, item 1b) put
building them in Phase 2 on the grounds that they need the muffin-tin
potential.

They do, and the potential is an SCF output. But it is also just an **input** to
this half of the calculation, and a converged one can be exported and held fixed
exactly as `STATE.OUT` already is. Patch **0015** does that, so the forward chain

$$v_s^{\rm MT} \;\longrightarrow\; u_{q\ell},\,v^{\rm lo}_p
 \;\longrightarrow\; h^{\alpha}_{qq';\ell\ell'\ell''m''},\,o^{\alpha}_{qp}
 \;\longrightarrow\; H,\,O \;\longrightarrow\; \varepsilon_j$$

can be closed and differentiated now, with no Phase 2 ingredient at all. What
remains imported is the interstitial pair (which genuinely does need the
*interstitial* potential) and the linearisation energies.

### What was built

`src/elkjax/radial.py` transcribes `hmlrad.f90` and `olprad.f90` —
pure quadrature, one weighted sum over the radial mesh per integral.
`src/elkjax/radial_functions.py` transcribes `rschrodint.f90`, `genapwfr.f90`
and `genlofr.f90` — the scalar-relativistic radial equation in Koelling-Harmon
form,

$$\frac{dP_\ell}{dr} = 2MQ_\ell + \frac{P_\ell}{r},\qquad
\frac{dQ_\ell}{dr} = -\frac{Q_\ell}{r}
 + \left[\frac{\ell(\ell+1)}{2Mr^2} + V - E\right]P_\ell,\qquad
M = 1 + \frac{E-V}{2c^2},$$

integrated outward on Elk's own logarithmic mesh by its own predictor-corrector:
a 3-point extrapolation of the derivative, then eight fixed-point corrections in
which the function is recovered by integrating the cubic through the last four
derivative samples. **The scheme is transcribed, not improved.** A Runge-Kutta
step or `jax.experimental.ode` would converge to the same continuum solution and
disagree with Elk at the mesh's own truncation error, four or five decades above
the roundoff this is checked at; an element-wise comparison against `apwfr` is
only available if the arithmetic is the same arithmetic.

Element-wise against Elk, on the three §1a fixtures:

| quantity | worst relative disagreement |
|---|---|
| `oalo`, `ololo` | 2.4e-16 |
| `haa`, `hloa`, `hlolo` | 3.3e-17 – 2.0e-16 |
| `apwfr`, `apwdfr` | 1.3e-14 |
| `lofr` | 7e-15 |
| `evalfv` from the potential alone | < 1e-8 Ha |

### The export was inconsistent with itself, and the split by potential found it

The first run of `hmlrad` in JAX reproduced Elk to $2.6\times10^{-16}$ on every
integral that does **not** touch the potential and was off by
$3\times10^{-10}$ on every one that does. That is not a transcription bug and it
is not roundoff: `gndstate.f90` calls `genapwlofr` at the *top* of an SCF
iteration and `potks`/`mixerifc` at the *bottom*, so on exit the radial
functions and integrals belong to the potential of the **previous** iteration
while `vsmt` is the current one. Nothing before patch 0015 could see this,
because nothing before it exported the potential.

Patch 0015 therefore calls `genapwlofr` before building anything. The lesson is
the split, not the fix: an aggregate "`haa` is off by 3e-10" would have read as a
subtle indexing error, and the two-way split named the cause in one run.

### The two channels of the potential are exactly complementary

The point of building the integrals rather than importing them is the
derivative, and the derivative separates into two pieces that can be computed
independently:

* **frozen basis** — hold `apwfr`/`lofr` fixed, let only the radial integrals
  respond. The overlap does not depend on the potential at all then, so
  first-order perturbation theory for the generalised eigenproblem collapses to
  $\delta\varepsilon_n = c_n^\dagger\,\delta H\,c_n$ with
  $c_n^\dagger Oc_n=1$: a closed form, computable from Elk's own `evecfv`, with
  no finite difference and no perturbed eigensolve in it.
* **full** — let the perturbation reach the radial functions, which `genapwfr`
  re-solves in the perturbed potential at the same linearisation energies.

Their difference is the **basis response**, the term any frozen-basis argument
drops. Splitting a random direction into its spherical ($\ell=0$) and
non-spherical halves turns that from a number into a structural statement, and
the two halves come out **exactly complementary** — on bulk Si, at
$k=(0.1,0.2,0.05)$, over the four occupied first-variational bands:

| direction | frozen-basis AD | full AD | basis response |
|---|---|---|---|
| spherical | **0** (exactly) | 1.3215e-1 | **100%** |
| non-spherical | 2.6258e-2 | 2.6258e-2 | **0** (7.0e-16 absolute) |
| random (their sum) | 2.6258e-2 | 1.5841e-1 | 83% |

Both zeros are structural, not numerical, and each has a one-line cause.

**The spherical potential never appears in a radial integral.** `hmlrad`'s
$\ell_2=0$ element is $\int u_{q\ell}(\hat Hu_{q'\ell})r^2dr$, and `genapwfr`
has already applied $\hat H$ — the radial functions *are* that operator's
solutions. This is the LAPW construction itself: the spherical potential is
absorbed into the basis, and only the non-spherical remainder survives as an
explicit matrix element. So a Hellmann-Feynman-shaped treatment of the
muffin-tin potential does not lose a small correction in the spherical channel;
**it loses the entire term.**

**The non-spherical potential never reaches the radial equation.**
`genapwfr`/`genlofr` integrate in the spherical part alone, so the basis cannot
respond and the two branches agree to $4\times10^{-16}$ absolute.

The random direction's 83% is then just the mixture, and is quoted only to say
that the effect is not a corner case of the split.

### The basis response has two halves, and one of them was missed first

A perturbed potential reaches $H$ and $O$ through the radial functions **twice**:
once inside the radial integrals, and once through the matrix $D$ of radial
derivatives at $R_{\rm MT}$ that `match` inverts for the matching coefficients.
The first version of this measurement rebuilt `apwfr`/`lofr` and the integrals
and left `apwalm` at its exported value — a basis whose shape at the sphere
boundary is frozen while its interior moves.

**Nothing in the gradient checks could see it.** AD and central FD then
differentiate the *same* truncated function and agree to $4\times10^{-10}$;
the closed form pins the frozen branch, which is unaffected; both structural
zeros survive (the frozen branch holds $D$ fixed by definition, and a
non-spherical perturbation moves neither `apwfr` nor $D$). This is Phase 0's
own carried-forward finding — *a green gradient test does not validate a
transcription* — recurring exactly as stated, and the check that exposes it is
the forward one: rebuild $D$ from `apwfr` and compare against the exported
`dmat` (bitwise at `apword=1`, 3.4e-16 at `apword=2`).

Quantitatively the omission was **not** small. With $D$ frozen the spherical
channel gave 9.8917e-2 against a true 1.3215e-1: the matching response alone is
a quarter of the basis response and 21% of the full derivative. So "freeze the
basis" has two distinct meanings inside the muffin tin, and neither is
negligible.

### AD against finite differences of the same function

On the random direction, central FD of the identical JAX function:

| step | frozen branch | full branch |
|---|---|---|
| $10^{-4}$ | 2.0e-9 | 4.9e-10 |
| $10^{-5}$ | 4.8e-9 | 8.7e-10 |
| $10^{-6}$ | 7.4e-8 | 3.8e-8 |

The disagreement **grows** as the step shrinks, which is the $1/h$ signature of
roundoff in the difference and not of a wrong gradient — the criterion
`docs/jax_port.md` states for Phase 1 and Phase 4 alike. The closed form is the
sharper check anyway: it pins the frozen branch to $1.2\times10^{-15}$ with no
step size in it.

### One thing the closed form got wrong first, and it is worth remembering

The map from the potential to the radial integrals is **affine, not linear**.
Building "the integrals of $\delta V$" and calling the result $\delta H$ carries
the $\ell_2=0$ block — which is the constant part, $\langle u|\hat Hu\rangle$ —
into the derivative at full strength. On bulk Si that does not merely degrade
the reference: it changes its sign ($-2.27\times10^{-1}$ against a true
$+2.63\times10^{-2}$). The fix is to zero the $\ell_2=0$ slice, which is the
same fact as the first zero in the table above, met from the other side.

### What this settles, and what it does not

Phase 1's item 1b is done, and done without Phase 2. The first-variational
spectrum is a differentiable function of the muffin-tin Kohn-Sham potential,
checked forward element-wise against Elk at every intermediate stage and
backward against both a closed form and finite differences.

**It is not the position derivative.** Moving an atom moves the potential
inside its sphere, and where that potential comes from is Phase 2. What is now
available that was not is the *frozen-potential* position derivative — the
`apwalm` structure factor alone — and the isolation `docs/jax_port.md` §Phase 4
asks for (the same force computed with and without `stop_gradient` on
`apwalm`), because the other half of that comparison now exists.

**The linearisation energies are held fixed** throughout, and are exported
rather than re-solved. Letting them float means differentiating `linengy`,
which finds them by bisection on the logarithmic derivative at $R_{\rm MT}$ —
a different object, and one Elk's own forces do not carry either.

**The interstitial blocks are still imported**, so this is a derivative with
respect to the muffin-tin potential only. $H^{\rm I}$ needs $V_s$ in the
interstitial region, which is genuinely Phase 2.

**A warning for Phase 2's own gradient criteria.** `docs/jax_port.md` asks for
$v_{xc}$ from `jax.grad` of $\varepsilon_{xc}$ against Elk's hand-coded
$v_{xc}$ pointwise. That is a check on the functional, and it is worth having,
but the table above says it would not catch the failure mode that matters here:
a Phase 2 chain that produced a perfectly correct $\delta v_s$ and then fed it
to a frozen LAPW basis would return **zero** for the spherical channel while
looking entirely healthy.

