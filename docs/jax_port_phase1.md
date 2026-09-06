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
| **1e** the adversarial `soc_scale` sweep, and the required refusal | not started; Phase 0b(ii) found it must extend below `soc_scale = 1` |

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
- **No degeneracy has been touched.** The fixture is a generic $k$ with every
  low band separated by $>10^{-3}$ Ha, asserted in the test. The adversarial
  `soc_scale` sweep and the required refusal at a closing gap — the only place
  blocker B1 actually bites, and the item 0b(ii) says must be extended below
  `soc_scale = 1` — are untouched.
- **`eigvalsh`, not the safe-$K$ rule.** The Cholesky reduction is here but it
  is not wired to `projector.py`'s custom rule or to a per-run $\kappa(O)$;
  at a multiplet this pipeline would fail exactly the way 0b describes.
