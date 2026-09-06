# Porting Elk to Python + JAX: a design study

**Status: study, not a plan of record.** Nothing here has been implemented. Every claim is
labelled by how it was obtained: *measured* (a number produced in this repository or in a JAX
session), *read* (a line of `vendor/elk/src/` with a file reference), *cited* (published work),
or *unverified* (reported but not checked here). Where the readers who produced the underlying
surveys disagreed, the disagreement is stated rather than resolved silently; see
§10.

Scope: Elk 11.0.2 as vendored at `vendor/elk/`, 821 `.f90` files, 74,714 lines (measured). The
*physics* scope is narrower than "Elk" and is stated once, in §2.2: everything from §2.2 onward
assumes a plain LSDA/GGA ground state with `dftu=0`, `ftmtype=0`, `fsmtype=0` and
`ksgwrho=.false.`. §3.5 lists the subsystems deliberately outside that boundary and §7 says what
happens to them.

---

## 1. Verdict

**This is a research project, and it is justified by differentiability, not by the GPU.**

The two stated motivations have opposite cost/benefit profiles, and the honest answer is that
one of them does not survive contact with the evidence.

**The GPU motivation, on its own, is a bad idea.** Four facts, none of them marginal:

1. *It already exists, in C++.* SIRIUS is a CUDA/ROCm domain-specific library implementing
   FP-LAPW that was created under PRACE-2IP WP8 explicitly "as a general-purpose solution for
   optimization and scaling of both Exciting and Elk full-potential LAPW codes" (cited). It is
   actively developed, in production at CSCS, and an Elk-lineage fork (Exciting-Plus, from Elk
   1.0.17) has already been interfaced to it with published results.
2. *The hot spots are already BLAS-3 at near-peak.* For FLEUR, H/S construction plus the
   eigensolver is >80% of runtime, and once H/S construction is restructured into
   `zherk`/`zher2k`/`zgemm`, 98.06–99.75% of its FLOPs live in those three kernels, reaching
   80–90% of vendor-BLAS peak (cited). JAX buys nothing over calling cuBLAS from Fortran there.
   FLEUR got 5x over optimised CPU (7.5–12.5x over original FLEUR) from roughly **100 lines** of
   cuBLAS-XT wrappers (cited).
3. *An all-electron code cannot leave FP64.* `rlmt` stores $r^\ell$ for $\ell=-\ell_{\max}-1$ to
   $\ell_{\max}+2$ on a logarithmic mesh running from $r_{\min}\approx2\times10^{-7}$ to
   $r_{\max}\approx43$ Bohr; for tungsten $r^{-7}=2.7\times10^{46}$ (float32 overflow) and
   $r^{8}=8.5\times10^{-54}$ (float32 underflow), both at the *first* mesh point (measured).
   Core eigenvalues reach $-2547.66$ Ha with a $10^{-12}$ relative convergence criterion.
   The non-tensor FP64 pipe is half rate on datacentre cards and 1/32–1/64 on consumer cards, so
   the FP32/TF32 advantage that makes JAX attractive for ML is unavailable. (The FP64 *tensor*
   path on H100 matches FP32 at 51 TFLOPS; it is the plain FP64 path that is halved. Later
   generations are reported to cut FP64 much harder still — *reported by review, not verified
   here*; if true it strengthens this fact rather than weakening it.)
4. *Every GPU-specific claim in the source material is unmeasured.* No machine used in preparing
   this study had a CUDA-enabled `jaxlib`. The one directly relevant CPU measurement is
   discouraging: `jax.vmap(jnp.linalg.eigh)` over 8 matrices of $n=400$ gave **1.03x** over a
   Python loop, against 1.65x for a vmapped matmul, because XLA lowers a batched `eigh` to one
   LAPACK call per batch element (measured, CPU). What the GPU lowering does is *unverified here*
   and the review process reports it has changed: recent `jaxlib` is said to dispatch
   `gpusolverDnXsyevBatched` — a genuinely batched kernel — above $n=32$ on CUDA $\ge$ 12.6.2,
   falling back to a host loop on older cuSOLVER. Even if so, the batch chunk is
   $\texttt{INT\_MAX}/(n^2\cdot16)\approx14$ matrices at $n=3000$ in complex128, and device memory
   binds first (100 k-points $\times\,3000^2\times16$ B $=14.4$ GB for $H$ alone). §6's 0d remains
   the right first benchmark; only the expectation attached to it changes.

**The differentiability motivation is real, and the gap it fills does not already exist —
provided the gap is stated narrowly.** No APW/LAPW/all-electron code with automatic
differentiation was found, in JAX, PyTorch or Julia; that is an absence-of-evidence result from
targeted searching, not a proof (§11). What *does* exist, and must not be overstated away: DFTK
(plane-wave, pseudopotential, Julia) shipped **forward mode only** — and still delivered
inverse design of a band gap and the fitting of exchange-correlation functional parameters with
it, so those two capabilities are *not* evidence for reverse mode; what DFTK actually defers to
reverse is narrower, "high-dimensional gradients … (such as training highly parameterized ML
exchange-correlation functionals)" — hand-writing the SCF
derivative as DFPT and leaving reverse mode as "a promising avenue for future extension" (cited);
and **PySCFAD does support periodic systems** (`pyscfad/pbc/{scf,dft,gto,df,tools}`, published as
"for molecules and materials at the mean-field level and beyond") and ships its own
`implicit_diff.py` with `custom_root`/`custom_fixed_point` — so "no periodic DFT code has implicit
differentiation" is false, though *whether reverse mode through a periodic SCF is exercised there
was not verified*. The defensible claim is therefore **first differentiable all-electron DFT**,
not first differentiable periodic DFT. That is also the claim $dE/dZ$ supports: only an
all-electron code can differentiate with respect to nuclear charge, because a pseudopotential is
a discontinuous function of the element.

**The single tightest constraint is that derivatives through the eigenproblem are undefined at
degeneracies and *silently wrong* near them, and crystal symmetry makes degeneracies generic
rather than exceptional.** This partly contradicts the received framing. Most of the
subsystem surveys and the prior-art scout reported the failure mode as `NaN`; the eigensolver
survey independently measured finite garbage ($4.3\times10^{14}$ at exact degeneracy, "fails
SILENTLY"); the adversarial lens supplied the mechanism. Both happen, and which one you get
depends on whether the degeneracy is bitwise. When a degenerate
Hamiltonian is assembled the way a real code assembles one ($U\,\mathrm{diag}(\varepsilon)\,U^H$,
i.e. out of FFTs and GEMMs), the degenerate pair comes back split by $8.9\times10^{-16}$ rather
than bitwise equal, so JAX's `Fmat = 1/(w_j - w_i)` is $\sim10^{15}$ and the gradient is a large
**finite** number whose sign is set by roundoff: five assemblies of the *identical* spectrum gave
$+1.43\times10^{15}$, $+6.59\times10^{14}$, $-3.84\times10^{15}$, $+7.10\times10^{14}$,
$+6.99\times10^{14}$ (measured, JAX 0.7.1, CPU, x64). `NaN` appears only where the degeneracy is
*bitwise* exact. That is not confined to unit tests: §3.2's own recommended padding scheme
($H_{\rm pad}=E_{\rm big}\mathbb 1$) manufactures a bitwise $(n_{\rm matmax}-n_{\rm mat})$-fold
degeneracy at every k-point, and a projector gradient through it returns `NaN` even when the
physical spectrum is perfectly non-degenerate (measured; §3.2 now pads with distinct values
instead). From the *physical* spectrum, in production, you get a number. This is
precisely the failure class this repository's `CLAUDE.md` is a record of: the Berry-curvature
sign that survived because every check was sign-blind; the cesium $Z_2$ that was
under-resolved rather than wrong-looking.

A second, quieter face of the same problem: **individual eigenvalues are not safe either**,
though occupation-weighted sums of them are. At an exact two-fold degeneracy, JAX returns the
derivatives of `eigh`'s arbitrarily-chosen branches, central finite differences of the *sorted*
spectrum return their average, and the true one-sided directional derivatives are neither —
measured $(-0.204,-0.793)$, $(-0.498,-0.498)$ and $(-1.553,+0.557)$ for the same perturbation,
with only the multiplet trace agreeing to machine precision (§8b). Every claim in this document
about $d\varepsilon_j/d\mathbf R$, deformation potentials and "exact" effective masses inherits
that caveat.

The fix is known and standard — differentiate the occupied-subspace projector through a custom
rule, never individual eigenvectors — and it *works* (measured: closed-window error at exact
degeneracy drops from $2.1\times10^{-2}$ to $3.0\times10^{-10}$). But it comes with a trap that
is worse than the original problem, and with one genuinely open question:

- **The fix launders an ill-posed case.** When the band window boundary sits *inside* a
  degenerate multiplet the quantity has no derivative at all. Naive AD returns $-7.4\times10^{11}$,
  which screams; the safe rule returns $-1.01$, which does not; the truth is $+2.8\times10^{5}$
  (measured). Adding the standard mitigation and stopping checking makes the bug invisible. The
  only defence is a hard gapped-window precondition — which this repository already enforces for
  independent physics reasons (`parsers.symmetry.check_window_gap`, added in §23 after exactly
  this class of confident wrong answer).
- **Second-order differentiation through a fixed point whose every iteration contains `eigh` is
  untested.** First-order reverse mode through the SCF is *not* the open question it was taken
  for: measured here, `jax.custom_vjp` around the fixed-point solver with GMRES on the transposed
  operator matches central finite differences to $3.4\times10^{-10}$, which is the standard
  library pattern (jaxopt, optimistix, PySCFAD's own `implicit_diff.py`). What fails is one
  specific composition — `lax.custom_root` with an *iterative* `tangent_solve` does not transpose
  (measured `NotImplementedError`), while the same `custom_root` with a dense `tangent_solve`
  does. The genuinely open item is the second derivative that §8f items 2, 3 and 8 and all of
  Phase 5 rest on: `jax.hessian` through a `custom_vjp` fixed point *does* work on a 6-dimensional
  toy, matching finite differences to $2.5\times10^{-9}$ even with the primal solve hidden behind
  a `lax.while_loop` (measured) — but it works by forward-differentiating the solver and the
  backward rule, so at Elk scale that outer JVP passes through `eigh`'s derivative at every SCF
  iteration. The safe-$K$ projector rule must therefore be twice-differentiable, and the padding
  block must not be exactly degenerate. Neither has been tested.

**Distinguish what kills the project from what makes it expensive.** The degeneracy pair above —
first order at a multiplet, and second order through an SCF full of `eigh` calls — can kill it.
Elk's mutable global state makes it expensive: 651 of 821 files (79%, measured) contain
`use modmain`, and `modmain.f90` is 1297 lines of pure declaration holding 654 module-level names
of which 159 are allocatable or pointer (measured). Converting that is the dominant cost and
carries no risk of being impossible — only of taking years. It is budgeted as §6's Phase S.

**The two motivations also pull the design apart in two concrete places**, which is why a
document that treats them as one goal will produce a design that serves neither:

| | GPU throughput wants | Reverse-mode AD wants |
|---|---|---|
| Precision | Elk's own single-precision path (`cgemv`/`sdot` in `eveqnsv`, `cgemm` in `wfmtsv_sp` and the single-precision SHT) | `complex128` downstream of the eigensolve — measured, `complex64` flips the *sign* of a gradient at a $10^{-6}$ Ha splitting where `complex128` is still correct |
| k-axis | `vmap` over as many k as fit, for batched GEMMs | `lax.scan` over k-chunks with a small local density matrix as carry, because a naive `vmap` materialises the whole tape |

**Recommendation, ranked.**

- **Best value for this repository today:** the hybrid in §9 — JAX kernels for new physics on top
  of Elk-converged eigenstates, Fortran keeping the SCF. It delivers $d\varepsilon_j/d\mathbf R$
  (i.e. `forcek.f90`'s content) and $\partial u/\partial\mathbf k$ (i.e. exact quantum geometry,
  killing §15's Löwdin apparatus) with **no fixed point at all**, because differentiating at
  fixed potential or with respect to $\mathbf k$ needs no SCF derivative. Cost, summed from the
  §3 rows it needs (`match`, `sbessel`/`genylmv`, `gengkvec`/`gensfacgp`, the radial functions and
  integrals, H/S assembly, the Cholesky-reduced eigensolve, and a `STATE.OUT` reader):
  **roughly 13–15 weeks**, which is a sum of estimates, not a measurement. It cannot deliver
  anything needing $dv^*/d\theta$: second derivatives, response functions, high-dimensional
  ML-XC training, reverse-mode inverse design.
- **If GPU speed is the goal:** interface Elk to SIRIUS, or restructure H/S into
  `zherk`/`zher2k` and hand those plus the eigensolver to cuBLAS/cuSOLVER/ELPA. Both have direct
  published precedent and bounded cost.
- **The full port is justified only if those capabilities are the goal**, and only after
  Phase 0 (§6) has passed. Note what DFTK's forward-only precedent removes from that list:
  low-dimensional inverse design and few-parameter functional fitting do *not* require the full
  port's reverse mode. What does is high-dimensional ML-XC training and everything second-order.
  Scope it as "differentiable all-electron DFT", not as "Elk on GPUs".

---

## 2. What Elk is, in the terms that matter for a port

### 2.1 The FP-LAPW basis

Space is partitioned into non-overlapping muffin-tin spheres $\mathrm{MT}_\alpha$ of radius
$R_\alpha$ centred on each nucleus, and the interstitial $I$. For each $\mathbf k$ and each
reciprocal-lattice vector $\mathbf G$ with $|\mathbf k+\mathbf G| < g_{\max}$, the basis function is

$$
\phi_{\mathbf G+\mathbf k}(\mathbf r) =
\begin{cases}
\Omega^{-1/2}\, e^{i(\mathbf G+\mathbf k)\cdot\mathbf r}, & \mathbf r \in I\\[4pt]
\displaystyle\sum_{\ell=0}^{\ell_{\rm apw}}\sum_{m=-\ell}^{\ell}\sum_{j=1}^{M_\ell^\alpha}
A^{\alpha,j}_{\ell m}(\mathbf G+\mathbf k)\, u^\alpha_{j\ell}(r)\, Y_{\ell m}(\hat{\mathbf r}),
& \mathbf r \in \mathrm{MT}_\alpha
\end{cases}
$$

with $\Omega$ the cell volume, $u^\alpha_{j\ell}$ numerical radial functions, $Y_{\ell m}$
complex spherical harmonics, $\ell_{\rm apw} = \texttt{lmaxapw} = 8$ at default
(`readinput.f90:80`, so $\texttt{lmmaxapw} = (\ell_{\rm apw}+1)^2 = 81$ — the 81 in §5's
`gntyry(49,81,81)`), and $M_\ell^\alpha = \texttt{apword}(\ell,\alpha)$ the APW order
(1 in every species file Elk ships — measured; the general case is reached only via `nxoapwlo`).
In addition there are $n_{\rm lo}$ **local orbitals**, functions $v^\alpha_p(r)Y_{\ell m}$ that
vanish outside their own sphere. The basis size at one $\mathbf k$ is
$n_{\rm mat}(\mathbf k) = n_{\mathbf G+\mathbf k}(\mathbf k) + n_{\rm lo,tot}$.

**Name the basis correctly: at Elk's shipped defaults this is APW+lo, not LAPW** (Sjöstedt,
Nordström & Singh, Solid State Commun. 114, 15 (2000)). $\texttt{apword}=1$ means the matching
condition below fixes the *value* only, with no slope matching — that is a fixed-energy APW, not
a linearised one. The derivative freedom is supplied instead by the local orbitals, and the
species files show them doing exactly that: `Si.in` has `nlorb=2` with `lorbl` $=0,1$,
`lorbord=2`, `lorbdm` $=0,1$ (i.e. $u$ and $\partial u/\partial E$ at the same energy) on the
*valence* channels — linearisation freedom, not semicore. `C.in` carries both roles, two valence
`lo`s of that shape plus a third at $E=-0.5012$ Ha with `lorbve=T`. This matters beyond
nomenclature: §1 fact 2's FLEUR BLAS-3 numbers are for an LAPW H/S restructuring, and APW+lo's
block structure (a large APW–APW block plus APW–lo and lo–lo blocks sized by `nlotot`) is not
the same shape, so those numbers are a proxy here, not a transfer.

The coefficients $A$ are fixed by continuity of the function and its first $M_\ell-1$
derivatives at $r = R_\alpha$:

$$
\sum_{j} D_{ij}\,A^{\alpha,j}_{\ell m} = b_i,\qquad
D_{ij} = \left.\frac{d^{\,i-1}u^\alpha_{j\ell}}{dr^{\,i-1}}\right|_{R_\alpha},\qquad
b_i = \frac{4\pi i^\ell}{\sqrt\Omega}\,|\mathbf G+\mathbf k|^{i-1}
 j_\ell^{(i-1)}(|\mathbf G+\mathbf k|R_\alpha)\, e^{i(\mathbf G+\mathbf k)\cdot\mathbf r_\alpha}\,
 Y^*_{\ell m}(\widehat{\mathbf G+\mathbf k}).
$$

For $M_\ell = 1$ this collapses to a closed form,

$$
A^{\alpha,1}_{\ell m} = \frac{4\pi i^\ell}{\sqrt\Omega}\,
\frac{j_\ell(|\mathbf G+\mathbf k|R_\alpha)}{u_{1\ell}(R_\alpha)}\,
e^{i(\mathbf G+\mathbf k)\cdot\mathbf r_\alpha}\,
Y^*_{\ell m}(\widehat{\mathbf G+\mathbf k}).
$$

**Note a trap that will cost days if missed** (read, `match.f90:78-93` + `genylmv.f90`): the
$4\pi i^\ell$ prefactor is *not* written anywhere in `match.f90`. It is carried by `genylmv`'s `t4pil` flag,
which returns $4\pi(-i)^\ell Y_{\ell m}$; `match` then takes `conjg`. Implementing `match`'s
docstring formula with ordinary spherical harmonics gives an $\ell$-dependent complex error in
`apwalm` whose $H$ and $S$ are still Hermitian and whose eigenvalues are still real — just wrong.

The radial functions solve the scalar-relativistic (Koelling–Harmon) radial equation at a **fixed
linearisation energy** $E_\ell$, with $P = rg_\ell$, $Q = (r/2M)\,dg_\ell/dr$:

$$
\frac{dP}{dr} = 2MQ + \frac{P}{r},\qquad
\frac{dQ}{dr} = -\frac{Q}{r} + \left[\frac{\ell(\ell+1)}{2Mr^2} + V_s(r) - E\right]P,\qquad
M = 1 + \frac{E-V_s}{2c^2}.
$$

Core states instead solve the radial **Dirac** equation and are found by shooting on the node
count.

### 2.2 The self-consistency loop

Read directly off `gndstate.f90:133-320` (the `do iscl` loop; `gencore` at 154, `genvsig` at 218,
`energy` at 230, the convergence test at 295–314), the loop body is

```
gencore -> linengy -> genapwlofr -> gensocfr -> genevfsv -> occupy
        -> rhomag | [gwrhomag if ksgwrho]
        -> [DFT+U / FTM: gendmatmt -> genvmatmt] -> potks -> mixerifc
        -> [FSM: bfieldfsm -> addbfsm] -> genvsig -> energy -> convergence
```

**Everything in brackets is outside the fixed-point picture §8(a) builds, and two of them are not
merely optional branches.** With `dftu /= 0` or `ftmtype /= 0`, `gendmatmt`/`genvmatmt` produce
`dmatmt`/`vmatmt`, which enter `eveqnsv` but are **not** part of `vsbs` (`init0.f90:633-645`) and
are never handed to the mixer — a second, unmixed fixed-point variable. With `fsmtype /= 0`,
`bfieldfsm` runs *after* `mixerifc` and is an integral controller,
$\mathbf B_{\rm fsm}\leftarrow\mathbf B_{\rm fsm}+\tau_{\rm fsm}(\mathbf m-\mathbf m_{\rm fix})$,
so `bfsmc`/`bfsmcmt` are persistent state outside the mixer whose converged condition is the
*constraint* $\mathbf m=\mathbf m_{\rm fix}$, not self-consistency. That is not academic for this
repository: `docs/design.md` §29's exchange tensor is built entirely on `fsmtype=-2` plus
per-atom `mommtfix`, so the one elkpy capability whose gradient would be worth the most sits in
the case the analysis below excludes. And `if (ksgwrho) call gwrhomag` (line 184) puts a GW
density branch inside this very loop. **Everything from here to §8 assumes
`dftu=0`, `ftmtype=0`, `fsmtype=0`, `ksgwrho=.false.`, LSDA or GGA.**

**Eigenproblem.** Because the basis is non-orthogonal, each $\mathbf k$ gives a *generalized*
Hermitian problem

$$
H(\mathbf k)\,c_n = \varepsilon_n\, S(\mathbf k)\, c_n,
$$

of order $n_{\rm mat}$, of which only the lowest $n_{\rm stfv}$ eigenpairs are wanted. Blockwise,
with $\tilde\Theta$ the Fourier transform of the interstitial characteristic function and
$\Gamma$ the Gaunt tensor `gntyry`,

$$
H^{I}_{\mathbf G\mathbf G'} = \tfrac12(\mathbf G+\mathbf k)\!\cdot\!(\mathbf G'+\mathbf k)\,
\tilde\Theta(\mathbf G-\mathbf G') + \widetilde{V_s\Theta}(\mathbf G-\mathbf G'),\qquad
S^{I}_{\mathbf G\mathbf G'} = \tilde\Theta(\mathbf G-\mathbf G'),
$$

$$
H^{\rm MT} = \sum_\alpha A_\alpha^{\dagger}\, \mathcal M_\alpha\, A_\alpha,\qquad
S^{\rm MT} = \sum_\alpha A_\alpha^{\dagger} A_\alpha,\qquad
\mathcal M_\alpha[i,j] = \sum_{\ell_2=\ell_0}^{\ell^{\rm o}_{\max}}
\sideset{}{'}\sum_{m_2}\Gamma_{\ell_2m_2,j,i}\;
\texttt{haa}[\ell_2 m_2, j, i, \alpha],
$$

where `haa` holds radial integrals $\int u_i V_{\ell_2m_2} u_j\, r^2 dr$ plus a kinetic surface
term (`hmlrad.f90:63`). Three details of that $\ell_2$ sum are load-bearing and easy to miss
(`hmlaa.f90:47-55`): it stops at $\ell^{\rm o}_{\max}=\texttt{lmaxo}=6$, not at
$\ell_{\rm apw}$; it runs in steps of **2** from
$\ell_0 = \texttt{merge}(0,1,\bmod(\ell_1{+}\ell_3,2)=0)$, the parity selection rule; and
`hmlrad` splits it at $\texttt{lmaxi}=1$ (see the two-block layout below). The second term of
$H^I$ is `vsig`, which `genvsig.f90` Fourier-transforms from `vsirc` — and `vsirc` is the
interstitial potential *already multiplied by the characteristic function* and truncated to the
coarse G-set, so it is the transform of $V_s\Theta$, not of $V_s$; `potks.f90` additionally calls
`trimrfg(vxcir)`, hard-zeroing $v_{xc}$ above $2g_{\max}$. $\tilde\Theta$ itself is not an
indicator function's transform either: `gencfun.f90` builds it analytically and truncates at
$G_{\max}$, so it is a Gibbs-truncated step.

There are therefore **two** reciprocal-space cutoffs, and only one of them is $g_{\max}$: the
density/potential cutoff `gmaxvr` (default 12, `readinput.f90:78`) sets `ngvec`, the FFT grid
`ngridg` and the pseudocharge order `npsd`, and is a first-order convergence parameter a port
must expose alongside `rgkmax`.

A **second variation** then diagonalises an $n_{\rm stsv}\times n_{\rm stsv}$ standard
Hermitian matrix in the first-variational eigenbasis, adding $\mathbf B_s\!\cdot\!\boldsymbol\sigma$,
spin-orbit $f_{\rm soc}(r)\,\hat{\mathbf L}\!\cdot\!\boldsymbol\sigma$, DFT+U and field terms;
$n_{\rm stsv} = n_{\rm stfv}\,n_{\rm spinor}$. **This is a variational approximation, not a change
of representation**: the spinor Hamiltonian is restricted to the span of the lowest $n_{\rm stfv}$
first-variational states, and $n_{\rm stfv} = \mathrm{nint}(Z_{\rm val}/2) + n_{\rm empty} + 1$
clamped by $\min_{\mathbf k}n_{\rm mat}$ (`init1.f90:319-330`), with
$n_{\rm empty} = \mathrm{nint}(\texttt{nempty0}\cdot n_{\rm atm})$ and `nempty0` $=4$. Convergence
in $n_{\rm empty}$ is a real and, with strong spin-orbit coupling, delicate parameter. A port has
a choice Elk does not offer — one-shot diagonalisation of the full $n_{\rm mat}n_{\rm spinor}$
spinor problem, which removes the truncation *and* removes one differentiated eigensolve (hence
one degeneracy hazard site) from the graph, at the cost of a $2\times$-larger dense eigenproblem.
Not evaluated here; worth evaluating before Phase 1 fixes the design.

The spin-orbit radial function is not an abstract $\xi(r)$: `gensocfr.f90` computes
$f_{\rm soc}(r) = \frac{1}{(2Mc)^2}\frac1r\frac{\partial V_s}{\partial r}$ with
$M = 1 + (E-V_s)/2c^2$, **with $E$ set to zero** and **$V_s$ the spherical ($\ell=m=0$) part of
the Kohn-Sham potential only** (Koelling & Harmon, J. Phys. C 10, 3107 (1977)). Both are
approximations, and both are inherited unexamined by this repository's patch 0001, which scales
this term per species (§9.1).

**Density and occupations.** `rhomag.f90` is `rhomagv` (valence) $\to$ `rhocore` $\to$ `charge`
$\to$ `rhonorm`, i.e. $\rho = \rho_{\rm val} + \rho_{\rm core}$ followed by a normalisation step.
The valence part is

$$
\rho_{\rm val}(\mathbf r) = \sum_{\mathbf k} w_{\mathbf k}\sum_n f_{n\mathbf k}
\sum_\sigma |\psi_{n\mathbf k\sigma}(\mathbf r)|^2,\qquad
\mathbf m(\mathbf r) = \sum_{\mathbf k} w_{\mathbf k}\sum_n f_{n\mathbf k}\,
\psi^\dagger_{n\mathbf k}\boldsymbol\sigma\psi_{n\mathbf k},
$$

where $w_{\mathbf k}$ is the k-point weight, $\boldsymbol\sigma$ the Pauli vector, and
$f_{n\mathbf k} = \texttt{occmax}\cdot\tilde\theta\!\left((\mu-\varepsilon_{n\mathbf k})
/\texttt{swidth}\right)$ with $\texttt{occmax}$ the maximum occupancy of one state (2 for an
unpolarised calculation, 1 otherwise), $\tilde\theta$ a smoothed step function, $\mu$ the Fermi
level and `swidth` the smearing width. $\mu$ is fixed by
$\sum_{\mathbf k}w_{\mathbf k}\sum_n f_{n\mathbf k} = Z_{\rm val}$, solved by bisection to
$10^{-12}$ (read, `occupy.f90`). Default `stype=3` (Fermi–Dirac), `swidth = 1e-3` Ha.

**The core density carries four approximations that a port inherits and that no equation above
shows** (read, `gencore.f90`/`rhocore.f90`/`rhonorm.f90`). (i) The core radial Dirac equation is
solved in the **spherical part** of $v_s$ only. (ii) That potential has the free-atom potential
appended for $r > R_{\rm MT}$. (iii) $\rho_{\rm core}$ is therefore spherically symmetric by
construction and is added to the $\ell=m=0$ component alone; core charge that leaks outside the
sphere is *recorded* as $\texttt{chgcrlk} = \texttt{chgcr} - \int_{\rm MT}\rho_{\rm core}$ and
never added to the interstitial. (iv) With `spincore`, Elk's own docstring calls the procedure
"a simple, but inexact, approach". Then `rhonorm` adds a **constant**
$t = (\texttt{chgtot}-\texttt{chgcalc})/\Omega$ to $\rho$ at every point of both regions, to
repair the charge lost in the $(\theta,\phi)\to Y_{\ell m}$ back-transform. That shift is
differentiable but *global* — it couples every grid point to every other — so hazard L's core
custom VJP is necessary and not sufficient on its own.

**Potential.** $v_s = v_C[\rho] + v_{xc}[\rho,\mathbf m]$, with $v_C$ from the Weinert
pseudocharge method (`zpotcoul.f90`). Written out, because the pseudocharge is the object the
method is named for and it does not appear in the prose version: the muffin-tin multipoles are
$q^{\rm MT}_{\ell m} = \int_0^{R}r^{\ell+2}\rho_{\ell m}(r)\,dr + z\,Y_{00}\delta_{\ell 0}$ with
$z$ the (negative) nuclear point charge; the multipoles $q^{I}_{\ell m}$ of the interstitial
density analytically continued into the sphere are subtracted; and a **pseudocharge** is built
that equals the true density in the interstitial and, inside each sphere, has the smooth radial
shape

$$
\rho^{P}_{\ell m}\,\frac{1}{R^{\ell+3}}\Big(\frac rR\Big)^{\ell}
\Big(1-\frac{r^2}{R^2}\Big)^{N},\qquad
\rho^{P}_{\ell m}=\frac{(2\ell+2N+3)!!}{2^{N}N!\,(2\ell+1)!!}
\big(q^{\rm MT}_{\ell m}-q^{I}_{\ell m}\big),
$$

with $N = \texttt{npsd} = \mathrm{nint}(\tfrac14 R_{\max}\,\texttt{gmaxvr})$
(`init0.f90:483-487`) chosen so the pseudocharge is representable on the finite G-set. Poisson is
then solved directly, $V^{P}(\mathbf G) = 4\pi\rho^{P}(\mathbf G)/G^2$ for $G>0$ and
**$V^{P}(\mathbf 0)=0$** (`gengclg.f90:12`, the neutral-cell convention — and the reason the
Madelung bookkeeping below exists at all). Inside each sphere the true potential comes from the
radial Green's function of the *real* muffin-tin density, plus a homogeneous $(r/R)^\ell$ term
whose coefficients are fixed by matching $V^P$ at the sphere boundary.

**The fixed-point variable is the POTENTIAL, not the density.** `mixrho` defaults to `.false.`
(read, `readinput.f90:110`), so `init0.f90:696-699` sets `vmixer => vsbs`, whose pointer views
are `vsmt(npmtmax,natmtot)` on the *fine* radial mesh, `vsirc(ngtc)` on the **coarse real-space**
interstitial grid, already multiplied by the characteristic function (`rfirftoc.f90` multiplies by
`cfunir` on the fine grid, forward-transforms, truncates to the coarse G-set and
back-transforms), and when spin-polarised `bsmt` on the *coarse* radial mesh and `bsirc`. The
G-space interstitial potential is a **different** array, `vsig`, built from `vsirc` afterwards by
`genvsig.f90` and not part of the mixing vector at all. A port that mixes the density, or that
builds the vector out of `vsir` on the fine grid, or that puts `vsig` in it, gets both the vector
length and the RMS convergence measure wrong while everything still appears to work.

The choice of $v$ rather than $\rho$ is not only a trap to reproduce; it is a design decision for
§8(a), because it fixes the implicit Jacobian: mixing the potential gives
$\mathbb 1 - K_{\rm Hxc}\chi_0$, mixing the density gives $\mathbb 1 - \chi_0 K_{\rm Hxc}$. Same
spectrum, different conditioning, different preconditioners, and different vector lengths
($\rho$ lives on the *fine* interstitial grid `ngtot` and the fine radial mesh; $v$ is coarse for
both the interstitial and $\mathbf B_s$), which directly sets the cost of every GMRES solve in
§8(d). Elk exposes both through one flag and a port should too; this study did not measure which
is better conditioned for LAPW.

Convergence requires **both** $\|v_{\rm out}-v_{\rm in}\|_2/\sqrt n < \texttt{epspot}$ ($10^{-6}$)
and $|E - \tilde E| < \texttt{epsengy}$ ($10^{-4}$ Ha), where
$\tilde E \leftarrow 0.75E + 0.25\tilde E$ is a *smoothed* history — which is what damps a
two-cycle oscillation into a convergence declaration.

**Total energy** (read, `energy.f90`), with $\langle f|g\rangle$ the combined MT + interstitial
inner product:

$$
E_{\rm tot} = T_s + \tfrac12 E_{\rm vcl} + E_{\rm Mad} + E_x + E_c + E_{ts},\qquad
T_s = \sum_{\rm states}n\varepsilon - \langle\rho|v_C\rangle - \langle\rho|v_{xc}\rangle
 - \sum_a\langle m_a|B_{s,a}\rangle .
$$

Here $T_s$ is the Kohn-Sham kinetic energy, $E_{\rm vcl} = \langle\rho|v_C\rangle$,
$E_x$ and $E_c$ the exchange and correlation energies, $E_{ts} = -\texttt{swidth}\cdot S/k_B$ the
smearing entropy term, and $E_{\rm Mad}$ the Madelung energy — the interaction of each nucleus
with the electrostatic potential of everything except its own point charge. Written as
`energy.f90:112` actually computes it,

$$
E_{\rm Mad} = \tfrac12\sum_\alpha \texttt{spzn}_\alpha
\big(v^{C}_{\alpha;00}(r_{\min}) - v^{\rm nuc}_{\alpha;00}\big)Y_{00},
$$

where two conventions must be stated or the sign comes out backwards: Elk's `vclmt` is the
**potential energy of an electron** (negative near a nucleus), and `spzn` is $-Z$ (verified:
`Si.in` gives $\texttt{spzn}=-14$), so this is $-\tfrac12\sum Z_\alpha(\dots)$ in terms of $Z$.
The evaluation point is the *first radial mesh point* $r_{\min}\approx5\times10^{-7}$ Bohr, not a
limit $r\to0$. The sum in $T_s$ runs over core states and over $(\mathbf k, n)$ with weight
$w_{\mathbf k}f_{n\mathbf k}$.

**Two facts about this expression matter enormously for §8.** First, it mixes eigenvalues
obtained from $v_{\rm in}$ with potentials built from $\rho_{\rm out}$, so it is meaningful only
*at* self-consistency and is not a stationary functional of any single argument. Second — verified
here and, as far as this study can tell, not noted in any of the surveys — the entropy term is
computed **only for Fermi–Dirac smearing**: `energy.f90:242` reads `if (stype == 3) then` around
the whole entropy block, so `engyts = 0` for Gaussian and Methfessel–Paxton. The Mermin free
energy is what is stationary with respect to occupation numbers; with `stype != 3` Elk's
`engytot` is not it, and the entire licence for freezing the density when differentiating fails
for those smearings.

### 2.3 Why all-electron + augmentation is structurally harder to port than plane-wave + pseudopotential

Seven reasons, each of which has a concrete consequence later in this document:

1. **Two representations, one of them intrinsically ragged — and the ragged one is a physical
   approximation, not just a storage trick.** Elk truncates angular momentum three different
   ways (`readinput.f90:80-85`): $\ell_{\rm apw}=\texttt{lmaxapw}=8$ for the basis
   ($\texttt{lmmaxapw}=81$); $\ell^{\rm o}_{\max}=\texttt{lmaxo}=6$ for the density and potential
   ($\texttt{lmmaxo}=49$); and $\ell^{\rm i}_{\max}=\texttt{lmaxi}=1$ for the *inner* region
   $r < \texttt{fracinr}\cdot R_\alpha$ with $\texttt{fracinr}=0.01$ ($\texttt{lmmaxi}=4$).
   The last of these says the potential and density near the nucleus are kept only to monopole
   plus dipole — a real approximation the two-block layout inherits, not an implementation
   detail. A muffin-tin field is therefore one flat array of length
   $\texttt{npmt} = 4\,n_{r}^{\rm i} + 49\,(n_r - n_r^{\rm i})$ at defaults, i.e. two different
   $\ell$ truncations concatenated along the radial axis, with per-species radial mesh lengths
   (measured: C 297, Si 397, Fe 497, W 697 points). Flattening to a dense
   $(n_r, \ell_{\max}^{\rm o})$ array costs **2.62x–2.93x** in storage and FLOPs (measured), because
   ~70% of radial points sit in the 4-harmonic inner region. A plane-wave code has one array.
2. **The basis depends on the potential.** `gndstate.f90:156,160,162` call `linengy`,
   `genapwlofr` and `gensocfr` *inside* the SCF loop, so `apwfr`, `lofr`, `haa`, `hloa`, `hlolo`
   and `socfr` are all functions of the converged density. The plane-wave basis is fixed once.
3. **Non-orthogonality.** $H c = \varepsilon S c$ is generalized, which is exactly what JAX does
   not provide. It also complicates the direct-minimisation escape hatch every JAX DFT code so
   far has used to avoid `eigh` entirely: minimising $\mathrm{Tr}[C^\dagger HC]$ subject to
   $C^\dagger SC=\mathbb 1$ is standard and the $S$-metric is a preconditioning question rather
   than an obstruction, so this is a cost, not a closed door. **It was not evaluated here**, and
   neither were iterative subspace eigensolvers (LOBPCG, ChASE, Davidson) — see B4.
4. **Integer-valued solvers inside the loop.** Core states are found by node counting
   (`rdirac.f90`) and linearisation energies by bracketing (`findband.f90`, `maxstp=250`), both
   every SCF iteration, both non-differentiable as written.
5. **Array shapes depend on atomic positions.** See §8(e) hazard D — this is the one that has no
   good fix.
6. **Dynamic range.** Fact 3 of §1. The muffin-tin representation is what forces FP64, and FP64
   is what removes the GPU's advantage.
7. **Cancellation.** $E_{\rm tot}\sim-578$ Ha for two Si atoms and $\sim-16157$ Ha for bcc W
   (measured), converged to $10^{-6}$ Ha: 9–10 significant figures out of differences of
   comparable large terms.

---

## 3. Subsystem map

Effort is for a competent JAX + LAPW developer and assumes the pytree scaffolding (§5) already
exists. Differentiability status uses five values: **DAW** (differentiable as written),
**CVJP** (needs a custom VJP/JVP), **IMPL** (needs implicit differentiation), **REFORM** (needs
the physics reformulated before differentiating), **ND** (not differentiable; freeze it).

### 3.1 SCF spine and program structure

| Elk files | Computes | JAX construct | Static shapes | Diff | Risk | Effort |
|---|---|---|---|---|---|---|
| `gndstate.f90` (402) | the SCF loop; 9 output units, MPI barriers, `STOP`-file polling | Python `while` over a jitted `step(state, basis) -> state`; I/O and convergence test outside the trace | state pytree constant across iterations | **IMPL** | med | 2 d |
| `init0.f90` (824), `init1.f90` (384) | all shape and index arithmetic; the `rhmg`/`vsbs` flat targets | a host-side `derive_shapes(structure, params)` returning a frozen pytree | this *is* the static-shape layer | ND (frozen) | med | 1–2 wk |
| `mixlinear` (34), `mixadapt` (77), `mixbroyden` (103), `mixerifc` (65) | potential mixing; Broyden history with `dgetrf`/`dgetri` | `jnp` arithmetic; fixed `(msd, n)` history + validity mask; `jnp.linalg.solve` not `inv` | `msd=5` static; warm-up `m=min(iscl+1,msd)` needs masking | **ND — and must be excluded from the gradient** | low | 1 d |
| `occupy.f90` (134) | Fermi level by bisection; smeared occupations; band-gap scan | `lax.custom_root` on the scalar charge constraint | fully static | **CVJP** (analytic $d\mu/d\varepsilon$) | low | 2 d |
| `findswidth.f90` (65) | `autoswidth`: mutates `swidth` from the spectrum | refuse it (off by default) | — | ND | low | — |
| `energy.f90` (268) | total energy assembly | `jnp` reductions | static | **REFORM** — see §2.2 and §8(b) | med | 3 d |
| `potks.f90` (63), `genvsig.f90` (30) | KS potential composition; one FFT + gather | composition; `jnp.fft.fftn` + `take` | static | DAW | low | 1 d |
| `rhoinit.f90` (141), `maginit.f90` (48) | atomic-superposition start | vmapped Bessel transform + structure-factor scatter | pad species radial range, mask | DAW | low | 3 d |
| `readstate`/`writestate` (504) | state persistence with regridding | checkpoint pytree; regridding as an explicit out-of-jit function | shapes change across a regrid by construction | ND | low | 1 wk |
| `modramdisk.f90` (332), `checkstop`/`checkwrite` (43) | in-process file emulation; filesystem polling | **not ported** (§7); a Python callback between traced steps | n/a | n/a | low | — |
| `elk.f90` (2143), `readinput.f90` (2353) | task dispatch; 316 input-block `case` clauses accepting 365 names (aliases included) | a Python dict and a dataclass | n/a | n/a | low | 3 d |
| `allatoms` (73), `atom.f90` (135), `readspecies` (277), `genspecies` (233) | the free-atom solver: species-file configuration, atomic potential; feeds `rhoinit`'s superposition start **and** `gencore`'s $r>R_{\rm MT}$ potential tail | host-side NumPy setup emitting frozen per-species tables | n/a | ND (constants) | low | 1 wk |
| `spline`/`splint*`/`wspline`/`wsplint*`/`fderiv`/`polynm`/`hermite` (625) | the quadrature and spline weights `wr2mt`/`wprmt`/`wcrmt` that §5's `Basis` treats as given | host-side NumPy; the weights become frozen arrays | n/a | ND (constants) | low | 3 d |
| `zfftifc_fftw`/`cfftifc_fftw`/`rzfftifc`/`nfftifc`/`gridsize` (233) | FFT interface; $\{2,3,5,7\}$-smooth grid sizing; the **packed** real-to-complex representation (`nfgrz`/`igrzf`) that `trimrfg` and the GGA path index directly | `jnp.fft` is *not* a one-to-one map: the packed real form has no `jnp.fft` equivalent and must be reimplemented or abandoned in favour of a full complex grid | grid sizes are host-computed integers | DAW | low | 1 wk |
| `modomp.f90` (78), `modmpi.f90` (73) | the parallel model: a dynamic nested-thread budget (`holdthd`/`freethd`) and MPI k-point distribution with per-k file writes | **deleted**; replaced by XLA's own scheduling and by §8(d)'s `lax.scan` over k-chunks. Note `holdthd`'s nested budget is the direct cause of the 83x timing artefact in §10 item 4 | n/a | n/a | low | — |

### 3.2 Basis, Hamiltonian and eigensolver

| Elk files | Computes | JAX construct | Static shapes | Diff | Risk | Effort |
|---|---|---|---|---|---|---|
| `gengkvec`, `findngkmax`, `gensfacgp` (186) | the per-k $\mathbf G+\mathbf k$ list; structure factors | host-side mask + slot scatter into `(nk, ngkmax)`; `exp(i vgkc @ atposc.T)` | Elk **already pads** to `ngkmax`; convert `ngk` to a boolean mask | DAW | low | 3 d |
| `sbessel.f90` (116), `genylmv.f90` (136) | $j_\ell$ (3 branches on $x$), $Y_{\ell m}$ recursion | evaluate all branches + `jnp.where`; `lax.scan` over `sbessel`'s Miller downward recursion, whose *start index* is $\texttt{lst}=\ell_{\max}+\ell_{\max}/8+14 = 23$ at the default $\ell_{\rm apw}=8$ (`sbessel.f90:72`), and separately over `genylmv`'s own $\ell$ axis (`do l=2,lmax`, length $\ell_{\rm apw}-1$) | $\ell_{\max}$ is a Python int | DAW | low | 1 wk |
| `match.f90` (157) | matching coefficients $A$ | one fused `einsum`; `apword=1` fast path | fully static | **DAW — and this is where Pulay comes from** | low | 1 wk |
| `rschrodint.f90` (147) | radial IVP at fixed $E$ | `lax.scan` over $r$, 4-point history carry, 8-pass corrector unrolled | pad `nrmt` per species, mask weights | DAW (double-`where` the overflow guard) | med | 3 wk |
| `rdiracint` (148) + `rdirac.f90` (142), `gencore.f90` (142) | core states by node-count shooting | `fori_loop(100)` shots over the scan; or keep on host | static | **CVJP** — $dE_c/dV = g_c^2+f_c^2$, exact from 1st-order PT | med | 1 wk |
| `findband.f90` (118), `linengy.f90` (134) | linearisation energies by bracketing | replace the algorithm: vmapped grid scan + fixed bisection | trip count fixed | **ND** — freeze $E_\ell$; state it as an approximation | high | 1 wk |
| `genapwfr` (137), `genlofr` (165), `hmlrad` (178), `olprad` (67) | radial functions, Gram–Schmidt, radial integrals | vmapped scans + unrolled GS; two `einsum`s per atom | keep inner/outer as **two** dense blocks | DAW (`dgesv` needs identity-padding) | low | 2 wk |
| `hmlaa/hmlalo/hmllolo/olpaa/olpalo/olplolo` (245), `hmlistl/olpistl` (96), `zmctmu` (33) | H and S blocks | two GEMMs per atom; precomputed `(ngkmax,ngkmax)` int index + two `take`s | pad to `nmatmax`; drop the $10^{-12}$ sparsity skip | DAW | low | 3–4 wk |
| `eveqnfv` (86), `zhegvxi` (47) | generalized eigenproblem, lowest `nstfv` | **hand-written Cholesky reduction** + full `jnp.linalg.eigh` | pad H/S: $S_{\rm pad}=I$ and $H_{\rm pad}=\mathrm{diag}(E_{\rm big}+j\delta)$ with **distinct** $\delta$-spaced values, so spurious eigenvalues sort away *without* manufacturing an exact degeneracy. $H_{\rm pad}=E_{\rm big}I$ is the obvious choice and is wrong: measured, a projector gradient through a bitwise-degenerate pad block returns `NaN` even for a non-degenerate physical spectrum (§8b, hazard A) | eigenvalues **DAW only when multiplet-summed** (§8b); eigenvectors **ND**; projector **CVJP** | **high** | 2–3 wk |
| `eveqnfvr` (264), `dsygvxi` (41) | real-symmetric path for centrosymmetric crystals (**default**) | same reduction in `float64`; the recombination is a fixed real similarity | static index/sign tables | DAW | med | 1 mo (defer) |
| `eveqnsv.f90` (387) | second variation: spin, $B_{xc}$, SOC, DFT+U | wavefunction expansion + 3 GEMMs + `jnp.linalg.eigh` | `nstsv` static; MT split preserved | as `eveqnfv` | med | 4–6 wk |
| `genevfsv.f90` (47) | the k-loop, writing eigenvectors per k to disk | **deleted**: `lax.map` over k-chunks with `vmap` inside; density accumulator as carry | — | n/a | low | 1 wk |

### 3.3 Muffin-tin machinery, density, potential, XC

| Elk files | Computes | JAX construct | Static shapes | Diff | Risk | Effort |
|---|---|---|---|---|---|---|
| `genrmesh` (131), `sphcover` (56), `genshtmat` (145), `gaunt`/`gauntyry`/`wigner3j` | meshes, quadrature weights, SHT matrices, Gaunt tensor | **not ported to JAX**: NumPy host setup emitting a frozen pytree | n/a | ND (constants) | low | 1 wk |
| `{c,r,z}{b,f}sht` + `*ip` (11 files, 325) | spherical-harmonic transform | two batched matmuls, $(4,4)$ and $(49,49)$ | static | DAW | low | 2 d |
| `gradzfmt` (210), `grad2rfmt` (80), `gradrfmt` (55) | MT gradient (radial spline + Clebsch–Gordan recoupling + Cartesian rotation) | three precomputed linear operators, ~10 lines of `einsum` | two operator sets (inner/outer) | DAW | med | 1 wk |
| `rfmtctof`/`rfirctof`/`rfirftoc` (171) | coarse↔fine interpolation | a fixed $(n_r, n_{rc})$ matrix per species; FFT zero-pad/truncate | static | DAW | low | 2 d |
| `rhomagk` (225), `rhomagv` (118), `wfmtsv` (200), `wfirsv` (85) | density and magnetisation from occupied states | fuse radial expansion + SHT into one per-species matrix $W$; **local density matrix** $\rho_y=\sum_k w\sum_n f\,yy^\dagger$ (49×49/atom) then $\rho_{\rm mt}=\mathrm{diag}(W\rho_yW^\dagger)$ | occupied count is data-dependent → fixed `nocc_max` + weight mask | DAW (`zdotu` is **unconjugated**) | med | 3 wk |
| `zpotcoul` (248), `zpotclmt` (128), `genzvclmt` (31) | Weinert Poisson solver | `cumsum` of a `conv1d` (**not** `lax.scan`); dense G-space contractions | static | DAW | med | 2 wk |
| `xc_*`, `x_*`, `c_*`, `k_*` (**1425 lines**, measured) | Elk's native functionals: PZ, PW92 (default), Xα, vBH, PBE/revPBE/PBEsol, WC06, AM05, TF/TFvW | reimplement the **scalar** $\varepsilon_{xc}$ only; `jax.grad` for $v_{xc}$, `jax.hessian` for $f_{xc}$ | elementwise | DAW (**double-`where` every `rho<1e-12` guard**) | low | 3 wk |
| `libxcifc.f90` (564) + libxc | **everything beyond LDA/GGA**: all meta-GGAs (`xcgrad` $\in\{3,4,5,6\}$), all hybrids, and Tran–Blaha. Elk's *native* set stops at `xcgrad` $\in\{-1,0,1\}$ (`modxcifc.f90:435-495`); `xc_c_tb09.f90` computes only TB09's $c$ constant, the functional itself is libxc's | the C library is unavailable inside a trace, but `jax_xc` covers much of it (B7) | n/a | **ND as libxc**; see B7 | med | — |
| `gentau.f90` + `taumt`/`tauir`/`taucr`/`wxcmt` plumbing | kinetic-energy density $\tau$, allocated and computed only when `xcgrad` $\in\{3,4,5,6\}$ (`potks.f90:35`, `init0.f90:660-670`) | needed **only if** meta-GGAs are wanted via `jax_xc`. $\tau$ is orbital-dependent, so $v_{xc}$ stops being a local functional derivative of $\rho$ and the velocity operator stops being $\hat{\mathbf p}$ — which matters directly for §§22/26 | occupied count as in `rhomagk` | DAW, but a genuine plumbing cost | med | 2 wk if wanted |
| `ggamt_*`, `ggair_*` (**1240 lines**, measured) | GGA gradient plumbing for two incompatible conventions | **deleted** — both conventions exist only because $v_{xc}$ was hand-derived | n/a | n/a | low | — |
| `potxcmt` (352), `potxcir` (284), `trimrfg` (12), `projsbf` (48) | XC driver; Kübler non-collinear rotation; the $\lvert G\rvert>2g_{\max}$ trim of $v_{xc}$; optional removal of $\mathbf B_{xc}$'s source term (`nosource`) | specialise the branch at trace time; `trimrfg` is an FFT mask; refuse `nosource` in phase one | static | DAW (**floor `sqrt(m·m)` unconditionally**, not only for GGA — but see below: the floor is a change of physics at $\lvert\mathbf m\rvert\to0$, not only a `NaN` guard) | med | 2 wk |
| `symrf`/`symrfir`/`symrfmt`/`symrvf`/`symrvfir`/`symrvfmt` (560) | symmetrisation | precomputed `(nsym, lmmax, lmmax)` Wigner-D + static gather; `einsum` | `nsymcrys` static per structure | DAW *given a frozen group* | low | 1 wk |
| `findsymcrys`/`findsym`/`findsymlat` (586) | symmetry determination; **mutates `atposl`** | **not ported** — spglib on the host, once | n/a | **ND** | low | 3 d |

**One physics note the table cannot carry.** The Kübler treatment (`potxcmt.f90:120-134`) is an
*approximation*, not a coordinate trick: $\mathbf B_{xc}$ is taken parallel to $\mathbf m$ at each
point, so a collinear $\varepsilon_{xc}$ is evaluated in the local spin frame. It has a genuine
non-analyticity where $|\mathbf m|\to0$ — which is where every antiferromagnetic node and the
whole vacuum region sit. Elk floors $\sqrt{\mathbf m\cdot\mathbf m}$ by `dncgga` in the GGA branch
and *not* in the LSDA branch. Flooring it unconditionally, as hazard H recommends, changes the
physics at exactly those points; that is the right trade for a differentiable code, but it must
be stated as a change and converged in the floor, not treated as a `NaN` guard.

### 3.4 Derived quantities (the subsystems autodiff is supposed to replace)

| Elk files | Computes | JAX construct | Static shapes | Diff | Risk | Effort |
|---|---|---|---|---|---|---|
| `force.f90` (241) | Hellmann–Feynman + core correction + MT integrals + `symveca` | partly `jax.grad`; the $\int_{\rm MT}v_s\nabla\rho$ and $\int_{\rm MT}\mathbf B_s\!\cdot\!\nabla\mathbf m$ blocks (lines 142–178) must be **ported**, not derived (§8c); `symveca` (50) survives as a constant projector | frozen G-set, `reducek=0` | **REFORM** — the recipe *and its frame* decide the answer (§8c) | high | 2 wk |
| `forcek.f90` (173) | incomplete-basis (Pulay/IBS) force per k | reverse VJP of the band energy | pad to `nmatmax` | REFORM | high | — |
| `genstress.f90` (52), `genstrain.f90` (66) | stress by **finite difference** over up to 7 full SCF runs | `jax.grad` w.r.t. a 3×3 strain on `avec` | freeze integer `ivg`, vary `vgc = ivg @ bvec(ε)` | REFORM (fixed-basis stress ≠ Elk's) | med | 1 wk |
| `geomopt.f90` (203), `atpstep` (59), `latvstep` (36) | sign-based damped steepest descent | Python + optax; do not jit the outer loop | n/a | n/a | low | 3 d |
| DFPT: `phonon` (279), `dforce` (155), `dforcek` (305), `deveqnfv` (108), `drhomagk` (233), `dhmlrad` (118), `dpotks`/`dpotxc` (187), `dmatch` (26), `doccupy` (51), `modphonon` (160), + G+q machinery | phonons and response | **$q=0$: `jax.hessian` over the implicit fixed point. $q\neq0$: NOT replaced** | k+q is a second ragged G-set | REFORM | high | 6–9 mo if ported |
| `modfxcifc`/`genfxcr`/`genspfxcr`/`genspfxcg`/`fxc_pwca` (**672 lines**, measured) | the XC kernel $f_{xc}$, hand-coded | `jax.hessian` of $\varepsilon_{xc}$ | elementwise | n/a | low | — |
| `phononsc.f90` (186) + supercell drivers | frozen phonons by $6N_{\rm sc}$ SCF runs | supercell Hessian: $3N_{\rm sc}$ linear solves | n/a | REFORM | med | — |

**Effort arithmetic, stated so it can be checked.** Summing every effort column in §§3.1–3.4 at
five-day weeks, and excluding the deferred `eveqnfvr` month and the 6–9-month DFPT row, gives
roughly
**45 weeks $\approx$ 10.4 months** — comparable to, not much smaller than, §6's phase total
(9.5–12.75 months to Phase 4). Both exclude the same thing: §5's scaffolding, which B3 calls the
dominant cost. That is now §6's Phase S.

### 3.5 Surveyed, and deliberately out of scope

These subsystems are real Elk code that a port would have to decide about. The decision here is
to leave every one of them in Fortran, reached through a checkpoint (§7). Listing them makes that
a decision rather than a gap. File counts measured by name over `vendor/elk/src/`.

| Subsystem | Files / lines | Why out of scope |
|---|---|---|
| Real-time TDDFT (`modtddft`, `tddft`, `timestep`, `tdinit`, `energytd`, `tdrestart`, `writetd*`) | 15 / 1652 | its own time propagation, not a ground state; the traced object is a trajectory |
| GW / Dyson (`modgw`, `gwsefm*`, `dysonr*`, `epsinv*`, `gwrhomag`, `gwefermi`, `gwdmat`, `acgwse`, `pade`) | 12 / 1068 | its own self-consistency; note `gwrhomag` is reachable *from inside* `gndstate` via `ksgwrho` |
| Electron–phonon (`ephcouple`, `genephmat`, `eveqneph`, `hmleph*`, `eliashberg`, `alpha2f`, `mcmillan`, `ephdos`, `modbog`, `gndsteph`) | 12 / 1458 | needs $q\neq0$ DFPT, which is explicitly not replaced (§7) |
| DFT+U / density-matrix / fixed tensor moments (`moddftu`, `genfdu*`, `genveedu`, `vmatmtdu`, `engyfdu`, `gendmatmt`, `genvmatmt`, `rotdmat`, `symdmat`, `dmatls`, `dmatuv`, `tm2todm`, `tm3todm`, `vmatmtftm`, `writedftu`, `writetm*`, `readdftu`) | 18 / 1507 | `vmatmt`/`dmatmt` are a **second, unmixed fixed-point variable** (§2.2), so this is a change to the fixed-point equation, not an added term |
| BSE | 8 / 842 | post-SCF response |
| Spin spirals (`eveqnss`, `genscss`, `spiralsc`, `ssfext`, `sstask`, `deveqnss`) | 5 / 419 | generalised Bloch condition; same periodicity problem as $q\neq0$ DFPT |
| Molecular dynamics (`moldyn`, `atptstep`, `afindtstep`) | 3 / 199 | a driver over ground states; belongs in Python if wanted |
| RDMFT (`rdmft`, tasks 300–), ULR (`gndstulr`, tasks 700/701), Hartree–Fock (task 5) | — | each runs its own self-consistency |
| Electric-field / vector-potential machinery (`potefield`, `efieldmt`, `genafieldt`, `genefieldt`, `potdmag`) | — | produces terms `energy.f90:229-236` and `potks.f90` do evaluate; phase one refuses them (§5) |

The load-bearing point is that the first three and the last three of these are **producers** — they
run their own self-consistency or propagation — so they are not covered by §7's "consumers of the
converged arrays" argument. §7 now says so.

---

## 4. Blockers, ranked

Ranked by whether they decide the outcome, not by how often they are mentioned.

### 4.1 Decides the outcome

**B1. Eigenvector, projector *and individual-eigenvalue* derivatives are undefined at
degeneracies and silently wrong near them.**
Evidence: JAX's `_eigh_jvp_rule` forms `Fmat = (Δw + I)^{-1} - I`; at a degeneracy `Δw` is zero
off-diagonal. Measured, at a controlled near-degeneracy the gradient of a single-eigenvector
observable ran $2.3\times10^{1}\to2.3\times10^{3}\to2.3\times10^{5}\to2.3\times10^{7}\to
2.3\times10^{11}$ as the splitting shrank $10^{-2}\to10^{-12}$, and $4.3\times10^{14}$ at exact
degeneracy. Measured, the *naive projector* — the quantity everyone calls safe — still carries
$2.08\times10^{-2}$ relative error at exact degeneracy, because `eigh`'s VJP divides before the
$(f_i-f_j)$ numerator can cancel. Measured, a $10^{-16}$ reassembly jitter moves that gradient by
**159%**, while a correctly-ruled one moves by $4.3\times10^{-15}$. Measured, an *individual* eigenvalue derivative is not safe either: at an exact two-fold
degeneracy AD returned $(-0.204,-0.793)$, central FD of the sorted spectrum returned their
average $(-0.498,-0.498)$, and the true one-sided directional derivatives were $(-1.553,+0.557)$
— three finite, plausible, mutually inconsistent answers, with only the multiplet trace
($-0.9966407299$) agreeing to machine precision across all three. Elk itself concedes the
singularity: `deveqnfv.f90:96-103` drops resolvent terms with $|\varepsilon_j-w_i| < \texttt{epsdev} =
0.0025$ Ha (68 meV — a wide window). Standard fixes exist for the projector (§8b); the residual
open items are that a window boundary inside a multiplet is genuinely ill-posed and the fix makes
it *look* fine, and that no fix makes an individual band's derivative meaningful inside a
multiplet.

**B2. Second-order differentiation through a fixed point containing `eigh` is untested; the
first-order reverse question that used to sit here is settled.** Measured on JAX 0.7.1:
`jax.custom_vjp` around the SCF with GMRES on the transposed operator gives a reverse-mode
gradient matching central finite differences to $3.4\times10^{-10}$ on a 6-dimensional nonlinear
fixed point — the pattern jaxopt, optimistix and PySCFAD's own `implicit_diff.py` all use, and it
needs no transposition of the iterative solver at all. What does fail is one specific
composition: `lax.custom_root` with `jax.scipy.sparse.linalg.gmres` as `tangent_solve` raises
`NotImplementedError` under `jax.grad` (measured; the guard is in
`jax/_src/lax/control_flow/solves.py`'s `_linear_solve_transpose_rule`), because nesting one
`custom_linear_solve` inside another's JVP transposition needs cotangents with respect to the
operator's closed-over constants. The same `custom_root` with a *dense* `tangent_solve`
transposes fine and agrees with the `custom_vjp` route to the last digit. So a port must use
`custom_vjp` with an explicit adjoint solve, not `custom_root`'s auto-transposition — an
engineering choice, not a gate.

The open question is second order, which §8(f) items 2, 3 and 8 and all of Phase 5 depend on.
Measured, `jax.hessian` through a `custom_vjp` fixed point *works* and matches finite differences
of the implicit gradient to $2.5\times10^{-9}$, including when the primal solve is hidden behind
a `lax.while_loop` so there is no unrolled tape. But the mechanism matters: the outer forward
pass differentiates the *solver* and the *backward rule*, never the `custom_vjp` primal. At Elk
scale that outer JVP therefore passes through `eigh`'s derivative at every SCF iteration, at every
k-point. Two consequences neither of which has been tested: the safe-$K$ projector rule of §8(b)
must itself be twice-differentiable, and §3.2's padding block must carry distinct eigenvalues or
the second derivative meets a bitwise degeneracy on iteration one. **The toy that established the
first-order result contains no eigensolve, and is therefore not evidence about this.**

**B3. Elk's mutable global state.** 651 of 821 files (79%, measured) contain `use modmain`;
`modmain.f90` is 1297 lines of pure declaration (zero `contains`, no executable statement)
holding 654 module-level names of which 159 are allocatable or pointer (re-counted here,
measured). Every Elk subroutine is a procedure over hidden state; JAX needs pure functions of
explicit arguments. There is no incremental path that keeps the Fortran. **This does not kill the
project; it sets its cost** — and it appears in *neither* effort estimate: §3's per-row figures
assume the scaffolding exists and §6's phases start after it. It now has its own line, Phase S,
whose estimate is an inference (survey 6 said 2–3 person-years for the spine) and not a
measurement.

### 4.2 Real, with known and bounded workarounds

**B4. No generalized Hermitian eigensolver in JAX, and no eigenvalue-subset selection.**
`jax/_src/scipy/linalg.py` raises `NotImplementedError("Only the b=None case of eigh is
implemented")` (issue #5461 open since January 2021); `jax/_src/lax/linalg.py` raises
`NotImplementedError("subset_by_index not supported on CPU and GPU")`. Workaround verified:
$L=\mathrm{chol}(S)$, $C=L^{-1}HL^{-\dagger}$, `eigh(C)`, back-transform — measured to reproduce
`scipy.linalg.eigh(A,B)` eigenvalues to $2.6$–$3.3\times10^{-16}$ with $B$-orthonormality
$1.1\times10^{-15}$, and it jits and vmaps. The subset gap costs a bounded factor: measured
2.68x at $n=600$/$m=60$ and 4.31x at $n=1200$/$m=120$ single-threaded; a second independent
measurement gave 3.57x for the whole Cholesky route at $n=920$/$m=49$. Extrapolation to
$n\approx3000$ is *inference*, not measurement.

**That factor is a floor on the differentiable path, not a performance tax a future upstream fix
removes.** `_eigh_jvp_rule` opens with
`raise NotImplementedError("Derivatives not defined for partial eigen decomposition.")` for any
`subset_by_index` other than the full range (read, JAX 0.7.1) — so even when subset selection
lands, the subset path will not be differentiable. Whenever a gradient is wanted, the full
spectrum must be computed.

Three further consequences worth naming. `jnp.linalg.cholesky` returns `NaN` rather than an error
on a near-singular $S$. The Cholesky reduction also sets the eigenvalue backward error at
$\sim\epsilon\,\kappa(S)\,\|H\|$, not $\epsilon\|H\|$, which is what §8(b)'s degeneracy tolerance
must be derived from — **and $\kappa(S)$ for Elk's APW+lo overlap was never measured here**, so
the tolerance should be computed at runtime from the Cholesky factor's own diagonal rather than
assumed. (An earlier draft cited `genapwfr.f90:86`'s $10^{-25}$ guard as evidence that the LAPW
overlap is near-singular by construction. That is wrong on two counts: the guard sits inside
`if (io > 1)` and concerns Gram–Schmidt orthogonalisation of *APW radial functions* against each
other, saying nothing about $S$; and it is unreachable at the $\texttt{apword}=1$ of every
shipped species file.) Finally, alternatives were not evaluated: **no iterative subspace
eigensolver (LOBPCG, ChASE, Davidson) was benchmarked**, which is what production LAPW codes use
for the lowest-$n_{\rm stfv}$ problem and which would move the factor the other way at
$n\approx3000$; and Elk's own history cuts both ways — it *removed* its iterative
first-variational solver (`readinput.f90:1371` still prints "variable 'tefvit' is no longer
used"), but it ships `zhegvxsi`, a diagonal-sorted subspace *reduction* before a dense solve
(off by default, `mefvs=-1`), which a port could reuse cheaply.

**B5. Ragged muffin-tin shapes under `jit`.** The packed array carries two $\ell$ truncations
concatenated, with per-species radial lengths. Dense padding costs **2.62x–2.93x** (measured
across C, Si, Fe, W). The recommended layout — two rectangular blocks per species, padded only
along the radial axis, one `jit` per species — is *better* than Elk's own packing, not merely
tolerable. Elk has already done the harder half of the ragged-to-dense conversion for the k-axis:
`igkig`, `vgkl`, `vgkc`, `sfacgk` and `evecfv` are already allocated to `ngkmax`/`nmatmax`, so
what a port adds is a *mask*, not a padding scheme (measured spread: Si diamond $n_{\mathbf
G+\mathbf k}\in[137,156]$, h-BN $[971,1060]$ — a few percent).

**B6. complex128 GPU throughput.** H100 PCIe: FP64 26 TFLOPS / FP64 tensor 51 vs FP32 51
(cited — note the halving applies to the non-tensor FP64 path, not the tensor one); consumer
cards 1/32–1/64. Later generations are reported to cut FP64 much harder (*reported by review,
unverified here*), which would strengthen this. A complex128 GEMM is four real FP64 GEMMs. Mixed
precision is possible in the forward pass — Elk itself already uses `complex(4)`, with
`cgemv`/`sdot` in `eveqnsv` and `cgemm` in `wfmtsv_sp` and the single-precision SHT — but **not
in the gradient**: measured, `complex64` flips
the sign of an eigen-derived gradient at a $10^{-6}$ Ha splitting where `complex128` is still
correct, and float32 $\epsilon = 1.2\times10^{-7}$ makes every splitting below $\sim10^{-5}$ Ha
(0.27 meV, i.e. essentially every SOC or valley splitting this repository studies)
roundoff-degenerate. A differentiable port is pinned to datacentre hardware.

### 4.3 Real, narrower in scope

**B7. libxc's C path has no JAX and no gradient — but the loss is much narrower than that
sounds.** `xctype=100` dispatches to a C library called elementwise on the host, which cannot run
inside a trace. Elk's *native* set (1425 lines, measured) is `xcgrad` $\in\{-1,0,1\}$ only, i.e.
LDA and the GGA workhorse range (`modxcifc.f90:435-495`); every meta-GGA, every hybrid and
Tran–Blaha reach Elk through libxc alone. **`jax_xc` covers most of them.** Checked directly
against the published wheel (`jax_xc` 0.0.11): 628 functionals translated from libxc's own Maple
source, including `mgga_x_scan`, `mgga_c_scan`, `mgga_x_tpss`, `mgga_x_r2scan`, `mgga_x_tb09`,
`hyb_gga_xc_b3lyp` and `hyb_gga_xc_hse06` — i.e. every functional this blocker previously named
as unavailable. So the correct statement is: the *library binding* is lost, not the functionals.

Two real costs survive. Meta-GGAs need the kinetic-energy density $\tau$, which is LAPW plumbing
(`gentau`, `taumt`/`tauir`/`taucr`, `wxcmt`) independent of the functional library — a §3.3 row,
not a free import. And Tran–Blaha is a **potential-only** functional with no parent $E_{xc}$, so
the "reimplement scalar $\varepsilon_{xc}$, `jax.grad` for $v_{xc}$" scheme is *structurally*
inapplicable to it, not merely unimplemented; its total energy is not stationary either, so every
§8(c) force recipe fails under it. Mitigating fact: libxc is already optional and **this build
links `libxcifc_stub.f90` (71 lines)**, so nothing currently in elkpy's test suite depends on it.
*Not checked: `jax_xc`'s own exclusion list, and whether its `mo`-based $\tau$ interface maps onto
an LAPW muffin-tin/interstitial partition.*

**B8. File-mediated dataflow.** `genevfsv` writes `EVECFV`/`EVECSV` per k and every consumer
reads them back; `modramdisk.f90` (332 lines) is a hand-rolled in-process file emulator existing
only because Fortran cannot hand arrays between subroutines without globals or files. Deleted
rather than translated. One thing must survive the deletion: `getevecfv.f90` also applies the
crystal **symmetry rotation** when the requested k lies outside the reduced set — real physics
hidden in an I/O routine, on which DFPT and this repository's own patches 0008/0010 depend.

**B9. 515 bare `stop` statements across 171 files** (measured here,
`grep -cE '^\s*stop\b' *.f90`; an earlier survey's "538 across 183" is not reproducible and is
dropped). Inside `jit` there is no `stop`, and
`jnp.linalg.eigh`/`cholesky` return `NaN` rather than an error code. Host-side preconditions plus
an explicit finiteness check after each step; `checkify` where an in-trace assertion is worth the
compile cost.

**B10. Batched `eigh` may not batch well enough to matter.** Measured on CPU: `vmap(eigh)` over 8
matrices of $n=400$ gave 1.03x over a serial loop. The GPU behaviour is *unverified here* — no
CUDA `jaxlib` was available — and the review process reports that recent `jaxlib` dispatches
`gpusolverDnXsyevBatched` (a real batched kernel) above $n=32$ on CUDA $\ge$ 12.6.2, rather than
only the small-$n$ Jacobi kernel this blocker originally assumed. Even taking that at face value,
the batch is shallow at LAPW sizes: the chunk is $\texttt{INT\_MAX}/(n^2\cdot16)\approx14$
matrices at $n=3000$ in complex128, device memory binds first, and $n=3000$ performance is
unmeasured by anyone. So the design conclusion is unchanged — `vmap` over k buys the assembly and
the density contraction, the eigensolve at best a shallow batch, and the k-axis should be
`lax.map` over chunks — but the *reason* is memory and chunk depth, not a host loop.
**This is still the first benchmark any real port should run.**

---

## 5. The global-state redesign

`modmain.f90` is pure declaration with no executable code, organised into ~30 commented sections.
Those sections *are* the dataclass design, already done by Elk's authors in the wrong language.
The port's job is to split them along one axis Elk does not distinguish: **what is static at
trace time** versus **what is a traced array**.

```python
# ---- static: closure constants, never traced, changing them retraces ----
@dataclass(frozen=True)
class Config:      # readinput.f90's 316 input-block cases (365 accepted names with aliases)
    xctype: tuple; stype: int; swidth: float; mixtype: int; msd: int
    epspot: float; epsengy: float; maxscl: int; spinpol: bool; spinorb: bool
    gmaxvr: float; rgkmax: float; lmaxapw: int; lmaxo: int; lmaxi: int
    reducek: int; tshift: bool; ...     # one block often sets several fields

@dataclass(frozen=True)
class Shapes:      # init0/init1's extent arithmetic — the whole static-shape layer
    nspecies: int; natmtot: int; nrmt: tuple; nrmti: tuple; nrcmt: tuple
    lmmaxi: int; lmmaxo: int; lmmaxapw: int; apwordmax: int
    ngkmax: int; nlotot: int; nmatmax: int; nstfv: int; nstsv: int
    ngridg: tuple; ngtot: int; ngdgc: tuple; ngtc: int; ngvec: int; ngvc: int
    nkpt: int; nsymcrys: int; nocc_max: int

@dataclass(frozen=True)
class Symmetry:    # frozen group: spglib on the host, once
    symlat: Array; symlatc: Array; ieqatom: Array   # integer tables
    wigner_d: Array          # (nsym, lmmaxo, lmmaxo), precomputed ONCE
    g_perm: Array; g_phase: Array                   # (nsym, ngvec)

# ---- geometry-derived tables: traced only if geometry gradients are wanted ----
@dataclass
class Basis:
    avec: Array; bvec: Array; omega: Array; atposc: Array   # <- traced for forces/stress
    ivg: Array                    # INTEGER G-list: frozen even under strain
    vgc: Array; gc: Array; gclg: Array; ylmg: Array; sfacg: Array; ffacg: Array
    igfft: Array; igfc: Array; ivgig: Array; cfunir: Array
    vkl: Array; wkpt: Array; igk_idx: Array; gk_mask: Array  # (nk, ngkmax)
    rmt: Array; rlmt: Array; wr2mt: Array; wprmt: Array; wcrmt: Array
    gntyry: Array                 # (49, 81, 81) complex, structure-INDEPENDENT
    sht_i: Array; sht_o: Array    # (4,4) and (49,49), forward = numerical inverse

# ---- traced state: the SCF variables ----
@dataclass
class Field:                      # one muffin-tin + interstitial field
    mt_in:  Array   # (natmtot, nri_max, lmmaxi)   <- do NOT flatten to lmmaxo
    mt_out: Array   # (natmtot, nro_max, lmmaxo)
    ir:     Array   # (ngtot,) or (ngtc,)

@dataclass
class State:
    v: Array          # THE fixed-point vector: flat vsbs (vsmt fine | vsirc coarse | bs...)
    rho: Field; mag: tuple[Field, ...]
    apwfr: Array; lofr: Array; haa: Array; hloa: Array; hlolo: Array
    oalo: Array; ololo: Array; socfr: Array        # basis: INSIDE the fixed point
    apwe: Array; lorbe: Array                      # linearisation energies: stop_gradient
    evalsv: Array; occsv: Array; efermi: Array
    rhocr: Array; evalcr: Array                    # core: custom_vjp
    mixer: MixerState                              # EXCLUDED from the gradient
    iscl: int; etp: float; dv: float
```

Three decisions carry the design:

1. **`Config` and `Shapes` are closure constants, not traced leaves.** `maxscl`/`mixtype`/`stype`
   branches then resolve at trace time rather than through `lax.cond`. Retracing on a shape change
   (a different `ngridk`, a new `rgkmax`) is correct and cheap relative to a DFT run.
2. **`State.v` keeps Elk's flat layout, with named views.** The Fortran pointer association is a
   struct-of-slices by another name, and the mixer wants one contiguous vector. But the *contents*
   must be right: `vsmt` fine-radial, `vsirc` on the **coarse real-space** grid already multiplied
   by `cfunir`, `bsmt` coarse-radial. `vsig` — the G-space interstitial potential `genvsig`
   derives from `vsirc` — is **not** in this vector.
3. **`Field` is two rectangular blocks, not one packed array.** This is the single most consequential
   layout choice in the port (§4 B5), and every other subsystem inherits it — which is why the
   scaffolding must be designed before anything else is written.

**Things that look like `Config` but are mutated and must live in `State`** (enumerated for the
ground-state path at the scope of §2.2, i.e. `dftu=0`, `ftmtype=0`, `fsmtype=0`): `apwe`/`lorbe`
(rewritten each iteration by `linengy`), `swidth` under `autoswidth`, `bfieldc`/`bfcmt`
(multiplied by `reducebf` each iteration), `rhosp` (deallocated inside `rhoinit`), `c_tb09`,
`filext`, and the loop scalars `iscl`/`tlast`/`tstop` — `iscl` read as a global by `occupy`
(`if ((autoswidth).and.(iscl > 1))`), `linengy` (inside a warning format), `rhonorm`, `gentau`
and `mixerifc`, which passes it *as an argument* to the three mixers, so those are already pure
in it (`mixlinear`/`mixadapt` are declared `pure`); `tlast` read by `energy` and `init0`;
`tstop` by `checkstop` and `gndstate`. (`genevfsv` reads none of the three.) Outside that scope,
add `vmatmt`/`dmatmt` (DFT+U/FTM — an unmixed second fixed-point variable) and
`bfsmc`/`bfsmcmt` (the FSM integral controller); §2.2 explains why those change the fixed-point
equation rather than adding a term. Phase one should refuse `autoswidth`, `reducebf`, DFT+U, FTM,
FSM, `nosource`, OEP/EXX and Hartree-Fock, which removes every conditional
in `energy.f90` and `potks.f90` except the spin-polarised branch.

---

## 6. Phased plan

Each phase has a **forward** success criterion and, where it produces a differentiable quantity,
a **gradient** criterion. The gradient criteria deliberately avoid high-symmetry silicon: at the
equilibrium diamond geometry the forces are zero by symmetry, `symveca` averages per-k garbage
toward zero over up to 48 operations, and a gapped insulator hides every Fermi-level and smearing
hazard. A green Si test suite is close to uninformative about hazards A–E and H–J of §8(e).

**Use Elk's own regression harness rather than inventing tolerances where it already has them.**
`modtest.f90` plus task 500 (`testcheck`) is exactly this facility, and 39 routines already emit
reference values through it — `energy.f90:265` writes `'total energy'` at tol $10^{-5}$,
`occupy.f90:96,131` write `'DOS at Fermi energy'` at $5\times10^{-3}$ and the estimated band gap
at $2\times10^{-2}$, `init0.f90:480` writes `'number of G-vectors'`. Separately, `modvars.f90`'s
`writevars` dumps `avec`, `bvec`, `omega`, `atposl`/`atposc`, `gmaxvr`, `ngridg`, `intgv`,
`ngvec`, `ivg`, `igfft` and `rmt` to `VARIABLES.OUT` — precisely the frozen static tables §5's
`Shapes`/`Basis` proposes to rebuild by hand, available as a free reference dump and as the right
ground truth for Phase 4's structural assertion.

### Phase S — the scaffolding (unestimated; the dominant cost, per B3)

§5's `Config`/`Shapes`/`Symmetry`/`Basis`/`Field`/`State` design, and the conversion of Elk's 654
`modmain` globals into explicit arguments. Every §3 row assumes this exists and no §3 or §6
figure includes it. Survey 6 put the spine at 2–3 person-years; that is an inference and this
document has nothing better. **It is listed as a phase precisely so that it stops being invisible
in the arithmetic.**

### Phase 0 — prove or kill (3–4 weeks, no Elk code)

| Item | What it settles | Kill criterion |
|---|---|---|
| **0a.** Reverse-mode implicit diff wired as `jax.custom_vjp` + GMRES on the transposed operator, where the tangent operator routes through the safe-$K$ projector rule on a small Hamiltonian with an *engineered degenerate pair* | whether reverse mode survives the coupling of B1 and B2 — which is the real risk, since the implicit solve's matvec **is** the JVP of one Kohn-Sham step and therefore passes through `eigh`'s derivative at every multiplet, on every GMRES iteration, and is then transposed | Agreement with central FD on the toy. Note what this is *not*: a smooth scalar fixed point with no eigensolve is already known to work (measured, $3.4\times10^{-10}$) and settles nothing. Do **not** use `lax.custom_root` with an iterative `tangent_solve` — measured, it raises `NotImplementedError`, which would read as a kill and is not one. |
| **0a′.** The same, second order: `jax.hessian` through that fixed point | whether §8(f) items 2, 3 and 8 and all of Phase 5 exist | This is the item that actually decides the full port over the hybrid. Measured, `jax.hessian` through a `custom_vjp` fixed point works on a toy *without* an eigensolve; with one in the loop, the safe-$K$ rule must be twice-differentiable and the pad block must not be exactly degenerate. If it does not work, Phase 5's Hessian is forward-over-forward at $3N\times3N$ JVP cost, not $3N$ implicit solves — say so rather than quietly reporting a number. |
| **0b.** safe-K projector `custom_jvp` + the $10^{-16}$ reassembly-jitter test, **and** the padding case — **DONE, `docs/jax_port_phase0.md`** | whether B1 has a working fix | Relative spread must be $\sim10^{-15}$, not the measured 159% of naive AD. Two sharpenings. (i) Include §3.2's own padding layout as a test case: $H_{\rm pad}=E_{\rm big}I$ makes a *bitwise* degeneracy and measured returns `NaN` from a projector gradient even for a non-degenerate physical spectrum — a Phase 1 developer meets this on day one. (ii) Repeat at $n\approx1000$ with a real Cholesky-reduced $S$, with a numeric criterion set at that size from the measured $\kappa(S)$, since the tolerance scales as $\epsilon\kappa(S)\|H\|$ and $\kappa(S)$ is unknown here. |
| **0c.** `jax.jvp(match)` vs `dmatch.f90` | whether the LAPW position-dependence is AD-tractable | Agreement to machine precision against $d(\texttt{apwalm})/dr = i(\mathbf G+\mathbf k)\,\texttt{apwalm}$. Exercises `sbessel`, `genylmv` (the `t4pil` trap) and `gensfacgp`, needs no SCF. If this fails nothing downstream is worth debugging. |
| **0d.** `vmap(eigh)` vs `lax.map` at $n=1000$, complex128, on a real GPU | the k-axis structure, before any k-loop is written | Not a kill; a design fork. Ratio $\approx1$ means the eigensolve is not helped by batching and the case for an iterative solver becomes decisive. **Requires GPU access this study did not have** (§1 fact 4); budget a cloud instance, since nothing else in Phase 0 needs one. |
| **0e.** `jit` compile time and peak device memory for **one** traced SCF step at production shapes ($n_{\rm mat}\approx3000$, $n_{\bf k}\approx100$) | whether hazard K is a real wall | No kill number is available in advance; the point is to have the number before Phase 3 designs around its absence. §8(d)'s reassuring 1.17–1.42x and 0.21–1.06 s are toy-scale ($n\le400$, 4 k-points) and must not be extrapolated: §3 proposes unrolled constructs (an 8-pass corrector, unrolled Gram–Schmidt, per-species `jit`s, a scan over ~700 radial points) inside a step that also contains per-k eigensolves, and XLA compile time grows superlinearly in HLO op count. |

### Phase 1 — one k-point, one species, no SCF (2–3 months)

Build `match` → `hmlfv`/`olpfv` → Cholesky-reduced `eigh`, reading a converged `STATE.OUT` from
the real binary as a fixed input.

- **Forward:** reproduce `EVALFV` at 4 chosen k-points of bulk Si to $10^{-8}$ Ha absolute, and
  at 4 k-points of monolayer h-BN. Check the local-orbital block **gauge-invariantly** — `EVECFV`
  is arbitrary within any degenerate multiplet (hazard C), so compare occupied-subspace projectors,
  $\|P_{\rm JAX}-P_{\rm Elk}\|_F < 10^{-8}$, or the principal angles between the LO-projected
  occupied subspaces, not the coefficients.
- **Gradient:** on **displaced h-BN** (one N pushed 0.05 Bohr along a general direction, `nosym`,
  `reducek=0`, `tshift=False` — low symmetry, non-degenerate, gapped), compare
  $d\varepsilon_j/d\mathbf R$ against central finite differences of the same eigenvalue at three
  step sizes ($10^{-3}$, $10^{-4}$, $10^{-5}$ Bohr), **for non-degenerate bands only**, and
  compare a multiplet-summed quantity ($\sum_{j\in{\rm occ}}\varepsilon_j$) everywhere else.
  **A step-size-independent disagreement is a wrong gradient**; a step-size-dependent one is
  truncation error.
- **Gradient, adversarial:** the same on graphene with `soc_scale` swept 3000 → 300 → 30 → 3
  (patch 0001 makes the Dirac gap a continuous knob from $\sim$1.4 eV to $\sim\mu$eV). Require
  relative agreement $<10^{-6}$ on the multiplet-summed quantity at every value, **and** require
  an explicit refusal (not a returned number) once the sampled gap falls below the tolerance
  §8(b) derives from $\epsilon\kappa(S)\|H\|$. "Agreement or a clean refusal" without a threshold
  at which refusal is *required* is unfalsifiable. This is the only test in the suite
  that reaches the near-degenerate regime where B1 actually bites.
- **Gradient, negative (required to fail):** at a k-point where two occupied bands are exactly
  degenerate (h-BN at $\Gamma$, or graphene at K with `soc_scale=0`), require that AD, central FD
  and one-sided FD of an **individual** eigenvalue *disagree*, and that the multiplet trace agrees
  across all three. This turns §8(b)'s degeneracy caveat into an asserted signal instead of a
  silent pass.

### Phase 2 — the density and potential half (2–3 months)

`rhomag` via the local-density-matrix route, Weinert Poisson, one LDA and one GGA functional as
scalar $\varepsilon_{xc}$, symmetrisation.

- **Forward:** total energy of bulk Si to $10^{-6}$ Ha against Elk, at a fixed input potential
  (one SCF step, not a converged run). Cell charge integral to $10^{-8}$ electrons.
- **Gradient:** `jnp.isfinite` on the full gradient for (i) the Cr-monolayer fixture with 24 Bohr
  of vacuum (every interstitial point at $\rho\approx0$) and (ii) a gapped insulator where
  occupations are exactly 0 and 1. Both fire the `where`-`NaN` hazard; **a metal does not**, because
  partial occupations keep $f$ strictly inside $(0,1)$. Measured, the naive transcription of Elk's
  own guards gives the *correct value* 0.0 and a `NaN` gradient in both cases.
- **Gradient:** $v_{xc}$ from `jax.grad` of $\varepsilon_{xc}$ against Elk's own hand-coded
  $v_{xc}$ pointwise, to $10^{-10}$.

### Phase 3 — the SCF loop with implicit differentiation (3–4 months)

- **Forward:** a converged ground state of bulk Si (2 atoms, $4^3$ mesh) reproducing Elk's
  `TOTENERGY.OUT` to $10^{-6}$ Ha and `EFERMI.OUT` to $10^{-8}$ Ha, and of monolayer h-BN, with
  the **iteration count within $\pm2$** of Elk's. Reproducing
  $\tilde E \leftarrow 0.75E+0.25\tilde E$ is necessary but does not make the iterate sequences
  identical — a different eigensolver, a different BLAS reduction order and Broyden's
  `dgetrf`/`dgetri` in different arithmetic give a different $v_{\rm out}$ at every step, so
  whether the residual crosses `epspot` on iteration $k$ or $k+1$ is a coin flip. "Same iteration"
  is not an achievable criterion and is the kind that gets met by tuning.
- **Gradient A (the implicit-diff signature):** compute the same gradient with `mixtype=0` and
  `mixtype=3` and require the inter-mixer difference to **fall linearly with `epspot`** over
  $10^{-4},10^{-6},10^{-8}$ under implicit differentiation, and to **plateau** under unrolling.
  Two mixers do not reach the same point — they reach different points inside the `epspot` ball —
  so implicit gradients differ at $O(\texttt{epspot}\times\partial(\mathrm{grad})/\partial v)$,
  not at zero; the *scaling*, not a fixed $10^{-8}$, is the signature. Measured, an *unrolled*
  gradient differs at $\sim10^{-3}$ between mixers at matched forward accuracy
  ($1.9215\times10^{3}$ vs $1.9252\times10^{3}$, truth $1.9281\times10^{3}$) regardless of
  `epspot`. This test needs no reference value and is the sharpest available.
- **Gradient B (metals and smearing):** on **bcc Fe**, compare $d\mu/d\varepsilon$ against
  central FD of the converged $\mu$. Measured, the literal bisection is **46% wrong with the same
  sign and the same order of magnitude**, and is *identical* at 40, 60, 100 and 200 iterations —
  so iteration-count invariance must not be mistaken for correctness. Additionally assert
  $dE/df_i = 0$ at the converged occupations to $10^{-9}$, and run the whole force suite under
  `stype = 0, 1, 2, 3`: under Methfessel–Paxton this **fails**, because `energy.f90:242` sets
  `engyts = 0`, and that failure is the force error one would otherwise chase for a week.
- **Gradient C (tolerance):** gradient converged in `epspot` over $10^{-4},10^{-6},10^{-8}$, and
  the plateau reached at a *looser* tolerance under implicit diff than under unrolling — DFTK's
  published result (cited) and the signature that implicit diff is actually wired up.

### Phase 4 — forces and stress (2 months)

- **Forward:** forces on displaced h-BN agreeing with `FORCES.OUT`. State this as a hypothesis
  with a stated fallback rather than a criterion, because §8(c) predicts the HF/IBS decomposition
  will *not* match and possibly the total will not either: if the total agrees to $10^{-4}$
  Ha/Bohr, recipe (d) is validated; if it disagrees, the fallback is the isolation test below,
  which localises the disagreement instead of accepting it.
- **Gradient:** $dE/d\mathbf R$ vs central FD of the total energy on the low-symmetry cell.
  **A fixed $10^{-6}$ Ha/Bohr tolerance is below the achievable floor** and would fail correct
  code: at $E_{\rm tot}\sim-578$ Ha with `epsengy` $=10^{-8}$ Ha, FD noise at $h=10^{-4}$ Bohr is
  $\sim10^{-4}$ Ha/Bohr, and even at machine precision on that total the floor is
  $\sim10^{-6}$. Require instead $|{\rm AD}-{\rm FD}| < \max(10^{-4}\ \text{Ha/Bohr},\,
  3\,\delta E/h)$ with $\delta E$ the *measured* run-to-run energy reproducibility, and require
  the discrepancy to **shrink as $h$ grows** over $10^{-4}$–$10^{-2}$ Bohr (truncation-limited)
  rather than sitting at a fixed value (wrong gradient) — the same step-size criterion Phase 1
  already uses.
- **Gradient, isolation:** compute the force twice, once with `apwalm`
  recomputed inside the trace and once with it `stop_gradient`ed, and check the difference against
  `forcek.f90`'s own output from the real binary. This is the only test that distinguishes the
  three force recipes. **Then a second isolation for recipe (d)'s frame problem** (§8c): compute
  $\nabla_{\mathbf R}$ with and without `stop_gradient` on `atposc` inside the
  $E_{\rm Hxc}[\rho_{\rm in}]$ evaluation, and compare the difference against `force.f90`'s own
  $\int_{\rm MT}v_s\nabla\rho$ block. Those two are the same shape with different arguments
  ($v_{\rm Hxc}[\rho_{\rm in}],\rho_{\rm in}$ versus $v_s,\rho_{\rm out}$) and should coincide at
  self-consistency; **whether they do is what this test establishes**, not an identity to assume.
  Under an atom-frame freeze the term is absent from AD entirely and must be added by hand.
- **Structural:** assert `nsymcrys`, `ngridg`, `ngtot`, `ngk`, `nstfv` and `rmt` are **bitwise
  unchanged** between the reference and every displaced geometry, and raise otherwise — against
  Elk's own `VARIABLES.OUT` dump, not a hand-built reference. Hazard D
  cannot be caught by finite differences, by construction.
- **Negative tests (required to raise, not to return a number):** a band window whose boundary
  sits *inside* a degenerate multiplet must make `check_window_gap` raise (hazard B); `stype=0`
  must make the differentiable entry point refuse rather than silently return the entropy-free
  force (hazard F); a request for $dE/dE_\ell$ must raise rather than silently return the
  `stop_gradient`ed zero (hazard M). Every other criterion in Phases 1–5 checks that a number is
  right; these check that a refusal fires, which is the path that regresses silently.
- **Core:** finite-difference a 1s core eigenvalue with respect to a localised bump in the
  spherical potential, on **W or Bi** (not Si), against the analytic $dE_c/dV = g_c^2 + f_c^2$.

### Phase 5 — second derivatives (open-ended, and gated on Phase 0a′)

$q=0$ Hessian, checked against Elk's task 205 at $q=0$ to 1 cm$^{-1}$; then explicitly verify that
the same code path yields *nothing* at $q=(\frac12,0,0)$ rather than a plausible number.

**A disagreement with task 205 may be Elk's, and the plan must be able to tell.** Two known
differences between the AD operator and Elk's DFPT operator, both from §8(a): `deveqnfv.f90:96-103`
zeroes every resolvent term with $|\varepsilon_j - w_i| \le \texttt{epsdev} = 0.0025$ Ha, and
`dhmlrad.f90` integrates the *unperturbed* `apwfr` against `dvsmt`, i.e. Elk's DFPT freezes the
radial basis with respect to the potential perturbation while §5's `State` traces it inside the
fixed point. The discriminator for the first is re-running Elk with a smaller `epsdev`; for the
second it is `stop_gradient` on `apwfr`/`lofr`, which should move the AD result *toward* Elk's.
Run both before concluding the port is wrong.

---

## 7. What would not be ported

| Not ported | Why | What happens instead |
|---|---|---|
| `elk.f90`'s task dispatch (2143 lines) | 87 `case` clauses plus `case default` over `elk.f90:109-289`, covering 148 distinct codes, plus ~2000 lines of embedded manual. Only task 0/1 is on the ground-state path | a Python dict of `{task: callable}` |
| `readinput.f90` (2353 lines) | 316 input-block `case` clauses accepting 365 names, with per-field validation and a bare `stop` each | a dataclass plus a small parser; elkpy already does this |
| `modramdisk.f90` (332), `putevecfv`/`getevecfv` I/O, `moddelf` | an in-process file emulator existing only because Fortran cannot pass arrays between subroutines without globals or files | eigenvectors are device arrays; a `lax.scan` carry inside the SCF, a resident array for post-processing. **Except**: `getevecfv`'s symmetry-rotation branch is physics and must be ported |
| `checkstop.f90`/`checkwrite.f90` | polls the filesystem for `STOP`/`WRITE` every iteration | a Python callback between traced steps; no `jit` equivalent should exist |
| 515 bare `stop`s across 171 files (measured) | no `stop` inside a trace; `eigh`/`cholesky` return `NaN` rather than error codes | host preconditions + a finiteness check per step |
| `findsymcrys`/`findsym`/`findsymlat` (586) | integer combinatorics; also **mutates `atposl`** | spglib on the host, once |
| `libxcifc.f90` + libxc | the C library cannot run inside a trace; already optional (this build links the 71-line stub). **Not** because the functionals are unavailable — `jax_xc` covers SCAN/TPSS/r2SCAN/B3LYP/HSE (B7) | Elk's native functionals reimplemented as scalar $\varepsilon_{xc}$; `jax_xc` if meta-GGA or hybrid coverage is wanted, plus the $\tau$ plumbing (§3.3) |
| the ~120 post-SCF task codes that are **consumers** of the converged arrays | they read `STATE.OUT` and produce output; nothing self-consistent | keep calling the Fortran through elkpy against a checkpoint. DOS, band structure, Fermi surfaces, effective mass, EFG/Mössbauer, Wannier90 export, plotting/volumetric output, phonon post-processing |
| the **producer** tasks: 5 (`hartfock`), 200–202/205 (phonons), 270/271 (`gndsteph`), 300 (`rdmft`), 350–352 (`spiralsc`), 420/421 (`moldyn`), 460–463 (`tddft`), 600/601 (`gwsefm`), 700/701 (`gndstulr`) | each runs its **own** self-consistency or time propagation — they are not covered by the consumer argument. §3.5 lists their file counts | keep the Fortran entirely; they are separate projects, not post-processing |
| `eveqnfvr` (264) in phase one | 264 lines of local-orbital recombination exploiting inversion symmetry; **it is the default path** for centrosymmetric crystals | pay ~2x memory and ~4x FLOPs, and know that a benchmark against Fortran is comparing against a code doing a quarter of the work |
| $q\neq0$ DFPT | $e^{i\mathbf q\cdot\mathbf r}\mathbf u$ breaks the traced cell's periodicity | keep Elk's Fortran, or use commensurate supercells |

**The boundary, stated once:** a JAX port owns **the ground-state path of a plain LSDA/GGA
calculation**, plus a checkpoint format. Everything downstream of the converged arrays stays where
it is, and so does everything with a self-consistency of its own — including, note, the
`ksgwrho` branch that sits *inside* `gndstate`'s loop. That is a real scoping relief — the port is
not "821 files" — but it is narrower than "the SCF path plus a file format": DFT+U and fixed spin
moments change the fixed-point equation itself (§2.2), and the second of those is what this
repository's §29 exchange tensor runs on.

---

## 8. Autodiff design

### 8(a) The fixed-point structure, and the identity that matters most

The ground state is defined by $v^* = F(v^*;\theta)$, where $\theta$ is any external parameter
(atomic positions, lattice vectors, `soc_scale`, an applied field, a functional parameter) and

$$
F = \texttt{potks}\circ\texttt{rhomag}\circ\texttt{occupy}\circ\{\texttt{eveqn}\}_{\mathbf k}
\circ(\texttt{gencore},\texttt{linengy},\texttt{genapwlofr},\texttt{gensocfr}),
$$

read off `gndstate.f90`, **at §2.2's stated scope** (`dftu=0`, `ftmtype=0`, `fsmtype=0`,
`ksgwrho=.false.`). Outside it the composition is wrong, not merely incomplete: DFT+U adds
`vmatmt`/`dmatmt` as a second fixed-point variable that never passes through the mixer, and
`fsmtype /= 0` adds a constraint whose Lagrange-multiplier-like field is itself an unknown, so
the correct object there is an augmented system in $(v,\mathbf B_{\rm fsm})$ with
$\mathbf m_\alpha = \mathbf m^{\rm target}_\alpha$ as a second equation. That augmented system is
not derived in this document. **$F$ is defined without the mixer**: `mixerifc` is the solver, not the
equation, and since the converged answer does not depend on it, the gradient must not either.

Differentiating the fixed-point condition and applying the implicit function theorem:

$$
\left(\mathbb 1 - \frac{\partial F}{\partial v}\right)\frac{dv^*}{d\theta}
= \frac{\partial F}{\partial\theta},
\qquad
\frac{\partial F}{\partial v} = K_{\rm Hxc}\,\chi_0 ,
$$

with $\chi_0$ the independent-particle susceptibility (the response of $\rho$ to a change in the
potential, through the eigenproblem) and $K_{\rm Hxc}$ the Hartree-plus-XC kernel. **$\chi_0$ is
exactly the derivative of the occupied projector** $P = \sum_j f_j c_jc_j^\dagger$, so it
inherits §8(b)'s custom rule *in full*: without it the implicit Jacobian is not merely
inaccurate, it is undefined at every multiplet — the safe-$K$ rule is what makes the linear
system well-posed, not just what makes the loss gradient right. For a scalar
loss $L$, reverse mode instead solves the transposed system
$(\mathbb 1 - K_{\rm Hxc}\chi_0)^{\mathsf T}\lambda = \partial L/\partial v$ once, then contracts
$\lambda$ with $\partial F/\partial\theta$ for every $\theta$ at once.

**The structural insight: that operator is the same *class* of operator as Elk's DFPT, already
written by hand.** Applying $(\mathbb 1 - K_{\rm Hxc}\chi_0)$ to a trial vector is `drhomagk` (the
density response to a potential perturbation) followed by `dpotks = dpotcoul + dpotxc` (the kernel
acting on that response) — one iteration of `phonon.f90`'s Broyden loop over `dvsbs`.

It is a *class*, not an identity, and the two differences both matter for Phase 5. (i) Elk's DFPT
**freezes the radial basis**: `dhmlrad.f90` integrates the unperturbed `apwfr`/`lofr` against
`dvsmt`, and nowhere propagates $\partial(\texttt{apwfr})/\partial v_s$ — whereas §5's `State`
carries `apwfr`/`lofr`/`haa`/`socfr` *inside* the fixed point (correctly, since `gndstate` calls
`linengy`/`genapwlofr`/`gensocfr` in the loop). Tracing them gives a strictly *larger* operator
than Elk's, and the two linear systems have different solutions. The port must choose and say
which: `stop_gradient` the radial functions to match Elk and call $\chi_0$ the frozen-basis
susceptibility, accepting that §8(f) item 2 will not reproduce task 205 term by term; or keep
them traced and call Elk's DFPT the approximation. (ii) `deveqnfv.f90:98` hard-truncates,
dropping every resolvent term inside `epsdev` $=0.0025$ Ha — a cruder version of exactly the
broadening this section rejects below — so agreement with Elk should be expected only where all
$|\varepsilon_j - w_i| \gg 68$ meV.

Implicit differentiation of
the SCF is not a machine-learning technique bolted onto DFT; it is *the same linear-response
calculation Elk already performs*, obtained by linearising one traced SCF step instead of
hand-deriving a perturbation expansion. DFTK reached the same place from the plane-wave side and
said so: "For the SCF algorithm we manually define its derivative as the matching DFPT algorithm"
(cited).

**Implicit versus unrolled — and a correction to the received view.** The usual argument is that
unrolling is *inaccurate*. Measured, it is not, particularly: at Elk's `epspot = 1e-6` the
unrolled gradient error was $0.9$–$1.9\times10^{-5}$ relative, only 1.4–1.9x the forward value
error, even at a mixing spectral radius of 0.98. The real objections are different and stronger:

- **Tape size.** One SCF iteration's residuals include every k-point's eigenvectors:
  $n_{\rm kpt}\,n_{\rm matmax}\,n_{\rm stfv}\times16$ B is 1.44 GB at (100, 3000, 300). At 200
  iterations that is $\sim$288 GB. Implicit differentiation is $O(1)$ in iterations.
- **Mixer dependence.** Measured, at matched forward accuracy $\sim10^{-3}$, linear mixing gave
  $1.9215\times10^{3}$ and Anderson/Pulay $1.9252\times10^{3}$ for a quantity whose true value is
  $1.9281\times10^{3}$. The user experiences this as nondeterminism.
- **`lax.while_loop` has no reverse rule at all**, so an unrolled reverse-mode SCF requires a
  fixed trip count in the first place.

Implicit gradients are exact only *at* a fixed point, so gradient work needs tighter convergence
than Elk's defaults — this repository already drops `epsengy` from $10^{-4}$ to $10^{-8}$ in §29
for the same reason.

### 8(b) Exact and free / ill-posed / custom rule / undefined

The eigenproblem is four different questions with four different answers.

| Object | Status | Why |
|---|---|---|
| **Multiplet-summed eigenvalues**: $\sum_j f_j\varepsilon_j$ and any quantity constant across each degenerate group | **exact and free** | The JVP is $d\lambda_j = c_j^\dagger(dH - \lambda_j\,dS)c_j$ with **no** $1/(\lambda_i-\lambda_j)$ anywhere. Inside a degenerate multiplet every member gets the same $f$ (occupations depend only on $\varepsilon$), so $\sum_{j\in\text{mult}} f\,d\lambda_j = f\,\mathrm{Tr}[P_{\rm mult}\,dA]$ is manifestly invariant under the arbitrary unitary `eigh` picks. Measured exact at every splitting including zero: at an exact two-fold degeneracy the trace agreed to $10^{-15}$ across AD, central FD and analytic degenerate PT. **This is the single fact that makes recipe (d) forces safe in an all-electron code.** |
| **Individual eigenvalues** $\varepsilon_j$ | **ill-posed at a degeneracy, contaminated near one** | An individual eigenvalue is only *directionally* differentiable at a multiplet, via the eigenvalues of $P\,dA\,P$, and JAX returns $c_j^\dagger dA\,c_j$ in whatever basis `eigh` happened to pick. Measured at an exact two-fold degeneracy: AD $(-0.204,-0.793)$, true one-sided $(-1.553,+0.557)$, central FD of the *sorted* spectrum $(-0.498,-0.498)$ — the branch average, because sorting swaps the branches between $\pm h$, so **an FD check cannot detect this failure**. Near-degeneracy is contaminated too: at $n=400$ with an all-electron-like 1100-Ha span, relative error was $4.2\times10^{-7}$ at $10^{-8}$ splitting, $3.9\times10^{-5}$ at $10^{-10}$, **7.7% at $10^{-12}$ and 10% at exact degeneracy**. Contamination floor $\sim\epsilon\,\kappa(S)\|H\|/\delta$. |
| **Individual eigenvectors** $c_j$ | **undefined; never differentiate** | Within a multiplet the eigenvectors carry a U($n$) gauge freedom resolved arbitrarily. Measured: a $10^{-14}$ perturbation rotates them by $\sim57^\circ$ ($|\langle v_2^A|v_2^B\rangle| = 0.539$) while the projector over the same multiplet is unchanged to $2.2\times10^{-14}$ — the numerical form of §14's observation that two diagonalisations pick different bases. |
| **Occupied projector** $P=\sum_j f_j c_jc_j^\dagger$ | **needs a custom rule** | Gauge-invariant and mathematically differentiable when the window is gapped, *and still returns garbage under JAX's default VJP*, because `eigh`'s rule divides by $(\lambda_i-\lambda_j)$ before the $(f_i-f_j)$ numerator can cancel. Fix: form $K_{ij} = (f_i-f_j)/(\lambda_i-\lambda_j)$ as **one** expression with `jnp.where` and the analytic $f'(\lambda)$ on the near-degenerate block. Measured: error at exact degeneracy $2.08\times10^{-2}\to3.0\times10^{-10}$; jitter reproducibility $159\%\to4.3\times10^{-15}$. **Scope caveat, RESOLVED — see `docs/jax_port_phase0.md`:** a check added after this document was drafted reported the claim overstated for a **hard integer window with the multiplet fully enclosed**, on the grounds that the two divergent terms are exact negatives and cancel bitwise (measured agreement with central FD of $1.5\times10^{-9}$ on two of three assemblies; the third's $1.3\times10^{-2}$ was attributed to FD noise). Phase 0b settled it against the **closed-form** derivative $dP = V(K\circ V^\dagger\,\delta H\,V)V^\dagger$ rather than FD, and the original claim stands: over 3 assemblies x 21 Hermitian directions the naive route's worst relative error is $1.1\times10^{1}$ forward and $3.4\times10^{0}$ reverse, against $2.9\times10^{-14}$ with the rule. FD is in fact *reliable* here (worst $3.7\times10^{-8}$, stable over three step sizes) — $\mathrm{Tr}[PM]$ is smooth in $H$ whenever the window BOUNDARY is gapped, however degenerate the interior — so the third assembly's disagreement was AD error, not FD noise. The earlier check saw agreement because it used one direction (a single real diagonal entry) in reverse mode only: that direction is nearly benign in reverse mode ($<10^{-7}$) and already wrong at $1.3\times10^{-2}$ in *forward* mode on the same matrix. Forward and reverse disagreeing is itself the proof, since for a scalar-in scalar-out function they are the same number. The mitigation for a boundary degeneracy is still the one this project applies to Berry curvature (CLAUDE.md §13): window the whole degenerate group — `elkjax.projector` returns `NaN` rather than a number there. |

Two further precision points on this rule. First, "we only use eigenvalues" is not enforced by
declining to write $v$ into the loss: JAX skips the `Fmat` branch only for a *symbolic* zero
cotangent, and an explicitly materialised zeros array still evaluates $\infty\times0=\text{NaN}$.
Use `eigvalsh`, or `stop_gradient(v)` at the point $v$ enters the density, or install the rule.
Second, the tolerance in `jnp.where` biases: pairs genuinely split just above `tol` get treated as
degenerate, an $O(\text{tol}/\delta)$ error. Choose `tol` from the eigenvalue backward error —
which for the **generalized** problem solved by the Cholesky reduction of B4 is
$\sim\epsilon\,\kappa(S)\,\|H\|$, not $\epsilon\|H\|$. The naive figure ($\sim10^{-13}$ Ha for a
spectrum spanning 2500 Ha) is therefore a *lower bound* by however many orders $\kappa(S)$
carries, and setting `tol` there would treat genuinely-degenerate pairs as split — reintroducing
hazard A exactly where the rule was installed to prevent it. **$\kappa(S)$ for Elk's APW+lo
overlap was never measured in this study.** So: compute it per k-point at runtime (cheaply, as
$\min\mathrm{diag}(L)^{-2}$ from the Cholesky factor the solver already forms), set `tol` from it,
and refuse the k-point when the implied `tol` exceeds a physically meaningful splitting. Then
require the gradient to be flat over at least two decades of `tol` — **at production $n$ with a
real LAPW $S$**, not at the 4×4 scale, since that is where the plateau can vanish.

Third, and easy to inflict on yourself: **do not manufacture degeneracies in the padding**.
§3.2's obvious choice $H_{\rm pad} = E_{\rm big}\mathbb 1$ creates a bitwise
$(n_{\rm matmax}-n_{\rm mat})$-fold degenerate block at every k-point, unconditionally. Measured
on an 8×8 with a 4-fold pad block at $E_{\rm big}=10^3$ and a *non-degenerate* physical block: a
projector-derived gradient returns `NaN` (FD gives $0.4299$), while an eigenvalue-only gradient is
correct to 10 digits, and spacing the pad eigenvalues by any $\delta$ fixes it exactly. This is
the one place where the `NaN` failure mode of §10 item 1 is the realistic one rather than the
toy one, because the degeneracy is bitwise by construction rather than roundoff-split.

**Do not reach for Lorentzian broadening** ($1/\Delta \to \Delta/(\Delta^2+\eta^2)$). It biases
*every* pair, not just degenerate ones, with the same sign, so it does not average out over
k-points or atoms: measured on a system with no degeneracy at all, relative bias
$4.19\times10^{-2}$ at $\eta=0.1$, $3.91\times10^{-3}$ at 0.03, $4.36\times10^{-4}$ at 0.01.

**The rest of the code, by status:**

- **DAW, and this is most of it:** `match`, the H/S assembly, `rschrodint` at fixed energy (there
  is no `rschrod.f90` in the tree; `genapwfr`/`genlofr` call the integrator at a *fixed*
  linearisation energy, so it is a plain IVP), `hmlrad`/`olprad`, the SHT, `gradzfmt`, the density
  build, the Poisson solve, FFTs, symmetrisation with a frozen group, and every native XC
  functional.
- **CVJP:** the projector (above); the Fermi level (analytic $d\mu/d\varepsilon_i = w_if'_i/
  \sum_j w_jf'_j$); core eigenvalues ($dE_c/dV(r) = g_c(r)^2 + f_c(r)^2$, exact from first-order
  perturbation theory, using the normalised radial density `rdirac` has already computed —
  five lines, not an approximation); optionally the radial ODE, whose natural adjoint is a
  backward integration of the same equation.
- **ND, freeze:** `findband` (a bracketing search with no scalar residual — `lax.custom_root`
  does not apply); the mixer; the symmetry group; `rmt`, the integer G-list and every FFT grid.
- **REFORM:** `energy.f90` (§2.2), the force recipe (§8c), $q\neq0$ DFPT.

### 8(c) What autodiff deletes — and the argument, per item

Line counts are measured from `vendor/elk/src/`.

| Elk code | Lines | Replaced by | Does it really? |
|---|---|---|---|
| `forcek.f90` (per-k incomplete-basis force) | **173** | reverse VJP of the band energy | **Yes, with an argument.** Atomic positions enter the *entire* LAPW basis through exactly one object: `sfacgp` $= e^{i(\mathbf G+\mathbf p)\cdot\mathbf r_\alpha}$ in `gensfacgp.f90` (`ffacgp` depends only on $\|\mathbf G\|$ and `rmt`). Hence $d(\texttt{apwalm})/dr_{\alpha,p} = i(\mathbf G+\mathbf k)_p\,\texttt{apwalm}$ — which is literally the whole of `dmatch.f90` (**26 lines**). The same structure factor enters the characteristic function, $\tilde\Theta(\mathbf G) = \delta_{\mathbf G,0} - \sum_\alpha \texttt{ffacg}\,\overline{\texttt{sfacg}}$ (`gencfun.f90`: the leading term is $\delta_{\mathbf G,0}$, not 1, and the exponential is $e^{-i\mathbf G\cdot\mathbf r_\alpha}$, *conjugate* to `gensfacgp`'s $e^{+i(\mathbf G+\mathbf p)\cdot\mathbf r_\alpha}$ — which is precisely the sign this argument turns on), i.e. the $\tilde\Theta$ in Elk's own $\delta H$/$\delta O$ formulae. Differentiating a traced $H(\mathbf R)$, $S(\mathbf R)$ therefore regenerates the $i(\mathbf G-\mathbf G')$ prefactors, the $\tilde\Theta$ terms, the APW–LO block and the muffin-tin surface terms automatically. **Confidence: high on the arithmetic, medium on getting the right thing** — see the recipe problem below. |
| `force.f90` (HF term, core correction, MT integrals, `symveca`) | **241** | partly `jax.grad`; `symveca` (50) survives as a constant projector | **Partly, and less than first stated.** The Hellmann–Feynman term: yes. **The $\int_{\rm MT}v_s\nabla\rho$ and $\int_{\rm MT}\mathbf B_s\!\cdot\!\nabla\mathbf m$ blocks: no** under the atom-frame freezing recipe (d) needs — see the frame discussion below; they must be ported. **The core correction is genuinely uncertain**: Elk's $\int_{\rm MT}v_s\nabla\rho_{\rm core}$ repairs the Hellmann–Feynman *formula* for a core that is an eigenstate only of the spherical part of $v_s$. Autodiff never invokes that formula; it differentiates whatever energy expression is written. It reproduces the term only if $\rho_{\rm core}$ enters $E_{\rm Hxc}$ as a rigid, sphere-local, position-dependent density **and** $\varepsilon_{\rm core}$ carries the PT custom VJP. This must be tested on W or Bi, not asserted. |
| `genstress.f90` + `genstrain.f90` | **118** | `jax.grad` w.r.t. a 3×3 strain on `avec` | **Yes on cost, with two caveats on identity.** Elk runs $1+n_{\rm strain}$ (up to 7) *complete* SCF calculations and forms a **forward** difference $(E_1-E_0)/\texttt{deltast}$, $\texttt{deltast}=0.005$, on an energy of order $10^3$–$10^5$ Ha. AD does it in one backward pass and never forms the difference. **But**: (i) freezing the integer G-list (which is mandatory, or the shapes change) makes AD the *fixed-basis analytic stress*; Elk's finite difference includes the basis-set Pulay stress from the G-set and $g_{\max}$ changing with the cell, so they agree only in the converged-`rgkmax` limit. (ii) Elk's `stress` array is **not** six Voigt components: `genstrain.f90` builds up to 7 symmetry-adapted directions — each $\delta_{ij}$ passed through `symmat`, converted by `r3mtm(ainv,·)`, Gram–Schmidt-orthogonalised against those already accepted, discarded below `epslat`, normalised, with `strain(:,:,1)` fixed as $\texttt{avec}/\|\texttt{avec}\|$ — so their number and identity depend on the crystal's symmetry. Any comparison must contract the AD $3\times3$ result against `strain(:,:,istrain)`, and carries Elk's own $O(\texttt{deltast})$ forward-difference truncation error on top of the basis-set difference. Report them as different objects. |
| `ggamt_*`/`ggair_*` (GGA gradient plumbing) | **1240** | implement only the scalar $\varepsilon_{xc}$; `jax.grad` for $v_{xc}$ | **Yes, unambiguously.** Elk carries *two incompatible conventions* — `xcgrad=1` supplying $\|\nabla\rho\|$, $\nabla^2\rho$ and $\nabla\rho\!\cdot\!\nabla\|\nabla\rho\|$; `xcgrad=2` supplying $\sigma$ and receiving $\partial\varepsilon/\partial\sigma$, with `ggamt_2b` applying $\nabla\!\cdot\!(2\,\partial\varepsilon/\partial\sigma\,\nabla\rho)$ by hand — *solely* because $v_{xc}$ was derived by hand. The existence of two conventions is itself the proof. `gradzfmt` (210) survives as real numerical content. |
| `modfxcifc`/`genfxcr`/`genspfxcr`/`genspfxcg`/`fxc_pwca` | **672** | `jax.hessian` of $\varepsilon_{xc}$ | **Yes, and it gains a capability**: Elk has no GGA $f_{xc}$ at all (`libxcifc.f90` exposes only `lda_fxc`). Pure elementwise arithmetic, no eigenvectors, no fixed point. The cleanest deletion in the list. |
| DFPT response path (`phonon`, `dforce`, `dforcek`, `deveqnfv`, `drhomagk`, `dhmlrad`, `dpotks`, `dpotxc`, `dmatch`, `doccupy`, `modphonon`, + G+q blocks) | **1622** for the 11 core files named here (measured, re-counted); *survey estimates for the full closure disagree — 2540 across 29 files vs 4733 exclusive of which ~3155 is the $q\neq0$ spine across 41 files* | $q=0$: `jax.hessian` over the implicit fixed point | **At $q=0$ only, and the win is code, not FLOPs.** $3N$ Hessian columns = $3N$ implicit solves = DFPT's $3N$ `dyntask` columns. What is gained is (i) ~2000+ lines collapsing to a `hessian` call plus the projector rule, and (ii) two upstream gaps closed for free: `phonon.f90` hard-`stop`s on `spinpol`, and `deveqnsv`/`deveqnss` are **commented out** (`devalsv := devalfv`), so the second-variational/SOC eigenvalue derivative does not exist upstream at all. **At $q\neq0$ it is not replaced**: $e^{i\mathbf q\cdot\mathbf r}\mathbf u$ breaks the traced cell's translational symmetry, $q$ is not an argument of the ground-state function, and no JVP recovers it. |
| `phononsc.f90` + supercell drivers | **186** + drivers | supercell Hessian | **Partly.** $3N_{\rm sc}$ linear solves instead of $6N_{\rm sc}$ nonlinear SCF runs, and exact rather than finite-difference — call it 3–5x, not free. |

**The recipe problem, which is the real risk in this table.** There are (at least) three defensible
places to put `stop_gradient`, and they give three different forces, all finite, all symmetric
under `symveca`, all convergent under geometry optimisation:

| Recipe | What it gives | Verdict |
|---|---|---|
| (a) `stop_gradient` the converged `vsbs`, then `jax.grad` of `energy.f90`'s expression | uncontrolled | **Invalid.** `energy.f90:226` mixes $\varepsilon$ from $v_{\rm in}$ with potentials from $\rho_{\rm out}$; it is not stationary in any single argument. The error shrinks as the SCF tightens, so it *looks* like a convergence issue. |
| (b) `stop_gradient` $\rho$ and `vsmt` | Hellmann–Feynman **only** | **Wrong, and it is the natural thing to write.** The entire 173 lines of `forcek.f90` silently absent. |
| (c) `stop_gradient` the eigenvector coefficients, rebuild everything inside the trace | HF + IBS, but risks also including a term Elk's Yu–Singh–Krakauer construction deliberately drops (the dependence of $u_\ell$ on $v_s$) | Better; if it includes that term it will agree with FD of $E$ *better* than `FORCES.OUT` does, which reads as a regression |
| **(d) Harris–Foulkes band energy at frozen $\rho_{\rm in}$** — **adopted here** | $E = \sum_{\mathbf k}w\sum_j f\varepsilon_j - \int v_{\rm Hxc}[\rho_{\rm in}]\rho_{\rm in} + E_{\rm Hxc}[\rho_{\rm in}] + E_{nn} + E_{ts}$, whose $\mathbf R$-derivative needs only the **occupation-weighted sum** $\sum_j f_j\,d\varepsilon_j/d\mathbf R$ with $d\varepsilon_j/d\mathbf R = c_j^\dagger(dH/d\mathbf R - \varepsilon_j\,dS/d\mathbf R)c_j$ | **Adopted, with the frame stated below.** Needs no eigenvector derivative at all, and only the multiplet-summed eigenvalue derivative, so hazards A and B do not fire; and $c^\dagger(dH-\varepsilon dS)c$ is exactly `forcek.f90`'s $F_{ij}^{\alpha k}$ as written in `force.f90`'s own docstring. Requires the *free* energy (include `engyts`) and requires stating that the HF/IBS split is not separately preserved — only the total is. |

**Recipe (d) is under-specified in exactly the place that decides the number: which frame the
frozen density is frozen in.** Read `force.f90` in full and the total IBS force is `forcek`'s
$\sum_{ij}c^*cF_{ij}$ **plus** further blocks the table above does not mention: a single
$\int_{\rm MT_\alpha}v_s\nabla\rho$ (line 142ff, `rfmtinp(vsmt, gradrfmt(rhomt))` per atom) and,
when `spinpol`, its $\int_{\rm MT_\alpha}\mathbf B_s\!\cdot\!\nabla\mathbf m$ analogue. Elk's own
docstring shows that single block is the **sum of two conceptually distinct terms**: the
incomplete-basis partition term $\int v_s\nabla[\rho-\rho^\alpha_{\rm core}]$ (line 70) and the
Yu–Singh–Krakauer core correction $\int v_s\nabla\rho^\alpha_{\rm core}$ (lines 23–27), which add
to $\int v_s\nabla\rho$. The row above already flags the second as "genuinely uncertain"; the
point here is that the **first** is missing too.

- **Atom frame** (freeze `vsmt`/`haa`/`oalo` in the *moving* atom's own frame, i.e. per-atom
  `State.vsmt` under `stop_gradient` — the natural thing under §5's layout, and what `forcek`
  itself does, differentiating only `sfacg`/`ffacg`): AD then reproduces `forcek.f90` and the
  partition-motion term is **silently absent** — the same failure mode this table calls fatal for
  recipe (b), one level down. What recipe (d) loses is
  $\int_{\rm MT_\alpha}v_{\rm Hxc}[\rho_{\rm in}]\nabla\rho_{\rm in}$, the same *shape* as Elk's
  block but with the frozen input density and potential rather than $v_s$ and $\rho_{\rm out}$;
  at self-consistency $\rho_{\rm in}=\rho_{\rm out}$ so the two are expected to coincide
  numerically, **but that identification is not derived here** and Phase 4's test should establish
  it rather than assume it.
- **Lab frame** (the frame in which Harris–Foulkes is actually stationary at self-consistency):
  moving an atom requires re-expanding $v_s$ in spherical harmonics about the displaced centre,
  an operation not well defined in the two-block muffin-tin layout §5 adopts and not costed
  anywhere in §3.

**Recommendation: freeze in the atom frame and add the two integrals by hand**, i.e. amend the
row above to "`forcek.f90` (173 lines) replaced; `force.f90`'s density- and
magnetisation-gradient blocks are **not** replaced and must be ported". Phase 4 now carries a test
that isolates precisely this term. This is the one place in §8(c) where the optimism was not
backed by a reading of the code being cited.

### 8(d) Forward versus reverse, per target

| Target | Inputs → outputs | Mode | Memory |
|---|---|---|---|
| Forces | 1 scalar → $3N$ | **reverse**, one pass | VJP contracts $\sum_j f_jc_jc_j^\dagger$ against $dH/d\mathbf R$, $dS/d\mathbf R$; needs $H$, $S$, $c$ live per k — at $n_{\rm mat}=3000$, complex128, that is $H$ 144 MB $+$ $S$ 144 MB $+$ $c$ ($3000\times300$) 14 MB $\approx$ **302 MB per k**, before any residuals or `remat` checkpoints → `lax.scan` over k-chunks with the cotangent in the carry, never `vmap` over the whole mesh |
| Stress | 1 scalar → up to 7 symmetry-adapted directions (9 raw) | **reverse** | as forces; additionally requires `ivg` frozen, and see §8(c) on Elk's `strain` basis |
| $d(\text{bands})/d(\texttt{soc\_scale})$, deformation potentials, $d/d(\texttt{rgkmax})$ | 1 → many | **forward**, one JVP through the implicit solve | $O(1)$ in outputs; one GMRES solve of $\sim$10–30 matvecs. **Per-band outputs inherit §8(b)'s individual-eigenvalue caveat** — meaningless inside a multiplet, and h-BN/WSe$_2$ valleys are exactly where those live |
| Effective masses, $\partial u/\partial\mathbf k$ | per-k, small | **forward-over-forward** | needs no fixed-point solve at all (fixed potential) — this is why the hybrid in §9 gets it cheaply. $\partial u/\partial\mathbf k$ must be written in **projector form** (hazards A/B); note JAX's `eigh` JVP produces $\partial u$ in the implicit parallel-transport gauge $\langle u_j|\partial u_j\rangle=0$, which is a property of the rule, not of $\partial u$, so only gauge-invariant combinations are meaningful. A degenerate band edge (Si's $\Gamma_{25'}$ triplet) needs a degenerate-$\mathbf k\!\cdot\!\mathbf p$ block treatment, not three separate second derivatives |
| $q=0$ Hessian (phonons, Born charges, elastic constants) | $3N\times3N$ | **forward-over-reverse**, $3N$ implicit solves — *if* Phase 0a′ passes | one solve at a time; wall-clock bound, not memory bound; requires the projector rule to be twice-differentiable. If 0a′ fails, this becomes forward-over-forward at $3N\times3N$ JVP cost |
| High-dimensional ML-XC training | many parameters → 1 loss | **reverse through the fixed point** | the mode DFTK deferred; the first-order machinery is settled (B2), so what remains is engineering, not a gate. Few-parameter functional fitting does not need it — DFTK did that in forward mode |

Measured for context, and **at toy scale only**: `grad`/forward runtime ratio for a
Cholesky-reduced generalized eigenproblem plus density contraction, vmapped over 4 k, was
1.17–1.42x at $n=64$–400 with compile times 0.21–1.06 s. That says one differentiated SCF step is
cheap *at $n\le400$ with 4 k-points*. It is **not** evidence about $n\approx3000$ with $\sim$100
k-points, where the traced program is orders of magnitude larger and XLA compile time grows
superlinearly in HLO op count; §6's item 0e exists to replace this extrapolation with a
measurement. The memory problem is the iteration axis and the k axis.

### 8(e) Hazard table

| # | Hazard | Symptom a user actually sees | Mitigation | Standard or open? |
|---|---|---|---|---|
| **A** | Eigenvector/projector VJP at a degeneracy returns **finite garbage** in a physical spectrum, and **`NaN`** wherever the degeneracy is bitwise — which §3.2's own $H_{\rm pad}=E_{\rm big}I$ padding creates at every k-point | A force/stress/response that is finite, plausible, symmetric, convergent — and wrong. Changing BLAS vendor or thread count changes its leading digits. Measured: five assemblies of one spectrum gave $\pm(0.7\text{–}3.8)\times10^{15}$; and a projector gradient through an exactly-degenerate pad block returned `NaN` with a non-degenerate physical spectrum | safe-$K$ `custom_jvp` with `tol` derived at runtime from $\epsilon\kappa(S)\|H\|$ (§8b); **pad with distinct eigenvalues**; **test by $10^{-16}$ reassembly jitter** (spread must be $\sim10^{-15}$, not the measured 159%) | standard |
| **B** | The fix **launders** an ill-posed window | Naive AD returns $-7.4\times10^{11}$ (screams); the fix returns $-1.01$ (does not); truth $+2.8\times10^{5}$. FD is *also* garbage here, so AD/FD **disagreement** is the only signal | refuse the calculation: `check_window_gap` as a hard precondition on $\min_k(\varepsilon_{n_{\rm occ}}-\varepsilon_{n_{\rm occ}-1})$ | no numerical fix — the physics is ill-posed |
| **C** | Eigenvector **gauge freedom** used to validate | An FD check of an eigenvector-derived quantity disagrees wildly with AD; you hunt for a bug in correct code. Measured: $10^{-14}$ perturbation ⇒ $\sim57^\circ$ rotation | validate only gauge-invariant quantities. **Corollary for elkpy**: gradients of *off-diagonal* elements of the `PROJECTION`/`ORBITAL`/`ANGMOM`/`SPIN` matrices are meaningless; only traces and window sums survive | standard |
| **D** | **Discrete structure depending on geometry *and on nuclear charge*.** Verified here: `gkmax = rgkmax/`(natoms-weighted **mean** `rmt`) at default `isgkmax=-1` (`init0.f90`); `rmt` itself is adjusted by `checkmt`'s `do` loop over `mtdmin` until surfaces clear `rmtdelta=0.05`; `nrmti` is the mesh index where $r<\texttt{fracinr}\cdot\texttt{rmt}$, i.e. an **array shape**; `ngridg` is $\lceil\cdot\rceil$ to a $\{2,3,5,7\}$-smooth integer; `nstfv` is clamped by $\min_k n_{\rm mat}$; `nsymcrys` and `nkpt` come from a symmetry search. **The same applies to $Z$**: `readspecies.f90:71` reads the electron configuration `nsp`/`lsp`/`ksp`/`occsp`/`spcore` as fixed integers and flags, `init0.f90:400-424` sums `occsp` over the non-core states into `chgval`, and `init1.f90:319` makes $n_{\rm stfv}=\mathrm{nint}(\texttt{chgval}/2)+n_{\rm empty}+1$ — an **array shape** that is a step function of $Z$, on top of a per-element core/valence partition, a species-dependent `rmt` and a per-species `nrmt` | The gradient is silently the **fixed-basis** gradient. Nothing warns. A small-step FD check never crosses a step and therefore **confirms** the wrong answer. Worst case: an optimiser walks across a symmetry-lowering boundary and the energy jumps | freeze everything and state the validity region: fixed `rmt`, frozen `ivg`, `reducek=0`, `tshift=False`, symmetry frozen or off. **Note `trmt0=.true.` is insufficient** — it restores `rmt0` *before* `checkmt` re-adjusts, so a port must skip `checkmt` and assert $\texttt{mtdmin} > \texttt{rmtdelta}$ as a precondition. Test structurally: assert shapes bitwise unchanged between geometries | standard mitigation, real loss of generality |
| **E** | **Occupation smearing and the Fermi solve.** The bisection differentiated literally | Measured 46% wrong, **same sign, same order**, and identical at 40/60/100/200 iterations — so tightening convergence does not move it | `custom_jvp` with $d\mu/d\varepsilon_i = w_if'_i/\sum_jw_jf'_j$ (measured $1.2\times10^{-7}$) | standard |
| **F** | **`stype != 3` breaks stationarity.** `energy.f90:242` computes the entropy only for Fermi–Dirac, so `engyts = 0` for Gaussian/MP | Forces on an MP-smeared metal wrong by the omitted $d(E_{ts})/d\mathbf R$, growing with `swidth`, converging normally | restrict to `stype=3` and enforce it, or implement the generalised entropy per smearing. **Test $dE/df_i = 0$ directly** | standard, but must be noticed |
| **G** | **Band crossings / degeneracy at $E_F$ in a metal.** Palenik & Dunlap: "Degenerate perturbation theory from quantum mechanics is inadequate in DFT because of nonlinearity in the Kohn–Sham potential" (*abstract only, not read in full*) | A gradient smooth on one side of a displacement and different on the other; or an optimiser oscillating. Measured proxy: AD tracks FD to $2.5\times10^{-7}$ down to $10^{-9}$ splitting, then degrades to $1.5\times10^{-3}$ | keep `swidth` above the smallest resolvable splitting; carry $df/d\mathbf R$. **Converge the gradient in `swidth`** and require a plateau | partly open research |
| **H** | **`where`-`NaN`.** Reverse mode evaluates both branches | Measured: `where(rho>1e-12, -rho**(1/3), 0)` has the *correct value* 0.0 and gradient `NaN` at $\rho=0$; the entropy guard `NaN`s at $f=0$ **and** $f=1$. One `NaN` poisons the whole reduction, with no indication which of $\sim2\times10^4$ grid points caused it | double-`where`: clamp the argument *inside* the branch. Audit every guard in one pass (native XC floors, `potxcmt`'s $|\rho_\uparrow-\rho_\downarrow|>10^{-8}$ division, Kübler's $\sqrt{\mathbf m\cdot\mathbf m}$ — floored under `dncgga` for GGA but **not** LSDA, `rdiracint`'s $10^{100}$ overflow guard). Note the Kübler floor is a change of physics at $|\mathbf m|\to0$, not only a guard (§3.3) | standard |
| **I** | **Non-converged SCF whose forward value looks converged.** Convergence uses the *smoothed* $\tilde E$ | The energy stops moving while the potential still oscillates; the gradient — far more sensitive, because $E$ is stationary and $\nabla E$ is not — is wrong by more than the energy error suggests | gate the differentiable entry point on the **residual**, never on the energy; converge the gradient in `epspot` | standard |
| **J** | **Precision.** `complex64` enlarges the set of pairs for which A fires | Measured: gradient $-1.047\times10^{3}$ (f64) vs $-1.104\times10^{3}$ (f32) at $10^{-4}$ splitting; 22% at $10^{-5}$; **sign flip** at $10^{-6}$ | `complex128` for the eigensolve and everything downstream in the gradient, even where the forward pass tolerates `complex64`. Elk's own single-precision choices are safe forward and **not** safe in reverse. Run the whole suite in both and require $10^{-4}$ agreement | standard, but costly (pins to datacentre GPUs) |
| **K** | **Reverse-mode memory / compile time** | OOM, or a compile that never finishes | implicit diff removes the iteration axis; `remat` the H/S assembly; `lax.map` over k-chunks. Assert the traced equation count does not grow with `maxscl`. **Measure it** (§6, item 0e) rather than extrapolating from the 1.17–1.42x / 0.21–1.06 s figures, which are $n\le400$, 4-k-point numbers and say nothing about a step containing per-k eigensolves at $n\approx3000$ | standard |
| **L** | **Core states via node-count shooting** | A finite, smooth, wrong core contribution. Large for heavy elements (W's 1s at $-2547.66$ Ha, 13 core states), nearly invisible on Si — so the bug ships from a silicon test suite | `custom_vjp` with $dE_c/dV = g_c^2+f_c^2$. Test on W or Bi | standard, exact |
| **M** | **Frozen linearisation energies** | Disagreement with a very careful FD that lets $E_\ell$ re-converge at each displaced geometry | `stop_gradient` and **say so**. Defensible: $E_\ell$ is a basis parameter, the energy is variational in the coefficients, and Elk's own `force.f90` makes the same assumption. `autolinengy` ($E_\ell = E_F + \texttt{dlefe}$) is the differentiable alternative | standard |

### 8(f) What differentiability buys, and what it costs

**Buys — in rough order of how much it is worth to this project.** Each carries the hazards that
gate it and the validity region it inherits; a "buy" with an unstated validity region is how §21's
cesium result happened.

1. **An analytic stress tensor.** Elk has none; `genstress.f90` runs up to 7 full SCF calculations
   and divides a difference of $\sim10^4$ Ha energies by 0.005. One reverse pass, better
   conditioned (caveat in §8c).
2. **A unified second-derivative machinery** replacing one code path each for Born effective
   charges (tasks 208/209/478), the dielectric tensor, piezoelectric and magnetoelectric tensors
   (tasks 380/390) and elastic constants. Adding a new one becomes a change of argument.
   *Gated on Phase 0a′ (B2): second order through a fixed point containing `eigh` is untested.*
3. **$q=0$ phonons with spin and spin-orbit**, which upstream Elk cannot do at all
   (`phonon.f90` hard-`stop`s on `spinpol`; `deveqnsv`/`deveqnss` are commented out).
   *Same gate as item 2; and see §8(a) on why term-by-term agreement with task 205 is not to be
   expected.*
4. **A GGA $f_{xc}$**, which Elk does not have (only `lda_fxc` via libxc) — genuinely new physics,
   enabling GGA-level linear response.
5. **$d(\text{anything})/d(\texttt{soc\_scale})$**, free. §§20/21/23 had to *sweep* this by brute
   force at `soc_scale=3000`, and the cesium retraction came from an under-resolved sweep.
6. **Effective masses** with neither a step size nor a state-sum truncation (§25 measured
   +3.1% error from truncation, converging from above). **Not "exact" at a degenerate band edge**:
   a mass is a second derivative of an *individual* band, and §8(b) shows that object is ill-posed
   inside a multiplet — Si's valence-band top is the degenerate $\Gamma_{25'}$ triplet §25 itself
   discusses, and needs a degenerate-$\mathbf k\!\cdot\!\mathbf p$ $3\times3$ block treatment, not
   three separate second derivatives.
7. **The quantum geometric tensor from $\partial u/\partial\mathbf k$ directly**, deleting §15's
   Löwdin-normalisation apparatus, its centered-stencil derivation, and the $\sim10^{-3}$
   `genolpq` truncation floor that made the raw metric *diverge* as $dk$ shrank (measured
   $g_{11}$: 40.5, 48.2, 52.5, 62.9, 131.8). **Hazards A and B, not merely a caveat:** this must
   be written in projector form, the K/K$'$ valleys of h-BN and WSe$_2$ are exactly where
   near-degeneracies live, and a window boundary inside a multiplet there is the ill-posed case
   the safe rule launders.
8. **An infinitesimal-rotation exchange constant by Hessian**, cheaper than §29's 36 constrained
   SCF runs (~60 CPU-hours, and `CLAUDE.md` records the nine-component NiO sweep as never having
   been run) — but note it is **a different quantity**. A Hessian of the energy in infinitesimal
   spin rotations about a collinear reference is the Liechtenstein-type magnetic-force-theorem
   $J$; §29's four-state method is a *large-angle total-energy mapping*. They coincide only when
   the mapping is exactly bilinear and the moment magnitude is rotation-independent — the
   assumption the four-state construction exists not to need. The two agreeing would be a good
   test of that assumption; substituting one for the other is not a free win. Differentiating the
   four-state energies *themselves* is a separate and harder problem: `fsmtype=-2` makes the
   constraining field an unknown fixed by the `mommtfix` constraint, so it is the augmented
   $(v,\mathbf B_{\rm fsm})$ system §8(a) declines to derive, on magnetic metals, i.e. squarely
   inside hazard G. The same gap applies to §29's `bforb` limitation. **Untested, and the least
   supported item in this list.**
9. **$dE/dZ$ at fixed electron configuration, fixed core/valence partition and fixed radial
   mesh** — a well-defined and useful derivative, available *only* in an all-electron code,
   because the radial solver integrates the Dirac equation in the bare nuclear potential with the
   actual $Z$. It is **not** the full alchemical path, which crosses integer boundaries in
   `occsp`/`spcore`/`chgval`/`nstfv`/`nrmt` (hazard D). Still the strongest scientific argument
   for doing this in Elk rather than a plane-wave code.
10. **Highly parameterised ML-XC functionals trained end-to-end**, the capability the 1D and
    molecular literature (Li *et al.* PRL 126, 036401; Kasim & Vinko PRL 127, 126403) shows buys
    real statistical efficiency. This is the item that genuinely needs reverse mode; few-parameter
    fitting does not (DFTK did it forward).
11. **Gradient-based inverse design** over structure, with objectives other than the energy —
    **only within a fixed discrete structure** (frozen `rmt`, frozen `ivg`, frozen symmetry).
    Hazard D means the objective is discontinuous across those boundaries and nothing warns; an
    optimiser walking across a symmetry-lowering boundary sees the energy jump. Low-dimensional
    inverse design does not require the full port at all (DFTK, forward mode).

**Costs:** `complex128` throughout the gradient (hazard J), which pins a differentiable port to
datacentre hardware and removes most of the GPU argument; a validity region bounded by frozen
`rmt`, frozen G-sets and frozen symmetry (hazard D); loss of libxc's C binding (narrower than it
sounds — see B7 — but it costs the $\tau$ plumbing and rules out potential-only functionals
entirely); the band gap as a
differentiable objective (it is a max/min over eigenvalues, non-differentiable at crossings —
DFTK concede exactly this, cited); and a permanent requirement that every gradient-producing
capability ship with its own adversarial test, because the failure mode is a plausible number.

---

## 9. Relation to elkpy

### 9.1 The twelve patches in a JAX world

| Patch | In a JAX port |
|---|---|
| 0001 per-species SOC scale | **Trivial, and becomes differentiable** — an array argument, and $d(\text{anything})/d(\texttt{soc\_scale})$ becomes one JVP instead of the brute-force sweeps §§20/21/23 had to run |
| 0002 Berry curvature (Wilson loop) | **Unnecessary.** `genolpq` exists as a Fortran export only because there is no other way to get wavefunctions out; with resident arrays the overlap is an `einsum`, and the $\sim10^{-3}$ truncation floor becomes a choice of quadrature |
| 0003 eigenstate session (task 9002) | **Evaporates entirely** — the most architecturally invasive of the twelve. `docs/design.md` §14 says explicitly it exists because Elk's per-query setup is file-mediated and process-bound |
| 0004 atom projection, 0005 orbital projection, 0006 angular momentum | **Trivial** — `einsum`s over the MT wavefunction with a radial weight and an $\ell$ mask, ~20 lines each. §18's `lmaxi` guard becomes a property of the two-block layout |
| 0007 momentum matrix elements | **Half free.** `genpmatk`'s hand-coded gradient becomes `jacfwd(H, k)`, making the Hellmann–Feynman check $v_{nn}=\partial\varepsilon_n/\partial k$ true by construction. The $(1/4c^2)[\boldsymbol\sigma\times\nabla V_s]$ SOC term is a physical addition, not a derivative — still hand-written |
| 0008 inversion parity, 0010 rotation operators | **Still hard.** `rotzflm`/`ylmrot` acting on the LAPW basis is genuine symmetry algebra that must be ported, not derived. 0010's `nspinor=1` restriction (the SU(2) factor is never applied) is a physics gap a port would have to *close*, and its unresolved $C_3$ generator-direction ambiguity is a convention question JAX does not answer |
| 0009 (`MOMENTUM` also returns `evecsv`) | **Unnecessary** — there is one diagonalisation and it returns everything, so §14's cross-diagonalisation degenerate-basis hazard cannot arise |
| 0011 spin-polarised STM | **Trivial** — `rhomagv` already computes $\mathbf m(\mathbf r,E)$ and discards it; a functional code returns it. The upstream `wkpt`-squared double count elkpy found cannot arise without a global accumulator |
| 0012 vertical transport | **Trivial** — the closed-form Gram matrix is a gather plus one `zgemm`; the energy sweep is a `vmap` |

### 9.2 The hybrid, and exactly where its boundary is

**The pragmatic answer for this repository today is a hybrid**: keep Elk's Fortran SCF, expose
$H(\mathbf R; v^*)$ and $S(\mathbf R)$ as JAX functions of a *converged, frozen* potential
(reachable through the existing task-9002 session or a checkpoint), and do new physics in JAX on
top.

**What the hybrid delivers, with no fixed point at all** — because differentiating at fixed
potential, or with respect to $\mathbf k$, needs no SCF derivative:

- $d\varepsilon_j/d\mathbf R = c_j^\dagger(dH/d\mathbf R - \varepsilon_j\,dS/d\mathbf R)c_j$,
  i.e. `forcek.f90`'s content, from `jax.grad` — for non-degenerate bands, or multiplet-summed
  (§8b);
- $\partial u/\partial\mathbf k$ exactly, hence Berry curvature, the quantum metric and the full
  quantum geometric tensor with no $dk$, no Löwdin renormalisation, no stencil order (§8f item 7)
  — **written in projector form**, because h-BN's and WSe$_2$'s K/K$'$ valleys are exactly where
  near-degeneracies live and a window boundary inside a multiplet there is hazard B;
- momentum matrix elements as $\partial H/\partial\mathbf k$, making §22's Hellmann–Feynman check
  exact by construction;
- every operator in §§16–19 as an `einsum`, on one diagonalisation.

**Cost:** roughly 13–15 weeks, summed from the §3 rows it needs — `match` (1 wk),
`sbessel`/`genylmv` (1 wk), `gengkvec`/`gensfacgp` (3 d), the radial functions and integrals
(2 wk), H/S assembly (3–4 wk), the Cholesky-reduced eigensolve (2–3 wk), plus about a week of the
`readstate` row for a `STATE.OUT` reader. That last item carries a risk the row does not surface:
`STATE.OUT` is an unformatted Fortran binary whose record layout is version-coupled, so it is the
piece most likely to break on an Elk upgrade.

**What the hybrid cannot deliver** — anything requiring $dv^*/d\theta$: second derivatives
(phonons, Born charges, elastic constants), response functions, high-dimensional ML-XC training,
and reverse-mode inverse design. **The full port is justified if and only if those are the goal**
— and note the list is shorter than it first looks, since DFTK's forward-only precedent already
covers low-dimensional inverse design and few-parameter functional fitting. That is the most
useful sentence in this section for this repository.

A third option worth naming: a JAX kernel library called *from* Fortran (the FLEUR/cuBLAS-XT
pattern, cited: ~100 lines of wrappers for a 5x win) is the GPU answer that does not require the
port at all — but it delivers no differentiability, so it belongs to the motivation this document
recommends against pursuing on its own.

---

## 10. Where the readers disagreed

Six subsystem surveys, one prior-art scout and two autodiff lenses produced this document's
inputs. They disagreed in eight places that matter. This section states each disagreement and what
was adopted. Where a disagreement was settled by one command against `vendor/elk/src/` — items 6
and 7, and the bare-`stop` count now in B9 — it is reported as *measured* rather than preserved as
a range; carrying an unverified label on a trivially checkable count costs more than it saves.

1. **`NaN` versus finite garbage at degeneracy.** Most subsystem surveys and the prior-art scout
   reported `NaN`; the eigensolver survey independently *measured* finite garbage
   ($4.3\times10^{14}$, "fails SILENTLY"); the adversarial lens supplied the mechanism.
   **Adopted: both, split by cause.** Finite garbage from a *physical* degeneracy, because a
   realistic assembly splits a degenerate pair by $\sim10^{-15}$ rather than bitwise; `NaN`
   wherever the degeneracy is bitwise — which includes §3.2's own identity padding, measured
   here, so the `NaN` case is not confined to toy matrices after all. §3.2 now pads with distinct
   values, which moves the padding case out of the `NaN` regime by construction.
   **Refined by Phase 0b** (`docs/jax_port_phase0.md`): the split is not by cause but by
   *eigensolver*. On one Cholesky-reduced matrix with an engineered pair, LAPACK returns a
   $1.5\times10^{-14}$ splitting and XLA returns the pair bitwise equal — so the same code on
   the same input gives finite garbage at $n=400$ and `NaN` at $n=1000$. Padding with distinct
   values removes one source of bitwise degeneracy; it does not remove this one.
2. **The force recipe.** Three incompatible proposals (§8c). **Adopted: Harris–Foulkes**, with the
   other three shown, because it needs no eigenvector derivative and its VJP is literally
   `forcek.f90`.
3. **Why not to unroll the SCF.** The architecture lens said accuracy; the adversarial lens
   *measured* $1.9\times10^{-5}$ relative error at Elk's own `epspot`. **Adopted: tape size and
   mixer-dependence, not accuracy** — this changes which mitigation is correct.
4. **Where the runtime is.** Survey 1 read 34.4% "charge density" from `INFO.OUT`; survey 5
   *measured* 3.5–15% serially and showed the large figure is an **83x artefact** of nested
   OpenMP plus FFTW planning/destroying a plan per transform inside a global critical section
   (48.06 s at `maxlvl=4` vs 0.58 s at `maxlvl=1`). Survey 4 measured a bcc-W run in which a
   *one-time* setup (`allatoms`) was 79% of wall time. **Adopted: no single split.** Elk's timers
   also exclude `gencore`, `linengy`, `genapwlofr` and `gensocfr` entirely, and `timetot` is a sum
   of timers, not wall clock. Any port should measure its own target cell before optimising.
5. **Effort.** "2–4 weeks for the spine once the callees are pure" (survey 1), "the modmain
   refactor *is* the project" (also survey 1), "4–7 person-months for the basis/eigensolver
   subsystem" (survey 2), "2–3 person-years for the spine" (survey 6). These differ by *what they
   assume is already ported*. **Adopted: no single number, but the arithmetic is now printed.**
   §3's per-row efforts sum to roughly 45 weeks $\approx$ 10.4 months (excluding the deferred
   `eveqnfvr` month and the 6–9-month DFPT row); §6's phases sum to 9.5–12.75 months to Phase 4.
   Those agree to within their own precision — an earlier claim that §3 summed to "far smaller"
   than §6 was simply wrong arithmetic. What *neither* includes is §5's scaffolding, which B3
   calls the dominant cost; it is now §6's Phase S, carrying survey 6's 2–3 person-years as an
   inference rather than being invisible.
6. **Task-code count.** Survey 1 said 88 `case` clauses covering 148 task codes; survey 6 said 134
   distinct codes in `elk.f90:109-270`. **Resolved by measurement, not adopted as a range**: the
   `select case(task)` block runs `elk.f90:109-289` and contains 87 `case(...)` clauses plus
   `case default`, covering **148 distinct codes**. Survey 6's figure came from a line range that
   truncates the block. §7 now counts producers and consumers against 148 explicitly.
7. **DFPT line counts.** 1622 for the 11 core files named in §8(c) (re-counted here; an earlier
   "1790 for 12 core files" did not reconcile with its own file list), 2540 across 29 files, 4733
   exclusive of which ~3155 is the $q\neq0$ spine. **Adopted: all three, labelled by scope.**
8. **Whether any periodic DFT code already has implicit differentiation.** The autodiff
   architecture lens said reverse-mode fixed-point differentiation is offered by "no periodic DFT
   code in any language"; the prior-art scout *verified* that PySCFAD contains
   `pyscfad/pbc/{scf,dft,gto,df,tools}` and `implicit_diff.py`, and is published as covering
   "molecules and materials". **Adopted: the scout's finding**, which narrows this document's
   claim from "first differentiable periodic DFT" to **"first differentiable all-electron DFT"**.
   Whether PySCFAD's reverse mode is actually exercised through a periodic SCF was not verified.

**One correction this study makes to several surveys.** Multiple readers stated
$g_{\max} = \texttt{rgkmax}/\min_s(\texttt{rmt})$. Read directly: `readinput.f90:198` sets
`isgkmax=-1` by default, and `init0.f90`'s `select case(isgkmax)` takes the `case(-1,0)` branch,
$g_{\max} = \texttt{rgkmax}\,/\,\big[\sum_s n_s\texttt{rmt}_s/n_{\rm atm}\big]$ — the
**natoms-weighted mean** radius, not the minimum. This strengthens hazard D rather than weakening
it: $g_{\max}$ depends on *every* species' radius, and every radius depends on atomic positions
through `checkmt`. Relatedly, `trmt0=.true.` (`readinput.f90:395`) restores `rmt0` at the *top* of
`checkmt`, *before* the adjustment loop runs, so "freeze the radii with `trmt0`" is insufficient;
a port must skip `checkmt` and assert `mtdmin > rmtdelta` as an input precondition.

---

## 11. References

**Verified — read in this study or in the surveys that produced it.**

- Elk 11.0.2 source, `vendor/elk/src/`: 821 files, 74,714 lines; 651 files contain `use modmain`;
  515 bare `stop`s across 171 files; `modmain.f90` 1297 lines, 654 declared module-level names of
  which 159 are allocatable or pointer, zero `contains` (all re-counted for this revision).
  Specific lines cited above: `gndstate.f90:133-320` (the SCF loop; `gencore` 154, `genvsig` 218,
  `energy` 230, convergence 295–314, `ksgwrho` 184, DFT+U 192–198, FSM 215–216),
  `init0.f90:400-424` (`chgval`), `:483-487` (`npsd`), `:633-645` (`vsbs` layout), `:696-699`
  (`vmixer => vsbs`), the `isgkmax` select case, `init1.f90:319-330` (`nstfv` clamp),
  `readinput.f90:78-110,158,197-198,240-241,286,395` (defaults),
  `energy.f90:112,226,242`, `occupy.f90`, `checkmt.f90`, `force.f90:55-74` (the IBS docstring) and
  `:142-178` (the $\int v_s\nabla\rho$ and $\int\mathbf B_s\!\cdot\!\nabla\mathbf m$ blocks),
  `potks.f90` (`gentau`, `xc_c_tb09`, `projsbf`, `trimrfg`, `rfirftoc`), `rfirftoc.f90`,
  `genvsig.f90`, `gencfun.f90`, `zpotcoul.f90` (the pseudocharge docstring), `gengclg.f90:12`,
  `rhomag.f90`/`rhocore.f90`/`rhonorm.f90`, `gencore.f90` (the core docstring),
  `gensocfr.f90` (the SOC docstring), `hmlaa.f90:47-55` (the parity rule),
  `hmlrad.f90:55-95`, `match.f90:78-93` + `genylmv.f90` (the `t4pil` prefactor),
  `modxcifc.f90:435-495` (native `xcgrad` values), `zhegvxi.f90`/`zhegvxsi.f90`,
  `genstrain.f90`/`genstress.f90`, `dhmlrad.f90` (unperturbed `apwfr` against `dvsmt`),
  `deveqnfv.f90:96-103` (`epsdev`), `bfieldfsm.f90`, `modtest.f90`/`modvars.f90`,
  `elk.f90:109-289` (87 `case` clauses + `case default`, 148 distinct codes), `phonon.f90` (the
  `spinpol` stop and the commented-out `deveqnsv`/`deveqnss`), `vendor/elk/species/{Si,C}.in`
  (`apword=1`; the APW+lo local-orbital pattern; `spzn` $=-Z$).
- JAX experiments run for this revision (JAX 0.7.1, CPU, x64): `custom_root`+`gmres` reverse
  transposition failure; `custom_root`+dense and `custom_vjp`+`gmres` both matching FD to
  $3.4\times10^{-10}$; `jax.hessian` through a `custom_vjp` fixed point (with and without a
  `lax.while_loop` primal) matching FD of the implicit gradient to $2.5\times10^{-9}$; the
  identity-padding `NaN`; the individual-eigenvalue-versus-multiplet-trace measurement.
- `jax_xc` 0.0.11 wheel, inspected directly: 628 functionals including `mgga_x_scan`,
  `mgga_c_scan`, `mgga_x_tpss`, `mgga_x_r2scan`, `mgga_x_tb09`, `hyb_gga_xc_b3lyp`,
  `hyb_gga_xc_hse06`.
- Sjöstedt, Nordström & Singh, *An alternative way of linearizing the augmented plane-wave
  method*, Solid State Commun. 114, 15 (2000) — APW+lo, which is what Elk's shipped species files
  give.
- Weinert, *Solution of Poisson's equation: beyond Ewald-type methods*, J. Math. Phys. 22, 2433
  (1981) — as cited by `zpotcoul.f90`'s own docstring.
- Koelling & Harmon, *A technique for relativistic spin-polarised calculations*, J. Phys. C 10,
  3107 (1977) — as cited by `gensocfr.f90`.
- Fabregat-Traver, Davidović, Höhnerbach & di Napoli, *High-performance generation of the
  Hamiltonian and Overlap matrices in FLAPW methods*, arXiv:1602.06589, Comput. Phys. Commun. —
  H/S setup + eigensolver >80% of runtime; 98.06–99.75% of setup in `zherk`/`zher2k`/`zgemm`;
  80–90% of BLAS peak.
- Fabregat-Traver *et al.*, *Hybrid CPU-GPU generation of the Hamiltonian and Overlap matrices in
  FLAPW methods*, arXiv:1611.00606 (2016) — ~100 lines of cuBLAS-XT wrappers, 5x over optimised
  CPU / 7.5–12.5x over original FLEUR, on 2x K20Xm / K80.
- SIRIUS, github.com/electronic-structure/SIRIUS — C++17 + CUDA/ROCm, PP-PW and FP-LAPW,
  "designed for GPU acceleration of popular community codes such as Exciting, Elk and Quantum
  ESPRESSO". Kozhevnikov, ICTP Trieste, 29 May 2015, for the PRACE-2IP WP8 origin and the
  LAPW-overlap → ZGEMM → blocked async `cudaZgemm` transformation.
- Zhang, Kozhevnikov, Schulthess, Cheng & Trickey, *Performance Enhancement of APW+lo Calculations
  by Simplest Separation of Concerns*, Computation 10(3), 43 (2022) — Exciting-Plus (an Elk
  1.0.17 derivative) interfaced to SIRIUS. **Note: the abstract credits task parallelism, not GPU,
  for the measured gain; the full text was not obtained.**
- Zhang, Kozhevnikov, Schulthess, Trickey & Cheng, *All-electron APW+lo calculation of magnetic
  molecules with the SIRIUS domain-specific package*, J. Chem. Phys. 158, 234801 (2023).
- Schmitz, Ploumhans & Herbst, *Algorithmic differentiation for plane-wave DFT*, npj Comput.
  Mater. 12, 6 (2025), arXiv:2509.07785 — DFTK. Forward mode only; the SCF derivative hand-written
  as DFPT and solved by GMRES with Sternheimer equations; reverse mode deferred; feasibility
  credited to "only about 10 000 lines of Julia code"; the band gap conceded non-differentiable
  at crossings.
- Li, Hoyer, Pederson, Sun, Cubuk, Riley & Burke, *Kohn-Sham Equations as Regularizer*, PRL 126,
  036401 (2021) — 1D; the strongest published argument that differentiating through the KS solve
  buys statistical efficiency in functional learning.
- Kasim & Vinko, *Learning the Exchange-Correlation Functional from Nature with Fully
  Differentiable DFT*, PRL 127, 126403 (2021); Kasim, Lehtola & Vinko, *DQC*, J. Chem. Phys. 156,
  084801 (2022) — PyTorch, molecular, 3D.
- Zhang & Chan, *Differentiable quantum chemistry with PySCF*, J. Chem. Phys. 157, 204801 (2022) —
  PySCFAD; the hybrid pattern (keep the optimised kernel, register its analytic derivative) and a
  worked `implicit_diff.py`.
- Li, Lin, Hu, Zheng, Vignale, Kawaguchi, Castro Neto, Novoselov & Yan, *D4FT*, ICLR 2023,
  arXiv:2303.00399; Li *et al.*, *Diagonalization without Diagonalization* (Jrystal),
  arXiv:2411.05033 — direct minimisation replacing SCF, which is how the JAX DFT codes avoid
  `eigh` entirely. Jrystal is plane-wave, "all-electron" only in the bare-Coulomb sense, tested on
  fcc Al and Si.
- JAX 0.7.1 source: `jax/_src/scipy/linalg.py` (`"Only the b=None case of eigh is implemented"`),
  `jax/_src/lax/linalg.py` (`"subset_by_index not supported on CPU and GPU"`; `_eigh_jvp_rule`'s
  `Fmat`). Issues jax-ml/jax#5461 (open since 2021-01-19) and #669 (open since 2019-05-04).
- Measurements from this study's sessions: JAX experiments at
  `.../scratchpad/ad/FINDINGS.md`; Elk timing runs at `.../scratchpad/` (`si`, `si1`, `si8`, `fe`,
  `symA/B`, `kA/kB`); eigensolver benchmarks at `.../scratchpad/eigensolver_survey_jl/FINDINGS.txt`;
  radial-solver benchmarks at `.../scratchpad/bench/`. These are session-scratch paths, not
  repository artefacts.

**Unverified — carry the label.**

- **cuSOLVER batching behaviour.** An earlier draft asserted that `syevd` is looped per batch
  element above $n\approx32$. Review reports the opposite for current `jaxlib` —
  `gpusolverDnXsyevBatched` above $n=32$ on CUDA $\ge$ 12.6.2, chunked at
  $\texttt{INT\_MAX}/(n^2\cdot\text{bytes})$ — which was **not verified here**; no CUDA `jaxlib`
  was available. Performance at $n\approx3000$ is unmeasured by anyone. Still the first benchmark
  (§6, item 0d).
- **FP64 throughput on post-Hopper datacentre GPUs.** Reported to fall sharply relative to H100
  (which would strengthen §1 fact 3); the H100 PCIe figures quoted are from the datasheet and are
  correct, but they are a generation old. Not verified here.
- **$\kappa(S)$ for Elk's APW+lo overlap matrix**, which sets the safe-$K$ tolerance (§8b) and the
  Cholesky-reduction backward error (B4). Never measured in this study; the recommendation is to
  compute it at runtime rather than assume a value.
- **Whether the second-order result generalises.** `jax.hessian` through a `custom_vjp` fixed
  point was measured on a 6-dimensional smooth toy with **no eigensolve in the loop**. That is not
  evidence about a fixed point whose every iteration contains `eigh` (§4.1 B2, §6 item 0a′).
- **`jax_xc`'s exclusion list**, and whether its `mo`-based kinetic-energy-density interface maps
  onto an LAPW muffin-tin/interstitial partition. The functional *coverage* above is measured;
  the LAPW integration is not.
- **Whether iterative subspace eigensolvers (LOBPCG, ChASE, Davidson) or $S$-metric direct
  minimisation change B4's arithmetic** at $n\approx3000$. Neither was benchmarked.
- **WIEN2k GPU support**: no native port was found; its GPU exposure appears to be via the
  GPU-accelerated ELPA eigensolver (Kůs *et al.*, arXiv:2002.10991). Unresolved.
- **IQC**, Xiaoyu Zhang, *End-to-End Differentiable Learning of a Single Functional for DFT and
  Linear-Response TDDFT*, arXiv:2602.05345 (2026) — claims AD through both the SCF *and* the
  eigenvalue problems. **Abstract only; no repository located.** Its degeneracy handling is
  precisely the open question in §4.1 and it should be read before committing to a design.
- **Palenik & Dunlap**, PRB 96, 045109 (2017) and PRB 94, 115108 (2016), on degenerate
  perturbation theory in DFT — **abstracts only**.
- **arXiv:2209.12747**, *Roadmap on Electronic Structure Codes in the Exascale Era* — downloaded
  but not read; verify its LAPW content before quoting.
- **The claim that no APW/LAPW/all-electron code exists in any autodiff framework** is an
  absence-of-evidence result: multiple targeted searches returned nothing. It is stated as "none
  was found", not as "none exists".
