# Phase 2 of the Elk-to-JAX port: measurements

Running log of what Phase 2 has actually measured, in the shape
`docs/jax_port_phase0.md` established and `docs/jax_port_phase1.md` continued:
what was at stake, what was run, and what is *not* settled by the result.
`docs/jax_port.md` §6 is the plan.

Phase 2 as written is "the density and potential half": `rhomag` via the
local-density-matrix route, Weinert Poisson, one LDA and one GGA functional as
scalar $\varepsilon_{xc}$, and symmetrisation. Its three stated criteria are a
forward total energy at fixed input potential, a `jnp.isfinite` check on the
full gradient where Elk's own guards fire, and $v_{xc}$ from `jax.grad` of
$\varepsilon_{xc}$ against Elk's hand-coded $v_{xc}$ pointwise.

**Phase 1 already took two bites out of Phase 2's territory and should be read
first.** §1k built the muffin-tin radial functions and integrals from the
Kohn-Sham potential — the plan had put those here — and §1l built the
interstitial characteristic function. What is left of the interstitial pair is
$V_s$ itself, which is what this phase is for.

| study item | status | where |
|---|---|---|
| **2a** the XC functional and its gradient | **done** — LDA (`xctype=3`) and PBE (`xctype=20`); `jax.grad` supplies PBE's functional derivative from the energy density alone | §2a, §2b |
| **2b** the density `rhomag` | **done for the valence density**, both regions, and on to Elk's converged `rhomt`/`rhoir`. The core-state SOLVER is not transcribed — `rhocr` is exported, an input at fixed potential like `vsmt` | §2h |
| **2c** the Weinert Poisson solver | **done** — `vclir` to 1.6e-15 relative, `vclmt` to 4e-20 ($l{=}0$) and 7e-14 ($l{>}0$) | §2e |
| **2d** a GGA functional | **done** — see 2a | §2b |
| **2e** symmetrisation | **done for the muffin tin** (`symrfmt`, exported as an operator). `symrfir`, the interstitial's, is not — so a symmetry-reduced mesh is out of scope | §2g |
| **2f** total energy at fixed input potential | **done for the density-functional terms**, to <1e-13 each. `evalsum`, `engyts` and `engynn` are imported: they need the second-variational step, a zone sum, and the lattice | §2f |

Everything above starts from Elk's own converged density. Two further sections
ask whether the pieces **compose**, which no study item does:

| | |
|---|---|
| §2i | the Kohn-Sham potential end to end, $v_{\rm cl}+\hat Sv_{xc}$ from three modules against Elk's own `vsir` |
| §2j | the loop closed — potential → $H,O$ at every $k$ → eigensolve → density, with nothing in the path reading an eigenvector |

Sections in order, since the numbering follows the work rather than the study:
§2a LDA · §2b PBE · §2c cell integrals · §2d the angular transform and
`symrfmt`'s effect · §2e Poisson · §2f the total energy · §2g `symrfmt` itself ·
§2h the density from the eigenvectors · §2i the potential composed · §2j the
loop closed.

Patches added by this phase: **0016** (`GROUNDSTATE`), **0017** (Poisson inputs
and `energy.f90`'s scalars), **0018** (`symrfmt`'s operator), **0019**
(`DENSITYK`), **0020** (the density's post-processing), **0021** (the core
density and `chgtot`).
---

## 2a. The exchange-correlation functional

### What was at stake

`docs/jax_port.md`'s Phase 2 gradient criterion is "$v_{xc}$ from `jax.grad` of
$\varepsilon_{xc}$ against Elk's own hand-coded $v_{xc}$ pointwise, to
$10^{-10}$". That is worth having — Elk differentiates the parameterisation by
hand through a long chain ($d\varepsilon/dr_s$, $d\varepsilon/d\zeta$,
$dr_s/d\rho$, $d\zeta/d\rho_\sigma$) and AD does not — but on its own it is an
**internal** check. It would pass just as well if the wrong functional had been
transcribed, which is the failure mode Phase 0 kept meeting.

So the functional is pinned three further ways, two of them against closed
forms that owe nothing to Elk.

### What was built

`src/elkjax/xc.py` transcribes `xc_pwca.f90` — the Perdew-Wang
parameterisation of the spin-polarised Ceperley-Alder electron gas (Perdew &
Wang, *Phys. Rev. B* **45**, 13244 (1992); Ceperley & Alder, *Phys. Rev. Lett.*
**45**, 566 (1980)), which is `xctype=3`, elkpy's `xc="PW"`, and Elk's own
default. Both the energy densities and Elk's hand-coded potentials, plus
`exc_density` for `jax.grad` to differentiate and `naive_pwca` for the hazard
below.

### The four checks, and what each alone cannot catch

**Exchange against Dirac.**
$\varepsilon_x=-\tfrac34(3/\pi)^{1/3}\rho^{1/3}$ and $v_x=\tfrac43\varepsilon_x$,
exactly (1e-15 relative). This pins $r_s$, the prefactors $p_1,p_2$ and the
$\zeta=0$ limit of the spin interpolation, and says nothing at all about
correlation.

**Exchange spin scaling.**
$\rho\varepsilon_x(\rho_\uparrow,\rho_\downarrow)=
\tfrac12[2\rho_\uparrow\varepsilon_x(2\rho_\uparrow)
+2\rho_\downarrow\varepsilon_x(2\rho_\downarrow)]$ and
$v_{x\sigma}=v_x(2\rho_\sigma)$, exact at any polarisation (1e-14). This is the
only check that exercises the $(1\pm\zeta)^{4/3}$ machinery; the unpolarised
case leaves it sitting at $\zeta=0$ where a wrong exponent is invisible.

**Correlation against Gell-Mann–Brueckner.** As $r_s\to0$,
$d\varepsilon_c/d\ln r_s\to(1-\ln2)/\pi^2$ (*Phys. Rev.* **106**, 364 (1957)),
which is the coefficient PW92's $A_0$ is fitted to reproduce
($0.0310907$ against $0.03109069$). Measured: 0.03104, 0.031084, 0.0310899 at
$r_s=10^{-3},10^{-4},10^{-5}$ — converging on it. This is the one analytic
anchor available on the correlation half, and neither exchange check touches
it.

**`jax.grad` against the hand-coded potential.** The study's own criterion, in
both spin channels and at $\zeta=0,0.3,0.85$: agreement is $<10^{-12}$
relative, against a stated tolerance of $10^{-10}$.

### The $\rho\to0$ guard, which is the hazard the study names

`xc_pwca` returns zero for $\rho<10^{-20}$. The obvious transcription is

```python
jnp.where(rho < RHO_MIN, 0.0, f(rho))
```

and it gives the correct **value** and a `NaN` **gradient**. `jnp.where`
evaluates both branches; the live one contains $r_s\propto\rho^{-1/3}$, whose
derivative at $\rho=0$ is infinite, and the vector-Jacobian product then
multiplies that infinity by a zero cotangent. Measured on
$\rho=(0,10^{-30},10^{-22},10^{-19},10^{-6})$: values bitwise identical between
the two forms, gradients identical on the last four entries, and `NaN` versus
`0.0` on the first. `naive_pwca` keeps that form deliberately so the failure is
asserted rather than described.

Note **where** it bites, since this is narrower than "vacuum": the value has to
be exactly zero, not merely tiny — at $10^{-30}$ the naive form is already
fine. A zero-initialised or zero-padded density array is exactly that case, and
so is any masked region, which is the study's §3.2 padding layout again.

Two things left as Elk leaves them. At full polarisation ($\zeta=1$) the
exchange term forms $(1-\zeta)^{4/3}/(1-\zeta)$, which is $0/0$ in Elk's code
too; patching it here would make this transcription disagree with the reference
it is checked against, and no Kohn-Sham density reaches it. And the guard makes
the derivative *discontinuous* across $\rho=10^{-20}$ rather than merely
finite — irrelevant for the XC energy, whose $v_{xc}\propto\rho^{1/3}\to0$, but
not for anything that differentiates $\varepsilon_{xc}$ itself, where the true
derivative diverges as $\rho^{-2/3}$ (measured $-1.1\times10^{12}$ just above
the cutoff).

### Against Elk's own $v_{xc}$: exact, once one more Elk step is reproduced

All four checks above are statements about the *form* of the functional. None
says it is the functional Elk used.

Patch **0016** adds a `GROUNDSTATE` query — the converged density and
potentials on the grids Elk holds them on, with no $k$-point, since this half
of the calculation is $k$-independent. `vxcir` is then the sharpest possible
reference: Elk evaluates the functional **pointwise** on the real-space FFT
grid, so it is literally this transcription applied to `rhoir`.

It agrees to **4.4e-16 absolute** — but only after one further Elk step is
reproduced. `potks.f90` passes `vxcir` through `trimrfg`, which zeroes every
Fourier component with $|\mathbf G| > 2k_{\max}$ (equivalently, every
G-vector whose index exceeds `ngvc`, Elk's list being sorted by $|\mathbf G|$).
Without that filter the same comparison stops at **2.5e-5 relative**, which
looks exactly like a mediocre transcription and is not one.
`elkjax.grid.trim` transcribes it, and the test asserts *both* numbers, so
"we reproduce Elk's $v_{xc}$" cannot quietly come to mean "to four digits" if
the filter is ever dropped.

**Two things the export carries that a transcription would not expect**, both
now in its docstring: `rhomt` includes the core density (`rhocore` adds it),
and `vclmt` includes the nuclear $-Z/r$ (`potnucl`). A third was a bug found
while writing it: `vsig` is allocated `ngvc` long, **not** `ngvec` —
`genvsig` builds it on the coarse grid — so the first version of the export
read past the end of the array.

**The muffin-tin side is in §2d below**, and it is where the one unexplained
measurement of this phase lives.

### Through `plot1d`, and why that tolerance is what it is

All four checks above are statements about the *form* of the functional. None
says it is the functional Elk used. The same test file also compares Elk's
$v_{xc}$ along a *line* through bulk silicon against $v_{xc}$ evaluated here on
Elk's density along the same line (`plot1d` puts both on the identical point
set, which the test asserts). That comparison is kept, even though the grid one
above supersedes it, because its number is worth knowing on its own.

Measured: **3.0e-5 median relative** over 200 points spanning
$\rho=2\times10^{-3}$ to $1.3\times10^{3}$, rising to 1.2e-2 in the shell
around $d\approx4.5$ Bohr.

**The residual is a commutation, not a transcription error**, and the mechanism
sets the tolerance. $v_{xc}$ is nonlinear, and neither of Elk's two
representations of a real-space function commutes with it: inside a muffin tin
Elk applies the functional on an angular grid and keeps $l_{\max}^{\rm o}$
harmonics of the *result*, while this comparison applies it to the truncated
*density*; in the interstitial Elk applies it pointwise on the FFT grid and
`plot1d` Fourier-interpolates the result, while this applies it to the
interpolated density. The 1.2e-2 shell is where the two representations meet.

It is still discriminating — correlation is 17–20% of $v_{xc}$ along this line
(asserted, not assumed) — but the point to carry forward is that **3e-5 is the
accuracy ceiling of any comparison made through Elk's plotting tasks**, four
orders of magnitude worse than the same comparison made on the grid Elk
computes on. Where an exported grid exists, use it.

### What this settles, and what it does not

`xctype=3` is done, forward and in gradient, with the study's own criterion met
at machine precision rather than at $10^{-10}$, and with the `NaN` hazard
turned into an assertion. The functional is a pure pointwise map, so it carries
no dependence on the rest of Phase 2 and will not need revisiting when the
density arrives.

**Not done**: a GGA (item 2d), which is a different shape — $\varepsilon_{xc}$
then depends on $\nabla\rho$, so the potential involves a divergence and the
`jax.grad` criterion becomes a functional-derivative check rather than a
pointwise one. And the study's second Phase 2 gradient criterion — `isfinite`
on the *full* gradient for the Cr monolayer and for a gapped insulator — is
only half addressed: the density-side guard is fixed here, and the
occupation-side one (occupations exactly 0 and 1) belongs with `rhomag`.


---

## 2b. PBE, and the functional derivative that costs nothing

### What was at stake

The LDA case (§2a) is a pointwise map, so `jax.grad` reproducing Elk's
hand-coded $v_{xc}$ there is a check on one chain rule. A GGA is the case that
actually tests the study's premise. Its energy density depends on
$\nabla\rho$, so the potential is a **functional** derivative,

$$v_{xc}(\mathbf r) = \frac{\partial(\rho\varepsilon_{xc})}{\partial\rho}
 - \nabla\cdot\frac{\partial(\rho\varepsilon_{xc})}{\partial\nabla\rho},$$

and Elk gets it from Perdew's own hand-derived expression — which needs
$\nabla^2\rho$ and $(\nabla\rho)\cdot(\nabla|\nabla\rho|)$ as *extra inputs*,
computed by `ggair_1.f90` with three more FFT round trips. If AD supplies that
divergence for free, the port's central claim has its first real demonstration
on a real quantity; if it does not, the claim is in trouble.

### What was built

`elkjax.xc.pbe` transcribes **only the energy densities** of `xc_pbe.f90`
(Perdew, Burke & Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)) —
$\varepsilon_x = \varepsilon_x^{\rm LDA}(2\rho_\sigma)F_x(s)$ with
$F_x = 1+\kappa-\kappa/(1+(\mu/\kappa)s^2)$, and
$\varepsilon_c = \varepsilon_c^{\rm PW92} + H$ with $H$ the usual
$g^3(\beta/\delta)\ln[1+\delta q_4t^2/q_5]$. `elkjax.grid.gradient` supplies
$\nabla\rho$ spectrally. **Nothing computes a Laplacian.**

Patch 0016 was extended with `exir`/`ecir` (and their muffin-tin partners) for
this: unlike `vxcir`, they are **not** passed through `trimrfg`, so they are the
raw pointwise output of the functional and an exact reference.

| quantity | agreement with Elk |
|---|---|
| $\varepsilon_x$ on the interstitial grid | **4.2e-16** |
| $\varepsilon_c$ on the interstitial grid | **3.4e-16** |
| $v_{xc}$ from `jax.grad`, against Elk's hand-coded one | 2.4e-5 median, 5.2e-4 max |

One convention is load-bearing and invisible in the formula: `ggair_1` zeroes
every Fourier component above `ngvc` before transforming the gradient back, so
`grid.gradient` truncates by default. (On this density it happens to matter
only at $10^{-15}$ — the density is already smooth — but on a density that is
not, it would not be optional.)

### The potential agrees to 2.4e-5, and that is not an error in either

The gradient terms are **11% of $v_{xc}$** here, so AD is reproducing from
nothing a correction whose own size is $5.5\times10^{-2}$, and disagreeing with
Elk by $5.2\times10^{-4}$ — about 1% of the thing being reproduced.

The gap has a cause, and it is worth stating precisely because it will recur
everywhere in Phase 3: **Elk discretises the exact continuum functional
derivative; AD returns the exact derivative of the discretised energy.** Those
are the same object only when the discretisation is exact, and it is not —
$|\nabla\rho|$ is not band-limited even when $\nabla\rho$ is, so the grid
representation of $\varepsilon_{xc}$ is aliased and its exact derivative is not
the sampled continuum derivative.

The prediction that follows is testable and is asserted: the residual should
track the reduced gradient $s$. Measured, median $|{\rm AD}-{\rm Elk}|$ is
5.8e-6 in the lowest quarter of $s$ and 1.25e-5 in the highest.

**Which one is "right" depends on what it is for.** For a variational or
force calculation the derivative of the energy *actually evaluated* is the one
that makes the energy stationary and the forces consistent — that is AD's. For
reproducing Elk's own numbers, Elk's is. This is the same distinction §1l met
from the other side, where the frozen assembly satisfied a gradient null that
the finite-shift comparison rejected.

### What this settles, and what it does not

Item 2d is done for PBE. The premise the port is justified by — that AD
supplies hand-coded derivative chains — is demonstrated on a real functional
against a real reference, with the residual explained rather than absorbed into
a tolerance.

**Not done**: the muffin-tin GGA, which needs the angular-grid transform
(`rbsht`/`rfsht`) and the muffin-tin gradient (`gradrfmt`), and the
spin-polarised GGA potential (the energy densities here are already
spin-resolved, but no spin-polarised fixture has been run). And the exchange
enhancement's own limits are checked ($F_x\to1+\mu s^2$ and $F_x\to1+\kappa$)
but the correlation's $H$ has no comparable closed-form limit checked beyond
$H(0)=0$.


---

## 2c. Integrals over the cell

### What was at stake

Elk splits every real-space function into a packed muffin-tin part and a value
on the interstitial FFT grid, so *every* scalar in Phase 2 and beyond — the
total charge, each energy component, any statement that a density is what it
should be — is one of two operations: `rfint`/`rfmtint` (the integral of one
function) or `rfinp` (the inner product of two). Both are short, and both are
easy to get subtly wrong in ways that stay plausible.

### What was built, and the three references

`elkjax.integrate` transcribes them:

$$\int_\Omega f\,d^3r
 = \frac{\Omega}{N_{\rm FFT}}\sum_i f_i\Theta_i
 + \sqrt{4\pi}\sum_\alpha\sum_r w^{(2)}_r f^\alpha_{00}(r),$$

with $\sqrt{4\pi}=1/Y_{00}$ from $\int R_{00}d\Omega$, and the interstitial sum
weighted by the characteristic function so the spheres are not counted twice.
The inner product differs in one structural way that a tolerance would never
reveal: it sums over **every** $(\ell,m)$, the real harmonics being
orthonormal, where a plain integral takes only $\ell=0$.

| reference | result |
|---|---|
| $\int_\Omega 1 = \Omega$ (pure geometry, and each half separately) | < 1e-9 relative |
| $\int_\Omega\rho = N_{\rm electrons}$ | **1.1e-14** (criterion: 1e-8) |
| $E_x = \langle\rho,\varepsilon_x\rangle$ vs Elk's INFO.OUT | 1e-9 (its print width) |
| $E_c = \langle\rho,\varepsilon_c\rangle$ vs Elk's INFO.OUT | 1e-9 (its print width) |

The first two owe nothing to Elk's own arithmetic. The electron count has one
trap: `rhomt` **includes the core density** (`rhocore` adds it), so the target
is 28 for two silicons and not the 8 valence electrons — an implementation that
assumed valence would miss by 20 electrons, not by a tolerance. And the energy
comparison is the check with teeth on the inner product specifically: an
implementation using only the $\ell=0$ coefficient passes both integral tests
and fails this one, which is asserted by measuring that the non-spherical
channels carry more than 1 Ha of $E_x$ here.

### What this settles

Half of the study's Phase 2 forward criterion — "cell charge integral to
$10^{-8}$ electrons" — is met at $10^{-14}$, and the machinery every remaining
energy component needs exists and is checked. The other half, the total energy
at a fixed input potential, needs the density (`rhomag`) and the Weinert
Poisson solve, neither of which is started.


---

## 2d. The muffin-tin angular transform, and why Elk's $v_{xc}$ is not $v_{xc}[\rho]$

### What was built

A nonlinear functional cannot be applied in the spherical-harmonic basis, so
Elk evaluates it on an angular grid: `rbsht` maps the $l_{\max}$ coefficients
at each radial point to $l_{\max}$ values on that grid, the functional is
applied pointwise, `rfsht` maps back. `elkjax.grid.to_angular`/`from_angular`
transcribe the pair; patch 0016 exports the four matrices (`rbshti`, `rfshti`,
`rbshto`, `rfshto`), which are small — $l_{\max}^{\rm i}=1$ and
$l_{\max}^{\rm o}=6$ give $4^2+49^2$ doubles each way.

| quantity | agreement with Elk |
|---|---|
| `rfsht(rbsht(rho)) - rho` | 2.7e-12 |
| `exmt` from the angular-grid density | **1.8e-15** |
| `ecmt` from the angular-grid density | **2.8e-16** |
| `vxcmt` from the same | **5.3e-3** on a scale of 45 (1.2e-4 relative) |
| `vxcmt`, same code, `symtype = 0` ground state | **1.4e-14** |

The last row is the explanation, and the four above it are what made it worth
finding.

### `potxc` symmetrises the potential and not the energy density

`potxc.f90` lines 55-58, after the per-atom `potxcmt` loop:

```fortran
if (tsh) then
  call symrfmt(nrmt,nrmti,npmt,npmtmax,vxcmt_)
  if (spinpol) call symrvfmt(.true.,ncmag,nrmt,nrmti,npmt,npmtmax,bxcmt_)
```

`vxcmt_` and `bxcmt_` are symmetrised; `exmt_`/`ecmt_` are returned exactly as
`potxcmt` computed them. So in Elk's muffin tins

$$v_{xc} = \hat S\,v_{xc}[\rho],\qquad \varepsilon_{xc}=\varepsilon_{xc}[\rho],$$

with $\hat S$ the average over the $n_{\rm symcrys}$ crystal operations
(`symrfmt`, each term a `rotrfmt` of the function at the atom that operation
maps in). The two are built from the same density by the same routine and then
treated differently.

$\hat S$ is not the identity on this function even though $\rho$ is already
symmetric, because **$v_{xc}[\rho]$ is not band-limited when $\rho$ is**.
Squeezing a nonlinear function through a finite angular grid leaks weight into
every harmonic, the symmetry-forbidden ones included, and `rfsht` truncates
that leak at $l_{\max}$ rather than removing it. `symrfmt` removes it.

Resolved by $l$ on bulk Si (atom 0, outer region, $l_{\max}^{\rm o}=6$):

| $l$ | Elk's `vxcmt` | pointwise $v_{xc}[\rho]$, residual |
|---|---|---|
| 0 | 3.5e+1 | 2.9e-4 |
| 1 | **1.7e-20** | 1.2e-3 |
| 2 | **1.5e-20** | 9.4e-4 |
| 3 | 1.5e-1 | 1.4e-3 |
| 4 | 2.9e-2 | 1.7e-3 |
| 5 | **6.9e-19** | 2.8e-3 |
| 6 | 1.9e-3 | 5.3e-3 |

$l=1,2,5$ are forbidden by the site symmetry and Elk carries exactly zero
there; the pointwise potential carries $10^{-3}$. In the *inner* region
($l\le l_{\max}^{\rm i}=1$, $r<0.30$ Bohr) the residual is 6e-14: near the
nucleus the density is spherical and there is nothing to project.

The `symtype = 0` run is the control that turns this from an argument into a
measurement. `readinput.f90:1311` shows the `nosym` input block doing nothing
but setting `symtype=0`, which leaves one symmetry operation and makes
`symrfmt` the identity. On that ground state the same transcription reproduces
`vxcmt` to 1.4e-14 at both atoms, while `exmt`/`ecmt` stay exact — so nothing
about the functional, the density or the transform changed, only $\hat S$.

### Two consequences, both real

**Elk's SCF cannot be reproduced without $\hat S$.** 1.2e-4 relative is far
above any tolerance in this port. Applying it needs `symlatc`, `lsplsymc`,
`ieqatom` and `isymlat` exported (patch 0016 exports none of them), plus a
transcription of `rotrfmt`'s Wigner-$D$ rotation of real harmonics — or the
port runs `symtype=0`, paying the cost of an unreduced $k$-set and an
unsymmetrised density. The second is the cheaper route for Phase 2 and is what
`tests/test_calculation_muffin_tin_xc.py` uses; the first will be needed
eventually, and it is one more patch, not a research problem.

**Inside a symmetric muffin tin, Elk's own $v_{xc}$ is not the pointwise
functional derivative of its own $E_{xc}$** — and §2f measured that this costs
**nothing** in any integral against $\rho$, which is not what this section
originally predicted.

The prediction here was that a total-energy or force check better than
$\sim10^{-4}$ relative "would be evidence of a mistake rather than of success."
That is wrong. $\hat S$ is a group average, hence an *orthogonal projection*,
and $\rho$ is already in its range, so
$\langle\rho,\hat Sv\rangle=\langle\hat S\rho,v\rangle=\langle\rho,v\rangle$
identically. The $5.3\times10^{-3}$ leak lives entirely in the harmonics $\rho$
does not have, so every integral against $\rho$ is blind to it. Measured (§2f,
both structures): the potentials differ by $5.3\times10^{-3}$ pointwise while
$E_{v_{xc}}$ agrees with Elk to $3\times10^{-16}$, and
$\langle\rho,v_{\rm mine}-v_{\rm Elk}\rangle$ is $10^{-16}$ *relative*.

So the asymmetry is real, it is why the pointwise comparison fails, and it is
invisible to the energy — which is presumably why Elk can carry it. What still
holds is the first consequence: a transcription that must match Elk's `vxcmt`
*pointwise*, as any SCF iteration does, needs $\hat S$.

### Why the wrong hypotheses took a while

Everything checked before the cause was found is listed here because each was
individually decisive and collectively misleading: it is entirely in
correlation (Elk's own $v_x=\tfrac43\varepsilon_x$ to roundoff); it is not a
density error ($\varepsilon_c$ agrees to 2.8e-16 at the same points); it is not
a function of $\rho$ (points at $\rho=11.2$ disagree, points at $\rho=0.78$
agree, and the two sets overlap); it is not mixing (`vsmt - vclmt - vxcmt` is
1.7e-7 on a scale of 9e7); it is not a post-processing filter (`potks` trims
only `vxcir`); it is not the interstitial's story (4.4e-16 there, §2a). All
true, and none of them looks at the *last four lines of the calling routine*.

The one observation that pointed at the answer was the $l$ decomposition —
Elk holding exact zeros in three channels where the transcription held
$10^{-3}$. A scalar residual, however carefully bounded, cannot show that.
**When a field-valued quantity disagrees, decompose it in the basis the code
stores it in before ruling anything out.**

---

## 2e. The Weinert Poisson solve

### What was at stake

Every other Phase 2 piece takes the converged potential as an *input*. This one
does not: the Coulomb potential is a functional of the density, and no export
supplies it as one — `vclmt`/`vclir` come out of `potcoul` and nowhere else. §2a
and §2b close the exchange-correlation half pointwise; without this the SCF loop
has a hole in it, and the total energy (item 2f) has no Hartree term.

### The method, and why it is not a Fourier transform

The interstitial Poisson equation would be trivial in $G$-space if the density
were smooth. It is not: an all-electron density has a nuclear cusp and a core
that varies over $10^{-4}$ Bohr, and its Fourier series is hopeless. Weinert's
construction (*J. Math. Phys.* **22**, 2433 (1981)) replaces the charge inside
each muffin tin by a smooth **pseudocharge** carrying the same multipole moments
$q_{lm}$. Outside the spheres the two are indistinguishable — a multipole
expansion knows nothing else — while the pseudocharge's Fourier series converges
quickly. The intra-sphere problem is then solved exactly on the radial mesh, and
the two are joined by adding the harmonic function $r^lY_{lm}$ that fixes the
boundary value.

`src/elkjax/poisson.py` is one function per step of `potcoul.f90`:

1. `real_to_complex` (`rtozfmt`) — the density to complex harmonics, the basis
   the multipole algebra is written in.
2. `intra_sphere` (`zpotclmt`) — the exact radial solution
   $$V_{lm}(r)=\frac{4\pi}{2l+1}\Big[r^{-l-1}\!\!\int_0^{r}\!\!\rho_{lm}r'^{l+2}dr'
   + r^{l}\!\!\int_r^{R}\!\!\rho_{lm}r'^{1-l}dr'\Big],$$
   both integrals by `wsplint`'s cumulative spline weights.
3. `add_nuclear` — `vcln` into the $l=0$ channel.
4. `pseudocharge_solve` (`zpotcoul`) — read $q_{lm}$ off the sphere boundary,
   subtract what the interstitial density already contributes there, add the
   pseudocharge in $G$-space, divide by $G^2$, and match the boundary.

### Against Elk

| quantity | agreement |
|---|---|
| `vclir` (Si / h-BN) | 1.6e-15 / 2.0e-14 relative |
| `vclmt`, $l=0$ (scale $10^7$, carrying the nucleus) | 3.9e-20 / 3.7e-19 relative |
| `vclmt`, $l>0$ (scale $10^{-1}$) | 7.2e-14 / 6.3e-14 relative |

Split by endpoint and by $l$ deliberately. The muffin-tin array's $l=0$ channel
is eight orders of magnitude larger than the other 48 harmonics, so comparing it
as one array would let any error in all of those pass at any relative tolerance
worth stating. h-BN is not a duplicate of Si either: every per-species array here
(`wprmt`, `vcln`, the Bessel table) is indexed by `idxis`, and a transcription
that indexes by *atom* instead is invisible on a one-species cell.

### Two checks that owe Elk's `vclmt` nothing

Most of the above is a comparison against the thing being reproduced, so two
checks are built to be independent of it.

**The monopole identity.** `zpotcoul` reads $q_{lm}$ off the sphere-boundary
*value* of the intra-sphere potential — not by integrating $\rho r^l$ — and by
that point the nucleus has been added, so $\sqrt{4\pi}\,q_{00}=N_{\rm MT}-Z$ with
$N_{\rm MT}$ the electron count inside the sphere. Taking $N_{\rm MT}$ from
`elkjax.integrate` (§2c, `rfmtint`, an unrelated code path) gives $Z=14.000000$
for silicon and $5.000000$ / $7.000000$ for boron and nitrogen. Exact integers
from a chain of splines, Bessel functions and an FFT.

**Two mutation tests**, each removing one thing Elk does:

* *the nuclear term, before the multipoles are read.* Skipping it leaves a
  perfectly smooth intra-sphere potential of the right order, and flips the sign
  of $q_{00}$. Nothing structural notices.
* *the outer region's own spline weights.* For $l>l_{\max}^{\rm i}$ the inner
  region does not store the harmonic at all, so Elk integrates from $r_{\rm iro}$
  outward using the **sub-mesh's** weights. Zero-padding and integrating from the
  origin uses the wrong weights at the lower boundary; the result is smooth, of
  the right order, and differs from Elk in the fifth digit.

Both mutants pass every structural check available. They are pinned because
arguing that they would be wrong is not the same as measuring it.

### Three details that were nearly wrong

**`genylmv`'s $4\pi(-i)^l$ prefactor removes every $l=0$ special case.**
`zpotcoul` writes the $l=0$ slot three times as `... * fourpi * y00 * z1`, with
no $Y_{lm}$ factor, and the $l\ge1$ slots as `... * conjg(ylmgp(...))`. Since
`ylmg[:,0]` *is* $4\pi y_{00}$ — a real constant — the uniform expression over
all $lm$ reproduces all four lines exactly. Writing the special case out
separately, as the Fortran does, would be a second transcription of one line and
a second place for it to drift.

**`vcln` is the $(0,0)$ coefficient, not the potential.** Its outermost value on
silicon is $-22.596$, which is $-Z/R$ divided by $y_{00}$, not $-Z/R=-6.374$.

**Elk leaves the FFT array's high-$G$ slots holding the raw density.** Only the
first `ngvec` of `ngtot` slots are divided by $G^2$; here that is 7799 of 21952.
Zeroing the rest is a *correction*, not a transcription — and the 8e-15 agreement
in `vclir` says Elk's version is what the round trip needs.

### Not differentiated, and that is the point

Poisson is **linear** in the density, so its linearisation is itself: an
AD-versus-FD check here would confirm that JAX can differentiate a linear map.
The content is forward exactness, and the standing rule from §1k and §1l — every
derivative check needs a forward check beside it — has nothing to check here in
the first place. The one forward instrument built and *not* used is
`intra_sphere_residual`, which evaluates $\nabla^2V_{lm}+4\pi\rho_{lm}$ directly
on the solution: on Elk's logarithmic mesh a five-point polynomial fit gives a
residual of $10^{-7}$ to $10^{-2}$ depending on the channel, which is a
measurement of the fit and not of the solve. It is kept because it owes Elk
nothing and would catch a gross error, and it is documented as the weak
instrument it is rather than quoted as a result.

---

## 2f. The total energy, and the prediction §2d got wrong

### What was at stake

Every Phase 2 piece so far was checked against Elk *on its own*: the functional
pointwise, the cell integral against `INFO.OUT`, the Poisson solve element-wise.
None of those says the pieces are consistent **with each other** — that the
potential one builds is the one another integrates, against the density they
share. `energy.f90`'s decomposition is the check that does, and it is a hard
one: ten terms, spanning three orders of magnitude, with heavy cancellation
between them.

### Scope: what is reproduced and what is imported

For a non-magnetic ground state `energy.f90` assembles

$$E_{\rm tot}=E_{\rm kin}+\tfrac12E_{v_{cl}}+E_{\rm Mad}+E_x+E_c+E_{TS},
\qquad E_{\rm kin}=\Sigma_\varepsilon-E_{v_{cl}}-E_{v_{xc}} .$$

Everything but three terms is a functional of the density and the potentials
built from it. The three are **imported**, and the module says so rather than
burying them in a total: $\Sigma_\varepsilon$ (the occupation-weighted eigenvalue
sum) and $E_{TS}$ (the smearing entropy) need the second-variational step and a
zone sum, which is Phase 3; $E_{nn}$ is a property of the lattice, not of the
density, and comes from `energynn`. **This is not a transcription of
`energy.f90`** — it is the density-functional half of it.

Patch 0017 exports Elk's own thirteen scalars at full precision rather than
leaving the comparison to `INFO.OUT`'s print width. That matters: a total energy
that agrees to $10^{-8}$ says nothing about which *convention* is right, and the
term-by-term table is the entire diagnostic value of the exercise.

### The table

Bulk Si (`src/elkjax/energy.py`'s own `report`), with the Coulomb potential taken
from §2e rather than from Elk:

| term | elkjax | Elk | relative |
|---|---|---|---|
| `engyvcl` | −841.805985476284 | −841.805985476285 | 5.4e-16 |
| `engymad` | −696.535809871263 | −696.535809871263 | 0 |
| `engyen` | −1219.434954924043 | −1219.434954924043 | 0 |
| `engyhar` | +188.814484723879 | +188.814484723879 | 1.2e-15 |
| `engycl` | −1117.438802609406 | −1117.438802609406 | 0 |
| `engyvxc` | −52.479468518512 | −52.479468518512 | 2.7e-16 |
| `engyx` | −37.574959142856 | −37.574959142855 | 3.8e-16 |
| `engyc` | −2.147752668912 | −2.147752668912 | 2.1e-16 |
| `engykn` | +579.178007953273 | +579.178007953274 | 5.9e-16 |
| `engytot` | −577.983515950320 | −577.983515950320 | 2.0e-16 |

Same on monolayer h-BN, where the vacuum makes the interstitial dominate the cell
integral rather than being a correction to the spheres. Every term is asserted
individually and not through the total, because `engykn` is $+579$ against
`engyen`'s $-1219$: an error of $10^{-3}$ in either would leave `engytot` looking
fine at $10^{-6}$ relative.

Two of these are cross-checks rather than repetitions. Substituting Elk's own
`vclmt`/`vclir` for §2e's changes nothing at $10^{-13}$ — the *integrated*
counterpart of §2e's pointwise check, which is not redundant with it, since a
pointwise agreement could still integrate differently if the solve and the
quadrature disagreed about the packing or the region split. And `engymad` is the
one term that is not an integral: it reads the $l=0$ potential at the **first**
radial point and subtracts `vcln` there, two numbers of order $10^7$ whose
difference is of order $10^2$. Dropping the subtraction leaves a smooth, finite
number wrong by five orders of magnitude, which the mutation test pins.

### §2d's prediction was wrong, and the reason is the interesting part

§2d found that `potxc` symmetrises the muffin-tin $v_{xc}$ and not
$\varepsilon_{xc}$, and predicted that $E_{v_{xc}}=\int\rho\,v_{xc}$ — an
integral against a symmetrised potential, computed here from an unsymmetrised
one — would therefore disagree with Elk at the same $\sim10^{-4}$ relative level.

It agrees at $2.7\times10^{-16}$.

$\hat S$ is a *group average*, hence an **orthogonal projection**, and $\rho$ is
already in its range (`rhomag` symmetrises it), so

$$\langle\rho,\hat Sv\rangle=\langle\hat S\rho,v\rangle=\langle\rho,v\rangle$$

identically. The leak lives entirely in the harmonics $\rho$ does not have, and
every integral against $\rho$ is blind to it. Measured directly on both
structures: the potentials differ by $5.3\times10^{-3}$ and
$2.6\times10^{-3}$ pointwise, while
$\langle\rho,v_{\rm mine}-v_{\rm Elk}\rangle/\langle\rho,v_{\rm Elk}\rangle$ is
$8.6\times10^{-17}$ and $4.7\times10^{-17}$.

So §2d's asymmetry is real, it is why the pointwise comparison fails, and it
costs nothing in the energy — which is presumably why Elk carries it. What
survives of §2d's consequences is the first: an SCF iteration compares potentials
*pointwise*, so reproducing Elk's loop still needs $\hat S$.

The methodological point is worth more than the physics. §2d's prediction was
made from a correct fact by a plausible argument, and it took a measurement to
find that the argument skipped a step. **A prediction derived from a verified
finding is not itself verified**, and the cheapest way to keep that honest is to
write predictions down where a later test will run into them — which is what
happened here.

---

## 2g. `symrfmt`, exported rather than transcribed

### What was at stake

§2d found the asymmetry — `potxc` symmetrises `vxcmt` and `bxcmt`, not
`exmt`/`ecmt` — and §2f found that it costs nothing in any integral against
$\rho$. What it does cost is the *pointwise* comparison, and an SCF iteration is
pointwise: a loop that builds $v_{xc}$ from a density and hands it to the next
Hamiltonian must produce Elk's `vxcmt`, not something $1.2\times10^{-4}$ away
from it. So closing the loop needs $\hat S$.

### The design choice, which is the interesting part

Transcribing `symrfmt` means transcribing `rotrflm`: `roteuler`'s extraction of
Euler angles from a Cartesian rotation, a real-harmonic Wigner-$D$ construction,
and the improper-rotation $(-1)^l$ branch. Two hundred lines of convention whose
**only consumer inside Elk is `symrfmt` itself** — so a Python re-derivation
would have no independent check except agreement with the thing it replaces. And
it does not stop there: `symrfmt`'s atom bookkeeping would have to come too —
`ieqatom` (the atom an operation maps *into* the target), `tfeqat`, and the
**inverse** lattice rotation `isymlat(lsplsymc(isym))` in the
rotate-into-equivalent-atoms loop.

Patch **0018** exports the operator instead. `elkpy_gsexport` calls upstream
`symrfmt` on basis vectors and writes back what comes out, so **none of that
bookkeeping is transcribed and none of it can be got wrong here**. What is tested
is the operator's *application*, which has no convention in it at all.

This is the same call `wprmt` got in patch 0017 — "no closed form worth
retyping" — but stronger, because there the alternative had a defining equation
to check against and here it does not.

The operator is one $l_{\max}^{\rm o}$-square matrix per **ordered atom pair**,
and that is the whole of it: a rotation is diagonal in the radial index (so the
mesh does not enter) and does not mix $l$ (so the inner region, carrying only
$l_{\max}^{\rm i}$ harmonics, uses the same matrix's top-left block — `rotrfmt`
calls `rotrflm` separately on the two regions with the same rotation). For bulk
Si that is $2^2\times49^2$ doubles.

### What it closes, and what it measures

| check | result |
|---|---|
| pointwise $v_{xc}[\rho]$ against Elk's `vxcmt` | 5.3e-3 (§2d) |
| $\hat S\,v_{xc}[\rho]$ against the same | **6.4e-14** |
| $\hat S\rho-\rho$ | 7e-18 — §2d's premise, asserted |
| $\hat S^2-\hat S$, bulk Si | 1e-16 |
| $\hat S^2-\hat S$, monolayer h-BN | **1.2e-11** |
| `symtype=0`: $\hat S$ against $\mathbb 1$ | exact, including zero between atoms |

Two of those need comment.

**The cross-atom block is not zero**, and that is what makes silicon a real
fixture here: its two equivalent atoms give `symop[0,1]` entries of $0.5$, so the
operator mixes *sites* and not only harmonics. On a cell where every atom is
alone in its species — h-BN's B and N — the off-diagonal blocks vanish
identically (measured: exactly $0.0$), and the file would be testing an angular
rotation only.

**Idempotence is exact on a cubic lattice and not on a hexagonal one**, by five
orders of magnitude, and the residual grows with $l$ within h-BN
($2.5\times10^{-12}$ at $l=1$ to $1.2\times10^{-11}$ at $l=5$). That is Elk's own
arithmetic: `roteuler`'s inverse trigonometry is exact when the Cartesian
`symlatc` entries are $0$ and $\pm1$, and hexagonal ones carry $\tfrac12$ and
$\tfrac{\sqrt3}2$; the $l$ dependence is the Wigner-$D$ order compounding the
angle error. It bounds how idempotent `symrfmt` **can** be, not the operator's
accuracy in use — which applies it once, and holds at $10^{-14}$ on both
structures.

The test asserts that contrast rather than covering both with one loose
tolerance. A single $10^{-10}$ would say the operator is good to $10^{-10}$
everywhere: wrong in one direction and uninformative in the other.

---

## 2h. The interstitial valence density, and the circle closing

### What was at stake

§§2a-2g all run in one direction: from a density to potentials to a total
energy. An SCF loop needs the other direction too — eigenvectors back to a
density — and until that exists the chain is a line, not a circle.

`rhomagv` → `rhomagk` is that step. This section does its **interstitial**
half.

### The formula, and why it is exact rather than approximate

For a non-magnetic ground state `rhomagk`'s `rmk3` branch is

$$\rho^{\rm I}(\mathbf r)=\frac1\Omega\sum_{\mathbf k}w_{\mathbf k}
\sum_n f_{n\mathbf k}\bigl|\psi_{n\mathbf k}(\mathbf r)\bigr|^2 ,$$

with $\psi$ built by scattering `evecfv(igp, n)` into the coarse FFT grid at
`igfc(igkig(igp))` and transforming. **In the interstitial an LAPW state is a
plain plane-wave sum** — no augmentation, no radial functions — so there is no
approximation here at all: the only truncation is the basis's own, and it is the
same truncation on both sides of the comparison.

Patch **0019** adds a `DENSITYK` query carrying exactly what `rhomagv` feeds to
`rhomagk`: `wkpt`, `vkl`, `occsv`, `ngk`, `igkig`, and `evecfv` read back through
`getevecfv` the same way `rhomagv` reads it, plus `ngdgc`/`ngtc`/`igfc`. It is a
query of its own because it is per-$k$ **and** over the whole zone, where `LAPW`
is one $k$-point of the caller's choosing and `GROUNDSTATE` is $k$-independent.
Rebuilding the $k$-set from `ngridk` would differ whenever `reducek` is nonzero,
which is the default.

### Three results

| | |
|---|---|
| unreduced mesh (`symtype=0`), `trhonorm` off | **9.0e-16** relative |
| symmetry-reduced mesh (Elk's default, 3 k-points) | **16%** |
| `rhonorm`'s shift, on vs off | 2.82e-05 vs **2.8e-18** |

**The first is the transcription.** Exact, as the plane-wave argument says it
should be.

**The second is the scope, and it is asserted rather than written down as a
caveat.** `rhomagv` calls `symrf` on the accumulated density, and on a reduced
mesh that is not the identity. 16% is a missing step, not a tolerance. This is
the same shape as §2g's muffin-tin `symrfmt` — in the interstitial, where the
operator is `symrfir` — and closing it would be the same kind of patch. The test
asserts the disagreement, so if `symrfir` is ever applied here the test fails and
gets rewritten rather than widened.

**The third identifies `rhonorm` by measurement.** `rhonorm` adds a *uniform
constant* $(N-N_{\rm calc})/\Omega$ rather than rescaling, so a correct
transcription differs from Elk's stored density by a constant — asserted as a
constant, which is sharper than any tolerance, since a constant is a
one-parameter family that a pointwise error would break (measured spread
1.7e-17). Switching `trhonorm` off then takes the constant itself to 2.8e-18.
Either measurement alone would only bound the discrepancy; the pair names it.
$\Omega$ times the shift is $7.6\times10^{-3}$ electrons, which is the
normalisation deficit of a converged density.

### The comparison introduces no error of its own

Elk's `rhoir` lives on the **fine** grid ($28^3$ here) and `rhomagk` accumulates
on the **coarse** one ($15^3$). `rfirctof` bridges them by transforming the
coarse function, copying its `ngvc` components into the fine array and zeroing
the rest — so the fine density carries no content the coarse one did not, and
taking those same components back is lossless. `elkjax.density.coarsen` does
that, and the property it relies on is **asserted on Elk's own `rhoir`** rather
than assumed: if `rhoir` ever carried content beyond `ngvc`, every comparison in
the file would quietly acquire an aliasing error instead of failing.

This is also why the comparison is made on the coarse grid rather than
interpolating upward: `rfirctof` goes through `rzfftifc`'s real-to-complex packed
representation with its own `nfgrz`/`igrzf` indexing, which is one more set of
conventions to get wrong for no gain.

### The mutation test

`igkig` takes a basis function to a global $\mathbf G$ index and `igfc` takes
that to a coarse FFT slot. Dropping the second step leaves a density that is
smooth, non-negative and correctly normalised — a permutation of the Fourier
content is still a density — and differs from Elk by 16%. Nothing structural
notices, which is why it is pinned.

### The muffin-tin half, and the reference that made it one routine

The muffin-tin density needs `wfmtsv` on the coarse *radial* mesh, and then
`rhomagsh`, `symrf`, `rfmtctof` and `rhocore` — the last of which brings in a
core-state solver. **None of those four is transcribed**, because patch 0019
exports the reference from *before* them: `elkpy_denskexport` loops `rhomagk`
over the k-set into a **local** array, touching nothing global, and writes the
result. That is patch 0018's design again — Elk's own routine produces the
reference — and it cuts the work to exactly one routine.

So the comparison is against the density in **spherical coordinates on the
coarse mesh**, which is what `rhomagk` actually produces. Measured on bulk Si
(`symtype=0`, `trhonorm` off):

| | |
|---|---|
| muffin tin, atom 0 / atom 1 | **9.0e-16 / 7.7e-16** |
| interstitial, against the same direct reference | **7.5e-16** |

**Two things were nearly wrong, and one of them was.**

*The local orbitals.* `evecfv` has $n_{\rm mat}=n_{gk}+n_{\rm lotot}$
coefficients — 177 against 169 here — and the first version of patch 0019
exported only $n_{gk}$ of them. The interstitial density stayed **exact**, local
orbitals vanishing there, while the muffin-tin density came out smooth,
non-negative, correctly scaled and 100% wrong. Only a reference catches that,
which is the argument for exporting one.

*The radial stride.* `wfmtsv`'s `zfzrf` declares `rf(lrstp, n)` and is handed
`apwfr(1,...)` for the inner region and `apwfr(iro,...)` with
$i_{ro}=n_r^{\rm i}+l_{\rm rstp}$ for the outer — so the outer region restarts
one full step *past* the inner boundary rather than continuing the stride. Off by
one step the density is still smooth and still the right order.

**And patch 0015's finding recurred, with a sharper consequence.** The first
measurement was 1.2e-10, not 9e-16 — close enough to read as a transcription
bug. The cause: `gndstate` mixes the potential *after* building the radial
functions, so on session entry `apwfr`/`lofr` belong to the previous iteration.
`elkpy_lapwexport` calls `genapwlofr` to fix that; `elkpy_denskexport` did not,
so its `rhomagk` reference used stale radial functions while the transcription
used regenerated ones. Diagnosed by reordering the two queries — asking for
`LAPW` first took the residual to 9e-16 — and then fixed properly in Fortran,
because **a query whose answer depends on which query ran before it is a trap**,
not a documentation problem. The test now asks for `DENSITYK` first, so the
ordering dependence cannot come back silently.

### The two post-processing steps, and `rhomagv` complete

`rhomagk` leaves the muffin-tin density in spherical *coordinates* on the
*coarse* radial mesh — a modulus is pointwise, so it has to be accumulated
there — and two steps take it to the representation the potential is built on.
Patch **0020** exports both intermediates, so each is checked on its own rather
than through their composition:

| stage | agreement |
|---|---|
| `rhomagk`, coordinates on the coarse mesh | 9.0e-16 / 7.7e-16 |
| after `rhomagsh` (`rfshtip`: back to harmonics) | 8.7e-16 / 7.3e-16 |
| after `rfmtctof` (coarse radial mesh → fine) | 1.0e-15 / 8.7e-16 |

`rhomagsh` is the **real** transform where the wavefunctions used the complex
one — a wavefunction is complex and a density is not, and Elk keeps both
matrices for that reason.

`rfmtctof` is a **fixed linear map**: `rfinterp`'s cubic-spline weights come
from `wspline` and depend only on the mesh. So patch 0020 exports it as a
matrix, built by calling `rfinterp` on basis vectors — patch 0018's call again,
and for the same reason: `splinew`'s weighted construction has no consumer that
would check a re-derivation, and the matrix is what a transcription needs anyway.

**Two matrices per species, and they are not interchangeable.** `rfmtctof`
interpolates the whole radial range for $l\le l_{\max}^{\rm i}$ and the outer
region *alone* above it, because only the outer region stores those harmonics.
Using the full-range map on an outer-only harmonic reads the inner region's
zeros as data — not a crash, not a discontinuity, just a smooth pull toward zero
near $R_{\rm MT}$. Pinned.

**With `symtype=0`, `symrf` is the identity, so those three stages are the whole
of `rhomagv`** for a non-magnetic cell. The chain from first-variational
eigenvectors to the valence density on the fine mesh is therefore closed and
exact.

### All of `rhomag`, against the converged arrays

Patch **0021** adds the last two ingredients — `rhocr` and `chgtot` — and with
them the chain runs to Elk's **own converged `rhomt`/`rhoir`** rather than to an
exported intermediate:

$$\text{rhomagk}\to\text{rhomagsh}\to
\text{rfmtctof}/\text{rfirctof}\to\text{rhocore}\to\text{rhonorm}$$

| | |
|---|---|
| `rhomt`, atom 0 / atom 1 | **3.2e-13 / 1.6e-13** |
| `rhoir` | **5.8e-12** |

`symrf` is absent because the fixtures run `symtype=0`, where it is the identity.

Two details. `rhocr` is stored the way `vcln` is — as the $(0,0)$ *coefficient*,
with the $1/y_{00}$ already folded in — and goes into that slot and nowhere else,
the core being spherical. And `rhonorm` shifts by a *constant*, so in the muffin
tin it too touches only the $l=0$ coefficient, with the same $y_{00}$ factor;
treating it as a rescaling would be wrong in a way that leaves the total charge
correct.

`rhocr` is **exported rather than solved for**, and that is the same call as
`vsmt` in patch 0015: the core states are a functional of the potential, so at
fixed potential the core density is an *input*. Everything else in the chain is
transcribed.

### A second structure, and the bug it found immediately

Every other Phase 2 result is checked on two structures; this section was
silicon-only, and multi-species indexing is precisely what one species cannot
test. The h-BN fixture **failed on its first run**.

`lorbl` is a *list over species* of per-species arrays — as the parser
documents — and **boron has two local orbitals to nitrogen's three**, so it is
genuinely ragged and `np.asarray` on the whole thing raises. On a one-species
cell it does not. One line.

The fixture's own premise needed correcting too, and that is the more useful
half: it asserted that the two species have different radial meshes, which is
**false** here — B and N share $n_{rc}=75$. What distinguishes them is the
local-orbital count, which is exactly what makes `lorbl` ragged. A premise that
names the wrong property is a test that starts passing for the wrong reason the
moment someone changes the structure.

### What is still not done

The core-state solver itself (`gencore`/`rdirac`), which a self-consistent loop
would need where a fixed-potential check does not; `symrfir`, for a
symmetry-reduced mesh — §2g has the muffin-tin operator, the interstitial's is
not exported; and the magnetic branches `rmk1`/`rmk2`.

---

## 2i. The Kohn-Sham potential, composed

### What was at stake

§§2a-2g each checked one piece against Elk, and §2f checked that the pieces are
consistent *inside an integral*. Nothing had checked the composition
**pointwise**, which is what an SCF iteration actually consumes:

$$v_s = v_{\rm cl}[\rho] + \hat S\,v_{xc}[\rho] .$$

Every term on the right is built here — the Weinert solve (§2e), the functional
on the angular grid (§2a/§2d), the symmetrisation (§2g) — from three different
modules, driven by one density. The left is Elk's own.

### Two references, and only one of them is independent

In the muffin tin Elk stores `vclmt` and `vxcmt` separately, so their sum is
`vsmt` by construction; comparing against it adds nothing beyond §2e and §2g
except that the two land in the same array. Measured $<10^{-14}$ relative on
bulk Si and monolayer h-BN.

In the interstitial Elk stores **`vsir` itself**, formed inside `potks` *after*
`trimrfg` has been applied to `vxcir` and **not** to `vclir`. That is an
independent reference, and it is the one with teeth: measured $<10^{-13}$.

### The mutation that only `vsir` catches

Trimming the Coulomb potential as well — a natural-looking symmetry, and wrong —
leaves a smooth interstitial potential of the right magnitude, since the
high-$|G|$ content it removes is small. It integrates against $\rho$ correctly
to the same order. The test asserts **both** that the mutant differs from `vsir`
*and* that the difference is below $10^{-2}$, i.e. that a loose tolerance would
have missed it. That is why the tolerance above is $10^{-13}$ and not something
comfortable.

### Where this leaves Phase 2

Both directions now exist and are exact:

* density → potential → energy (§§2a-2g, and this section for the potential
  itself),
* eigenvectors → density (§2h, all of `rhomagv` at `symtype=0`).

What is missing to close the loop is not a transcription problem: `rhocore` (an
input at fixed potential, exactly as `vsmt` is), `rhonorm` (one constant), a
zone-summed Fermi level (§1i has it at a single $k$), and `symrfir` for a
symmetry-reduced mesh. The open question is whether the **composition** is
stable, which nothing here has tested — every check in Phase 2 starts from Elk's
own converged density.

---

## 2j. The loop closed: potential → eigenvectors → density

### What was at stake

§2i takes a density to a potential. §2h takes eigenvectors to a density. The
link between them — potential to eigenvectors — is Phase 1's assembly, but only
at **one** $k$-point: `eigenproblem_at` recovers $\tilde v_s$ as a *matrix* in
the exported $k$-point's own $\mathbf G$ basis, which cannot be carried to
another $k$. A zone sum needs the whole mesh.

### The one thing that was missing

`hmlistl`/`olpistl` build the interstitial blocks from $\tilde v_s$ and
$\tilde\Theta$ in $G$-space:

$$O^{\rm I}_{ij}=\tilde\Theta(\mathbf G_i-\mathbf G_j),\qquad
H^{\rm I}_{ij}=\tilde v_s(\mathbf G_i-\mathbf G_j)
+\tfrac12(\mathbf G_i{+}\mathbf k)\!\cdot\!(\mathbf G_j{+}\mathbf k)\,
\tilde\Theta(\mathbf G_i-\mathbf G_j),$$

which works at any $k$ whose $\mathbf G$ set is known — and patch 0019 already
exports Elk's own `igkig` for every $k$ of the mesh. `vsig` and `cfunig` come
from patch 0016; `ivgig` is rebuilt from `ivg` as a dense lookup, so the
difference-vector map is one fancy-index rather than $n_{gp}^2$ dictionary
probes. Elk fills only the upper triangle and the full matrix is formed here
instead, which is equivalent because both are transforms of real functions.

Against Elk's own exported matrices at $\Gamma$: $H$ to **2.7e-15** and $O$ to
**5.6e-16**.

`vsig`'s `ngvc` allocation is exactly the $|G|\le2g_{k\max}$ range a difference
of two $|\mathbf G+\mathbf k|<g_{k\max}$ vectors can reach, so the lookup must
land inside it. That is asserted rather than assumed — the Fortran would read
past the end.

### The result

`density_from_potential` does the whole half-step: radial integrals from the
potential (§1k) → $H$, $O$ at every $k$ → Cholesky-reduced eigensolve →
`wfmtsv` and the interstitial FFT → density. **Nothing in that path reads an
eigenvector.** Against patch 0019's reference: **5e-11** relative in both
regions.

### Why 5e-11 and not 1e-15, measured rather than excused

The residual is **Elk's own two exports of `evecfv` disagreeing with each
other**, by 8.5e-09.

`elkpy_lapwexport` calls `genapwlofr` and then `eveqnfv` — a *fresh*
diagonalisation with the regenerated radial functions. `elkpy_denskexport`
calls `genapwlofr` and then `getevecfv` — the *stored* eigenvectors, which were
computed with the previous iteration's radial functions, because `gndstate`
mixes the potential after building them. The reference is built from the
stored ones, as `rhomagv` builds it; this chain produces fresh ones.

The isolation, all at $\Gamma$ on the occupied-subspace projector (gauge
invariant, so a degenerate multiplet cannot confuse it):

| | |
|---|---|
| this solve vs the `LAPW` query's fresh `evecfv` | **1.6e-14** |
| re-diagonalising Elk's own $H,O$ vs its stored `evecfv` | 1.0e-14 |
| this assembly's $H,O$ vs Elk's, same solver | 2.2e-14 |
| **`DENSITYK`'s stored `evecfv` vs `LAPW`'s fresh one** | **3.7e-11** |

So the assembly, the solver and the density construction are each at $10^{-14}$,
and the only disagreement is between two of Elk's own arrays. **This is patch
0015's finding for the third time** — §1k measured it at 3e-10 in `haa`, §2h at
1.2e-10 in the muffin-tin density, and here it is visible directly as two
exports of the same object differing. The test asserts *both* halves: the two
exports disagree, and this solve matches the fresh one — without the second, the
loose tolerance would be an excuse rather than a measurement.

### What the loop still does not have

This is one *half-step* with Elk's occupations, not a fixed-point iteration.
Missing: `rhocore` (an input at fixed potential, like `vsmt`), `rhonorm` (one
constant), a zone-summed Fermi level (§1i has it at a single $k$), and
`symrfir` for a symmetry-reduced mesh. What is now demonstrated is that both
directions exist, are exact to their references, and compose.
