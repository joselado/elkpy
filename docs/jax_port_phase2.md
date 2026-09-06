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

| item | result |
|---|---|
| **2a** the XC functional, and its gradient | **done for LDA (`xctype=3`)**: exchange exact against Dirac, correlation anchored on Gell-Mann–Brueckner, `jax.grad` reproduces Elk's hand-coded $v_{xc}$ to machine precision, Elk's own `vxcir` reproduced to **4.4e-16** once `trimrfg` is reproduced with it, and the $\rho\to0$ guard's `NaN` hazard is asserted rather than described |
| **2a′** the `GROUNDSTATE` export (patch 0016) | **done** — density and potentials on Elk's own grids, $k$-independent; the reference for everything below |
| **2b** the density `rhomag` | not started |
| **2c** the Weinert Poisson solver | not started |
| **2d** a GGA functional | **done for PBE (`xctype=20`)** (§2b): energy densities exact against Elk's own `exir`/`ecir` (4e-16), and `jax.grad` of the discretised energy reproduces Elk's hand-coded potential to 2.4e-5 median — with the gap identified as discretise-then-differentiate versus differentiate-then-discretise, not as an error in either |
| **2e** symmetrisation | not started |
| **2d′** the muffin-tin angular transform | **done, with an anomaly** (§2d): `rbsht`/`rfsht` are mutual inverses to 2.7e-12 and give `exmt`/`ecmt` exactly, but `vxcmt` misses by 1.2e-4 relative — localised to correlation, not a function of the density, and unexplained |
| **2f** total energy at fixed input potential | **partly** (§2c): the cell integral and inner product are built (`rfint`/`rfinp`), so the charge integrates to the electron count within 1.1e-14 — the study's forward criterion asks 1e-8 — and $E_x$/$E_c$ match Elk's own INFO.OUT to 1e-9. The total energy needs the density and the Poisson solve |

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

## 2d. The muffin-tin angular transform, and one thing that does not add up

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

The first three say the transform is right and the density it is applied to is
right. The fourth does not follow from them, and it should.

### What has been ruled out

* **It is entirely in correlation.** Elk's own $v_x$ equals $\tfrac43$ times
  its own $\varepsilon_x$ to roundoff at every point, and $\varepsilon_x$
  itself is exact, so the exchange half is right.
* **It is not a density error.** $\varepsilon_c$ agrees to 2.8e-16 at the very
  same points, and $\varepsilon_c$ and $v_c$ have comparable sensitivity to
  $\rho$ — a $\delta\rho$ big enough to produce this would show in both.
* **It is not a function of $\rho$.** Points at $\rho=11.2$ disagree while
  points at $\rho=0.78$ agree; the two sets overlap in density. What it tracks
  is *radius*, growing smoothly from below $10^{-9}$ at $r=0.30$ Bohr to
  $5.3\times10^{-3}$ at $R_{\rm MT}$.
* **It is not the interstitial's story.** The identical transcription
  reproduces Elk's `vxcir` to 4.4e-16 (§2a) over an overlapping density range.
* **It is not mixing.** `vsmt - vclmt - vxcmt` is $1.7\times10^{-7}$ on a
  scale of $9\times10^{7}$, so the exported potentials are mutually
  consistent, and comparing against `vsmt - vclmt` instead gives the identical
  $5.3\times10^{-3}$.
* **It is not a post-processing filter.** `potks` trims only `vxcir`, and no
  routine outside `potxc`/`oepmain` writes `vxcmt` at all (grep-verified).

That is where it stands. `tests/test_calculation_muffin_tin_xc.py` asserts the
solid results *and* pins the anomaly's size and its
not-a-function-of-density character, so a later change that explains it makes
the test fail — which is the point of pinning it rather than leaving it in
prose.

### Why this is worth a section rather than a footnote

The two exact rows above are what make the fourth interesting. If the density
or the transform were wrong, $\varepsilon_x$ and $\varepsilon_c$ would be wrong
too, and they are exact to roundoff. So one of the following is true and it is
not yet known which: Elk's muffin-tin $v_c$ is computed from something other
than the angular-grid density that its own $\varepsilon_c$ is computed from; or
the export carries a `vxcmt` from a different state than its `exmt`/`ecmt`
despite `potxcmt` writing all three in the same loop; or there is a reading of
`xcifc`'s unpolarised branch that has been missed. **Do not build the
muffin-tin GGA or the total energy on this path until it is settled** — the
interstitial path is exact and is the one to extend meanwhile.
