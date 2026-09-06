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
| **2a** the XC functional, and its gradient | **done for LDA (`xctype=3`)**: exchange exact against Dirac, correlation anchored on Gell-Mann–Brueckner, `jax.grad` reproduces Elk's hand-coded $v_{xc}$ to machine precision, and the $\rho\to0$ guard's `NaN` hazard is asserted rather than described |
| **2b** the density `rhomag` | not started |
| **2c** the Weinert Poisson solver | not started |
| **2d** a GGA functional | not started |
| **2e** symmetrisation | not started |
| **2f** total energy at fixed input potential | not started |

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

### Against Elk's own $v_{xc}$, and why that tolerance is what it is

All four checks above are statements about the *form* of the functional. None
says it is the functional Elk used. `tests/test_calculation_xc.py` takes Elk's
own $v_{xc}$ along a line through bulk silicon and compares it against
$v_{xc}$ evaluated here on Elk's own density along the same line (`plot1d` puts
both on the identical point set, which the test asserts).

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

It is still discriminating. Correlation is 17–20% of $v_{xc}$ along this line
(asserted, not assumed), so a 3e-5 median agreement identifies the
parameterisation to about four significant figures — four orders of magnitude
finer than the difference between having this correlation functional and having
none.

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
