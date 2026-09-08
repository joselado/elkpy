# Phase 3 of the Elk-to-JAX port: measurements

Running log in the shape `docs/jax_port_phase{0,1,2}.md` established: what was
at stake, what was run, and what the result does *not* settle.
`docs/jax_port.md` §6 is the plan.

Phase 3 as written is "the SCF loop with implicit differentiation". Its
criteria are a converged ground state reproducing Elk's `TOTENERGY.OUT` and
`EFERMI.OUT`, and three gradient signatures: the inter-mixer difference falling
linearly with `epspot` under implicit differentiation and plateauing under
unrolling (A), $d\mu/d\varepsilon$ on a metal against central FD (B), and the
gradient's plateau reached at a *looser* tolerance under implicit
differentiation than under unrolling (C).

**Phase 2 left two half-steps that compose and had never been cycled.** §2i
takes a density to a potential and §2j takes a potential to a density, each
pointwise against Elk. What joined them was the zone-summed Fermi level, and
that is §3a.

| study item | status | where |
|---|---|---|
| the occupations and the zone-summed Fermi level | **done** — Elk's `efermi` and `occsv` reproduced bitwise on bulk Si and fcc Al, with the assembly out of the path | §3a |
| **Forward:** a converged ground state reproducing Elk's total energy and Fermi level | **done** — 3.0e-8 Ha and 1.4e-9 Ha on bulk Si, from a start 0.30 away in potential norm | §3b |
| **Gradient A/B/C:** the implicit-differentiation signatures | **not started** — the loop runs on concrete arrays; §3b says exactly what blocks tracing | §3b |

---

## 3a. `occupy.f90`: the Fermi level over the zone

### What was at stake

Phase 1i built $\mu$ at **one** k-point. That is not a scaled-down version of
the real thing: study §8(b)'s rule is

$$d\mu=\frac{\sum_{i\bf k}w_{\bf k}f'_{i\bf k}\,d\varepsilon_{i\bf k}}
{\sum_{i\bf k}w_{\bf k}f'_{i\bf k}},$$

and at a single k-point the weights cancel identically, so *the half of the
rule that carries them had never been evaluated*. §1i said so explicitly and
left it open.

It is also the last piece of Fortran between the two Phase 2 half-steps. §2j's
loop runs at **Elk's own occupations**; a fixed-point iteration cannot.

### The reference, and why it has the assembly taken out

Patch **0023** exports `occupy`'s inputs — `stype`, `swidth`, `occmax`,
`chgval`, `e0min`, `epsocc` — together with Elk's own `efermi` and, the one
that matters, `evalsv` over the **whole** k-set. No existing query could
supply that: `LAPW` returns `evalfv` at one k-point of the caller's choosing
and `DENSITYK` wrote only the occupations.

With it the transcription is checked on **Elk's own eigenvalues**. That is the
point of exporting `evalsv` rather than computing it: through the assembly, a
wrong bisection and a wrong Hamiltonian are the same symptom.

### The result

`src/elkjax/occupations.py` transcribes `occupy.f90` with `stheta`/`sdelta` for
`stype` 0–3 (Methfessel-Paxton orders 0–2 and Fermi-Dirac; 4 and 5 raise rather
than silently returning Fermi-Dirac occupations).

| fixture | $\mu$ vs Elk's `efermi` | `occsv` vs Elk's |
|---|---|---|
| bulk Si, 2×2×2 reduced to 3 k-points | **0.0 (bitwise)** | **0.0 (bitwise)** |
| fcc Al, 4×4×4 reduced to 8 k-points | **0.0 (bitwise)** | **0.0 (bitwise)** |

Bitwise is a stronger statement than a tolerance here and is worth stating as
one: the bisection is a deterministic sequence of ~45 comparisons
`chg < chgval`, so agreeing bitwise means every one of them went the same way
even though the electron count is accumulated in a different order (a Fortran
loop against `jnp.sum`). The tests assert $10^{-12}$ — the bisection's own
bracket width — rather than equality, so a different BLAS cannot turn a
measurement into a failure.

Three details of Elk's own routine are transcribed rather than tidied, because
each changes the answer:

* the bracket is $[\min\varepsilon,\max\varepsilon]$ over the whole set, and
  the value returned is the **last midpoint tested**, not the midpoint of the
  final bracket — which is the one `occsv` was filled from, so the two must be
  produced together;
* `e0min` gates to zero any state below the minimum linearisation energy
  (`min(0, every apwe0 and lorbe0) - 2`), which is a ghost of the
  linearisation rather than an electron;
* the smeared step's hard cutoffs at $|x|>50$ (Fermi-Dirac) and $|x|>12$
  (Methfessel-Paxton). The exponent is clipped *before* the exponential, so
  the discarded branch of the `jnp.where` cannot overflow — §2a's hazard, and
  the gradient is zero outside the cutoff in Elk too, so that is a
  transcription and not an approximation.

### The gate does not fire on either fixture, and that is said rather than hidden

Si's deepest eigenvalue is $-0.236$ Ha against an `e0min` of $-2.0$; Al's is
$-2.13$ against $-4.56$. So the `e0min` branch is exercised on a synthetic
spectrum instead, with a state placed below the cutoff, and asserted to change
the electron count by exactly `occmax`. A gate that never fires on the fixtures
it ships with is not tested by them.

### The derivative, and the half a single k-point could not see

$\mu$ is a `custom_jvp` over a bisection that is **never** differentiated —
Phase 0a's lesson about unrolled solvers in miniature. Differentiating the
count constraint gives, for the three differentiable arguments,

$$d\mu=\frac{\sum_{i\bf k}w_{\bf k}\tilde\delta_{i\bf k}\,d\varepsilon_{i\bf k}
+\frac{\sigma}{n_{\max}}\big(dN-n_{\max}\sum_{i\bf k}
\tilde\Theta_{i\bf k}\,dw_{\bf k}\big)}
{\sum_{i\bf k}w_{\bf k}\tilde\delta_{i\bf k}} .$$

On fcc Al, along a general direction (not a single entry — that is the trap
§0b measured), forward mode and reverse mode agree to $<10^{-13}$ relative and
central FD to $<10^{-6}$. All three arguments are checked, because getting one
term of that expression wrong leaves the other two right.

**And the weights do not cancel.** Elk reduces Si's 2×2×2 mesh to 3 k-points
with unequal weights; replacing them with uniform ones — the shape the Phase 1
rule has — moves $\mu$ by more than $10^{-4}$ Ha. Asserted to move, so "the
weights are wired in" cannot regress into "the weights happened to be equal".

### The denominator vanishing is physics, and it is refused

In a gap at small $\sigma$ no state responds: the electron count stops
determining $\mu$, every occupancy is exactly 0 or $n_{\max}$, and $d\mu$ is
genuinely $0/0$. `check_fermi_level_determined` raises, the zone-summed
sibling of `projector.check_fermi_level_determined`.

Si at Elk's **default** `swidth` does *not* trigger it: the response is
$5.5\times10^{-4}$, which places the conduction band about seven smearing
widths above $\mu$ on this mesh — the 2×2×2 grid does not sample the
conduction-band minimum. Narrow the smearing to $10^{-4}$ and it underflows to
exactly zero and the refusal fires. The forward Fermi level is unaffected
either way; only the derivative is refused.

### What this does not settle

`sdelta`/`stheta` for `stype` 0–2 are checked against each other (AD of the
step is the delta) and against the constraint, **not** against Elk: `occupy`
was run at Elk's default `stype=3` on both fixtures. The study's Phase 3
"Gradient B" wants the whole force suite under `stype = 0,1,2,3`, and that is
where those branches acquire an external reference.

Nothing here is spin-polarised. `eveqnsv` is not transcribed, so `evalsv` is
`evalfv` and `occmax` is 2; a magnetic ground state needs the
second-variational step before its occupations mean anything.

---

## 3b. The loop, closed and iterated

### What was at stake

§2i and §2j each reproduce Elk pointwise, and §2j's own closing line says what
was missing: "This is one *half-step* with Elk's occupations, not a fixed-point
iteration." With §3a the cycle can be closed —

$$v \to \{u_\ell,u^{\rm lo}_\ell\} \to H_{\bf k},O_{\bf k}
\to \varepsilon_{i\bf k},c_{i\bf k} \to \mu,n_{i\bf k} \to \rho
\to v_{\rm cl}[\rho]+\hat Sv_{xc}[\rho] = F(v)$$

— and the open question §2i named was never a transcription problem: it is
whether the composition is **stable**, which nothing in Phase 2 tested.

### What is held fixed, and why Elk's own answer is still a fixed point

Four things are inputs here that `gndstate` recomputes every iteration: the
core density `rhocr` and its eigenvalue sum `evalsumcr` (`gencore`), the target
charge `chgtot`, and the linearisation energies (`linengy`). Each is a
functional of $v$, so freezing them changes $F$. But every one of them is
evaluated **at Elk's converged potential**, so Elk's own $v^*$ still satisfies
$v^*=F(v^*)$ and every forward check below is valid. What is *not* valid is
calling this "the SCF": the trajectory is not Elk's.

### Two forward pins, and the first one was wrong

`genvsig` had never been transcribed. The trap is one factor:

$$\tilde v_s({\bf G}) = \frac1N\sum_{\bf r}
v_s({\bf r})\,\Theta({\bf r})\,e^{-i{\bf G}\cdot{\bf r}},$$

the transform of the potential **times the characteristic function**.
`rfirftoc.f90`'s first line is `zfft = rfir*cfunir`; `rfirctof`, the map back
to the fine grid that the density uses, has no such factor — so
`density.coarsen`, which is `rfirctof`'s exact inverse, is *not* `rfirftoc`.
Using it is wrong by a **factor of 12**, and the symptom downstream is not a
slightly wrong potential: it pushes the whole spectrum below `e0min`, §3a's
gate zeroes every occupancy, and the density comes back empty. Pinned before
the loop, as an assertion in both directions.

| | |
|---|---|
| `vsig` from this transform, vs Elk's own | **2.6e-8** |
| the same without `cfunir` | 12.1 |
| `vsmt` from `potks`, vs `lapw["vsmt"]` | **2.0e-15** |
| `vsir` from `potks`, vs `groundstate["vsir"]` | **1.4e-15** |

The second pin is not a repeat of §2i, which compared against
`vclmt + vxcmt`: this compares against the array `radial.potential_arrays`
actually unpacks, which is the one the loop feeds back. A right potential in
the wrong packing fails here and passes there.

### And the 2.6e-8 is Elk's own mixing step, in a place worth remembering

`init0.f90:633-641` makes the mixer's target array `vsbs` = [`vsmt`, `vsirc`]
— the **coarse** interstitial potential — while `vsir` is a separate array that
`potks` fills from $v_{\rm cl}+v_{xc}$ and *nothing mixes*. `genvsig` reads
`vsirc`. So the export carries a `vsir` one un-mixed step ahead of the `vsig`
built from it. Measured: 2.6e-8 at Elk's default `epspot` (1e-6) and 2.0e-9 at
`epspot=1e-8`, i.e. it tracks where Elk's SCF stopped.

Two consequences. Elk's own SCF variable is
$(v^{\rm MT},\tilde v^{\rm I}_{\rm coarse})$ while this module iterates
$(v^{\rm MT},v^{\rm I}_{\rm fine})$ — the extra high-$|G|$ tail of the fine
array is inert, nothing in the step reads it, so the two have the same fixed
point. And this is the **fourth** instance of the same pattern: §1k's `haa` at
3e-10 (which is what patch 0015's `genapwlofr` call fixed), §2h's muffin-tin
density at 1.2e-10, §2j's two `evecfv` exports at 3.7e-11, and now `vsig`. Elk
mixes in the middle of its own iteration, and any two arrays written on
opposite sides of that line disagree by the last mixing step.

### Elk's converged potential is a fixed point of the map

One whole iteration of $F$ starting at Elk's own answer: radial functions,
`apwalm`, both interstitial blocks, the zone eigensolve, `occupy`, all of
`rhomag` including both symmetrisations, the Weinert solve and the functional
— composed once and landing back where they started.

**Split by region, because the packed norm is not one number.** The muffin-tin
half carries the nuclear $-Z/r$ at the first radial point and is of order
$4.8\times10^8$; the interstitial half is of order $10^2$. A single relative
bound on the packed vector is therefore a statement about the muffin tin alone,
and would admit an interstitial error of $10^{-6}$ absolute:

| | $\lVert r\rVert$ | $\lVert v\rVert$ | relative |
|---|---|---|---|
| muffin tin | 8.6e-7 | 4.8e8 | **1.8e-15** |
| interstitial | 1.1e-7 | 1.1e2 | **1.0e-9** |

The second is not roundoff and is not claimed to be. The start mixes Elk's own
**mixed** `vsmt` with its **unmixed** `vsir` — the two sides of the line the
previous subsection describes — so a residual at the scale of Elk's last mixing
step is what this start point can give. The muffin-tin figure is the one that
says the cycle closes.

### The energy no longer imports the eigenvalue sum or the entropy

§2f had to import `evalsum` and `engyts`: both need a zone sum and the
occupations. With §3a they are computed, and every term of `energy.f90` this
port builds now agrees with Elk's exported scalars:

| term | relative |
|---|---|
| `evalsum` | 2.5e-11 |
| `engyts` | 7.6e-7 (of $-9.5\times10^{-6}$ Ha, i.e. 7e-12 Ha) |
| `engyvcl` | 1.3e-11 |
| `engymad` | 0.0 |
| `engyx` / `engyc` | 2.7e-11 / 4.0e-11 |
| `engykn` | 3.0e-11 |
| **`engytot`** | **2.2e-11** ($1.3\times10^{-8}$ Ha) |

$\mu$ from this port's own eigenvalues lands $4.3\times10^{-10}$ Ha from Elk's
`efermi` — §3a's bitwise agreement was on Elk's eigenvalues, this one has the
assembly in the path. What is still imported is the **core** half of `evalsum`
(an input at fixed potential, exactly as `rhocr` is) and `engynn`, a property
of the lattice rather than of the density.

### It converges, and to Elk's fixed point rather than its own

Starting at $v^*$ tests nothing. The start is §1k's smooth valence-region bump
at 5% of the muffin-tin potential: initial residual $\lVert F(v)-v\rVert=0.80$
and initial distance $\lVert v-v^*\rVert=0.30$. Linear mixing at $\beta=0.4$:

| iteration | $\lVert F(v)-v\rVert$ | $\lVert v-v^*\rVert$ |
|---|---|---|
| 0 | 8.0e-1 | 3.0e-1 |
| 5 | 1.0e-1 | 1.2e-1 |
| 10 | 1.4e-2 | 1.5e-2 |
| 20 | 1.3e-4 | 1.4e-4 |
| 29 | **1.8e-6** | **2.6e-6** |

The two columns track each other all the way down, which is the result: the
iteration converges to **Elk's** fixed point and not to one of its own. After
the first few steps the residual falls geometrically at about 0.62 per
iteration. At the stopping point,

| | |
|---|---|
| `engytot` vs Elk's exported value | **3.0e-8 Ha** |
| $\mu$ vs Elk's `EFERMI.OUT` value | **1.4e-9 Ha** |

both inside the study's own $10^{-6}$ Ha and $10^{-8}$ Ha, and both set by
where the iteration was stopped rather than by the map. The **iteration count
is deliberately not compared** with Elk's: the study withdraws that criterion
itself, since a different eigensolver and a different mixer make the step at
which the residual crosses a threshold a coin flip.

Cost, since it decides what can be a test: about 41 s for the first iteration
(compilation) and 13-17 s after, so ~8 minutes for 30 — gated behind
`ELKPY_RUN_SLOW_TESTS`, like the DFPT phonon suite.

### What this does not settle, named rather than implied

**It is forward only.** `rhomagk`'s `epsocc` skip is a Python `continue` on the
occupation value, so the density accumulation needs concrete arrays and the
step cannot be traced. Turning that skip into a zeroed weight is exactly
equivalent (the state contributes nothing either way) and is what study Phase 3
"Gradient A" needs. Nothing about the implicit-differentiation machinery has
been exercised here; `elkjax.fixedpoint` supplies the forward iteration only.

> **Superseded by §3c.** The `epsocc` skip is a zeroed weight now
> (`density.skip_below_epsocc`), `elkjax.response` replaced JAX's own `eigh`
> rule, and `scf.step` traces — its `jvp` is checked against a central
> difference in `tests/test_calculation_scf.py`. What remains true is the last
> sentence: nothing has been differentiated *through* the converged fixed
> point.

**One structure, one mixer, no metal.** Bulk Si, linear mixing, `stype=3`.
Anderson is available (`history>0`) and untried; the study's Gradient A is
precisely the comparison between the two, and it needs the traced step first.

**Scalar only.** `eveqnsv` is not transcribed, so a spin-polarised or
spin-orbit ground state is refused (`scf.check_scalar`) rather than silently
treated as first-variational.

---


## 3c. A calculation from the input file

### What was at stake

§3b closed the loop and §3b's own last section says what it did not do: the
starting potential was Elk's converged one, perturbed by hand. Elk's converged
potential is a fixed point of $F$, so a loop started there has nothing to do,
and a loop started 0.30 away from it in potential norm is still a loop whose
starting point was *built out of the answer*. The port could not be handed an
`elk.in` and asked for a ground state.

Nothing was missing from the map. What was missing was a way to reach the
export queries without `readstate`.

### The seam, and why it is one line of Fortran rather than a transcription

`gndstate.f90` has exactly two branches:

```
if (trdstate) then
  call readstate
else
  call rhoinit; call maginit; call potks(.true.)
end if
call genvsig
```

Task 1 takes the first. Every export in patches 0013-0023 is reached through
task 1, which is why every measured number in Phases 1-3 describes a converged
calculation. Patch **0024** adds task **9006**
(`src/elkpy_initstate.f90`), which takes the second, and then runs the top of
`gndstate`'s first iteration — `gencore`, `linengy`, `genapwlofr`, `gensocfr`,
`genevfsv`, `occupy` — and stops. No `rhomag`, no `potks` on a new density, no
`mixerifc`. Every array the 9002 queries read then holds its iteration-zero
value.

Two consequences worth stating. The density is `rhoinit`'s superposition of
free atomic densities, and it is **not normalised** — `rhonorm` acts on the
density the SCF produces, not on the starting guess, so the count comes out
27.99358 against 28 on Si, 6.4e-3 short. That is a useful marker rather than a
defect: a converged density is normalised to machine precision, so the electron
count alone distinguishes this export from every other one. And because no
mixing has happened, `vsmt` and `vsir` are for once on the **same** side of
`mixerifc` — the trap CLAUDE.md records four separate encounters with does not
apply to this export.

The alternative was transcribing `init0`/`init1`/`rhoinit`, which needs
`allatoms` → `atom.f90`, a radial Dirac solver for every free atom, plus Elk's
grid/symmetry/species bookkeeping. None of that is a functional of the density
— it is identical at every iteration and for every potential, so no gradient
passes through it. This is exactly the case `docs/design.md` §8 says to export.

### The 2.0 Ha that was a wrong formula, not a tolerance

The first run converged cleanly and missed Elk's total energy by **2.025 Ha**,
which is 3.5e-3 relative — far too large to be the frozen core and far too
small to be a broken map. It was neither. It was `evalsumcr`.

`energy.f90` builds the kinetic energy as
$\Sigma_\varepsilon-\int\rho v_{cl}-\int\rho v_{xc}$, and the core's share of
that is `energykncr`,

$$T_{\rm core} \;=\; \sum_{\rm core} f\,\varepsilon \;-\; \int\rho_{\rm core}\,v_s ,$$

with **both** halves at the current potential: Elk recomputes
$\varepsilon_{\rm core}$ every iteration in `gencore`, and the $\rho$ in
$\int\rho v_{cl}$ includes the core. §3b imported `evalsumcr` — the first term
alone — and that was harmless there only because the loop converged to the very
potential `evalsumcr` had been evaluated at. Started from the atomic
superposition it is not harmless: the same $\rho_{\rm core}$ is integrated
against a potential its eigenvalues never saw, and the mismatch is *first
order* in $v-v^{\rm init}$ over 20 core electrons sitting exactly where the
potential moves most.

The numbers say it plainly. Between the initial and the converged export,

| | initial | converged | moved |
|---|---|---|---|
| `evalsumcr` | $-317.3833$ | $-315.3483$ | **2.035** |
| `engykncr` ($T_{\rm core}$) | $567.7239$ | $567.7254$ | **1.5e-3** |

so the quantity that may be frozen is $T_{\rm core}$, and the quantity that was
being frozen moved by three orders of magnitude more. Patch 0024 exports
`engykncr`; `elkjax.energy.core_eigenvalue_sum` holds it fixed and rebuilds
$\int\rho_{\rm core}v_s$ at whatever potential it is handed.
`core_potential_energy` reproduces Elk's own $\int\rho_{\rm core}v_s$ — which
is `evalsumcr - engykncr` — to **5.1e-16** at the initial export and
**3.9e-16** at the converged one, the two potentials being 2 Ha apart in that
integral, so the transcription is pinned where it matters and not only where it
is easy. At the export's own potential the correction returns `evalsumcr`
exactly, so §3b's numbers are unchanged.

This is **not** another instance of the mixing-side trap CLAUDE.md records
four encounters with — those were two arrays written on opposite sides of
`mixerifc`, and the fix was always to move a call. Here both arrays are
correct and the hidden potential dependence is in the *formula* that combines
them. The symptom is the same shape, so it is worth separating: a mixing-side
disagreement is the size of `epspot`, and this one is 2 Ha.

### It runs, and the gap is the frozen core and nothing else

Bulk Si, `xc=PW`, `ngridk=(2,2,2)`, `rgkmax=7`, Elk's own reduced k-set,
linear mixing at $\beta=0.4$, `tol=1e-7` on $\|F(v)-v\|$. The start is
`rhoinit`'s atomic superposition, whose Fermi level is 0.1249 Ha against the
converged 0.2140 — not a perturbation of the answer.

**40 iterations, residual 9.3e-8**, and then

| | elkjax | Elk (task 0) | difference |
|---|---|---|---|
| $E_{\rm tot}$ | $-577.9838797268$ | $-577.9835159872$ | $-3.64\times10^{-4}$ Ha |
| $E_F$ | $0.2139904581$ | $0.2140331833$ | $-4.27\times10^{-5}$ Ha |

$6.3\times10^{-7}$ relative on a $-578$ Ha all-electron total. Cost: about 3
minutes wall, dominated by one XLA compile of the step (`scf.run(jit=True)`,
new here — the un-jitted loop is ~16 s/iteration and this is ~2 s).

**And the whole of that difference is the frozen core.** The one experiment
that settles it: take the *converged* export's `rhocr` and `engykncr`, put them
into the *initial* triple, and start from the same initial potential. Nothing
else changes — same map, same mixer, same start, same 40 iterations, residual
9.2e-8. The result is

$$E_{\rm tot}-E_{\rm tot}^{\rm Elk} = -3.8\times10^{-8}\ {\rm Ha},\qquad
E_F-E_F^{\rm Elk} = -4.5\times10^{-9}\ {\rm Ha},$$

i.e. exactly §3b's own agreement, reached from a start that has nothing to do
with Elk's answer. So the map, the Fermi level, the density, the potential and
the total-energy assembly are all right to 1e-8 Ha from cold; what the port is
missing to close the last $3.6\times10^{-4}$ Ha is `gencore` — a radial Dirac
solver in the loop — and nothing else. That is a concrete, bounded, named piece
of work rather than an unexplained residual.

### The linearisation energies are frozen too, and here that is exact

`linengy.f90` calls `findband` only for an APW or local orbital whose species
file sets `apwve`/`lorbve` true; otherwise `apwe`/`lorbe` keep the
`apwe0`/`lorbe0` that `init1` copied in, for the whole run. Elk's stock species
files — Si's included — set every flag false, and `autolinengy` defaults false,
so freezing them across this loop changes nothing at all.

That is a property of the input, not of the port, so it is checked rather than
assumed: patch 0024 exports `autolinengy` and the per-orbital flags, and
`elkjax.driver.check_linearisation_frozen()` raises on a species file that
would have searched. `apwe` alone cannot detect this — a searched energy and a
default one are the same kind of number.

### What this does not settle

**The core is still frozen.** Named above with its price measured: 3.6e-4 Ha
in the total energy and 4.3e-5 Ha in the Fermi level on bulk Si. Elk's own
`gencore` runs `rdirac` per core state per atom per iteration; transcribing it
is the next forward step and is independent of everything else here.

**One structure, one mixer, no metal.** Bulk Si, linear mixing, `stype=3`.
Anderson (`history>0`) is available and was not needed — linear mixing at 0.4
converged from the atomic superposition without trouble — so it remains
untried at this distance.

**Still scalar only.** `eveqnsv` is not transcribed;
`scf.check_scalar` refuses a spin-polarised or spin-orbit ground state.

**Forward only, still.** The step is now traceable (`elkjax.response` replaced
JAX's own `eigh` rule, and §3b's `jvp` check passes), but nothing here
differentiates *through the converged fixed point*: `elkjax.fixedpoint`'s
implicit route has not been exercised on this map, and none of Phase 3's three
gradient signatures has been started.

---
