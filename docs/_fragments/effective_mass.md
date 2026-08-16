# The effective mass tensor from the k.p sum rule

*Fragment for `docs/design.md` (a subsection of §22, which already covers the
momentum matrix elements this is built on) and for a `\part` of
`docs/physics.tex`. Not published as-is.*

`Calculation.get_effective_mass_sum_rule(k, ist)` /
`parsers.optical.effective_mass_tensor()` compute the band-curvature (inverse
effective mass) tensor from momentum matrix elements alone — the third consumer
of §22's `MOMENTUM` query, after the optical selection rules and the Kubo-form
quantum geometry, and, like them, an *independent code path* for something elkpy
already computes another way.

## The formula

In the Bloch Hamiltonian obtained by pulling the plane-wave factor out of the
wavefunction,

$$ H(\mathbf k)=e^{-i\mathbf k\cdot\mathbf r}He^{i\mathbf k\cdot\mathbf r}
=\tfrac12(\mathbf p+\mathbf k)^2+V_s(\mathbf r) $$

(Hartree atomic units, $\hbar=m_e=1$), the $\mathbf k$-dependence is explicit and
elementary:

$$ \frac{\partial H}{\partial k_a}=p_a+k_a=v_a,\qquad
\frac{\partial^2H}{\partial k_a\partial k_b}=\delta_{ab}. $$

Ordinary non-degenerate second-order perturbation theory in $\mathbf k$ then gives
the band curvature exactly, with no derivative of anything numerical:

$$ \boxed{\;\Big(\frac{1}{m^*}\Big)^{ab}_n=\frac{\partial^2\varepsilon_n}{\partial k_a\partial k_b}
=\delta_{ab}+2\sum_{m\neq n}\frac{\mathrm{Re}\big[p^a_{nm}p^b_{mn}\big]}{\varepsilon_n-\varepsilon_m}\;} $$

with $p^a_{nm}=\langle\psi_{n\mathbf k}|(-i\nabla+\text{SOC})_a|\psi_{m\mathbf k}\rangle$
the momentum matrix elements §22 exports (for a local Kohn–Sham potential these are
equally the velocity matrix elements). The mass tensor $m^*_{ab}$ is the matrix
inverse; `"inverse_mass"` is the primitive here, and `"mass"` is `None` when that
inverse does not exist — a direction of vanishing curvature is an infinite mass, not
an error.

Each symbol: $n$ the band whose mass is asked for, $m$ every other state at the same
$\mathbf k$, $\varepsilon$ the Kohn–Sham eigenvalues (Hartree), $a,b$ Cartesian axes.

**Reading the formula physically.** The $\delta_{ab}$ is the bare free-electron term:
an electron with no interband coupling at all has $m^*=m_e$. Every deviation from unit
mass is interband repulsion — states *below* band $n$ ($\varepsilon_m<\varepsilon_n$,
positive denominator) push the curvature up, states *above* push it down, each weighted
by how strongly the velocity operator connects them. That decomposition — *which*
coupling to *which* band produces the mass — is real physics a finite-difference
curvature cannot deliver, and `decompose=True` returns it term by term.

**Why `free_electron_term=False` exists.** The $\delta_{ab}$ comes from
$\tfrac12(\mathbf p+\mathbf k)^2$, i.e. from the fact that the crystal Hamiltonian's
basis is the complete Hilbert space. A model Hamiltonian written directly in a finite
band basis — a $\mathbf k\cdot\mathbf p$ or tight-binding $H(\mathbf k)$ — has
$\partial^2H/\partial k_a\partial k_b=0$ instead, and the sum over the model's own bands
is the whole answer. The flag is what makes the massive-Dirac unit tests an exact,
analytic pin rather than an approximate one.

## The Thomas–Reiche–Kuhn f-sum, stated honestly

`parsers.optical.oscillator_strength_sum()` returns

$$ f^{ab}_n=\sum_{m\neq n}\frac{2\,\mathrm{Re}\big[p^a_{nm}p^b_{mn}\big]}{\varepsilon_m-\varepsilon_n}, $$

the usual velocity-form oscillator strengths of the transitions out of band $n$. This is
the *same arithmetic* as the mass with the denominator reversed, so identically, at every
$\mathbf k$-point,

$$ f^{ab}_n=\delta_{ab}-\Big(\frac{1}{m^*}\Big)^{ab}_n . $$

"One unit of oscillator strength per electron" is therefore **not** a pointwise statement
in a crystal — read as one, it would assert that every band is flat. The true statement is
the Brillouin-zone average: for a filled band the zone average of
$\partial^2\varepsilon_n/\partial k_a\partial k_b$ vanishes (it is the second derivative of
a periodic function integrated over a period), hence

$$ \big\langle f^{ab}_n\big\rangle_{\rm BZ}=\delta_{ab}, $$

which is the f-sum rule behind the optical conductivity's spectral weight,
$\int\sigma_1(\omega)\,d\omega=\pi n/2$. Its practical role here is as a truncation
diagnostic with an exactly known target.

## Convergence: the first power of $\Delta\varepsilon$ is the whole story

The denominator appears to the **first** power, unlike §22's Kubo geometric sums
$T_{ab}\sim1/(\Delta\varepsilon)^2$. High-lying intermediate states are suppressed only as
$1/\Delta\varepsilon$, so the sum converges markedly more slowly in `nempty`, and three
distinct kinds of missing state matter:

1. everything above `nempty` (fixable, and the thing the sweep below measures);
2. the core states, which are not among the `nstsv` valence states at all — they lie
   *below* band $n$, so their omission biases the mass in the **opposite** direction from
   (1), which is what makes the observed overshoot self-consistently attributable to (1);
3. the high continuum the finite (`rgkmax`), energy-linearized LAPW basis cannot represent
   even in principle.

**Sign logic, which is what makes an under-converged number readable.** Every state removed
by truncation lies *above* band $n$, so each omitted diagonal term
$-2|p^a_{nm}|^2/|\Delta\varepsilon|$ is negative: a truncated $(1/m^*)^{aa}$ is an
**overestimate**, and falls monotonically as `nempty` rises. That monotonicity does not
extend to off-diagonal components, whose terms carry either sign.

`nstates=N` truncates the $m$-sum from Python. That is *exactly* equivalent to having run
Elk with the corresponding smaller `nempty` — verified directly on bulk Si, where the
`nempty=8` run's own full sum and the `nempty=40` run truncated to the same 21 states agree
to $10^{-11}$ (their eigenvalues agree to $10^{-11}$ too), which is what makes a whole
convergence sweep affordable out of a single ground state.

## Verification against a real compiled binary (bulk Si)

`tests/test_calculation_effective_mass.py`; Si, `ngridk=(4,4,4)`, `rgkmax=7.0`,
`nempty=40` (85 states), compared against `get_effective_mass()` (Elk task 25,
`src/effmass.f90`), which fits a polynomial to *eigenvalues* on a 27-point mesh of
Cartesian displacements `deltaem=0.025` around the point and differentiates it. Task 25's
"matrix of eigenvalue derivatives" is the inverse-mass tensor; its "effective mass tensor"
is that matrix inverted.

**Γ, band 1** (the non-degenerate $\Gamma_1$ $s$-bonding band, isolated by 12 eV):

| states retained | $(1/m^*)^{xx}$ | deviation from task 25 | ${\rm Tr}\,f/3$ |
|---|---|---|---|
| 4 (occupied only) | 1.0000 | +16.1% | 0.000 |
| 8 | 0.9276 | +7.7% | 0.217 |
| 20 | 0.8984 | +4.3% | 0.305 |
| 40 | 0.8914 | +3.5% | 0.326 |
| 60 | 0.8900 | +3.4% | 0.330 |
| 85 (all) | 0.8876 | +3.1% | 0.337 |
| task 25 (finite difference) | 0.8611 | — | 0.139 |

Monotone from above, exactly as the sign argument requires, and still descending at 85
states: the sum rule recovers $0.1124/0.1389\approx81\%$ of the interband repulsion that
lowers the curvature below the free-electron 1. The residual 19% is the missing continuum —
the highest computed state sits at only 3.7 Ha above the band.

**A generic, low-symmetry k-point** $(0.2,0.15,0.1)$, band 1, where the tensor is not forced
isotropic and the off-diagonal components are a real test:

| | $xx$ | $yy$ | $zz$ | $xy$ | $xz$ | $yz$ |
|---|---|---|---|---|---|---|
| sum rule (85 states) | 0.8675 | 0.8656 | 0.8642 | −0.0199 | −0.0073 | −0.0050 |
| task 25 | 0.8369 | 0.8360 | 0.8352 | −0.0208 | −0.0076 | −0.0053 |

3.5–3.7% on the diagonal (same convergence trend: 15.1% → 3.6%), and 4–6% on the
off-diagonals, which are two orders of magnitude smaller and carry the correct sign and
magnitude.

**Where the two routes disagree more, and whose fault it is.** For bands with a close
neighbour (min gap $\sim0.02$ Ha) at the generic point, differences reach a few tenths of an
atomic unit (e.g. band 5, $xx$: −1.14 vs −1.36). Part of that is task 25's, not the sum
rule's: its polynomial fit samples over $\pm0.025$ Bohr$^{-1}$, comparable to the scale on
which those bands curve near an avoided crossing, so the quadratic fit is itself
questionable there. Neither route is ground truth in that regime.

**The decomposition, and a selection rule that could not be a numerical accident.** At Γ,
diamond Si's states have definite parity about the bond centre and $\mathbf p$ is odd, so a
transition between two even states is forbidden. Band 1 ($\Gamma_1$, even) accordingly gets
*exactly nothing* from the even valence triplet $\Gamma_{25'}$ (bands 2–4): those
contributions are $\sim10^{-30}$, i.e. zero at machine precision, not merely small. Its
entire mass comes from the odd conduction triplet $\Gamma_{15}$ (bands 5–7, $-0.0241$ each,
together 64% of the total deviation from 1) and its higher analogues at 1.18 Ha. This is the
textbook selection rule behind Si's optical spectrum ($\Gamma_{25'}\to\Gamma_{15}$ allowed,
$\Gamma_1\to\Gamma_{25'}$ forbidden), read straight off the mass decomposition.

**The f-sum rule, Brillouin-zone averaged.** Summing $f_n$ over Si's four occupied bands on a
generic (unshifted-symmetry-free) $\mathbf k$-mesh and averaging:

| mesh | $\langle{\rm Tr}f/3\rangle$ per occupied band, 85 states | at 40 states |
|---|---|---|
| $4^3=64$ points | 1.103 | 1.096 |
| $6^3=216$ points | 0.971 | 0.964 |
| $8^3=512$ points | 0.932 | 0.924 |

against the exact value 1. The two error sources are separable and pull in opposite
directions. Truncation makes $f$ too *small* (equivalently $\langle1/m^*\rangle_{\rm BZ}$ too
*large*, since every omitted term is negative pointwise): at fixed mesh, adding states raises
$f$ monotonically (0.888 → 0.932 going from 12 to 85 states on the $8^3$ mesh), and the
remaining deficit of $\sim7\%$ is the missing continuum, the same physics as the $19\%$ at Γ
above but averaged over a whole band manifold rather than one state. Mesh discretization of
the average is the other error and is *not* small at $4^3$ — it is what pushes that row above
1, i.e. to the physically impossible side, and it is why $\langle1/m^*\rangle_{\rm BZ}$ comes
out $-0.103$ there instead of the required non-negative value. From $6^3$ on the sign is at
least correct ($+0.029$, $+0.069$), but the truncation deficit is mesh-independent by
construction while those two differ by $2.4\times$, so mesh error still cancels roughly half
the truncation bias at $6^3$; only at $8^3$ is the residual mostly truncation, and it is still
drifting (successive differences $-0.132$, $-0.039$, extrapolating to $\approx0.92$, a deficit
of $\approx8\%$). Treat this as a consistency check at the several-percent level, not a sharp
one.

**Synthetic pins, ahead of any Elk run** (`tests/test_parsers_optical.py`): on the massive
Dirac model $H_\tau=v(\tau k_x\sigma_x+k_y\sigma_y)+\Delta\sigma_z$, where
$\mathbf p=\partial H/\partial\mathbf k$ is exact and the two-band basis is complete, the sum
rule reproduces the analytic Hessian
$\partial^2\varepsilon_\pm/\partial k_a\partial k_b=\pm(v^2\delta_{ab}/E-v^4k_ak_b/E^3)$,
$E=\sqrt{v^2k^2+\Delta^2}$, to machine precision at every $k$ tried and for both bands
(with `free_electron_term=False`, per the argument above) — the same
eigenvalue-derivative-vs-matrix-element comparison the Si test makes, here with no truncation
to blur it. Plus: symmetry of the tensor, the two-band cancellation
$(1/m^*)_1+(1/m^*)_2=0$, the decomposition summing to the total, `nstates` truncation
matching a genuinely shorter input, the degeneracy guard, and $f^{ab}=\delta_{ab}-(1/m^*)^{ab}$
together with its closed form $f_{xx}=v^2/\Delta$ at the Dirac point.

## Guard

The sum rule above is *non-degenerate* perturbation theory. If band `ist` is within
`degeneracy_tol` (default $10^{-4}$ Ha) of any other retained state, `ValueError` is raised
rather than dividing by an arbitrary splitting: within a degenerate multiplet a single band's
curvature is not defined at all (the partners' dispersions cross), and Elk's own
finite-difference task 25 is equally meaningless there. Si's three-fold $\Gamma_{25'}$ valence
top is the canonical case, and is asserted to raise.

## How to use in code

```python
si = Structure(SI_AVEC, SI_SPECIES).get_calculation(
    "si", xc="PW", ngridk=(4, 4, 4), rgkmax=7.0,
    extra_blocks={"nempty": [40]},      # the k.p sum's slow 1/dE tail needs these
)
si.get_energy()

# one-off wrapper: momentum matrix elements at k, then the sum rule
kp = si.get_effective_mass_sum_rule((0.0, 0.0, 0.0), ist=1)
kp["inverse_mass"]     # (3,3) d^2 eps/dk_a dk_b, atomic units
kp["mass"]             # its inverse, or None if a direction is flat

# the independent route Elk already provides, for comparison
fd = si.get_effective_mass((0.0, 0.0, 0.0))
fd[0]["derivative_tensor"]   # <- compare against "inverse_mass"
fd[0]["tensor"]              # <- compare against "mass" (that matrix inverted)

# convergence sweep and the interband decomposition, from ONE run
from elkpy.parsers import optical

with si.eigenstate_session() as session:
    m = session.momentum((0.0, 0.0, 0.0))

for n in (8, 20, 40, len(m.energies)):
    optical.effective_mass_tensor(m.energies, m.pmat, 1, nstates=n)["inverse_mass"]

per_band = optical.effective_mass_tensor(
    m.energies, m.pmat, 1, decompose=True
)["contributions"]          # (nstates, 3, 3): which coupling makes the mass

optical.oscillator_strength_sum(m.energies, m.pmat, 1)   # TRK f-sum tensor
```
