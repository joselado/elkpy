# 24. Spin Berry curvature and the intrinsic spin Hall conductivity

*Draft fragment, to be folded into `docs/design.md` as §24 with a matching
`\part` in `docs/physics.tex`.*

`parsers.spin_hall.spin_berry_curvature()` /
`EigenstateSession.spin_current_operator()` (no new task number;
`patches/0009-momentum-evecsv.patch`) give the spin Berry curvature of an
occupied band window, and `parsers.spin_hall.spin_hall_conductivity()`
folds a Brillouin-zone mesh of those into the intrinsic spin Hall
conductivity.

## What was blocking this, and why the fix is one array

Everything this needs already existed, in two halves that could not legally
be multiplied together:

- the **velocity matrix elements** $v_a=p_a$ of §22's `MOMENTUM` query,
  which returned `energies` and `pmat`;
- the **spin operators** $S_x,S_y,S_z$ of §17, built in pure Python from
  `evecsv`'s spin-up/spin-down row blocks ($i=p+(\mathrm{ispn}-1)\,$`nstfv`),
  obtained from the `EIGENSTATES` query.

Those are two *separate* diagonalisations at the same $k$. §14 spells out
why that matters: a degenerate multiplet's eigenvectors are only defined up
to a unitary rotation within the multiplet, and two independent
diagonalisations are free to — and empirically do — pick different ones. A
product $S_z v_a$ formed across that boundary is meaningless, and, crucially,
**nothing cheap detects it**: $S_z$ stays Hermitian with eigenvalues in
$[-\tfrac12,\tfrac12]$, `evecsv` stays unitary, `pmat` stays Hermitian, and
the resulting curvature is a plausible finite number. The failure is silent.

Patch 0009 is therefore small on purpose: `elkpy_momentum` gains one
`intent(out)` argument, `evecsv_out(nstsv,nstsv)`, written where the routine
previously wrote a local array it then discarded; and the session's
`MOMENTUM` case prints that array in the same `do b; do a` column-major
block the `EIGENSTATES` response already uses. No new upstream subroutine is
called, no new task number, no extra computation — a `MOMENTUM` response is
now literally an `EIGENSTATES` response with the three momentum components
appended. `Momentum` gains an `evecsv` field; `EigenstateSession.momentum()`
returns it unwindowed even when `ist0`/`ist1` slice `energies`/`pmat`, since
its *row* index is the first-variational spinor basis, which a band window
has no meaning for (and truncating it would break
`compute_spin_operator()`'s `nstsv == 2*nstfv` check).

`EigenstateSession.spin_current_operator(k, direction, spin)` exists so the
pairing cannot be got wrong by accident: it issues one `MOMENTUM` query and
builds $S_s$ from *that response's own* `evecsv`.

## The physics

The **conventional spin current operator** is the symmetrized product

$$ J^s_a \;=\; \tfrac12\{S_s,v_a\} \;=\; \tfrac12\left(S_s v_a + v_a S_s\right), $$

with $S_s$ the spin operator for projection $s\in\{x,y,z\}$ and $v_a$ the
velocity operator along Cartesian axis $a$. The symmetrization is not
cosmetic: $S_s v_a$ alone is not Hermitian once $[S_s,v_a]\neq0$, which is
exactly the spin-orbit-coupled case of interest. It is also load-bearing for
the discretization below — the exact cancellation of intra-window terms
needs *both* operators Hermitian.

Its Kubo linear response to an electric field along $b$ gives the **spin
Berry curvature** of an occupied band window $W$,

$$ \Omega^{s}_{ab}(\mathbf k) \;=\; -2\,\mathrm{Im}\sum_{n\in W}\sum_{m\notin W}
   \frac{\langle n|J^s_a|m\rangle\,\langle m|v_b|n\rangle}{(\varepsilon_n-\varepsilon_m)^2}, $$

and the **intrinsic spin Hall conductivity** is its Brillouin-zone integral,

$$ \sigma^{s}_{ab} \;=\; \int_{\mathrm{BZ}}\frac{d^d k}{(2\pi)^d}\;\Omega^{s}_{ab}(\mathbf k). $$

Structurally this is §22's Kubo quantum geometric tensor with the first
velocity factor replaced by the spin current, so it reuses that module's
arithmetic verbatim (`parsers.optical.kubo_sum`, generalized in this change
from "a `pmat` plus two axis indices" to "two Hermitian operator matrices").
It therefore inherits §22's sign convention unchanged —
$\mathbf A=i\langle u|\nabla_{\mathbf k}u\rangle$, $\Omega=\nabla\times\mathbf A$
(Xiao, Chang & Niu, RMP **82**, 1959 (2010)) — and setting $S\to\mathbb 1$
recovers the ordinary charge Berry curvature exactly, which is the cheapest
available pin on that shared sign and on the absence of a stray factor in
the anticommutator.

**Why only the window sum is exposed.** The textbook per-band $\Omega^s_n$
sums over *all* $m\neq n$, including states inside $W$. For the
window-summed quantity those intra-window terms cancel **exactly**, for any
Hermitian $J$ and $v$: writing $X=\langle n|J|m\rangle$ and
$Y=\langle m|v|n\rangle$, the reversed pair contributes
$\langle m|J|n\rangle\langle n|v|m\rangle=X^*Y^*=(XY)^*$ over the identical
denominator, so the two imaginary parts cancel. Hence
$\sum_{n\in W}\sum_{m\neq n} = \sum_{n\in W}\sum_{m\notin W}$, and the sum
never touches an intra-window energy denominator — whereas a per-band
$\Omega^s_n$ diverges on any degenerate occupied pair. The remaining
degeneracy guard is §22's unchanged: a window not separated from the states
outside it raises `ValueError` rather than being clipped.

**The $\tfrac12$, stated once because it is exactly a factor of 2.** elkpy's
$S_a$ (§17) has eigenvalues $\pm\tfrac12$, i.e. $\hbar=1$ and
$S=\vec\sigma/2$. So for a system with conserved $S_z$, where the bands
decouple into two sectors and $J^z_a=s_\sigma v_a$ within each,

$$ \Omega^{s} \;=\; \sum_\sigma s_\sigma\,\Omega^{\sigma} \;=\; \tfrac12\left(\Omega^{\uparrow}-\Omega^{\downarrow}\right), $$

*half* the bare difference. Papers quoting SHC in units of $\hbar/2e$ work
with the $\pm1$-normalized spin, so multiply by 2 to compare.

**Caveat, and it is a real one.** $\tfrac12\{S_z,v\}$ is the *conventional*
spin current, which is not conserved once spin-orbit coupling breaks spin
conservation; a "proper" definition adds a torque-dipole term. elkpy
computes the conventional one — what essentially all first-principles SHC
numbers in the literature use — and names it for what it is rather than
quietly implying conservation.

## Verification

**Synthetic** (`tests/test_parsers_spin_hall.py`, no Elk run), on §22's
massive Dirac model $H_\tau=v(\tau k_x\sigma_x+k_y\sigma_y)+\Delta\sigma_z$
where $\mathbf p=\partial H/\partial\mathbf k$ is exact. Two Dirac sectors
are glued into one four-state system with conserved $S_z$ (states sorted by
energy, `pmat` and $S_z$ permuted together so the window really is "both
valence bands"), and the result is checked against
`parsers.optical.kubo_berry_curvature` — an independent, already
sign-pinned code path — on each sector alone:

- $\Omega^s=\sum_\sigma s_\sigma\Omega^\sigma$ exactly (to $10^{-10}$), over
  all four valley combinations. Two of those combinations are the ones with
  teeth, and neither alone suffices:
  - **the time-reversal pair** ($\tau_\uparrow=+1$, $\tau_\downarrow=-1$):
    charge curvature cancels to zero while the spin curvature is maximal —
    a $J$-vs-$v$ swap would return 0 here instead of a large number;
  - **two copies of one valley**: charge curvature doubles while the spin
    curvature cancels — the opposite pattern, catching a dropped or
    mis-signed $S$ factor that the first case's coincidence hides.
- $S\to\mathbb 1$ reproduces the ordinary curvature exactly.
- $J$ is Hermitian for non-commuting random Hermitian $S$ and $v$ (with the
  unsymmetrized product asserted *not* Hermitian, so the test isn't vacuous).
- the window sum equals an explicit brute-force sum over all $m\neq n$ on a
  random Hermitian model — a direct check of the cancellation argument
  above, not a restatement of it.

**Against a real compiled binary** (`tests/test_calculation_spin_hall.py`),
on the graphene + `soc_scale={"C": 3000}` fixture §20 already uses. One
practical note first, because it bit this test: band windows here are
**explicit and gap-checked**, not read off occupation numbers. Scaling
carbon's SOC by 3000 reorders the band structure so drastically that the
cell's 8 valence electrons (confirmed in `INFO.OUT`) do not fill a fixed 8
states at every $k$ — Elk's own occupancies show 6 occupied at the first
$k$-point — so `sum(occ > 0.5)` at one $k$-point, the idiom §13/§22's tests
use on genuine insulators, does not define a band group here. Every
quantity in this section is a property of a *gapped group's projector*, so
the tests assert the boundary gap of each window they use. (This is worth
carrying over to §20's own use of the same fixture.)

Measured, at both valleys, for the two gapped groups used:

| window | boundary gap | $\Omega^{\text{charge}}$ | $\Omega^{s}(K)$ | $\Omega^{s}(K')$ |
|---|---|---|---|---|
| [1,4] | 1.25 eV | $\sim2\times10^{-4}$ | $-78.4035$ | $-78.4016$ |
| [1,6] | 10.4 eV | $<10^{-5}$ | $-0.31936$ | $-0.31936$ |

The $K/K'$ agreement is to $\sim2\times10^{-5}$ relative, across windows
whose spin curvatures differ by more than two orders of magnitude, while
the charge curvature is numerical noise about zero in both. The absolute
sign of $\Omega^s$ is a regression pin, not a prediction — it depends on
this structure's own conventions and on which `evecsv` row block is
physically "up", exactly as §17 documents for $S_z(K)$.

Specifically:

- the `MOMENTUM` response's new `evecsv` is unitary to machine precision
  (not the $\sim10^{-3}$ `genolpq` truncation floor overlaps carry — these
  are eigenvectors of one Hermitian problem);
- $S_z$ built from it has the same *spectrum* over the window as
  $S_z$ from an `EIGENSTATES` query at the same $k$, and the two responses'
  energies agree exactly. The comparison is of the spectrum, not of matrix
  elements: graphene is Kramers degenerate at *every* $k$ (inversion +
  time reversal), so an elementwise comparison would fail with nothing
  wrong — which is precisely the ambiguity patch 0009 exists to remove;
- the physics check: graphene's inversion **and** time-reversal symmetry
  together force the charge Berry curvature to vanish pointwise
  ($\Omega(-\mathbf k)=+\Omega(\mathbf k)$ from inversion,
  $-\Omega(\mathbf k)$ from time reversal), while the spin Berry curvature
  is under no such constraint. $\Omega^s$ is **even** under each symmetry
  separately: $J^z_a$ picks up sign flips from *both* $S_z$ and $v$ where
  $v$ alone picks up one, so the two flips cancel. Hence
  $\Omega^s(K)=+\Omega^s(K')$ — the *opposite* relative sign to the $K/K'$
  antisymmetry §13's Berry curvature, §17's $S_z$, §19's $L_z$ and §22's
  circular dichroism all assert, and the reason a spin Hall response is
  allowed in a time-reversal-symmetric crystal where an anomalous Hall
  response is not. A large, valley-**symmetric** spin curvature sitting on
  a vanishing, valley-antisymmetric charge one is the defining signature of
  a quantum spin Hall system. Note what the test would show if the two
  operator factors were swapped or $S$ silently dropped: the charge
  curvature, i.e. zero — so the *magnitude* is as diagnostic as the sign
  pattern here.

Not asserted: a converged $\sigma^s_{xy}$ against a literature value. The
spin Berry curvature is sharply peaked near the gapped Dirac points, so the
BZ integral converges far more slowly than a total energy on the same mesh —
the same resolution problem §20 documents for the $Z_2$ mesh. The
conductivity helper is exercised end-to-end on a coarse mesh for shape and
units only, and says so.

## How to use in code

```python
from elkpy.parsers import optical, spin_hall
from elkpy.parsers.spin import compute_spin_operator

calc = Structure(GRAPHENE_AVEC, GRAPHENE_SPECIES).get_calculation(
    "graphene", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
    spinorb=True, soc_scale={"C": 3000.0},
    extra_blocks={"nempty": [12]},   # states for the Kubo sums
)
calc.get_energy()

ist0, ist1 = 1, 4   # a gapped band group -- CHECK the boundary gap, see above

with calc.eigenstate_session() as session:
    # one MOMENTUM query: energies, pmat AND evecsv from ONE diagonalisation
    m = session.momentum((1 / 3, 1 / 3, 0))
    nstsv = m.evecsv.shape[0]
    sz = compute_spin_operator(m.evecsv, nstsv // 2, 1, nstsv)["sz"]

    # spin Berry curvature of the occupied window (Bohr^2), sigma^{s_z}_xy
    spin_hall.spin_berry_curvature(m.energies, m.pmat, sz, ist0, ist1,
                                   directions=(1, 2))

    # the charge curvature of the same window, for comparison -- zero here,
    # forced by graphene's inversion x time-reversal symmetry
    optical.kubo_berry_curvature(m.energies, m.pmat, ist0, ist1)

    # or let the session pair S_z and v for you (same single query):
    m, j = session.spin_current_operator((1 / 3, 1 / 3, 0), direction=1, spin="z")

# fold a mesh of curvatures into the conductivity (atomic units)
sigma = spin_hall.spin_hall_conductivity(curvatures, cell_volume=volume)
```

`directions` here indexes **Cartesian** axes ($1,2,3=x,y,z$), the same as
`parsers.optical` and for the same reason (`genpmatk`'s components) — not
the reciprocal-lattice convention of `get_berry_curvature()`.

## References

- Guo, Yao & Niu, *Ab initio calculation of the intrinsic spin Hall effect
  in semiconductors*, PRL **94**, 226601 (2005), arXiv:cond-mat/0505146 —
  the ancestor Kubo formula both papers below derive from, and the one whose
  $\omega\to0$ limit fixes the $e\hbar/V_c$ prefactor used here.
- Yao & Fang, *Sign Changes of Intrinsic Spin Hall Effect in Semiconductors
  and Simple Metals*, PRL **95**, 156601 (2005), arXiv:cond-mat/0502351 —
  prints the curvature with $-2\,\mathrm{Im}$, the form elkpy's own
  convention coincides with, and states the sign in words.
- Guo, Murakami, Chen & **Nagaosa**, *Intrinsic spin Hall effect in platinum
  metal*, PRL **100**, 096401 (2008), arXiv:0705.0409 — prints the same
  expression with $+2\,\mathrm{Im}$; both papers nevertheless report a
  positive SHC of comparable size for hole-doped GaAs, so the printed
  difference is absorbed in the charge vertex, not physical. (Note the
  fourth author is Nagaosa, not Niu — an easy misattribution given the
  ancestor paper above.)
- Shi, Zhang, Xiao & Niu, *Proper definition of spin current in spin-orbit
  coupled systems*, PRL **96**, 076604 (2006), arXiv:cond-mat/0503505 — the
  conserved-spin-current objection to $\tfrac12\{S_z,v\}$, not implemented
  here.
- Kane & Mele, PRL **95**, 226801 (2005), arXiv:cond-mat/0411737 (the
  current-level reduction $\mathbf J_s=(\hbar/2e)(\mathbf J_\uparrow-\mathbf
  J_\downarrow)$) and PRL **95**, 146802 (2005), arXiv:cond-mat/0506581 (the
  invariant-level one, already cited by §20) — the quantum spin Hall effect
  in graphene, the fixture's own prediction.
- Sinova, Valenzuela, Wunderlich, Back & Jungwirth, *Spin Hall effects*, RMP
  **87**, 1213 (2015), arXiv:1411.3249 — review covering both the Kubo
  conventions and the proper-current caveat.
- Xiao, Chang & Niu, RMP **82**, 1959 (2010) — the Berry-phase sign
  convention elkpy uses throughout (§22).
