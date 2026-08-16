# WCC mesh-convergence audit (follow-up to §23's cesium retraction)

*Draft fragment — not yet merged into `docs/design.md`. Results below come from a
real compiled binary, run 2026-08-16.*

§23 retracted §21's cesium ($\nu_0=1$) result: the exact Fu-Kane parity indicator
gives $(0;000)$, while refining the WCC mesh on the disputed plane gives
$z = 1, 0, 1, 0$ for $(n_{kx},n_t) = (12,7), (18,9), (24,13), (32,17)$ — an
oscillation, not convergence. The open question §23 left is **how far that
generalizes**: is the WCC route under-converged at practical meshes *in general*,
or was cesium a special case?

Two independent probes. **Answer up front: cesium was the special case.** Both real
2D systems tested here give the same $Z_2$ at every mesh tried — graphene at four
meshes spanning a 9× range in k-point count, bismuthene at four meshes spanning the
same range — with no wobble anywhere, and bismuthene's parity indicator independently
confirms its WCC answer. Nothing here supports a general convergence warning on the
2D WCC method. What the data does support is narrower and is stated in the conclusion
below.

## Probe 2 — graphene WCC mesh ladder

Monolayer graphene with `soc_scale={"C": 3000.0}` (the §20 fixture, reproduced
verbatim from `tests/test_calculation_z2.py`: `ngridk=(6,6,1)`, `rgkmax=7.0`,
`spinorb=True`), where $Z_2=1$ is trusted and the parity route already agrees
(`tests/test_calculation_parity.py::test_graphene_fu_kane_matches_wcc_z2`).
Band window `[1, ist1]` with `ist1 = 6`, taken from `EIGVAL.OUT`'s own occupation
numbers exactly as the fixture does.

| $(n_{kx}, n_t)$ | loop points $n_{kx}$ | pump points $2(n_t{-}1)$ | is $K$ on the mesh? | $z$ | wall time |
|---|---|---|---|---|---|
| (8, 5)   | 8  | 8  | no  | **1** | 354 s |
| (12, 7)  | 12 | 12 | yes | **1** | 984 s |
| (18, 9)  | 18 | 16 | no  | **1** | 1425 s |
| (24, 13) | 24 | 24 | yes | **1** | (not re-run here — this is the checked-in `test_z2_enhanced_soc_graphene_is_quantum_spin_hall` configuration, which asserts `z2 == 1`) |

**Graphene is rock-stable.** Four meshes spanning a 9× range in total k-point
count (64 → 576) all give $z=1$, including the two rungs where $K$ does *not*
land on the mesh. There is no wobble to explain.

This also rules out one tempting mechanical explanation of the cesium
oscillation. On cesium's disputed plane the pump mesh has
$2(n_t{-}1) = 12, 16, 24, 32$ points, i.e. $0, 1, 0, 2 \bmod 3$ — which matches
the observed $z = 1, 0, 1, 0$ exactly, while $n_{kx} \bmod 3$ ($0,0,0,2$) does
not. That correlation suggested a commensurability effect: the WCC answer
flipping according to whether the pump mesh samples the $k=1/3$ line. Graphene's
ladder has the same divisibility pattern (yes/yes for (12,7) and (24,13),
no/no for (8,5) and (18,9)) on a system whose topological physics lives *at*
$K = (1/3,1/3)$ — and shows no such flip. The commensurability hypothesis is
therefore not supported; whatever goes wrong on cesium is not simply "the
$1/3$ line is or isn't sampled".

### A fixture caveat found in passing (not fixed here)

`ist1 = 6` is not the occupied-manifold count for this fixture. Graphene has 8
valence electrons per cell and `nspinor = 2`, so 8 second-variational states
should be occupied — but at `soc_scale = 3000` the SCF solution is **metallic**:
the per-k-point occupied counts on the $6\times6\times1$ ground-state mesh are
6, 8, 8, 8, 8, 8, 9, summing correctly to 8 electrons on average. `ist1` is read
off the *first* k-point ($\Gamma$), where states 7–8 sit just above $E_F$, giving
6.

So the window actually tested — here, in `tests/test_calculation_z2.py` and in
`tests/test_calculation_parity.py` — is the lowest 6 bands, a legitimately and
very widely gapped group (gap to state 7 is 0.13–0.40 Ha $\approx$ 3.5–11 eV
across the ground-state mesh), not "the occupied valence manifold" as §20
describes it. The $Z_2$ of that group is perfectly well defined, and WCC and
parity agree on it, so the cross-validation stands. But two things follow:

1. §20's prose ("`ist0`/`ist1` spans every occupied valence band") is inaccurate
   for this fixture, and the connection to Kane & Mele's half-filled $\pi/\pi^*$
   prediction is looser than stated — at $3000\times$ SOC the band structure is
   heavily rearranged (the $\Gamma$ spectrum shows what should be a degenerate
   $\sigma$ pair split by $\sim10$ eV).
2. **Graphene at this window is an easy case for WCC.** A 3.5–11 eV gap gives
   smooth, well-separated WCC flow. Cesium's disputed plane had a minimum direct
   gap of 0.19 eV. So graphene's stability is evidence that the WCC
   implementation is sound and converges quickly *when the gap is large*; it is
   weaker evidence about the small-gap regime where cesium failed.

## Probe 1 — bismuthene parity cross-check

Freestanding buckled-honeycomb monolayer Bi (`BI_AVEC`/`BI_SPECIES` from
`tests/test_calculation_z2.py`; $a = 4.34$ Å, buckling 1.73 Å, 28 Bohr vacuum,
`spinorb=True`, no `soc_scale`), where WCC gives $Z_2 = 1$ and Murakami,
PRL 97, 236805 (2006) predicts a QSH insulator.

### 1a — is it centrosymmetric? **Yes.**

Elk's `tsyminv` is true and `session.parity()` returns a matrix rather than refusing.
At $\Gamma$: $\max|P-P^\dagger| = 5.0\times10^{-3}$, $\max|P^2-\mathbb 1| = 1.3\times10^{-2}$
— both at the `genolpq` real-space truncation floor already documented in §14, and
inside `parsers.symmetry`'s own `tol=5\times10^{-2}`. 14 of the 30 states carry
$\xi=-1$ (even, as Kramers requires).

The inversion centre is the midpoint of the Bi-Bi bond. Elk's `findsymcrys` shifts the
basis onto it, putting the two atoms at $\pm(1/3,1/6,0.0584)$ (`GEOMETRY.OUT`).

### 1b — Fu-Kane parity invariant: $\nu = 1$, agreeing with WCC

| window `ist0` | $\nu$ |
|---|---|
| 1 (full occupied manifold, `ist1` = 30) | **1** |
| 21 (above a 0.316 Ha $\approx$ 8.6 eV gap) | **1** |
| 25 (above a 0.250 Ha $\approx$ 6.8 eV gap) | **1** |

Cost: 13 s for the four TRIM, against 93–1154 s for one WCC mesh.

This is the **second independent 2D validation of the parity route** (graphene was
the first), and equally the second independent confirmation of the WCC $Z_2 = 1$ that
`tests/test_calculation_z2.py::test_z2_buckled_bismuthene_is_quantum_spin_hall`
asserts and Murakami, PRL 97, 236805 (2006) predicts. The two routes share no
arithmetic.

### A trap worth documenting: the individual $\delta_i$ are origin-dependent

The measured per-TRIM values look, at first sight, like a bug:

$$\delta(\Gamma)=-1,\quad \delta(0,\tfrac12)=-1,\quad \delta(\tfrac12,0)=+1,\quad \delta(\tfrac12,\tfrac12)=-1$$

The three $M$ points of a hexagonal cell are related by the 3-fold rotation, so a
naive reading says all three $\delta$ must be equal — and they are not. **They need
not be.** Elk chose the *bond-midpoint* inversion centre, which is displaced from the
3-fold axis by the half-lattice vector $t=(0,\tfrac12,0)$; `SYMCRYS.OUT` confirms this
directly, reporting both $C_3$ elements with non-lattice translations
($(\tfrac12,0,0)$ and $(\tfrac12,\tfrac12,0)$) — a symmorphic group written in a
non-standard origin.

Moving the inversion origin by $t$ replaces $\hat I$ by $T_{2t}\hat I$, and $2t$ here
*is* a lattice vector, so every $\xi$ at a TRIM $\mathbf k$ picks up
$e^{2\pi i\,\mathbf k\cdot 2t}=\pm1$ and

$$\delta(\mathbf k)\;\longrightarrow\;\delta(\mathbf k)\,\bigl(e^{2\pi i\,\mathbf k\cdot 2t}\bigr)^{N}$$

with $N$ the number of occupied Kramers pairs. Bismuthene has $N=15$ (odd), so the
phases survive. Applying them ($+1,-1,+1,-1$ at $\Gamma, (0,\tfrac12), (\tfrac12,0),
(\tfrac12,\tfrac12)$) transports the measured values to the on-axis origin:

$$\delta(\Gamma)=-1,\qquad \delta(M_1)=\delta(M_2)=\delta(M_3)=+1$$

— manifestly $C_3$-symmetric, and physically the right picture: the odd $\delta$ sits
at $\Gamma$ alone, i.e. **the band inversion is at $\Gamma$**, exactly the HgTe/CdTe-style
$s$-$p$ inversion `test_bi_gap_minimum_is_at_gamma_not_k` already measures for this
fixture (gap minimum ~0.6 eV at $\Gamma$, >2 eV at $K$). The product over all four TRIM
is invariant under the shift, so $\nu=1$ either way. This is the same origin/gauge
freedom §23 already noted when six of cesium's eight $\delta_i$ flipped together with
$\nu_0$ unchanged; here it shows up *within* one calculation, across TRIM.

The group theory was checked independently (per this project's standing rule of asking
Fable about crystal-symmetry questions) and confirmed, with one sharpening worth
keeping: of the four inversion centres of this structure — $(1/6,1/3)$, $(2/3,1/3)$,
$(1/6,5/6)$, $(2/3,5/6)$, all at $z=1/2$, obtained from $2c = \mathbf r_1+\mathbf r_2$
mod a lattice vector — **only the hexagon centre $(2/3,1/3)$ lies on the $C_3$ axis**;
the other three are the honeycomb's three bond midpoints, permuted cyclically by $C_3$.
Elk picked $(2/3,5/6)$. Conjugating an off-axis inversion by $C_3$ gives inversion
through a *different* centre, differing by a lattice translation, and that translation
supplies exactly the compensating sign — which is why $C_3$ does not force the three
$\delta(M)$ equal. The sharpening: the inequality **only appears when the occupied
Kramers-pair count $N$ is odd**. For even $N$ the phases are $(-1)^N = +1$ and the three
$\delta(M)$ come out equal even in the off-axis origin, so a passing $C_3$-uniformity
check is not evidence the origin is on-axis. Bismuthene has $N=15$. Origin-independence
of $\nu$ is exact for the same reason in general: the four 2D TRIM sum to an integer
vector, so the accumulated phase is $1$ whatever $t$ and $N$ are.

**Practical consequence:** do not assert $C_3$-uniformity of `deltas` in a test, and do
not read an individual $\delta_i$ as a physical statement without first pinning the
inversion origin. Only the products are meaningful. Neither operator is "wrong" — both
are genuine crystal symmetries and either is legitimate input to the Fu-Kane formula.

### 1c — bismuthene's own WCC mesh ladder

Bismuthene sits between the two extremes of probe 2: its gap ($\approx0.5$ eV,
minimum at $\Gamma$) is far smaller than graphene's tested 3.5-11 eV window and
of the same order as cesium's disputed plane (0.19 eV).

| $(n_{kx}, n_t)$ | $z$ | wall time |
|---|---|---|
| (8, 5)   | **1** | 93 s |
| (12, 7)  | **1** | 196 s |
| (18, 9)  | **1** | 381 s |
| (24, 13) | **1** | 1154 s |

**Bismuthene is rock-stable too** — and this is the more informative of the two
ladders, because it is a real, unenhanced material with a ~0.5 eV gap, the same order
as cesium's 0.19 eV, rather than graphene's enormous artificially-widened window.


A denser rung, $(32,17)$, was started and killed at ~37 min before completing; the
four rungs above are the ones that landed. A fresh $(24,13)$ graphene rung was also
queued behind it and not reached, which is why probe 2's last row cites the
checked-in test rather than a run made here.

## Conclusion

**The data supports: cesium was a special case; the 2D WCC method is not
under-converged at practical meshes in general.**

| | rungs tried | result |
|---|---|---|
| graphene, `soc_scale=3000` (probe 2) | (8,5), (12,7), (18,9), (24,13) | $z = 1, 1, 1, 1$ |
| bismuthene (probe 1c) | (8,5), (12,7), (18,9), (24,13) | $z = 1, 1, 1, 1$ |
| cesium, $k_3=0$ plane (§23) | (12,7), (18,9), (24,13), (32,17) | $z = 1, 0, 1, 0$ |

and, independently of any mesh:

| | parity $\nu$ | WCC $Z_2$ | agree? |
|---|---|---|---|
| graphene (§23) | 1 | 1 | yes |
| **bismuthene (new here)** | **1** | **1** | **yes** |
| cesium (§23) | 0 | oscillates | n/a |

So §23's retraction is confirmed as *local to cesium*, not a symptom of a broadly
unreliable method, and the two 2D validations of the parity route (graphene,
bismuthene) now both stand.

### What this does *not* establish

Three honest limits on the claim.

1. **Both stable systems are 2D and both have a comfortable gap.** Graphene's tested
   window has a 3.5-11 eV gap; bismuthene's is ~0.5 eV. Cesium's disputed plane had
   0.19 eV. Bismuthene narrows that distance a lot — it is a real material with no
   SOC enhancement, and its gap is within a factor of ~3 of cesium's — but it does not
   reach it, and neither probe tests a 3D six-plane sweep, which is what §23 actually
   impugned.
2. **The commensurability hypothesis is ruled out, but no positive mechanism is
   established.** Cesium's flips track $2(n_t{-}1) \bmod 3$ exactly ($0,1,0,2$ vs
   $z = 1,0,1,0$) while $n_{kx} \bmod 3$ does not; graphene's ladder has the same
   divisibility pattern and does not flip, on a system whose topology lives at
   $k = 1/3$. So "the $1/3$ line is or isn't sampled" is not the explanation. What is
   left is the ordinary suspect — the largest-gap crossing count miscounting when the
   WCC flow is under-resolved between pumping steps — but that was not demonstrated
   here, and would need the saved `wannier_centers` from cesium's own rungs to check.
3. **§21's Bi$_2$Se$_3$ open item is untouched.** §23 named WCC under-convergence as
   the leading explanation for its $\nu_0 = 0$ against a literature $(1;000)$. The
   present result makes that explanation *less* attractive, not more: two systems now
   converge immediately. Since Bi$_2$Se$_3$ is centrosymmetric, the parity route
   settles it in minutes once its structure is a checked-in fixture — still the
   cheapest next step, and now more clearly worth doing.

### Recommendation for the docs

Do **not** add a general "the WCC method may be under-converged" warning to §20.
`get_z2_invariant()`'s docstring already carries the right advice (start modest,
increase if `wannier_centers` looks jagged, watch for `_unitarize` raising) and the
mesh-aliasing warning §20 hit on graphene at `soc_scale=100`. What is worth adding is
narrower: **for the 3D six-plane sweep specifically (§21), report a per-plane $z$ at
two different meshes and treat disagreement as "not converged" rather than picking
one** — cesium's failure was visible at zero extra insight the moment a second mesh
was tried.

### Two things found in passing, neither fixed here

* The `soc_scale=3000` graphene fixture is **metallic** and its `ist1 = 6` is not the
  occupied-manifold count (see probe 2's caveat above). §20's prose describing the
  window as "every occupied valence band" is inaccurate for it. The tested band group
  is well-gapped and the WCC-vs-parity comparison is unaffected.
* The per-TRIM $\delta_i$ depend on which inversion centre the code picked, and on a
  buckled honeycomb Elk picks an off-axis one, so $C_3$-related TRIM can carry
  different $\delta$ (see probe 1's trap above). Worth a sentence in §23, which
  currently mentions only the global all-$\delta$-flip version of this freedom.

### Where the raw evidence lives

Run logs, per-rung `wannier_centers` arrays (`*_wcc_*.npy`) and copies of bismuthene's
`SYMCRYS.OUT`/`GEOMETRY.OUT` are in `.wcc_audit_runs/` (untracked, ~100 KB after the
Elk run directories were deleted). Every number quoted above is reproducible from
`.wcc_audit_runs/audit.py` and `probe1.py`.

### Tests added

`tests/test_calculation_parity.py::test_bismuthene_fu_kane_matches_wcc_z2` and
`::test_bismuthene_parity_is_window_independent` — the probe-1 result, which is exact
(no mesh) and cheap (~88 s for both, including the ground state). Both pass against a
real binary. The mesh ladders are deliberately **not** added as tests: 8 WCC runs
costing ~40 minutes buys a regression pin on a number no code change is likely to move.
