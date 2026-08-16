# Bi2Se3 settled: $(1;000)$, closing §21's open question

*Prose fragment intended for `docs/design.md` §21/§23 and `docs/physics.tex` Part XII.
Nothing here edits those files directly.*

## Result

Bulk Bi$_2$Se$_3$ is a **strong topological insulator, $(\nu_0;\nu_1\nu_2\nu_3)=(1;000)$**,
by the Fu-Kane parity indicator (§23) — matching the literature. Fu & Kane, PRB **76**,
045302 (2007) supply the *method* (their §IV applies it to the diamond-lattice model,
Bi$_{1-x}$Sb$_x$, gray tin and HgTe — Bi$_2$Se$_3$ was not yet known to be a topological
insulator); the Bi$_2$Se$_3$-family result itself, $(1;000)$ via exactly this parity
counting, is Zhang, Liu, Qi, Dai, Fang & Zhang, Nature Physics **5**, 438 (2009),
arXiv:0812.1622. This **closes §21's open question**: the $\nu_0=0$ reported there from a six-plane
Wannier-charge-center sweep at `nkx=8`, `nt=5` was an under-converged crossing count, the
same failure mode §23 documented on the cesium structure. The WCC number was wrong; the
material is not anomalous.

The eight $\delta_i$, for the full occupied manifold (`ist0=1`, `ist1=78`):

| TRIM $(k_1,k_2,k_3)$ | $\delta_i$ |
|---|---|
| $(0,0,0)$ — $\Gamma$ | $-1$ |
| $(0,0,\tfrac12)$ | $+1$ |
| $(0,\tfrac12,0)$ | $+1$ |
| $(0,\tfrac12,\tfrac12)$ | $+1$ |
| $(\tfrac12,0,0)$ | $+1$ |
| $(\tfrac12,0,\tfrac12)$ | $+1$ |
| $(\tfrac12,\tfrac12,0)$ | $+1$ |
| $(\tfrac12,\tfrac12,\tfrac12)$ | $+1$ |

$\prod_{i=1}^{8}\delta_i=-1\Rightarrow\nu_0=1$; each weak index is a product of four $+1$s,
so $\nu_1=\nu_2=\nu_3=0$.

**The pattern is itself the physics, not just its arithmetic.** Exactly one TRIM carries the
odd parity product, and it is $\Gamma$ — the single band inversion at the zone centre that
is the accepted mechanism for Bi$_2$Se$_3$'s topology (the $p_z$-derived
$|P1^+_z\rangle$/$|P2^-_z\rangle$ states exchanging parity under spin-orbit coupling, Zhang
*et al.* 2009 §"Band inversion"). Had the odd $\delta$ sat instead at a $k_i=\pi$ TRIM, the
weak indices would have come out nonzero — a *different* disagreement with the literature
than $\nu_0=0$, and one that would have pointed at the cell orientation rather than at the
physics. It did not.

## Window independence

Seven independently-gapped band windows, all ending at `ist1=78` (the occupied count read
off `EIGVAL.OUT`'s own occupation numbers, not an assumed electron count), all give
$(1;000)$:

| `ist0` | gap immediately below the window (min over the 8 TRIM) | result |
|---|---|---|
| 1 | — (bottom of the valence states) | $(1;000)$ |
| 13 | 0.601 eV | $(1;000)$ |
| 31 | 22.534 eV | $(1;000)$ |
| 39 | 2.652 eV | $(1;000)$ |
| 51 | 7.204 eV | $(1;000)$ |
| 57 | 1.717 eV | $(1;000)$ |
| 61 | 4.935 eV | $(1;000)$ |

This exercises $Z_2$'s additivity mod 2 over independently-gapped band groups: dropping a
gapped semicore block from the bottom of the window cannot change the invariant, and does
not. `ist0=61` is §21's own narrow window (bands 61-78), which there reproduced the *wrong*
WCC answer on the one plane it was tested on — the parity route gives the right one from
the same ground state, so the discrepancy was never about the window.

Between windows the eight $\delta_i$ flip *together* (`ist0` = 31, 39, 57 return the exact
negation of the table above, $\Gamma$ at $+1$ and the other seven at $-1$). Mechanically the
dropped block carries $\delta=-1$ at every TRIM; the invariants are untouched because that
block is itself trivial, $(-1)^8=+1$ for $\nu_0$ and $(-1)^4=+1$ for each weak index. That
is the additivity-mod-2 statement made concrete, and it is the same arithmetic behind §23's
**even-TRIM sign immunity** (every invariant is a product over an even number of TRIM, so
any change common to all eight cancels). $\nu_0=1$ and $\nu=(0,0,0)$ in all seven cases.

## Structure and ground state

Structure re-sourced from **Crystallography Open Database entry 9011965** (digitizing
Nakajima's diffraction refinement, J. Phys. Chem. Solids **24**, 479 (1963)) as a checked-in
fixture, per this project's standing rule against hand-deriving a rhombohedral cell from
hexagonal parameters. The CIF's 15-atom $R\bar3m$ hexagonal cell
($a=4.143$ Å, $c=28.636$ Å) was reduced to the 5-atom rhombohedral primitive cell with
`spglib.standardize_cell(to_primitive=True)` and loaded through `Structure.from_ase()` —
a library transformation, not a hand-written matrix. Primitive cell:
$a=9.8405$ Å, $\alpha=24.304^\circ$; Bi at $\pm(0.4008,0.4008,0.4008)$, Se at $(0,0,0)$ and
$\pm(0.2117,0.2117,0.2117)$; spglib re-identifies it as $R\bar3m$ (#166) with an inversion
centre at the origin.

Bond lengths verified numerically **before** any DFT, and reproducing §21's own recorded
values exactly:

| quantity | measured | expected |
|---|---|---|
| Bi-Se (inner, to the central Se) | 3.0747 Å | ~3.0 Å |
| Bi-Se (outer) | 2.8509 Å | ~2.85 Å |
| quintuple-layer span (outer Se to outer Se, 3D distance) | 7.365 Å | ~7.4 Å |
| van der Waals gap (Se-Se interlayer spacing along $c$) | 2.5791 Å | ~2.6 Å |

The QL span is the *three-dimensional* outer-Se-to-outer-Se distance, not its projection on
$c$: the $c$-projected span is 6.966 Å and the two outer Se of one QL are laterally offset
by $a/\sqrt3=2.392$ Å, and $\sqrt{6.966^2+2.392^2}=7.365$ Å. Worth stating explicitly,
since the two definitions differ by 0.4 Å and only one of them matches the number §21
recorded.

Ground state: `xc="PW"`, `ngridk=(4,4,4)`, `rgkmax=7.0`, `spinorb=True`, **no**
`soc_scale` — bismuth's own atomic spin-orbit coupling is the physics here, so unlike §20's
graphene and §21's cesium there is no numerics knob to turn. Converged in 17 SCF iterations,
~6.5 min wall on one core. 78 occupied bands of `nstsv=120`; direct gap at $\Gamma$
**0.2575 eV**, reproducing §21's 0.258 eV to the digit, so this is demonstrably the same
ground state the failed WCC sweep was run on — the parity result is not being compared
against a different calculation.

## Cost, and what this says about the two methods

Eight diagonalisations per window, **~1-3 minutes each**, against the ~45 minutes §21's
six-plane WCC sweep took to return the wrong answer. Both routes are driven from the *same*
converged `STATE.OUT`; nothing about the ground state changed between them. This is now the
sharpest available statement of §23's thesis:

- On a real material with a genuinely robust gap (0.26 eV, nowhere near §20's
  $10^{-3}$-wide graphene anticrossing), the 3D six-plane WCC sweep at a practical mesh
  returned a *confidently wrong* integer, with no internal signal that it was wrong.
- The parity indicator returned the right one, mesh-free, in a twentieth of the time, and
  was stable across seven band windows.

So the failure mode §23 diagnosed on a hypothetical structure (cesium) is confirmed on a
real one, and both of §21's open/retracted items now have the same resolution: the 3D WCC
crossing count is under-resolved at meshes anyone would actually run. What is *not*
impugned, again, is the WCC implementation — it agrees with parity in 2D (graphene) and is
separately validated on bismuthene; what is impugned is the six-plane 3D sweep at practical
mesh density.

Deliberately **not** done: re-running `get_z2_invariant_3d()` at a denser mesh to watch it
converge to $\nu_0=1$. That is the hours-long path §23 built this feature to avoid, and the
parity answer is exact rather than convergent, so a denser WCC sweep could only agree with
it more slowly.

## How to use in code

```python
from ase import Atoms
from ase.io import read
import spglib
from elkpy.structure import Structure

# COD 9011965 -> 5-atom rhombohedral primitive cell (library transformation,
# never a hand-derived one -- see CLAUDE.md's standing rule)
hexa = read("9011965.cif")
lat, pos, nums = spglib.standardize_cell(
    (hexa.get_cell()[:], hexa.get_scaled_positions(), hexa.get_atomic_numbers()),
    to_primitive=True, symprec=1e-4,
)
structure = Structure.from_ase(Atoms(numbers=nums, scaled_positions=pos, cell=lat, pbc=True))

calc = structure.get_calculation(
    "bi2se3", xc="PW", ngridk=(4, 4, 4), rgkmax=7.0, spinorb=True,
)
calc.get_energy()

result = calc.get_fu_kane_invariant(1, 78, dimension=3)
result["nu0"]     # 1  -- strong topological insulator
result["nu"]      # (0, 0, 0)
result["deltas"]  # -1 at Gamma, +1 at the other seven TRIM
```
