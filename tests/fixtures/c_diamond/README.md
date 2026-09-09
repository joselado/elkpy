# `c_diamond` — diamond, two carbon atoms

The companion to `h_sc`. Hydrogen has no core, so nothing in `h_sc` can test
what happens when `rhomt` holds a **frozen core**; carbon's 1s carries two of
its six electrons, so this one does. Two atoms of one species also exercise
the **atom-inner** half of `STATE.OUT`'s `ias` ordering, which one atom per
cell cannot. (The species-outer half is `sic_zb`.)

The cell is verbatim the diamond cell in the sibling project's own Quantum
ESPRESSO test input, so the two are comparable exactly rather than
approximately.

| | |
|---|---|
| Structure | fcc, $a = 6.74$ Bohr = 3.56665 Å, C at $(0,0,0)$ and $(\tfrac14,\tfrac14,\tfrac14)$ |
| Checked before running | C-C = 2.91851 Bohr = 1.54441 Å, four neighbours, against a literature 1.5445 Å |
| Functional | **PBE** (`xctype 20`), set explicitly, not the default LSDA |
| Spin | non-spin-polarised (`ndmag = 0`, no magnetic records) |
| k-mesh | $4\times4\times4$, 8 irreducible points |
| Everything else | Elk defaults — `gmaxvr`, `rgkmax`, `swidth`, `autormt` as it falls |
| `tshift` | **`.false.`**, and here it is *not* a no-op (see below) |
| Tasks | 0 (ground state) then 33 (3D density) in one run; **9006** (initial state) in a second |
| `plot3d` | the unit cell, `np3d = 16 16 16` |
| Cost | 0.4 s for `STATE_INIT.OUT`, 4 s for the rest, one core |

Regenerate with `./regenerate.sh` (needs `./build_elk.sh` at the repo root
first, **with the elkpy patch series** — task 9006 is `patches/0024`).

## Numbers this run produces

Converged in 14 loops (`INFO.OUT` carries `Convergence targets achieved`).

| | |
|---|---|
| $r_{\rm MT}$ | 1.434252805 Bohr, $= r_{\rm sp}(n_{r{\rm MT}})$ in `STATE.OUT` |
| `nrmt` / `nrmti` | 297 / 201 |
| `lmmaxo` / `lmmaxi` | 49 / 4 (`lmaxo = 6`, `lmaxi = 1`) |
| `ngridg` / `ngtot` | 20 20 20 / 8000 |
| `ngvec` | 2229 |
| `xcgrad` | **1** — GGA; `h_sc` has 0 |
| $E_F$ | 0.45408306917170 Ha |
| core / valence charge | 4.0 / 8.0 (2 core electrons per atom) |
| muffin-tin charge, per atom | 4.397030726, both atoms |
| core leakage, per atom | 8.878318689e-5 |
| interstitial charge | 3.205938547 |
| total calculated / error | 11.99536778 / 4.632e-3 (**pre-`rhonorm`** — see `h_sc`'s README) |
| total energy | −76.193493799303 Ha |
| indirect gap | 0.16394 Ha = 4.46 eV — the $4\times4\times4$ mesh does not sample the $\Delta$-line conduction minimum, so this is an upper bound, not a converged PBE gap |

$\rho$ at either nucleus: **129.8314269** e/Bohr³, against hydrogen's 0.286.
That factor of 450 is the 1s, and it is the answer to "does `rhomt` include the
frozen core" — it does; 2 of carbon's 6 electrons are core, 45% of `chgmt`.
(`STATE.OUT` holds only the sum. The core density on its own is patch 0021's
`rhocr` export, not this file.)

Reintegrating the $l=0$ channel recovers `chgmt` to 2.3e-5, which is Simpson
against Elk's spline weights and nothing else (`h_sc`'s residual is 9e-6).

## Why `tshift = .false.` matters here, measured

In `h_sc` it was a no-op. Diamond's inversion centre is the bond midpoint at
$(\tfrac18,\tfrac18,\tfrac18)$, so Elk's default moves the origin there. Run
with the default and the same input gives:

```
  1 :   0.37500000  0.37500000  0.37500000
  2 :  -0.37500000 -0.37500000 -0.37500000
Crystal has inversion symmetry
Real symmetric eigensolver will be used
```

Not $(0,0,0)$ and $(\tfrac14,\tfrac14,\tfrac14)$. `GEOMETRY.OUT` is written
after the shift and `elk.in`'s `atoms` block is not, so a reader that took the
positions from `elk.in` would put the muffin tins in the wrong place entirely.
The price of pinning the frame is that Elk no longer sees the inversion and
uses the complex Hermitian eigensolver — a few seconds, and worth it.

## `STATE_INIT.OUT` — and what it is *not*

Task 9006 (`patches/0024-initial-state.patch`) runs `gndstate`'s own
`trdstate = .false.` initialisation and the top of its first iteration, then
stops:

```
init0; init1; rhoinit; maginit; potks(.true.); genvsig
gencore; linengy; genapwlofr; gensocfr; genevfsv; occupy; writestate
```

So `STATE_INIT.OUT` holds **`rhoinit`'s superposition of free atomic
densities**, not a density after one SCF iteration — `rhomag` never runs. Two
consequences worth having written down:

- **It is the one internally consistent pair Elk writes.** `mixerifc` is never
  called, so its `vsmt`/`vsir` is `potks` of exactly the `rhomt` in the same
  file. The converged `STATE.OUT` is not: its density comes from `rhomag`
  (`gndstate.f90:189`), its potential is mixed at `:211`, and the file is
  written at `:338`. Those two are one mixing step apart, ≤ `epspot` at
  convergence but not zero.
- **The core does not cancel in $\rho_{\rm SCF}-\rho_{\rm init}$.** The initial
  file's core is `rhosp`'s free-atom core; the converged file's is `gencore`'s
  core in the crystal potential. Measured here: they differ by 1.67 e/Bohr³ out
  of 460 at the innermost mesh points (0.36%), and by 1.9e-3 electrons inside
  $r < 0.2$ Bohr. The difference is valence change **plus** core relaxation.

| | `STATE.OUT` | `STATE_INIT.OUT` |
|---|---|---|
| $E_F$ | 0.45408306917170 | 0.31921436078665 |
| $\rho(0)$ per atom | 129.8314268824 | 130.1870636725 |
| reintegrated `chgmt` | 4.39705347 | 4.38207053 |

Elk prints `Warning(linengy): could not find 1 linearisation energies` in every
loop, including this one. It is the second radial function of carbon's third
local orbital, the only one with `lorbve = T`; `findband` cannot bracket it and
the energy stays at the species file's own −0.5012 Ha. Benign, and it is
exactly the case patch 0024's exported `lorbve` flags exist to distinguish: a
loop run outside Elk that freezes this energy is **exact**, not approximate.

## What to check against what

Same as `h_sc`, plus two things only two atoms can give:

- **Both nuclei.** `rfpts` clamps $r$ up to $r_{\rm sp}(1)$, so `RHO3D.OUT` at
  each site is exactly $\rho_{00}(r_1)y_{00}$ from that atom's `ias` column.
  With `np3d = 16` the second atom at $\tfrac14$ is grid index $(4,4,4)$ —
  `plotpt3d.f90` steps $i/n$ without an endpoint.
- **The `ias` ordering.** The two carbons sit at the two ends of a bond whose
  midpoint is an inversion centre, so their $\rho_{lm}$ differ by $(-1)^l$:
  $l = 0, 4, 6$ identical to 1e-14, $l = 3$ exactly opposite, $l = 1, 2, 5$
  zero (Td site symmetry). The $l = 0$ check therefore **cannot** see an `ias`
  swap on this fixture — only the odd-$l$ channels can. `sic_zb` settles it
  with no symmetry argument at all.

The layout of `STATE.OUT` itself, and the conventions inside it, are
`docs/design.md` §34.
