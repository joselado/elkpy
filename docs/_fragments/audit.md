# Post-correction staleness audit

Scope: hunt for text left stale by (1) the Berry-curvature sign-convention fix
(design.md §22 — `parsers.berry._berry_phase()` now applies the King-Smith–Vanderbilt/
Resta negation, so elkpy reports the standard Xiao-Chang-Niu $\mathbf A=i\langle
u|\nabla_{\mathbf k}u\rangle$, $\Omega=\nabla\times\mathbf A$) and (2) the cesium
retraction (design.md §23 — the [111]-dimerized diamond structure is trivial, $(0;000)$;
the WCC *implementation* is not impugned).

Rule applied throughout: statements that **record history** ("originally failed with a
factor of $-1$", "gave $\nu_0=1$ — retracted") are correct and were left alone. Only
**present-tense claims about current behaviour** that are now false are reported.

`design.md`, `physics.tex`, `CLAUDE.md`, `README.md` are report-only (being edited
concurrently). Everything else was fixable and the fixes applied are listed at the end.

Note: `src/elkpy/calculation.py`, `optical.py`, `session.py`, `spec.py`,
`tests/test_parsers_optical.py` are being actively extended in the working tree right
now (an unrelated magneto-optical Kerr / k·p feature, `src/elkpy/parsers/moke.py`), so
line numbers in those files drifted by ~180 lines during this audit. Line refs below for
those files are as-of-now; the quoted text is the reliable anchor.

---

## 1. Genuine contradictions (present-tense claims that are now false)

### 1.1 `docs/physics.tex:2536` — asserts the Kubo and Wilson-loop routes are OPPOSITE in sign

This sits inside Part XI's **"Verification against a real compiled binary"** list — a
list of what `tests/test_calculation_momentum.py` checks *now*. The test now asserts
agreement (`pytest.approx(wilson_point["curvature"], rel=0.05)`), and `docs/design.md`'s
matching bullet already says "agreeing to 5% in **both magnitude and sign**". This line
was not updated.

Current text:

```
the Kubo curvature against Part~\ref{part:berry}'s Wilson loop at
$K/K'$ (measured $\mp8.194$ against $\pm8.101$ Bohr$^2$: 1.2\% in magnitude, opposite in
sign per the previous section); stability of the Kubo sum between \texttt{nempty}$=12$ and
```

Suggested replacement:

```
the Kubo curvature against Part~\ref{part:berry}'s Wilson loop at
$K/K'$, agreeing in sign and to 1.2\% in magnitude (this comparison originally came out
$\mp8.194$ against $\pm8.101$ Bohr$^2$ --- opposite in sign, which is how the previous
section's missing negation was found); stability of the Kubo sum between
\texttt{nempty}$=12$ and
```

This is the highest-priority item: it is the one place in the docs that still states the
**old** Kubo-vs-Wilson relationship as a live fact.

### 1.2 `README.md:122-123` — `nu0_by_axis` comment internally contradicts the `nu0` comment

```python
result = cs.get_z2_invariant_3d(1, ist1, nkx=12, nt=7)
result["nu0"]         # strong index (this structure: 0 -- see docs/design.md #23,
                      # which retracts an earlier unconverged nu0=1 for it)
result["nu0_by_axis"] # (1, 1, 1): the strong index agrees identically across all three axes
```

Two problems: (a) `nu0` and `nu0_by_axis` cannot be `0` and `(1,1,1)` from the same call
— `nu0_by_axis` is by construction three copies of `nu0`; (b) the `nu0` comment claims
the call *returns* 0, which it does not — at `nkx=12, nt=7` it returns 1 (confirmed by
`notebooks/13_z2_invariant_3d.ipynb`'s checked-in output `(1, (1, 1, 1), (0, 0, 0))`).
The retraction is about the physics, not about what the code prints.

Suggested replacement:

```python
result = cs.get_z2_invariant_3d(1, ist1, nkx=12, nt=7)
result["nu0"]         # 1 at this mesh -- NOT converged; the exact parity indicator gives
                      # 0 for this structure, see docs/design.md #23
result["nu0_by_axis"] # (1, 1, 1): whatever nu0 is, it agrees identically across all
                      # three axis splits -- the algebraic check, which held
```

### 1.3 `docs/design.md:1369` and `docs/physics.tex:2062` — "The verified example" headings on a retracted result

`design.md:1369`:
```
**The verified example: cesium on a dimerized diamond lattice — Fu & Kane's own toy
model, not a proxy for it.**
```
Suggested: `**The retracted example: cesium on a dimerized diamond lattice — Fu & Kane's
own toy model, not a proxy for it (the $\nu_0$ reported here is retracted; see §23).**`

`physics.tex:2062`:
```latex
\section{The verified example: cesium on a dimerized diamond lattice}
```
Suggested: `\section{The cesium dimerized-diamond example, and its retracted $\nu_0$}`

### 1.4 `docs/design.md:1371` and `docs/physics.tex:2065` — "this feature is verified directly against"

Both read:
```
... hoping its published Z2 classification carries over, this feature is verified
directly against the minimal lattice model Fu & Kane use to *introduce* ...
```
Present tense, and no longer true for the value of $\nu_0$ (only the axis-split algebra
survives). Suggested for both: "... this feature was **originally** verified against the
minimal lattice model Fu & Kane use to *introduce* ... — a verification that has since
been retracted for $\nu_0$'s value (§23 / Part~\ref{part:parity}), leaving only the
algebraic axis-split check."

### 1.5 `docs/physics.tex:2092` — `\subsection{Verified}` heading over a retraction

The body immediately below correctly says "**That value is retracted**". Only the heading
is stale. Suggested: `\subsection{Verified --- and, for $\nu_0$, retracted}`

### 1.6 `docs/design.md:1444` and `docs/physics.tex:2144` — "the verified cesium model above"

Both, inside the $\alpha$-Sn dead-end discussion:
```
[111] distortion (the verified cesium model above) sidesteps the question entirely ...
```
Suggested for both: "[111] distortion (the cesium model above, whose $\nu_0$ was later
retracted but whose symmetry analysis stands) sidesteps the question entirely ...". The
symmetry point being made ($R\bar3m$ symmorphic, no zone-boundary sticking) is unaffected
by the retraction — only the word "verified" is wrong.

### 1.7 `README.md:126` — image caption asserts the retracted crossing pattern

```
![Alt text](images/cs_dimerized_z2_invariant_3d.png?raw=true "Wannier charge centers on
the k1=0 and k1=pi planes of a dimerized diamond lattice, showing an odd vs. even number
of crossings")
```
The "odd vs. even" split *is* what the code produced at `nkx=12` — but the odd count on
the $k_1=0$ plane is exactly the number §23 shows oscillates ($z=1,0,1,0$) with mesh, so
presenting it as the headline result contradicts the retraction. `images/cs_dimerized_
z2_invariant_3d.png` is dated 14 Aug, predating the parity commit `ff856b5`, i.e. it was
never regenerated.

Two options: (a) keep the figure and change the caption to "... showing the WCC
trajectories on two planes; the $k_1=0$ plane's crossing count is mesh-dependent and its
$\nu_0=1$ is retracted (docs/design.md #23)"; or (b) drop this EXAMPLES entry and promote
the parity notebook's figure instead. Option (a) is cheaper and keeps the section's real
value (it demonstrates the method). Regenerating the PNG requires an Elk run, which was
out of scope for this audit.

---

## 2. Cross-references

### 2.1 BROKEN — `tests/test_calculation_atom_projection.py:166-167` (FIXED, see §4)

Pointed at "the K/K' Berry-curvature-antisymmetry check in `test_calculation_berry.py`".
`tests/test_calculation_berry.py` runs on **bulk Si only** — it has no h-BN fixture and
no K/K' curvature test (its 6 tests are: Si Chern number, gap-check rejection, `dk`
convergence, path-vs-mesh agreement, and two `kpath=` plumbing tests). The h-BN K/K'
curvature antisymmetry is actually asserted in
`tests/test_calculation_quantum_geometry.py::test_hbn_gkm_valley_quantum_geometry` and
`tests/test_calculation_momentum.py::test_kubo_curvature_matches_wilson_loop`.

### 2.2 Weak — `docs/design.md:485-490` claims an h-BN result with no test named

§13's path-mode verification paragraph says the h-BN K/K' antisymmetry was "**exercised
on** monolayer h-BN ($\Omega(K)=-8.568$, $\Omega(K')=+8.568$ Bohr$^2$)" but names no test
file, and — per §2.1 — no test in `test_calculation_berry.py` covers it. This is the one
"verified" claim in the audit (task 4) that does not resolve to a test. It is not false
(the check exists, in two other files), just unattributed.

Suggested: append the attribution
`` (`tests/test_calculation_quantum_geometry.py::test_hbn_gkm_valley_quantum_geometry`
asserts this antisymmetry via the independent `EigenstateSession.overlap()` path;
`tests/test_calculation_momentum.py::test_kubo_curvature_matches_wilson_loop` pins the
same two points against the Kubo route) ``

Same applies to the mirrored passage at `docs/physics.tex:503-506`.

### 2.3 Number inconsistency worth a footnote — $8.568$ vs $8.101$

`design.md:488` / `physics.tex:504` / `CLAUDE.md:69` quote $|\Omega(K)|=8.568$ Bohr².
`design.md:1630` / `physics.tex:2536` / `notebooks/05_berry_curvature.ipynb` (executed
output) all give $8.101$. Both are `get_berry_curvature_path()` on h-BN's occupied
manifold at $K$; the difference is presumably `dk`/`rgkmax`/`nempty`, but nothing says so,
and after a sign-convention correction two differing magnitudes for the same quantity in
the same doc invite a false "did the fix change the magnitude?" reading. Suggest a
parenthetical on the $8.568$ figure naming the `dk` it was measured at.

### 2.4 All other `docs/design.md #N` pointers resolve correctly

Mechanically checked every `docs/design.md #N` / `§N` pointer in `src/`, `tests/`,
`notebooks/`, `docs/` against design.md's section headers. All 40+ resolve to the section
that actually discusses the topic. Spot-checks worth recording:
- `parsers/wilson.py:82` "#22" → §22 (the sign convention) ✓ — and its note that the WCC
  angles deliberately bypass `_berry_phase()`, and that a crossing **parity** is invariant
  under $\theta\to-\theta$, is correct and consistent with §22's "Scope of the fix".
- `parsers/quantum_geometry.py:36` "#14" → §14 (eigenstate session / genolpq truncation
  floor) ✓; `:127` "#15" ✓; `:138` "#22" ✓.
- `parsers/symmetry.py` "#20/#21", "#23", "#14" ✓. `parsers/spin.py` "#14"/"#17" ✓.
- `config.py`/`spec.py` "#8", `spec.py` "#7", `structure.py` "#9", `launcher.py` "#6" ✓.
- `docs/design.md §23` correctly names `patches/0008-inversion-parity-operator.patch`,
  which exists.

Roman-numeral `docs/physics.tex Part N` pointers were checked the same way, against the
`\part{}` order (§12→I, §13→II, §14→III, §15→IV, §16→V, §17→VI, §18→VII, §19→VIII,
§20→IX, §21→X, §22→XI, §23→XII). All 35 hits in `src/`, `tests/`, `docs/design.md` and
`docs/roadmap.md` resolve correctly — including the five in `session.py` (Parts V, VII,
VIII, XI, XII, one per query type) and the four in `calculation.py` (Parts II, IV, IX, X).
No mismatches.

### 2.5 MISSING — `docs/design.md` §22 and §23 carry no `physics.tex` Part pointer

Every section from §12 to §21 ends its opening with an explicit handoff ("Full physics
writeup ...: `docs/physics.tex` Part N"). §22 and §23 — the two newest sections, and the
two that carry both corrections — have none: `awk` over lines 1488-1900 of design.md finds
zero occurrences of `physics.tex`. `CLAUDE.md` and `physics.tex` both agree the targets
are Part XI and Part XII respectively, so the parts exist; only design.md's pointer is
absent.

Suggested: add to §22's opening paragraph "Full physics writeup (the velocity-operator
identity, the $C_3$ selection rule, the Kubo derivation, the sign analysis):
`docs/physics.tex` Part XI." and to §23's "Full physics writeup (the FKM formulas, the
Kramers-pairing derivation, the even-TRIM sign immunity, why $P^2=\mathbb 1$ survives
truncation, the retraction): `docs/physics.tex` Part XII."

### 2.6 `docs/physics.tex` compiles clean

`pdflatex -interaction=nonstopmode physics.tex` run twice in `docs/`: 49 pages, **zero**
undefined references, zero multiply-defined labels, zero LaTeX warnings. Every `\ref` /
`\eqref` target resolves. Build artifacts (`physics.aux`, `.log`, `.out`, `.pdf`, `.toc`)
were deleted afterwards; `docs/` verified clean.

### 2.7 `docs/design.md` does not name `tests/test_parsers_symmetry.py`

§23 describes `parsers.symmetry.check_window_gap()`, `parity_eigenvalues()`,
`trim_delta()` and the global-sign-immunity argument, all of which have synthetic pins in
`tests/test_parsers_symmetry.py` (17 tests, including
`test_global_sign_flip_cannot_change_any_invariant` and
`test_check_window_gap_rejects_a_cut_band_group`). Every other section names its synthetic
unit-test file; §23 does not. Cosmetic gap, not an error.

---

## 3. Non-findings (checked, correct as-is — recorded so they are not re-flagged)

- **`src/elkpy/parsers/berry.py:294` "area is Bohr^-2"** is CORRECT. Full text: `"curvature":
  flux / loop area (Bohr^2 -- flux is dimensionless and the area is Bohr^-2, since bvec is
  Cartesian reciprocal Bohr^-1)`. The curvature is labelled Bohr², the *area* Bohr⁻² —
  exactly the relation the correction established. Do not "fix" this.
- **No stale "Bohr⁻²" curvature label survives anywhere.** Swept `docs/`, `src/`, `tests/`,
  `notebooks/` (including ipynb JSON and TeX `^{-2}` spellings) — the only remaining hits
  are berry.py:294 above and CLAUDE.md's own record of having made the fix. Notebook 05
  and 07 axis labels both read `Bohr$^2$` ✓; `parsers/optical.py:213` reads "Bohr^2" ✓.
- **`notebooks/05_berry_curvature.ipynb` and `07_quantum_geometry.ipynb` executed outputs
  are POST-fix.** 05 prints `K: -8.100931...`, `Kprime: 8.100931...`; 07 prints
  `Omega(K), Omega(K'): -8.086..., 8.081...`. §22 records the pre-fix path value as
  $+8.101$ at $K$, so the checked-in sign is the corrected one. `images/hbn_berry_
  curvature.png` and `hbn_quantum_geometry.png` are both dated 15 Aug 22:45, after the
  sign-fix commit `7565c36` — regenerated.
- **`notebooks/15_parity_invariants.ipynb`'s executed outputs are consistent with the
  retraction**: graphene gives `nu = 1` with `deltas = {-1, -1, +1, -1}` (product $-1$,
  i.e. $\nu=1$ ✓), matching `get_z2_invariant()`'s WCC answer. It does not run the cesium
  structure at all, so there is no retracted number to go stale there.
- **Notebook 05's markdown already states the new convention explicitly**: "the Berry
  phase of each plaquette is $-\arg$ of its link-variable product, not $+\arg$" ✓.
- **`notebooks/13_z2_invariant_3d.ipynb` already carries a retraction cell** (markdown
  cell 5, "Note: this structure's $\nu_0$ has since been retracted"), which is why its
  checked-in `(1, (1,1,1), (0,0,0))` output is acceptable in place. Its later cell 7
  ("The $k_1=0$ plane's WCCs wind ... an odd number of times") describes the plot honestly
  and is covered by cell 5's "read the numbers below as mechanics". Left alone.
- **pyqula mentions**: the only *agreement* claim is `docs/design.md:1665-1673` /
  `physics.tex:189-191`, both of which correctly state elkpy now **differs** from pyqula.
  The remaining mentions (`parsers/berry.py:225,241`, `calculation.py`, design.md §13)
  describe pyqula's `berry_curvature(h,k,dk)` **API/corner-ordering** convention, not its
  sign — correct as written.
- **CP¹ / Provost-Vallée pin** is $-\tfrac12\sin\theta$ everywhere it appears
  (`tests/test_quantum_geometry_gauge_invariance.py:206` asserts `-0.5*np.sin(theta0)`;
  `physics.tex:975`; `design.md:1634-1640`), with the old $+\tfrac12\sin\theta$ correctly
  marked historical.
- **`README.md` notebook table**: 15 rows, exactly matching `notebooks/01..15` (`_scratch/`
  excluded), and the stated count "Fifteen notebooks" is right. Every one of the 15 is also
  reachable from a `FUNCTIONALITIES` bullet's `[[notebook]]` link. No discrepancy.
- **Every test file named in `docs/design.md` exists** (17 distinct files, all present in
  `tests/`), and each names a test that matches its claim — with the single exception
  logged as §2.2 above. Notably §21's surviving claim now correctly points at
  `test_z2_3d_axis_splits_are_algebraically_consistent` (which asserts
  `len(set(nu0_by_axis)) == 1` and `result["nu0"] in (0, 1)`), not a $\nu_0=1$ assertion.
- **`docs/design.md` §22/§23 and `tests/test_calculation_momentum.py`,
  `tests/test_parsers_optical.py`, `tests/test_calculation_parity.py`,
  `src/elkpy/parsers/berry.py`, `optical.py`, `wilson.py`** are all consistent with both
  corrections. No assertion anywhere inserts a minus sign between the Kubo and Wilson-loop
  routes.

---

## 4. Fixes applied in this audit (editable files only)

1. `tests/test_calculation_atom_projection.py:166-169` — retargeted the broken
   cross-reference (§2.1) from `test_calculation_berry.py` to
   `test_calculation_quantum_geometry.py`, with a parenthetical noting why the former is
   not the right file.
2. `tests/test_calculation_z2_3d.py:48-49` — repaired the mangled retraction splice
   ("what IS asserted, and IS basis-independent, was reported as nu0=1") to "What this
   module ONCE asserted, and what IS basis-independent, was nu0=1 -- RETRACTED".
3. `tests/test_calculation_z2_3d.py:105-108` — "matching the parameters this module's
   result was verified with" → now says the mesh is known too coarse to resolve $\nu_0$'s
   value but still fine for the axis-split algebra the module asserts.
4. `tests/test_calculation_z2_3d.py:134-137` — the `DELTA` comment's bare
   "the strong-topological-insulator sign" now says this is the sign in FKM's *model*, and
   that this Cs structure does not realize that phase.

Verified after editing: both files byte-compile and `pytest --collect-only` collects them
cleanly (10 tests; no Elk binary invoked, no DFT run).
