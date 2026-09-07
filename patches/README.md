# Patch series

Tracks every edit elkpy makes to a `build/elk/` copy of `vendor/elk/` (see
`docs/design.md` §8 — `vendor/elk/` itself is never touched). Applied in
order by `build_elk.sh` via `patch -p1`.

When bumping `vendor/elk/` to a new upstream release, check each row's
"upstream file" against the new version before assuming a patch still
applies — that's the actual cost of this approach, and this table is meant
to make it a quick diff-checklist instead of a re-read of the raw patches.

| # | New file(s) added | Upstream file touched | Hook point | Adds |
|---|---|---|---|---|
| [0001](0001-per-species-soc-scale.patch) | — | `src/modmain.f90`, `src/readinput.f90`, `src/gensocfr.f90`, `src/elk.f90` (docs) | `modmain.f90`: new `socscfsp(maxspecies)` array. `readinput.f90`: new `case('elkpy_socscale')` block parser. `gensocfr.f90`: existing per-atom SOC loop reads `socscfsp(is)` instead of global `socscf` when set | `elkpy_socscale` input block — per-species spin-orbit scale factor override |
| [0002](0002-berry-curvature-wilson-loop.patch) | `src/elkpy_berry.f90` | `src/elk.f90` (task dispatch `case`), `src/modmain.f90` (new module vars), `src/readinput.f90` (`elkpy_berry`/`elkpy_berry_path` block parsers), `src/Makefile` (`SRC_ELKPY` var) | `elk.f90`: two new `case` arms in the task-number dispatch | Tasks 9000/9001 — Berry curvature via Wilson loop, on a k-mesh (9000) or an arbitrary k-point list/path (9001, via `elkpy_wfcorner`) |
| [0003](0003-eigenstate-session.patch) | `src/elkpy_eigenstates.f90` | `src/elk.f90` (task dispatch `case`), `src/Makefile` (`SRC_ELKPY` var) | `elk.f90`: one new `case` arm | Task 9002 — interactive stdin/stdout eigenstate/overlap query session (`elkpy_diagonalize` factored out of patch 0002's `elkpy_wfcorner`) |
| [0004](0004-atom-projection.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | New `elkpy_atomproj` subroutine + a call from `elkpy_eigenstate_session`'s query loop | `PROJECTION` query on the task-9002 session — per-atom muffin-tin projection operators |
| [0005](0005-orbital-projection.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | New `elkpy_orbitalproj` subroutine + a call from `elkpy_eigenstate_session`'s query loop | `ORBITAL` query on the task-9002 session — per-atom, l-resolved (s/p/d/f) muffin-tin projection operators |
| [0006](0006-angular-momentum.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | New `elkpy_angmomproj` subroutine (reuses upstream `lopzflm.f90` unmodified) + a call from `elkpy_eigenstate_session`'s query loop | `ANGMOM` query on the task-9002 session — per-atom, l-resolved orbital angular momentum operators $L_x,L_y,L_z$ |
| [0007](0007-momentum-matrix-elements.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | New `elkpy_momentum` subroutine (reuses upstream `genpmatk.f90` unmodified — the same one `putpmat.f90`/task 120 calls) + a call from `elkpy_eigenstate_session`'s query loop | `MOMENTUM` query on the task-9002 session — momentum (velocity) matrix elements $p^a_{nm}$ for all `nstsv` states, plus that diagonalisation's eigenvalues |
| [0008](0008-inversion-parity-operator.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | New `elkpy_parity` subroutine (lifts getevecfv.f90's symmetry transformation of the first-variational coefficients; reuses upstream `rotzflm`, `genwfsv`, `genolpq` unmodified) + a call from `elkpy_eigenstate_session`'s query loop | `PARITY` query on the task-9002 session — the inversion operator $\langle\psi_m\vert\hat I\vert\psi_n\rangle$ at a time-reversal-invariant momentum, for the Fu-Kane $Z_2$ symmetry indicators |
| [0009](0009-momentum-evecsv.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | One new `intent(out)` argument on 0007's `elkpy_momentum` (`evecsv_out`, written where the discarded local `evecsv` was) + one extra write block in the `MOMENTUM` case of `elkpy_eigenstate_session`'s query loop | `MOMENTUM` also returns that diagonalisation's `evecsv`, so the $\S17$ spin operators and the $\S22$ velocity matrix elements share one eigenbasis — the prerequisite for the spin current operator $J^z_a=\tfrac12\{S_z,v_a\}$, spin Berry curvature and spin Hall conductivity ($\S24$) |
| [0010](0010-symmetry-operators.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | New `elkpy_symop` subroutine (generalises 0008's `elkpy_parity` from inversion to any space-group element; reuses `rotzflm`/`genwfsv`/`genolpq` unmodified) + `SYMLIST` and `SYMMETRY` arms in the query loop | `SYMLIST` (Elk's crystal symmetries) and `SYMMETRY` (the operator $\langle\psi_m\vert\hat O\vert\psi_n\rangle$ at a fixed k-point), for rotation-eigenvalue symmetry indicators |
| [0011](0011-spin-polarized-stm.patch) | `src/elkpy_stm.f90` | `src/elk.f90` (task dispatch `case` + docs), `src/modmain.f90` (new module vars), `src/readinput.f90` (`elkpy_stmdir`/`elkpy_stmpol`/`elkpy_stmbias`/`elkpy_stmint` block parsers), `src/Makefile` (`SRC_ELKPY` var) | `elk.f90`: one new `case` arm (9003, 9004) | Tasks 9003/9004 — spin-polarised STM images (Tersoff-Hamann): upstream task 162's own occupation-replacement + `rhomagv` route, but keeping the magnetisation it discards, and plotting $n$, $\mathbf m\cdot\hat{\mathbf e}_T$ and $n+P_T\,\mathbf m\cdot\hat{\mathbf e}_T$ |
| [0012](0012-vertical-transport.patch) | `src/elkpy_transport.f90` | `src/elk.f90` (task dispatch `case` + docs), `src/modmain.f90` (new module vars), `src/readinput.f90` (`elkpy_transport_exit`/`_window`/`_kgrid`/`_koffset`/`_sdir`/`_spol` block parsers), `src/Makefile` (`SRC_ELKPY` var) | `elk.f90`: one new `case` arm (9005) | Task 9005 — vertical tunnelling transport through a 2D material: the exit-plane Gram matrices $S_{\bf k}[n,n']=\int_{\rm plane}\psi^*_{n\bf k}\hat P_{\rm s}\psi_{n'\bf k}$ (closed form, via the in-plane G-vector orthogonality collapse) and the tip amplitudes $\psi_{n\bf k}({\bf r}_p)$, on a k-mesh the task generates and diagonalises itself |
| [0013](0013-lapw-export.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | New `elkpy_lapwexport` subroutine (reuses upstream `gengkvec`/`gensfacgp`/`match`/`eveqnfv`/`hmlfv`/`olpfv`/`hmlistl`/`olpistl` unmodified; copies `match.f90`'s own four-line construction of the derivative matrix $D$, which `zgesv` destroys in place) + a `LAPW` arm in `elkpy_eigenstate_session`'s query loop | `LAPW` query on the task-9002 session — every ingredient of the first-variational LAPW eigenvalue problem at one k-point: the $\bf G+k$ set, `apwalm` (which nothing upstream writes at all), the derivative matrices $D$, the radial-function tails behind them, $H$ and $O$ with their interstitial parts written separately, and Elk's own `evalfv`/`evecfv`. An export for the JAX port (`docs/jax_port.md`, `docs/design.md` §33), not a physical observable |
| [0014](0014-lapw-export-radial-integrals.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | Extra `write` statements at the end of 0013's `elkpy_lapwexport`, behind a header of their own; no new subroutine and no new call site | The `LAPW` query also returns the muffin-tin radial integrals `oalo`, `ololo`, `haa`, `hloa`, `hlolo`, the local-orbital bookkeeping (`nlorb`, `lorbl`, `idxlo`) and the complex Gaunt array `gntyry` — together, every remaining input of `olpfv`/`hmlfv` beyond `apwalm` and the interstitial blocks 0013 already writes. This is what lets the port assemble each muffin-tin block itself and compare it against Elk's, one upstream routine at a time (`src/elkjax/hamiltonian.py`, `docs/jax_port_phase1.md`) |

| [0015](0015-lapw-export-potential-radial-functions.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | One `call genapwlofr` at the top of 0013's `elkpy_lapwexport`, plus extra `write` statements at the end, behind a header of their own; no new subroutine and no new call site | The `LAPW` query also returns the inputs of 0014's radial integrals: the muffin-tin Kohn-Sham potential `vsmt` in Elk's own packing, the radial mesh `rlmt` and its quadrature weights `wr2mt`, the APW/local-orbital linearisation energies and derivative orders, and `apwfr`, `apwdfr` and `lofr` in full. With them the port closes the chain vsmt → apwfr/lofr → radial integrals → H, O → evalfv (`src/elkjax/radial.py`, `src/elkjax/radial_functions.py`). The `genapwlofr` call is load-bearing, not tidiness: `gndstate` mixes the potential AFTER building the radial functions, so without it the export carries a `vsmt` one SCF iteration ahead of its own `haa` — measured, 3e-10 relative, and indistinguishable from a transcription bug |
| [0016](0016-groundstate-export.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | New `elkpy_gsexport` subroutine + a `GROUNDSTATE` arm in `elkpy_eigenstate_session`'s query loop | `GROUNDSTATE` query on the task-9002 session — the converged density and potentials on the grids Elk holds them on, plus everything needed to integrate and transform on those grids. Muffin tin, in Elk's own packing: `rhomt`, `vclmt`, `vxcmt`, `exmt`, `ecmt`, the radial mesh `rlmt` and weights `wr2mt`, and the four spherical-harmonic transform matrices `rbshti`/`rfshti`/`rbshto`/`rfshto`. Interstitial, on the real-space FFT grid: `rhoir`, `vclir`, `vxcir`, `exir`, `ecir`, `vsir`, `cfunir`; in G-space `cfunig` and `vsig`, with `ivg`, `igfft`, `gc` and `vgc`. Plus `omega`, `avec`, `bvec`, `idxis` and the shape integers. Takes **no k-point** — unlike every other query, this half of the calculation is k-independent. The Phase 2 counterpart of 0013-0015's `LAPW` query (`docs/jax_port_phase2.md`). Written 4 per line in `ES25.16E3` rather than by list-directed output: these arrays are two orders of magnitude larger than anything 0013-0015 writes |
| [0017](0017-groundstate-export-poisson.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | A further block of `write` statements at the end of 0016's `elkpy_gsexport`; no new subroutine and no new call site | Four more fields on the `GROUNDSTATE` query, and **only** the four the Weinert Poisson solve cannot rebuild from what 0016 already exports: `wprmt` (`wsplint`'s cumulative spline weights, which `zpotclmt`'s `splintwp` consumes four at a time — NOT `wr2mt`, and no closed form worth retyping), `vcln` (the nuclear potential from `potnucl`, which `potcoul` adds to the l=0 channel BEFORE `zpotcoul` reads the sphere-boundary multipoles, so omitting it gets every $q_{lm}$ wrong), `npsd`/`lnpsd` and `atposc`, plus **`spzn` and `energy.f90`'s own thirteen converged scalars** (`evalsum`, `engykn`, `engyvcl`, `engyvxc`, `engymad`, `engyen`, `engyhar`, `engycl`, `engynn`, `engyx`, `engyc`, `engyts`, `engytot`) — exported so a total-energy transcription can be checked TERM BY TERM at full precision rather than against `INFO.OUT`'s print width, which is the whole diagnostic value: a total that agrees to 1e-8 says nothing about which convention is right. Everything else `potcoul` needs is rebuilt in `src/elkjax/poisson.py`: $r^\ell$ and $R^\ell$ from the mesh, $4\pi/G^2$ from `gc`, and `ylmg`/`sfacg`/`jlgrmt` from `elkjax.lapw`'s `genylmv`/`gensfacgp`/`sbessel`, which patch 0013 already pinned against Elk element-wise. Exporting `ylmg` alone would be ~38 MB of text to avoid reusing code that is already checked |
| [0018](0018-symmetrisation-operator.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | A further block of `write` statements at the end of `elkpy_gsexport`, which now calls upstream `symrfmt` on basis vectors; no new subroutine and no new call site | `symrfmt`'s muffin-tin symmetrisation as a **linear operator**, one $l_{\max}^{\rm o}$-square matrix per ordered atom pair. `potxc.f90:55-58` symmetrises `vxcmt`/`bxcmt` and not `exmt`/`ecmt`, so a pointwise transcription of the functional reproduces the energy densities exactly and misses `vxcmt` by 1.2e-4 relative; applying this closes it to 6.4e-14 (`docs/jax_port_phase2.md` §2d/§2g). **Exported rather than transcribed on purpose**: `rotrflm`'s Euler-angle and Wigner-$D$ construction has no consumer inside Elk but `symrfmt`, so a re-derivation would have no independent check except agreement with what it replaces — and Elk's whole atom bookkeeping (`ieqatom`, `tfeqat`, the *inverse* lattice rotation in the rotate-into-equivalent loop) would have to come with it. Built by calling `symrfmt` on basis vectors: a rotation is diagonal in the radial index and does not mix $\ell$, so one matrix per atom pair is the whole of it, and the inner region uses its top-left $l_{\max}^{\rm i}$ block |
| [0019](0019-densityk-export.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | New `elkpy_denskexport` subroutine + a `DENSITYK` arm in the query loop; it calls upstream `match` and `rhomagk` the way `rhomagv` does | `DENSITYK` query — what `rhomagv` feeds to `rhomagk` for the ground state's OWN k-set (`wkpt`, `vkl`, `occsv`, `ngk`, `nmat`, `igkig`, `evecfv`), the COARSE muffin-tin and interstitial meshes (`lradstp`, `nrcmt`, `nrcmti`, `npcmt`, `npcmti`, `ngdgc`, `ngtc`, `igfc`), the COMPLEX SHT matrices `zbshti`/`zbshto` that `wfmtsv` applies with `tsh=.false.`, and **the reference itself**: `rhomagk` looped over the k-set into a LOCAL array, i.e. the density before `rhomagsh`, `symrf`, `rfmtctof` and `rhocore` — each of which is a step the port does not transcribe, and comparing against the converged `rhomt` would fold all four into one number. Same design as 0018: Elk's own routine produces the reference. Used by `src/elkjax/density.py`, which reproduces both halves to 9e-16 (`docs/jax_port_phase2.md` §2h) |

## Notes

- 0001/0002/0003/0011 each touch `src/elk.f90`'s task dispatch — the one
  genuinely unavoidable shared hook point (per `docs/design.md` §8, this is
  kept to a single added `case`/line per patch, marked `! elkpy: ...`). If a
  future upstream version restructures `elk.f90`'s dispatch, all four will
  need re-hooking together.
- 0002, 0003 and 0011 each edit `src/Makefile`'s `SRC_ELKPY` variable to
  register their new file — a likely conflict point if upstream ever adds its own
  `SRC_ELKPY`-shaped variable or restructures the source list.
- 0004-0010 only touch elkpy's own `elkpy_eigenstates.f90` (added by 0003),
  not any upstream file — lowest risk of the eleven on an upstream bump. They
  do each *call* an upstream subroutine unmodified (`wfmtsv`, `lopzflm`,
  `genpmatk`, `rotzflm`/`genwfsv`/`genolpq`), so an upstream signature change
  to one of those is the realistic breakage mode, not a patch-application
  conflict. 0009 is the one patch in the series that touches no upstream
  subroutine at all — it only widens an elkpy-added routine's own argument
  list and prints an array that was already being computed — so it is the
  cheapest to re-evaluate, but it *does* depend on 0007's `elkpy_momentum`
  textually and must be re-applied after it. 0008 additionally *copies* a block of `getevecfv.f90` (the
  symmetry transformation of first-variational coefficients) rather than
  calling it, since upstream exposes it only inline — so an upstream change
  to that transformation would need the copy re-synced, and it is the one
  place in the series where a silent divergence from upstream is possible.
- 0012 also adds a new `.f90` file, and so touches the same two shared hook
  points (`elk.f90`'s dispatch, `Makefile`'s `SRC_ELKPY`) as 0002/0003/0011.
  Its own new routine calls upstream `plotpt2d`, `wfirsv`, `gengkvec`,
  `match`, `eveqnfv` and `eveqnsv` unmodified — the same set patch 0002's
  `elkpy_wfcorner` already depends on, plus `wfirsv`/`plotpt2d` — so an
  upstream signature change to one of those is the realistic breakage mode
  rather than a patch-application conflict. It writes only exported
  ingredients, never a physical result: all of the transport arithmetic is in
  `src/elkpy/parsers/transport.py`.
- 0011 is the first patch since 0002/0003 to add a new `.f90` file, so like
  them it edits `src/Makefile`'s `SRC_ELKPY` variable and `elk.f90`'s task
  dispatch — the same two shared hook points, and the same re-hooking cost on
  an upstream bump. It is also the only patch in the series whose new code
  deliberately *differs* from the upstream routine it mirrors:
  `wfplot.f90`'s task-162 branch folds `wkpt(ik)` into `occsv` on top of the
  factor `rhomagv` already applies, and `elkpy_stm.f90` does not (see
  `docs/design.md` §30). If a future upstream release fixes that, the
  divergence disappears rather than conflicting — but the elkpy-vs-162
  regression test's expected factor of $N_{\mathbf k}$ would then need to
  become 1.
- 0013 is, like 0004-0010, confined to elkpy's own `elkpy_eigenstates.f90`
  and touches no upstream file — but it is the one patch in the series whose
  correctness depends on details of upstream routines it does not call.
  Two are load-bearing and would fail silently on an upstream change: it
  forces `tefvr=.false.` while building `H` and `O`, because `olpaa`/`hmlaa`
  otherwise route through `rzmctmu`, which accumulates only the real part of
  the muffin-tin APW-APW block (correct for `eveqnfvr`, wrong as an export,
  and Hermitian and positive definite either way); and it writes only the
  upper triangles, because `olpistl`/`hmlistl` and every muffin-tin
  contribution run `do i=1,j` and the rest of the array is never assigned.
  Like 0008 it also *copies* a block of an upstream routine rather than
  calling it — `match.f90`'s construction of the derivative matrix $D$,
  which cannot be captured because `zgesv` overwrites it in place — so that
  copy is the second place in the series where a silent divergence from
  upstream is possible. `docs/design.md` §33 states all three.
- 0014 appends to 0013's own subroutine and so inherits every one of its
  hazards; it adds one of its own. `hmlrad.f90` builds `hlolo`'s
  $\ell_2=0$ element as the UNSYMMETRISED $\int u^{\rm lo}_i(\hat H
  u^{\rm lo}_j)r^2dr$ — no averaging over the two orderings and no kinetic
  surface term, unlike `haa`, whose counterpart is explicitly averaged and
  whose transpose is explicitly assigned. `hmllolo` therefore evaluates each
  local-orbital pair in one order only and Hermitises the rest, and a
  consumer of the exported array must do the same. The export writes the
  array as Elk holds it, asymmetry included; `src/elkjax/hamiltonian.py`'s
  `_hermitise_evaluated_half` is where the restriction lives. Only a species
  with two local orbitals sharing an $\ell$ can see this at all (nitrogen
  has two $\ell=0$; silicon has one s and one p and cannot).
- 0016 writes `vsig(1:ngvc)`, NOT `vsig(1:ngvec)`. `init0.f90` allocates it
  to `ngvc` because `genvsig` fills it from the COARSE grid, so it only
  carries $|G|\le 2g_{k\max}$; the two differ by a factor of six here (1243
  against 7799) and the longer loop reads past the end of the array. Written
  down because the wrong version runs, produces plausible numbers for the
  first `ngvc` of them, and is only caught by a bounds check.
- 0018's exported operator is idempotent to 1e-16 on a CUBIC lattice and
  only to 1.2e-11 on a hexagonal one, growing with $\ell$. That is Elk's
  own arithmetic, not the export's: `symrfmt` builds each rotation through
  `roteuler`, whose inverse trigonometry is exact when the Cartesian
  `symlatc` entries are $0$ and $\pm1$ and is not otherwise. It bounds how
  idempotent `symrfmt` can be, not the operator's accuracy in use — which
  applies it once, and holds at 1e-14 on both.
- 0019 writes `evecfv(1:nmat)`, NOT `evecfv(1:ngk)`. The coefficients past
  `ngk` are the local orbitals, which `wfmtsv` reads as
  `evecfv(ngp+idxlo(...))`. Truncating at `ngk` leaves the INTERSTITIAL
  density exact -- local orbitals vanish there -- and the muffin-tin one
  smooth, positive, correctly scaled and 100% wrong. Hit for real.
- 0019 calls `genapwlofr` on entry for the same reason 0015 does, and the
  consequence here is sharper: without it the exported `rhomagk` reference
  is built from the previous iteration's radial functions while a
  transcription uses the regenerated ones, so the query's answer depended
  on whether `LAPW` had been asked for first. Measured 1.2e-10 against
  9e-16 -- close enough to read as a transcription bug.
- Task numbers 9000-9005 and the `elkpy_`-prefixed block/variable names are
  deliberately in an unused-by-upstream range (`docs/design.md` §8) to
  minimize collision risk on a version bump.
