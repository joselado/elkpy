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
| [0008](0008-inversion-parity-operator.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | New `elkpy_parity` subroutine (lifts getevecfv.f90's symmetry transformation of the first-variational coefficients; reuses upstream `rotzflm`, `genwfsv`, `genolpq` unmodified) + a call from `elkpy_eigenstate_session`'s query loop | `PARITY` query on the task-9002 session — the inversion operator $\langle\psi_m\|\hat I\|\psi_n\rangle$ at a time-reversal-invariant momentum, for the Fu-Kane $Z_2$ symmetry indicators |
| [0009](0009-momentum-evecsv.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | One new `intent(out)` argument on 0007's `elkpy_momentum` (`evecsv_out`, written where the discarded local `evecsv` was) + one extra write block in the `MOMENTUM` case of `elkpy_eigenstate_session`'s query loop | `MOMENTUM` also returns that diagonalisation's `evecsv`, so the $\S17$ spin operators and the $\S22$ velocity matrix elements share one eigenbasis — the prerequisite for the spin current operator $J^z_a=\tfrac12\{S_z,v_a\}$, spin Berry curvature and spin Hall conductivity ($\S24$) |
| [0010](0010-symmetry-operators.patch) | — | `src/elkpy_eigenstates.f90` (elkpy's own file, added by 0003) | New `elkpy_symop` subroutine (generalises 0008's `elkpy_parity` from inversion to any space-group element; reuses `rotzflm`/`genwfsv`/`genolpq` unmodified) + `SYMLIST` and `SYMMETRY` arms in the query loop | `SYMLIST` (Elk's crystal symmetries) and `SYMMETRY` (the operator $\langle\psi_m|\hat O|\psi_n\rangle$ at a fixed k-point), for rotation-eigenvalue symmetry indicators |
| [0011](0011-spin-polarized-stm.patch) | `src/elkpy_stm.f90` | `src/elk.f90` (task dispatch `case` + docs), `src/modmain.f90` (new module vars), `src/readinput.f90` (`elkpy_stmdir`/`elkpy_stmpol`/`elkpy_stmbias`/`elkpy_stmint` block parsers), `src/Makefile` (`SRC_ELKPY` var) | `elk.f90`: one new `case` arm (9003, 9004) | Tasks 9003/9004 — spin-polarised STM images (Tersoff-Hamann): upstream task 162's own occupation-replacement + `rhomagv` route, but keeping the magnetisation it discards, and plotting $n$, $\mathbf m\cdot\hat{\mathbf e}_T$ and $n+P_T\,\mathbf m\cdot\hat{\mathbf e}_T$ |
| [0012](0012-vertical-transport.patch) | `src/elkpy_transport.f90` | `src/elk.f90` (task dispatch `case` + docs), `src/modmain.f90` (new module vars), `src/readinput.f90` (`elkpy_transport_exit`/`_window`/`_kgrid`/`_koffset`/`_sdir`/`_spol` block parsers), `src/Makefile` (`SRC_ELKPY` var) | `elk.f90`: one new `case` arm (9005) | Task 9005 — vertical tunnelling transport through a 2D material: the exit-plane Gram matrices $S_{\bf k}[n,n']=\int_{\rm plane}\psi^*_{n\bf k}\hat P_{\rm s}\psi_{n'\bf k}$ (closed form, via the in-plane G-vector orthogonality collapse) and the tip amplitudes $\psi_{n\bf k}({\bf r}_p)$, on a k-mesh the task generates and diagonalises itself |

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
- Task numbers 9000-9005 and the `elkpy_`-prefixed block/variable names are
  deliberately in an unused-by-upstream range (`docs/design.md` §8) to
  minimize collision risk on a version bump.
