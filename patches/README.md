# Patch series

Tracks every edit elkpy makes to a `build/elk/` copy of `vendor/elk/` (see
`docs/design.md` §8 — `vendor/elk/` itself is never touched). Applied in
order by `scripts/build_elk.sh` via `patch -p1`.

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

## Notes

- 0001/0002/0003 each touch `src/elk.f90`'s task dispatch — the one
  genuinely unavoidable shared hook point (per `docs/design.md` §8, this is
  kept to a single added `case`/line per patch, marked `! elkpy: ...`). If a
  future upstream version restructures `elk.f90`'s dispatch, all three will
  need re-hooking together.
- 0002 and 0003 both edit `src/Makefile`'s `SRC_ELKPY` variable to register
  their new file — a likely conflict point if upstream ever adds its own
  `SRC_ELKPY`-shaped variable or restructures the source list.
- 0004 and 0005 only touch elkpy's own `elkpy_eigenstates.f90` (added by
  0003), not any upstream file — lowest risk of the five on an upstream
  bump.
- Task numbers 9000-9002 and the `elkpy_`-prefixed block/variable names are
  deliberately in an unused-by-upstream range (`docs/design.md` §8) to
  minimize collision risk on a version bump.
