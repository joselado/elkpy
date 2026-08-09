# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Roadmap Tiers 1-3 from `docs/roadmap.md` are implemented: `Structure` → `Calculation` with
`get_energy()`, `get_bands()`, `get_dos()`, `get_forces()`, `get_relaxed()`, `get_effective_mass()`,
`get_density()`, `get_phonon_dos()`/`get_phonon_dispersion()`, and a `run_tasks()` escape hatch, plus a
`spec.py` module for version-coupled task/xctype/filename knowledge. Verified end-to-end against a real
compiled Elk binary on bulk Si/Fe (`tests/test_calculation_si.py`, `tests/test_calculation_fe.py`,
`tests/test_calculation_si_phonons.py`). Also present: `vendor/elk/` (vendored Elk 11.0.2, unmodified,
tracked in git), `docs/elk_manual.pdf`/`.txt` (official manual, plain-text version for grepping), and
`docs/design.md` + `docs/roadmap.md` (architecture strategy and forward plan — read before adding code).
Not implemented: symbolic k-path's disconnected-segment support (`,` breaks), the classical supercell
phonon method (task 200, DFPT/task 205 only), scheduler-backed launchers, MPI, named `get_*` methods for
potential/ELF volumetric plots (reachable via `run_tasks()` + `parsers.volumetric`). Check `src/elkpy/`
directly rather than assuming the docs describe current code; update both as they diverge.

Also implemented, as the first real entry in the Fortran patch series described below:
`Calculation(spinorb=, soc_scale={"Fe": 1.5, ...})` — per-species scaling of the spin-orbit coupling
term, on top of upstream Elk's single global `socscf` scalar. `patches/0001-per-species-soc-scale.patch`
adds a `socscfsp(maxspecies)` array (`modmain.f90`) and an `elkpy_socscale` input block (`readinput.f90`)
overriding `socscf` per species inside `gensocfr.f90`'s existing per-atom loop — the smallest possible
hook, verified against a real compiled binary to reproduce the global-`socscf` result exactly for a
single-species cell and to scale independently per species in a two-species cell
(`tests/test_calculation_soc.py`). Physics writeup (Koelling-Harmon SOC term, why a global scale is
the wrong shape for multi-species cells): `docs/design.md` §12 and `docs/physics.tex` (Part I).

Also implemented, as the second entry in the Fortran patch series:
`Calculation.get_berry_curvature(ist0, ist1, directions=(1, 2))` — Berry curvature and Chern number via
the Wilson-loop (Fukui-Hatsugai-Suzuki) method. `patches/0002-berry-curvature-wilson-loop.patch` adds a
new task (9000) and `elkpy_berry.f90`, reusing Elk's own `genwfsvp`/`genolpq` (the machinery behind
Wannier90 export, task 550) to export mesh-neighbour wavefunction overlap matrices without needing
task 550's own neighbour-shell search (which calls the external Wannier90 library, unavailable in this
build) — the two loop directions are exact `ngridk` mesh generators, so the neighbour is constructed
directly. All Wilson-loop arithmetic (link variables, plaquette flux, Chern number, admissibility) is
done in Python (`parsers/berry.py`), independently unit-tested against synthetic overlap matrices, both
for gauge invariance and, separately, for sign -- gauge invariance alone can't catch a conjugation-sign
flip, since `conj(M)` is exactly as gauge-invariant as `M`, so the Python arithmetic's sign is pinned
directly against FHS eq. 8 with hand-built matrices of known phase (`tests/test_berry_gauge_invariance.py`).
The Fortran conjugation convention itself rests on a `zgemv` BLAS-semantics derivation, not a runtime
test. Against a real compiled binary, bulk Si's trivial valence manifold gives a Chern number of
floating-point zero (`tests/test_calculation_berry.py`) -- which, being zero either way, would not itself
catch a sign error.

Also implemented: `Calculation.get_berry_curvature_path(kpoints, ist0, ist1, directions=(1, 2), dk=)`
(task 9001) -- Berry curvature at an arbitrary, explicit list of k-points (e.g. a band-structure-style
path), via one small Wilson loop per point (`pyqula`'s `berry_curvature(h, k, dk=...)` convention)
rather than a periodic mesh. Also accepts `kpath="GKMG"` (pyqula/ASE-style symbolic path, same
convention as `get_bands()`/`get_phonon_dispersion()`'s `kpath=`, resolved via `_kpath_to_points()`)
instead of `kpoints=`, discretized into `npoints` points with each returned point additionally carrying
a `"distance"` entry for plotting; unlike `get_bands()`/`get_phonon_dispersion()` (which hand vertices to
Elk's own `plot1d` task, so a disconnected `,` path can't be interpolated), `get_berry_curvature_path()`
evaluates every point independently, so a disconnected `,` path (e.g. to jump straight to a specific K′
zone image) is fully supported. Needed one genuinely new piece of Fortran, not just a smaller task 9000:
`elkpy_wfcorner` expands a wavefunction at an arbitrary k-point via fresh on-the-fly diagonalisation
(`eveqnfv`/`eveqnsv`, the same pattern `src/bandstr.f90` uses for a band-structure path) rather than
`genwfsvp`'s file-backed `getevecfv`/`getevecsv`, which only knows previously-diagonalised mesh points --
so this mode needs no `ngridk` alignment and no `reducek=0` at all, and a Γ-only ground state is
sufficient (not a stopgap), since task 9001 never touches the ground state's own mesh. Trade-off: no
Chern number (needs a closed cover of the zone) and no automatic gap check (no eigenvalues exported in
this mode). Verified against a real compiled binary two ways: mesh mode and path mode agree (to ordinary
numerical tolerance) when asked to evaluate the literal same four k-points on bulk Si
(`tests/test_calculation_berry.py::test_path_and_mesh_conventions_agree`); and on monolayer h-BN (broken
sublattice inversion symmetry, unlike Si), the occupied manifold's curvature is exactly antisymmetric
between the two physically inequivalent valleys K/K′ (8.568 vs -8.568 Bohr⁻², agreeing to 0.01%) --
required by time-reversal symmetry and a much sharper check than Si's Chern number ($0=-0$ either way).
That h-BN run also surfaced two real pitfalls worth remembering for any future band-window usage: the
occupied-band count should come from Elk's own `EIGVAL.OUT` occupation numbers, not assumed from a total
(core + valence) electron count (core electrons aren't among the valence bands `nstsv` indexes at all);
and a single band's curvature can diverge near a point where it's degenerate with a *neighbouring
occupied* band, not just the first unoccupied one, even when the requested window's own outer boundary
stays gapped -- the fix is windowing the full degenerate group together, not picking a different single
band. Physics writeup (both modes, the FHS link-variable/plaquette-flux construction, and exactly what
is/isn't verified): `docs/design.md` §13 and `docs/physics.tex` (Part II).

Also implemented, as the third entry in the Fortran patch series:
`Calculation.eigenstate_session()` (task 9002, `patches/0003-eigenstate-session.patch`,
`src/elkpy_eigenstates.f90`) — an interactive, long-lived `elk` subprocess for fast repeated
eigenstate/overlap queries, plus one-off convenience wrappers `get_eigenstates(k)` and
`get_overlap(k_a, k_b, ist0, ist1)`. Initially proposed as an f2py in-memory bridge; rejected
because the dominant per-query cost is Elk's ground-state-dependent setup, not process-spawn
overhead, and making an f2py bridge robust would mean converting Elk's pervasive bare `stop`
calls into catchable errors across most of `vendor/elk/src/` — not achievable additively. The
persistent-subprocess design gets the same "stay warm" benefit through a stdin/stdout query
loop instead, with no new build machinery. `EIGENSTATES`/`OVERLAP` queries reuse
`elkpy_wfcorner` (task 9001's fresh on-the-fly diagonalisation) and a new
`elkpy_diagonalize` (the diagonalisation-only half of the same code, kept separate rather
than extending `elkpy_wfcorner` with optional arguments, since a Fortran `optional` dummy
argument needs an explicit interface at every call site, which task 9001's existing call
site doesn't have). Verified against a real compiled binary: `evecsv` is unitary at an
arbitrary k-point; `overlap(k, k, ...)` is the identity matrix to the ~1e-3 tolerance set by
`genwfsvp`/`genolpq`'s own real-space truncation error (not a bug — observed at the same
order for a single non-degenerate band too); a session survives many repeated queries; and
`overlap(Γ, Γ+e₁, ...)` matches task 9000's mesh-exported overlap for the same pair, checked
for one non-degenerate band only, since Si's bands 2-4 are degenerate at Γ and two
independent diagonalisations are free to (and empirically do) pick different bases within
that degenerate subspace (`tests/test_calculation_eigenstates.py`). Physics writeup (the
LAPW generalized eigenproblem, why `evecfv` isn't a valid raw-overlap basis but `evecsv` is
— only within one diagonalisation — and the cross-k overlap integral): `docs/design.md` §14
and `docs/physics.tex` (Part III).

Also implemented, and — unlike the three entries above — needing no new Fortran at all:
`Calculation.get_quantum_geometry(kpoints, ist0, ist1, directions=(1, 2), dk=)` — the full
quantum geometric tensor $Q_{ab}=g_{ab}-\tfrac i2 F_{ab}$ of a band window at an arbitrary
k-point: Berry curvature $F_{ab}$ (already covered by `get_berry_curvature_path()`) *and*
the quantum metric $g_{ab}$ (Fubini-Study/Provost-Vallee metric) it had been missing. Every
overlap this needs — including each loop corner's own self-overlap
`session.overlap(k, k, ...)`, the one new ingredient curvature alone never needed a name
for — is already exposed by the task 9002 interactive session, so this is driven entirely
from Python-side `EigenstateSession.overlap()` queries (9 per k-point) plus discretization
arithmetic in `parsers/quantum_geometry.py`; no `elkpy_quantum_geometry.f90`, no new task
number. The one real subtlety: Elk's `genolpq` overlap carries a ~1e-3 real-space
truncation floor that Berry curvature is immune to (it survives only in a closed-loop phase
product, which cancels a common-mode modulus deficiency exactly) but the quantum metric,
built directly from `1 - |overlap|^2`, is not — left uncorrected, this swamps the metric's
genuine `O(dk^2)` signal with a spurious offset that *diverges* as `dk` shrinks (measured
directly: raw g11 at dk=0.05→0.002 goes 40.5, 48.2, 52.5, 62.9, 131.8, i.e. blowing up).
The fix is exact Löwdin symmetric normalization (`M -> S_a^{-1/2} M S_b^{-1/2}` using each
corner's own self-overlap `S`) before computing anything from the raw overlaps — provably
inert for curvature (`S^{-1/2}` is Hermitian positive-definite, so it cannot shift
`arg(det M)`), and confirmed to fix the metric (same dk sweep, normalized: 40.5, 47.6,
49.7, 50.5, 51.0 — converging). Verified against a real compiled binary: that dk
divergence/convergence contrast on bulk Si; that curvature from `get_quantum_geometry()`
matches `get_berry_curvature_path()` on literally identical loop corners (task 9001's
corner-1 anchor, double the step size); and on monolayer h-BN (same structure as the Berry
curvature K/K′ check above), the metric is positive semi-definite at Γ/K/M/K′, its
diagonal is K/K′-symmetric to <1% (time-reversal symmetry, $g_{ab}(-k)=g_{ab}(k)$, unlike
curvature's sign flip), curvature reproduces the existing K/K′ antisymmetry via an entirely
independent Python code path, and at all four points $\det g \geq (F_{12}/2)^2$ — an exact
theorem ($Q_{ab}=g_{ab}-\tfrac i2F_{ab}$ is PSD as a 2x2 Hermitian matrix, being a sum of
Gram-matrix-like $\langle\cdot|Q|\cdot\rangle$ terms, so $\det Q\geq0$), not a loose
plausibility band — holds with margin at K/K′ and trivially at Γ/M, tying the new metric's
scale to the already-trusted curvature value without dividing by a curvature that
vanishes at two of the four points
(`tests/test_calculation_quantum_geometry.py`,
`tests/test_quantum_geometry_gauge_invariance.py`). Also pinned against an analytically
known case, the spin coherent state / CP¹ Fubini-Study metric (Provost & Vallée 1980's own
worked example): converges to the exact $g=\tfrac14\mathrm{diag}(1,\sin^2\theta)$ and
$F_{\theta\phi}=\tfrac12\sin\theta$. One genuine remaining wrinkle, not a bug: the metric's
off-diagonal component $g_{12}$ (built from a polarization-identity difference of three
comparable-magnitude quantities) converges markedly more slowly with `dk` than $g_{11}$,
$g_{22}$, or curvature — worth checking its own dk-convergence before trusting a single
value, same discipline `get_berry_curvature_path()` already asks for regarding curvature
generally. Physics writeup (the quantum geometric tensor, the Marzari-Vanderbilt/Resta
discretization, the `Tr[P∂P∂P]` derivation, the Löwdin-normalization fix):
`docs/design.md` §15 and `docs/physics.tex` (Part IV).

Also implemented, as the fourth entry in the Fortran patch series:
`Calculation.get_atom_projection(k, ist0, ist1)` (task 9002's new `PROJECTION` query,
`patches/0004-atom-projection.patch`) — the atom-projection operator $P_\alpha$ (the
muffin-tin-sphere restriction of the identity) for every atom in the cell at once, as an
`nst`$\times$`nst` Hermitian matrix in the second-variational eigenbasis of one fresh
diagonalisation at $k$. Reuses upstream `wfmtsv.f90` unchanged — the same per-atom,
$(\ell m,\sigma)$-resolved muffin-tin wavefunction expansion and `wr2cmt` radial
quadrature weight that `gendmatk.f90` already uses for the `dos`/`bandstr` tasks'
atom/lm-projected DOS and band-character output — but generalizes `gendmatk`'s
per-state diagonal occupation-matrix entry to the full off-diagonal `nst`$\times$`nst`
operator (summed over all $\ell,m$, not $\ell$-resolved), computed as one `zgemm` per
spin channel rather than `gendmatk`'s explicit per-$(\ell,m)$ loop. All `natmtot`
matrices come from the same single diagonalisation (not one query per atom), since — as
with `evecsv` generally (`docs/design.md` §14) — matrices from separate diagonalisations
aren't guaranteed to share a basis, and the operator's main correctness identity
($\sum_\alpha P_\alpha + P_\text{interstitial} = \mathbb 1$) needs every atom projected
consistently. Verified against a real compiled binary: every returned matrix is
Hermitian and positive semi-definite (immediate from `wr2cmt` being a positive
quadrature weight); summing every atom and subtracting from the identity is still
Hermitian PSD (the interstitial remainder can't be negative), with each atom's own
weight a substantial, physically reasonable fraction of bulk Si's cell; diamond Si's two
atoms — related by inversion through the bond midpoint, which sends $k\to-k$ and
therefore fixes Γ — have identical weight on band 1 at Γ (non-degenerate; bands 2-4 are
degenerate there, the same caveat already documented for `evecsv`); a diagonal entry
agrees to 5 decimal places with an entirely independent Fortran code path — upstream
`bandstr.f90` task 21's own atom-projected band-character output, same `gendmatk`/
`wfmtsv` machinery but a separate call site and reduction — catching what the
Hermitian/PSD and sum-below-identity checks alone would miss (a silent undercount, e.g.
a packing bug, that's still internally consistent); a spin-polarized run (`nspinor`=2)
stays Hermitian PSD, exercising the spin-channel accumulation the unpolarized checks
above never run twice (SOC remains untested for this feature); and on monolayer
h-BN at $K=(1/3,1/3,0)$, the occupied valence-top ($\pi$) band is N-dominated and the
unoccupied conduction-bottom ($\pi^*$) band is B-dominated — the standard qualitative
picture for h-BN's band character (the more electronegative N pulling the bonding
state's weight toward itself), a sharp sign-of-the-effect prediction rather than a
plausibility band (`tests/test_calculation_atom_projection.py`). Physics writeup (the
projection operator, the exact muffin-tin/interstitial partition identity, why it's not
gauge-comparable across separate diagonalisations): `docs/design.md` §16 and
`docs/physics.tex` (Part V).

## Architecture

- `src/elkpy/structure.py` — `Structure`: lattice vectors (`avec`, Bohr) + species, each atom either a
  bare `(x,y,z)` position or a `(position, bfcmt)` pair (per-atom magnetic field, manual sec. 5.2).
  `species_files` optionally overrides the `{symbol}.in` species-filename convention. `from_ase()`/
  `to_ase()` convert Angstrom/Cartesian ASE `Atoms` (optional dependency).
- `src/elkpy/calculation.py` — `Calculation`: owns one run directory and the ground-state-defining
  parameters (`xc`, `spinpol`, `spinorb`, `soc_scale`, `rgkmax`, `ngridk`, `extra_blocks` for anything
  else — e.g. `maxscl`). `soc_scale={"Fe": 1.5}` requires `spinorb=True` and needs the
  `patches/0001-per-species-soc-scale.patch` Fortran extension applied (i.e. a binary built via
  `scripts/build_elk.sh` after the patch was added — see git log for when). `get_*` methods block and
  can be expensive — real Elk subprocesses, not in-memory work.
  `ensure_ground_state()` reuses a prior task-0 run only if a JSON manifest (`.elkpy_manifest.json`)
  shows the basis/structure/functional-defining parameters (including `extra_blocks`) and the Elk
  binary identity are unchanged; sampling parameters (e.g. a denser `ngridk` passed to `get_dos()`) are
  free to differ. Every other `get_*` runs via `_run_resumed()` in its own **wiped-clean** subdirectory,
  never in `self.workdir` — this isn't just tidiness, it's load-bearing correctness: some tasks (phonon
  DFPT, task 205) treat the mere presence of prior output files as "already done" and resume from them,
  so a stale subdirectory from an earlier killed/crashed run silently corrupts the next result instead
  of erroring (hit this for real while implementing phonons — see git log). `converged` property +
  `raise_on_nonconvergence=` control whether non-convergence raises or must be checked explicitly.
- `src/elkpy/spec.py` — version-coupled knowledge (task codes, `xctype` codes, output filenames) as
  data, each entry cross-checked against `vendor/elk/src/` (not just the manual) — an Elk version bump
  should mean editing this one file.
- `src/elkpy/inputfile.py` — generic `elk.in` block writer (block name + value lines) and `read_blocks()`
  reader (used to parse `GEOMETRY_OPT.OUT`, which Elk writes in the same block syntax).
- `src/elkpy/launcher.py` — `LocalLauncher`: `run()` is blocking local subprocess execution, used by
  every task except the eigenstate session; `start_session()` instead returns a non-blocking `Popen`
  with stdin/stdout pipes, for `Calculation.eigenstate_session()` (task 9002) to drive interactively.
  Refuses `nprocs > 1` since `build-config/make.inc` builds serial (`mpi_stub.f90`) — that combination
  would silently launch N racing copies into one directory, not parallelize.
- `src/elkpy/session.py` — `EigenstateSession`: owns the interactive task-9002 subprocess started by
  `eigenstate_session()`, sending `EIGENSTATES`/`OVERLAP`/`PROJECTION` queries over stdin and parsing
  responses off stdout until closed (context manager) or told to `QUIT`. See `docs/design.md` §14 for
  why this is a persistent worker process rather than an f2py in-memory bridge, and §16 for
  `PROJECTION` (atom-projection operators).
- `src/elkpy/parsers/` — one small module per output file family (`info`, `totenergy`, `band` — reused
  for phonon dispersion, since `PHDISP.OUT` shares `BAND.OUT`'s exact layout — `dos`, reused for phonon
  DOS, `forces`, `geometry`, `effmass`, `volumetric`), each verified against real Elk output, not
  assumed from the manual. `berry` is the exception to "just a parser": it also does all of the
  Wilson-loop/Chern-number arithmetic in Python (`compute_berry_curvature()`), deliberately kept out of
  Fortran so it's unit-testable against synthetic overlap matrices (`tests/test_berry_gauge_invariance.py`)
  without an Elk run. `eigenstates` parses `EigenstateSession`'s stdout token stream (not a file) into
  energies/`evecsv`/overlap/atom-projection arrays, independently unit-testable the same way
  (`tests/test_parsers_eigenstates.py`).
- `src/elkpy/config.py` — locates the built `elk` binary (`build/elk/src/elk` by default, override via
  `ELKPY_ELK_BIN`) and the species directory (`vendor/elk/species/` by default).

## Project purpose

A Python interface to Elk, an all-electron full-potential linearized augmented-plane-wave (LAPW)
density-functional-theory (DFT) code written in Fortran. On top of the interface, this project adds
extra functionality that Elk itself does not provide.

## Development practices

Whenever the user asks for a new piece of functionality to be implemented, checking arXiv for a
relevant paper is encouraged where it plausibly helps — not limited to new physics — to ground the
implementation in an actual published source (method, formalism, convention, algorithm) rather than
guessing. This is a standing option to reach for, not something that needs to be requested each time.

This applies with the most force to new physics (a new Fortran capability, a new formula, a new
numerical scheme — not routine wrapping of an existing Elk task): checking arXiv for the relevant
method/paper is encouraged where it fits the task, to ground the implementation in the actual
published formalism (e.g. matching sign/normalization conventions, confirming which approximation a
term corresponds to) rather than guessing from the code alone.

Whenever a new formalism is added or an existing one is modified (new physics, a changed formula, a
different numerical scheme — not routine wrapping of an existing Elk task), update the documentation
describing it in both forms: the Markdown docs (`docs/design.md`/`docs/roadmap.md` or wherever the
capability is described) and the corresponding LaTeX writeup, added as a new `\part{}` (with a `\label`)
inside the single shared file `docs/physics.tex` — not a new `.tex` file per addition. Both must be
physics-focused, not just an API description: state the relevant formula(e), define each symbol, and
explain the physical meaning/approximation being made (what it captures, what it neglects, how it
relates to the underlying published method) — not merely "this function computes X". Each writeup must
also include a "how to use in code" part showing the actual elkpy call(s) (e.g. `Calculation(...)`, the
relevant `get_*()`) that exercise the formalism, so the physics and the API surface stay tied together.
Keep both in sync with the code in the same change — don't defer either to a follow-up.

## README and notebook style

Whenever the user gives style feedback on `README.md`/`notebooks/` (or documentation
style generally) — a correction, a preference, a "make it more like X" — record it
durably in this section (or add a new section here) as part of that same change, not
just apply it to the current diff and let it lapse next time. This section is itself
the product of that process (see git log) and is the standing reference to keep
current, not a one-time writeup.

`README.md` and `notebooks/` follow [`pyqula`](https://github.com/joselado/pyqula)'s style
(the same physics-code-lineage project this one borrows its `Structure`/`Calculation`
object-model naming from) — physics-first, not an API/engineering writeup:

- **README**: pyqula's `SUMMARY`/`INSTALLATION`/`FUNCTIONALITIES`/`EXAMPLES` section
  structure (all-caps `#`-level headers). `FUNCTIONALITIES` is a short bullet list per
  category, each bullet a physics statement with its defining formula where one is
  illuminating — not a paragraph explaining how it's implemented (patch series, Fortran
  file names, caching, subprocess architecture; that belongs in `docs/design.md`, not
  the README). List elkpy's own physics — the capabilities genuinely beyond stock Elk
  (currently: per-species spin-orbit scaling, Berry curvature/Chern numbers, arbitrary-k
  eigenstates/overlaps) — before the routine wrapping of Elk's standard DFT workflow
  (energy/bands/DOS/forces/relaxation/phonons/...), which gets one condensed "also
  wraps" mention, not equal billing. `EXAMPLES` pairs a short code snippet with a real
  PNG generated from an executed notebook cell (`images/`, extracted via
  `nbformat`+`base64`, same pattern as pyqula's own `images/*.png` gallery) — never a
  hand-drawn or synthetic figure. No "Project layout"/directory-tour section — that
  reads as internal engineering documentation, not user-facing README material.
- **Notebooks** (`notebooks/`, one per feature area, table linked from the README):
  pyqula's `jupyter-notebooks/*/main.ipynb` rhythm — a one-line "This notebook shows
  how to compute X" title cell, minimal imports, then repeating
  `[markdown: formula + one clause defining symbols] → [code: 2-6 terse lines, one #
  comment per line] → [plot]`. Cut engineering context rather than compress it into
  shorter prose; where a mechanism genuinely matters to a result (e.g. a value that's
  silently wrong if you get it from the wrong place), it becomes a `#` comment on the
  line it affects, not a markdown paragraph. Every notebook runs against a real
  compiled Elk binary and is checked in with its actual output cells — the one
  exception is DFPT phonons, left unexecuted with a note on why (~11-13 min/call) and
  the command to run it yourself. Add a new notebook (and a README table row) alongside
  any new physics capability, same trigger as the `docs/physics.tex` writeup rule above.
- **LaTeX gotchas hit in practice**: matplotlib's mathtext needs braced arguments
  (`\mathbf{r}`, not `\mathbf r` — the latter raises `ParseFatalException` at render
  time, not at notebook-generation time, so it only surfaces when a cell actually
  executes); keep every inline math span's `$...$` balanced without splitting a token
  across delimiters (`$K'=-K$`, not `K$'=-$K`, which opens/closes math mid-token and
  renders garbled on both GitHub and in Jupyter); don't cram two separate relations
  into one display equation ending in a trailing comma that runs into unrelated prose
  on the next line — end a display equation cleanly and give the second relation its
  own sentence.

## Core constraint: isolate changes to vendored Elk source, don't avoid Fortran

Elk's own Fortran source will be vendored into this repository (not treated as an installed system
dependency), so that the Python interface has a fixed, buildable copy to target. Because upstream Elk
is expected to be swapped for a newer release in the future, **changes must stay isolated** — Fortran
itself is a fine implementation choice for new capability; what's constrained is how it touches the
vendored tree:

- `vendor/elk/` always stays byte-for-byte what was downloaded from upstream — never edited directly,
  including `make.inc`. All building and any Fortran changes happen from a separate out-of-tree copy
  (see `docs/design.md` §8).
- Prefer Elk's existing export tasks (matrix elements, wavefunction/Wannier90 export, `STATE.OUT`
  post-processing — see `docs/design.md` §8) for new physics when they're sufficient; this is cheaper
  and carries zero Fortran risk, not a mandate to avoid Fortran altogether.
- When new Fortran is genuinely needed, prefer additive new files (new modules/subroutines) over
  editing existing upstream files. When hooking into existing control flow is unavoidable (e.g. the
  task dispatch in `elk.f90`), keep the edit to the smallest possible footprint, clearly marked, and
  track it as one hunk in a maintained patch series applied to the build copy — never committed as a
  direct change to `vendor/elk/`.
- New functionality should live in Python or in clearly separated new Fortran files rather than being
  folded into Elk's existing modules, so the patch series stays small and easy to re-evaluate against a
  new upstream version.

## Commands

- Build Elk out-of-tree (copies `vendor/elk/` to `build/elk/`, applies `patches/*.patch` if any, drops
  in `build-config/make.inc`, builds — never touches `vendor/elk/`):
  `./scripts/build_elk.sh`. Must be run before any test/example that actually invokes Elk.
  Serial by design — `make -j` races on an implicit ordering dependency in Elk's own `src/Makefile`
  (stub files like `mpi_stub.f90`/`libxcifc_stub.f90` must compile before the modules that `use` them,
  and that isn't expressed as an explicit prerequisite upstream); this is a pre-existing upstream
  issue, not something to fix by editing `vendor/elk/`.
- Install elkpy (editable): `python3 -m pip install -e .`
- Run tests: `python3 -m pytest tests/`. Run a single test: `python3 -m pytest tests/test_calculation_si.py::test_get_bands`.
  `tests/test_calculation_*.py` are integration suites that run the real `elk` binary on bulk Si/Fe —
  they self-skip if `build/elk/src/elk` doesn't exist yet, so run `./scripts/build_elk.sh` first to
  actually exercise them. `tests/test_structure.py`'s ASE round-trip test self-skips if `ase` isn't
  installed (`pip install -e .[ase]`).
- `tests/test_calculation_si_phonons.py` (DFPT phonon dispersion/DOS) is skipped by default even with
  the binary built — confirmed ~11-13 minutes per test on a minimal 2-atom, `ngridq=(2,2,2)` grid, cost
  dominated by DFPT's per-perturbation-per-q-point work, not anything elkpy controls. Set
  `ELKPY_RUN_SLOW_TESTS=1` to actually run them.
- `build-config/make.inc` targets GNU Fortran + OpenBLAS/LAPACK + FFTW, serial (no MPI); edit it (not
  `vendor/elk/make.inc`) to change compiler/library configuration.
