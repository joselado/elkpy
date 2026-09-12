# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository. It is loaded into every session, so it holds **rules and routing**, not narrative:
the per-feature record lives in `docs/`, and the pointers below say which file to open for
which task.

## Which document to read, and when

| Doing this | Read first |
|---|---|
| Anything touching an existing elkpy capability | `docs/status.md` (feature ledger: what is implemented and **how far it is verified**) |
| Adding/changing physics, or checking a convention | `docs/design.md` §N and `docs/physics.tex` Part N for that feature |
| Planning new work | `docs/roadmap.md`, `docs/design.md` |
| Picking up a cold session | `docs/continue_here.md` |
| Known bugs in the task-family wrappers | `docs/review_findings.md` (17 findings, severity-ranked) |
| **Any JAX work** | `docs/jax_port_status.md` — **including its memory/CPU discipline, before running a single JAX command** |
| Elk input blocks / task codes / filenames | `src/elkpy/params.py`, `src/elkpy/spec.py` (data, not prose) |

Check `src/elkpy/` directly rather than assuming the docs describe current code; update both
as they diverge.

## Communication style

The user is a physicist with some Python background, not a software engineer. Write to that
audience: simple English, short sentences, concrete statements, no verbosity. Physics vocabulary
is the right register; keep programming jargon at an everyday-Python level and explain anything
deeper. Lead with the result, not with the narration of how it was obtained.

## Project purpose

A Python interface to Elk, an all-electron full-potential linearized augmented-plane-wave (LAPW)
density-functional-theory (DFT) code written in Fortran. On top of the interface, this project adds
extra functionality that Elk itself does not provide.

## Project status

Roadmap Tiers 1-3 are implemented, plus a thirteen-entry Fortran patch series (`patches/`) adding
physics Elk does not have. **143 of the 146 live task codes sit behind a named method (97.9%);
`Calculation` exposes 116 `get_*` methods.** Full narrative, with the verification evidence for
each row: `docs/status.md`.

**Adding a capability: one row in this table, the verification narrative in `docs/status.md`, the
design reasoning in `docs/design.md` §N, the physics in `docs/physics.tex`. No new prose in this
file** — this section is a routing table and grew to 940 lines once by not being one.

| § | Capability | Patch | Verification |
|---|---|---|---|
| 12 | Per-species spin-orbit scaling (`soc_scale=`) | 0001 | verified vs binary |
| 13 | Berry curvature, Chern number; arbitrary-k path | 0002 | verified (h-BN K/K′ antisymmetry) |
| 14 | `eigenstate_session()`, eigenstates/overlaps at any k | 0003 | verified |
| 15 | Quantum geometric tensor (metric + curvature) | — | verified (Löwdin fix, centered stencil) |
| 16 | Atom-projection operators | 0004 | verified vs task 21 |
| 17 | Spin operators $S_x,S_y,S_z$ | — | verified (WSe2 spin-valley locking) |
| 18 | l-resolved (s/p/d/f) projection | 0005 | verified vs task 21 |
| 19 | Orbital angular momentum $L_x,L_y,L_z$ | 0006 | verified (WSe2 $L_z$ valley locking) |
| 20 | 2D $Z_2$ by Wannier-charge-center pumping | — | verified (graphene, bismuthene) |
| 21 | 3D $(\nu_0;\nu_1\nu_2\nu_3)$ | — | **implementation sound, cesium result RETRACTED** — see §23 |
| 22 | Momentum matrix elements, Kubo geometry, dichroism | 0007 | verified; **fixed a global Berry-sign error** |
| 23 | Inversion parity, Fu-Kane indicator | 0008 | verified (graphene; Bi2Se3 gives the literature $(1;000)$) |
| 24 | Dielectric function, circular absorption | — | verified vs Elk task 121 |
| 25 | k·p effective-mass sum rule | — | verified vs task 25 (+3.1%, converging from above) |
| 26 | Spin Berry curvature / spin Hall | 0009 | verified (graphene, same sign at both valleys) |
| 27 | Potential, ELF, MOKE wrappers | — | verified |
| 28 | Rotation-eigenvalue symmetry indicators | 0010 | **operator verified; the corner charge is NOT** |
| 29 | Anisotropic exchange tensor (four-state mapping) | — | **Python pinned only; NOT validated end-to-end** |
| 30 | Spin-polarised STM (Tersoff-Hamann) | 0011 | verified; found an upstream `wkpt` double-count |
| 31 | Vertical tunnelling transport | 0012 | verified; **magnetic substrate has no physics test** |
| 32 | Six task-family mixins + `params.py` input table | — | **uneven, labelled per method** — many are format-derived, i.e. untested |
| 33 | LAPW/ground-state export for the JAX port | 0013-0024 | see `docs/jax_port_status.md` |
| 34 | `STATE.OUT` format + conventions; `spec.ELK_VERSION`, `parse_charges()` | — | verified (`tests/fixtures/` `h_sc`, `c_diamond`, `sic_zb`; no binary needed) |
| 35 | Momentum-resolved tunnelling Fermi surface (planar tip) | 0025 | verified (**exactly** equals §31's map integrated over the tip plane) |

**What is open, and must not be quietly asserted as done**

- §21's cesium $(0;000)$ retraction: the 3D six-plane WCC sweep at a practical mesh is
  under-resolved, not wrong in its algebra. Bi2Se3 by WCC is still unexplained; by parity it is right.
- §28's corner charge $Q^{(3)}$: two candidate conventions, both self-consistent, differing by
  $e/3$. The concrete unfinished fix is pinning the rotation sense with §19's $L_z$ on an
  on-axis atom (h-BN has none).
- §29's NiO nine-component sweep (~60 CPU-hours) has never been run;
  `check_constraint()` has never fired on a completed run. `mommtfix` constrains spin only —
  insufficient for a $j_{\rm eff}=1/2$ system.
- The graphene `soc_scale=3000` fixture is **metallic**; the tested window is a legitimately
  gapped 6-band group, so §20/§23 stand, but §20's "spans every occupied valence band" overstates it.
- Tasks 2, 201, 271 have no named method (restarts needing a previous run's files, which the
  wiped-subdirectory invariant cannot supply).

**Traps that will bite, all measured rather than theorised**

- `tshift=False` is **mandatory** for §28 and §31: Elk otherwise moves the origin onto the
  inversion centre while plotting planes stay in the input frame.
- Every `get_*` runs in a **wiped-clean subdirectory**; some tasks resume from stale output
  instead of erroring. Never reuse a run directory.
- Elk's default `epsengy` (1e-4 Ha) exceeds the whole exchange-coupling signal; elkpy uses 1e-8.
- A one-species Elk run cannot fail two `STATE.OUT` reader bugs: `natmtot` is the sum of
  `natoms` over species, and the rows past `nrmt(is)` are uninitialised buffer. `nrmt` is
  set by periodic-table row (197/297/397/497) and `checkmt` never changes it, so two
  species from the SAME row are equally blind — `tests/fixtures/sic_zb` exists for this.
- Occupied-band counts come from `EIGVAL.OUT` occupations, never from an electron count (core
  electrons are not among the `nstsv` valence bands). The `sum(occ > 0.5)` idiom reads only the
  FIRST k-point and is valid only when the filling is k-independent.
- Window a whole degenerate group, not a single band — a band can diverge against a *neighbouring
  occupied* band even when the window boundary is gapped.
- `parsers.optical`'s `directions` is **Cartesian**; `get_berry_curvature()`'s identically named
  argument is **reciprocal-lattice**. Same shape of trap one layer down:
  `parsers.transport.compute_transmission()`'s `energies` are **absolute**, while
  `get_vertical_transport()`'s are **relative to $E_F$** — it now raises rather than returning a
  plausible map at the wrong energy, and `amplitude_weights()`'s `occmax` is required for the
  same reason (its old 2.0 default was a silent factor of two on every spinor run).
- One Berry-phase convention everywhere (Xiao-Chang-Niu), set in `parsers.berry._berry_phase()`
  alone. It differs in sign from pyqula's, deliberately. Berry curvature is in **Bohr²**.
- Matrices from two separate diagonalisations do not share a basis inside a degenerate multiplet;
  anything combining operators must come from ONE diagonalisation.
- §35's vacuum weighting has a **`rgkmax` floor, and the large-|k| pocket hits it first** — the
  very pocket whose suppression is the result. Measured on NbSe2 at `rgkmax=7`: past ~3.5 Å the
  K weight flattens onto ~1e-6 and the *apparent* contrast then FALLS (131x at 4.5 Å, 4.6x at
  6.5 Å). Sweep the height and check both pockets are still straight on a log plot before
  quoting any ratio. The check that a run IS clean is two distances: the pocket weights must
  give a κ each, and those κ must predict how the contrast grew (measured 4.17 vs 4.17).
- `soc_scale=` requires `spinorb=True` and a binary built from the patch series.
- `patch` exits 0 on a FUZZY apply — so "it applied" is not "it applied cleanly". There is
  no CI; grep the `patch` output for `fuzz` by hand after any `vendor/elk/` bump.

## Architecture

One line per module. `docs/design.md` carries the reasoning; the parenthetical warnings below are
the parts that cause silent wrong answers if ignored.

- `structure.py` — `Structure`: `avec` (Bohr) + species; each atom a position or a
  `(position, bfcmt)` pair. `from_ase()`/`to_ase()` convert Angstrom/Cartesian (optional dep).
- `calculation.py` — `Calculation`: owns one run directory and the ground-state-defining
  parameters (`xc`, `spinpol`, `spinorb`, `soc_scale`, `rgkmax`, `ngridk`, `extra_blocks`).
  `get_*` methods block on real Elk subprocesses. `ensure_ground_state()` reuses a prior task-0
  run only if `.elkpy_manifest.json` shows the basis/structure/functional parameters and the binary
  identity unchanged; sampling parameters may differ. **Every other `get_*` runs via
  `_run_resumed()` in its own wiped-clean subdirectory** — load-bearing, not tidiness: some tasks
  treat prior output as "already done" and resume from it. `class Calculation(*ALL_MIXINS)`;
  `_ndmag()` and its block helpers live here, not in a mixin, so the `init0.f90` rule has one
  transcription.
- `tasks/` — one mixin per Elk task family (`groundstate`, `spectra`, `optics`, `phonons`,
  `magnetism_manybody`, `params`), composed in `tasks/__init__.py`'s `ALL_MIXINS`. No mixin holds
  state, defines `__init__`, or imports `..calculation`; `Calculation`'s own methods come first in
  the MRO. Verification is **labelled per method** — treat a format-derived one as untested.
- `spec.py` — version-coupled data (152 task codes, 190 filenames, 22 templates), each entry
  cross-checked against `vendor/elk/src/`. An Elk version bump should mean editing this one file.
- `params.py` — all 315 `case(...)` branches of `readinput.f90` as data, plus validator, exact-text
  renderer and `describe`/`search`/`categories`. Its completeness test fails in BOTH directions.
- `inputfile.py` — generic `elk.in` block writer and `read_blocks()` reader.
- `launcher.py` — `LocalLauncher.run()` (blocking subprocess) and `start_session()` (non-blocking
  `Popen` pipes, for task 9002). Refuses `nprocs > 1` (the build is serial). Pins BLAS threads and
  holds an flock semaphore, `ELKPY_MAX_CONCURRENT` (default 4).
- `session.py` — `EigenstateSession`: the interactive task-9002 subprocess. Queries:
  `EIGENSTATES`, `OVERLAP`, `PROJECTION` (§16), `ORBITAL` (§18), `MOMENTUM` (§22, the one query
  taking no band window), `PARITY` (§23), `SYMMETRY`/`SYMLIST` (§28), `LAPW` (§33).
- `exchange.py` — orchestration for four-state energy mapping (§29), including `check_constraint()`.
- `parsers/` — one module per output family (`info`, `totenergy`, `band` — reused for phonon
  dispersion, `dos`, `forces`, `geometry`, `effmass`, `volumetric`, `eigenstates`, `stm`,
  `symmetry`). Six do arithmetic rather than parsing, deliberately kept out of Fortran so they are
  unit-testable without an Elk run: `berry` (Wilson loop, Chern number, and the single
  `_berry_phase()` that sets the sign convention), `wilson` ($Z_2$/WCC), `optical` (dichroism,
  Kubo geometry), `quantum_geometry`, `exchange`, `transport`.
- `config.py` — locates the `elk` binary (`ELKPY_ELK_BIN`) and species directory; diagnoses a
  non-editable `pip install .`.

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

When a real material's crystal `Structure` is needed (lattice vectors + atomic positions), prefer
pulling it from an actual structure database/file (a CIF from the Crystallography Open Database or
Materials Project, a published paper's POSCAR/Quantum Espresso input, etc.), loaded via ASE
(`Structure.from_ase()`) where possible, over hand-deriving it from reported lattice
parameters/Wyckoff positions. A hand conversion (e.g. hexagonal-to-rhombohedral primitive vectors from
a,c and a Wyckoff z-parameter) is an extra, error-prone derivation step even when every input number is
correct — hit for real building Bi2Se3's rhombohedral cell, where a wrong hand-derived transformation
matrix gave physically nonsensical bond lengths (~11 Å instead of ~3 Å) despite starting from correct
literature z-parameters; re-deriving it wasted significant real-DFT compute chasing a structure that
was never right. When a database/file source isn't available or a hand derivation is unavoidable,
numerically verify the result (e.g. actual computed bond lengths/layer spacing against known physical
values) before running any DFT on it, not just before trusting the final answer.

Whenever picking which real material to use for something (a demonstration, a test, choosing between
candidate structures), or trying to understand something about a material in terms of its structure
(crystal symmetry, distortion geometry, why a particular structure does or doesn't have a given
property), ask Fable (the `fable` model, e.g. via `Agent(..., model="fable")`) rather than relying
solely on your own judgement or a general-purpose research agent.

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
  reads as internal engineering documentation, not user-facing README material. Every
  `FUNCTIONALITIES` bullet ends with a `[[notebook]](notebooks/NN_name.ipynb)` link to
  the notebook that demonstrates it — same inline-link-per-bullet pattern pyqula's own
  README uses (see pyqula's `FUNCTIONALITIES` section) — so a reader goes straight from
  the one-line physics claim to the worked example, not just from a separate summary
  table at the bottom of the page.
- **Notebooks** (`notebooks/`, one per feature area, table linked from the README):
  pyqula's `jupyter-notebooks/*/main.ipynb` rhythm — a one-line "This notebook shows
  how to compute X" title cell, minimal imports, then repeating
  `[markdown: formula + one clause defining symbols] → [code: 2-6 terse lines, one #
  comment per line] → [plot]`. Cut engineering context rather than compress it into
  shorter prose; where a mechanism genuinely matters to a result (e.g. a value that's
  silently wrong if you get it from the wrong place), it becomes a `#` comment on the
  line it affects, not a markdown paragraph. Concretely, this means no standalone
  "verify the result" cell — a Hermiticity check, an identity/partition check
  (`sum_alpha P_alpha + P_interstitial = 1`), an eigenvalue-sign check — sitting between
  the formula and the headline calculation with no plot of its own: that correctness
  check already lives in `tests/` and is asserted in the prose of `docs/design.md`/
  `docs/physics.tex`, so repeating it in the notebook is exactly the kind of engineering
  context pyqula's own notebooks don't carry (see e.g. `jupyter-notebooks/08_chern_insulator/
  main.ipynb`: Hamiltonian → bands/curvature/Chern number → plot, nothing else). A
  notebook cell should either feed the next cell or feed a plot; if it does neither,
  cut it. When a quantity is naturally a function of k (or of some other physically
  meaningful axis), prefer showing it that way over a bar chart of one or two isolated
  numbers — e.g. a band structure colored by an operator's expectation value (a
  spin/orbital-texture plot), not `ax.bar(["K", "K'"], [...])` — even when the
  headline physics claim is about just two points; a categorical comparison across
  atoms/orbital channels (not indexed by k) is the one case where a grouped bar chart
  is still the right call (see [[feedback_notebook_plots_not_bar_charts]] in the
  auto-memory). Every notebook runs against a real compiled Elk binary and is checked
  in with its actual output cells — the one exception is DFPT phonons, left unexecuted
  with a note on why (~11-13 min/call) and the command to run it yourself. Add a new
  notebook (and a README table row + `FUNCTIONALITIES` link) alongside any new physics
  capability, same trigger as the `docs/physics.tex` writeup rule above.
- **LaTeX gotchas hit in practice**: matplotlib's mathtext needs braced arguments
  (`\mathbf{r}`, not `\mathbf r` — the latter raises `ParseFatalException` at render
  time, not at notebook-generation time, so it only surfaces when a cell actually
  executes); keep every inline math span's `$...$` balanced without splitting a token
  across delimiters (`$K'=-K$`, not `K$'=-$K`, which opens/closes math mid-token and
  renders garbled on both GitHub and in Jupyter); don't cram two separate relations
  into one display equation ending in a trailing comma that runs into unrelated prose
  on the next line — end a display equation cleanly and give the second relation its
  own sentence.


## JAX port (Workstream B)

A research project justified by **differentiability, not by the GPU** (SIRIUS already does FP-LAPW
on CUDA/ROCm). **Nothing about it is a plan of record; Phase 0 is designed to kill it, not to start
it.** Code lives in `src/elkjax/` — a *sibling* package to `elkpy`, deliberately not `elkpy.jax`,
so elkpy's fast unit tests never acquire a `jax` dependency. Install `pip install -e .[jax]`;
tests are `tests/test_jax_*.py` and self-skip without jax.

Status: Phase 0 closed (it did not kill the project), Phase 1 done through §1m, Phase 2 done as
forward checks (§§2a-2k, one open question in §2k), Phase 3 has its **forward** criterion (§§3a-3c:
the Kohn-Sham loop closes, and `elkjax.driver.run()` takes a `Calculation` **from its `elk.in`
alone** — patch 0024's task 9006 stops Elk at the top of its own first iteration — and converges to
Elk's total energy and Fermi level; the remaining 3.6e-4 Ha on bulk Si is entirely the frozen core)
and none of its gradient ones. **Full narrative, phase tables, every measured tolerance, and the compute discipline:
`docs/jax_port_status.md`** — plus `docs/jax_port.md` (the study) and
`docs/jax_port_phase{0,1,2,3}.md` (the logs). `docs/continue_here.md` §3 is the cold start.
**New measurements go in `docs/jax_port_phaseN.md` and `docs/jax_port_status.md`; only a rule that
must hold even if those files are never opened belongs here.**

**Read `docs/jax_port_status.md`'s "Memory and CPU discipline" before running any JAX here.**
The five rules that must hold even if you never open it:

- **Never allocate production shapes on this box** (12 cores, 39 GB): $H+S$ at
  $n_{\rm mat}\approx3000$, $n_{\bf k}\approx100$ is 26.8 GiB before eigenvectors. Execute at
  $n\le1500$, $n_{\bf k}\le4$ and measure the exponent; for compile-cost questions **lower and
  compile** (`.lower(...).compile()`, `.memory_analysis()`) rather than executing.
- **Cap every script and test** with `elkjax.memory.limit_address_space()` (default 16 GB).
- **`taskset -c 0-3` every JAX invocation.** `OMP_NUM_THREADS` does NOT govern XLA — measured, one
  matmul spawns 40 threads under it. `launcher.py`'s semaphore sees none of this.
- **`lax.map`/`lax.scan` over the k-axis is the default; `vmap(eigh)` is opt-in** (0.411 GiB vs
  40.2 GiB of temporaries at production shape). A **Python** loop over k (or over atoms, or over
  $\ell$) is the same mistake one level up: it unrolls into the compiled program, so XLA compile
  time and HLO size grow linearly with the loop count. Measured on the SCF step (§3d): 4.5 s of
  compile per k-point, and 47 s of 93 s spent on one `lax.scan` emitted 26 times. Batching both
  made the compiled program independent of the k-mesh — same 86,839 HLO lines at 3, 8 and 16
  k-points. **Padding to a common shape is what makes that possible, and padding a matrix that
  gets Cholesky-factorised needs a diagonal, not a mask** — with the shift measured, since it
  enters the norm the eigensolver works on (1e3 Ha costs 8e-15 Ha, 1e6 costs 6e-10).
- **Import `elkjax` first** — `jax_enable_x64` must be set before the first array exists. All of
  this work is float64/complex128; an all-electron spectrum spans ~2500 Ha.

Working rules distilled from what Phase 0/1 measured (each is a measurement, not a style opinion):

- **A green gradient test does not validate a transcription** — a constant factor commutes with
  differentiation. Put a *forward* check beside every AD check; three separate bugs here were
  invisible to the gradient and obvious to the forward comparison.
- **Always compare forward-mode against reverse-mode**: for scalar-in/scalar-out they must agree
  exactly, so disagreement is proof and costs nothing.
- **Use an analytic reference, not finite differences, wherever a degeneracy is in play**; and test
  along general directions with the protecting symmetry broken (a single real diagonal direction,
  or a symmetry-respecting perturbation, makes a wrong rule look right).
- **`grad(grad)`, never `jax.hessian`** — a `custom_vjp` cannot be forward-differentiated. Second
  order needs `sign_projector` (Newton-Schulz, no eigensolve); the safe-$K$ rule is first-order only.
- **`lax.scan` the Newton-Schulz tape**, not an unroll (230x the HLO instructions at 80 steps).
- **Recompute $\kappa(O)$ per run** — it is set by `rgkmax`, not matrix size; the study's §8b
  Cholesky estimate is uninformative and must not set a threshold.
- **Freezing a quantity "at fixed potential" is not free once the potential actually moves.**
  What may be held fixed is the physically frozen thing, not whichever array Elk happens to
  export: the core's contribution is $T_{\rm core}$ (`engykncr`), NOT the core eigenvalue sum
  (`evalsumcr`), because `energy.f90` builds the kinetic energy as
  $\Sigma_\varepsilon-\int\rho v_{cl}-\int\rho v_{xc}$ and the two halves must sit at the
  same potential. Measured: 2.0 Ha the wrong way, 3.6e-4 Ha the right way. Invisible on any
  run that starts at Elk's converged answer.
- **Never unroll an SCF to differentiate it**: unrolled Anderson's forward value is fine while its
  gradient is wrong by $10^{17}$. Use the implicit route — which also means "implicit agrees between
  mixers" proves nothing.
- `tol` is inert for smeared occupations and load-bearing for a hard window.
- **Elk mixes the potential in the MIDDLE of its own iteration**, so any two exported arrays
  written on opposite sides of `gndstate.f90`'s `mixerifc` call disagree by the last mixing step —
  measured four times now (§1k's `haa`, which patch 0015's `genapwlofr` call fixed; §2h's
  density; §2j's two `evecfv`; §3b's `vsig`). When two Elk arrays disagree at roughly `epspot`, that is what it is;
  check where each is written before hunting a transcription bug.

## Commands

- Build Elk out-of-tree (copies `vendor/elk/` to `build/elk/`, applies `patches/*.patch` if any, drops
  in `build-config/make.inc`, builds — never touches `vendor/elk/`):
  `./build_elk.sh`. Must be run before any test/example that actually invokes Elk.
  Serial by design — `make -j` races on an implicit ordering dependency in Elk's own `src/Makefile`
  (stub files like `mpi_stub.f90`/`libxcifc_stub.f90` must compile before the modules that `use` them,
  and that isn't expressed as an explicit prerequisite upstream); this is a pre-existing upstream
  issue, not something to fix by editing `vendor/elk/`.
- Install elkpy (editable): `python3 -m pip install -e .`
- Run tests: `python3 -m pytest tests/`. Run a single test: `python3 -m pytest tests/test_calculation_si.py::test_get_bands`.
  `tests/test_calculation_*.py` are integration suites that run the real `elk` binary on bulk Si/Fe —
  they self-skip if `build/elk/src/elk` doesn't exist yet, so run `./build_elk.sh` first to
  actually exercise them. `tests/test_structure.py`'s ASE round-trip test self-skips if `ase` isn't
  installed (`pip install -e .[ase]`).
- `tests/test_calculation_si_phonons.py` (DFPT phonon dispersion/DOS) is skipped by default even with
  the binary built — confirmed ~11-13 minutes per test on a minimal 2-atom, `ngridq=(2,2,2)` grid, cost
  dominated by DFPT's per-perturbation-per-q-point work, not anything elkpy controls. Set
  `ELKPY_RUN_SLOW_TESTS=1` to actually run them.
- `build-config/make.inc` targets GNU Fortran + OpenBLAS (which bundles LAPACK — do NOT re-add
  `-llapack`, it is absent in Spack-built OpenBLAS and redundant where it exists) + FFTW3 double and
  single precision, serial (no MPI); edit it (not `vendor/elk/make.inc`) to change compiler/library
  configuration. Those are the workstation defaults: `build_elk.sh` link-tests them and, when they
  fail, loads `${ELKPY_MODULES:-openblas fftw}`, adds `-Wl,-rpath` for every `LIBRARY_PATH` entry (so
  the binary still runs in a batch job with no modules loaded), and swaps `-march=native` for
  `-march=haswell -mtune=generic` whenever an environment-module system is present — a login node is
  routinely newer than the compute nodes, so `native` builds fine and then `SIGILL`s. Overrides:
  `ELKPY_F90_LIB` (verbatim, and a link failure is fatal), `ELKPY_MARCH`, `ELKPY_MODULES`. The
  resolved values are appended to the copied `build/elk/make.inc`, which is therefore a
  self-contained record of what was built. See `docs/design.md` §8.
