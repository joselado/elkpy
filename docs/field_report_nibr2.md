# Field report: driving tasks 9003/9005 on a NiBr2 monolayer spin spiral

**Source.** A separate Claude Code session (working on this user's behalf), post-processing a
converged Elk NiBr2 monolayer spin spiral on Triton: 15 Ni per magnetic period, in-plane helix,
`spinorb`, 45 atoms, patched binary at
`/scratch/work/ladovj1/apps/elkpy/build/elk/src/elk`.

**Triaged 2026-09-07.** Every item below has now been checked against the code; the verdict
is recorded inline under each one, and the substance of what was acted on lives in
`docs/design.md` §31 ("Four traps that are NOT in the Fortran" and the transverse-sampling
section), not here. The text above each verdict is the reporter's, left as written.

| Item | Verdict |
|---|---|
| 1 absolute vs relative energies | **Confirmed and fixed** — `parsers.transport._check_energy_window()` |
| 2 `occmax` default | **Confirmed and fixed** — required, `None` sentinel raises |
| 3 spin-resolved LDOS helper | **Accepted, not merged** — code received, kept at `docs/field_report_nibr2_spin_ldos.py` |
| 4 `OMP_STACKSIZE` | **Fixed, plus the half the report did not name** — `RLIMIT_STACK` too |
| 5 `ramdisk` | **Confirmed for the hand-driven path only** — elkpy's own path is immune; documented |
| 6 adopt an external ground state | **Deferred to the user** — a real workflow against a load-bearing invariant |
| 7 `plot2d` transverse aliasing | **Confirmed and documented** in both example READMEs and §31 |

Two details of the report were corrected against the vendored source while triaging, both
minor: the rejecting routine is `getevalsv`, not `getevalsy`; and `nstfv` is
`nint(chgval/2) + nempty + 1` (`init1.f90:319`), with `nempty` itself scaled as
`nint(nempty0*natmtot)` (`init1.f90:316`) — which is why `nempty 180` on 45 atoms asks for
8100 states and `nstfv` ends up clamped to the matrix size.

**The positive result, which is the context for everything else:** tasks 9003 (§30,
spin-polarised STM) and 9005 (§31, vertical transport) both worked on a 45-atom spin spiral and
**agreed with each other to 0.2% on every well-sampled harmonic** — two independent code paths
(`rhomagv` in Fortran versus the Python Gram-matrix contraction). That is a stronger end-to-end
check of §31 than anything in `tests/`, and it is worth capturing as a test fixture if the run is
still on disk. Everything below is about the edges.

---

## 1. `compute_transmission(energies=)` is absolute, `get_vertical_transport(energies=)` is relative to $E_F$

Same parameter name, adjacent layers, opposite conventions — `get_vertical_transport` quietly does
`absolute = [efermi + e for e in energies]`. Dropping to the parser level is the *documented*
workflow when driving Elk by hand, and this fails **silently**: `amplitude_weights`' smeared delta
still has exponential tails at the wrong energy, so the result is plausible small numbers rather
than an error. Cost the reporter a full re-run; caught only because the `exit_region="cell"`
cross-check came out at 1.4e0 instead of ~1e-16.

Suggested fix, cheapest first:

- have `compute_transmission` check the requested energies lie inside the exported window
  (`data["window"]` is relative, `data["efermi"]` is right there) and raise or warn — this alone
  would have caught it instantly;
- better: rename to `absolute_energies`, or accept `bias=` and do the shift.

This is the item the reporter would call a genuine silent-failure bug in the API surface.

> **Verdict: confirmed, fixed.** The convention is now pinned from the Fortran rather than
> inferred: `elkpy_transport.f90` writes `elkpy_trans_window(1:2)` to line 3 as the *relative*
> input values and selects states in `[efermi+w0, efermi+w1]` (its `e0`/`e1`), so the bound is
> `efermi + window[0] <= E <= efermi + window[1]`.
> `parsers.transport._check_energy_window()` enforces it and names the shift in the message.
> A **raise**, not a warning, for the reason given: outside the window there are no states,
> so there is no correct answer to degrade to. The kwarg was deliberately not renamed — it
> fans out to the tests, the notebook, the example and `design.md`, and the check alone closes
> the silent failure. The reporter's own guard carried a `3*broadening` margin inside the
> bounds; that is not taken, because it would refuse energies that genuinely have states, and
> `get_vertical_transport`'s `nsigma=8` padding means it never arises through the wrapper.
> Pinned by `tests/test_parsers_transport.py::test_an_energy_outside_the_exported_window_is_refused`,
> which uses this run's own $E_F$ and window, both signs of the boundary, and asserts the
> correct call does not raise.

## 2. `amplitude_weights(..., occmax=2.0)` defaults to the value that is wrong for spinors

The docstring correctly says 2 for `nspinor=1` and 1 for `nspinor=2`, and `compute_transmission`
derives it properly — but the standalone function cannot see `nspinor`, so its default silently
introduces a factor 2 for any spin-orbit run. Since the whole design point of §31 is that the
arithmetic is Python and reusable, users will call it directly.

Suggested fix: make `occmax` required, or take the parsed `data` dict.

> **Verdict: confirmed, fixed.** `occmax=None` is now a sentinel that raises with the rule
> (`2.0 if data["nspinor"] == 1 else 1.0`) in the message. Kept as a defaulted positional
> rather than made keyword-only, so every existing positional call — including the one in the
> handover module — still works; the reported failure is the wrong *value*, not the wrong
> position. Pinned by `test_amplitude_weights_refuses_to_guess_the_spin_degeneracy`, which
> also checks the two settings differ by exactly $\sqrt{2}$.

## 3. Missing spin-resolved LDOS helper on the transport path

The 9005 export carries spinor-resolved amplitudes `(nspinor, nsel, np)` and **nothing in it
depends on energy**, so one diagonalisation is enough to build $n(\mathbf r,E)$ and all three
$m_i(\mathbf r,E)$ at every bias. `parsers/transport.py` does not expose that, so the Pauli
contraction gets hand-rolled.

This matters in practice because `get_spin_stm()` needs **one Elk run per bias**, whereas this
route gives a whole $dI/dV(x,E)$ map for free. Suggested API:
`spin_ldos(data, energies, broadening) -> (n, m)`. The reporter has a working version validated
against task 9003 (`rhomagv`, a completely separate Fortran path) to 0.2% on every harmonic, and
offered to hand it over — ask before rewriting it.

> **Verdict: accepted, not merged.** The code was handed over and is at
> `docs/field_report_nibr2_spin_ldos.py`; its self-test passes here (`PYTHONPATH=src python3
> docs/field_report_nibr2_spin_ldos.py`, no Elk run needed) against the fixed
> `amplitude_weights` signature. Promoting `spin_ldos()`/`tip_image()` into
> `parsers/transport.py` is a new public API and a `physics.tex` writeup, which is a decision
> rather than a fix — so it is staged here, working, rather than taken unilaterally. The
> argument for taking it is strong: `get_spin_stm()` needs one Elk run per bias *and* per tip
> direction, each re-reading 7.3 GB of eigenvectors on a cell that size, while this route gets
> the whole $dI/dV(x,E)$ map and every tip direction out of one export because nothing in that
> export depends on energy.

## 4. The launcher should set `OMP_STACKSIZE`

`rhomagv` segfaults at ~42 GB RSS with 32 threads: an OpenMP thread-stack overflow — `SIGSEGV`,
not the cgroup's `SIGKILL`, so it reads as a crash rather than a limit.
`ulimit -s unlimited; export OMP_STACKSIZE=1G` fixes it.

`LocalLauncher._thread_pinned_env` already owns `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`MKL_NUM_THREADS`, so this is a one-line addition in exactly the right place.

> **Verdict: fixed, and the report named only half of it.** `_thread_pinned_env` now does
> `env.setdefault("OMP_STACKSIZE", "1G")` — `setdefault`, unlike the three thread counts
> beside it, which deliberately override: this one is a floor a user is entitled to raise.
> But `OMP_STACKSIZE` governs the OpenMP *worker* stacks only, so at the launcher's default
> `omp_threads=1` it is inert and the `ulimit -s unlimited` half is the one that matters.
> That is now a `preexec_fn` (`_raise_stack_limit`) raising `RLIMIT_STACK` soft-to-hard in the
> child before exec, on both `run()` and `start_session()`, best-effort so a child that cannot
> raise it still runs.

## 5. `ramdisk` interaction is undocumented and bites the reuse workflow

Elk 11 defaults `ramdisk .true.`, so a resumed ground state writes no `EVEC*.OUT` at all. Anything
downstream wanting to reuse eigenvectors needs `ramdisk .false.` — task 9003 does: `rhomagv` calls
`getevecfv`/`getevecsv` from file and does **not** re-diagonalise, which is what makes a bias sweep
cheap.

Suggested fix: a docs line at minimum; better, set it automatically for the tasks that read
eigenvectors.

> **Verdict: confirmed for the hand-driven path; elkpy's own path is already immune, so
> documented rather than changed.** `_run_resumed()` prepends task 1 in the *same* elk process
> as 9003, so the RAM disk task 1 fills is the one `rhomagv` reads — setting `ramdisk .false.`
> there would only make elkpy slower and would write `EVEC*.OUT` nobody reads. It bites when
> the ground state is converged in a separate invocation, which is exactly the report's
> workflow. Documented in `docs/design.md` §31 and in `examples/spin-stm/README.md`, where an
> `elk.in` is driven by hand. Confirmed against the vendored source: `readinput.f90:412` sets
> `ramdisk=.true.`; task 9005 is immune either way, `elkpy_transport_wf` diagonalising fresh
> at its own k-mesh.

## 6. No supported way to adopt an externally converged ground state

There is no path to point a `Calculation` at someone else's converged `STATE.OUT` without
`ensure_ground_state()` re-running task 0, so the only option is hand-writing
`.elkpy_manifest.json` with a matching `_basis_signature()`.

`get_relaxed()`'s docstring argues against seeding a manifest, and that reasoning is right *for
relaxation* — but "post-process a colleague's converged run" is a common real workflow. An explicit
`adopt_ground_state(workdir, converged=True)` with a loud caveat would be safer than everyone
forging the manifest. **Weigh this against §4's wiped-subdirectory invariant before implementing.**

> **Verdict: real, and deferred to the user rather than decided here.** The workflow is
> legitimate and the current alternative — hand-forging `.elkpy_manifest.json` to match
> `_basis_signature()` — is strictly worse, since a forged manifest is unfalsifiable while an
> explicit `adopt_ground_state()` could at least re-read the adopted `elk.in` and *check* the
> basis/functional parameters against the `Calculation`'s own rather than assert them.
> Recommendation if it is taken: adopt by **copying** `STATE.OUT` into `self.workdir` and
> writing a manifest marked `adopted`, never by pointing at someone else's directory —
> that keeps the wiped-subdirectory invariant untouched, since every `get_*` still runs in a
> fresh subdirectory of a workdir elkpy owns. It is a new public method on the ground-state
> contract, so it is the user's call, not a triage fix.

## 7. Docs: `plot2d` transverse sampling can alias the atomic lattice into low harmonics

Both `examples/spin-stm/` and `examples/vertical-transport/` invite harmonic analysis, so this
belongs in both.

In the reporter's cell both sublattices satisfy $p = 2m \pmod{15}$ for $\mathbf G = m\mathbf b_1 +
p\mathbf b_2$, and an $n_2$-point average over $\mathbf a_2$ keeps $p = 0 \pmod{n_2}$ — so atomic
weight leaks into x-harmonic $m$ whenever $2m = 0 \pmod{\gcd(15, n_2)}$. With $n_2 = 6$ the
$(m,p) = (3,6)$ component landed on the $3q$ harmonic at 3.4e-3 where the true value is 2.9e-6.

**Rule: pick $n_2$ sharing the supercell's periodicity.** Also worth stating that a single line cut
is useless here, since it shows every $(m,p)$ at harmonic $m$: theirs read 66% charge modulation at
$q$ where the correct transverse-averaged value is 1.4e-4.

> **Verdict: confirmed, documented.** Added to both `examples/spin-stm/README.md` and
> `examples/vertical-transport/README.md` as a "Reading harmonics off the map" section, with
> the aliasing condition and the measured $3.4\times10^{-3}$ against $2.9\times10^{-6}$, and to
> `docs/design.md` §31.

---

## Not elkpy's fault, but relevant to version handling

- Elk 11.0.2 computes `nstfv = int(chgval/2) + nempty + 1` where 10.2.4 had no `+1`, so **10.2.4
  eigenvector files are rejected** by `getevalsy`/`getevecfv` ("differing nstsv") while `readstate`
  accepts the 10.2.4 density with only a version warning.
- The `nempty` block is **scaled internally**: `nempty 180` gives `nstsv` 10768, not 1110.
- Clean workaround for both: tasks 1 + `maxscl 1` + `ramdisk .false.` to regenerate eigenvectors
  from the old density.

This is `spec.py`-shaped knowledge — if it is acted on, it belongs there rather than in prose.

## Attribution note from the reporter

Items 1 and 2 were their bugs to write, not elkpy's to have. They were reported because both are
cheap for the library to make impossible, and both fail with no error message. No action was
requested; triage as you see fit.
