# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

No Python package exists yet — this is still pre-implementation. What does exist:
`vendor/elk/` (the vendored Elk 11.0.2 source, unmodified, tracked in git), `docs/elk_manual.pdf` /
`docs/elk_manual.txt` (the official Elk manual, plain-text version pdftotext-extracted for grepping),
and `docs/design.md` (the architecture strategy for the Python interface — read it before adding code;
it covers the object model, task-coverage strategy, run-directory/caching semantics, the build/patch
mechanism for the minimal-touch constraint, and the "additional implementations" strategy). Treat
`docs/design.md` as intent, not fact once implementation starts — verify against actual code before
relying on any path or API it describes, and keep it updated as decisions change.

## Project purpose

A Python interface to Elk, an all-electron full-potential linearized augmented-plane-wave (LAPW)
density-functional-theory (DFT) code written in Fortran. On top of the interface, this project adds
extra functionality that Elk itself does not provide.

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

No Python package, lint, or test tooling exists yet for elkpy itself. Do not invent commands (e.g.
`pytest`) — check for their actual presence (`pyproject.toml`, `setup.py`, CI config) before assuming
any exist, and update this section once real tooling is added.

Elk's own build (`vendor/elk/Makefile`, driven by `vendor/elk/make.inc`) and test suite
(`vendor/elk/tests/test.sh`) exist and work standalone, but per `docs/design.md` §8, elkpy is not meant
to build or run Elk directly out of `vendor/elk/` — the planned build/patch mechanism builds from a
separate out-of-tree copy so the vendored source stays untouched.
