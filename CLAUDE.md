# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is a fresh, empty project. Nothing has been implemented yet — no source layout, build
system, or tests exist on disk. Treat the "Planned structure" section below as intent, not fact: verify
against the actual repository contents before relying on any path or command, and update this file as
real structure lands.

## Project purpose

A Python interface to Elk, an all-electron full-potential linearized augmented-plane-wave (LAPW)
density-functional-theory (DFT) code written in Fortran. On top of the interface, this project adds
extra functionality that Elk itself does not provide.

## Core constraint: minimize changes to vendored Elk source

Elk's own Fortran source will be vendored into this repository (not treated as an installed system
dependency), so that the Python interface has a fixed, buildable copy to target. Because upstream Elk
is expected to be swapped for a newer release in the future, **the vendored Elk source should be
modified as little as possible**:

- Prefer wrapping Elk (subprocess calls, file-based I/O against its input/output files, or a thin
  binding layer) over editing its Fortran.
- If a change to the Fortran source is unavoidable, keep it small, isolated, and clearly marked (e.g.
  a small, well-documented patch file rather than scattered inline edits), so it can be re-applied or
  reassessed against a new upstream version with minimal effort.
- New functionality — the "additional implementations" this project provides beyond stock Elk — should
  live in Python (or in clearly separated new files) rather than being folded into Elk's existing
  modules.
- Before touching any file under the vendored Elk source tree, prefer a design that avoids the edit
  entirely; if you do need to change it, note in the commit/PR what upstream file and behavior is being
  patched and why, so the patch is easy to re-evaluate on an Elk version bump.

## Commands

No build, lint, or test tooling exists yet. Do not invent commands (e.g. `pytest`, `make`) — check for
their actual presence (`pyproject.toml`, `setup.py`, `Makefile`, CI config) before assuming any exist,
and update this section once real tooling is added.
