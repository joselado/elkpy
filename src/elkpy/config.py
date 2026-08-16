"""Locating the built Elk binary and the vendored species files.

See docs/design.md #8: vendor/elk/ is never built in place. The expected
binary lives at build/elk/src/elk, produced by copying vendor/elk/ into
build/elk/, applying build-config/make.inc, and running `make` there.
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _is_source_checkout(root):
    """Does `root` look like the elkpy source tree (as opposed to a venv's
    site-packages parent)? Both defaults below are computed relative to this
    file, which only lands on the repository root for a source checkout or an
    editable install."""
    return (root / "vendor" / "elk").is_dir() and (root / "scripts").is_dir()


def _layout_hint():
    """Extra diagnosis appended to the "not found" errors below, for the one
    failure mode whose symptom is otherwise baffling: a non-editable
    `pip install .` puts this module in site-packages, so parents[2] resolves
    to the venv's lib/pythonX.Y and every default path below is nonsense
    (e.g. .../lib/python3.14/vendor/elk/species). elkpy needs the vendored Elk
    tree and a binary built from it, neither of which pip installs, so an
    editable install (or explicit environment variables) is required."""
    if _is_source_checkout(_REPO_ROOT):
        return ""
    return (
        f"\n\nNote: {_REPO_ROOT} does not look like an elkpy source checkout "
        f"(no vendor/elk/ or scripts/ there), so this path was almost certainly "
        f"derived from a non-editable installation. elkpy needs the vendored Elk "
        f"source and a binary built from it, which pip does not install. Use "
        f"`pip install -e .` from a checkout, or set ELKPY_ELK_BIN and "
        f"ELKPY_SPECIES_PATH explicitly."
    )


def repo_root():
    return _REPO_ROOT


def default_elk_binary():
    env = os.environ.get("ELKPY_ELK_BIN")
    if env:
        return Path(env)
    return _REPO_ROOT / "build" / "elk" / "src" / "elk"


def default_species_path():
    env = os.environ.get("ELKPY_SPECIES_PATH")
    if env:
        return Path(env)
    return _REPO_ROOT / "vendor" / "elk" / "species"


def resolve_elk_binary(path=None):
    binary = Path(path) if path is not None else default_elk_binary()
    if not binary.is_file():
        raise FileNotFoundError(
            f"Elk binary not found at {binary}. Build it first: copy vendor/elk/ "
            f"to build/elk/, drop in build-config/make.inc, and run `make` there "
            f"(see docs/design.md #8), or set ELKPY_ELK_BIN." + _layout_hint()
        )
    return binary


def resolve_species_path(path=None):
    species_path = Path(path) if path is not None else default_species_path()
    if not species_path.is_dir():
        raise FileNotFoundError(
            f"Species directory not found at {species_path}." + _layout_hint()
        )
    return species_path
