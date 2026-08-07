"""Locating the built Elk binary and the vendored species files.

See docs/design.md #8: vendor/elk/ is never built in place. The expected
binary lives at build/elk/src/elk, produced by copying vendor/elk/ into
build/elk/, applying build-config/make.inc, and running `make` there.
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
            f"(see docs/design.md #8), or set ELKPY_ELK_BIN."
        )
    return binary


def resolve_species_path(path=None):
    species_path = Path(path) if path is not None else default_species_path()
    if not species_path.is_dir():
        raise FileNotFoundError(f"Species directory not found at {species_path}.")
    return species_path
