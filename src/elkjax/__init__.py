"""JAX side of the Elk port (Workstream B) — Phase 0 only.

This is a *sibling* package to :mod:`elkpy`, not a submodule of it, and importing it
must never become a side effect of importing ``elkpy``: Phase 0 of ``docs/jax_port.md``
§6 is explicitly "no Elk code", and elkpy's fast unit suite must not acquire a ``jax``
import (JAX costs ~1 s of import time and spawns a 40-thread XLA pool on this box —
measured; see CLAUDE.md's "JAX port" section for the CPU/memory rules this work runs
under).

Everything here is float64/complex128.  An all-electron LAPW spectrum spans ~2500 Ha,
so float32 is not an option, and ``jax_enable_x64`` must be set before the first array
is created — this module does it on import, which is why it should be imported before
any array-producing JAX call.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

from . import memory, reference, projector, fixedpoint, scftoy  # noqa: E402

__all__ = ["memory", "reference", "projector", "fixedpoint", "scftoy"]
