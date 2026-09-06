"""Phase 0e of the Elk-to-JAX port: what one traced SCF step costs to compile and hold.

Study §6 item 0e.  Nothing here is executed: `jax.jit(f).lower(*avals).compile()` builds
the executable from abstract shapes, so the 26.8 GiB of H and S at production shapes --
more than this machine has -- is never allocated.  That is the only way to get the number
the item asks for on this hardware, and it is also what makes these tests cheap.

Memory facts are asserted exactly; compile TIMES only as ratios large enough to survive a
different machine, and only behind ELKPY_RUN_SLOW_TESTS.
"""

import os

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import elkjax  # noqa: E402
from elkjax import memory, phase0e  # noqa: E402

memory.limit_address_space(16.0)

SLOW = pytest.mark.skipif(
    not os.environ.get("ELKPY_RUN_SLOW_TESTS"),
    reason="set ELKPY_RUN_SLOW_TESTS=1 for the compile-time sweep",
)


def test_nothing_is_allocated_at_shapes_this_machine_cannot_hold():
    """The production shape compiles here, and reports 26.8 GiB of arguments."""
    before = memory.peak_rss_bytes()
    row = phase0e.measure(3000, 100)
    assert row["arguments"] == 2 * 100 * 16 * 3000 ** 2 + 8 * 3000   # H, S, and v
    assert 26.0 < row["arguments"] / memory.GB < 27.0            # CLAUDE.md's figure
    assert memory.peak_rss_bytes() - before < 2 * memory.GB


def test_the_k_axis_is_a_memory_decision():
    """The half of item 0d a CPU can settle -- its timing question needs a GPU.

    Three ways to walk the k-axis, and they differ by orders of magnitude in what has to
    be resident: `vmap` holds every k-point's intermediates, `lax.map` holds one but
    still stacks the per-k outputs, and a `lax.scan` accumulator holds nothing per-k.
    """
    scan_small = phase0e.measure(400, 4, k_axis="scan")
    scan_large = phase0e.measure(400, 16, k_axis="scan")
    map_large = phase0e.measure(400, 16, k_axis="map")
    vmap_small = phase0e.measure(400, 4, k_axis="vmap")
    vmap_large = phase0e.measure(400, 16, k_axis="vmap")
    assert scan_small["temp"] == scan_large["temp"]              # exactly flat in n_k
    assert map_large["temp"] > scan_large["temp"]                # ... stacked outputs
    assert map_large["temp"] < 1.05 * scan_large["temp"]         # ... but only just
    assert vmap_large["temp"] > 3 * vmap_small["temp"]           # linear in n_k
    assert vmap_large["temp"] > 10 * scan_large["temp"]


def test_the_production_k_loop_does_not_fit_under_vmap():
    """Measured: 0.413 GiB of temporaries under `lax.map`, 40.2 GiB under `vmap`."""
    scan = phase0e.measure(3000, 100, k_axis="scan")
    vmap = phase0e.measure(3000, 100, k_axis="vmap")
    assert scan["temp"] / memory.GB < 1.0
    assert vmap["temp"] / memory.GB > 30.0
    assert vmap["temp"] > 50 * scan["temp"]


@SLOW
def test_compile_time_is_flat_in_the_shapes():
    """Hazard K is not a wall from problem size: XLA compiles on op count."""
    small = phase0e.measure(200, 4)
    production = phase0e.measure(3000, 100)
    assert production["seconds"] < 4 * small["seconds"]


@SLOW
def test_compile_time_is_superlinear_in_unrolled_op_count():
    """... and that is exactly where hazard K does live.

    Measured: unrolled Gram-Schmidt over 8/32/64/128 columns compiles in
    0.38/3.44/6.23/34.4 s at one fixed shape, while a `lax.scan` over 2800 radial points
    costs the same as one over 700.  The design rule follows: `scan` repeated structure,
    unroll only what must be.
    """
    baseline = phase0e.measure(800, 4, n_lo=8)
    unrolled = phase0e.measure(800, 4, n_lo=128)
    assert unrolled["seconds"] > 5 * baseline["seconds"]
    assert unrolled["temp"] == baseline["temp"]        # op count, not memory

    short = phase0e.measure(800, 4, radial=700)
    long = phase0e.measure(800, 4, radial=2800)
    assert long["seconds"] < 2 * short["seconds"]      # a scan is a loop, not a tape
