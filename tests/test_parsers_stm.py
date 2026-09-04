"""Unit tests for the spin-polarised STM output parsers (no Elk run).

parse_plot2d reads the generic plot2d writer (src/plot2d.f90), shared with
upstream tasks 62/162; parse_stm_dos reads elkpy_stm.f90's own
ELKPY_STMDOS.OUT. The check with teeth here is the grid ORDER: plotpt2d.f90
runs the first plotting vector's index in the inner loop, so a column
reshapes as (n2, n1). Getting that backwards transposes every image without
changing a single number, which no numerical check downstream can catch --
so it is pinned directly against a file whose values encode their own
indices.
"""

import numpy as np
import pytest

from elkpy.parsers import stm, volumetric


def write_plot2d(path, n1, n2, nf=1):
    """Write a plot2d-format file whose values encode their own (i1, i2),
    laid out exactly as src/plotpt2d.f90 orders the points."""
    with open(path, "w") as fh:
        fh.write(f"{n1:6d}{n2:6d} : grid size\n")
        for i2 in range(n2):
            for i1 in range(n1):
                vals = "  ".join(f"{100 * i1 + i2 + 10 ** jf:.10E}" for jf in range(nf))
                fh.write(f"{0.5 * i1:.10E}  {0.25 * i2:.10E}  {vals}\n")


def test_parse_plot2d_grid_order(tmp_path):
    path = tmp_path / "STM2D.OUT"
    write_plot2d(path, n1=5, n2=3)
    points, values, grid = volumetric.parse_plot2d(path)
    assert grid == (5, 3)
    assert values.shape == (15,)
    image = values.reshape(grid[1], grid[0])
    # value = 100*i1 + i2 + 1, so image[i2, i1] recovers both indices
    for i1 in range(5):
        for i2 in range(3):
            assert image[i2, i1] == pytest.approx(100 * i1 + i2 + 1)
    assert points[1, 0] == pytest.approx(0.5)  # second point is i1=1, i2=0
    assert points[1, 1] == pytest.approx(0.0)


def test_parse_plot2d_multiple_functions(tmp_path):
    path = tmp_path / "ELKPY_STM2D.OUT"
    write_plot2d(path, n1=4, n2=2, nf=3)
    points, values, grid = volumetric.parse_plot2d(path, nf=3)
    assert grid == (4, 2)
    assert values.shape == (8, 3)
    # the three columns differ by 1, 10, 100 at every point
    assert np.allclose(values[:, 1] - values[:, 0], 9.0)
    assert np.allclose(values[:, 2] - values[:, 0], 99.0)


def test_parse_stm_dos(tmp_path):
    path = tmp_path / "ELKPY_STMDOS.OUT"
    path.write_text(
        "     42.5181697829     : integral of the spin-summed LDOS over the unit cell\n"
        "    -1.25000000000     : integral of the projected magnetisation LDOS\n"
        "   -0.924362903287E-01 : sampling energy e0 = efermi + bias\n"
        "   -0.984362903287E-01 : efermi\n"
        "   0.5773502692      0.5773502692      0.5773502692     : direction\n"
    )
    d = stm.parse_stm_dos(path)
    assert d["dos"] == pytest.approx(42.5181697829)
    assert d["spin_dos"] == pytest.approx(-1.25)
    assert d["energy"] == pytest.approx(-0.0924362903287)
    assert d["efermi"] == pytest.approx(-0.0984362903287)
    assert d["direction"] == pytest.approx(np.array([1, 1, 1]) / np.sqrt(3), abs=1e-9)
