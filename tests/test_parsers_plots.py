"""Unit tests for the generic plot1d/plot3d parsers (no Elk run).

Every fixture below is written by a Python transcription of the Fortran
``write`` statement that produces the real file, cited on the function that
writes it, so these are FORMAT-DERIVED checks -- they pin the parser against
the Fortran source, not against a file a real binary produced.

The check with teeth is the GRID ORDER. ``src/plotpt3d.f90`` runs::

    do i3=0,np3d(3)-1
      do i2=0,np3d(2)-1
        do i1=0,np3d(1)-1

so the FIRST plotting vector's index is innermost and a column reshapes to
a volume as ``(n3, n2, n1)``. Getting that backwards permutes the axes
without changing a single number, which no downstream numerical check can
catch -- so it is pinned directly against a file whose values encode their
own indices.
"""

import math

import numpy as np
import pytest

from elkpy.parsers import plots


def g(v, w=18, d=10):
    """One Fortran ``Gw.d`` edit-descriptor field.

    Fortran prints ``Gw.d`` in F form with four trailing blanks whenever
    0.1 <= |v| < 10^d, and in E form otherwise; reproduced here so the
    fixtures have the same column widths and trailing blanks a real Elk
    file does.
    """
    if v == 0.0:
        n = 0
    else:
        n = math.floor(math.log10(abs(v))) + 1
    if 0 <= n <= d:
        return f"{v:{w - 4}.{d - n}f}" + "    "
    return f"{v:{w}.{d - 1}E}"


def write_plot1d(path, npoints, nf=1):
    """src/plot1d.f90::

        do ip=1,npp1d
          write(fnum1,'(5G18.10)') dpp1d(ip),(fp(ip,jf),jf=1,nf)
        end do

    No header line. Values encode their own point index and column so a
    misread shows up as a wrong number, not just a wrong shape.
    """
    with open(path, "w") as fh:
        for ip in range(npoints):
            row = [0.25 * ip] + [100.0 * jf + ip for jf in range(nf)]
            fh.write("".join(g(v) for v in row) + "\n")


def write_plot_lines(path, vertex_distances, fmin=-1.5, fmax=2.5):
    """src/plot1d.f90's second unit::

        do iv=1,nvp1d
          write(fnum2,'(2G18.10)') dvp1d(iv),fmin
          write(fnum2,'(2G18.10)') dvp1d(iv),fmax
          write(fnum2,*)
        end do
    """
    with open(path, "w") as fh:
        for d in vertex_distances:
            fh.write(g(d) + g(fmin) + "\n")
            fh.write(g(d) + g(fmax) + "\n")
            fh.write("\n")


def write_plot3d(path, n1, n2, n3, nf=1):
    """src/plot3d.f90::

        write(fnum,'(3I6," : grid size")') np3d(:)
        do ip=1,np
          call r3mv(avec,vpl(:,ip),v1)
          write(fnum,'(7G18.10)') v1(:),(fp(ip,jf),jf=1,nf)
        end do

    with src/plotpt3d.f90's i3-outer / i1-inner point order. Each value is
    ``i1 + 10*i2 + 100*i3`` (plus 1000*column), so it names its own indices.
    """
    with open(path, "w") as fh:
        fh.write(f"{n1:6d}{n2:6d}{n3:6d} : grid size\n")
        for i3 in range(n3):
            for i2 in range(n2):
                for i1 in range(n1):
                    code = i1 + 10 * i2 + 100 * i3
                    row = [float(i1), float(i2), float(i3)]
                    row += [1000.0 * jf + code for jf in range(nf)]
                    fh.write("".join(g(v) for v in row) + "\n")


def test_parse_plot1d_scalar(tmp_path):
    path = tmp_path / "RHO1D.OUT"
    write_plot1d(path, 7)
    distances, values = plots.parse_plot1d(path)
    assert distances.shape == (7,)
    assert values.shape == (7,)
    np.testing.assert_allclose(distances, 0.25 * np.arange(7))
    np.testing.assert_allclose(values, np.arange(7.0))


def test_parse_plot1d_vector_field(tmp_path):
    """Vector plots (MAG1D.OUT, JPR1D.OUT, RHOS1D.OUT) write nf=3 columns."""
    path = tmp_path / "MAG1D.OUT"
    write_plot1d(path, 5, nf=3)
    distances, values = plots.parse_plot1d(path, nf=3)
    assert values.shape == (5, 3)
    np.testing.assert_allclose(values[:, 0], np.arange(5.0))
    np.testing.assert_allclose(values[:, 1], 100.0 + np.arange(5.0))
    np.testing.assert_allclose(values[:, 2], 200.0 + np.arange(5.0))
    np.testing.assert_allclose(distances[-1], 1.0)


def test_parse_plot_lines_collapses_the_duplicate_vertex_rows(tmp_path):
    path = tmp_path / "RHOLINES.OUT"
    write_plot_lines(path, [0.0, 0.75, 1.9])
    assert plots.parse_plot_lines(path) == pytest.approx([0.0, 0.75, 1.9])


def test_parse_plot3d_grid_and_values(tmp_path):
    path = tmp_path / "RHO3D.OUT"
    write_plot3d(path, 3, 4, 5)
    points, values, grid = plots.parse_plot3d(path)
    assert grid == (3, 4, 5)
    assert points.shape == (60, 3)
    assert values.shape == (60,)
    # the header's grid size and the number of data rows must agree
    assert grid[0] * grid[1] * grid[2] == values.size


def test_parse_plot3d_axis_order_is_i1_fastest(tmp_path):
    """THE ordering pin: values encode (i1, i2, i3), so a transposed
    reshape gives a wrong NUMBER here rather than merely a wrong shape."""
    path = tmp_path / "ELF3D.OUT"
    write_plot3d(path, 3, 4, 5)
    _points, values, grid = plots.parse_plot3d(path)
    volume = plots.reshape_plot3d(values, grid)
    assert volume.shape == (5, 4, 3)  # (n3, n2, n1)
    for i1 in range(3):
        for i2 in range(4):
            for i3 in range(5):
                assert volume[i3, i2, i1] == pytest.approx(i1 + 10 * i2 + 100 * i3)


def test_parse_plot3d_vector_field(tmp_path):
    path = tmp_path / "MAG3D.OUT"
    write_plot3d(path, 2, 3, 2, nf=3)
    _points, values, grid = plots.parse_plot3d(path, nf=3)
    assert values.shape == (12, 3)
    component = plots.reshape_plot3d(values[:, 2], grid)
    assert component[1, 2, 1] == pytest.approx(2000.0 + 1 + 20 + 100)


def test_reshape_plot2d_is_n2_by_n1(tmp_path):
    """src/plotpt2d.f90 shares plotpt3d's convention: i1 innermost."""
    values = np.array([i1 + 10 * i2 for i2 in range(3) for i1 in range(4)], dtype=float)
    image = plots.reshape_plot2d(values, (4, 3))
    assert image.shape == (3, 4)
    assert image[2, 3] == pytest.approx(23.0)
