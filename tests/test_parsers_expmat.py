"""Unit tests for parsers/expmat.py (tasks 130 and 135) -- no Elk run
needed.

EXPIQR.OUT fixtures are transcribed from vendor/elk/src/writeexpmat.f90's
write statements (the header block, then per k-point two labelled
3-vectors, then per state a labelled header and nstsv rows of
`(I6,3G18.10)` holding j, Re, Im, |.|^2).

WFPW.OUT is written by vendor/elk/src/writewfpw.f90 as a DIRECT-access
UNFORMATTED file whose record length comes from

    inquire(iolength=recl) vkl(:,1), nhkmax, nspinor, nstsv, wfpw

with `write(270,rec=ik) vkl(:,ik), nhkmax, nspinor, nstsv, wfpw`. A
direct-access record carries no record markers, so the fixture below is
built byte-for-byte with struct/numpy: 3 float64, 3 int32, then
wfpw(nhkmax, nspinor, nstsv) complex128 in Fortran order. This assumes
4-byte default integers and a byte-valued `iolength`, which is gfortran's
default and which build-config/make.inc does not override (no
-fdefault-integer-8, no -frecord-marker).
"""

import numpy as np
import pytest

from elkpy.parsers.expmat import parse_expiqr, read_wfpw


def _write_expiqr(path, vecql, vecqc, kpoints, matrices):
    """Reproduce writeexpmat.f90's EXPIQR.OUT layout exactly."""
    nstsv = matrices[0].shape[0]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write("q-vector (lattice coordinates) :\n")
        fh.write("".join(f"{v:18.10G}" for v in vecql) + "\n")
        fh.write("q-vector (Cartesian coordinates) :\n")
        fh.write("".join(f"{v:18.10G}" for v in vecqc) + "\n")
        fh.write("\n")
        fh.write(f"{len(kpoints):8d} : number of k-points\n")
        fh.write(f"{nstsv:6d} : number of states per k-point\n")
        for (vkl, vkc), matrix in zip(kpoints, matrices):
            fh.write("\n")
            fh.write(" k-point (lattice coordinates) :\n")
            fh.write("".join(f"{v:18.10G}" for v in vkl) + "\n")
            fh.write("\n")
            fh.write(" k-point (Cartesian coordinates) :\n")
            fh.write("".join(f"{v:18.10G}" for v in vkc) + "\n")
            for i in range(nstsv):
                fh.write("\n")
                fh.write(
                    f"{i + 1:6d} : state i; state j, <...>, |<...>|² below\n"
                )
                for j in range(nstsv):
                    z = matrix[i, j]
                    fh.write(
                        f"{j + 1:6d}{z.real:18.10G}{z.imag:18.10G}"
                        f"{abs(z) ** 2:18.10G}\n"
                    )


def test_parse_expiqr_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    nstsv = 3
    matrices = [
        rng.normal(size=(nstsv, nstsv)) + 1j * rng.normal(size=(nstsv, nstsv))
        for _ in range(2)
    ]
    kpoints = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
               ((0.5, 0.0, 0.0), (0.3, 0.3, -0.3))]
    path = tmp_path / "EXPIQR.OUT"
    _write_expiqr(path, (0.5, 0.0, 0.0), (0.3, 0.3, -0.3), kpoints, matrices)

    result = parse_expiqr(path)
    assert result["vecql"] == pytest.approx([0.5, 0.0, 0.0])
    assert result["vecqc"] == pytest.approx([0.3, 0.3, -0.3])
    assert len(result["kpoints"]) == 2
    assert result["kpoints"][1]["vkl"] == pytest.approx([0.5, 0.0, 0.0])
    for parsed, expected in zip(result["kpoints"], matrices):
        # the printed precision is G18.10, so ~10 significant figures
        assert parsed["matrix"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_parse_expiqr_detects_row_misordering(tmp_path):
    """A silently dropped row would shift every subsequent matrix element
    by one state; the `j` column is what catches it."""
    matrix = np.eye(2, dtype=complex)
    path = tmp_path / "EXPIQR.OUT"
    _write_expiqr(path, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                  [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))], [matrix])
    lines = path.read_text(encoding="utf-8").splitlines()
    # drop the "j = 1" row of state 1
    del lines[next(i for i, l in enumerate(lines) if l.strip().startswith("1 ")
                   and len(l.split()) == 4)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out of order"):
        parse_expiqr(path)


def _write_wfpw(path, kpoints, wfpw):
    """Byte-exact DIRECT-access records, matching writewfpw.f90."""
    nkpt, nhkmax, nspinor, nstsv = wfpw.shape
    with open(path, "wb") as fh:
        for ik in range(nkpt):
            fh.write(np.asarray(kpoints[ik], dtype="<f8").tobytes())
            fh.write(np.array([nhkmax, nspinor, nstsv], dtype="<i4").tobytes())
            fh.write(
                np.asarray(wfpw[ik], dtype="<c16").flatten(order="F").tobytes()
            )


def test_read_wfpw_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    nkpt, nhkmax, nspinor, nstsv = 3, 5, 2, 4
    wfpw = (rng.normal(size=(nkpt, nhkmax, nspinor, nstsv))
            + 1j * rng.normal(size=(nkpt, nhkmax, nspinor, nstsv)))
    kpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]])
    path = tmp_path / "WFPW.OUT"
    _write_wfpw(path, kpoints, wfpw)

    result = read_wfpw(path)
    assert result["kpoints"] == pytest.approx(kpoints)
    assert result["wfpw"].shape == (nkpt, nhkmax, nspinor, nstsv)
    assert result["wfpw"] == pytest.approx(wfpw)


def test_read_wfpw_rejects_inconsistent_record_length(tmp_path):
    """The record length is derived from the header integers, so a
    truncated or differently-dimensioned file must not be reshaped into
    plausible nonsense."""
    rng = np.random.default_rng(2)
    wfpw = (rng.normal(size=(2, 3, 1, 2)) + 1j * rng.normal(size=(2, 3, 1, 2)))
    path = tmp_path / "WFPW.OUT"
    _write_wfpw(path, np.zeros((2, 3)), wfpw)
    with open(path, "ab") as fh:
        fh.write(b"\x00" * 8)
    with pytest.raises(ValueError, match="not a multiple of the record length"):
        read_wfpw(path)
