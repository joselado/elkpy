"""Unit tests for parsers.geomfile (crystal.xsf, crystal.ascii,
crystal.axsf).

The XSF and V_Sim fixtures are verbatim output of a real compiled Elk
binary (bulk Si, diamond structure, task 190); the animated-XSF fixture is
format-derived from vendor/elk/src/writeaxsf.f90, since molecular dynamics
was not run.

The physics check with teeth here is the ORIGIN. Elk's default
``tshift=.true.`` shifts the atomic basis so that a centrosymmetric
crystal has an inversion centre at the origin (``findsymcrys.f90``).
Silicon was handed to Elk as (0,0,0) and (1/4,1/4,1/4); it comes back as
+-(3/8,3/8,3/8), one of the several inversion centres of the diamond
lattice, so the two atoms are symmetric about the origin and still
separated by (1/4,1/4,1/4) in lattice coordinates. Reading these files
back is exactly how that shift becomes visible; the lattice vectors are
untouched, and only the XSF file is in Angstrom.
"""

import numpy as np
import pytest

from elkpy.parsers import geomfile

# vendor/elk/src/geomplot.f90:16-32
#   write(50,'("CRYSTAL")')
#   write(50,'("PRIMVEC")')
#   write(50,'(3G18.10)') avec(:,1)*br_ang    ! and (:,2), (:,3)
#   write(50,'("PRIMCOORD")')
#   write(50,'(2I8)') natmtot,1
#   write(50,'(A,3G18.10)') trim(spsymb(is)),atposc(:,ia,is)*br_ang
# Real output, bulk Si with avec = 5.13 Bohr diamond vectors.
CRYSTAL_XSF = """
CRYSTAL

PRIMVEC
   2.714679092       2.714679092       0.000000000
   2.714679092       0.000000000       2.714679092
   0.000000000       2.714679092       2.714679092

PRIMCOORD
       2       1
Si   2.036009319       2.036009319       2.036009319
Si  -2.036009319      -2.036009319      -2.036009319
"""

# vendor/elk/src/geomplot.f90:56-70 -- note NO br_ang on this path, so the
# V_Sim file is in Bohr while the XSF one is in Angstrom.
#   write(50,'(3G18.10)') dxx,dyx,dyy
#   write(50,'(3G18.10)') dzx,dzy,dzz
#   write(50,'(3G18.10," ",A)') v4,trim(spsymb(is))
# Real output, same run.
CRYSTAL_ASCII = """
   7.254915575       3.627457787       6.282941190
   3.627457787       2.094313730      -5.923613762

   5.441186681       3.141470595      -2.221355161     Si
  -5.441186681      -3.141470595       2.221355161     Si
"""

# vendor/elk/src/writeaxsf.f90:13-31 -- format-derived.
#   write(50,'("ANIMSTEPS ",I8)') (ntimes-2)/ntsforce+1
#   write(50,'("PRIMCOORD ",I8)') (itimes-1)/ntsforce+1
CRYSTAL_AXSF = """ANIMSTEPS        2
CRYSTAL
PRIMVEC
   2.714679092       2.714679092       0.000000000
   2.714679092       0.000000000       2.714679092
   0.000000000       2.714679092       2.714679092

PRIMCOORD        1
       2       1
Si   2.036009319       2.036009319       2.036009319
Si  -2.036009319      -2.036009319      -2.036009319

PRIMCOORD        2
       2       1
Si   2.038009319       2.036009319       2.036009319
Si  -2.038009319      -2.036009319      -2.036009319
"""

SI_AVEC_BOHR = np.array([[5.13, 5.13, 0.0], [5.13, 0.0, 5.13], [0.0, 5.13, 5.13]])


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_xsf_lattice_is_the_input_lattice_in_angstrom(tmp_path):
    result = geomfile.parse_xsf(_write(tmp_path, "crystal.xsf", CRYSTAL_XSF))
    assert result["symbols"] == ["Si", "Si"]
    assert result["avec"] == pytest.approx(
        SI_AVEC_BOHR * geomfile.BOHR_IN_ANGSTROM, rel=1e-6
    )
    assert result["positions"].shape == (2, 3)


def test_parse_xsf_shows_the_tshift_origin_move(tmp_path):
    """The positions are NOT the input ones: Elk put an inversion centre at
    the origin, so the two atoms sit at +-(3/8,3/8,3/8) rather than at
    (0,0,0) and (1/4,1/4,1/4)."""
    result = geomfile.parse_xsf(_write(tmp_path, "crystal.xsf", CRYSTAL_XSF))
    fractional = result["positions"] @ np.linalg.inv(result["avec"])
    assert fractional[0] == pytest.approx([0.375, 0.375, 0.375], abs=1e-6)
    assert fractional[1] == pytest.approx([-0.375, -0.375, -0.375], abs=1e-6)
    # inversion symmetric about the origin ...
    assert fractional[0] == pytest.approx(-fractional[1], abs=1e-9)
    # ... and still the diamond (1/4,1/4,1/4) basis separation, modulo a
    # lattice translation
    separation = (fractional[1] - fractional[0]) % 1.0
    assert separation == pytest.approx([0.25, 0.25, 0.25], abs=1e-6)


def test_parse_vsim_ascii_is_in_bohr_and_rotated(tmp_path):
    result = geomfile.parse_vsim_ascii(_write(tmp_path, "crystal.ascii", CRYSTAL_ASCII))
    assert result["symbols"] == ["Si", "Si"]
    dxx = result["lower_triangle"][0]
    # dxx = |a1| in Bohr (no br_ang on this path)
    assert dxx == pytest.approx(np.linalg.norm(SI_AVEC_BOHR[0]), rel=1e-6)
    # the rotated frame preserves lengths and angles
    assert np.linalg.norm(result["avec"][1]) == pytest.approx(
        np.linalg.norm(SI_AVEC_BOHR[1]), rel=1e-6
    )
    assert abs(np.linalg.det(result["avec"])) == pytest.approx(
        abs(np.linalg.det(SI_AVEC_BOHR)), rel=1e-6
    )
    assert result["positions"][0] == pytest.approx(-result["positions"][1], rel=1e-9)


def test_parse_axsf_frames(tmp_path):
    frames = geomfile.parse_axsf(_write(tmp_path, "crystal.axsf", CRYSTAL_AXSF))
    assert [f["frame"] for f in frames] == [1, 2]
    for frame in frames:
        assert frame["symbols"] == ["Si", "Si"]
        assert frame["avec"] is not None
        assert frame["positions"].shape == (2, 3)
    # the second frame is displaced along x only
    delta = frames[1]["positions"] - frames[0]["positions"]
    assert delta[0] == pytest.approx([0.002, 0.0, 0.0], abs=1e-9)


def test_parse_xsf_rejects_a_file_without_primcoord(tmp_path):
    text = CRYSTAL_XSF.split("PRIMCOORD")[0]
    with pytest.raises(ValueError):
        geomfile.parse_xsf(_write(tmp_path, "crystal.xsf", text))
