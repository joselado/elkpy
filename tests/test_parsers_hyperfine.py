"""Unit tests for parsers.hyperfine (EFG.OUT, MOSSBAUER.OUT).

The non-magnetic fixtures are verbatim output of a real compiled Elk
binary (bulk Si, diamond structure, ngridk 2x2x2, rgkmax 5, lmaxi 2). The
spin-polarised and dipolar Moessbauer sections are format-derived --
transcribed from the `write` statements cited above the fixture, since
that run was not spin polarised and those sections are exactly the ones a
non-magnetic run never writes.

The real EFG.OUT is itself a physics check: both silicon sites in diamond
have T_d point symmetry, and a traceless symmetric rank-2 tensor is
forbidden at a site with cubic symmetry, so every component must vanish.
Measured: |V_ij| <= 1.3e-16 Hartree/Bohr^2, i.e. floating-point zero.
"""

import numpy as np
import pytest

from elkpy.parsers import hyperfine

# vendor/elk/src/writeefg.f90:51-99
#   write(50,'("(electric field gradient tensor is in Cartesian coordinates)")')
#   write(50,'("Species : ",I4," (",A,"), atom : ",I4)') is,trim(spsymb(is)),ia
#   write(50,'(" EFG tensor :")')
#   do i=1,3
#     write(50,'(3G18.10)') (efg(i,j),j=1,3)
#   end do
#   write(50,'(" trace : ",G18.10)') efg(1,1)+efg(2,2)+efg(3,3)
#   write(50,'(" eigenvalues :")')
#   write(50,'(3G18.10)') w
# Real output, bulk Si: both sites are T_d, so the EFG is an exact null.
EFG_OUT = """
(electric field gradient tensor is in Cartesian coordinates)


Species :    1 (Si), atom :    1

 EFG tensor :
 -0.5373340867E-16 -0.1469210663E-16 -0.4246876190E-17
 -0.1469210663E-16 -0.6905589630E-16 -0.1113600013E-15
 -0.4246876190E-17 -0.1113600013E-15  0.1227893050E-15
 trace :  -0.9860761315E-31
 eigenvalues :
 -0.1233950528E-15 -0.5047107119E-16  0.1738661240E-15


Species :    1 (Si), atom :    2

 EFG tensor :
 -0.4928609732E-16 -0.1437796897E-16 -0.1180267126E-16
 -0.1437796897E-16 -0.6306261188E-16 -0.1019323574E-15
 -0.1180267126E-16 -0.1019323574E-15  0.1123487092E-15
 trace :   0.2465190329E-31
 eigenvalues :
 -0.1147760597E-15 -0.4444614437E-16  0.1592222041E-15
"""

# Format-derived, same write statements, with a genuinely anisotropic
# (axially symmetric) EFG so that the tensor/eigenvalue plumbing is
# exercised on nonzero numbers as well.
EFG_OUT_ANISOTROPIC = """
(electric field gradient tensor is in Cartesian coordinates)


Species :    1 (Cu), atom :    1

 EFG tensor :
  -0.500000000       0.000000000       0.000000000
   0.000000000      -0.500000000       0.000000000
   0.000000000       0.000000000       1.000000000
 trace :   0.000000000
 eigenvalues :
  -0.500000000      -0.500000000       1.000000000
"""

# vendor/elk/src/mossbauer.f90:76-108
#   write(50,'("Species : ",I4," (",A,"), atom : ",I4)') is,trim(spsymb(is)),ia
#   write(50,'(" approximate nuclear radius : ",G18.10)') rnucl(is)
#   write(50,'(" number of mesh points to nuclear radius : ",I6)') nrn
#   write(50,'(" Thomson radius : ",G18.10)') rtmsn(is)
#   write(50,'(" number of mesh points to Thomson radius : ",I6)') nrt
#   write(50,'(" Contact density :")')
#   write(50,'("  at nuclear center         : ",G18.10)') r0
#   ... four more labelled densities ...
# Real output, bulk Si (spinpol false -> no hyperfine section at all).
MOSSBAUER_OUT = """

Species :    1 (Si), atom :    1

 approximate nuclear radius :   0.5963282101E-04
 number of mesh points to nuclear radius :    124
 Thomson radius :   0.7455189633E-03
 number of mesh points to Thomson radius :    189

 Contact density :
  at nuclear center         :    2074.784372
  at nuclear surface        :    1886.941068
  average in nuclear volume :    1893.935143
  at Thomson radius         :    1803.705630
  average in Thomson volume :    1818.534402


Species :    1 (Si), atom :    2

 approximate nuclear radius :   0.5963282101E-04
 number of mesh points to nuclear radius :    124
 Thomson radius :   0.7455189633E-03
 number of mesh points to Thomson radius :    189

 Contact density :
  at nuclear center         :    2074.784372
  at nuclear surface        :    1886.941068
  average in nuclear volume :    1893.935143
  at Thomson radius         :    1803.705630
  average in Thomson volume :    1818.534402
"""

# vendor/elk/src/mossbauer.f90:118-172 -- the sections a spin-polarised run
# with tbdip=.true. adds. Format-derived (the real run above was not
# spin polarised).
#   write(50,'(" Contact average in nuclear volume :")')
#   write(50,'("  moment (mu_B) : ",3G18.10)') mn(1:ndmag)
#   write(50,'("  magnetic field : ",3G18.10)') bn(1:ndmag)
#   write(50,'("   tesla : ",3G18.10)') b_si*bn(1:ndmag)
#   write(50,'(" Average dipole field in nuclear volume :")')
#   write(50,'("  spin : ",3G18.10)') bn         ! "spin and orbital" if tjr
#   write(50,'("   tesla : ",3G18.10)') b_si*bn
MOSSBAUER_OUT_MAGNETIC = """

Species :    1 (Fe), atom :    1

 approximate nuclear radius :   0.7000000000E-04
 number of mesh points to nuclear radius :    130
 Thomson radius :   0.8000000000E-03
 number of mesh points to Thomson radius :    195

 Contact density :
  at nuclear center         :    15000.00000
  at nuclear surface        :    14000.00000
  average in nuclear volume :    14100.00000
  at Thomson radius         :    13000.00000
  average in Thomson volume :    13100.00000

 Contact average in nuclear volume :
  moment (mu_B) :  -0.1200000000E-01
  magnetic field :  -0.5000000000E-04
   tesla :   -11.76000000

 Contact average in Thomson volume :
  moment (mu_B) :  -0.1100000000E-01
  magnetic field :  -0.4600000000E-04
   tesla :   -10.82000000

 Average dipole field in nuclear volume :
  spin and orbital :   0.1000000000E-05  0.2000000000E-05  0.3000000000E-05
   tesla :   0.2352000000      0.4704000000      0.7056000000

 Average dipole field in Thomson volume :
  spin and orbital :   0.1000000000E-05  0.2000000000E-05  0.3000000000E-05
   tesla :   0.2352000000      0.4704000000      0.7056000000

Note that the contact term is implicitly included in the spin dipole field
 but may not match exactly with the directly calculated value.
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_efg_cubic_site_is_an_exact_null(tmp_path):
    atoms = hyperfine.parse_efg(_write(tmp_path, "EFG.OUT", EFG_OUT))
    assert len(atoms) == 2
    for entry in atoms:
        assert entry["symbol"] == "Si"
        assert entry["tensor"].shape == (3, 3)
        assert entry["eigenvalues"].shape == (3,)
        # T_d site symmetry forbids a rank-2 traceless tensor
        assert np.abs(entry["tensor"]).max() < 1e-15
        assert abs(entry["trace"]) < 1e-15
    assert [e["atom"] for e in atoms] == [1, 2]


def test_parse_efg_anisotropic(tmp_path):
    atoms = hyperfine.parse_efg(_write(tmp_path, "EFG.OUT", EFG_OUT_ANISOTROPIC))
    assert len(atoms) == 1
    entry = atoms[0]
    assert entry["symbol"] == "Cu"
    assert np.diag(entry["tensor"]) == pytest.approx([-0.5, -0.5, 1.0])
    # the EFG is traceless by construction (the l=m=0 part is removed)
    assert entry["trace"] == pytest.approx(0.0)
    assert entry["trace"] == pytest.approx(np.trace(entry["tensor"]))
    # dsyev returns ascending eigenvalues, not the |Vzz| >= |Vyy| convention
    assert entry["eigenvalues"] == pytest.approx([-0.5, -0.5, 1.0])
    assert np.all(np.diff(entry["eigenvalues"]) >= 0)


def test_parse_efg_tensor_is_symmetric(tmp_path):
    atoms = hyperfine.parse_efg(_write(tmp_path, "EFG.OUT", EFG_OUT))
    for entry in atoms:
        assert entry["tensor"] == pytest.approx(entry["tensor"].T)


def test_parse_mossbauer_nonmagnetic(tmp_path):
    atoms = hyperfine.parse_mossbauer(_write(tmp_path, "MOSSBAUER.OUT", MOSSBAUER_OUT))
    assert len(atoms) == 2
    entry = atoms[0]
    assert (entry["species"], entry["symbol"], entry["atom"]) == (1, "Si", 1)
    assert entry["nuclear_radius"] == pytest.approx(0.5963282101e-04)
    assert entry["nuclear_mesh_points"] == 124
    assert entry["thomson_radius"] == pytest.approx(0.7455189633e-03)
    assert entry["thomson_mesh_points"] == 189
    density = entry["contact_density"]
    assert set(density) == {
        "nuclear_center",
        "nuclear_surface",
        "nuclear_volume_average",
        "thomson_radius",
        "thomson_volume_average",
    }
    assert density["nuclear_center"] == pytest.approx(2074.784372)
    # the density falls monotonically outwards from the nucleus
    assert density["nuclear_center"] > density["nuclear_surface"] > density["thomson_radius"]
    # no hyperfine sections at all for a non-magnetic run
    assert "contact_nuclear" not in entry
    assert "dipole_nuclear" not in entry


def test_parse_mossbauer_magnetic_sections(tmp_path):
    atoms = hyperfine.parse_mossbauer(
        _write(tmp_path, "MOSSBAUER.OUT", MOSSBAUER_OUT_MAGNETIC)
    )
    entry = atoms[0]
    assert entry["symbol"] == "Fe"
    contact = entry["contact_nuclear"]
    assert contact["moment"] == pytest.approx([-0.012])
    assert contact["field"] == pytest.approx([-5.0e-05])
    assert contact["field_tesla"] == pytest.approx([-11.76])
    assert entry["contact_thomson"]["field_tesla"] == pytest.approx([-10.82])
    dipole = entry["dipole_nuclear"]
    assert dipole["includes_orbital"] is True
    assert dipole["field"] == pytest.approx([1e-06, 2e-06, 3e-06])
    assert dipole["field_tesla"] == pytest.approx([0.2352, 0.4704, 0.7056])
    # the "tesla" line under a dipole section must not leak into the
    # contact section that precedes it
    assert entry["contact_thomson"]["field_tesla"].shape == (1,)


def test_parse_mossbauer_rejects_empty_file(tmp_path):
    with pytest.raises(ValueError):
        hyperfine.parse_mossbauer(_write(tmp_path, "MOSSBAUER.OUT", "\n\n"))
