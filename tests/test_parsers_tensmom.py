"""Unit tests for parsers.tensmom (TENSMOM.OUT, task 400).

Format-derived: every fixture is transcribed from the `write` statements
of vendor/elk/src/writetm3.f90 cited above it. Task 400 was not run
against the binary -- it needs a converged DFT+U ground state, which is
outside this bucket's one permitted run.

The structural facts the parser must respect, all fixed by writetm3's own
loop bounds ``do k=0,2*l; do p=0,1; do r=abs(k-p),k+p``:

- k is the orbital rank (0..2l), p the spin rank (0 or 1), r the total
  rank obtained by coupling them, and there are 2r+1 components t;
- ``tm3type`` = 1 prints two numbers per component (the complex van der
  Laan convention), 0 and 2 print one.
"""

import numpy as np
import pytest

from elkpy.parsers import tensmom

# vendor/elk/src/writetm3.f90:38-100
#   write(50,'("Density matrix decomposition in coupled tensor moments")')
#   write(50,'("Tensor moment type :")')
#   write(50,'(" real, corresponding to Hermitian Gamma matrices")')
#   write(50,'("Species : ",I4," (",A,"), atom : ",I4)') is,spsymb,ia
#   write(50,'(" l = ",I1)') l
#   write(50,'("  k = ",I1,", p = ",I1,", r = ",I1)') k,p,r
#   write(50,'("   t = ",I2," : ",F14.8)') t,t0*wkpr(t)
#   write(50,'("  magnitude : ",F14.8)') t1
#   write(50,'("  Hartree + exchange energy : ",F14.8)') ehx
TENSMOM_OUT = """Density matrix decomposition in coupled tensor moments
Tensor moment type :
 real, corresponding to Hermitian Γ matrices

Species :    1 (Ni), atom :    1
 l = 2

  k = 0, p = 0, r = 0
   t =  0 :     8.12345678
  magnitude :     8.12345678
  Hartree + exchange energy :    -0.05000000

  k = 0, p = 1, r = 1
   t = -1 :     0.00000000
   t =  0 :     1.60000000
   t =  1 :     0.00000000
  magnitude :     1.60000000
  Hartree + exchange energy :    -0.01000000

Species :    1 (Ni), atom :    2
 l = 2

  k = 0, p = 0, r = 0
   t =  0 :     8.12345678
  magnitude :     8.12345678
  Hartree + exchange energy :    -0.05000000
"""

# vendor/elk/src/writetm3.f90:82-85, the tm3type = 1 branch:
#   write(50,'("   t = ",I2," : ",2F14.8)') t,wkpr_v(t)
TENSMOM_OUT_COMPLEX = """Density matrix decomposition in coupled tensor moments
Tensor moment type :
 complex, see Appendix A of G. van der Laan and B. T. Thole,
 J. Phys.: Condens. Matter 7, 9947 (1995)

Species :    2 (U), atom :    1
 l = 3

  k = 1, p = 1, r = 2
   t = -2 :     0.10000000    0.20000000
   t = -1 :     0.00000000    0.00000000
   t =  0 :     0.30000000    0.00000000
   t =  1 :     0.00000000    0.00000000
   t =  2 :     0.10000000   -0.20000000
  magnitude :     0.42426407
  Hartree + exchange energy :    -0.00200000
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_tensor_moments_real(tmp_path):
    moments = tensmom.parse_tensor_moments(_write(tmp_path, "TENSMOM.OUT", TENSMOM_OUT))
    assert len(moments) == 3

    occupation = moments[0]
    assert (occupation["species"], occupation["symbol"], occupation["atom"]) == (1, "Ni", 1)
    assert occupation["l"] == 2
    assert (occupation["k"], occupation["p"], occupation["r"]) == (0, 0, 0)
    # rank 0 has exactly one component; (0,0,0) is the shell occupation
    assert occupation["t"] == pytest.approx([0])
    assert occupation["value"] == pytest.approx([8.12345678])
    assert occupation["magnitude"] == pytest.approx(8.12345678)
    assert occupation["energy"] == pytest.approx(-0.05)

    spin = moments[1]
    assert (spin["k"], spin["p"], spin["r"]) == (0, 1, 1)
    # rank r carries 2r+1 components, t = -r..r
    assert spin["t"] == pytest.approx([-1, 0, 1])
    assert spin["value"] == pytest.approx([0.0, 1.6, 0.0])
    assert spin["magnitude"] == pytest.approx(np.linalg.norm(spin["value"]))

    # the second atom's block inherits its own species/atom header
    assert moments[2]["atom"] == 2


def test_parse_tensor_moments_complex(tmp_path):
    moments = tensmom.parse_tensor_moments(
        _write(tmp_path, "TENSMOM.OUT", TENSMOM_OUT_COMPLEX)
    )
    assert len(moments) == 1
    entry = moments[0]
    assert entry["l"] == 3
    assert (entry["k"], entry["p"], entry["r"]) == (1, 1, 2)
    assert entry["value"].dtype == np.complex128
    assert entry["t"] == pytest.approx([-2, -1, 0, 1, 2])
    assert entry["value"][0] == pytest.approx(0.1 + 0.2j)
    assert entry["value"][4] == pytest.approx(0.1 - 0.2j)


def test_parse_tensor_moments_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        tensmom.parse_tensor_moments(_write(tmp_path, "TENSMOM.OUT", "nothing\n"))
