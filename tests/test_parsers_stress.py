"""Unit tests for parsers.stress (STRAIN.OUT, STRESS.OUT, PIEZOELT.OUT,
MAGNETOELT.OUT).

The STRAIN/STRESS fixtures are verbatim output of a real compiled Elk
binary (bulk Si, diamond structure, ngridk 2x2x2, rgkmax 5, lmaxi 2,
tasks 0 115 110 195 430 190 440 in one run) -- non-ASCII header
characters included. The PIEZOELT/MAGNETOELT fixtures are format-derived:
transcribed from the `write` statements cited above each one, since those
tasks were not run.

Two things the real STRAIN.OUT confirms rather than assumes: cubic Si has
``nstrain = 1`` (``symmat`` averages every anisotropic delta_ij to a
multiple of the identity, which the isotropic tensor already spans), and
the entries are 1/sqrt(6) -- the diamond ``avec`` has six equal nonzero
components, so normalising to unit Frobenius norm gives exactly that.
"""

import numpy as np
import pytest

from elkpy.parsers import stress

# vendor/elk/src/writestrain.f90:18-24
#   do k=1,nstrain
#     write(50,*)
#     write(50,'("Strain tensor : ",I0)') k
#     do j=1,3
#       write(50,'(3G18.10)') (strain(i,j,k),i=1,3)
#     end do
#   end do
# Real output, bulk Si (cubic -> a single, isotropic strain tensor).
STRAIN_OUT = """
Strain tensor : 1
  0.4082482905      0.4082482905       0.000000000
  0.4082482905       0.000000000      0.4082482905
   0.000000000      0.4082482905      0.4082482905
"""

# Same format, two tensors, to exercise multi-block parsing (a
# lower-symmetry crystal). Format-derived, not from a run; both tensors
# carry unit Frobenius norm as genstrain guarantees.
STRAIN_OUT_TWO = """
Strain tensor : 1
  0.4082482905      0.4082482905       0.000000000
  0.4082482905       0.000000000      0.4082482905
   0.000000000      0.4082482905      0.4082482905

Strain tensor : 2
   0.000000000       0.000000000      0.7071067812
   0.000000000       0.000000000       0.000000000
  0.7071067812       0.000000000       0.000000000
"""

# vendor/elk/src/writestress.f90:20-38
#   write(50,'("Lattice vector matrix, A, changed by")')
#   write(50,'("     A -> A + e_k dt,")')           [non-ASCII in the source]
#   do k=1,nstrain
#     write(50,'("Strain tensor k : ",I1)') k
#     do j=1,3
#       write(50,'(3G18.10)') (strain(i,j,k),i=1,3)
#     end do
#     write(50,'("Stress : ",G18.10)') stress(k)
#   end do
# Real output, same run as STRAIN_OUT above.
STRESS_OUT = """
Lattice vector matrix, A, changed by

     A → A + eₖ dt,

where dt is an infinitesimal scalar and eₖ is a strain tensor

Stress is given by the derivative of the total energy dE/dt

Strain tensor k : 1
  0.4082482905      0.4082482905       0.000000000
  0.4082482905       0.000000000      0.4082482905
   0.000000000      0.4082482905      0.4082482905
Stress :  -0.8938399606E-02
"""

# vendor/elk/src/piezoelt.f90:67-89 -- format-derived
#     write(50,'("Strain tensor k : ",I1)') k
#     ... three (3G18.10) rows ...
#     write(50,'("Piezoelectric tensor components dP_i/dt, i=1...3 :")')
#     write(50,'(" lattice coordinates : ",3G18.10)') pelt(:,k)
#     write(50,'(" Cartesian coordinates : ",3G18.10)') vc
#     write(50,'("  length : ",G18.10)') norm2(vc(1:3))
PIEZOELT_OUT = """
Lattice vector matrix, A, changed by

     A → A + eₖ dt,

where dt is an infinitesimal scalar and eₖ is a strain tensor

The piezoelectric tensor is the derivative of the polarisation vector dPᵢ/dt, i=1...3

Strain tensor k : 1
  0.5773502692       0.000000000       0.000000000
   0.000000000      0.5773502692       0.000000000
   0.000000000       0.000000000      0.5773502692
Piezoelectric tensor components dPᵢ/dt, i=1...3 :
 lattice coordinates :   0.1000000000E-01  0.2000000000E-01  0.3000000000E-01
 Cartesian coordinates :   0.4000000000E-01  0.5000000000E-01  0.6000000000E-01
  length :   0.8774964387E-01

Strain tensor k : 2
  0.7071067812     -0.7071067812       0.000000000
 -0.7071067812      0.7071067812       0.000000000
   0.000000000       0.000000000       0.000000000
Piezoelectric tensor components dPᵢ/dt, i=1...3 :
 lattice coordinates :  -0.1000000000E-02   0.000000000       0.000000000
 Cartesian coordinates :  -0.2000000000E-02   0.000000000       0.000000000
  length :   0.2000000000E-02
"""

# vendor/elk/src/magnetoelt.f90:70-81 -- format-derived
#   do j=1,3
#     write(50,'("Magnetic field Cartesian component j : ",I1)') j
#     write(50,'("Magnetoelectric tensor components dP_i/dB_j, i=1...3")')
#     write(50,'(" lattice coordinates : ",3G18.10)') felt(:,j)
#     write(50,'(" Cartesian coordinates : ",3G18.10)') vc
#     write(50,'("  length : ",G18.10)') norm2(vc(1:3))
#   end do
MAGNETOELT_OUT = """
The magnetoelectric tensor is the change in the polarisation
with respect to the external magnetic field dPᵢ/dBⱼ, for
components i,j=1...3

Magnetic field Cartesian component j : 1
Magnetoelectric tensor components dPᵢ/dBⱼ, i=1...3
 lattice coordinates :   0.1000000000       0.000000000       0.000000000
 Cartesian coordinates :   0.1100000000       0.000000000       0.000000000
  length :   0.1100000000

Magnetic field Cartesian component j : 2
Magnetoelectric tensor components dPᵢ/dBⱼ, i=1...3
 lattice coordinates :    0.000000000      0.2000000000       0.000000000
 Cartesian coordinates :    0.000000000      0.2200000000       0.000000000
  length :   0.2200000000

Magnetic field Cartesian component j : 3
Magnetoelectric tensor components dPᵢ/dBⱼ, i=1...3
 lattice coordinates :    0.000000000       0.000000000      0.3000000000
 Cartesian coordinates :    0.000000000       0.000000000      0.3300000000
  length :   0.3300000000
"""

SI_AVEC = np.array([[5.13, 5.13, 0.0], [5.13, 0.0, 5.13], [0.0, 5.13, 5.13]])


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_strain_cubic_gives_one_isotropic_tensor(tmp_path):
    tensors = stress.parse_strain(_write(tmp_path, "STRAIN.OUT", STRAIN_OUT))
    assert len(tensors) == 1
    # genstrain sets strain(:,:,1) = avec/||avec||_F before anything else
    assert tensors[0] == pytest.approx(SI_AVEC / np.linalg.norm(SI_AVEC), abs=1e-9)
    assert np.linalg.norm(tensors[0]) == pytest.approx(1.0, abs=1e-9)


def test_parse_strain_multiple_blocks(tmp_path):
    tensors = stress.parse_strain(_write(tmp_path, "STRAIN.OUT", STRAIN_OUT_TWO))
    assert len(tensors) == 2
    for t in tensors:
        assert t.shape == (3, 3)
        assert np.linalg.norm(t) == pytest.approx(1.0, abs=1e-9)
    # genstrain orthogonalises each tensor against the previous ones
    assert np.sum(tensors[0] * tensors[1]) == pytest.approx(0.0, abs=1e-9)


def test_parse_stress(tmp_path):
    result = stress.parse_stress(_write(tmp_path, "STRESS.OUT", STRESS_OUT))
    assert result["stress"].shape == (1,)
    assert result["stress"][0] == pytest.approx(-0.008938399606)
    assert len(result["strain"]) == 1
    assert result["strain"][0][1, 2] == pytest.approx(0.4082482905)


def test_parse_stress_rejects_mismatched_file(tmp_path):
    text = STRESS_OUT.replace("Stress :  -0.8938399606E-02", "")
    with pytest.raises(ValueError):
        stress.parse_stress(_write(tmp_path, "STRESS.OUT", text))


def test_pressure_from_stress_matches_a_numerical_volume_derivative():
    """P = -stress[0] * ||A||_F / (3V) is checked against a numerical
    dV/dt along the very deformation genstrain/strainabg apply,
    A -> A + t e_1, rather than against the closed form being tested."""
    e1 = SI_AVEC / np.linalg.norm(SI_AVEC)
    dt = 1e-6
    volume = abs(np.linalg.det(SI_AVEC))
    dvolume = (abs(np.linalg.det(SI_AVEC + dt * e1)) - volume) / dt

    dedt = -0.008938399606  # the real STRESS.OUT value above
    assert stress.pressure_from_stress([dedt], SI_AVEC) == pytest.approx(
        -dedt / dvolume, rel=1e-5
    )


def test_pressure_sign_is_positive_for_expansion():
    """Energy falling under expansion (stress < 0) must give P > 0."""
    assert stress.pressure_from_stress([-0.01], SI_AVEC) > 0
    assert stress.pressure_from_stress([+0.01], SI_AVEC) < 0


def test_parse_piezoelectric(tmp_path):
    entries = stress.parse_piezoelectric(_write(tmp_path, "PIEZOELT.OUT", PIEZOELT_OUT))
    assert len(entries) == 2
    assert entries[0]["lattice"] == pytest.approx([0.01, 0.02, 0.03])
    assert entries[0]["cartesian"] == pytest.approx([0.04, 0.05, 0.06])
    assert entries[0]["length"] == pytest.approx(np.linalg.norm([0.04, 0.05, 0.06]))
    assert entries[1]["strain"][0, 1] == pytest.approx(-0.7071067812)
    assert entries[1]["cartesian"][0] == pytest.approx(-0.002)


def test_parse_magnetoelectric(tmp_path):
    result = stress.parse_magnetoelectric(_write(tmp_path, "MAGNETOELT.OUT", MAGNETOELT_OUT))
    assert result["cartesian"].shape == (3, 3)
    # row j is dP_i/dB_j: the fixture is diagonal, 0.11 / 0.22 / 0.33
    assert np.diag(result["cartesian"]) == pytest.approx([0.11, 0.22, 0.33])
    assert result["cartesian"][0, 1] == pytest.approx(0.0)
    assert result["length"] == pytest.approx([0.11, 0.22, 0.33])
    assert result["lattice"][2, 2] == pytest.approx(0.3)


def test_parse_magnetoelectric_rejects_truncated_file(tmp_path):
    text = MAGNETOELT_OUT.split("Magnetic field Cartesian component j : 3")[0]
    with pytest.raises(ValueError):
        stress.parse_magnetoelectric(_write(tmp_path, "MAGNETOELT.OUT", text))
