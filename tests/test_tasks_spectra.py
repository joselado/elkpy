"""Unit tests for elkpy.tasks.spectra's input-block construction and its
up-front refusals (no Elk run).

Every method in that mixin either builds an ``elk.in`` block or refuses a
combination the Fortran would ``stop`` on. Those refusals are the part worth
testing without a binary: each one mirrors a specific guard (or, in the
vecplot case, a specific GAP in a guard) in vendor/elk/src/, and getting one
wrong means a real run dies with a bare Fortran ``stop`` and no traceback.
"""

import pytest

from elkpy.calculation import Calculation
from elkpy.structure import Structure
from elkpy.tasks.spectra import SpectraTasks


# Calculation now inherits SpectraTasks (see elkpy/tasks/__init__.py).
SpectraCalculation = Calculation


AVEC = [(5.13, 5.13, 0.0), (5.13, 0.0, 5.13), (0.0, 5.13, 5.13)]


def make(tmp_path, **kwargs):
    s = Structure(AVEC, {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]})
    return SpectraCalculation(s, tmp_path / "si", ngridk=(2, 2, 2), **kwargs)


# ---------------------------------------------------------------- blocks


def test_plot1d_block_layout():
    """src/readinput.f90 case('plot1d') reads ``nvp1d npp1d`` then one line
    per vertex."""
    lines = SpectraTasks._plot1d_lines([(0, 0, 0), (0.5, 0.5, 0.0)], 128)
    assert lines[0] == (2, 128)
    assert lines[1:] == [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0)]


def test_plot1d_block_refuses_npoints_below_the_vertex_count():
    """src/plotpt1d.f90 stops on ``np < nv``."""
    with pytest.raises(ValueError, match="fewer than"):
        SpectraTasks._plot1d_lines([(0, 0, 0), (1, 0, 0), (1, 1, 0)], 2)


def test_plot1d_block_refuses_a_single_vertex():
    with pytest.raises(ValueError, match="at least two vertices"):
        SpectraTasks._plot1d_lines([(0, 0, 0)], 100)


# ------------------------------------------------------------- ndmag


def test_ndmag_unpolarised(tmp_path):
    assert make(tmp_path)._ndmag() == 0


def test_ndmag_collinear(tmp_path):
    assert make(tmp_path, spinpol=True)._ndmag() == 1


def test_ndmag_noncollinear_from_spinorb(tmp_path):
    assert make(tmp_path, spinpol=True, spinorb=True)._ndmag() == 3


def test_ndmag_noncollinear_from_an_in_plane_bfcmt(tmp_path):
    """src/init0.f90 lines 176-181: an x or y component of any atom's
    ``bfcmt`` makes the whole calculation non-collinear."""
    s = Structure(AVEC, {"Si": [((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
                                (0.25, 0.25, 0.25)]})
    calc = SpectraCalculation(s, tmp_path / "si", ngridk=(2, 2, 2), spinpol=True)
    assert calc._ndmag() == 3


def test_ndmag_cmagz_forces_collinear(tmp_path):
    """src/init0.f90 line 190: ``cmagz`` overrides everything else."""
    calc = make(tmp_path, spinpol=True, spinorb=True, extra_blocks={"cmagz": [True]})
    assert calc._ndmag() == 1


# ---------------------------------------------------------- refusals


def test_magnetisation_refuses_an_unpolarised_calculation(tmp_path):
    """src/vecplot.f90's own guard covers tasks 72/73/82/83 but NOT 71 or
    81, which would then read an unallocated ``magmt``/``bxcmt``; refused
    here for every dimension instead."""
    calc = make(tmp_path)
    for dim in (1, 2, 3):
        with pytest.raises(ValueError, match="spin-polarised"):
            calc.get_magnetisation(dim=dim)


def test_bxc_and_divergence_refuse_an_unpolarised_calculation(tmp_path):
    calc = make(tmp_path)
    with pytest.raises(ValueError, match="spin-polarised"):
        calc.get_bxc(dim=1)
    with pytest.raises(ValueError, match="spin-polarised"):
        calc.get_bxc_divergence(dim=3)


def test_magnetic_torque_refuses_a_collinear_calculation(tmp_path):
    """src/vecplot.f90 stops on ``.not.ncmag`` for tasks 151-153: a
    collinear m x B_xc is zero by construction."""
    calc = make(tmp_path, spinpol=True)
    with pytest.raises(ValueError, match="NON-COLLINEAR"):
        calc.get_magnetic_torque(dim=3)


def test_band_character_spin_kinds_require_spin_polarisation(tmp_path):
    """src/bandstr.f90 jumps to its error label for tasks 23 and 24 when
    ``.not.spinpol``."""
    calc = make(tmp_path)
    for kind in ("spin", "moment"):
        with pytest.raises(ValueError, match="spin-polarised"):
            calc.get_band_character(kind=kind, vertices=[(0, 0, 0), (0.5, 0, 0)])


def test_band_character_rejects_an_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="unknown band-character kind"):
        make(tmp_path).get_band_character(kind="j", vertices=[(0, 0, 0), (1, 0, 0)])


def test_fermi_surface_rejects_an_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="unknown Fermi-surface kind"):
        make(tmp_path).get_fermi_surface(kind="isosurface")


def test_potential_rejects_an_unknown_component(tmp_path):
    calc = make(tmp_path)
    with pytest.raises(ValueError, match="unknown potential component"):
        calc.get_potential_1d(component="hartree")
    with pytest.raises(ValueError, match="unknown potential component"):
        calc.get_potential_2d(component="hartree")


def test_elnes_refuses_a_q_incommensurate_with_the_k_mesh(tmp_path):
    """src/elnes.f90 checks ``ngridk*q`` is integral and stops otherwise --
    the k+q state has to be another mesh point."""
    calc = make(tmp_path)  # ngridk = (2, 2, 2)
    with pytest.raises(ValueError, match="incommensurate"):
        calc.get_elnes(q=(0.25, 0.0, 0.0))


def test_electron_momentum_density_refuses_an_offset_mesh(tmp_path):
    """src/emdplot.f90 stops on any nonzero ``vkloff``."""
    calc = make(tmp_path, vkloff=(0.25, 0.5, 0.625))
    with pytest.raises(ValueError, match="vkloff"):
        calc.get_electron_momentum_density(dim=1)


def test_lsj_states_refuses_an_empty_list(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        make(tmp_path).get_lsj_states(kstlist=[])


def test_plot_dimension_is_validated(tmp_path):
    calc = make(tmp_path)
    with pytest.raises(ValueError, match="dim must be 1, 2 or 3"):
        calc.get_wavefunction(dim=4)
    with pytest.raises(ValueError, match="dim must be 1, 2 or 3"):
        calc.get_electric_field(dim=0)


# ------------------------------------------------------- atom filenames


def test_atom_filenames_follow_elks_species_and_atom_indexing(tmp_path):
    """src/bandstr.f90 / src/dos.f90 / src/wfcrplot.f90 all name their
    per-atom files ``"...S",I2.2,"_A",I4.4``, with the species index taken
    from the position in the ``atoms`` block."""
    s = Structure(AVEC, {"B": [(0.0, 0.0, 0.0)],
                         "N": [(0.33, 0.66, 0.0), (0.66, 0.33, 0.0)]})
    calc = SpectraCalculation(s, tmp_path / "hbn", ngridk=(2, 2, 2))
    names = calc._atom_filenames("BAND_S{species:02d}_A{atom:04d}.OUT")
    assert names[("B", 1)] == "BAND_S01_A0001.OUT"
    assert names[("N", 1)] == "BAND_S02_A0001.OUT"
    assert names[("N", 2)] == "BAND_S02_A0002.OUT"
