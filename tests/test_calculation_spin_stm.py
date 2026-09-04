"""Integration tests for Calculation.get_spin_stm() -- the spin-polarised
STM image (elkpy task 9003, patches/0011-spin-polarized-stm.patch, see
docs/design.md #30) -- against a real compiled elk binary.

The physics fixture is a freestanding Cr monolayer on the triangular lattice
of Cr/Ag(111), which orders in a coplanar 120-degree Neel state (Kurz,
Foerster, Nordstroem, Bihlmayer and Bluegel, PRB 69, 024415 (2004)); it is
Elk's own examples/magnetism/Cr-monolayer with the vacuum opened up so a tip
plane exists. This is the system SP-STM was proposed for in the first place
(Wortmann, Heinze, Kurz, Bihlmayer and Bluegel, PRL 86, 4132 (2001)) and has
since been simulated in detail (Palotas, Hofer and Szunyogh, PRB 84, 174428
(2011)).

Three chemically identical atoms with moments 120 degrees apart make this a
sharp test rather than a plausibility band:

- the spin-summed LDOS is IDENTICAL above all three atoms (they are related
  by a C3 rotation combined with a spin rotation, a symmetry Elk finds and
  enforces), while the tip projection is not -- that difference IS the
  SP-STM signal;
- the projection ratios over the three sublattices are fixed by the moment
  directions alone: 0 : +1 : -1 for a tip along x, -1 : +1/2 : +1/2 for a
  tip along y. Rotating the tip changes which sublattice is bright.
- the moments are coplanar in xy, so a tip along z gives identically zero --
  an exact null, and the check that the Cartesian component being projected
  is the requested one rather than, say, |m|.

The absolute normalisation is pinned separately against a genuinely
independent Elk code path: src/occupy.f90's own FERMIDOS.OUT, the density of
states at the Fermi energy, which must equal the unit-cell integral of the
spin-summed LDOS this task plots. (Upstream's task 162 fails that check by a
factor of the k-point weight, which it folds into occsv on top of the one
rhomagv already applies -- see elkpy_stm.f90's note.)

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.
"""

import numpy as np
import pytest

from elkpy import config, spec
from elkpy.parsers import volumetric
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

# Cr monolayer, triangular lattice with the Cr/Ag(111) nearest-neighbour
# distance 7.79/sqrt(2) Bohr, three atoms per cell (the sqrt(3) x sqrt(3)
# magnetic cell), 20 Bohr of vacuum along z so a tip plane can sit far from
# the layer's own periodic image. Same numbers as Elk's own
# examples/magnetism/Cr-monolayer, with scale3 opened up from 7 to 20.
A = 5.50836
C = 20.0
CR_AVEC = [(1.5 * A, 0.86602540378 * A, 0.0), (1.5 * A, -0.86602540378 * A, 0.0), (0.0, 0.0, C)]
# bfcmt seeds 120 degrees apart in the xy-plane (Cartesian, manual sec. 5.2);
# reducebf switches them off during the s.c. loop so the converged state is
# field-free. Elk's moments come out ANTIPARALLEL to the seed (measured, see
# docs/design.md #29), which is why the moment directions asserted below are
# the negatives of these.
CR_SPECIES = {
    "Cr": [
        ((0.0, 0.0, 0.0), (0.0, 0.1, 0.0)),
        ((1 / 3, 1 / 3, 0.0), (-0.086602540378, -0.05, 0.0)),
        ((2 / 3, 2 / 3, 0.0), (0.086602540378, -0.05, 0.0)),
    ]
}

HEIGHT = 0.25  # fractional -> 5 Bohr above the layer
NGRID = 24  # divisible by 3, so the three atoms land on grid points
SITES = (0, NGRID // 3, 2 * NGRID // 3)  # i1 = i2 index of each Cr atom
SWIDTH = 0.005


@pytest.fixture(scope="module")
def cr_calculation(tmp_path_factory):
    """Ground state of the Neel monolayer, shared by every test in this
    module (module-scoped: each get_spin_stm() call below still runs its own
    resumed Elk process, but the s.c. loop is paid for once)."""
    calc = Structure(CR_AVEC, CR_SPECIES).get_calculation(
        tmp_path_factory.mktemp("cr") / "cr",
        xc="PW",
        spinpol=True,
        ngridk=(4, 4, 1),
        # nempty: non-collinearity is resolved in the second-variational
        # step, so the empty-state count is load-bearing, not padding.
        # reducebf: switch the seeding fields off geometrically.
        extra_blocks={"nempty": [8], "reducebf": [0.5]},
    )
    calc.get_energy()
    return calc


@pytest.fixture(scope="module")
def stm_x(cr_calculation):
    return cr_calculation.get_spin_stm(
        direction=(1, 0, 0), height=HEIGHT, grid=(NGRID, NGRID), swidth=SWIDTH
    )


def site_values(field_grid):
    """The field at the three Cr sites, which sit at fractional (0,0),
    (1/3,1/3) and (2/3,2/3) -- i.e. on the grid diagonal."""
    return np.array([field_grid[i, i] for i in SITES])


def test_requires_spin_polarization(tmp_path):
    """Without spinpol/spinorb Elk computes no magnetisation density at all,
    so this must fail loudly in Python rather than launch a run that returns
    a meaningless zero projection."""
    calc = Structure(CR_AVEC, CR_SPECIES).get_calculation(tmp_path / "np", xc="PW")
    with pytest.raises(ValueError, match="spin-polarized"):
        calc.get_spin_stm(direction=(1, 0, 0))


def test_spin_summed_ldos_is_identical_above_every_atom(stm_x):
    """The three Cr atoms are chemically identical and related by a C3
    rotation combined with a 120-degree spin rotation, so the SPIN-SUMMED
    vacuum LDOS cannot distinguish them: a conventional STM sees a 1x1
    lattice. This is the baseline the spin contrast below is measured
    against -- if it did not hold, a difference in the projected image would
    not be magnetic in origin."""
    n = site_values(stm_x["ldos_grid"])
    assert np.ptp(n) / n.mean() < 1e-6


def test_tip_along_x_gives_the_neel_sublattice_pattern(stm_x):
    """With moments at -(0,1), -(-sqrt3/2,-1/2), -(sqrt3/2,-1/2) times |m|,
    a tip along x projects out 0, -sqrt3/2, +sqrt3/2 -- i.e. one sublattice
    dark and the other two of equal magnitude and OPPOSITE sign. That
    0 : +1 : -1 pattern is fixed by the 120-degree structure alone."""
    m = site_values(stm_x["spin_ldos_grid"])
    scale = np.abs(m).max()
    assert abs(m[0]) / scale < 1e-4
    assert m[1] / scale == pytest.approx(1.0, abs=1e-4)
    assert m[2] / scale == pytest.approx(-1.0, abs=1e-4)
    # ... and the contrast is a real signal, not numerical dust: the
    # spin-resolved modulation between sublattices is far larger than the
    # spin-summed image's own corrugation
    assert scale / stm_x["ldos_grid"].mean() > 0.05


def test_projection_vanishes_at_the_hollow_site(stm_x):
    """A site equidistant from all three sublattices sees the three moments
    at 120 degrees sum to zero, for ANY tip direction. (0, 1/3, 2/3) in the
    a1-a2 plane is such a point."""
    m = stm_x["spin_ldos_grid"][2 * NGRID // 3, NGRID // 3]
    assert abs(m) / np.abs(stm_x["spin_ldos_grid"]).max() < 1e-4


def test_tip_along_z_is_an_exact_null(cr_calculation):
    """The Neel state is coplanar in xy (without spin-orbit coupling the
    seed's m_z = 0 subspace is invariant under the Kohn-Sham flow), so the
    z-projection must vanish everywhere -- an exact null rather than a
    bound, and a direct check that the Cartesian component asked for is the
    one that comes back."""
    r = cr_calculation.get_spin_stm(
        direction=(0, 0, 1), height=HEIGHT, grid=(NGRID, NGRID), swidth=SWIDTH
    )
    assert np.abs(r["spin_ldos"]).max() / r["ldos"].mean() < 1e-4


def test_tip_along_y_reverses_which_sublattice_is_bright(cr_calculation, stm_x):
    """Rotating the tip by 90 degrees changes the projection ratios from
    0 : +1 : -1 to -1 : +1/2 : +1/2 -- the tip-direction dependence that
    identifies a non-collinear structure experimentally (Gao, Wulfhekel and
    Kirschner, PRL 101, 267205 (2008)). The spin-summed image is unchanged."""
    r = cr_calculation.get_spin_stm(
        direction=(0, 1, 0), height=HEIGHT, grid=(NGRID, NGRID), swidth=SWIDTH
    )
    m = site_values(r["spin_ldos_grid"])
    scale = np.abs(m).max()
    assert m[0] / scale == pytest.approx(-1.0, abs=1e-4)
    assert m[1] / scale == pytest.approx(0.5, abs=1e-4)
    assert m[2] / scale == pytest.approx(0.5, abs=1e-4)
    assert r["ldos"] == pytest.approx(stm_x["ldos"], rel=1e-5)


def test_arbitrary_direction_is_the_linear_projection(cr_calculation, stm_x):
    """A (1,1,1) tip -- the general, off-axis case -- must give exactly
    (m.x + m.y + m.z)/sqrt(3), since m . e is linear in e. Three separate
    Elk runs are combined here, so this also checks that the direction is
    normalised the same way every time."""
    d = cr_calculation.get_spin_stm(
        direction=(1, 1, 1), height=HEIGHT, grid=(NGRID, NGRID), swidth=SWIDTH
    )
    y = cr_calculation.get_spin_stm(
        direction=(0, 1, 0), height=HEIGHT, grid=(NGRID, NGRID), swidth=SWIDTH
    )
    z = cr_calculation.get_spin_stm(
        direction=(0, 0, 1), height=HEIGHT, grid=(NGRID, NGRID), swidth=SWIDTH
    )
    combined = (stm_x["spin_ldos"] + y["spin_ldos"] + z["spin_ldos"]) / np.sqrt(3)
    assert d["direction"] == pytest.approx(np.ones(3) / np.sqrt(3))
    assert d["spin_ldos"] == pytest.approx(combined, abs=1e-6 * np.abs(combined).max())


def test_spin_ldos_is_bounded_by_the_spin_summed_ldos(stm_x):
    """Elk's magnetisation is n_up - n_down (not the spin operator's
    <S> = m/2), so |m . e| <= n pointwise for a unit e -- the local spin
    polarisation the tip sees cannot exceed 1. A factor-of-two convention
    slip in the projection would break this."""
    assert (np.abs(stm_x["spin_ldos"]) <= stm_x["ldos"]).all()


def test_image_is_the_tip_polarisation_recombination(cr_calculation):
    """The third plotted field is n + P m.e by definition, so a run at
    P = 0.4 must reproduce it from its own first two fields -- which is what
    makes P a post-processing knob rather than a reason to re-run."""
    r = cr_calculation.get_spin_stm(
        direction=(1, 0, 0),
        height=HEIGHT,
        grid=(NGRID, NGRID),
        swidth=SWIDTH,
        polarization=0.4,
    )
    assert r["image"] == pytest.approx(r["ldos"] + 0.4 * r["spin_ldos"], rel=1e-9)


def test_integrated_mode_over_a_zero_width_window_is_exactly_zero(cr_calculation):
    """Constant-current mode weights each state by the smeared window between
    E_F and E_F + eV. At zero bias that window has zero width, so every field
    must be identically zero -- an exact null that pins the two smeared
    step functions against each other (a sign slip or a swapped argument in
    the pair would leave a finite residue), and that exercises the
    otherwise-untested branch at all."""
    r = cr_calculation.get_spin_stm(
        direction=(1, 0, 0),
        height=HEIGHT,
        grid=(NGRID, NGRID),
        swidth=SWIDTH,
        integrated=True,
        bias=0.0,
    )
    assert np.abs(r["ldos"]).max() == 0.0
    assert np.abs(r["spin_ldos"]).max() == 0.0
    assert r["dos"] == 0.0


def test_integrated_mode_counts_states_in_the_bias_window(cr_calculation, stm_x):
    """At a finite positive bias the same mode counts the empty states
    between E_F and E_F + eV, so the fields are positive and their cell
    integral is a state count rather than a density of states -- roughly the
    Fermi-level DOS times the window width for a window narrow enough that
    the DOS is still slowly varying."""
    bias = 0.02
    r = cr_calculation.get_spin_stm(
        direction=(1, 0, 0),
        height=HEIGHT,
        grid=(NGRID, NGRID),
        swidth=SWIDTH,
        integrated=True,
        bias=bias,
    )
    assert (r["ldos"] > 0).all()
    assert (np.abs(r["spin_ldos"]) <= r["ldos"]).all()
    assert r["energy"] == pytest.approx(r["efermi"] + bias)
    assert 0.2 < r["dos"] / (stm_x["dos"] * bias) < 5.0


def test_cell_integral_matches_elks_own_density_of_states(cr_calculation, stm_x):
    """At zero bias the unit-cell integral of the plotted spin-summed LDOS
    IS the density of states at the Fermi energy, which src/occupy.f90
    computes by an entirely separate route (a sum over eigenvalues, no
    charge density and no plotting machinery at all) and writes to
    FERMIDOS.OUT. Agreement pins the absolute normalisation, which every
    other test here -- all of them ratios or nulls -- leaves free. The
    tolerance is set by the density's own muffin-tin/interstitial
    representation error, the same one INFO.OUT reports for the total
    charge (order 1e-4 relative)."""
    # FERMIDOS.OUT comes from the run directory get_spin_stm() itself used --
    # the resumed task-1 loop writes it there with THIS call's swidth, which is
    # why the two smearings match. Every test in this module passes the same
    # SWIDTH, so whichever call wrote the directory last agrees; a call with a
    # different swidth would not, and would need its own subdirectory.
    fermidos = np.loadtxt(cr_calculation.workdir / "spin_stm" / spec.OUTPUT_FILES["fermidos"])
    assert stm_x["dos"] == pytest.approx(float(fermidos[-1]), rel=1e-3)


def test_matches_upstream_task_162_image_up_to_the_k_point_weight(cr_calculation):
    """Upstream Elk's own task 162 plots the same spin-summed vacuum LDOS by
    the same route (wfplot.f90: replace the occupations by a delta function
    at E_F, call rhomagv, plot the charge density), so the two images must
    agree pointwise -- except that task 162 folds the k-point weight into
    occsv on top of the one rhomagv already applies to it, making its image
    carry wkpt^2 (see elkpy_stm.f90's note; the same file's task-61/62/63
    branch sets occsv = 1/wkpt precisely to cancel that second factor).

    Both runs use reducek=0, so every weight is exactly 1/nkpt and the
    predicted discrepancy is a single number, nkpt -- which makes this not
    just a proportionality check but a quantitative confirmation of what the
    difference between the two routines IS. On a REDUCED mesh the weights
    differ between k-points and upstream's image is genuinely distorted, not
    merely rescaled.
    """
    plot = {
        "plot2d": cr_calculation._plot2d_lines(
            [(0, 0, HEIGHT), (1, 0, HEIGHT), (0, 1, HEIGHT)], (NGRID, NGRID)
        ),
        "swidth": [SWIDTH],
        "reducek": [0],
    }
    sub162 = cr_calculation.run_tasks([spec.TASKS["stm"]], plot, label="stm162")
    sub9003 = cr_calculation.run_tasks(
        [spec.TASKS["spin_stm_2d"]], {**plot, "elkpy_stmdir": [(1.0, 0.0, 0.0)]}, label="stm9003"
    )
    _, upstream, _ = volumetric.parse_plot2d(sub162 / spec.OUTPUT_FILES["stm_2d"])
    _, ours, _ = volumetric.parse_plot2d(sub9003 / spec.OUTPUT_FILES["spin_stm_2d"], nf=3)

    ratio = ours[:, 0] / upstream
    assert np.ptp(ratio) / ratio.mean() < 1e-6
    assert ratio.mean() == pytest.approx(4 * 4 * 1, rel=1e-6)  # nkpt, unreduced ngridk
