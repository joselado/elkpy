"""The momentum-resolved tunnelling Fermi surface against a real compiled Elk
binary (elkpy task 9007 -- docs/design.md #35).

As with task 9005 one level down, no other code computes this quantity, so
the validation has to close inside the package. It does so by tying the new
task to the ALREADY-VERIFIED point-tip one through an exact identity rather
than a plausibility band:

    int_{tip cell} T(r; E) d^2r  =  sum_k w_k W(k; E),

the left-hand side from task 9005's real-space map and the right from task
9007's two Gram matrices. Nothing on the two sides is shared: 9005 samples
the tip plane on a plot2d grid and contracts point by point, 9007 collapses
the G-sphere in closed form and contracts one trace per k-point.

The rest, in the order they would catch a bug:

* the two tasks' EXIT Gram matrices must agree to round-off -- the closed-form
  collapse is deliberately duplicated in src/elkpy_fermitunnel.f90 rather than
  shared, so that the verified 9005 is not edited, and this is what stops the
  duplicate drifting;
* 9007's TIP Gram matrix against a real-space quadrature of 9005's amplitudes
  sampled on the same plane;
* both regions "cell" must give occmax sum_n delta_eta(E - eps) exactly, which
  is what pins occmax, the smearing type and the k-weights together;
* the physics: the weight must decay into the vacuum as exp(-2 kappa z) with
  kappa set by the state's own in-plane momentum. Graphene is the sharpest
  case -- its Fermi surface is only at K, so Gamma's weight is identically
  zero and K's decay rate can be held against |K| = 4 pi / 3a directly.
"""

import numpy as np
import pytest

from elkpy import Structure, config
from elkpy.parsers import fermitunnel as F, transport as T

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built (run ./build_elk.sh)",
)

_BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

# monolayer graphene, the same cell tests/test_calculation_transport.py uses
A = 2.46 * _BOHR_PER_ANGSTROM
VACUUM = 20.0
GRAPHENE_AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3**0.5 / 2, 0.0), (0.0, 0.0, VACUUM)]
GRAPHENE_SPECIES = {"C": [(0.0, 0.0, 0.5), (1 / 3, 2 / 3, 0.5)]}
EXIT, TIP = 0.38, 0.62          # 2.4 Bohr of vacuum to each plane, symmetric

# `tshift=False` is mandatory, not stylistic -- see test_calculation_transport
BASE = {"tshift": [False], "nempty": [10]}

# a mesh deliberately OFFSET onto generic k-points: at a high-symmetry point
# both Gram matrices are real to round-off and a transposed convention is
# indistinguishable from the right one
KGRID, KOFFSET = (2, 2, 1), (0.31, 0.17, 0.0)
BROADENING, WINDOW = 0.01, (-0.3, 0.3)


@pytest.fixture(scope="module")
def graphene(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("fermitunnel")
    calc = Structure(GRAPHENE_AVEC, GRAPHENE_SPECIES).get_calculation(
        workdir / "graphene", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
        spinpol=True, extra_blocks=dict(BASE),
    )
    calc.get_energy()
    return calc


@pytest.fixture(scope="module")
def planar(graphene):
    """Task 9007 on the two-plane junction geometry."""
    return graphene.get_tunnelling_fermi_surface(
        tip_height=TIP, exit_height=EXIT, kgrid=KGRID, koffset=KOFFSET,
        broadening=BROADENING, window=WINDOW,
    )


@pytest.fixture(scope="module")
def point_tip(graphene):
    """Task 9005 with its plot2d tip plane spanning the WHOLE cell at the same
    tip height, on the same k-mesh -- so its map can be integrated over the
    plane and compared with the planar-tip weight.

    The grid has to exceed twice the largest in-plane G index for the
    rectangle rule to be exact; 24 x 24 is comfortably past that at
    rgkmax = 7 for this cell, and `test_the_tip_gram_matrix_is_the_plane_
    quadrature` is what actually demonstrates it converged.
    """
    return graphene.get_vertical_transport(
        exit_height=EXIT, height=TIP, grid=(24, 24), kgrid=KGRID,
        koffset=KOFFSET, broadening=BROADENING, window=WINDOW,
    )


# --------------------------------------------------------------------------
# the identity that ties the new task to the verified one
# --------------------------------------------------------------------------

def test_the_planar_weight_is_the_plane_integral_of_the_point_tip_map(
        planar, point_tip):
    """int_cell T(r; E) d^2r == sum_k w_k W(k; E), the exact relation between
    task 9005 and task 9007.

    The two sides share no code: one samples psi at 576 points on the tip
    plane and contracts point by point, the other collapses the G-sphere onto
    its in-plane shadow in closed form and contracts one trace per k-point.
    A wrong normalisation, a wrong k-weight, a wrong spinor sum or a
    transposed contraction on either side breaks this.
    """
    area = point_tip["raw"]["area"]
    npts = point_tip["raw"]["npoints"]
    integrated = (area / npts) * point_tip["transmission"][0].sum()
    assert planar["total"][0] == pytest.approx(integrated, rel=1.0e-9)
    assert integrated > 0.0


def test_the_two_tasks_agree_on_the_exit_gram_matrix(planar, point_tip):
    """src/elkpy_fermitunnel.f90 carries its own copy of the closed-form Gram
    collapse rather than calling into the verified src/elkpy_transport.f90.
    This is what stops the copy drifting from the original."""
    checked = 0
    for mine, theirs in zip(planar["raw"]["overlaps"], point_tip["raw"]["overlaps"]):
        assert mine.shape == theirs.shape
        if mine.shape[0] == 0:
            continue
        assert np.abs(mine - theirs).max() < 1.0e-12 * np.abs(theirs).max()
        checked += 1
    assert checked > 0


def test_the_tip_gram_matrix_is_the_plane_quadrature(planar, point_tip):
    """The tip-plane Gram matrix, in closed form, against a literal rectangle
    rule over task 9005's own samples of the same states on the same plane."""
    checked = 0
    for mine, amplitude in zip(planar["raw"]["tip_overlaps"],
                               point_tip["raw"]["amplitudes"]):
        if mine.shape[0] == 0:
            continue
        quad = T.gram_matrix_by_quadrature(amplitude, point_tip["raw"]["area"])
        assert np.abs(mine - quad).max() < 1.0e-10 * np.abs(mine).max()
        checked += 1
    assert checked > 0


def test_the_contraction_is_the_literal_double_integral(planar, point_tip):
    """THE conjugation check on real Elk coefficients, one level up from
    task 9005's: Tr[D G_t D S] against int_tip int_exit |G|^2 built by
    quadrature from two DISTINCT planes, and the transposed form must differ.

    Task 9005's export samples the tip plane; its Gram matrix at the EXIT
    plane comes from the same run, so the exit side is taken in closed form
    and the tip side by quadrature -- which is exactly the mixture that fails
    if the conjugation sits on the wrong variable.
    """
    tested_wrong = False
    for gtip, sexit, amp, eig in zip(
        planar["raw"]["tip_overlaps"], planar["raw"]["overlaps"],
        point_tip["raw"]["amplitudes"], planar["raw"]["eigenvalues"],
    ):
        if gtip.shape[0] == 0:
            continue
        occmax = 2.0 if planar["raw"]["nspinor"] == 1 else 1.0
        g = T.amplitude_weights(
            eig, planar["raw"]["efermi"], BROADENING, occmax=occmax)
        fast = F.weight_at_k(gtip, sexit, g)
        # the tip side by quadrature over 9005's samples, the exit side in
        # closed form: the mixture a wrong conjugation cannot survive
        area, npts = point_tip["raw"]["area"], point_tip["raw"]["npoints"]
        tip_quad = (area / npts) * np.einsum(
            "snp,smp->nm", amp.conj(), amp, optimize=True)
        assert F.weight_at_k(tip_quad, sexit, g) == pytest.approx(fast, rel=1e-9)
        wrong = float(np.real(np.sum((g[:, None] * gtip * g[None, :]) * sexit)))
        if abs(wrong - fast) > 1.0e-8 * abs(fast):
            tested_wrong = True
    assert tested_wrong, "no k-point had an off-diagonal to distinguish the two"


# --------------------------------------------------------------------------
# the identity limits
# --------------------------------------------------------------------------

def test_both_cell_regions_give_the_plain_fermi_surface(graphene):
    """G_t = S = 1 must give occmax sum_n delta_eta(E - eps) exactly -- which
    pins occmax, Elk's smearing type and the assembly together -- and must
    equal the "bare" column every call returns whatever the regions are."""
    result = graphene.get_tunnelling_fermi_surface(
        tip_height=TIP, exit_height=EXIT, kgrid=(3, 3, 1),
        broadening=BROADENING, window=WINDOW,
        tip_region="cell", exit_region="cell",
    )
    raw = result["raw"]
    occmax = 2.0 if raw["nspinor"] == 1 else 1.0
    expect = np.array([
        occmax * float(np.sum(
            T.smeared_delta((raw["efermi"] - eig) / BROADENING, 3) / BROADENING
        )) if eig.size else 0.0
        for eig in raw["eigenvalues"]
    ])
    assert result["weight"][0] == pytest.approx(expect, rel=1.0e-12)
    assert result["bare"][0] == pytest.approx(expect, rel=1.0e-12)


def test_the_bare_column_does_not_depend_on_the_planes(graphene):
    """"bare" is the band structure alone, so moving the tip must not touch
    it while it moves the tunnelling weight by orders of magnitude. Sharing
    one export for both is the point of the quantity: the ratio is taken on
    identical k-points and identical eigenvalues."""
    near = graphene.get_tunnelling_fermi_surface(
        tip_height=0.58, exit_height=EXIT, kgrid=(3, 3, 1),
        broadening=BROADENING, window=WINDOW)
    far = graphene.get_tunnelling_fermi_surface(
        tip_height=0.66, exit_height=EXIT, kgrid=(3, 3, 1),
        broadening=BROADENING, window=WINDOW)
    assert near["bare"][0] == pytest.approx(far["bare"][0], rel=1.0e-12)
    assert far["weight"][0].max() < 0.5 * near["weight"][0].max()


# --------------------------------------------------------------------------
# the physics
# --------------------------------------------------------------------------

def test_the_vacuum_decay_is_set_by_the_in_plane_momentum(graphene):
    """kappa = sqrt(2(V0 - E) + |k + G|^2) is the entire reason this quantity
    differs from the plain Fermi surface, so it is measured rather than
    assumed.

    Graphene is the sharpest case available: its Fermi surface is ONLY at K,
    so with a narrow window there is nothing at Gamma at all -- the weight
    there is identically zero, not merely small -- and the K states decay into
    the vacuum at a rate the in-plane momentum |K| = 4 pi / 3a sets.

    The measured 2 kappa comes out ~6% BELOW the free-electron floor 2|K|
    (1.70 vs 1.80 Bohr^-1), and further still below sqrt(2(V0-E) + |K|^2) for
    graphene's ~4.6 eV work function: the interstitial plane-wave basis
    represents a vacuum tail that decays too slowly, the same finite-rgkmax
    limitation whose far-field end is the floor documented in
    docs/design.md #35. The bounds below are therefore deliberately loose
    around 2|K| -- they catch a wrong sign, wrong units or no decay at all,
    which is what a test can honestly claim here.
    """
    heights = (0.58, 0.61, 0.64, 0.67)
    narrow = dict(exit_height=EXIT, kgrid=(3, 3, 1), broadening=0.01,
                  window=(-0.1, 0.1), incoherent=False)
    kpoint, two_plane = [], []
    for h in heights:
        tip_only = graphene.get_tunnelling_fermi_surface(
            tip_height=h, exit_region="cell", **narrow)
        both = graphene.get_tunnelling_fermi_surface(tip_height=h, **narrow)
        k = tip_only["kpoints"]
        igamma = int(np.argmin(np.linalg.norm(k, axis=1)))
        ik = int(np.argmin(np.linalg.norm(k - [1 / 3, 1 / 3, 0], axis=1)))
        # graphene's Fermi surface is only at K: Gamma holds no state in the
        # window, so this is exactly zero rather than a small number
        assert tip_only["raw"]["eigenvalues"][igamma].size == 0
        assert tip_only["weight"][0][igamma] == 0.0
        assert tip_only["bare"][0][igamma] == 0.0
        kpoint.append(tip_only["weight"][0][ik])
        two_plane.append(both["weight"][0][ik])

    z = np.array(heights) * VACUUM
    logw = np.log(kpoint)
    slope, intercept = np.polyfit(z, logw, 1)
    # a clean exponential, not just a decay. Measured deviation is 2% of the
    # total decay: K carries four states here and their kappa differ slightly,
    # so the sum of exponentials is not exactly one exponential
    residual = np.abs(logw - (slope * z + intercept)).max()
    assert residual < 0.05 * (logw[0] - logw[-1])
    # the rate is the one |K| sets, to within the basis's own error
    twok = 2.0 * 4.0 * np.pi / (3.0 * A)
    assert 0.8 * twok < -slope < 2.0 * twok
    # moving the TIP leaves the exit plane's factor untouched, so the
    # two-plane weight must decay at the same rate -- the factorisation of
    # Tr[D G_t D S] into one factor per electrode, measured
    slope_two = np.polyfit(z, np.log(two_plane), 1)[0]
    assert slope_two == pytest.approx(slope, rel=0.02)
    # and it costs an EXACTLY constant factor -- measured 0.05242 at all four
    # heights to nine digits. That is the factorisation itself, not an
    # estimate of it: with the exit plane held still, S is literally the same
    # matrix at every tip height
    ratio = np.array(two_plane) / np.array(kpoint)
    assert ratio == pytest.approx(ratio[0], rel=1.0e-9)
    assert 0.0 < ratio[0] < 1.0
