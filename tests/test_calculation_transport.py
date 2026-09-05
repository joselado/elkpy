"""Vertical tunnelling transport through a 2D material, against a real
compiled Elk binary (elkpy task 9005 -- docs/design.md #31).

NEITHER Elk NOR any plane-wave DFT code in common use computes this quantity,
so there is no reference output anywhere to compare a number against, and the
whole of the validation has to close inside the package. That makes the choice
of checks the point:

* the **Tersoff-Hamann limit**, which is exact rather than approximate: widen
  the substrate from a plane to the whole cell and every S_k becomes the
  identity, so the transmission becomes the tunnelling density of states at
  the tip -- the SAME number get_spin_stm() (elkpy task 9003) reaches through
  `rhomagv`, a Fortran path sharing no line of code with this one. With no
  factor between them, which is what pins occmax, the k-weights, the spinor
  sum, the Omega^-1/2 normalisation and the tip sampler at once;
* the **closed-form Gram matrix against a real-space quadrature** of the same
  integral, and the **contraction against a literal evaluation of
  int |G(r,r')|^2** -- both on real Elk coefficients, using the task's own
  coincident-plane self-check configuration. The second is the only thing that
  discriminates a_n a*_m S[n,m] from a*_n a_m S[n,m] (see
  parsers/transport.py);
* the **physics**, as an exact symmetry statement rather than a plausibility
  band. Monolayer graphene's zero-energy states at K are a doublet, and the
  subgroup of K's little group that the tip/substrate geometry LEAVES INTACT
  is C3v = {E, 2C3, 3sigma_v} -- the mirror sigma_v exchanges the two
  sublattices without flipping z, so the doublet is still the 2D irrep E and
  by Schur's lemma S_K is a multiple of the identity. The substrate cannot
  tell the two partners apart, there is nothing off-diagonal to interfere
  through, and the transmission collapses onto the local density of states.
"""

import os

import numpy as np
import pytest

from elkpy import Structure, config
from elkpy.parsers import transport as T

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built (run ./build_elk.sh)",
)

_BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

# monolayer graphene, the same cell every other 2D fixture in this suite uses
A = 2.46 * _BOHR_PER_ANGSTROM
VACUUM = 20.0
GRAPHENE_AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3**0.5 / 2, 0.0), (0.0, 0.0, VACUUM)]
GRAPHENE_SPECIES = {"C": [(0.0, 0.0, 0.5), (1 / 3, 2 / 3, 0.5)]}
EXIT, TIP = 0.38, 0.62          # 2.4 Bohr of vacuum to each plane, symmetric

# `tshift=False` is mandatory, not stylistic: Elk otherwise relocates the
# origin onto the inversion centre (for a honeycomb, the bond midpoint) while
# both plotting planes stay in the input frame -- the same trap docs/design.md
# #28 documents for the rotation indicators. Here it puts the two planes on
# the wrong side of the sheet, which the task itself refuses.
BASE = {"tshift": [False], "nempty": [10]}


@pytest.fixture(scope="module")
def graphene(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("transport")
    calc = Structure(GRAPHENE_AVEC, GRAPHENE_SPECIES).get_calculation(
        workdir / "graphene", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
        spinpol=True, extra_blocks=dict(BASE),
    )
    calc.get_energy()
    return calc


@pytest.fixture(scope="module")
def coincident(graphene):
    """The task's own self-check configuration: tip plane = exit plane, so the
    exported tip amplitudes ARE a sampling of the exit plane and both the Gram
    matrix and the contraction can be checked against the definition."""
    # the mesh is deliberately OFFSET onto generic k-points: at a
    # high-symmetry point S_k is real to round-off, and the transposed
    # convention below is then indistinguishable from the right one
    return graphene.get_vertical_transport(
        exit_height=EXIT, height=EXIT, grid=(24, 24), kgrid=(2, 2, 1),
        koffset=(0.31, 0.17, 0.0), broadening=0.01, window=(-0.3, 0.3),
    )["raw"]


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def test_the_gram_matrix_is_hermitian_and_positive_semidefinite(graphene):
    """On real wavefunctions rather than random vectors. It being a Gram
    matrix is what makes the transmission non-negative with nothing clipping
    it."""
    result = graphene.get_vertical_transport(
        exit_height=EXIT, height=TIP, grid=(8, 8), kgrid=(3, 3, 1))
    assert result["hermiticity"] < 1.0e-12
    assert result["least_eigenvalue"] > -1.0e-12
    assert result["transmission"].min() > 0.0


def test_the_closed_form_gram_matrix_is_the_plane_quadrature(coincident):
    """The G-parallel orthogonality collapse src/elkpy_transport.f90 uses --
    one gather and one matrix product, no quadrature and no convergence
    parameter -- against a literal rectangle rule over the same plane."""
    checked = 0
    for overlap, amplitude in zip(coincident["overlaps"], coincident["amplitudes"]):
        if overlap.shape[0] == 0:
            continue
        quad = T.gram_matrix_by_quadrature(amplitude, coincident["area"])
        assert np.abs(overlap - quad).max() < 1.0e-12 * np.abs(overlap).max()
        checked += 1
    assert checked > 0


def test_the_contraction_is_the_literal_green_function_integral(coincident):
    """THE conjugation check, on real Elk coefficients: build
    G(r, r') = sum_n a_n(r) psi*_n(r') and integrate |G|^2 by quadrature.

    The transposed convention a*_n a_m S[n,m] is real, non-negative, blind to
    a degenerate rotation and exactly right in the Tersoff-Hamann limit, so
    nothing but this comparison separates the two.
    """
    tested_wrong = False
    for overlap, amplitude, eigenvalues in zip(
        coincident["overlaps"], coincident["amplitudes"], coincident["eigenvalues"]
    ):
        if overlap.shape[0] == 0:
            continue
        weights = T.amplitude_weights(eigenvalues, coincident["efermi"], 0.02)
        fast = T.transmission_at_k(amplitude, overlap, weights)
        literal = T.green_function_transmission(
            amplitude, amplitude, coincident["area"], weights)
        assert np.abs(fast - literal).max() < 1.0e-10 * fast.max()
        wrong = T.transmission_at_k(amplitude, overlap.T.copy(), weights)
        if np.abs(wrong - literal).max() > 1.0e-8 * fast.max():
            tested_wrong = True
    assert tested_wrong, "no k-point had an off-diagonal to distinguish the two"


# --------------------------------------------------------------------------
# the Tersoff-Hamann limit: the check that shares no code with what it checks
# --------------------------------------------------------------------------

def test_the_whole_cell_exit_region_is_the_tersoff_hamann_image(graphene):
    """S_k -> identity, and get_spin_stm()'s own local density of states comes
    back -- through `rhomagv` and Elk's plotting machinery on one side and an
    interstitial plane-wave sum on the other, with NO factor between them.

    The residual is the density representation's own error (the STM image
    interpolates `rhoir` off the coarse FFT grid where this evaluates the
    plane-wave sum exactly), the same ~1e-5 order docs/design.md #30 measures
    for its own FERMIDOS check.
    """
    geometry = dict(height=TIP, grid=(12, 12))
    ours = graphene.get_vertical_transport(
        exit_height=EXIT, kgrid=(6, 6, 1), broadening=0.005,
        exit_region="cell", incoherent=False, **geometry)["transmission"][0]
    reference = np.asarray(graphene.get_spin_stm(
        direction=(0, 0, 1), polarization=0.0, swidth=0.005,
        ngridk=(6, 6, 1), **geometry)["ldos"])
    assert np.abs(ours - reference).max() / reference.max() < 1.0e-4


# --------------------------------------------------------------------------
# the physics
# --------------------------------------------------------------------------

def test_the_dirac_doublet_sees_one_substrate_channel(graphene):
    """Schur's lemma at K, measured.

    The zero-energy pair spans the 2D irrep E of the geometry's own symmetry
    group C3v, so the exit-plane Gram matrix restricted to it must be a
    multiple of the identity -- and therefore the transport map must be the
    STM image. This is an exact statement about the two eigenvalues of a 2x2
    matrix, far sharper than the correlation it implies.
    """
    result = graphene.get_vertical_transport(
        exit_height=EXIT, height=TIP, grid=(6, 6), kgrid=(3, 3, 1),
        broadening=0.002, window=(-0.003, 0.003))
    data = result["raw"]
    doublets = 0
    for overlap, eigenvalues in zip(data["overlaps"], data["eigenvalues"]):
        if overlap.shape[0] == 0:
            continue
        # nspinor = 2 on this fixture, so the Dirac pair appears as four
        # states: the two sublattice partners times a spin degeneracy that is
        # exact here (no SOC, zero converged moment)
        assert overlap.shape[0] == 2 * data["nspinor"]
        assert np.ptp(eigenvalues) < 1.0e-5                # degenerate at K
        spectrum = np.linalg.eigvalsh(0.5 * (overlap + overlap.conj().T))
        assert np.ptp(spectrum) / spectrum.mean() < 1.0e-4
        doublets += 1
    assert doublets == 2, "a 3x3x1 mesh carries K and K'"
    # two equal channels, and nothing off-diagonal for a current to interfere
    # through -- so the coherent and incoherent maps coincide
    assert abs(result["channels"] - 2.0 * data["nspinor"]) < 1.0e-6
    assert result["offdiagonal_weight"] < 1.0e-6
    assert np.abs(result["interference"]).max() < 1.0e-8 * result["transmission"].max()


def test_the_monolayer_transport_map_is_its_tunnelling_image(graphene):
    """The consequence of the theorem above, as the map an experiment would
    see: for a single sheet, "what gets through" and "what the tip sees" are
    the same picture."""
    geometry = dict(height=TIP, grid=(12, 12), kgrid=(6, 6, 1), broadening=0.005)
    flow = graphene.get_vertical_transport(exit_height=EXIT, **geometry)
    local = graphene.get_vertical_transport(
        exit_height=EXIT, exit_region="cell", incoherent=False, **geometry)
    correlation = np.corrcoef(flow["transmission"][0],
                              local["transmission"][0])[0, 1]
    assert correlation > 0.999


# --------------------------------------------------------------------------
# the other half of the physics: a bilayer, where the same symmetry argument
# fails for a reason that can be named
# --------------------------------------------------------------------------

# AB (Bernal) bilayer graphene: graphite's own interlayer spacing, layer 2
# shifted so that one of its atoms sits directly over one of layer 1's. Do NOT
# relax this -- PBE has no interlayer minimum at all, and LDA reproduces
# 3.35 A only fortuitously; the K-point layer polarization below is exact for
# any spacing anyway.
BILAYER_C = 32.0
_D = 3.35 * _BOHR_PER_ANGSTROM
_Z1, _Z2 = 0.5 - _D / (2 * BILAYER_C), 0.5 + _D / (2 * BILAYER_C)
BILAYER_AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3**0.5 / 2, 0.0), (0.0, 0.0, BILAYER_C)]
BILAYER_SPECIES = {"C": [(0.0, 0.0, _Z1), (1 / 3, 2 / 3, _Z1),
                         (1 / 3, 2 / 3, _Z2), (2 / 3, 1 / 3, _Z2)]}
# the same 2.5 Bohr standoff on both sides, which is what makes the two
# layer-polarized states' TIP weights comparable and isolates the substrate
_GAP = 2.5 / BILAYER_C
BILAYER_EXIT, BILAYER_TIP = _Z1 - _GAP, _Z2 + _GAP


@pytest.mark.skipif(
    os.environ.get("ELKPY_RUN_SLOW_TESTS") != "1",
    reason="4-atom bilayer ground state plus two transport sweeps, ~7 minutes",
)
def test_the_bilayer_substrate_tells_the_two_partners_apart(tmp_path_factory):
    """The AB bilayer's zero-energy pair at K is degenerate for a symmetry
    reason too -- but the element that exchanges its members is an in-plane
    C2', which FLIPS z and is therefore broken by having a tip above and a
    substrate below. The intact subgroup is C3 alone, which is abelian, so
    Schur's lemma gives nothing: the two states are the non-dimer sites of the
    two DIFFERENT layers, and the substrate below sees them through very
    different amounts of carbon.

    Measured against the monolayer's 1.0000002, the ratio of S_K's two
    eigenvalues here is of order a thousand. This is an exact statement about
    a 2x2 matrix, which is why it is asserted rather than the correlation it
    implies.
    """
    workdir = tmp_path_factory.mktemp("transport_bilayer")
    calc = Structure(BILAYER_AVEC, BILAYER_SPECIES).get_calculation(
        workdir / "bilayer", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
        extra_blocks={"tshift": [False], "nempty": [12]},
    )
    calc.get_energy()

    sharp = calc.get_vertical_transport(
        exit_height=BILAYER_EXIT, height=BILAYER_TIP, grid=(6, 6),
        kgrid=(3, 3, 1), broadening=0.002, window=(-0.004, 0.004))["raw"]
    doublets = 0
    for overlap, eigenvalues in zip(sharp["overlaps"], sharp["eigenvalues"]):
        if overlap.shape[0] == 0:
            continue
        assert overlap.shape[0] == 2
        assert abs(eigenvalues[1] - eigenvalues[0]) < 1.0e-5
        spectrum = np.linalg.eigvalsh(0.5 * (overlap + overlap.conj().T))
        assert spectrum.max() / spectrum.min() > 100.0
        doublets += 1
    assert doublets == 2

    geometry = dict(height=BILAYER_TIP, grid=(12, 12), kgrid=(6, 6, 1),
                    broadening=0.005)
    flow = calc.get_vertical_transport(exit_height=BILAYER_EXIT, **geometry)
    local = calc.get_vertical_transport(
        exit_height=BILAYER_EXIT, exit_region="cell", incoherent=False,
        **geometry)
    # half of S_k now sits off its diagonal, where the monolayer had nothing
    # there at all, and the map departs from any local picture
    assert flow["offdiagonal_weight"] > 0.2
    assert np.corrcoef(flow["transmission"][0],
                       local["transmission"][0])[0, 1] < 0.95
    assert np.abs(flow["interference"]).max() > 0.1 * flow["transmission"].max()


# --------------------------------------------------------------------------
# the geometry the whole construction rests on
# --------------------------------------------------------------------------

def test_tshift_is_required(tmp_path):
    """Refused in Python, before any Elk run: with Elk free to relocate the
    origin, the planes and the atoms live in different frames."""
    calc = Structure(GRAPHENE_AVEC, GRAPHENE_SPECIES).get_calculation(
        tmp_path / "noshift", xc="PW", ngridk=(2, 2, 1), rgkmax=6.0)
    with pytest.raises(ValueError, match="tshift"):
        calc.get_vertical_transport(exit_height=EXIT, height=TIP)


def test_a_plane_inside_a_muffin_tin_sphere_is_refused(graphene):
    """Both formulas use the interstitial plane-wave representation, which is
    NOT the wavefunction inside a sphere -- and would return a perfectly
    plausible number there."""
    with pytest.raises(RuntimeError, match="muffin-tin"):
        graphene.get_vertical_transport(exit_height=0.5, height=TIP, grid=(4, 4),
                                        kgrid=(1, 1, 1))


def test_planes_on_the_wrong_side_of_the_material_are_refused(graphene):
    """A cell is periodic, so "above" and "below" are only meaningful relative
    to where the atoms are: with both planes below the sheet nothing tunnels
    through it."""
    # going up from the exit plane at 0.30 one meets the tip at 0.40 BEFORE
    # the sheet at 0.50, so nothing tunnels through anything. (Note 0.40/0.35
    # would be perfectly valid: the cell is periodic, so a tip below the exit
    # plane is a tip above the sheet, reached the other way round.)
    with pytest.raises(RuntimeError, match="between the two planes"):
        graphene.get_vertical_transport(exit_height=0.30, height=0.40,
                                        grid=(4, 4), kgrid=(1, 1, 1))
