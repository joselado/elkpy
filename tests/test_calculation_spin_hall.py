"""Integration tests for the spin Berry curvature / spin Hall machinery
(elkpy.parsers.spin_hall, plus the evecsv patches/0009 adds to the task-9002
MOMENTUM query), against a real compiled elk binary.

Two things are being checked here that the synthetic pins in
tests/test_parsers_spin_hall.py structurally cannot:

  * that the MOMENTUM response's NEW evecsv block is really this
    diagonalisation's second-variational eigenvector matrix -- unitary, and
    describing the same spin content as the EIGENSTATES query's own evecsv
    at the same k;
  * a sign-of-the-effect physics prediction on a real band structure.

The fixture is the graphene + soc_scale=3000 system tests/
test_calculation_z2.py already uses (Kane & Mele, PRL 95, 146802 (2005);
real carbon SOC is ~1 microeV and unresolvable on any practical mesh, so
the enhancement is a numerics knob -- see docs/design.md #20). It is the
right fixture here because graphene has BOTH inversion symmetry and
time-reversal symmetry, which forces the ordinary (charge) Berry curvature
to vanish pointwise while leaving the SPIN Berry curvature free to be
large: a spin Hall response with no anomalous Hall response at all.

The K/K' comparison is the sharp part. Charge Berry curvature is
time-reversal ODD (Omega(-k) = -Omega(k)) while the spin Berry curvature
is EVEN -- the spin current operator J^z_a = (1/2){S_z, v_a} picks up two
sign flips where v alone picks up one, so they cancel (the same holds
under inversion, which graphene also has, so both symmetries agree here).
So Omega^s(K) = +Omega^s(K'), the OPPOSITE relative sign to the K/K'
antisymmetry every other K/K' test in this suite asserts (docs/design.md
#13's Berry curvature, #17's S_z, #19's L_z, #22's circular dichroism) --
and the reason a spin Hall response is allowed in a
time-reversal-symmetric crystal where an anomalous Hall response is not.

Note what the failure modes look like: if the two operator factors were
swapped, or S silently dropped, this would return the CHARGE curvature,
i.e. zero. So the magnitude assertion is as diagnostic as the sign one.

Band windows are EXPLICIT and gap-checked here rather than read off
occupation numbers, which matters for this particular fixture: scaling
carbon's SOC by 3000 reorders the bands so drastically that the cell's 8
valence electrons no longer fill a fixed 8 states at every k-point (Elk's
own occupancies show 6 occupied at the first k-point), so an
occupancy-derived window would not be a well-defined band group. Every
quantity here is a property of a gapped group's projector, so a gap-checked
group is what the test uses -- and the gap is asserted, not assumed.

Deliberately NOT asserted: an elementwise comparison of S_z built from the
MOMENTUM evecsv against EigenstateSession.spin_operator()'s own. Graphene
with inversion + time-reversal symmetry is Kramers degenerate at EVERY
k-point, so two independent diagonalisations resolve each doublet in
arbitrary, equally valid bases (docs/design.md #14) and an elementwise
comparison would fail with nothing wrong. The basis-INVARIANT comparison
(the eigenvalue spectrum of S_z over a window) is what is checked instead
-- and that invariance is exactly why patch 0009 was needed: it is also
why no cheap check could have caught the mismatched-basis product this
patch exists to prevent.

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.
"""

import numpy as np
import pytest

from elkpy import config
from elkpy.parsers import optical, spin_hall
from elkpy.parsers.spin import compute_spin_operator
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

_BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
_HARTREE_EV = 27.211386245988
A = 2.46 * _BOHR_PER_ANGSTROM
VACUUM = 20.0
GRAPHENE_AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3**0.5 / 2, 0.0), (0.0, 0.0, VACUUM)]
GRAPHENE_SPECIES = {"C": [(0.0, 0.0, 0.5), (1 / 3, 2 / 3, 0.5)]}

K = (1 / 3, 1 / 3, 0.0)
KPRIME = (-1 / 3, -1 / 3, 0.0)

# Two independently-gapped band groups of this SOC-scaled band structure,
# both closed under Kramers pairing. The boundary gap of each is asserted
# below rather than assumed. [1, 4] is where the spin curvature is large
# (a ~1.25 eV boundary gap, so the 1/(delta eps)^2 weight is substantial);
# [1, 6] sits below a ~10 eV gap, which suppresses it by two orders of
# magnitude -- included precisely so the K/K' symmetry is checked at both
# scales, not only where the effect is big.
WINDOWS = ((1, 4), (1, 6))
MIN_BOUNDARY_GAP_EV = 0.5


@pytest.fixture(scope="module")
def graphene_soc(tmp_path_factory):
    """Monolayer graphene with intrinsic SOC scaled 3000x, with nempty
    raised so the Kubo sums have unoccupied states to run over."""
    workdir = tmp_path_factory.mktemp("spin_hall")
    calc = Structure(GRAPHENE_AVEC, GRAPHENE_SPECIES).get_calculation(
        workdir / "graphene_soc",
        xc="PW",
        ngridk=(6, 6, 1),
        rgkmax=7.0,
        spinorb=True,
        soc_scale={"C": 3000.0},
        extra_blocks={"nempty": [12]},
    )
    calc.get_energy()
    return calc


def test_momentum_evecsv_is_unitary(graphene_soc):
    """The new evecsv block of the MOMENTUM response must be a genuine
    second-variational eigenvector matrix: eigenvectors of one Hermitian
    problem, hence exactly orthonormal (machine precision, not the ~1e-3
    genolpq truncation floor overlaps carry -- docs/design.md #14).

    This is the cheapest check that the block was packed with the right
    shape and ordering at all: a transposed or misaligned column-major
    read would generally not be unitary."""
    with graphene_soc.eigenstate_session(label="evecsv_unitary") as session:
        m = session.momentum((0.13, 0.21, 0.0))  # generic, low symmetry
    nstsv = m.evecsv.shape[0]
    assert m.evecsv.shape == (nstsv, nstsv)
    assert m.pmat.shape == (3, nstsv, nstsv)
    assert np.allclose(m.evecsv.conj().T @ m.evecsv, np.eye(nstsv), atol=1e-10)


def test_momentum_evecsv_gives_the_same_spin_spectrum_as_the_eigenstates_query(
    graphene_soc,
):
    """S_z built from the MOMENTUM response's evecsv must describe the
    same physical spin content as S_z from an EIGENSTATES query at the
    same k -- but only up to the unitary freedom inside each degenerate
    multiplet, which graphene (inversion + time reversal, so Kramers
    degenerate everywhere) has at every k-point.

    So the comparison is of the SPECTRUM of S_z over the window, a basis
    invariant, not of the matrix elements. It also pins the energies: two
    diagonalisations of the same Hermitian problem must agree on those
    exactly.
    """
    k = (0.13, 0.21, 0.0)
    with graphene_soc.eigenstate_session(label="evecsv_spectrum") as session:
        m = session.momentum(k)
        state = session.get_eigenstates(k)
    nstfv = m.evecsv.shape[0] // 2
    ist0, ist1 = WINDOWS[-1]
    sz_mom = compute_spin_operator(m.evecsv, nstfv, ist0, ist1)["sz"]
    sz_eig = compute_spin_operator(state.evecsv, nstfv, ist0, ist1)["sz"]
    assert m.energies == pytest.approx(state.energies, abs=1e-10)
    assert np.linalg.eigvalsh(sz_mom) == pytest.approx(
        np.linalg.eigvalsh(sz_eig), abs=1e-6
    )
    # and it is a real spin operator, not a near-zero artefact: with SOC
    # this strong the window carries substantial |S_z|
    assert np.max(np.abs(np.linalg.eigvalsh(sz_mom))) > 0.1


def _curvatures(session, k):
    """Charge and spin Berry curvature of each WINDOWS entry at one k, plus
    that window's boundary gap, all from ONE MOMENTUM query."""
    m, j = session.spin_current_operator(k, direction=1, spin="z")
    assert np.allclose(j, j.conj().T, atol=1e-8), f"spin current not Hermitian at {k}"
    nstsv = m.evecsv.shape[0]
    sz = compute_spin_operator(m.evecsv, nstsv // 2, 1, nstsv)["sz"]
    out = {}
    for ist0, ist1 in WINDOWS:
        out[(ist0, ist1)] = {
            "gap_ev": (m.energies[ist1] - m.energies[ist1 - 1]) * _HARTREE_EV,
            "charge": optical.kubo_berry_curvature(
                m.energies, m.pmat, ist0, ist1, directions=(1, 2)
            ),
            "spin": spin_hall.spin_berry_curvature(
                m.energies, m.pmat, sz, ist0, ist1, directions=(1, 2)
            ),
        }
    return out


def test_spin_curvature_is_valley_symmetric_where_charge_curvature_vanishes(
    graphene_soc,
):
    """The physics check. Graphene has inversion AND time-reversal
    symmetry, whose product forces the charge Berry curvature to vanish
    pointwise, Omega(k) = 0 for all k. The spin Berry curvature is under
    no such constraint: it is EVEN under each (J^z_a = (1/2){S_z, v_a}
    picks up two sign flips where v alone picks up one), so
    Omega^s(K) = +Omega^s(K') rather than the antisymmetry every other
    K/K' test in this suite asserts.

    Together: a large, valley-SYMMETRIC spin Berry curvature sitting on
    top of a vanishing, valley-ANTIsymmetric charge one -- the defining
    signature of a quantum spin Hall system, and a pattern no sign
    convention or basis ambiguity can produce by accident.

    Asserted for BOTH gapped groups, whose spin curvatures differ by two
    orders of magnitude, so the symmetry claim is not carried by one
    fortunate window. The ABSOLUTE sign of Omega^s is a regression pin,
    not a prediction: it depends on this structure's own conventions
    (lattice handedness, which evecsv row block is physically "up" --
    docs/design.md #17), exactly as with S_z(K) there.
    """
    with graphene_soc.eigenstate_session(label="valley_spin_curvature") as session:
        at_k = _curvatures(session, K)
        at_kp = _curvatures(session, KPRIME)

    big = WINDOWS[0]
    for window in WINDOWS:
        rk, rkp = at_k[window], at_kp[window]
        # the window really is a separated band group at both valleys
        assert rk["gap_ev"] > MIN_BOUNDARY_GAP_EV
        assert rkp["gap_ev"] > MIN_BOUNDARY_GAP_EV
        # spin Berry curvature is valley-SYMMETRIC (even under T and under
        # inversion), to much better than the 1e-3 asserted here
        assert rk["spin"] == pytest.approx(rkp["spin"], rel=1e-3)
        # the charge curvature is forbidden by inversion x time reversal
        # and is numerical noise about zero, at both valleys
        assert abs(rk["charge"]) < 1e-2
        assert abs(rkp["charge"]) < 1e-2

    # and where the boundary gap is ~1 eV rather than ~10, the spin
    # curvature is large in absolute terms -- what a swapped J/v (which
    # would return the vanishing charge curvature instead) cannot fake
    assert abs(at_k[big]["spin"]) > 10.0
    assert abs(at_k[big]["spin"]) > 100 * abs(at_k[big]["charge"])


def test_spin_hall_conductivity_integrates_a_mesh_of_curvatures(graphene_soc):
    """End-to-end shape/units smoke test of the Brillouin-zone integral:
    a coarse mesh of spin Berry curvatures folded into a conductivity.

    Deliberately NOT a converged number and not compared to a literature
    value -- the spin Berry curvature is sharply peaked near avoided
    crossings, so a mesh this coarse cannot resolve the integral (the same
    convergence problem docs/design.md #20 documents for the Z2 mesh).
    What is asserted is only that the pipeline runs on real data and that
    the helper's normalization is the documented one.
    """
    ist0, ist1 = WINDOWS[0]
    nk = 4
    curvatures = []
    with graphene_soc.eigenstate_session(label="shc_mesh") as session:
        for i in range(nk):
            for j in range(nk):
                # offset off the high-symmetry mesh so no sampled point
                # lands exactly on a degeneracy
                k = ((i + 0.37) / nk, (j + 0.19) / nk, 0.0)
                m = session.momentum(k)
                nstsv = m.evecsv.shape[0]
                sz = compute_spin_operator(m.evecsv, nstsv // 2, 1, nstsv)["sz"]
                curvatures.append(
                    spin_hall.spin_berry_curvature(m.energies, m.pmat, sz, ist0, ist1)
                )
    assert len(curvatures) == nk * nk
    assert all(np.isfinite(c) for c in curvatures)
    volume = abs(np.linalg.det(np.array(GRAPHENE_AVEC)))
    sigma = spin_hall.spin_hall_conductivity(curvatures, cell_volume=volume)
    assert sigma == pytest.approx(float(np.mean(curvatures)) / volume, rel=1e-12)
