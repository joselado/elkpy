"""The 3D Z2 classification of bulk Bi2Se3 from the Fu-Kane parity
indicator (docs/design.md #23), against a real compiled elk binary.

This settles the open question left at the end of docs/design.md #21. There,
a six-plane Wannier-charge-center sweep (get_z2_invariant_3d, #21) at
nkx=8/nt=5 returned nu0 = 0 for bulk Bi2Se3 -- the WRONG answer against the
well-established literature (1;000) (Zhang, Liu, Qi, Dai, Fang & Zhang,
Nature Physics 5, 438 (2009), arXiv:0812.1622, who obtain it by exactly this
parity counting; Fu & Kane, PRB 76, 045302 (2007) supply the method, their
own section IV applications being the diamond-lattice model, Bi(1-x)Sb(x),
gray tin and HgTe -- Bi2Se3 was not yet known to be a topological insulator).
Mesh convergence was never tested there because a materially denser sweep
would have cost many times the ~45 minutes already spent.

The parity indicator settles it exactly and in minutes, because Bi2Se3 is
centrosymmetric: 8 diagonalisations, no mesh at all. The answer is
(nu0; nu1 nu2 nu3) = (1; 000) -- Bi2Se3 IS a strong topological insulator,
and #21's nu0 = 0 was an under-converged crossing count, exactly the failure
mode docs/design.md #23 documented on the cesium structure. The WCC
implementation itself is not impugned (it agrees with the parity route in 2D
on graphene, and is separately validated on bismuthene); the six-plane 3D
sweep at a practical mesh is.

Two things make this a sharper confirmation of #23's thesis than the cesium
retraction was:

  * Bi2Se3 is a real material with an independently known answer, not a
    hypothetical structure with SOC scaled 3000x. No soc_scale is used here
    at all -- bismuth's own atomic spin-orbit coupling is the physics.
  * The gap is robust (0.26 eV at Gamma), so this is NOT the mesh-aliasing
    mode of docs/design.md #20 (graphene's ~1e-3-wide anticrossing at
    soc_scale=100). A confidently wrong integer came out of a well-resolved,
    well-gapped band structure.

The delta pattern carries physics beyond the invariant: exactly one TRIM has
the odd parity product and it is Gamma, the single zone-centre band
inversion that is the accepted mechanism for Bi2Se3's topology (the
p_z-derived states exchanging parity under SOC, Zhang et al. 2009). An odd
delta at a k_i = pi TRIM instead would have produced nonzero WEAK indices --
a different disagreement with the literature, pointing at the cell
orientation rather than the physics.

Structure: Crystallography Open Database entry 9011965 (digitizing
Nakajima's diffraction refinement, J. Phys. Chem. Solids 24, 479 (1963)),
reduced from its 15-atom R-3m hexagonal cell to the 5-atom rhombohedral
primitive cell with spglib.standardize_cell(to_primitive=True) and converted
through Structure.from_ase(). The numbers are hard-coded below rather than
re-derived at test time (no network, no spglib dependency in the test path).
Independently bond-length-verified before any DFT was run, reproducing
docs/design.md #21's own recorded values: 3.0747 / 2.8509 Angstrom Bi-Se
distances, 7.365 Angstrom quintuple-layer span (the 3D outer-Se-to-outer-Se
distance -- its projection on c is 6.966 Angstrom, and the two differ by
0.4 Angstrom because the outer Se are laterally offset by a/sqrt(3)), and a
2.5791 Angstrom van der Waals gap.

Skipped by default: the ground state alone is ~6.5 minutes (5 atoms,
all-electron, spin-orbit), plus ~1-3 minutes per band window. Set
ELKPY_RUN_SLOW_TESTS=1 to run. Also skipped if the elk binary hasn't been
built, same as test_calculation_si.py.
"""

import os
import re

import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = [
    pytest.mark.skipif(
        not config.default_elk_binary().is_file(),
        reason="elk binary not built; see docs/design.md #8",
    ),
    pytest.mark.skipif(
        os.environ.get("ELKPY_RUN_SLOW_TESTS") != "1",
        reason="~6.5 min ground state plus a parity sweep per window; "
        "set ELKPY_RUN_SLOW_TESTS=1 to run",
    ),
]

# Bohr. The rhombohedral primitive cell (a = 9.8405 Angstrom,
# alpha = 24.304 degrees) of COD 9011965's R-3m hexagonal cell
# (a = 4.143, c = 28.636 Angstrom), via spglib + Structure.from_ase().
BI2SE3_AVEC = [
    (3.914567667, 2.260076696, 18.038065768),
    (-3.914567667, 2.260076696, 18.038065768),
    (0.000000000, -4.520153393, 18.038065768),
]
# The quintuple layer Se-Bi-Se-Bi-Se: the central Se sits at the inversion
# centre (0,0,0), the other two species come in +-z pairs about it.
BI2SE3_SPECIES = {
    "Bi": [(0.4008, 0.4008, 0.4008), (0.5992, 0.5992, 0.5992)],
    "Se": [(0.0, 0.0, 0.0), (0.2117, 0.2117, 0.2117), (0.7883, 0.7883, 0.7883)],
}

_HARTREE_IN_EV = 27.211386245988


def _occupied_count_and_gap(calc):
    """Occupied-band count and the direct gap above it, from EIGVAL.OUT's own
    occupation numbers rather than an assumed electron count -- core
    electrons are not among the nstsv valence bands at all, the pitfall
    documented in docs/design.md #13. The first k-point block of a
    ngridk=(4,4,4) reduced mesh is Gamma."""
    first_block = (calc.workdir / "EIGVAL.OUT").read_text().split("k-point")[1]
    state_line = re.compile(r"^\s*\d+\s+(\S+)\s+(\S+)\s*$")
    rows = [
        (float(m.group(1)), float(m.group(2)))
        for m in map(state_line.match, first_block.splitlines())
        if m
    ]
    nocc = sum(occ > 0.5 for _, occ in rows)
    gap_ev = (rows[nocc][0] - rows[nocc - 1][0]) * _HARTREE_IN_EV
    return nocc, gap_ev


@pytest.fixture(scope="module")
def bi2se3(tmp_path_factory):
    """No soc_scale: bismuth's own atomic spin-orbit coupling is what opens
    the inverted gap, unlike the graphene (#20) and cesium (#21) fixtures
    where the real coupling is unresolvable and the scaling is a pure
    numerics knob."""
    workdir = tmp_path_factory.mktemp("bi2se3")
    calc = Structure(BI2SE3_AVEC, BI2SE3_SPECIES).get_calculation(
        workdir / "bi2se3", xc="PW", ngridk=(4, 4, 4), rgkmax=7.0, spinorb=True
    )
    calc.get_energy()
    nocc, gap_ev = _occupied_count_and_gap(calc)
    return calc, nocc, gap_ev


def test_ground_state_matches_the_one_section_21_used(bi2se3):
    """A gate on the comparison, not a physics claim: the parity result below
    is only a correction of docs/design.md #21's WCC result if it is computed
    from the same ground state. #21 recorded 78 occupied bands and a 0.258 eV
    direct gap at Gamma, the Brillouin-zone minimum along Gamma-Z-F-L-Gamma."""
    _calc, nocc, gap_ev = bi2se3
    assert nocc == 78
    assert gap_ev == pytest.approx(0.258, abs=0.01)


def test_bi2se3_is_a_strong_topological_insulator(bi2se3):
    """(nu0; nu1 nu2 nu3) = (1; 000), the literature answer -- correcting
    docs/design.md #21's nu0 = 0 from an under-converged six-plane WCC sweep
    on this same ground state.

    The delta pattern is asserted too, not just the invariant: exactly one
    TRIM must differ from the other seven, and it must be Gamma. That is the
    single zone-centre band inversion behind Bi2Se3's topology, and it is
    what makes the weak indices vanish -- an odd delta at a k_i = pi TRIM
    would give the same nu0 with nonzero (nu1, nu2, nu3).

    Only the RELATIVE pattern is asserted. All eight deltas may flip together
    depending on where the window starts (observed directly in
    test_parity_is_window_independent below), which cannot change any
    invariant here: every one is a product over an even number of TRIM. See
    parsers.symmetry's "sign immunity" note.
    """
    calc, ist1, _gap = bi2se3
    result = calc.get_fu_kane_invariant(1, ist1, dimension=3)
    assert result["nu0"] == 1
    assert result["nu"] == (0, 0, 0)

    deltas = result["deltas"]
    assert len(deltas) == 8
    gamma = deltas[(0.0, 0.0, 0.0)]
    others = [d for k, d in deltas.items() if k != (0.0, 0.0, 0.0)]
    assert all(d == -gamma for d in others)


def test_parity_is_window_independent(bi2se3):
    """Z2 is additive mod 2 over independently-gapped band groups, so
    dropping a gapped block off the bottom of the window cannot change the
    answer. Checked on two further legitimate windows, each starting above a
    real gap at all eight TRIM (measured: 22.5 eV below ist0=31, 4.9 eV below
    ist0=61). Windows starting at 13, 39, 51 and 57 were also run and also
    give (1; 000).

    ist0 = 61 is docs/design.md #21's own narrow window (bands 61-78, chosen
    there to rule out semicore contamination), which reproduced the wrong WCC
    answer on the one plane it was tested on. The parity route gives (1; 000)
    from it, so the discrepancy was never about the band window.

    ist0 = 31 additionally exercises the even-TRIM sign immunity on a real
    material: all eight of its deltas come out negated relative to the
    ist0 = 1 table above (Gamma at +1, the other seven at -1), with nu0 and
    the weak indices unchanged.
    """
    calc, ist1, _gap = bi2se3
    semicore_dropped = calc.get_fu_kane_invariant(31, ist1, dimension=3, label="fk_31")
    upper_valence = calc.get_fu_kane_invariant(61, ist1, dimension=3, label="fk_61")
    assert semicore_dropped["nu0"] == upper_valence["nu0"] == 1
    assert semicore_dropped["nu"] == upper_valence["nu"] == (0, 0, 0)
    # the global flip, asserted rather than merely tolerated
    assert semicore_dropped["deltas"][(0.0, 0.0, 0.0)] == 1
    assert upper_valence["deltas"][(0.0, 0.0, 0.0)] == -1
