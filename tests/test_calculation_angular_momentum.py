"""Integration tests for Calculation.get_angular_momentum()/
EigenstateSession.angular_momentum() (task 9002's ANGMOM query, backed by
elkpy_angmomproj in src/elkpy_eigenstates.f90 -- patches/0006-angular-momentum.patch),
against a real compiled elk binary.

The physics test is monolayer WSe2 with spin-orbit coupling, the same
structure test_calculation_spin.py uses for the spin-valley-locking check.
Xiao, Liu, Feng, Xu & Yao ("Coupled Spin and Valley Physics in Monolayers of
MoS2 and Other Group-VI Dichalcogenides," Phys. Rev. Lett. 108, 196802
(2012), arXiv:1112.3144) -- the same paper already cited for the Sz(K) =
-Sz(K') check -- also gives the valence-band-top Bloch state's orbital
character in the group-VIB TMD family: predominantly transition-metal
d_{x^2-y^2} +- i*d_{xy} (i.e. m=+-2 in the complex-harmonic basis), with the
sign tied to the valley index tau=+-1 for K/K'. Since d_{x^2-y^2}+-i*d_{xy}
is (up to normalization) exactly Y_2^{+-2}, this is a sharp, literal
prediction for the ON-SITE (not Berry-curvature-derived) atomic orbital
angular momentum this feature computes: <Lz>_{W,d} should be large and of
opposite sign at K vs. K' -- a much more direct match to what
elkpy_angmomproj actually evaluates than a Berry-curvature-based "orbital
magnetization" argument would be (that is a different, k-space-derived
quantity, not the muffin-tin on-site operator here).

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.
"""

import re

import numpy as np
import pytest

from elkpy import config
from elkpy.session import ORBITAL_LABELS
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}

# Bohr, monolayer 2H-WSe2, same structure/convention as test_calculation_spin.py
# (ase.build.mx2(formula="WSe2", kind="2H", a=3.32, thickness=3.34, vacuum=12)).
A = 6.273890733757557
C = 51.66511224726855
WSE2_AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3**0.5 / 2, 0.0), (0.0, 0.0, C)]
WSE2_SPECIES = {
    "W": [(0.0, 0.0, 0.5)],
    "Se": [(2 / 3, 1 / 3, 0.5610826627651793), (2 / 3, 1 / 3, 0.43891733723482085)],
}

K = (1 / 3, 1 / 3, 0)
KPRIME = (-1 / 3, -1 / 3, 0)


@pytest.fixture
def si_calculation(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    return s.get_calculation(tmp_path / "si", xc="PW", ngridk=(4, 4, 4))


@pytest.fixture
def wse2_calculation(tmp_path):
    calc = Structure(WSE2_AVEC, WSE2_SPECIES).get_calculation(
        tmp_path / "wse2", xc="PW", ngridk=(3, 3, 1), rgkmax=7.0, spinorb=True
    )
    calc.get_energy()
    # occupied-band count from EIGVAL.OUT's own occupation numbers -- same
    # pitfall/fix as test_calculation_spin.py's wse2_calculation fixture.
    first_block = (calc.workdir / "EIGVAL.OUT").read_text().split("k-point")[1]
    state_line = re.compile(r"^\s*\d+\s+(\S+)\s+(\S+)\s*$")
    occ = [float(m.group(2)) for m in map(state_line.match, first_block.splitlines()) if m]
    ist1 = sum(o > 0.5 for o in occ)
    return calc, ist1


def test_orbital_labels_order():
    assert ORBITAL_LABELS == ("s", "p", "d", "f")


def test_matrices_are_hermitian_at_generic_k(si_calculation):
    """Each (atom, l, component) matrix is Hermitian by construction (a
    weighted Gram-type contraction against a Hermitian kernel -- see
    elkpy_angmomproj's docstring) -- checkable independent of any external
    reference data. Unlike get_atom_projection()/get_orbital_projection(),
    NOT expected to be positive semi-definite (an angular momentum
    expectation value can be negative)."""
    lm = si_calculation.get_angular_momentum((0.1, 0.2, 0.05), ist0=1, ist1=4)
    natmtot = lm.lx.shape[0]
    assert natmtot == 2  # two Si atoms
    assert lm.lx_orbital.shape == (2, 4, 4, 4)
    for name in ("lx", "ly", "lz"):
        totals = getattr(lm, name)
        by_l = getattr(lm, f"{name}_orbital")
        for a in range(natmtot):
            assert totals[a] == pytest.approx(totals[a].conj().T, abs=1e-8), f"{name} not Hermitian"
            # the headline total is the Python-side sum over l=0..3
            assert totals[a] == pytest.approx(by_l[a].sum(axis=0), abs=1e-12)
            for l in range(4):
                m = by_l[a, l]
                assert m == pytest.approx(m.conj().T, abs=1e-8), f"{name}_orbital l={l} not Hermitian"


def test_valence_band_orbital_angular_momentum_valley_locking(wse2_calculation):
    """<Lz> restricted to W's d channel, evaluated on the valence-band-top
    state, must be large in magnitude and of opposite sign at K vs. K'
    (Xiao, Liu, Feng, Xu & Yao, PRL 108, 196802 (2012)) -- a sign-of-the-
    effect prediction, the same spirit as this test suite's other K/K'
    checks (Berry curvature antisymmetry, spin-valley locking, quantum-
    metric parity).

    Measured: lz_d(K) = -1.1380, lz_d(K') = +1.1380, and W's own d-weight
    (get_orbital_projection()'s P_{W,d}) at K is 0.56900 -- to 5 decimal
    places, |lz_d(K)| = 2 * P_{W,d}(K) exactly. That is a much sharper
    statement than "large and sign-flipped": the cited paper's valence-band
    Bloch state, d_{x^2-y^2} -+ i*d_{xy} = Y_2^{-+2} (up to normalization),
    is a PURE m=+-2 state within the d-channel -- so its <Lz> should be
    exactly +-2 times whatever fraction of the wavefunction the d-channel
    itself carries, no smaller (which a mixture of several m values within
    d would give) and no larger (Lz's operator norm on l=2 is 2). This ties
    the new operator's absolute scale to an already-independently-verified
    quantity (P_{W,d}, itself cross-checked against upstream bandstr.f90 in
    test_calculation_orbital_projection.py) rather than to a bare
    plausibility threshold."""
    calc, ist1 = wse2_calculation
    w_index = calc.structure.atom_index("W")
    d_index = ORBITAL_LABELS.index("d")

    with calc.eigenstate_session() as session:
        lm_k = session.angular_momentum(K, ist1, ist1)
        lm_kprime = session.angular_momentum(KPRIME, ist1, ist1)
        d_weight_k = session.orbital_projection(K, ist1, ist1).matrices[w_index, d_index, 0, 0].real

    lz_k = lm_k.lz_orbital[w_index, d_index, 0, 0].real
    lz_kprime = lm_kprime.lz_orbital[w_index, d_index, 0, 0].real

    assert lz_k == pytest.approx(-lz_kprime, abs=1e-3)
    assert abs(lz_k) == pytest.approx(2 * d_weight_k, abs=1e-3)
    # Lx, Ly (off-diagonal ladder-operator content) vanish for a diagonal
    # matrix element of a pure Lz eigenstate -- consistent with (not an
    # independent test of) the pure-m=+-2 picture above.
    assert lm_k.lx_orbital[w_index, d_index, 0, 0].real == pytest.approx(0.0, abs=1e-5)
    assert lm_k.ly_orbital[w_index, d_index, 0, 0].real == pytest.approx(0.0, abs=1e-5)


def test_l0_s_channel_has_zero_angular_momentum(si_calculation):
    """The l=0 (s) channel is one-dimensional (m=0 only), so Lx, Ly, Lz are
    all exactly zero there -- a trivial but sharp regression check on the
    l-masking (elkpy_angmomproj's inner/outer wgt guard) independent of any
    external reference."""
    lm = si_calculation.get_angular_momentum((0.1, 0.2, 0.05), ist0=1, ist1=4)
    s_index = ORBITAL_LABELS.index("s")
    for name in ("lx_orbital", "ly_orbital", "lz_orbital"):
        by_l = getattr(lm, name)
        assert by_l[:, s_index] == pytest.approx(np.zeros_like(by_l[:, s_index]), abs=1e-10)
