"""Integration tests for the optical absorption spectrum, against a real
compiled elk binary: Calculation.get_dielectric_function() (Elk's own task
121, src/dielectric.f90) and Calculation.get_circular_absorption()
(elkpy's polarization-resolved spectrum, built in Python from the
task-9002 MOMENTUM query -- docs/design.md #24).

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.

The headline check is the second one: the polarization-SUMMED elkpy
spectrum against Elk's own Im eps_xx + Im eps_yy on the same mesh, the
same nempty and the same swidth. That is an entirely independent Fortran
implementation of the same Kubo-Greenwood physics, reading momentum matrix
elements off PMAT.OUT (task 120) on the reduced mesh and rotating them,
where elkpy re-diagonalises at every non-reduced point and does all the
arithmetic in NumPy -- the project's standard cross-check idiom
(cf. #22's Kubo-vs-Wilson-loop curvature, #18's projector-vs-BAND_S*.OUT).

Monolayer h-BN, the same slab the Berry-curvature/momentum tests use.
"""

import numpy as np
import pytest

from elkpy import config
from elkpy.parsers import optical
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

A, VACUUM = 4.743210000, 20.0  # Bohr, same hBN slab as tests/test_calculation_momentum.py
HBN_AVEC = [(A, 0.0, 0.0), (-A / 2, A * 3**0.5 / 2, 0.0), (0.0, 0.0, VACUUM)]
HBN_SPECIES = {"B": [(1 / 3, 2 / 3, 0.5)], "N": [(2 / 3, 1 / 3, 0.5)]}

K = (1 / 3, 1 / 3, 0.0)
KPRIME = (-1 / 3, -1 / 3, 0.0)

NGRIDK = (6, 6, 1)
NEMPTY = 12       # both sides must see the same nstsv, so this lives on the Calculation
SWIDTH = 0.005    # Hartree; Elk's relaxation-time broadening, matched on both sides
WPLOT = (0.0, 0.5)
NWPLOT = 500


@pytest.fixture(scope="module")
def hbn(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("absorption")
    calc = Structure(HBN_AVEC, HBN_SPECIES).get_calculation(
        workdir / "hbn", xc="PW", ngridk=NGRIDK, rgkmax=7.0,
        extra_blocks={"nempty": [NEMPTY]},
    )
    calc.get_energy()
    return calc


@pytest.fixture(scope="module")
def elk_dielectric(hbn):
    """Elk task 121's own dielectric tensor -- the xx and yy components,
    both from one run so no isotropy assumption enters the comparison."""
    return hbn.get_dielectric_function(
        components=((1, 1), (2, 2)), wplot=WPLOT, nwplot=NWPLOT, swidth=SWIDTH,
    )


@pytest.fixture(scope="module")
def elkpy_absorption(hbn):
    return hbn.get_circular_absorption(
        wplot=WPLOT, nwplot=NWPLOT, swidth=SWIDTH,
    )


def test_task_121_gives_a_physical_absorption_spectrum(hbn, elk_dielectric):
    """Sanity on the wrapper itself before it is used as a reference: the
    spectrum is real-valued and non-negative, vanishes below the gap, and
    the in-plane components agree with each other (h-BN's three-fold axis
    makes the in-plane dielectric tensor isotropic, which Elk computes
    twice over without ever being told so)."""
    w = elk_dielectric["energies"]
    eps_xx = elk_dielectric["epsilon"][(1, 1)]
    eps_yy = elk_dielectric["epsilon"][(2, 2)]
    assert w[0] == 0.0 and w[-1] < WPLOT[1]  # Elk's grid excludes the upper endpoint
    assert eps_xx.imag.min() > -1e-8
    assert eps_xx.imag.max() > 1.0
    # essentially nothing absorbs below the Kohn-Sham gap: what is left
    # there is the Lorentzian tail of the band-edge transitions, not a
    # separate feature
    assert eps_xx.imag[w < 0.10].max() < 0.01 * eps_xx.imag.max()
    # in-plane isotropy, which Elk is never told about
    assert np.abs(eps_xx.imag - eps_yy.imag).max() < 1e-3 * eps_xx.imag.max()


def test_circular_channels_sum_to_elks_own_dielectric_function(
    elk_dielectric, elkpy_absorption
):
    """THE cross-check. |P_+|^2 + |P_-|^2 = 2(|p^x|^2 + |p^y|^2) makes the
    polarization-summed circular spectrum identically Im eps_xx + Im
    eps_yy, so elkpy's Python sum over the non-reduced mesh must reproduce
    Elk's Fortran one point by point.

    Nothing but the arithmetic differs: same ground state, same k-mesh,
    same nempty (hence the same nstsv truncation), same broadening, same
    energy grid, occupations read from the same EIGVAL.OUT -- but Elk sums
    in Fortran over reduced-mesh momentum matrix elements read back from
    PMAT.OUT and rotated by getpmat, where elkpy re-diagonalises at every
    non-reduced point through the task-9002 session and sums in NumPy.
    """
    w = elk_dielectric["energies"]
    reference = elk_dielectric["epsilon"][(1, 1)].imag + elk_dielectric["epsilon"][(2, 2)].imag
    ours = elkpy_absorption["eps2_total"]
    assert np.allclose(w, elkpy_absorption["omega"])

    mask = reference > 0.02 * reference.max()
    assert mask.sum() > 100
    relative = np.abs(ours - reference)[mask] / reference[mask]
    assert relative.max() < 5e-3
    assert np.median(relative) < 5e-4


def test_the_delta_lineshape_costs_a_few_percent_at_this_broadening(
    elk_dielectric, elkpy_absorption
):
    """The same sum with the textbook delta-function lineshape instead of
    dielectric.f90's own finite-swidth response is NOT the same function at
    a finite width: they differ by (2w - Delta)/Delta, first order in
    (w - Delta)/Delta and so a few percent across a linewidth. Pinning that
    here keeps the headline test's near-exact agreement attributable to the
    lineshape being matched, rather than to a tolerance loose enough to
    hide it -- and documents that the delta form is a s -> 0 statement.

    Re-uses the momentum data the fixture already collected: no extra Elk
    run."""
    w = elk_dielectric["energies"]
    reference = elk_dielectric["epsilon"][(1, 1)].imag + elk_dielectric["epsilon"][(2, 2)].imag
    delta_form = optical.circular_absorption(
        elkpy_absorption["kdata"], w, elkpy_absorption["occupations"],
        elkpy_absorption["volume"], occmax=elkpy_absorption["occmax"],
        swidth=SWIDTH, broadening="lorentzian",
    )["eps2_total"]
    mask = reference > 0.2 * reference.max()
    relative = np.abs(delta_form - reference)[mask] / reference[mask]
    assert relative.max() > 0.01     # visibly different from the exact response
    assert relative.max() < 0.30     # but the same spectrum, not a different one


def test_the_zone_integrated_circular_channels_are_equal(elkpy_absorption):
    """h-BN is non-magnetic, so time reversal (k -> -k, which exchanges
    sigma+ and sigma-) makes the two circular channels equal once summed
    over a mesh closed under negation -- however strong the local
    dichroism. This is the statement that the interesting physics is
    k-RESOLVED, and a check that the +/- bookkeeping is not accidentally
    symmetric or accidentally biased."""
    plus, minus = elkpy_absorption["eps2_plus"], elkpy_absorption["eps2_minus"]
    assert plus.max() > 1.0
    assert np.abs(plus - minus).max() < 1e-6 * plus.max()


def test_valley_restricted_absorption_is_circularly_selective(hbn):
    """Restricting the k-sum to a single valley recovers what the
    zone-integrated spectrum cancels: the K valley absorbs only one
    circular polarization and K' only the other (Yao, Xiao & Niu, PRB 77,
    235406 (2008); Xiao, Liu, Feng, Xu & Yao, PRL 108, 196802 (2012)).
    This is the beyond-stock-Elk output -- task 121 computes sigma_xx and
    sigma_xy, never sigma_+/sigma_-, and cannot resolve a single valley at
    all.

    The RELATIVE sign is the published prediction; the absolute sign is a
    convention pin (which sublattice carries B vs N), as in
    test_calculation_momentum.py's eta(K)."""
    valleys = hbn.get_circular_absorption(
        wplot=WPLOT, nwplot=NWPLOT, swidth=SWIDTH, kpoints=[K, KPRIME],
        label="valley_absorption",
    )
    w = valleys["omega"]
    etas = {}
    for name, weights in (("K", [1.0, 0.0]), ("K'", [0.0, 1.0])):
        one = optical.circular_absorption(
            valleys["kdata"], w, valleys["occupations"], valleys["volume"],
            occmax=valleys["occmax"], swidth=SWIDTH, weights=weights,
        )
        edge = int(np.nanargmax(one["eps2_total"]))
        etas[name] = one["eta"][edge]
    assert abs(etas["K"]) > 0.9
    assert etas["K"] == pytest.approx(-etas["K'"], rel=1e-3)
    assert etas["K"] < 0  # convention pin, matching test_calculation_momentum.py
