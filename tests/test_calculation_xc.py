r"""Phase 2 of the JAX port: the XC functional against Elk's own v_xc.

`tests/test_jax_xc.py` pins `src/elkjax/xc.py` against closed forms -- Dirac
exchange, the exact exchange spin scaling, the Gell-Mann-Brueckner
high-density limit -- and against `jax.grad`.  All of those are statements
about the FORM of the functional.  None of them says it is the functional Elk
actually used, which is what this file checks: Elk's own v_xc along a line
through bulk silicon, against v_xc evaluated here on Elk's own density along
the same line.

There are two comparisons here and the difference between them is itself the
result.

**On the interstitial grid the agreement is exact** (4.4e-16 absolute).  Elk
evaluates the functional pointwise on the real-space FFT grid, so `vxcir` is
literally this transcription applied to `rhoir` -- once one further step is
reproduced: `potks.f90` passes `vxcir` through `trimrfg`, which zeroes every
Fourier component with |G| > 2 k_max.  WITHOUT that filter the same comparison
stops at 2.5e-5 relative, which looks exactly like a mediocre transcription and
is not one.  `elkjax.grid.trim` transcribes it.

**Through `plot1d` the agreement is 3e-5, and that is a commutation, not an
error.**  v_xc is nonlinear and neither of Elk's representations of a
real-space function commutes with it: inside a muffin tin Elk applies the
functional on an angular grid and keeps lmaxo harmonics of the RESULT, while a
comparison through the plot applies it to the truncated DENSITY; in the
interstitial Elk applies it pointwise and `plot1d` Fourier-interpolates the
result.  Measured 3.0e-5 median over 200 points, rising to 1.2e-2 in the shell
where the two representations meet at the muffin-tin boundary.  Kept as a test
because that number is worth knowing: it is the accuracy ceiling of ANY
comparison made through Elk's plotting tasks, and it is four orders of
magnitude worse than the same comparison made on the grid Elk computes on.

Skipped without the elk binary, and without jax.
"""

import importlib.util

import numpy as np
import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = [
    pytest.mark.skipif(not config.default_elk_binary().is_file(),
                       reason="elk binary not built; see docs/design.md #8"),
    pytest.mark.skipif(importlib.util.find_spec("jax") is None,
                       reason="jax not installed; pip install -e .[jax]"),
]

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
LINE = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]      # atom, interstitial, atom
NPOINTS = 200


@pytest.fixture(scope="module")
def line_profile(tmp_path_factory):
    """Elk's own density and v_xc along one line of a converged bulk Si.

    Two separate task runs on the same ground state; `plot1d` puts them on the
    identical point set, which the test asserts rather than assumes.
    """
    workdir = tmp_path_factory.mktemp("xc") / "si"
    calculation = Structure(
        avec=SI_AVEC,
        species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    ).get_calculation(workdir, xc="PW", ngridk=(2, 2, 2), rgkmax=7.0)
    calculation.ensure_ground_state()
    density = calculation.get_density_1d(line=LINE, npoints=NPOINTS)
    potential = calculation.get_potential_1d(
        component="xc", line=LINE, npoints=NPOINTS)
    return density, potential


def test_elks_own_potential_is_this_functional_of_elks_own_density(
        line_profile):
    density, potential = line_profile
    from elkjax import xc
    assert np.abs(density["distance"] - potential["distance"]).max() == 0.0
    rho = density["values"]
    assert (rho > 0).all()
    assert rho.max() > 1e2 and rho.min() < 1e-2, "no dynamic range on this line"
    _, _, vx, _, vc, _ = xc.pwca(0.5 * rho, 0.5 * rho)
    got = np.asarray(vx) + np.asarray(vc)
    reference = potential["values"]
    relative = np.abs(got - reference) / np.abs(reference)
    assert np.median(relative) < 1e-3, np.median(relative)
    assert relative.max() < 5e-2, relative.max()


def test_the_comparison_discriminates_the_correlation_functional(line_profile):
    """The premise of the tolerance above: correlation is a large fraction of
    v_xc here, so agreeing with Elk to 3e-5 is not something an
    exchange-only functional could do."""
    density, _ = line_profile
    from elkjax import xc
    rho = density["values"]
    _, _, vx, _, vc, _ = xc.pwca(0.5 * rho, 0.5 * rho)
    fraction = np.abs(np.asarray(vc)) / np.abs(np.asarray(vx) + np.asarray(vc))
    assert 0.05 < np.median(fraction) < 0.5, np.median(fraction)


@pytest.fixture(scope="module")
def groundstate(tmp_path_factory):
    """Elk's converged density and potentials on its own grids (patch 0016)."""
    workdir = tmp_path_factory.mktemp("xc_gs") / "si"
    calculation = Structure(
        avec=SI_AVEC,
        species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    ).get_calculation(workdir, xc="PW", ngridk=(2, 2, 2), rgkmax=7.0)
    calculation.ensure_ground_state()
    with calculation.eigenstate_session() as session:
        return session.ground_state()


def test_vxcir_is_exactly_this_functional_of_rhoir(groundstate):
    """The sharp check: pointwise on the grid Elk computes on.

    Both halves are required.  Without `trimrfg` the residual is 2.5e-5
    relative -- asserted here, so that "we reproduce Elk's v_xc" cannot
    silently come to mean "to four digits" if the filter is ever dropped.
    """
    import numpy as np
    from elkjax import grid, xc
    rho, reference = groundstate["rhoir"], groundstate["vxcir"]
    assert (rho > 0).all()
    _, _, vx, _, vc, _ = xc.pwca(0.5 * rho, 0.5 * rho)
    raw = np.asarray(vx) + np.asarray(vc)
    trimmed = np.asarray(grid.trim(raw, groundstate))
    assert np.abs(trimmed - reference).max() < 1e-14
    assert np.abs(raw - reference).max() > 1e-6


def test_the_muffin_tin_l0_channel_is_spherical_almost_everywhere(groundstate):
    """What a transcription of `potxcmt` would be up against, measured.

    Elk builds the muffin-tin v_xc on an angular grid; treating the density as
    if it were spherical reproduces its l = 0 channel to 8e-16 over most of the
    sphere and to 1.4e-2 at worst, where the non-spherical part of the density
    reaches 35% of the spherical part.  So the angular machinery is needed, but
    only in the outer shell -- which is where to look first if it ever
    disagrees.
    """
    import numpy as np
    from elkjax import xc
    from elkpy.parsers.eigenstates import unpack_muffin_tin
    y00 = 1.0 / np.sqrt(4.0 * np.pi)
    lmmaxi, lmmaxo = int(groundstate["lmmaxi"]), int(groundstate["lmmaxo"])
    for ias in range(int(groundstate["natmtot"])):
        isp = int(groundstate["idxis"][ias]) - 1
        nr = int(groundstate["nrmt"][isp])
        nri = int(groundstate["nrmti"][isp])
        rho = unpack_muffin_tin(groundstate["rhomt"][ias], nr, nri,
                                lmmaxi, lmmaxo)
        vxc = unpack_muffin_tin(groundstate["vxcmt"][ias], nr, nri,
                                lmmaxi, lmmaxo)
        spherical = rho[:, 0] * y00
        assert (spherical > 0).all()
        _, _, vx, _, vc, _ = xc.pwca(0.5 * spherical, 0.5 * spherical)
        got = (np.asarray(vx) + np.asarray(vc)) / y00
        relative = np.abs(got - vxc[:, 0]) / np.abs(vxc[:, 0])
        assert np.median(relative) < 1e-13
        assert relative.max() > 1e-4, "the non-spherical part is doing nothing"
        assert relative.max() < 1e-1


def test_the_ground_state_export_is_self_consistent(groundstate):
    """Cheap identities on the export itself, which catch a mis-ordered or
    truncated stream far more clearly than any physics check.

    `vsir = vclir + vxcir` is `potks.f90`'s own last line.  And `vsig` is
    shorter than `cfunig`: `genvsig` builds it on the COARSE grid, so it
    carries only |G| <= 2 k_max and `init0` allocates it `ngvc` long -- an
    export that wrote `ngvec` of them would read past the end of the array.
    """
    import numpy as np
    assert np.abs(groundstate["vsir"]
                  - (groundstate["vclir"] + groundstate["vxcir"])).max() == 0.0
    assert groundstate["vsig"].shape == (int(groundstate["ngvc"]),)
    assert groundstate["cfunig"].shape == (int(groundstate["ngvec"]),)
    assert int(groundstate["ngvc"]) < int(groundstate["ngvec"])
    assert np.isfinite(groundstate["vsig"]).all()
    # the characteristic function integrates to the interstitial volume
    fraction = float(groundstate["cfunig"][0].real)
    assert 0.1 < fraction < 0.95
    assert abs(groundstate["cfunir"].mean() - fraction) < 1e-10
