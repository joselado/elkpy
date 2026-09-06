r"""Phase 2 of the JAX port: Poisson's equation on the exported ground state.

Patch 0016 exports the density and the Coulomb potential on the same
real-space FFT grid.  In the INTERSTITIAL region -- where Elk's characteristic
function is 1 -- they must satisfy

    laplacian V_cl(r) = -4 pi rho(r)

exactly, and nothing else in this repository checks that.  It is not a
transcription check: Elk's Weinert solver is not transcribed here.  What it
does check is the whole G-space bookkeeping the rest of Phase 2 will be built
on -- that `igfft` puts a list entry in the right FFT slot, that `ivg` times
`bvec` gives the Cartesian G at that slot, and that the export's arrays are
mutually consistent.  A transposed grid or an off-by-one in `igfft` produces
a plausible-looking array and fails here immediately.

**Why it is not machine precision.**  `vclir` is a band-limited representation
of a function that is not band-limited near a muffin-tin sphere, where Weinert's
pseudocharge takes over, so the spectral Laplacian carries the grid's own
truncation error.  Measured on bulk Si: 1.2e-5 median against a right-hand side
of order 1, over the ~8000 grid points with `cfunir > 0.999`.  The identity is
asserted at 1e-3, which is three decades below what any indexing error would
give and two above the truncation floor.

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


@pytest.fixture(scope="module")
def groundstate(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("poisson") / "si"
    calculation = Structure(
        avec=SI_AVEC,
        species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    ).get_calculation(workdir, xc="PW", ngridk=(2, 2, 2), rgkmax=7.0)
    calculation.ensure_ground_state()
    with calculation.eigenstate_session() as session:
        return session.ground_state()


def test_the_g_vectors_agree_with_elks_own(groundstate):
    """`bvec @ ivg` must reproduce `vgc` on the entries Elk wrote it for.

    The forward check that makes the identity below a physics statement
    rather than a coincidence of two wrong things.
    """
    from elkjax import grid
    ngvec = int(groundstate["ngvec"])
    gfft = grid.reciprocal_vectors(groundstate)
    igfft = np.asarray(groundstate["igfft"]) - 1
    reference = np.asarray(groundstate["vgc"])[:, :ngvec]
    assert np.abs(gfft[:, igfft[:ngvec]] - reference).max() < 1e-12
    # and the list really is sorted by |G|, which `trimrfg` relies on
    magnitude = np.asarray(groundstate["gc"])
    assert (np.diff(magnitude) > -1e-10).all()


def test_poissons_equation_holds_in_the_interstitial(groundstate):
    """The identity, on the points where the characteristic function is 1."""
    from elkjax import grid
    laplacian = np.asarray(grid.laplacian(groundstate["vclir"], groundstate))
    rhs = -4.0 * np.pi * groundstate["rhoir"]
    inside = groundstate["cfunir"] > 0.999
    assert inside.sum() > 1000, "no interstitial points to test on"
    residual = np.abs(laplacian[inside] - rhs[inside])
    scale = np.abs(rhs[inside]).max()
    assert scale > 1e-2
    assert np.median(residual) / scale < 1e-3, np.median(residual) / scale
    assert residual.max() / scale < 1e-2, residual.max() / scale


def test_the_identity_is_not_free(groundstate):
    """Inside a muffin-tin sphere it must FAIL, and by a lot.

    Weinert's method replaces the true muffin-tin charge with a smooth
    pseudocharge carrying the same multipoles, so `vclir` there solves
    Poisson's equation for a different density entirely.  Without this, a
    check that accidentally compared two smooth things would look just as
    green.
    """
    from elkjax import grid
    laplacian = np.asarray(grid.laplacian(groundstate["vclir"], groundstate))
    rhs = -4.0 * np.pi * groundstate["rhoir"]
    deep = groundstate["cfunir"] < 0.01
    assert deep.sum() > 100
    scale = np.abs(rhs[groundstate["cfunir"] > 0.999]).max()
    assert np.abs(laplacian[deep] - rhs[deep]).max() / scale > 1.0
