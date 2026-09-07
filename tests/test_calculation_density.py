r"""Phase 2 item 2b: the interstitial valence density from the eigenvectors.

Sections 2a-2g go from a density to a total energy.  This goes the other way --
from `evecfv` back to a density -- which is the step that closes the SCF loop's
circle.  What is here is the INTERSTITIAL half; the muffin-tin half needs
`wfmtsv` on the coarse radial mesh plus `rhomagsh`, `rfmtctof` and `rhocore`,
and so the core states, and is not done.

Three results, and the second and third are the ones with teeth:

  * on an UNREDUCED mesh the density is reproduced to 9e-16 relative -- exact,
    since in the interstitial an LAPW state is a plain plane-wave sum.
  * on a SYMMETRY-REDUCED mesh it is 16% off, because `rhomagv` calls `symrf`
    afterwards and this does not.  That is the same shape as section 2g's
    muffin-tin `symrfmt`, in the interstitial, and it is asserted rather than
    left as a caveat: 16% is a missing step, not a tolerance.
  * the residual difference on the unreduced mesh is `rhonorm`'s UNIFORM shift.
    Switching `trhonorm` off takes it from 2.82e-05 to 2.8e-18, which
    identifies it by measurement rather than by reading the source.

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


def _run(workdir, extra_blocks, lapw=False):
    calculation = Structure(
        avec=SI_AVEC,
        species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    ).get_calculation(workdir, xc="PW", ngridk=(2, 2, 2), rgkmax=7.0,
                      extra_blocks=extra_blocks)
    calculation.ensure_ground_state()
    with calculation.eigenstate_session() as session:
        # DENSITYK deliberately FIRST, so the test would fail if the query
        # depended on LAPW having regenerated the radial functions -- which it
        # did until `genapwlofr` was added to it (see the docstring below)
        out = (session.ground_state(), session.density_k())
        return out + ((session.lapw_problem((0.0, 0.0, 0.0)),) if lapw else ())


@pytest.fixture(scope="module")
def unreduced(tmp_path_factory):
    """`symtype=0`, so `symrf` is the identity and the 2x2x2 mesh is unreduced."""
    return _run(tmp_path_factory.mktemp("density") / "si_nosym",
                {"symtype": [0], "trhonorm": [False]}, lapw=True)[:2]


@pytest.fixture(scope="module")
def unreduced_with_lapw(tmp_path_factory):
    """The same, keeping the `LAPW` query the muffin-tin half needs for the
    radial functions, the Gaunt-free matching machinery and `idxlo`."""
    return _run(tmp_path_factory.mktemp("density") / "si_mt",
                {"symtype": [0], "trhonorm": [False]}, lapw=True)


@pytest.fixture(scope="module")
def normalised(tmp_path_factory):
    """The same, with `rhonorm` left on -- the only difference from `unreduced`."""
    return _run(tmp_path_factory.mktemp("density") / "si_norm",
                {"symtype": [0]})


@pytest.fixture(scope="module")
def reduced(tmp_path_factory):
    """Elk's default symmetry reduction: 3 k-points instead of 8."""
    return _run(tmp_path_factory.mktemp("density") / "si_red",
                {"trhonorm": [False]})


def _compare(fixture):
    from elkjax import density
    groundstate, densityk = fixture
    mine = np.asarray(density.interstitial_density(
        densityk, float(groundstate["omega"])))
    elk = np.asarray(density.coarsen(groundstate["rhoir"], groundstate,
                                     densityk))
    return mine, elk


def test_the_zone_sum_is_a_real_sum(unreduced):
    """More than one k-point, with weights summing to one.

    Without this the file could pass on a Gamma-only cell where the weight is
    1 and the sum has one term -- and the k-weight handling, which is what
    differs between a reduced and an unreduced mesh, would never run.
    """
    _, densityk = unreduced
    assert int(densityk["nkpt"]) == 8
    assert abs(float(np.asarray(densityk["wkpt"]).sum()) - 1.0) < 1e-12


def test_the_fine_density_carries_no_content_beyond_the_coarse_cutoff(
        unreduced):
    """The property that makes `coarsen` exact rather than an interpolation.

    `rfirctof` copies `ngvc` Fourier components into the fine array and zeroes
    the rest, so taking those same components back is lossless.  Asserted on
    Elk's own `rhoir` -- if it ever carried content beyond `ngvc`, every
    comparison in this file would silently acquire an aliasing error.
    """
    groundstate, densityk = unreduced
    grid = tuple(int(n) for n in groundstate["ngridg"])
    spectrum = np.fft.fftn(
        np.asarray(groundstate["rhoir"]).reshape(grid, order="F")
    ).reshape(-1, order="F") / np.prod(grid)
    ngvc = int(densityk["ngvc"])
    kept = np.asarray(groundstate["igfft"])[:ngvc] - 1
    dropped = np.setdiff1d(np.arange(spectrum.size), kept)
    assert np.abs(spectrum[dropped]).max() / np.abs(spectrum).max() < 1e-14


def test_the_interstitial_density_is_exact_on_an_unreduced_mesh(unreduced):
    """The headline: 9e-16 relative, with `rhonorm` off.

    In the interstitial an LAPW state is an exact plane-wave sum, so this is
    not an approximation -- the only truncation is the basis's own, and it is
    the same truncation on both sides.
    """
    mine, elk = _compare(unreduced)
    assert np.abs(elk).max() > 1e-3
    assert np.abs(elk - mine).max() / np.abs(elk).max() < 1e-13


def test_rhonorm_is_a_uniform_shift_and_switching_it_off_removes_it(
        unreduced, normalised):
    """`rhonorm` ADDS a constant rather than rescaling, identified by
    measurement.

    With it on, the difference from Elk is a constant to 1.7e-17 -- which is a
    sharper statement than any tolerance, a constant being a one-parameter
    family that a pointwise error would break.  With it off the constant is
    2.8e-18, i.e. gone.  The pair identifies the mechanism; either alone would
    only bound it.
    """
    from elkjax import density
    mean_off, spread_off = density.normalisation_offset(*_compare(unreduced))
    mean_on, spread_on = density.normalisation_offset(*_compare(normalised))
    _, elk = _compare(normalised)
    scale = np.abs(elk).max()

    assert spread_on / scale < 1e-14
    assert spread_off / scale < 1e-14
    assert mean_on > 1e4 * abs(mean_off)
    assert abs(mean_off) / scale < 1e-14


def test_a_reduced_mesh_needs_the_symmetrisation_this_does_not_apply(reduced):
    """The scope of the claim, asserted rather than written down.

    `rhomagv` calls `symrf` on the accumulated density, and on a reduced mesh
    that is not the identity -- the same shape as section 2g's muffin-tin
    `symrfmt`, in the interstitial.  Measured 16%, which is a missing step and
    not a tolerance.  If a future change applies `symrfir` here this test
    fails, which is the signal to rewrite it rather than to widen it.
    """
    _, densityk = reduced
    assert int(densityk["nkpt"]) < 8, "this fixture is not actually reduced"
    mine, elk = _compare(reduced)
    assert np.abs(elk - mine).max() / np.abs(elk).max() > 1e-2


def test_the_grid_indirection_is_load_bearing(unreduced):
    """A mutation test on `igkig` -> `igfc`, the two-step map.

    `igkig` takes a basis function to a global G index and `igfc` takes that to
    a coarse FFT slot.  Using the identity for the second step leaves a
    perfectly smooth, positive, correctly normalised density on the same grid
    -- it is a permutation of the Fourier content, so nothing structural
    notices.
    """
    from elkjax import density
    groundstate, densityk = unreduced
    good, elk = _compare(unreduced)

    grid = tuple(int(n) for n in densityk["ngdgc"])
    total = np.zeros(int(densityk["ngtc"]))
    for ik in range(int(densityk["nkpt"])):
        weight = float(densityk["wkpt"][ik])
        vectors = densityk["evecfv"][(ik, 0)]
        igkig = np.asarray(densityk["igkig"][(ik, 0)])
        for ist, occupation in enumerate(np.asarray(densityk["occsv"][ik])):
            if abs(occupation) < 1e-8:
                continue
            array = np.zeros(int(densityk["ngtc"]), dtype=complex)
            array[igkig - 1] = vectors[ist][:igkig.size]   # the mutation: no igfc
            psi = np.fft.ifftn(array.reshape(grid, order="F")
                               ).reshape(-1, order="F") * np.prod(grid)
            total += (occupation * weight
                      / float(groundstate["omega"])) * np.abs(psi) ** 2
    assert total.min() >= 0.0, "the mutant is still a density, as claimed"
    assert np.abs(total - elk).max() / np.abs(elk).max() > 1e-2
    assert np.abs(good - elk).max() / np.abs(elk).max() < 1e-13


def test_the_muffin_tin_density_matches_rhomagk(unreduced_with_lapw):
    """`wfmtsv` + `rmk3`, against Elk's own accumulation.

    The reference is built by patch 0019 calling `rhomagk` over the k-set in a
    LOCAL array -- so it is the density BEFORE `rhomagsh` (spherical
    coordinates to harmonics), `symrf`, `rfmtctof` (coarse to fine radial mesh)
    and `rhocore`.  That is deliberate: each of those is a step this does not
    transcribe, and comparing against the converged `rhomt` would fold all four
    into one number.
    """
    from elkjax import density
    groundstate, densityk, lapw = unreduced_with_lapw
    values = density.muffin_tin_density(densityk, groundstate, lapw)
    for ias in range(int(groundstate["natmtot"])):
        packed = np.asarray(density.pack_coarse(values[ias], densityk,
                                                groundstate, ias))
        reference = np.asarray(densityk["rhomt_coarse"][ias])[:packed.size]
        assert np.abs(reference).max() > 1.0
        assert np.abs(packed - reference).max() \
            / np.abs(reference).max() < 1e-13


def test_the_interstitial_matches_the_direct_reference(unreduced):
    """The same comparison for the interstitial, against patch 0019's own
    pre-`symrf`, pre-`rhonorm` array rather than through `coarsen`.

    This and `test_the_interstitial_density_is_exact_on_an_unreduced_mesh`
    check the same arithmetic against two different references -- one built by
    Elk before any post-processing, one recovered from the stored density by
    inverting `rfirctof`.  Agreement of both is what says `coarsen` is right.
    """
    from elkjax import density
    groundstate, densityk = unreduced
    mine = np.asarray(density.interstitial_density(
        densityk, float(groundstate["omega"])))
    reference = np.asarray(densityk["rhoir_coarse"])
    assert np.abs(mine - reference).max() / np.abs(reference).max() < 1e-13


def test_the_local_orbital_coefficients_are_load_bearing(unreduced_with_lapw):
    """A mutation test on the one bug this feature actually hit.

    `evecfv` has `nmat = ngk + nlotot` coefficients; the first `ngk` are plane
    waves and the rest are local orbitals.  An export truncated at `ngk` -- the
    first version of patch 0019 -- leaves the interstitial density EXACT, since
    local orbitals vanish there, and the muffin-tin density smooth, positive,
    correctly scaled and 100% wrong.  Nothing but a reference catches it.
    """
    from elkjax import density
    groundstate, densityk, lapw = unreduced_with_lapw
    assert int(densityk["nmat"][0, 0]) > int(densityk["ngk"][0, 0])

    truncated = dict(densityk)
    truncated["evecfv"] = {
        key: np.pad(value[:, :int(densityk["ngk"][key[0], key[1]])],
                    ((0, 0), (0, value.shape[1]
                              - int(densityk["ngk"][key[0], key[1]]))))
        for key, value in densityk["evecfv"].items()}
    values = density.muffin_tin_density(truncated, groundstate, lapw)
    packed = np.asarray(density.pack_coarse(values[0], truncated,
                                            groundstate, 0))
    reference = np.asarray(densityk["rhomt_coarse"][0])[:packed.size]
    assert packed.min() >= 0.0, "the mutant is still a density"
    assert np.abs(packed - reference).max() / np.abs(reference).max() > 0.1


def test_the_two_radial_regions_restart_the_stride(unreduced_with_lapw):
    """`wfmtsv`'s outer region does not continue the inner one's stride.

    `zfzrf` is handed `apwfr(iro, ...)` with `iro = nrmti + lradstp`, one full
    step PAST the inner boundary rather than continuing from it.  Off by one
    step, the outer half of the density is still smooth and still the right
    order; only the reference sees it.
    """
    from elkjax import density
    groundstate, densityk, _ = unreduced_with_lapw
    inner, outer = density.coarse_radial_indices(densityk, groundstate, 0)
    step = int(densityk["lradstp"])
    isp = int(groundstate["idxis"][0]) - 1
    assert inner[0] == 0
    assert inner[-1] == int(groundstate["nrmti"][isp]) - 1
    assert outer[0] == inner[-1] + step
    assert outer[-1] == int(groundstate["nrmt"][isp]) - 1
    assert inner.size + outer.size == int(densityk["nrcmt"][isp])
