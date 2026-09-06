r"""Phase 2 of the JAX port: the muffin-tin angular transform, and why Elk's
own `vxcmt` is NOT the pointwise XC potential of its own density.

A nonlinear functional cannot be applied in the spherical-harmonic basis, so
Elk evaluates it on an angular grid: `rbsht` maps the `lmmax` coefficients at
each radial point to `lmmax` values on that grid, the functional is applied
pointwise, and `rfsht` maps back.  `elkjax.grid.to_angular`/`from_angular`
transcribe the pair, with patch 0016 exporting the four matrices.

That round trip is exact on a band-limited function, but v_xc[rho] is NOT
band-limited even when rho is: squeezing a nonlinear function through a finite
angular grid leaks weight into every harmonic, including the ones the site
symmetry forbids.  **`potxc.f90` lines 55-58 then remove it** --

    if (tsh) then
      call symrfmt(nrmt,nrmti,npmt,npmtmax,vxcmt_)
      if (spinpol) call symrvfmt(.true.,ncmag,...,bxcmt_)

-- symmetrising the POTENTIAL and the field, and NOT `exmt_`/`ecmt_`, which
are returned as computed.  So in Elk the two are built from the same density
by the same code and then treated differently, and

    v_xc(muffin tin) = S v_xc[rho],   e_xc(muffin tin) = e_xc[rho]

with S the average over the `nsymcrys` crystal operations (`symrfmt`).  This
is measured here rather than argued: the pointwise transcription reproduces
`exmt`/`ecmt` to 1.8e-15 but misses `vxcmt` by 5.3e-3 on a scale of 45; the
same transcription against a `symtype=0` ground state, where `symrfmt` is the
identity, reproduces `vxcmt` to 1.4e-14 as well.

Two consequences for the port, both real:

  * S is not roundoff.  1.2e-4 relative, growing from below 1e-9 near the
    nucleus (where the density is spherical and there is nothing to project)
    to 5.3e-3 at R_MT.  Any transcription of Elk's SCF must apply it, which
    means exporting `symlatc`/`lsplsymc`/`ieqatom`, or must run `symtype=0`.
  * inside a symmetric muffin tin Elk's own v_xc is not the functional
    derivative of its own E_xc.  The discrepancy is variational noise of the
    SHT truncation, but it is there, and a total-energy or force check at
    better than ~1e-4 relative will see it.

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


def _ground_state(workdir, extra_blocks=None):
    calculation = Structure(
        avec=SI_AVEC,
        species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    ).get_calculation(workdir, xc="PW", ngridk=(2, 2, 2), rgkmax=7.0,
                      extra_blocks=extra_blocks)
    calculation.ensure_ground_state()
    with calculation.eigenstate_session() as session:
        return session.ground_state()


@pytest.fixture(scope="module")
def groundstate(tmp_path_factory):
    """Bulk Si with its full 48-operation symmetry."""
    return _ground_state(tmp_path_factory.mktemp("mtxc") / "si")


@pytest.fixture(scope="module")
def unsymmetrised(tmp_path_factory):
    """The same cell with `symtype = 0`, which is what the `nosym` input block
    sets (`readinput.f90:1311`) and which makes `symrfmt` the identity."""
    return _ground_state(tmp_path_factory.mktemp("mtxc") / "si_nosym",
                         extra_blocks={"symtype": [0]})


def _potential(groundstate, ias):
    """The pointwise XC potential of Elk's own density, in packed harmonics."""
    from elkjax import grid, xc
    isp = int(groundstate["idxis"][ias]) - 1
    npmt = int(groundstate["npmt"][isp])
    rho = np.asarray(grid.to_angular(groundstate["rhomt"][ias],
                                     groundstate, ias))[:npmt]
    _, _, vx, _, vc, _ = xc.pwca(0.5 * rho, 0.5 * rho)
    values = np.asarray(vx) + np.asarray(vc)
    return np.asarray(grid.from_angular(values, groundstate, ias))[:npmt]


def test_the_angular_transforms_are_mutual_inverses(groundstate):
    """`rfsht(rbsht(f)) = f`, on Elk's own density rather than a random
    vector: the packing (lmmaxi per point inside `nrmti`, lmmaxo outside) is
    what a transcription gets wrong, and a random vector would exercise it
    just as well only if the packing were right to begin with."""
    from elkjax import grid
    for ias in range(int(groundstate["natmtot"])):
        packed = groundstate["rhomt"][ias]
        back = np.asarray(grid.from_angular(
            grid.to_angular(packed, groundstate, ias), groundstate, ias))
        assert np.abs(back - packed).max() / np.abs(packed).max() < 1e-14


def test_the_energy_densities_are_exact_in_the_muffin_tin(groundstate):
    """`exmt` and `ecmt` from the angular-grid density, element-wise.  These
    are the two arrays `potxc` does NOT symmetrise, and they are exact."""
    from elkjax import grid, xc
    for ias in range(int(groundstate["natmtot"])):
        rho = np.asarray(grid.to_angular(groundstate["rhomt"][ias],
                                         groundstate, ias))
        ex, ec = xc.pwca(0.5 * rho, 0.5 * rho)[:2]
        for values, key in ((ex, "exmt"), (ec, "ecmt")):
            got = np.asarray(grid.from_angular(np.asarray(values),
                                               groundstate, ias))
            reference = groundstate[key][ias]
            assert np.abs(reference).max() > 1e-2
            assert np.abs(got - reference).max() < 1e-13


def test_the_potential_is_exact_only_without_symmetrisation(
        groundstate, unsymmetrised):
    """The measurement that identifies `symrfmt`, and the reason it needs two
    ground states: the pointwise potential is a 5.3e-3 miss on the symmetric
    cell and machine precision on the `symtype=0` one, from the same code.

    A single-fixture version of this test could only pin the discrepancy, not
    attribute it -- which is what this file did before the cause was found.
    """
    symmetric = max(
        np.abs(_potential(groundstate, ias)
               - groundstate["vxcmt"][ias][:len(_potential(groundstate, ias))]
               ).max()
        for ias in range(int(groundstate["natmtot"])))
    scale = np.abs(groundstate["vxcmt"][0]).max()

    assert symmetric > 1e-4, (
        "the symmetric cell no longer shows the projection; if `potxc` has "
        "stopped calling `symrfmt`, this whole file is about nothing")
    assert symmetric / scale < 1e-3

    for ias in range(int(unsymmetrised["natmtot"])):
        mine = _potential(unsymmetrised, ias)
        elk = unsymmetrised["vxcmt"][ias][:len(mine)]
        assert np.abs(mine - elk).max() < 1e-12, (
            "with symmetrisation switched off the pointwise potential must be "
            "exact -- if it is not, the cause is not `symrfmt` after all")


def test_the_projection_removes_the_symmetry_forbidden_harmonics(groundstate):
    """What `symrfmt` actually does to this function, resolved by l.

    Diamond Si's site symmetry forbids l = 1, 2 and 5 in the muffin tin, and
    Elk's `vxcmt` carries 1e-20 there -- exactly zero.  The pointwise
    potential carries 1e-3, because the SHT round trip of a nonlinear
    function is not band-limited.  That is the leak, seen directly.

    The inner region (l <= lmaxi = 1) is exact to 6e-14 in the same run: near
    the nucleus the density is spherical, so there is nothing to project.
    """
    ias = 0
    isp = int(groundstate["idxis"][ias]) - 1
    nr, nri = int(groundstate["nrmt"][isp]), int(groundstate["nrmti"][isp])
    lmmaxi, lmmaxo = int(groundstate["lmmaxi"]), int(groundstate["lmmaxo"])
    npmt = int(groundstate["npmt"][isp])

    mine = _potential(groundstate, ias)
    elk = groundstate["vxcmt"][ias][:npmt]

    inner_size = lmmaxi * nri
    inner = (mine - elk)[:inner_size].reshape(nri, lmmaxi)
    assert np.abs(inner).max() < 1e-12

    outer_mine = mine[inner_size:].reshape(nr - nri, lmmaxo)
    outer_elk = elk[inner_size:].reshape(nr - nri, lmmaxo)
    for l in (1, 2, 5):
        block = slice(l * l, (l + 1) ** 2)
        assert np.abs(outer_elk[:, block]).max() < 1e-15, (
            f"l={l} is symmetry-forbidden here and Elk's vxcmt must be zero")
        assert np.abs(outer_mine[:, block]).max() > 1e-4, (
            f"l={l} carries the leak; if it has vanished the SHT round trip "
            "has become band-limited, which it cannot be for a nonlinear "
            "functional")
