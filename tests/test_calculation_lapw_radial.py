"""Phase 1 of the JAX port: build the muffin-tin radial integrals in JAX.

`tests/test_calculation_lapw_assembly.py` takes the five radial integrals
(`haa`, `hloa`, `hlolo`, `oalo`, `ololo`) from the LAPW export and checks the
assembly built on top of them.  This checks the integrals themselves --
`src/elkjax/radial.py`, a transcription of Elk's `hmlrad.f90` and
`olprad.f90` -- against Elk's own, element by element, from the muffin-tin
Kohn-Sham potential and the radial functions that patch 0015 adds to the
export.

Two things here have teeth that a total-agreement check would not have.

The comparison is split by whether the integral involves the potential at
all.  The `lm2 = 0` element is not a potential integral -- it is
<u|H u>, with the Hamiltonian already applied by `genapwfr` -- so it shares
every input with Elk and must agree to roundoff.  The rest contracts against
`vsmt`, whose PACKING (lmmaxi harmonics per radial point inside `nrmti`,
lmmaxo outside) is the part a transcription gets wrong.  Splitting them
localises a failure to one of the two.

This split is also what identified the reason patch 0015 has to call
`genapwlofr` before exporting: without it every potential-free integral
agreed to 2.6e-16 while every potential integral was off by 3e-10, because
`gndstate` mixes the potential after building the radial functions and the
export therefore carried a `vsmt` one iteration ahead of its own `haa`.

Entries Elk never assigns are masked out.  `haa` and the rest are allocated
at `apwordmax`/`nlomax` -- maxima over all species -- so for an atom with
fewer APW orders or local orbitals than the maximum, the unwritten entries of
the exported array are uninitialised memory.

Same three fixtures as the assembly test, for the same reasons: Si at
apword 1 and 2 (the order axis is length 1 otherwise) and monolayer h-BN
(two species, and a nitrogen with two l=0 local orbitals, which is the only
one of the three that can see `hlolo`'s asymmetry).

Skipped without the elk binary, and without jax.
"""

import importlib.util

import numpy as np
import pytest

from elkpy import config

pytestmark = [
    pytest.mark.skipif(not config.default_elk_binary().is_file(),
                       reason="elk binary not built; see docs/design.md #8"),
    pytest.mark.skipif(importlib.util.find_spec("jax") is None,
                       reason="jax not installed; pip install -e .[jax]"),
]

from conftest import LAPW_CASES as CASES  # noqa: F401

# Every integral is one weighted sum over a few hundred radial points, so
# roundoff is the only scale in play.  Measured worst case across the three
# fixtures: 2.4e-16 relative.
TOL_REL = 1e-14


def _relative(got, ref, mask):
    got, ref = np.asarray(got)[mask], np.asarray(ref)[mask]
    scale = np.abs(ref).max()
    assert scale > 1e-3, "reference block is trivially small"
    return np.abs(got - ref).max() / scale


@pytest.mark.parametrize("case", CASES)
def test_radial_integrals_match_elk(case, exports):
    """`haa`, `hloa`, `hlolo`, `oalo`, `ololo`, split by potential
    involvement so a failure names one half of `hmlrad`."""
    from elkjax import radial
    export = exports[case]
    if "vsmt" not in export:
        pytest.skip("binary predates patch 0015 (no potential exported)")
    masks = radial.assigned_masks(export)
    haa, hloa, hlolo = radial.hamiltonian_integrals(export)
    oalo, ololo = radial.overlap_integrals(export)
    for key, got in (("oalo", oalo), ("ololo", ololo)):
        rel = _relative(got, export[key], masks[key])
        assert rel < TOL_REL, f"{key}: {rel:.3e}"
    for key, got in (("haa", haa), ("hloa", hloa), ("hlolo", hlolo)):
        mask = masks[key]
        for tag, sel in (("l2=0", 0), ("l2>0", slice(1, None))):
            part = np.zeros_like(mask)
            part[sel] = mask[sel]
            rel = _relative(got, export[key], part)
            assert rel < TOL_REL, f"{key} {tag}: {rel:.3e}"


@pytest.mark.parametrize("case", CASES)
def test_assembly_from_jax_built_integrals_reproduces_elk(case, exports):
    """The whole point of building them: feed the JAX-built integrals back
    into the assembly and recover Elk's own first-variational eigenvalues.

    This is the end-to-end statement -- the potential, not a set of imported
    radial integrals, is now the input to the spectrum.
    """
    import jax.numpy as jnp
    from elkjax import hamiltonian as ham, radial
    export = exports[case]
    if "vsmt" not in export:
        pytest.skip("binary predates patch 0015 (no potential exported)")
    rebuilt = radial.integrals_from_export(export)
    h, o = ham.assemble_from_export(rebuilt)
    reduced, _ = ham.cholesky_reduce(h, o)
    evals = np.asarray(jnp.linalg.eigvalsh(reduced))
    ref = np.asarray(export["evalfv"])
    assert np.abs(evals[:len(ref)] - ref).max() < 1e-8


@pytest.mark.parametrize("case", CASES)
def test_potential_unpacking_is_lossless(case, exports):
    """The dense `(nr, lmmaxo)` potential must carry every packed value and
    nothing else: the inner region's missing harmonics ZERO, which is what
    makes `hmlrad`'s `l2 <= lmaxi` guard automatic rather than a branch."""
    from elkjax import radial
    export = exports[case]
    if "vsmt" not in export:
        pytest.skip("binary predates patch 0015 (no potential exported)")
    idxis = np.asarray(export["idxis"]) - 1
    lmmaxi = int(export["lmmaxi"])
    for ias, dense in enumerate(radial.potential_arrays(export)):
        is_ = int(idxis[ias])
        nri = int(np.asarray(export["nrmti"])[is_])
        npmt = int(np.asarray(export["npmt"])[is_])
        assert np.all(dense[:nri, lmmaxi:] == 0.0)
        assert np.count_nonzero(dense) <= npmt
        # the packed stream, reconstructed from the dense array
        packed = np.concatenate([dense[:nri, :lmmaxi].ravel(),
                                 dense[nri:, :].ravel()])
        assert np.array_equal(packed, np.asarray(export["vsmt"])[ias, :npmt])
