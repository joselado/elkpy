"""Phase 1 of the JAX port: assemble H and O in JAX, compare against Elk's.

`src/elkjax/hamiltonian.py` transcribes olpfv/hmlfv -- the muffin-tin half of
the first-variational LAPW eigenproblem -- and this checks it against the
matrices Elk itself built at the same k-point, block by block, through the
LAPW export (patches/0013 + 0014, docs/design.md #33).

Six blocks are compared SEPARATELY rather than only in total, so a failure
names one upstream routine (olpaa, olpalo, olplolo, hmlaa, hmlalo, hmllolo)
instead of "H is wrong somewhere". The interstitial contributions are taken
from the export and not rebuilt: the Hamiltonian's needs the interstitial
Kohn-Sham potential, which is Phase 2 of the port.

Three fixtures, and each one catches something the others cannot:

  * bulk Si at apword=1  -- the stock case, every species file Elk ships.
  * bulk Si at apword=2  -- the only way to exercise haa's and hloa's APW
    ORDER axes, which are length 1 otherwise, so an io/l index swap is
    invisible at apword=1. Same generated species file the export test uses.
  * monolayer h-BN       -- two species (idxis indexing into rmt/apword/nlorb)
    and, decisively, a nitrogen carrying TWO l=0 local orbitals. hlolo's
    l2=0 element is <u_i|H u_j>, built with no symmetrisation, so the two
    orderings differ (measured: 1.3e-2 Ha) and only the half Elk evaluates
    may be used. Si cannot see this at all -- its local orbitals are one s
    and one p, so no pair shares an l.

Skipped without the elk binary, and without jax.
"""

import numpy as np
import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = [
    pytest.mark.skipif(not config.default_elk_binary().is_file(),
                       reason="elk binary not built; see docs/design.md #8"),
    pytest.mark.skipif(
        pytest.importorskip("importlib").util.find_spec("jax") is None,
        reason="jax not installed; pip install -e .[jax]"),
]

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
KPOINT = (0.1, 0.2, 0.05)          # generic: no symmetry, no degeneracy

# Machine precision is the right scale for every block: each is a small dense
# contraction of exported doubles. The one exception is Elk's own side --
# hmlaa/hmlalo skip any Gaunt-contracted z1 with |Re| + |Im| <= 1e-12
# (hmlaa.f90's zaxpy guard), which the dense transcription here keeps. Measured
# on h-BN that costs 3.3e-14, and reproducing the guard recovers 3.0e-17, so
# the tolerance is set by the guard and not by the transcription.
TOL = 1e-12


def _apword2_species_dir(tmp_path):
    """Elk's own Si.in with the APW order raised from 1 to 2 -- derivative
    orders 0 and 1, the textbook LAPW basis u_l and du_l/dE. Every species
    file Elk ships sets apword = 1, so without this the order axis of haa and
    hloa is length 1 and never tested."""
    source = config.resolve_species_path() / "Si.in"
    out = []
    for line in source.read_text().splitlines(keepends=True):
        if ": apword" in line:
            out.append("   2                                        : apword\n")
            out.append("    0.1500   0  F                           : apwe0, apwdm, apwve\n")
            out.append("    0.1500   1  F\n")
            continue
        if ": apwe0, apwdm, apwve" in line and out and "apword" in out[-3]:
            continue
        out.append(line)
    directory = tmp_path / "species_apword2"
    directory.mkdir()
    (directory / "Si.in").write_text("".join(out))
    return directory


@pytest.fixture(scope="module")
def _module_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("lapw_assembly")


def _export(structure, workdir, sppath=None, **kwargs):
    from elkpy.calculation import Calculation
    calc = Calculation(structure=structure, workdir=workdir, **kwargs)
    if sppath is not None:
        calc.sppath = sppath
    calc.ensure_ground_state()
    with calc.eigenstate_session() as session:
        return session.lapw_problem(KPOINT)


@pytest.fixture(scope="module")
def exports(_module_tmp):
    """All three cases, each ground state converged once.

    A single dict rather than three parametrized fixtures because the tests
    below select among them by name: pytest's `getfixturevalue` cannot reach
    into a parametrized fixture.
    """
    out = {}
    silicon = Structure(
        avec=SI_AVEC,
        species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]})
    out["si_apword1"] = _export(silicon, _module_tmp / "si1",
                                ngridk=(2, 2, 2), rgkmax=7.0)
    out["si_apword2"] = _export(
        silicon, _module_tmp / "si2",
        sppath=_apword2_species_dir(_module_tmp), ngridk=(2, 2, 2),
        rgkmax=7.0)
    a, c = 4.746, 20.0
    hbn = Structure(
        avec=[(a, 0.0, 0.0), (-a / 2, a * 3 ** 0.5 / 2, 0.0), (0.0, 0.0, c)],
        species={"B": [(0.0, 0.0, 0.0)], "N": [(1 / 3, 2 / 3, 0.0)]})
    out["hbn"] = _export(hbn, _module_tmp / "hbn",
                         ngridk=(2, 2, 1), rgkmax=6.0)
    return out


CASES = ["si_apword1", "si_apword2", "hbn"]


def _reference_blocks(export):
    """Elk's own muffin-tin blocks: its H and O with the exported interstitial
    contribution (which lives in the APW-APW corner alone -- a local orbital
    vanishes outside its sphere) subtracted off."""
    ngp = int(export["ngp"])
    h = np.array(export["hmat"])
    o = np.array(export["omat"])
    h[:ngp, :ngp] -= export["hmat_istl"]
    o[:ngp, :ngp] -= export["omat_istl"]
    return h, o


def _compare_blocks(export):
    from elkjax import hamiltonian as ham
    ngp = int(export["ngp"])
    h_ref, o_ref = _reference_blocks(export)
    o_jax = np.asarray(ham.muffin_tin_overlap(export))
    h_jax = np.asarray(ham.muffin_tin_hamiltonian(export))
    corners = {"apw_apw": (slice(None, ngp), slice(None, ngp)),
               "apw_lo": (slice(None, ngp), slice(ngp, None)),
               "lo_lo": (slice(ngp, None), slice(ngp, None))}
    out = {}
    for name, sel in corners.items():
        for tag, jax_mat, ref in (("olp", o_jax, o_ref), ("hml", h_jax, h_ref)):
            out[f"{tag}_{name}"] = (np.abs(jax_mat[sel] - ref[sel]).max(),
                                    np.abs(ref[sel]).max())
    return out


@pytest.mark.parametrize("case", CASES)
def test_muffin_tin_blocks_match_elk(case, exports):
    """Each of the six muffin-tin blocks, separately.

    Every block is also required to be non-trivial: a transcription that
    returns zeros would otherwise pass the ones Elk also leaves near zero.
    """
    export = exports[case]
    for name, (diff, scale) in _compare_blocks(export).items():
        assert scale > 1e-3, f"{name} reference block is trivially small"
        assert diff < TOL, f"{name}: max|diff| = {diff:.3e}"


@pytest.mark.parametrize("case", CASES)
def test_assembled_matrices_reproduce_elks_eigenvalues(case, exports):
    """H and O assembled here, diagonalised, against Elk's own evalfv.

    This is the whole-object check the block-by-block one cannot be: evalfv
    comes from eveqnfv through Elk's own configured path, so agreement pins
    the assembly, the local-orbital column numbering and the interstitial
    padding together.
    """
    import scipy.linalg as sla
    from elkjax import hamiltonian as ham

    export = exports[case]
    h, o = (np.asarray(m) for m in ham.assemble_from_export(export))
    evalfv = np.asarray(export["evalfv"])
    computed = sla.eigh(h, o, eigvals_only=True)[:len(evalfv)]
    assert np.abs(computed - evalfv).max() < 1e-10


def test_hbn_nitrogen_has_two_l0_local_orbitals(exports):
    """The premise of the h-BN fixture, asserted rather than assumed.

    hmllolo evaluates a local-orbital pair in one order only, and hlolo's
    l2 = 0 element <u_i|H u_j> is not symmetric under exchanging them. That
    can only be caught by a species with two local orbitals sharing an l --
    which is why this fixture exists, and which stops being true silently if
    Elk's shipped N.in changes.
    """
    hbn = exports["hbn"]
    lorbl = [np.asarray(x).tolist() for x in hbn["lorbl"]]
    repeated = [l for species in lorbl for l in set(species)
                if species.count(l) > 1]
    assert repeated, f"no species has a repeated l: {lorbl}"

    hlolo = np.asarray(hbn["hlolo"])
    worst = 0.0
    for ias, is_ in enumerate(np.asarray(hbn["idxis"]) - 1):
        n = len(lorbl[is_])
        block = hlolo[0, :n, :n, ias]
        worst = max(worst, np.abs(block - block.T).max())
    assert worst > 1e-4, (
        f"hlolo's l2=0 block is symmetric to {worst:.1e}; the ordering this "
        "test exists to pin would then be unobservable")
