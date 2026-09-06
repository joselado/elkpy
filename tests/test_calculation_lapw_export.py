"""Integration test for the LAPW export (task 9002's LAPW query, added by
patches/0013-lapw-export.patch) -- every ingredient of the first-variational
LAPW eigenvalue problem at one k-point.

This exists for the JAX port, not for physics. Two things it settles that
docs/jax_port_phase0.md records as open:

  * item 0c's FORWARD half. `src/elkjax/lapw.py`'s transcription of `match`
    had only ever been checked against its own defining equation, because
    nothing in vendor/elk/src/ writes `apwalm` at all. It is now compared
    against Elk's own array, element by element, in BOTH of match.f90's
    branches -- the `omax == 1` division (every species file Elk ships) and
    the general linear solve, reached here through a species file this test
    generates with `apword = 2`.

  * kappa(O) for a REAL LAPW overlap, which sets the tolerance the study's
    section 8(b) derives for the projector rule (eps kappa(S) ||H||) and which
    had been measured only on synthetic matrices with a prescribed condition
    number.

Skipped if the elk binary hasn't been built, same as test_calculation_si.py.
"""

import itertools

import numpy as np
import pytest

from elkpy import config
from elkpy.structure import Structure

pytestmark = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}
KPOINT = (0.1, 0.2, 0.05)          # generic: no symmetry, no degeneracy


def _apword2_species_dir(tmp_path):
    """A copy of Elk's own Si.in with the APW order raised from 1 to 2.

    Every species file Elk ships sets `apword = 1` -- plain APW plus local
    orbitals -- so match.f90's `omax == 1` fast path is the ONLY one a stock
    calculation ever reaches, and its general branch (build D from polynm's
    divided-difference fit, then zgesv) would otherwise go untested on both
    sides of the comparison. Order 2 with derivative orders 0 and 1 is the
    textbook LAPW basis, u_l and du_l/dE.
    """
    source = config.resolve_species_path() / "Si.in"
    lines = source.read_text().splitlines(keepends=True)
    out = []
    for line in lines:
        if ": apword" in line:
            out.append("   2                                        : apword\n")
            out.append("    0.1500   0  F                           : apwe0, apwdm, apwve\n")
            out.append("    0.1500   1  F\n")
            continue
        if ": apwe0, apwdm, apwve" in line and out and "apword" in out[-3]:
            continue                          # the order-1 line just replaced
        out.append(line)
    directory = tmp_path / "species_apword2"
    directory.mkdir()
    (directory / "Si.in").write_text("".join(out))
    return directory


@pytest.fixture(scope="module")
def _module_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("lapw_export")


@pytest.fixture(scope="module", params=[1, 2], ids=["apword1", "apword2"])
def exported(request, _module_tmp):
    """The LAPW export at one generic k-point, at APW order 1 and 2.

    Module-scoped: each parameter converges its own ground state, and the
    export itself is a fraction of a second next to that.
    """
    apword = request.param
    sppath = None if apword == 1 else str(_apword2_species_dir(_module_tmp))
    structure = Structure(SI_AVEC, SI_SPECIES, sppath=sppath)
    calculation = structure.get_calculation(
        _module_tmp / f"si_apword{apword}", xc="PW", ngridk=(4, 4, 4))
    with calculation.eigenstate_session() as session:
        data = session.lapw_problem(KPOINT)
    assert data["apwordmax"] == apword
    return data


def test_exported_matrices_reproduce_elks_own_eigenvalues(exported):
    """Diagonalising the exported H and O must return Elk's own evalfv.

    This is the check that validates the export as a whole, and it is not a
    tautology: evalfv is computed by eveqnfv through Elk's configured path,
    BEFORE elkpy_lapwexport disables the `tefvr` real-matrix shortcut to
    build the matrices it writes. Diamond silicon has an inversion centre,
    so `tefvr` is true here and olpaa/hmlaa would otherwise accumulate only
    the real part of the muffin-tin APW-APW contribution (rzmctmu's
    dgemv strides by two over the complex array) -- an exported matrix that
    is wrong while remaining perfectly Hermitian and positive definite.

    It also pins the two parsing conventions in one shot: column-major
    ordering, and the upper-triangle-only fill that Elk actually performs.
    """
    scipy_linalg = pytest.importorskip("scipy.linalg")
    nstfv = exported["nstfv"]
    eigenvalues = scipy_linalg.eigh(
        exported["hmat"], exported["omat"], eigvals_only=True)
    assert eigenvalues[:nstfv] == pytest.approx(exported["evalfv"], abs=1e-10)


def test_apw_apw_block_is_the_matching_coefficient_gram_matrix(exported):
    """O's APW-APW block, minus its interstitial part, is exactly
    sum_{lm,io} conj(A_{i,io,lm}) A_{j,io,lm} summed over atoms.

    That is olpaa's zmctmu written out, and it ties the exported apwalm to
    the exported overlap through Elk's own assembly -- so a packing error in
    either would have to be shared by both to survive. The interstitial
    contribution is exported separately for exactly this reason: it is
    added into the same block by olpistl and would otherwise have to be
    reconstructed from the characteristic function.
    """
    ngp = exported["ngp"]
    muffin_tin = exported["omat"][:ngp, :ngp] - exported["omat_istl"]
    gram = np.zeros((ngp, ngp), dtype=complex)
    for atom in range(exported["natmtot"]):
        species = exported["idxis"][atom] - 1
        for l in range(exported["lmaxapw"] + 1):
            for lm in range(l * l, (l + 1) ** 2):
                for io in range(exported["apword"][l, species]):
                    a = exported["apwalm"][:, io, lm, atom]
                    gram += np.conj(a)[:, None] * a[None, :]
    assert np.abs(muffin_tin - gram).max() < 1e-12 * np.abs(muffin_tin).max()


def test_derivative_matrix_matches_an_independent_polynomial_fit(exported):
    """D_ij = d^{i-1} u_j / dr^{i-1} at R, rebuilt from the exported radial
    functions by numpy's own polynomial fit rather than by transcribing
    polynm.

    match.f90 evaluates rows 2.. of D by fitting a polynomial of order npapw
    through the last npapw mesh points of u_j and differentiating it; row 1
    is the value at the boundary point itself. Refitting with numpy and
    differentiating analytically is an independent implementation of that
    same object, so agreement checks both the D export and the alignment of
    the apwfr/rsp tails that Phase 1 will build D from. At apword=1 only
    row 1 exists and this reduces to D = u(R), which still pins that the
    last exported radial point is the muffin-tin boundary.

    At apword=2 it additionally pins D's ORIENTATION -- row is the
    derivative order, column the APW index -- because u and du/dE are
    unrelated functions, so a transposed D would compare d^1 u_0 against
    d^0 u_1 and fail. That is worth having explicitly: handing `match` a
    transposed D is one of the two errors docs/jax_port_phase0.md records
    as leaving the dmatch identity passing at 3e-16 while every coefficient
    is wrong by O(1).
    """
    npapw = exported["npapw"]
    worst = 0.0
    for atom in range(exported["natmtot"]):
        species = exported["idxis"][atom] - 1
        mesh = np.asarray(exported["rsp"][species], dtype=float)
        radius = float(exported["rmt"][species])
        assert mesh[-1] == pytest.approx(radius, rel=1e-12)
        for l in range(exported["lmaxapw"] + 1):
            matrix = exported["dmat"][atom][l]
            for column in range(matrix.shape[0]):
                values = np.asarray(
                    exported["apwfr"][atom][l][column], dtype=float)
                fit = np.polynomial.Polynomial.fit(mesh, values, deg=npapw - 1)
                for row in range(matrix.shape[0]):
                    reference = values[-1] if row == 0 else fit.deriv(row)(radius)
                    worst = max(worst, abs(matrix[row, column] - reference)
                                / max(abs(reference), 1e-30))
    assert worst < 1e-10


def test_jax_match_reproduces_elks_apwalm(exported):
    """Phase 0c's forward half, closed against Elk itself.

    docs/jax_port_phase0.md records that src/elkjax/lapw.py's `match` had
    only its own defining equation to check against, and that the identity
    dmatch asserts is blind to a constant factor per l block -- dropping
    genylmv's 4pi(-i)^l prefactor leaves that identity passing at 7e-16
    while every coefficient is wrong. An element-wise comparison against
    Elk's array is not blind to any of that.

    Elk's own G+k set and atomic positions are fed in rather than
    regenerated: an element-wise comparison needs identical ordering, and
    `tshift` has moved the origin relative to the input file, so the
    structure factor would otherwise be built at the wrong position.
    """
    pytest.importorskip("jax")
    import jax.numpy as jnp

    import elkjax  # noqa: F401  -- enables float64 before any array exists
    from elkjax.lapw import match

    ngp = exported["ngp"]
    vgkc = jnp.asarray(np.ascontiguousarray(exported["vgpc"][:, :ngp].T))
    gkc = jnp.asarray(exported["gpc"][:ngp])
    for atom in range(exported["natmtot"]):
        species = exported["idxis"][atom] - 1
        derivative_matrices = [
            jnp.asarray(m, dtype=jnp.complex128) for m in exported["dmat"][atom]]
        computed = np.asarray(match(
            exported["lmaxapw"], vgkc, gkc,
            jnp.asarray(exported["atposc"][:, atom]), derivative_matrices,
            float(exported["rmt"][species]), float(exported["omega"])))
        reference = exported["apwalm"][:, :, :, atom]
        scale = np.abs(reference).max()
        assert np.abs(computed - reference).max() < 1e-11 * scale


def test_overlap_condition_number_and_its_cholesky_lower_bound(exported):
    """kappa(O) on a real LAPW overlap, and how far the cheap estimate is
    from it.

    The study's section 8(b) proposes estimating kappa(S) from the spread of
    the Cholesky diagonal, and notes it is a provable LOWER bound (each
    L_ii^2 is a Schur complement pivot, hence between the smallest and
    largest eigenvalue) -- the dangerous direction, since the tolerance it
    feeds is meant to be an upper bound. This asserts the bound holds and
    records the real gap; the measured factors are in
    docs/jax_port_phase0.md.
    """
    overlap = exported["omat"]
    eigenvalues = np.linalg.eigvalsh(overlap)
    assert eigenvalues.min() > 0
    kappa = eigenvalues.max() / eigenvalues.min()
    pivots = np.abs(np.diag(np.linalg.cholesky(overlap))) ** 2
    estimate = pivots.max() / pivots.min()
    assert estimate <= kappa
    # a real LAPW overlap at a standard cutoff is ill-conditioned enough to
    # matter for the projector rule's tolerance, but nowhere near singular
    assert 1e2 < kappa < 1e6


def test_atomic_positions_are_elks_own(exported):
    """atposc is what Elk holds, not what the input file said.

    `tshift` is on by default and moves the origin onto the inversion
    centre -- for diamond silicon that is the bond midpoint, so both atoms
    come back displaced from (0,0,0) and (1/4,1/4,1/4). Their SEPARATION is
    invariant, and that is what this checks: the export is the right
    positions in the frame the structure factor actually uses.
    """
    avec = exported["avec"]
    separation = exported["atposc"][:, 1] - exported["atposc"][:, 0]
    expected = avec @ np.array([0.25, 0.25, 0.25])
    lattice_offset = np.linalg.solve(avec, separation - expected)
    assert lattice_offset == pytest.approx(np.round(lattice_offset), abs=1e-10)


def test_the_gk_set_is_exactly_the_cutoff_sphere(exported):
    """Elk's G+k set is {G : |G+k| < rgkmax / min(rmt)}, and the exported
    lattice/Cartesian pair is consistent with it.

    src/elkjax/lapw.py's `gkvectors` assumes exactly that rule, and had no
    way to check it -- the assumption is not the formula but the details:
    which cutoff (`gkmax` is derived from the SMALLEST muffin-tin radius,
    itself shrunk by checkmt from the species file's value), strict versus
    inclusive comparison, and whether Elk's own G-vector set is large enough
    not to truncate the sphere. Enumerating every integer triple in a
    generous box and selecting on |G+k| is an independent construction of
    the same set, so this checks all three at once. On bulk Si at rgkmax=7
    the two sets agree exactly, 153 vectors.
    """
    ngp = exported["ngp"]
    bvec = exported["bvec"]
    vgpl = exported["vgpl"][:, :ngp]
    # bvec's columns are the reciprocal lattice vectors, so vgpc = bvec @ vgpl
    assert np.abs(exported["vgpc"][:, :ngp] - bvec @ vgpl).max() < 1e-12
    assert np.abs(np.linalg.norm(exported["vgpc"][:, :ngp], axis=0)
                  - exported["gpc"][:ngp]).max() < 1e-12

    k_lattice = np.linalg.solve(bvec, exported["vkc"])
    assert k_lattice == pytest.approx(KPOINT, abs=1e-10)
    gvectors = vgpl - k_lattice[:, None]
    assert np.abs(gvectors - np.round(gvectors)).max() < 1e-10

    gkmax = 7.0 / exported["rmt"].min()          # Calculation's default rgkmax
    box = 8
    candidates = np.array(list(itertools.product(range(-box, box + 1), repeat=3)))
    lengths = np.linalg.norm((candidates + k_lattice) @ bvec.T, axis=1)
    enumerated = {tuple(v) for v in candidates[lengths < gkmax]}
    exported_set = {tuple(v) for v in np.round(gvectors).astype(int).T}
    assert enumerated == exported_set


def test_exported_eigenvectors_solve_the_exported_eigenproblem(exported):
    """evecfv satisfies H V = O V diag(evalfv) and V^H O V = 1.

    Nothing else here touches evecfv, and Phase 1's forward criterion is
    stated on it -- gauge-invariantly, as the occupied-subspace projector
    P = V V^H O, since EVECFV is arbitrary within a degenerate multiplet.
    So this checks the object that criterion will be measured against, and
    that it is idempotent.

    It also confirms the tefvr reasoning from the other direction. evecfv
    comes out of eveqnfvr, Elk's real symmetric solver, which never forms
    the complex matrices at all; H and O are built afterwards with the
    shortcut disabled. A residual of order machine epsilon between the two
    is evidence that the exported matrices are the ones Elk's own solution
    belongs to -- had the real-part-only accumulation survived into the
    export, this residual would be O(1) rather than 1e-15.
    """
    hamiltonian, overlap = exported["hmat"], exported["omat"]
    vectors, values = exported["evecfv"], exported["evalfv"]
    nstfv = exported["nstfv"]

    gram = vectors.conj().T @ overlap @ vectors
    assert np.abs(gram - np.eye(nstfv)).max() < 1e-12

    residual = hamiltonian @ vectors - (overlap @ vectors) * values[None, :]
    assert np.abs(residual).max() < 1e-12 * np.linalg.norm(hamiltonian, 2)

    occupied = vectors[:, :4]                      # silicon's valence manifold
    projector = occupied @ occupied.conj().T @ overlap
    assert np.linalg.norm(projector @ projector - projector) < 1e-11
