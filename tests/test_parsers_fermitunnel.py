"""Unit tests for elkpy.parsers.fermitunnel -- the momentum-resolved
tunnelling Fermi surface (docs/design.md #35).

No Elk binary: everything here runs on synthetic Gram matrices, exactly as
tests/test_parsers_transport.py does for the point-tip map one level down. The
test that carries the weight is `test_contraction_is_the_double_integral`: the
contraction Tr[D G_t D S] differs from the tempting elementwise form only by a
transpose of a Hermitian matrix, so the wrong one is real, non-negative, blind
to a degenerate rotation and exactly right whenever the exit Gram is diagonal.
Only a literal evaluation of the double plane integral separates them, and
only when the two planes are DISTINCT -- which is why that test builds two
independent samplings and asserts the two forms actually disagree.
"""

import numpy as np
import pytest

from elkpy.parsers import fermitunnel, transport


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------

def _random_states(rng, nspinor, nst, npts):
    return (rng.normal(size=(nspinor, nst, npts))
            + 1j * rng.normal(size=(nspinor, nst, npts)))


def _gram(psi, area):
    """The same integral parsers.transport.gram_matrix_by_quadrature does,
    spin-summed: S[n, m] = int psi*_n psi_m."""
    npts = psi.shape[2]
    return (area / npts) * np.einsum(
        "snp,smp->nm", psi.conj(), psi, optimize=True
    )


def _f(x):
    """A number the way Fortran's G24.15 writes it -- numpy's own repr()
    renders as np.float64(0.3), which the parser cannot read."""
    return f"{float(x):.17g}"


def _write_export(path, kpoints, eigenvalues, tips, exits, *, nspinor=1,
                  grid=(2, 1, 1), efermi=0.3, window=(-0.04, 0.04)):
    """Write a file in src/elkpy_fermitunnel.f90's own format."""
    lines = ["elkpy tunnelling Fermi surface: synthetic"]
    lines.append(
        f"{len(kpoints):8d}{nspinor:8d}{grid[0]:8d}{grid[1]:8d}{grid[2]:8d}"
        f"{3:8d} : nkpt nspinor kgrid(1:3) plane-axis"
    )
    lines.append(f"{_f(efermi)} 0.75 0.25 12.0 240.0 : efermi tip exit area omega")
    lines.append("0.0 0.0 0.0 : koffset")
    lines.append(f"{_f(window[0])} {_f(window[1])} : window")
    lines.append("0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 : tpol tdir spol sdir")
    for ik, kv in enumerate(kpoints):
        eig = eigenvalues[ik]
        cart = np.asarray(kv) * 2.0
        lines.append(
            " ".join(_f(x) for x in list(kv) + list(cart))
            + f" {_f(1.0 / len(kpoints))} {len(eig):8d} : k, kc, weight"
        )
        for ist, e in enumerate(eig):
            lines.append(f"{ist + 1:8d} {_f(e)}")
        for matrix in (tips[ik], exits[ik]):
            # column by column, Fortran's own storage order
            for column in np.asarray(matrix).T:
                for z in column:
                    lines.append(f"{_f(z.real)} {_f(z.imag)}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# (a) the format
# ---------------------------------------------------------------------------

def test_parse_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    eig = [np.array([0.29, 0.31]), np.array([0.30])]
    tips = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)),
            np.array([[1.5 + 0j]])]
    exits = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)),
             np.array([[0.5 + 0j]])]
    path = tmp_path / "ELKPY_FERMITUNNEL.OUT"
    _write_export(path, [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], eig, tips, exits)
    data = fermitunnel.parse_fermitunnel(path)

    assert data["nspinor"] == 1
    assert data["grid"] == (2, 1, 1)
    assert data["efermi"] == pytest.approx(0.3)
    assert data["tip_height"] == pytest.approx(0.75)
    assert data["exit_height"] == pytest.approx(0.25)
    assert data["window"] == pytest.approx((-0.04, 0.04))
    assert data["kpoints"].shape == (2, 3)
    assert data["kpoints"][1] == pytest.approx([0.5, 0.0, 0.0])
    # the Cartesian column is independent data, not recomputed by the parser
    assert data["kpoints_cartesian"][1] == pytest.approx([1.0, 0.0, 0.0])
    assert data["weights"] == pytest.approx([0.5, 0.5])
    assert data["eigenvalues"][0] == pytest.approx([0.29, 0.31])
    # the column-major unpacking is what a transposed matrix would catch
    assert data["tip_overlaps"][0] == pytest.approx(tips[0])
    assert data["overlaps"][0] == pytest.approx(exits[0])
    assert data["overlaps"][1].shape == (1, 1)


def test_parse_handles_a_k_point_with_no_states(tmp_path):
    path = tmp_path / "ELKPY_FERMITUNNEL.OUT"
    _write_export(
        path, [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
        [np.array([0.3]), np.empty(0)],
        [np.array([[2.0 + 0j]]), np.empty((0, 0), dtype=complex)],
        [np.array([[1.0 + 0j]]), np.empty((0, 0), dtype=complex)],
    )
    data = fermitunnel.parse_fermitunnel(path)
    assert data["eigenvalues"][1].size == 0
    result = fermitunnel.compute_fermi_weight(data, broadening=0.01)
    assert result["weight"][0, 1] == 0.0
    assert result["weight"][0, 0] > 0.0


def test_reshape_grid_runs_the_first_index_fastest():
    # the write loop is ik-1 = i1 + n1*(i2 + n2*i3), so a (nk,) column folds
    # as (n3, n2, n1) -- getting this backwards transposes every image
    grid = (4, 3, 1)
    values = np.arange(12.0)
    folded = fermitunnel.reshape_grid(values, grid)
    assert folded.shape == (3, 4)          # (n2, n1), n3 == 1 dropped
    assert folded[0] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert folded[1, 0] == pytest.approx(4.0)
    # a stack of energies keeps its leading axis
    stacked = fermitunnel.reshape_grid(np.stack([values, values + 12]), grid)
    assert stacked.shape == (2, 3, 4)


# ---------------------------------------------------------------------------
# (b) the contraction -- the transpose trap
# ---------------------------------------------------------------------------

def test_contraction_is_the_double_integral():
    """Tr[D G_t D S] must reproduce int_tip int_exit |G|^2 evaluated
    literally, and the elementwise form must NOT.

    Two DISTINCT planes are essential: with one plane the two Gram matrices
    coincide and both forms give the same real number, so the test would pass
    while proving nothing.
    """
    rng = np.random.default_rng(11)
    nst, tip_area, exit_area = 5, 3.5, 3.5
    for nspinor in (1, 2):
        psi_tip = _random_states(rng, nspinor, nst, 40)
        psi_exit = _random_states(rng, nspinor, nst, 31)
        gtip = _gram(psi_tip, tip_area)
        sexit = _gram(psi_exit, exit_area)
        g = rng.uniform(0.2, 1.5, size=nst)

        literal = fermitunnel.quadrature_weight(
            psi_tip, psi_exit, tip_area, exit_area, g
        )
        assert fermitunnel.weight_at_k(gtip, sexit, g) == pytest.approx(literal)

        # the wrong form: Tr[D G_t D S^T]. Real, non-negative, and different
        wrong = float(np.real(np.sum(
            (g[:, None] * gtip * g[None, :]) * sexit
        )))
        assert wrong == pytest.approx(wrong.real)
        assert abs(wrong - literal) > 1e-3 * abs(literal)


def test_weight_is_real_and_non_negative():
    """Both Gram matrices are Hermitian positive semi-definite, so the weight
    is a real, non-negative number for any state weights."""
    rng = np.random.default_rng(3)
    for nst in (1, 2, 6):
        psi_tip = _random_states(rng, 1, nst, 25)
        psi_exit = _random_states(rng, 1, nst, 25)
        g = rng.uniform(0.0, 2.0, size=nst)
        w = fermitunnel.weight_at_k(_gram(psi_tip, 2.0), _gram(psi_exit, 2.0), g)
        assert w >= 0.0


def test_shape_mismatch_is_refused():
    g = np.ones(3)
    with pytest.raises(ValueError, match="tip-plane"):
        fermitunnel.weight_at_k(np.eye(2, dtype=complex),
                                np.eye(3, dtype=complex), g)
    with pytest.raises(ValueError, match="exit-plane"):
        fermitunnel.weight_at_k(np.eye(3, dtype=complex),
                                np.eye(2, dtype=complex), g)


# ---------------------------------------------------------------------------
# (c) the two identity limits
# ---------------------------------------------------------------------------

def test_both_cell_is_the_plain_fermi_surface(tmp_path):
    """G_t = S = 1 gives occmax sum_n delta_eta(E - eps), which is the plain
    smeared-delta Fermi surface -- and it must equal the "bare" column that
    every call returns."""
    rng = np.random.default_rng(5)
    eig = [np.array([0.295, 0.302, 0.311])]
    tips = [rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))]
    tips[0] = tips[0] + tips[0].conj().T
    exits = [np.diag([1.0, 2.0, 3.0]).astype(complex)]
    path = tmp_path / "ELKPY_FERMITUNNEL.OUT"
    _write_export(path, [(0.0, 0.0, 0.0)], eig, tips, exits)
    data = fermitunnel.parse_fermitunnel(path)

    eta = 0.01
    got = fermitunnel.compute_fermi_weight(
        data, broadening=eta, tip_region="cell", exit_region="cell"
    )
    expect = 2.0 * np.sum(
        transport.smeared_delta((0.3 - eig[0]) / eta, 3) / eta
    )
    assert got["weight"][0, 0] == pytest.approx(expect)
    assert got["bare"][0, 0] == pytest.approx(expect)
    # "bare" is the plain Fermi surface whatever the regions are set to
    plane = fermitunnel.compute_fermi_weight(data, broadening=eta)
    assert plane["bare"][0, 0] == pytest.approx(expect)
    assert plane["weight"][0, 0] != pytest.approx(expect)


def test_exit_cell_is_the_tip_diagonal(tmp_path):
    """S = 1 alone leaves sum_n g_n^2 G_t[n, n] -- a planar Tersoff-Hamann
    image in momentum space, blind to every off-diagonal of the tip Gram."""
    rng = np.random.default_rng(7)
    eig = [np.array([0.298, 0.303])]
    gtip = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    gtip = gtip + gtip.conj().T
    path = tmp_path / "ELKPY_FERMITUNNEL.OUT"
    _write_export(path, [(0.0, 0.0, 0.0)], eig, [gtip],
                  [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))])
    data = fermitunnel.parse_fermitunnel(path)

    eta = 0.01
    got = fermitunnel.compute_fermi_weight(
        data, broadening=eta, exit_region="cell"
    )
    g = transport.amplitude_weights(eig[0], 0.3, eta, 3, 2.0)
    assert got["weight"][0, 0] == pytest.approx(
        float(np.dot(g**2, np.real(np.diag(gtip))))
    )


# ---------------------------------------------------------------------------
# (d) gauge freedom inside a degenerate multiplet
# ---------------------------------------------------------------------------

def test_coherent_weight_survives_a_degenerate_rotation():
    """A degenerate eigensolver is free to return any basis inside a
    multiplet (docs/design.md #14). The coherent weight is invariant because D
    is a multiple of the identity there; the incoherent one is a diagonal and
    is not, which is why it is taken in the substrate's channel basis."""
    rng = np.random.default_rng(13)
    nst = 4
    eig = np.array([0.30, 0.30, 0.30, 0.34])   # a triplet plus a spectator
    psi_tip = _random_states(rng, 1, nst, 30)
    psi_exit = _random_states(rng, 1, nst, 30)
    gtip, sexit = _gram(psi_tip, 2.0), _gram(psi_exit, 2.0)
    g = transport.amplitude_weights(eig, 0.30, 0.01, 3, 2.0)

    # a random unitary acting only inside the triplet
    u = np.eye(nst, dtype=complex)
    block = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    u[:3, :3] = np.linalg.qr(block)[0]

    coherent = fermitunnel.weight_at_k(gtip, sexit, g)
    rotated = fermitunnel.weight_at_k(
        u.conj().T @ gtip @ u, u.conj().T @ sexit @ u, g
    )
    assert rotated == pytest.approx(coherent)

    incoherent = fermitunnel.weight_at_k(gtip, sexit, g, False, eig)
    rotated_inc = fermitunnel.weight_at_k(
        u.conj().T @ gtip @ u, u.conj().T @ sexit @ u, g, False, eig
    )
    assert rotated_inc == pytest.approx(incoherent)
    # and the naive diagonal, taken in whatever basis arrived, is NOT
    naive = float(np.real(np.dot(
        g**2, np.diag(gtip) * np.diag(sexit)
    )))
    naive_rot = float(np.real(np.dot(
        g**2,
        np.diag(u.conj().T @ gtip @ u) * np.diag(u.conj().T @ sexit @ u),
    )))
    assert abs(naive - naive_rot) > 1e-8 * abs(naive)


def test_interference_is_the_coherent_minus_incoherent_part(tmp_path):
    rng = np.random.default_rng(17)
    eig = [np.array([0.299, 0.305])]
    psi_tip = _random_states(rng, 1, 2, 20)
    psi_exit = _random_states(rng, 1, 2, 20)
    path = tmp_path / "ELKPY_FERMITUNNEL.OUT"
    _write_export(path, [(0.0, 0.0, 0.0)], eig,
                  [_gram(psi_tip, 2.0)], [_gram(psi_exit, 2.0)])
    data = fermitunnel.parse_fermitunnel(path)
    got = fermitunnel.compute_fermi_weight(data, broadening=0.01)
    assert got["interference"] == pytest.approx(got["weight"] - got["incoherent"])
    assert got["hermiticity"] < 1e-12
    assert got["least_eigenvalue"] > -1e-12


# ---------------------------------------------------------------------------
# (e) the traps the point-tip parser already documents, one level up
# ---------------------------------------------------------------------------

def test_an_energy_outside_the_window_is_refused(tmp_path):
    """The bias/absolute trap: the smeared delta's exponential tails would
    otherwise return a plausible small map at the wrong energy."""
    path = tmp_path / "ELKPY_FERMITUNNEL.OUT"
    _write_export(path, [(0.0, 0.0, 0.0)], [np.array([0.3])],
                  [np.array([[1.0 + 0j]])], [np.array([[1.0 + 0j]])],
                  efermi=0.3, window=(-0.04, 0.04))
    data = fermitunnel.parse_fermitunnel(path)
    fermitunnel.compute_fermi_weight(data, energies=0.32)      # inside
    with pytest.raises(ValueError, match="ABSOLUTE Hartree"):
        fermitunnel.compute_fermi_weight(data, energies=0.02)  # a bias
    with pytest.raises(ValueError, match="ABSOLUTE Hartree"):
        fermitunnel.compute_fermi_weight(data, energies=0.5)


def test_unknown_region_is_refused(tmp_path):
    path = tmp_path / "ELKPY_FERMITUNNEL.OUT"
    _write_export(path, [(0.0, 0.0, 0.0)], [np.array([0.3])],
                  [np.array([[1.0 + 0j]])], [np.array([[1.0 + 0j]])])
    data = fermitunnel.parse_fermitunnel(path)
    with pytest.raises(ValueError, match="unknown tip_region"):
        fermitunnel.compute_fermi_weight(data, tip_region="vacuum")
    with pytest.raises(ValueError, match="unknown exit_region"):
        fermitunnel.compute_fermi_weight(data, exit_region="vacuum")


def test_total_is_the_k_integral(tmp_path):
    rng = np.random.default_rng(19)
    kpts = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    eig = [np.array([0.30]), np.array([0.31])]
    tips = [np.array([[2.0 + 0j]]), np.array([[0.5 + 0j]])]
    exits = [np.array([[1.0 + 0j]]), np.array([[3.0 + 0j]])]
    path = tmp_path / "ELKPY_FERMITUNNEL.OUT"
    _write_export(path, kpts, eig, tips, exits)
    data = fermitunnel.parse_fermitunnel(path)
    got = fermitunnel.compute_fermi_weight(data, broadening=0.01)
    assert got["total"][0] == pytest.approx(
        float(got["weight"][0] @ data["weights"])
    )
    assert rng is not None


# ---------------------------------------------------------------------------
# (f) the three levels: DOS, Tersoff-Hamann, full Green's function
# ---------------------------------------------------------------------------

def test_the_three_levels_are_three_contractions_of_one_export(tmp_path):
    """"bare", "tersoff_hamann" and "weight" must be the plain DOS, the LDOS
    at the tip plane, and the full Tr[D G_t D S] -- from ONE export, on the
    same eigenvalues, which is what makes comparing them meaningful.

    The Tersoff-Hamann one adds bands as PROBABILITIES (a diagonal) and the
    full one as AMPLITUDES, so they may not agree once the tip Gram has an
    off-diagonal; that difference is the quantity the comparison exists for.
    """
    rng = np.random.default_rng(23)
    eig = [np.array([0.298, 0.303])]
    psi_tip = _random_states(rng, 1, 2, 30)
    psi_exit = _random_states(rng, 1, 2, 30)
    gtip, sexit = _gram(psi_tip, 2.0), _gram(psi_exit, 2.0)
    path = tmp_path / "ELKPY_FERMITUNNEL.OUT"
    _write_export(path, [(0.0, 0.0, 0.0)], eig, [gtip], [sexit])
    data = fermitunnel.parse_fermitunnel(path)

    eta = 0.01
    got = fermitunnel.compute_fermi_weight(data, broadening=eta)
    g = transport.amplitude_weights(eig[0], 0.3, eta, 3, 2.0)

    assert got["bare"][0, 0] == pytest.approx(float(np.sum(g**2)))
    assert got["tersoff_hamann"][0, 0] == pytest.approx(
        float(np.dot(g**2, np.real(np.diag(gtip))))
    )
    assert got["weight"][0, 0] == pytest.approx(
        fermitunnel.weight_at_k(gtip, sexit, g)
    )
    # the three are genuinely different numbers on this data
    assert got["bare"][0, 0] != pytest.approx(got["tersoff_hamann"][0, 0])
    assert got["tersoff_hamann"][0, 0] != pytest.approx(got["weight"][0, 0])
    # and each has its k-integral
    for column, total in (("bare", "total_bare"),
                          ("tersoff_hamann", "total_tersoff_hamann"),
                          ("weight", "total")):
        assert got[total][0] == pytest.approx(
            float(got[column][0] @ data["weights"])
        )


def test_tersoff_hamann_ignores_the_region_switches(tmp_path):
    """It is a fixed reference, like "bare": setting tip_region="cell" asks
    for `weight` without the tip weighting, and must not silently redefine the
    Tersoff-Hamann curve it is being compared against."""
    rng = np.random.default_rng(29)
    eig = [np.array([0.30])]
    gtip = np.array([[3.0 + 0j]])
    path = tmp_path / "ELKPY_FERMITUNNEL.OUT"
    _write_export(path, [(0.0, 0.0, 0.0)], eig, [gtip],
                  [np.array([[0.5 + 0j]])])
    data = fermitunnel.parse_fermitunnel(path)
    reference = fermitunnel.compute_fermi_weight(data, broadening=0.01)
    for tip, exit_ in (("cell", "plane"), ("cell", "cell"), ("plane", "cell")):
        got = fermitunnel.compute_fermi_weight(
            data, broadening=0.01, tip_region=tip, exit_region=exit_)
        assert got["tersoff_hamann"] == pytest.approx(
            reference["tersoff_hamann"])
        assert got["bare"] == pytest.approx(reference["bare"])


def test_an_energy_sweep_gives_three_spectra(tmp_path):
    """Sweeping the energy is the whole point of the comparison and costs one
    contraction per energy: neither the wavefunctions nor the Gram matrices
    depend on it, only the on-shell weights do."""
    rng = np.random.default_rng(31)
    eig = [np.array([0.28, 0.30, 0.32])]
    psi_tip = _random_states(rng, 1, 3, 30)
    path = tmp_path / "ELKPY_FERMITUNNEL.OUT"
    _write_export(path, [(0.0, 0.0, 0.0)], eig, [_gram(psi_tip, 2.0)],
                  [_gram(_random_states(rng, 1, 3, 30), 2.0)],
                  window=(-0.06, 0.06))
    data = fermitunnel.parse_fermitunnel(path)
    energies = np.linspace(0.26, 0.34, 17)
    got = fermitunnel.compute_fermi_weight(data, energies=energies,
                                           broadening=0.005)
    for name in ("bare", "tersoff_hamann", "weight"):
        assert got[name].shape == (17, 1)
        assert np.all(got[name] > 0.0)
    # the DOS must peak where the bands are, not where the tip weighting is
    assert np.argmax(got["bare"][:, 0]) in (7, 8, 9)      # 0.30, the middle
    # and one energy at a time gives the same numbers as the whole sweep
    single = fermitunnel.compute_fermi_weight(data, energies=energies[5],
                                              broadening=0.005)
    for name in ("bare", "tersoff_hamann", "weight"):
        assert single[name][0] == pytest.approx(got[name][5])
