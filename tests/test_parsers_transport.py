"""Synthetic pins for elkpy.parsers.transport -- no Elk run.

Everything here is arithmetic on hand-built overlaps and amplitudes, in the
same spirit as tests/test_berry_gauge_invariance.py and
tests/test_parsers_optical.py. The checks that carry weight:

* the CONJUGATION. G(r, r') carries psi conjugated in the exit variable, so
  the plane integral of |G|^2 is sum_nm a_n a*_m S[n,m] and not
  sum_nm a*_n a_m S[n,m]. The two differ by a transpose of a Hermitian
  matrix, so the wrong one is real, non-negative, blind to a rotation inside
  a degenerate multiplet, and exactly right in the Tersoff-Hamann limit where
  S is the identity. Only a literal evaluation of the definition tells them
  apart, which is what green_function_transmission() is for;
* the Tersoff-Hamann limit with NO factor -- the one thing a wrong overall
  normalisation could not hide in, since every other check here is a ratio;
* the gauge freedom inside a degenerate multiplet: the coherent map must be
  blind to it and the incoherent one is not, until channel_basis() fixes it.
"""

import numpy as np
import pytest

from elkpy.parsers import transport as T


# --------------------------------------------------------------------------
# a synthetic band-limited "wavefunction" on a plane, from which both the
# closed-form Gram matrix and a real-space quadrature can be built exactly
# --------------------------------------------------------------------------

def _synthetic_plane(nst=3, hmax=2, ngrid=16, nspinor=1, seed=0, area=7.5):
    """Random in-plane Fourier coefficients, and the samples they give.

    The rectangle rule over `ngrid` x `ngrid` fractional points is EXACT for
    these once ngrid > 2*(2*hmax), the integrand being periodic and band
    limited -- the same statement that makes src/elkpy_transport.f90's
    G-parallel collapse exact rather than a quadrature.
    """
    rng = np.random.default_rng(seed)
    hs = np.array([(h1, h2) for h1 in range(-hmax, hmax + 1)
                   for h2 in range(-hmax, hmax + 1)])
    c = (rng.normal(size=(nspinor, nst, len(hs)))
         + 1j * rng.normal(size=(nspinor, nst, len(hs))))
    frac = np.stack(np.meshgrid(np.arange(ngrid) / ngrid,
                                np.arange(ngrid) / ngrid, indexing="ij"),
                    axis=-1).reshape(-1, 2)
    phase = np.exp(2j * np.pi * frac @ hs.T)          # (npts, nh)
    psi = np.einsum("snh,ph->snp", c, phase, optimize=True)
    # the closed form: S = A sum_h conj(c_n) c_n'
    overlap = area * np.einsum("snh,smh->nm", c.conj(), c, optimize=True)
    return psi, overlap, area


def test_the_closed_form_gram_matrix_is_the_plane_quadrature():
    """The G-parallel orthogonality collapse against a literal rectangle rule.

    This is the Python-side twin of what tests/test_calculation_transport.py
    checks on real Elk coefficients; here it pins the CONVENTION
    S[n,n'] = int psi*_n psi_n' (conjugate on the first index) as well as the
    arithmetic.
    """
    psi, overlap, area = _synthetic_plane()
    quad = T.gram_matrix_by_quadrature(psi, area)
    assert np.abs(quad - overlap).max() < 1.0e-12 * np.abs(overlap).max()


def test_the_gram_matrix_is_hermitian_and_positive_semidefinite():
    """Not decoration: it is what makes the transmission non-negative by
    construction rather than by luck."""
    _, overlap, _ = _synthetic_plane(nst=5, seed=3)
    assert np.abs(overlap - overlap.conj().T).max() < 1.0e-12
    assert np.linalg.eigvalsh(overlap).min() > -1.0e-12


def test_the_contraction_is_the_literal_green_function_integral():
    """THE conjugation pin, and the reason green_function_transmission exists.

    Evaluate G(r, r') = sum_n a_n(r) psi*_n(r') on the plane and integrate
    |G|^2 by quadrature, against the fast sum_nm a_n a*_m S[n,m].
    """
    psi, overlap, area = _synthetic_plane(nst=4, seed=7)
    weights = np.array([0.3, 1.0, 0.7, 0.2])
    fast = T.transmission_at_k(psi, overlap, weights)
    literal = T.green_function_transmission(psi, psi, area, weights)
    assert np.abs(fast - literal).max() < 1.0e-10 * fast.max()


def test_the_transposed_convention_is_wrong_and_nothing_else_catches_it():
    """a^dagger S a passes every structural check and is still wrong.

    It is real, non-negative, and agrees exactly in the Tersoff-Hamann limit;
    it differs from the definition only where S has an off-diagonal, which is
    where the interference this module exists for lives.
    """
    psi, overlap, area = _synthetic_plane(nst=4, seed=11)
    weights = np.ones(4)
    literal = T.green_function_transmission(psi, psi, area, weights)
    wrong = T.transmission_at_k(psi, overlap.T.copy(), weights)
    assert np.all(wrong >= -1.0e-12)                     # still non-negative
    assert np.abs(wrong - literal).max() > 1.0e-3 * literal.max()
    # and it coincides in the Tersoff-Hamann limit, where S is the identity
    identity = np.eye(4, dtype=complex)
    assert np.allclose(T.transmission_at_k(psi, identity, weights),
                       T.transmission_at_k(psi, identity.T.copy(), weights))


def test_the_whole_cell_limit_is_the_tunnelling_density_of_states():
    """S -> identity gives sum_n |psi_n(r)|^2 w_n^2 with NO factor.

    The normalisation of amplitude_weights is chosen so that this is exactly
    the local density of states elkpy's task-9003 STM image computes; a factor
    here would be a factor everywhere and no ratio-shaped check could see it.
    """
    psi, _, _ = _synthetic_plane(nst=3, seed=5)
    weights = np.array([0.5, 1.5, 0.9])
    got = T.transmission_at_k(psi, np.eye(3, dtype=complex), weights)
    want = np.einsum("n,snp->p", weights**2, np.abs(psi)**2)
    assert np.abs(got - want).max() < 1.0e-12 * want.max()


def test_the_coherent_map_is_blind_to_a_rotation_inside_a_multiplet():
    """T_coherent is a quadratic form, so the arbitrary basis a degenerate
    eigensolver returns inside a multiplet (docs/design.md #14) cannot change
    it -- while the incoherent diagonal can, until channel_basis fixes it."""
    rng = np.random.default_rng(2)
    psi, overlap, _ = _synthetic_plane(nst=4, seed=13)
    weights = np.array([1.0, 1.0, 0.4, 0.2])
    eigenvalues = np.array([0.0, 0.0, 0.5, 0.9])         # states 0,1 degenerate
    q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    u = np.eye(4, dtype=complex)
    u[:2, :2] = q
    psi_r = np.einsum("ni,snp->sip", u, psi, optimize=True)
    overlap_r = u.conj().T @ overlap @ u

    coherent = T.transmission_at_k(psi, overlap, weights)
    coherent_r = T.transmission_at_k(psi_r, overlap_r, weights)
    assert np.abs(coherent - coherent_r).max() < 1.0e-10 * coherent.max()

    naive = np.einsum("n,snp->p", np.real(np.diag(overlap)),
                      np.abs(psi * weights[None, :, None])**2)
    naive_r = np.einsum("n,snp->p", np.real(np.diag(overlap_r)),
                        np.abs(psi_r * weights[None, :, None])**2)
    assert np.abs(naive - naive_r).max() > 1.0e-6 * naive.max()

    fixed = T.transmission_at_k(psi, overlap, weights, coherent=False,
                                eigenvalues=eigenvalues)
    fixed_r = T.transmission_at_k(psi_r, overlap_r, weights, coherent=False,
                                  eigenvalues=eigenvalues)
    assert np.abs(fixed - fixed_r).max() < 1.0e-10 * fixed.max()


def test_a_multiple_of_the_identity_leaves_nothing_to_interfere_through():
    """Schur's lemma, as arithmetic: when the substrate cannot tell the members
    of a degenerate multiplet apart, S restricted to it is a multiple of the
    identity and the coherent map collapses onto the local one. This is exactly
    why monolayer graphene's transport map IS its STM image."""
    psi, _, _ = _synthetic_plane(nst=2, seed=17)
    weights = np.ones(2)
    s = 0.37 * np.eye(2, dtype=complex)
    coherent = T.transmission_at_k(psi, s, weights)
    incoherent = T.transmission_at_k(psi, s, weights, coherent=False,
                                     eigenvalues=np.zeros(2))
    assert np.abs(coherent - incoherent).max() < 1.0e-12 * coherent.max()


def test_the_spinor_substrate_projector_partitions_the_transmission():
    """A substrate polarized along n plus one along -n is a substrate that
    takes both spins, exactly. The projector 1 + P n.sigma sums to 2 over the
    two signs, which is twice the unpolarized 1 -- Elk's own STM
    normalisation, n + P m.e, not the Landauer (1 + P n.sigma)/2."""
    psi, _, area = _synthetic_plane(nst=3, nspinor=2, seed=19)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    up = T.gram_matrix_by_quadrature(psi, area, np.eye(2) + sigma_z)
    down = T.gram_matrix_by_quadrature(psi, area, np.eye(2) - sigma_z)
    both = T.gram_matrix_by_quadrature(psi, area, np.eye(2))
    assert np.abs(up + down - 2 * both).max() < 1.0e-12 * np.abs(both).max()


# --------------------------------------------------------------------------
# the energy weight
# --------------------------------------------------------------------------

def test_methfessel_paxton_is_refused_because_an_amplitude_has_no_square_root():
    for stype in (1, 2):
        with pytest.raises(ValueError, match="Methfessel-Paxton"):
            T.smeared_delta(np.linspace(-3, 3, 11), stype)


@pytest.mark.parametrize("stype, tol", [(0, 1.0e-6), (3, 1.0e-6), (5, 2.0e-3)])
def test_the_smeared_deltas_are_normalised_and_positive(stype, tol):
    """Positive (so the on-shell amplitude, their square root, exists) and
    normalised. The Lorentzian's looser tolerance is the point rather than a
    concession: it falls off as 1/x^2, so its SQUARE ROOT falls off only as
    1/|x| and the band sum built on it converges slowly -- which is why a
    Gaussian (stype 0) is the right choice when the export window is narrow."""
    x = np.linspace(-400.0, 400.0, 400001)
    d = T.smeared_delta(x, stype)
    assert d.min() >= 0.0
    assert abs(np.trapezoid(d, x) - 1.0) < tol


def test_the_amplitude_weight_squares_to_the_density_of_states_weight():
    """|a|^2 = occmax delta(E - eps)/eta, which is src/occupy.f90's own
    dI/dV weight -- the statement that makes the Tersoff-Hamann limit exact."""
    eps = np.array([-0.01, 0.0, 0.02])
    w = T.amplitude_weights(eps, 0.0, 0.005, stype=3, occmax=2.0)
    want = 2.0 * T.smeared_delta((0.0 - eps) / 0.005, 3) / 0.005
    assert np.allclose(w**2, want)
    assert np.all(w >= 0.0)


# --------------------------------------------------------------------------
# the file format
# --------------------------------------------------------------------------

def test_parse_transport_round_trip(tmp_path):
    """Pins the two index orders src/elkpy_transport.f90 writes in: the Gram
    matrix column by column (Fortran's own storage order), the amplitudes with
    the plotting-point index fastest."""
    nsel, npts, nspinor = 2, 3, 1
    s = np.array([[1.0 + 0j, 2.0 + 3.0j], [2.0 - 3.0j, 4.0 + 0j]])
    amp = np.arange(nsel * npts).reshape(1, nsel, npts) + 1j
    lines = [
        "elkpy vertical transport",
        f"  1  {nspinor}  {npts}  3  1  3 : nkpt nspinor npoints np2d(1) np2d(2) axis",
        " -0.1  0.38  7.5  100.0 : efermi exit-height area omega",
        " -0.05 0.05 0.0 0.0 0.0 1.0 : window polarisation direction",
        "tip points:",
    ]
    for ip in range(npts):
        lines.append(f" {ip}.0 0.0 {ip/npts} 0.0 0.62")
    lines.append(" 0.0 0.0 0.0 1.0 2 : k, weight, states")
    lines += ["  4 -0.02", "  5 0.01"]
    for j in range(nsel):                       # column by column
        for i in range(nsel):
            lines.append(f" {s[i, j].real} {s[i, j].imag}")
    for isp in range(nspinor):
        for i in range(nsel):
            for ip in range(npts):
                lines.append(f" {amp[isp, i, ip].real} {amp[isp, i, ip].imag}")
    path = tmp_path / "ELKPY_TRANSPORT.OUT"
    path.write_text("\n".join(lines) + "\n")

    d = T.parse_transport(path)
    assert d["npoints"] == npts and d["nspinor"] == nspinor
    assert d["grid"] == (1, 3)                  # (n2, n1): first index fastest
    assert np.allclose(d["overlaps"][0], s)
    assert np.allclose(d["amplitudes"][0], amp)
    assert np.allclose(d["states"][0], [4, 5])
    assert np.allclose(d["eigenvalues"][0], [-0.02, 0.01])
