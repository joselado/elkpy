r"""A Kohn-Sham-shaped fixed point small enough to differentiate exactly — Phase 0a.

Study §6 item 0a asks for reverse-mode implicit differentiation "on a small Hamiltonian
with an engineered degenerate pair", and says explicitly what would not count: a smooth
scalar fixed point with no eigensolve.  This model has the structure that matters and
nothing else:

.. math::

    H(v;\theta) = \mathcal L\big[h_0 + \mathrm{diag}(v)\big] + \theta\,W,
    \qquad
    \rho_a(v;\theta) = \textstyle\sum_{\rm copies}[P]_{aa},
    \qquad
    F(v;\theta) = K_{\rm Hxc}\,\rho(v;\theta),

with :math:`P` the occupied projector of :math:`H` and :math:`\mathcal L` the doubling
below.  ``diag(v)`` is the local potential entering the Hamiltonian, the diagonal of the
projector is the density, and a fixed symmetric kernel maps it back — so
:math:`\partial F/\partial v = K\chi_0` is a genuine independent-particle susceptibility
and the implicit solve's matvec really does pass through ``eigh``'s derivative.

**The degeneracy is engineered by doubling, not by tuning.**  With
``degeneracy=2`` the Hamiltonian actually diagonalised is
:math:`Q\,(\mathbb 1_2\otimes h)\,Q^\dagger` for a fixed unitary :math:`Q`, so *every*
level is exactly two-fold — which is what spin degeneracy in an ``nspinor=1`` code is,
and the reason the study says "crystal symmetry makes degeneracy the normal case".
Tuning a parameter to make two levels cross would not survive the SCF moving :math:`v`;
a symmetry does.  ``rotate=False`` leaves it block-diagonal and therefore *bitwise*
degenerate.  ``rotate=True`` was meant to split it at :math:`\epsilon\|h\|` like a real
assembly, and does so at some sizes and not others: measured here, XLA's ``eigh``
returns a splitting of 1.9e-16 at :math:`m=6`, 3.9e-16 at :math:`m=12`, and **exactly
zero** at :math:`m=8`.  So the "rotated" and "bitwise" cases are not a controlled
distinction — which is the same backend-dependence Phase 0b measures directly.

**Whether the perturbation respects that symmetry decides everything.**  With
``break_symmetry=True`` the parameter couples to a general Hermitian :math:`W` on the
doubled space, so :math:`\theta` *splits* the degenerate partners — which is what a
displacement or a strain lowering a crystal symmetry does, and therefore the case that
matters for forces and phonons.  With ``W`` of the symmetric form instead, the
perturbation cannot couple the partners and the naive route survives by accident:
measured, its error is :math:`1.5\times10^{-14}` symmetric and :math:`4.7\times10^{-1}`
symmetry-broken, on the same Hamiltonian.

**The reference uses no autodiff.**  Every derivative below is computed in NumPy from
the closed form of :mod:`elkjax.reference` — which is exact at a multiplet, unlike
``eigh``'s own rule — and a dense LU solve of :math:`(\mathbb 1-K\chi_0)`, against the
AD's GMRES on the transposed operator.  Central finite differences of the converged
fixed point are reported alongside as a third opinion.
"""

import dataclasses

import numpy as np

import jax
import jax.numpy as jnp

from . import projector as pj, reference as ref

__all__ = ["ScfToy", "build"]


@dataclasses.dataclass(frozen=True)
class ScfToy:
    h0: np.ndarray            # (m, m) Hermitian: kinetic + external
    w: np.ndarray             # (m, m) Hermitian: what theta couples to
    kernel: np.ndarray        # (m, m) real symmetric: K_Hxc
    observable: np.ndarray    # (m, m) Hermitian: M in the loss Tr[P M]
    nocc: int
    degeneracy: int = 1
    rotation: np.ndarray = None   # (gm, gm) unitary, used when degeneracy > 1
    tol: float = 1e-11
    smearing: tuple = None        # (mu, width) for Fermi-Dirac, else a hard window
    rule: str = "safe"            # "naive" is the control; "sign" the eigensolver-free route
    sign_steps: int = 30          # Newton-Schulz iterations when rule == "sign"
    w_full: np.ndarray = None     # a SYMMETRY-BREAKING perturbation, on the doubled space
    m_full: np.ndarray = None     # ... and a symmetry-breaking observable

    # ---------------------------------------------------------------- forward

    @property
    def size(self):
        return self.h0.shape[0]

    def block(self, v):
        """The :math:`m\\times m` Kohn-Sham block, without the parameter."""
        return jnp.asarray(self.h0) + jnp.diag(v)

    def perturbation(self):
        r"""What :math:`\theta` couples to, on the space actually diagonalised.

        ``w_full`` is the interesting case: a perturbation that does NOT commute with
        the doubling symmetry, so it couples the degenerate partners.  Physically that
        is the ordinary situation — a displacement or a strain that lowers the crystal
        symmetry — and it is where a symmetry-protected degeneracy stops being benign.
        """
        if self.w_full is not None:
            return jnp.asarray(self.w_full)
        return self._lift(self.w)

    def observable_full(self):
        if self.m_full is not None:
            return jnp.asarray(self.m_full)
        return self._lift(self.observable)

    def hamiltonian(self, v, theta):
        """What is actually diagonalised: the block, doubled, rotated, then perturbed."""
        return self._lift(self.block(v)) + theta * self.perturbation()

    def _lift(self, m):
        if self.degeneracy == 1:
            return jnp.asarray(m)
        big = jnp.kron(jnp.eye(self.degeneracy, dtype=jnp.asarray(m).dtype), jnp.asarray(m))
        if self.rotation is None:
            return big
        q = jnp.asarray(self.rotation)
        return q @ big @ q.conj().T

    def projector(self, v, theta):
        h = self.hamiltonian(v, theta)
        if self.smearing is None:
            if self.rule == "naive":
                return pj.naive_hard_window_projector(h, self.degeneracy * self.nocc)
            if self.rule == "sign":
                return pj.sign_projector(h, self.degeneracy * self.nocc,
                                         steps=self.sign_steps)
            return pj.hard_window_projector(h, self.degeneracy * self.nocc, self.tol)
        mu, width = self.smearing
        if self.rule == "naive":
            return pj.naive_smeared_projector(h, mu, width)
        return pj.smeared_projector(h, mu, width, self.tol)

    def density(self, v, theta):
        r""":math:`\rho_a`, summed over the degenerate copies — a real vector of length m."""
        p = self.projector(v, theta)
        if self.degeneracy == 1:
            return jnp.real(jnp.diag(p))
        if self.rotation is not None:
            q = jnp.asarray(self.rotation)
            p = q.conj().T @ p @ q
        diagonal = jnp.real(jnp.diag(p))
        return sum(diagonal[k * self.size:(k + 1) * self.size]
                   for k in range(self.degeneracy))

    def step(self, v, theta):
        r""":math:`F(v;\theta)=K_{\rm Hxc}\,\rho`.  The mixer is NOT part of this."""
        return jnp.asarray(self.kernel) @ self.density(v, theta)

    def loss(self, v, theta):
        r""":math:`\mathrm{Tr}[PM]`, per degenerate copy.  Gauge-invariant by construction."""
        p = self.projector(v, theta)
        return jnp.real(jnp.trace(p @ self.observable_full())) / self.degeneracy

    def band_energy(self, v, theta):
        r""":math:`\sum_{i<n_{\rm occ}}\varepsilon_i` per copy — study §8(b)'s safe quantity."""
        evals = jnp.linalg.eigvalsh(self.hamiltonian(v, theta))
        return jnp.sum(evals[:self.degeneracy * self.nocc]) / self.degeneracy

    # -------------------------------------------------------------- reference

    def _lift_np(self, m):
        m = np.asarray(m)
        if self.degeneracy == 1:
            return m
        big = np.kron(np.eye(self.degeneracy, dtype=m.dtype), m)
        if self.rotation is None:
            return big
        q = np.asarray(self.rotation)
        return q @ big @ q.conj().T

    def _perturbation_np(self):
        return np.asarray(self.w_full) if self.w_full is not None else self._lift_np(self.w)

    def _observable_np(self):
        return np.asarray(self.m_full) if self.m_full is not None else self._lift_np(self.observable)

    def _full_np(self, v, theta):
        return (self._lift_np(np.asarray(self.h0) + np.diag(np.asarray(v)))
                + theta * self._perturbation_np())

    def _occupations(self, evals, size):
        if self.smearing is None:
            return ref.hard_occupations(size, self.degeneracy * self.nocc), None
        return ref.fermi_dirac(evals, *self.smearing)

    def _dprojector(self, h, dh):
        """The closed-form :math:`dP` — no autodiff, and safe at an exact multiplet."""
        evals, evecs = np.linalg.eigh(h)
        occ, docc = self._occupations(evals, h.shape[0])
        kernel = ref.divided_difference_kernel(evals, occ, docc, self.tol)
        a = evecs.conj().T @ dh @ evecs
        return evecs @ (kernel * a) @ evecs.conj().T

    def _fold(self, matrix):
        r"""Contract a full-space operator down to the m-vector :math:`\rho_a`."""
        if self.degeneracy > 1 and self.rotation is not None:
            q = np.asarray(self.rotation)
            matrix = q.conj().T @ matrix @ q
        diagonal = np.real(np.diag(matrix))
        return sum(diagonal[k * self.size:(k + 1) * self.size]
                   for k in range(self.degeneracy))

    def _dv_directions(self):
        """:math:`\\partial H/\\partial v_b` on the space actually diagonalised."""
        for b in range(self.size):
            direction = np.zeros((self.size, self.size), dtype=complex)
            direction[b, b] = 1.0
            yield self._lift_np(direction)

    def chi0(self, v, theta):
        r""":math:`\chi_{0,ab}=\partial\rho_a/\partial v_b`, from the closed form.

        Exactly the operator whose transpose GMRES is asked to invert, built with no
        autodiff and with a different linear-algebra path.
        """
        h = self._full_np(v, theta)
        out = np.zeros((self.size, self.size))
        for b, direction in enumerate(self._dv_directions()):
            out[:, b] = self._fold(self._dprojector(h, direction))
        return out

    def reference_gradient(self, v, theta, quantity="loss"):
        r""":math:`dL/d\theta` at the fixed point, by a dense implicit solve.

        :math:`dv^*/d\theta=(\mathbb 1-K\chi_0)^{-1}K\,\partial\rho/\partial\theta`,
        then the chain rule with the loss's explicit :math:`\theta` dependence.
        """
        v = np.asarray(v, dtype=float)
        h = self._full_np(v, theta)
        w = self._perturbation_np()
        kernel = np.asarray(self.kernel)
        drho_dtheta = self._fold(self._dprojector(h, w))
        jacobian = np.eye(self.size) - kernel @ self.chi0(v, theta)
        dv = np.linalg.solve(jacobian, kernel @ drho_dtheta)

        if quantity == "band_energy":
            evals, evecs = np.linalg.eigh(h)
            occ, _ = self._occupations(evals, h.shape[0])
            p = (evecs * occ) @ evecs.conj().T
            partial_theta = float(np.real(np.trace(p @ w))) / self.degeneracy
            partial_v = np.array([float(np.real(np.trace(p @ d))) / self.degeneracy
                                  for d in self._dv_directions()])
        else:
            m = self._observable_np()
            partial_theta = float(np.real(np.trace(self._dprojector(h, w) @ m))) / self.degeneracy
            partial_v = np.array([float(np.real(np.trace(self._dprojector(h, d) @ m)))
                                  / self.degeneracy for d in self._dv_directions()])
        return partial_theta + float(partial_v @ dv)

    def spectral_radius(self, v, theta):
        """:math:`\\rho(K\\chi_0)` — how hard the fixed point is, and whether it converges."""
        eigenvalues = np.linalg.eigvals(np.asarray(self.kernel) @ self.chi0(v, theta))
        return float(np.max(np.abs(eigenvalues)))


def build(size=8, nocc=3, seed=0, degeneracy=1, rotate=True, coupling=0.25,
          smearing=None, gap=1.0, tol=1e-11, rule="safe", break_symmetry=False,
          sign_steps=30):
    """A reproducible instance with a gapped window and a convergent fixed point.

    ``gap`` is pushed into the *unperturbed* block so the occupied window has somewhere
    to be; ``coupling`` scales the kernel, which sets the spectral radius of
    :math:`K\\chi_0` and therefore how far the answer is from the non-self-consistent
    one.  A radius near zero would make the implicit solve trivially right.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.sort(rng.normal(scale=1.0, size=size))
    spectrum[nocc:] += gap
    h0 = ref.hermitian_from_spectrum(spectrum, seed)
    w = ref.random_hermitian_direction(size, seed + 1)
    m = ref.random_hermitian_direction(size, seed + 2)
    a = rng.normal(size=(size, size))
    kernel = coupling * (a + a.T) / (2 * np.sqrt(size))
    rotation = None
    if degeneracy > 1 and rotate:
        big = degeneracy * size
        b = rng.normal(size=(big, big)) + 1j * rng.normal(size=(big, big))
        q, r = np.linalg.qr(b)
        rotation = q * (np.diag(r) / np.abs(np.diag(r)))
    w_full = m_full = None
    if break_symmetry:
        if degeneracy == 1:
            raise ValueError("break_symmetry needs a degeneracy to break")
        big = degeneracy * size
        w_full = ref.random_hermitian_direction(big, seed + 11) * np.sqrt(big)
        m_full = ref.random_hermitian_direction(big, seed + 12) * np.sqrt(big)
    return ScfToy(h0=h0, w=w, kernel=kernel, observable=m, nocc=nocc,
                  degeneracy=degeneracy, rotation=rotation, tol=tol,
                  smearing=smearing, rule=rule, w_full=w_full, m_full=m_full,
                  sign_steps=sign_steps)
