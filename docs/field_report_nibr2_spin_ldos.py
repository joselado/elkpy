"""Charge and spin LDOS at the tip plane, from one task-9005 export.

Contributed alongside docs/field_report_nibr2.md (item 3 of that report). NOT
wired into the package: this is the reference implementation the report refers
to, kept here so it survives the session that wrote it.

Why it is worth having
----------------------
`parsers.transport` already notes that the export does not depend on the
energy, only the per-state weight does. It carries one more thing the
transmission path never uses: the tip amplitudes are SPINOR-RESOLVED,
`(nspinor, nsel, npoints)`. That is enough to build

    n(r,E)   = sum_kn w_k |a_n(E)|^2  a^dag a          conventional STM
    m_i(r,E) = sum_kn w_k |a_n(E)|^2  a^dag sigma_i a  spin-polarised STM

for EVERY energy from a single diagonalisation. The task-9003 route
(`Calculation.get_spin_stm`) needs one full Elk run per bias and per tip
direction, and each of those re-reads the eigenvector files, which on a
45-atom spiral supercell is 7.3 GB of Lustre traffic per bias. Getting a whole
dI/dV(x,E) map this way instead took one 20-minute run rather than dozens.

Note the tip direction is applied HERE, in Python, not in Fortran: all three
components of m come out of the same export, so rotating the tip (and choosing
P_T) is a recombination, never a re-run. Task 9003 has to be told
`elkpy_stmdir` up front and returns only the projection.

Validation
----------
Against Elk's own task 9003 (`rhomagv`, an entirely separate Fortran path)
on a converged NiBr2 monolayer spiral: 15 Ni + 30 Br, 15x1 supercell carrying
one full turn of an in-plane helix, PBE, spinpol + spinorb, 6x18x1 mesh. At
bias -0.080 Ha, comparing relative harmonic amplitudes A_m = 2|c_m|/nbar of the
a2-averaged profile along the spiral:

    harmonic        task 9003        this route
    A_2q (charge)   1.404e-03        1.401e-03
    A_4q (charge)   2.964e-05        2.939e-05
    A_15q (charge)  2.748e-01        2.748e-01
    A_q  (spin x)   2.376e-01        2.381e-01

The charge channel is additionally exact against elkpy's own arithmetic:
`compute_transmission(..., exit_region="cell")` replaces every S_k by the
identity, which is the same quantity, and the two agree to 3.5e-16. That
identity is a genuinely load-bearing check, not decoration: it is what caught
the absolute-versus-relative energy bug reported as item 1.

Two traps this code exists to encode
------------------------------------
1. `energies` here are ABSOLUTE (Hartree), matching `compute_transmission` and
   NOT `Calculation.get_vertical_transport`, whose `energies` are relative to
   E_F. Passing a bias where an absolute energy belongs fails SILENTLY, because
   the smeared delta still has exponential tails and returns small non-zero
   numbers. Hence the explicit window check below, which is the fix suggested
   as item 1 of the report.
2. `occmax` is 1 for nspinor = 2, not the 2.0 that `amplitude_weights`
   defaults to (item 2). It is derived from the data here, never defaulted.
"""

import numpy as np

from elkpy.parsers import transport as _tr


def spin_ldos(data, energies, broadening=0.005, stype=3, check_window=True):
    """Charge and vector spin LDOS at the tip points, for each energy.

    `data` is `parsers.transport.parse_transport`'s output. `energies` are
    ABSOLUTE Hartree (add `data["efermi"]` to a bias yourself; see the module
    docstring). Returns a dict:

      "charge" (nE, npoints)     n, the spin-summed vacuum LDOS
      "spin"   (3, nE, npoints)  m_x, m_y, m_z in the Cartesian frame
      "energies", "grid" (n2, n1), "efermi"

    Units are elkpy's on-shell amplitude convention, i.e. the true LDOS times
    1/eta, exactly as the transmission is. Every physical statement here is a
    ratio, so the common factor never appears; do not read the absolute value
    as states/Hartree/Bohr^3.

    A tip of polarisation P_T along a unit vector e_T then measures
    n + P_T (m . e_T), which is a recombination of these arrays.
    """
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    npoints = data["npoints"]
    efermi = data["efermi"]
    occmax = 2.0 if data["nspinor"] == 1 else 1.0

    if check_window:
        lo, hi = (efermi + w for w in data["window"])
        pad = 3.0 * broadening
        if energies.min() < lo + pad or energies.max() > hi - pad:
            raise ValueError(
                "requested energies [%.4f, %.4f] Ha reach outside the exported "
                "window [%.4f, %.4f] Ha (efermi %.4f). These are ABSOLUTE "
                "energies: if you meant a bias relative to E_F, add "
                "data['efermi'] first. Passing a bias here does not raise on "
                "its own, it just samples the delta's tails and returns small "
                "wrong numbers. Pass check_window=False to override."
                % (energies.min(), energies.max(), lo, hi, efermi)
            )

    charge = np.zeros((energies.size, npoints))
    spin = np.zeros((3, energies.size, npoints))
    for amp, eig, wk in zip(data["amplitudes"], data["eigenvalues"], data["weights"]):
        if eig.size == 0:
            continue
        if amp.shape[0] != 2:
            raise ValueError(
                "spin_ldos needs nspinor = 2 (spinpol and/or spinorb); the "
                "export has nspinor = %d, for which the spin density is not "
                "defined" % amp.shape[0]
            )
        for ie, energy in enumerate(energies):
            w = _tr.amplitude_weights(eig, energy, broadening, stype, occmax)
            a = amp * w[None, :, None]
            up, dn = a[0], a[1]
            nu = np.real(np.conj(up) * up).sum(axis=0)
            nd = np.real(np.conj(dn) * dn).sum(axis=0)
            cross = (np.conj(up) * dn).sum(axis=0)
            charge[ie] += wk * (nu + nd)
            spin[0, ie] += wk * 2.0 * np.real(cross)
            spin[1, ie] += wk * 2.0 * np.imag(cross)
            spin[2, ie] += wk * (nu - nd)

    return {
        "charge": charge,
        "spin": spin,
        "energies": energies,
        "grid": data["grid"],
        "efermi": efermi,
    }


def tip_image(result, direction=(0, 0, 1), polarization=1.0):
    """n + P_T (m . e_T) on the tip plane: the Tersoff-Hamann differential
    conductance a magnetic tip measures, the same combination task 9003 writes
    as the third column of ELKPY_STM2D.OUT."""
    e = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(e)
    if norm < 1e-12:
        raise ValueError("zero-length tip direction")
    e = e / norm
    return result["charge"] + polarization * np.einsum(
        "i,iep->ep", e, result["spin"], optimize=True
    )


def _selftest():
    """Synthetic check of the spinor algebra alone: no Elk run needed.

    One k-point, one state, one point, unit weight and a delta placed exactly
    on the eigenvalue, so the weight is a known scalar and the spinor algebra
    is the only thing under test. Four spinors with analytically known
    polarisations.
    """
    cases = [
        ((1.0 + 0j, 0.0 + 0j), (0.0, 0.0, +1.0)),
        ((0.0 + 0j, 1.0 + 0j), (0.0, 0.0, -1.0)),
        ((1 / np.sqrt(2) + 0j, 1 / np.sqrt(2) + 0j), (+1.0, 0.0, 0.0)),
        ((1 / np.sqrt(2) + 0j, 1j / np.sqrt(2)), (0.0, +1.0, 0.0)),
    ]
    eps = 0.3
    for (up, dn), want in cases:
        data = {
            "npoints": 1, "nspinor": 2, "grid": (1, 1), "efermi": 0.0,
            "window": (-1.0, 1.0),
            "amplitudes": [np.array([[[up]], [[dn]]], dtype=complex)],
            "eigenvalues": [np.array([eps])],
            "weights": np.array([1.0]),
        }
        r = spin_ldos(data, [eps], broadening=0.01)
        n = r["charge"][0, 0]
        m = r["spin"][:, 0, 0] / n
        assert np.allclose(m, want, atol=1e-12), (m, want)
        assert np.isclose(n, _tr.amplitude_weights(np.array([eps]), eps, 0.01,
                                                   3, 1.0)[0] ** 2)
    # the tip projection must reproduce the spin-summed image for P_T = 0
    data["amplitudes"] = [np.array([[[0.6 + 0j]], [[0.8 + 0j]]], dtype=complex)]
    r = spin_ldos(data, [eps], broadening=0.01)
    assert np.allclose(tip_image(r, (0, 0, 1), 0.0), r["charge"])
    # and n +- m_z must be the two spin-resolved channels
    assert np.isclose(tip_image(r, (0, 0, 1), 1.0)[0, 0],
                      r["charge"][0, 0] + r["spin"][2, 0, 0])
    # The window guard must fire on the real failure mode: the NiBr2 run's own
    # numbers, where a bias of +0.10 Ha relative to E_F = -0.162 is absolute
    # -0.062 and legal, while passing that bias straight through as if it were
    # absolute lands at +0.10, outside the exported [-0.322, +0.058].
    data["efermi"] = -0.162176
    data["window"] = (-0.16, 0.22)
    lo, hi = (data["efermi"] + w for w in data["window"])
    assert lo < data["efermi"] + 0.10 < hi, "the correct absolute energy is in range"
    spin_ldos(data, [data["efermi"] + 0.10], broadening=0.005)   # must not raise
    try:
        spin_ldos(data, [0.10], broadening=0.005)                # the bug
    except ValueError as exc:
        assert "ABSOLUTE" in str(exc)
    else:
        raise AssertionError("window guard did not fire")
    print("spin_ldos self-test passed")


if __name__ == "__main__":
    _selftest()
