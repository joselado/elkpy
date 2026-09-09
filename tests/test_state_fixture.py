"""tests/fixtures/h_sc: the committed Elk run that pins the STATE.OUT layout.

Nothing in elkpy reads STATE.OUT -- Elk's own `readstate` does. These tests
exist so that the conventions written down in docs/design.md #34 are checked
against a real file rather than remembered, since a sibling project reads the
format directly and every one of them has a plausible wrong answer.

No elk binary needed: the fixture is committed.  Regenerate it with
tests/fixtures/h_sc/regenerate.sh.
"""

import struct

import numpy as np
import pytest

from elkpy import config, spec
from elkpy.parsers import geometry, info, volumetric

FIXTURE = config.repo_root() / "tests" / "fixtures" / "h_sc"

RMT = 1.400000000        # INFO.OUT, after autormt
NRMT, NRMTI = 197, 129
EFERMI = 0.73970527984559e-01


def read_records(path):
    """Yield the payload of each gfortran sequential unformatted record: a
    4-byte length, the bytes, then the same length again."""
    with open(path, "rb") as fh:
        while True:
            head = fh.read(4)
            if len(head) < 4:
                return
            (length,) = struct.unpack("<i", head)
            payload = fh.read(length)
            (tail,) = struct.unpack("<i", fh.read(4))
            assert tail == length, "record markers disagree -- not this layout"
            yield payload


def test_fixture_converged():
    assert info.parse_convergence(FIXTURE / "INFO.OUT") is True


def test_charges():
    charges = info.parse_charges(FIXTURE / "INFO.OUT")
    assert charges["core"] == 0.0          # hydrogen: no core, AE == valence
    assert charges["valence"] == pytest.approx(1.0, abs=1e-9)
    assert charges["muffin_tin"] == [0.6125761996]
    assert charges["core_leakage"] == [0.0]
    assert charges["species"] == ["H"]
    assert charges["interstitial"] == 0.3874238004
    assert charges["muffin_tin_total"] == charges["muffin_tin"][0]
    assert "excess" not in charges          # chgexs == 0, so writechg omits it

    # chgmt and chgir are POST-rhonorm and chgcalc is not (src/rhonorm.f90
    # updates chgmt/chgmttot, then sets chgir = chgtot - chgmttot, and never
    # touches chgcalc).  So the printed "error" is what rhonorm CORRECTED,
    # not a floor under a reintegration check -- and the corrected charges
    # close on the total exactly.
    assert charges["total"] == 1.0
    assert charges["muffin_tin_total"] + charges["interstitial"] == 1.0
    assert abs(charges["total_calculated"] - 1.0) == pytest.approx(charges["error"])
    assert 1e-4 < charges["error"] < 1e-2


def test_charges_index_selects_the_scf_iteration():
    """writechg runs once per SCF loop, so INFO.OUT holds one block each and
    the default must be the converged one."""
    first = info.parse_charges(FIXTURE / "INFO.OUT", index=0)
    last = info.parse_charges(FIXTURE / "INFO.OUT")
    assert first["interstitial"] != last["interstitial"]
    assert abs(first["interstitial"] - last["interstitial"]) < 1e-3


def test_state_out_header_order():
    """The header records of writestate.f90, in order, with the radial mesh
    read back.  This is what a reader has to get right before it reaches a
    single density value."""
    records = read_records(FIXTURE / "STATE.OUT")

    assert struct.unpack("<3i", next(records)) == spec.ELK_VERSION
    (spinpol,) = struct.unpack("<i", next(records))     # 4-byte logical
    assert spinpol == 0
    (nspecies,) = struct.unpack("<i", next(records))
    (lmmaxo,) = struct.unpack("<i", next(records))
    (nrmtmax,) = struct.unpack("<i", next(records))
    (nrcmtmax,) = struct.unpack("<i", next(records))
    assert nspecies == 1
    assert lmmaxo == 49                                  # default lmaxo = 6
    assert nrmtmax == NRMT

    (natoms,) = struct.unpack("<i", next(records))
    (nrmt,) = struct.unpack("<i", next(records))
    rsp = np.frombuffer(next(records), dtype=np.float64)
    (nrcmt,) = struct.unpack("<i", next(records))
    rcmt = np.frombuffer(next(records), dtype=np.float64)
    assert natoms == 1
    assert nrmt == NRMT and len(rsp) == NRMT
    assert len(rcmt) == nrcmt <= nrcmtmax

    # rmt is the LAST point of the radial mesh (genrmesh.f90:56-59 builds the
    # mesh so that r(nrmt) = rmt, up to the round-off of the exp/log round
    # trip), and it is the post-autormt value the species file does not carry.
    assert rsp[-1] == pytest.approx(RMT, rel=1e-14)
    assert rcmt[-1] == pytest.approx(RMT, rel=1e-14)

    # ... and the mesh really is logarithmic, which is what poly4 interpolates on
    ratios = rsp[1:] / rsp[:-1]
    assert np.allclose(ratios, ratios[0])

    ngridg = struct.unpack("<3i", next(records))
    (ngvec,) = struct.unpack("<i", next(records))
    (ndmag,) = struct.unpack("<i", next(records))
    assert ndmag == 0                        # unpolarised: no magnetic records
    # the interstitial Fourier sum runs over ngvec, NOT over the whole FFT box
    assert ngvec < np.prod(ngridg)

    for _ in range(6):                       # nspinor fsmtype ftmtype dftu
        next(records)                        # lmmaxdm xcgrad
    (efermi,) = struct.unpack("<d", next(records))
    assert efermi == pytest.approx(EFERMI)   # no EFERMI.OUT needed
    (dlefe,) = struct.unpack("<d", next(records))

    # The density record holds BOTH rhomt and rhoir, concatenated.  A reader
    # that expects one array per record desyncs exactly here.
    density = next(records)
    ngtot = int(np.prod(ngridg))
    assert len(density) == (lmmaxo * nrmtmax * natoms + ngtot) * 8

    rhomt = np.frombuffer(density[: lmmaxo * nrmtmax * natoms * 8],
                          dtype=np.float64).reshape(nrmtmax, lmmaxo)
    rhoir = np.frombuffer(density[lmmaxo * nrmtmax * natoms * 8:], dtype=np.float64)

    # rfmtpack(.false.) zeroes lm > lmmaxi on the inner part, so nrmti is
    # recoverable from the file even though it is never written to it.
    nonzero = np.abs(rhomt).max(axis=1) > 0
    lmmaxi = 4                                            # default lmaxi = 1
    assert np.all(rhomt[:NRMTI, lmmaxi:] == 0.0)
    assert np.any(rhomt[NRMTI:, lmmaxi:] != 0.0)
    assert nonzero.all()

    # rhomt stores rho_lm(r), NOT r^2 rho_lm(r).  At the innermost mesh point
    # r = 2.8e-5 Bohr, so the two differ by ten orders of magnitude and there
    # is nothing to argue about: the density at the nucleus is O(0.1), while
    # r^2 rho would be O(1e-13).
    y00 = 1.0 / np.sqrt(4.0 * np.pi)
    rho_at_nucleus = rhomt[0, 0] * y00
    assert 0.1 < rho_at_nucleus < 1.0
    assert rsp[0] < 1e-4

    # ... and it is the same number Elk's own reconstruction prints at the
    # origin.  rfpts clamps r up to rsp(1) at the nucleus, so RHO3D.OUT's
    # first value IS rhomt(lm=0, ir=1) * y00 -- which closes the whole chain
    # at once: the record layout, the lm-fastest ordering of the reshape, the
    # y00 factor, and rho-not-r^2-rho.
    _, reference = volumetric.parse_plot3d(FIXTURE / "RHO3D.OUT")
    assert rho_at_nucleus == pytest.approx(reference[0], rel=1e-9)

    # rhoir is raw -- extended over the whole cell, never multiplied by the
    # characteristic function, so it is finite and smooth inside the sphere too.
    assert len(rhoir) == ngtot
    assert np.isfinite(rhoir).all()


def test_rho3d_is_a_pointwise_reference():
    """RHO3D.OUT is Elk's own muffin-tin + interstitial reconstruction, with
    the Cartesian coordinate beside every value -- the ground truth a reader
    is checked against pointwise rather than only through an integral."""
    points, density = volumetric.parse_plot3d(FIXTURE / "RHO3D.OUT")
    assert points.shape == (16 ** 3, 3)
    assert density.shape == (16 ** 3,)
    assert (density > 0).all()

    # the first point is the origin, where the atom sits.  The density there
    # is finite, not divergent -- the 1s wavefunction has a cusp, not a pole --
    # and it lands just under the free-atom 1/pi, the cell being compressed.
    assert np.allclose(points[0], 0.0)
    assert density[0] == density.max()
    assert 0.85 < density[0] / (1.0 / np.pi) < 1.0
    assert density[0] > 5 * np.median(density)


def test_geometry_round_trip():
    """The lattice vectors and positions STATE.OUT does not carry come from
    GEOMETRY.OUT beside it -- written after any `tshift`, which is why the
    fixture pins tshift = .false. rather than relying on it being a no-op."""
    avec, species = geometry.parse_last_geometry(FIXTURE / "GEOMETRY.OUT")
    assert np.allclose(avec, 3.0 * np.eye(3))
    assert list(species) == ["H"]
    (position, bfcmt), = species["H"]
    assert position == (0.0, 0.0, 0.0)
    assert bfcmt == (0.0, 0.0, 0.0)


def test_muffin_tin_charge_is_the_clean_integrated_check():
    """chgmt is a pure radial integral inside the sphere, and rhonorm updates
    it, so it matches the rhomt that STATE.OUT actually holds.  Reintegrating
    the l=0 channel here recovers it to quadrature accuracy -- this is the
    integrated check a density reader can trust.

    Elk integrates with cubic-spline weights (wsplint); Simpson on the
    log mesh is a different quadrature, which is the whole 9e-6 residual.
    """
    records = list(read_records(FIXTURE / "STATE.OUT"))
    # header record order, writestate.f90: version spinpol nspecies lmmaxo
    # nrmtmax nrcmtmax | natoms nrmt rsp nrcmt rcmt | ngridg ngvec ndmag
    # nspinor fsmtype ftmtype dftu lmmaxdm xcgrad efermi dlefe, then rho.
    rsp = np.frombuffer(records[8], dtype=np.float64)
    lmmaxo = 49
    rhomt = np.frombuffer(records[22][: lmmaxo * NRMT * 8],
                          dtype=np.float64).reshape(NRMT, lmmaxo)

    # chgmt = 4 pi y00 \int rho_00(r) r^2 dr, on a mesh uniform in ln r
    y00 = 1.0 / np.sqrt(4.0 * np.pi)
    integrand = rhomt[:, 0] * y00 * rsp ** 3        # r^2 dr = r^3 d(ln r)
    x = np.log(rsp)
    weights = np.zeros(NRMT)                        # composite Simpson, odd n
    weights[0:-1:2] += 1.0
    weights[1::2] += 4.0
    weights[2::2] += 1.0
    charge = 4.0 * np.pi * np.sum(weights * integrand) * (x[1] - x[0]) / 3.0

    printed = info.parse_charges(FIXTURE / "INFO.OUT")["muffin_tin"][0]
    assert charge == pytest.approx(printed, abs=2e-5)
