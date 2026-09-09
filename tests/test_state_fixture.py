"""The committed Elk runs that pin the STATE.OUT layout.

Nothing in elkpy reads STATE.OUT -- Elk's own `readstate` does. These tests
exist so that the conventions written down in docs/design.md #34 are checked
against a real file rather than remembered, since a sibling project reads the
format directly and every one of them has a plausible wrong answer.

Three fixtures, each closing something the others cannot:

- `h_sc`      simple cubic hydrogen, one atom.  No core at all, so the
              all-electron and valence densities are the same function.
- `c_diamond` diamond, two carbon atoms.  A frozen 1s core inside rhomt, and
              the atom-inner half of the ias ordering.
- `sic_zb`    3C-SiC, one carbon and one silicon.  Two species, so
              natmtot != natoms(1) and nrmt(1) != nrmtmax -- the two traps
              that no one-species fixture can fail.

`c_diamond` and `sic_zb` also carry a STATE_INIT.OUT: the same run stopped at
the top of Elk's first self-consistent iteration (task 9006, patches/0024).

No elk binary needed: the fixtures are committed.  Regenerate each with its
own regenerate.sh.
"""

import struct

import numpy as np
import pytest

from elkpy import config, inputfile, spec
from elkpy.parsers import geometry, info, volumetric

FIXTURES = config.repo_root() / "tests" / "fixtures"
H_SC = FIXTURES / "h_sc"
DIAMOND = FIXTURES / "c_diamond"
SIC = FIXTURES / "sic_zb"

ALL_FIXTURES = [H_SC, DIAMOND, SIC]
TWO_STATE_FIXTURES = [DIAMOND, SIC]

RMT = 1.400000000        # h_sc INFO.OUT, after autormt
NRMT, NRMTI = 197, 129
EFERMI = 0.73970527984559e-01

Y00 = 1.0 / np.sqrt(4.0 * np.pi)


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


def read_state(path):
    """writestate.f90's header plus the first (density) record, for ANY number
    of species.

    Everything here that is per-species is a LIST, and the muffin-tin block is
    indexed by ias -- species outer, atom inner (init0.f90's idxas loop). The
    two things a one-species file cannot teach are both in this function:
    natmtot is the sum over species, and rows past nrmt(is) of the ias block
    are uninitialised buffer, so `muffin_tin()` below slices them off.
    """
    r = read_records(path)
    state = {"version": struct.unpack("<3i", next(r))}
    for key in ("spinpol", "nspecies", "lmmaxo", "nrmtmax", "nrcmtmax"):
        (state[key],) = struct.unpack("<i", next(r))
    state.update(natoms=[], nrmt=[], rsp=[], nrcmt=[], rcmt=[])
    for _ in range(state["nspecies"]):
        (natoms,) = struct.unpack("<i", next(r))
        (nrmt,) = struct.unpack("<i", next(r))
        rsp = np.frombuffer(next(r), dtype=np.float64)
        (nrcmt,) = struct.unpack("<i", next(r))
        rcmt = np.frombuffer(next(r), dtype=np.float64)
        state["natoms"].append(natoms)
        state["nrmt"].append(nrmt)
        state["rsp"].append(rsp)
        state["nrcmt"].append(nrcmt)
        state["rcmt"].append(rcmt)
    state["ngridg"] = struct.unpack("<3i", next(r))
    for key in ("ngvec", "ndmag", "nspinor", "fsmtype", "ftmtype", "dftu",
                "lmmaxdm", "xcgrad"):
        (state[key],) = struct.unpack("<i", next(r))
    (state["efermi"],) = struct.unpack("<d", next(r))
    (state["dlefe"],) = struct.unpack("<d", next(r))

    natmtot = sum(state["natoms"])                 # NOT natoms[0]
    ngtot = int(np.prod(state["ngridg"]))
    state["natmtot"] = natmtot
    state["ngtot"] = ngtot
    # ias -> species index, the same order writestate.f90 packs rfmt in
    state["idxis"] = [i for i, n in enumerate(state["natoms"]) for _ in range(n)]

    density = next(r)
    nmt = state["lmmaxo"] * state["nrmtmax"] * natmtot
    state["density_record"] = density
    state["rhomt"] = np.frombuffer(density[: nmt * 8], dtype=np.float64).reshape(
        natmtot, state["nrmtmax"], state["lmmaxo"])
    state["rhoir"] = np.frombuffer(density[nmt * 8:], dtype=np.float64)
    return state


def muffin_tin(state, ias):
    """rho_lm(r) inside sphere `ias`, with the uninitialised padding rows
    dropped, and the radial mesh that goes with it."""
    nrmt = state["nrmt"][state["idxis"][ias]]
    return state["rsp"][state["idxis"][ias]], state["rhomt"][ias, :nrmt, :]


def reintegrate_chgmt(rsp, rhomt):
    """chgmt = 4 pi y00 \\int rho_00(r) r^2 dr, by composite Simpson on the
    mesh uniform in ln r.  Elk uses cubic-spline weights (wsplint), so the
    residual against the printed chgmt is quadrature and nothing else."""
    integrand = rhomt[:, 0] * Y00 * rsp ** 3        # r^2 dr = r^3 d(ln r)
    x = np.log(rsp)
    weights = np.zeros(len(rsp))                    # composite Simpson, odd n
    weights[0:-1:2] += 1.0
    weights[1::2] += 4.0
    weights[2::2] += 1.0
    return 4.0 * np.pi * np.sum(weights * integrand) * (x[1] - x[0]) / 3.0


# --------------------------------------------------------------------------
# what must hold on every fixture
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda p: p.name)
def test_fixture_converged(fixture):
    assert info.parse_convergence(fixture / "INFO.OUT") is True


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda p: p.name)
def test_density_record_length(fixture):
    """One Fortran record holds rhomt AND rhoir concatenated, and its length
    counts natmtot spheres -- the sum over species, not natoms(1).  On h_sc
    and c_diamond those two are equal and this assertion cannot fail; sic_zb
    is why it is written this way."""
    state = read_state(fixture / "STATE.OUT")
    expected = (state["lmmaxo"] * state["nrmtmax"] * state["natmtot"]
                + state["ngtot"]) * 8
    assert len(state["density_record"]) == expected
    assert len(state["rhoir"]) == state["ngtot"]
    assert np.isfinite(state["rhoir"]).all()
    # the interstitial Fourier sum runs over ngvec, NOT over the whole FFT box
    assert state["ngvec"] < state["ngtot"]


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda p: p.name)
def test_radial_mesh_is_logarithmic_and_ends_at_rmt(fixture):
    """genrmesh.f90:56-59 builds r_i = rmin exp[(i-1) ln(rmt/rmin)/(nr-1)], so
    rmt is the last mesh point -- and it is the value AFTER checkmt/autormt
    has moved it, which the species file does not carry."""
    state = read_state(fixture / "STATE.OUT")
    for isp in range(state["nspecies"]):
        rsp = state["rsp"][isp]
        assert len(rsp) == state["nrmt"][isp]
        ratios = rsp[1:] / rsp[:-1]
        assert np.allclose(ratios, ratios[0])
        assert state["rcmt"][isp][-1] == pytest.approx(rsp[-1], rel=1e-14)


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda p: p.name)
def test_density_at_every_nucleus_matches_rho3d(fixture):
    """rfpts clamps r up to rsp(1) at the nucleus and its poly4 window starts
    at ir=1, so RHO3D.OUT at an atomic site IS rhomt(lm=0, ir=1) * y00.  One
    number per atom, and it pins the record layout, the lm-fastest reshape,
    the y00 factor and rho-not-r^2-rho at once."""
    state = read_state(fixture / "STATE.OUT")
    points, density = volumetric.parse_plot3d(fixture / "RHO3D.OUT")
    avec, species = geometry.parse_last_geometry(fixture / "GEOMETRY.OUT")
    sites = [np.array(position) @ np.array(avec)
             for symbol in species for position, _ in species[symbol]]
    assert len(sites) == state["natmtot"]

    for ias, site in enumerate(sites):
        distance = np.linalg.norm(points - site, axis=1)
        nearest = int(np.argmin(distance))
        assert distance[nearest] < 1e-9, "the site is not on the plot3d grid"
        _, rhomt = muffin_tin(state, ias)
        assert rhomt[0, 0] * Y00 == pytest.approx(density[nearest], rel=1e-9)


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda p: p.name)
def test_reintegrated_chgmt_matches_the_printed_one(fixture):
    """chgmt is the clean integrated check: a pure radial integral inside the
    sphere with the characteristic function nowhere in it, and rhonorm updates
    it, so it describes the rhomt the file holds."""
    state = read_state(fixture / "STATE.OUT")
    printed = info.parse_charges(fixture / "INFO.OUT")["muffin_tin"]
    assert len(printed) == state["natmtot"]
    for ias in range(state["natmtot"]):
        rsp, rhomt = muffin_tin(state, ias)
        assert reintegrate_chgmt(rsp, rhomt) == pytest.approx(printed[ias], abs=1e-4)


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda p: p.name)
def test_rhonorm_closes_the_charge_sum_but_not_chgcalc(fixture):
    """src/rhonorm.f90 updates chgmt/chgmttot and sets chgir = chgtot -
    chgmttot, and never touches chgcalc.  So the printed "error" is what
    rhonorm CORRECTED, not a floor under a reintegration check."""
    charges = info.parse_charges(fixture / "INFO.OUT")
    # everything here is INFO.OUT's G18.10, so the slack is the printing
    assert (charges["muffin_tin_total"] + charges["interstitial"]
            == pytest.approx(charges["total"], abs=1e-8))
    assert charges["muffin_tin_total"] == pytest.approx(
        sum(charges["muffin_tin"]), abs=1e-8)
    assert abs(charges["total"] - charges["total_calculated"]) == pytest.approx(
        charges["error"])
    assert charges["error"] > 1e-4          # bigger than the check above's slack
    assert "excess" not in charges          # chgexs == 0, so writechg omits it


# --------------------------------------------------------------------------
# h_sc: one atom, no core
# --------------------------------------------------------------------------

def test_h_sc_charges():
    charges = info.parse_charges(H_SC / "INFO.OUT")
    assert charges["core"] == 0.0          # hydrogen: no core, AE == valence
    assert charges["valence"] == pytest.approx(1.0, abs=1e-9)
    assert charges["muffin_tin"] == [0.6125761996]
    assert charges["core_leakage"] == [0.0]
    assert charges["species"] == ["H"]
    assert charges["interstitial"] == 0.3874238004
    assert charges["total"] == 1.0
    assert 1e-4 < charges["error"] < 1e-2


def test_charges_index_selects_the_scf_iteration():
    """writechg runs once per SCF loop, so INFO.OUT holds one block each and
    the default must be the converged one."""
    first = info.parse_charges(H_SC / "INFO.OUT", index=0)
    last = info.parse_charges(H_SC / "INFO.OUT")
    assert first["interstitial"] != last["interstitial"]
    assert abs(first["interstitial"] - last["interstitial"]) < 1e-3


def test_h_sc_state_out_header_order():
    """The header records of writestate.f90, in order, read positionally.
    This is what a reader has to get right before it reaches a single density
    value, and getting the ORDER wrong is not detectable from the values."""
    records = read_records(H_SC / "STATE.OUT")

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
    assert rsp[-1] == pytest.approx(RMT, rel=1e-14)

    ngridg = struct.unpack("<3i", next(records))
    (ngvec,) = struct.unpack("<i", next(records))
    (ndmag,) = struct.unpack("<i", next(records))
    assert ndmag == 0                        # unpolarised: no magnetic records

    for _ in range(5):                       # nspinor fsmtype ftmtype dftu lmmaxdm
        next(records)
    (xcgrad,) = struct.unpack("<i", next(records))
    assert xcgrad == 0                       # h_sc is LSDA; c_diamond is not
    (efermi,) = struct.unpack("<d", next(records))
    assert efermi == pytest.approx(EFERMI)   # no EFERMI.OUT needed
    next(records)                            # dlefe

    density = next(records)
    ngtot = int(np.prod(ngridg))
    assert len(density) == (lmmaxo * nrmtmax * natoms + ngtot) * 8


def test_h_sc_inner_muffin_tin_is_packed_out():
    """rfmtpack(.false.) zeroes lm > lmmaxi on the inner part of the sphere,
    so nrmti is recoverable from the file even though it is never written to
    it -- which is why a reader needs neither nrmti nor lmmaxi."""
    state = read_state(H_SC / "STATE.OUT")
    _, rhomt = muffin_tin(state, 0)
    lmmaxi = 4                                            # default lmaxi = 1
    assert np.all(rhomt[:NRMTI, lmmaxi:] == 0.0)
    assert np.any(rhomt[NRMTI:, lmmaxi:] != 0.0)


def test_h_sc_density_is_rho_not_r2_rho():
    """At the innermost mesh point r = 2.8e-5 Bohr, so rho and r^2 rho differ
    by ten orders of magnitude and there is nothing to argue about: the
    density at the nucleus is O(0.1), while r^2 rho would be O(1e-13)."""
    state = read_state(H_SC / "STATE.OUT")
    rsp, rhomt = muffin_tin(state, 0)
    assert rsp[0] < 1e-4
    assert 0.1 < rhomt[0, 0] * Y00 < 1.0


def test_h_sc_rho3d_is_a_pointwise_reference():
    """RHO3D.OUT is Elk's own muffin-tin + interstitial reconstruction, with
    the Cartesian coordinate beside every value -- the ground truth a reader
    is checked against pointwise rather than only through an integral."""
    points, density = volumetric.parse_plot3d(H_SC / "RHO3D.OUT")
    assert points.shape == (16 ** 3, 3)
    assert (density > 0).all()
    # the density at the nucleus is finite, not divergent -- the 1s
    # wavefunction has a cusp, not a pole -- and it lands just under the
    # free-atom 1/pi, the cell being compressed at a = 3 Bohr.
    assert np.allclose(points[0], 0.0)
    assert density[0] == density.max()
    assert 0.85 < density[0] / (1.0 / np.pi) < 1.0


def test_h_sc_geometry_round_trip():
    """The lattice vectors and positions STATE.OUT does not carry come from
    GEOMETRY.OUT beside it -- written after any `tshift`, which is why every
    fixture pins tshift = .false. rather than relying on it being a no-op."""
    avec, species = geometry.parse_last_geometry(H_SC / "GEOMETRY.OUT")
    assert np.allclose(avec, 3.0 * np.eye(3))
    assert list(species) == ["H"]
    (position, bfcmt), = species["H"]
    assert position == (0.0, 0.0, 0.0)
    assert bfcmt == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# c_diamond: a frozen core inside rhomt, and two atoms of one species
# --------------------------------------------------------------------------

def test_diamond_rhomt_contains_the_frozen_core():
    """rhocore.f90 adds the core density into rhomt and STATE.OUT holds only
    the sum, so there is no valence-only density in the file.  Hydrogen's
    density at the nucleus is 0.286 e/Bohr^3; carbon's is 130, which is the 1s.
    Two of carbon's six electrons are core, and they are 45% of chgmt."""
    state = read_state(DIAMOND / "STATE.OUT")
    _, rhomt = muffin_tin(state, 0)
    assert rhomt[0, 0] * Y00 > 100.0

    charges = info.parse_charges(DIAMOND / "INFO.OUT")
    assert charges["core"] == pytest.approx(4.0)          # 1s^2 on each atom
    assert charges["valence"] == pytest.approx(8.0)
    assert 0.4 < 2.0 / charges["muffin_tin"][0] < 0.5
    # the core is inside the sphere but not entirely: chgcrlk is the part that
    # leaked out, and it is small enough that "the core is frozen" holds.
    assert 0 < charges["core_leakage"][0] < 1e-3


def test_diamond_two_atoms_are_related_by_inversion():
    """Diamond's two carbons sit at the two ends of a bond whose midpoint is
    an inversion centre, so their rho_lm differ by (-1)^l.  The site symmetry
    is Td, which kills l = 1, 2 and 5 outright.

    This is what makes an ias swap detectable on a ONE-species fixture at all:
    l = 0 is identical between the two columns, so the nucleus check cannot
    see a swap -- only the odd-l channels can.
    """
    state = read_state(DIAMOND / "STATE.OUT")
    assert state["nspecies"] == 1 and state["natoms"] == [2]
    _, first = muffin_tin(state, 0)
    _, second = muffin_tin(state, 1)

    for l in (1, 2, 5):                                     # forbidden by Td
        for column in (first, second):
            assert np.abs(column[:, l ** 2:(l + 1) ** 2]).max() < 1e-15
    for l in (0, 3, 4, 6):                                  # Td keeps these
        block = first[:, l ** 2:(l + 1) ** 2]
        other = second[:, l ** 2:(l + 1) ** 2]
        assert np.abs(block).max() > 1e-3
        sign = (-1) ** l
        assert np.abs(block - sign * other).max() < 1e-14


def test_diamond_is_pbe_and_carries_xcgrad():
    """xctype 20 sets xcgrad = 1 (modxcifc.f90:460), and xcgrad is a header
    record.  A reader that inferred 0 from h_sc breaks on a GGA state."""
    assert read_state(DIAMOND / "STATE.OUT")["xcgrad"] == 1
    assert read_state(H_SC / "STATE.OUT")["xcgrad"] == 0


def test_diamond_tshift_is_load_bearing():
    """h_sc could not show this: with one atom at the origin the inversion
    centre is already there and tshift is a no-op.  Diamond's is the bond
    midpoint, so Elk's default would move BOTH atoms -- to +-(3/8,3/8,3/8),
    not to (0,0,0) and (1/4,1/4,1/4).  GEOMETRY.OUT is written after the
    shift and elk.in's atoms block is not, so a reader that took positions
    from elk.in would put the muffin tins in the wrong place entirely."""
    blocks = dict(inputfile.read_blocks(DIAMOND / "elk.in"))
    assert blocks["tshift"] == [[".false."]]
    _, species = geometry.parse_last_geometry(DIAMOND / "GEOMETRY.OUT")
    positions = [position for symbol in species for position, _ in species[symbol]]
    assert positions == [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]


# --------------------------------------------------------------------------
# sic_zb: two species -- the traps no one-species fixture can fail
# --------------------------------------------------------------------------

def test_sic_natmtot_is_not_natoms_of_the_first_species():
    state = read_state(SIC / "STATE.OUT")
    assert state["nspecies"] == 2
    assert state["natoms"] == [1, 1]
    assert state["natmtot"] == 2 != state["natoms"][0]
    assert state["idxis"] == [0, 1]


def test_sic_padding_rows_are_uninitialised_not_zero():
    """rfmt is dimensioned to nrmtmax and rfmtpack writes only rows 1..nrmt(is),
    so for the shorter species the rest of the block is leftover buffer.  Elk's
    species files set nrmt by periodic-table row (300 for C, 400 for Si, less
    the lradstp rounding), and checkmt moves rmt but never nrmt -- which is why
    this needs two species from two DIFFERENT rows to show up at all.

    The leftover values here are O(1e-3): sixteen orders of magnitude above
    the O(1e-19) floor of a symmetry-forbidden channel, and 2% of the real
    l > 0 density at the sphere boundary.  Nothing about them looks like
    roundoff to a reader that keeps them.
    """
    state = read_state(SIC / "STATE.OUT")
    assert state["nrmt"] == [297, 397]
    assert state["nrmtmax"] == 397
    padding = state["rhomt"][0, state["nrmt"][0]:, :]
    assert padding.size == (397 - 297) * state["lmmaxo"]
    assert not np.all(padding == 0.0)
    _, rhomt = muffin_tin(state, 0)
    assert np.abs(padding).max() > 1e-6
    assert np.abs(padding).max() > 0.01 * np.abs(rhomt[-1, 1:]).max()


def test_sic_the_two_columns_are_wholly_different():
    """Unlike diamond's two carbons, silicon and carbon share no symmetry
    relation, so an ias swap is visible in the l=0 channel alone: 2095
    e/Bohr^3 at the silicon nucleus against 130 at the carbon."""
    state = read_state(SIC / "STATE.OUT")
    _, carbon = muffin_tin(state, 0)
    _, silicon = muffin_tin(state, 1)
    assert carbon[0, 0] * Y00 == pytest.approx(129.90248354, rel=1e-9)
    assert silicon[0, 0] * Y00 == pytest.approx(2094.51777265, rel=1e-9)

    charges = info.parse_charges(SIC / "INFO.OUT")
    assert charges["species"] == ["C", "Si"]
    assert charges["core"] == pytest.approx(12.0)      # C 1s^2, Si 1s^2 2s^2 2p^6
    assert charges["muffin_tin"][1] > 2 * charges["muffin_tin"][0]
    # the heavier species leaks two orders of magnitude more core charge
    assert charges["core_leakage"][1] > 100 * charges["core_leakage"][0]


# --------------------------------------------------------------------------
# STATE_INIT.OUT: the top of Elk's first iteration (task 9006)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", TWO_STATE_FIXTURES, ids=lambda p: p.name)
def test_initial_state_has_the_same_layout(fixture):
    """Task 9006 stops before rhomag ever runs, so STATE_INIT.OUT holds
    rhoinit's superposition of free atomic densities.  Same layout, same
    meshes, same grids -- only the functions and efermi differ."""
    converged = read_state(fixture / "STATE.OUT")
    initial = read_state(fixture / "STATE_INIT.OUT")
    for key in ("version", "nspecies", "lmmaxo", "nrmtmax", "natoms", "nrmt",
                "ngridg", "ngvec", "ndmag", "xcgrad", "natmtot"):
        assert initial[key] == converged[key]
    assert len(initial["density_record"]) == len(converged["density_record"])
    assert initial["efermi"] != converged["efermi"]


@pytest.mark.parametrize("fixture", TWO_STATE_FIXTURES, ids=lambda p: p.name)
def test_initial_state_core_does_not_cancel_in_a_difference(fixture):
    """The reason to be careful with rho_SCF - rho_init.  The initial file's
    core is rhoinit's FREE-ATOM core (rhosp, rhoinit.f90:115-122); the
    converged file's is gencore's core in the crystal potential.  They are
    not the same function, so the difference is valence change PLUS core
    relaxation, and the core term is not negligible at the nucleus."""
    converged = read_state(fixture / "STATE.OUT")
    initial = read_state(fixture / "STATE_INIT.OUT")
    rsp, first = muffin_tin(converged, 0)
    _, second = muffin_tin(initial, 0)
    difference = (first[:, 0] - second[:, 0]) * Y00
    deep = rsp < 0.01                                  # well inside the 1s
    assert np.abs(difference[deep]).max() > 1e-3 * first[0, 0] * Y00


@pytest.mark.parametrize("fixture", TWO_STATE_FIXTURES, ids=lambda p: p.name)
def test_the_two_input_files_differ_only_in_the_tasks_block(fixture):
    """elk_init.in must be elk.in with `tasks 9006` -- same cell, same
    species, same k-mesh, same functional -- or the two STATE.OUT files in
    the fixture are not two states of the same calculation."""
    main = dict(inputfile.read_blocks(fixture / "elk.in"))
    init = dict(inputfile.read_blocks(fixture / "elk_init.in"))
    assert main.pop("tasks") == [["0"], ["33"]]
    assert init.pop("tasks") == [["9006"]]
    assert main == init
