"""Parsers for Elk's time-dependent DFT output -- both the LINEAR-RESPONSE
family (tasks 320, 330/331) and the REAL-TIME family (tasks 450, 455, 456,
460-463, 480/481).

Every format here is transcribed from the `write` statement that produces
it; the citing comment on each function names the source file.

Two shared shapes recur:

  * "two-block" files -- nw lines of `(2G18.10)` (abscissa, value), a blank
    `write(50,*)` line, then the same number of lines again: real part
    first, imaginary part second. This is src/dielectric.f90's layout and
    the reader lives in parsers/dielectric.py (parse_two_blocks).

  * "vector-block" files -- three blank-line-separated blocks, one per
    Cartesian direction, each nw lines of `(3G18.10)` (abscissa, Re, Im).
    EFIELDW.OUT and JTOTW.OUT use this; parse_two_blocks cannot read them
    (wrong block count AND wrong column count), so parse_column_blocks()
    below is the general reader.

FREQUENCY GRIDS differ between tasks and getting them wrong silently
mislabels every spectrum:

  * task 320/330/331 (src/init3.f90): wrf(iw) = wplot(1)
    + (wplot(2)-wplot(1))/nwplot * (iw-1) + i*swidth, for iw = 1..nwplot.
    tddftlr writes only iw = 2..nwrf, so its files hold nwplot-1 rows;
    tddftsplr writes all nwplot rows.
  * task 456/480/481 (src/writeefieldw.f90, src/dielectric_tdrt.f90):
    w1 = max(wplot(1),0), w(iw) = w1 + (w2-w1)/nwplot*(iw-1), all nwplot
    rows.

TIME GRID (src/gentimes.f90): ntimes = nint(tstime/dtimes) + 1,
times(its) = (its-1)*dtimes, atomic units of time (1 a.u. ~ 24.19 as).
"""

import numpy as np

from .dielectric import parse_two_blocks


# --------------------------------------------------------------------------
# generic block reader
# --------------------------------------------------------------------------

def parse_column_blocks(path, ncols=None):
    """Split a blank-line-separated Elk output file into blocks of numeric
    rows, returning a list of (nrows, ncols) float arrays.

    `ncols`, when given, is asserted -- catching the case where a format
    changed underneath (e.g. a `(4G18.10)` line that only wrote 2 items
    because the item list was shorter than the format, which Elk does in
    JTOTM_TD.OUT and MOMENT_TD.OUT).
    """
    blocks = []
    current = []
    with open(path) as fh:
        for line in fh:
            tokens = line.split()
            if not tokens:
                if current:
                    blocks.append(current)
                    current = []
                continue
            current.append([float(t) for t in tokens])
    if current:
        blocks.append(current)
    out = []
    for block in blocks:
        widths = {len(row) for row in block}
        if len(widths) != 1:
            raise ValueError(f"ragged block in {path}: row widths {sorted(widths)}")
        width = widths.pop()
        if ncols is not None and width != ncols:
            raise ValueError(f"{path}: expected {ncols} columns per row, got {width}")
        out.append(np.array(block, dtype=float))
    return out


def _complex_from_two_blocks(path):
    energies, real_part, imag_part = parse_two_blocks(path)
    return energies, real_part + 1j * imag_part


# --------------------------------------------------------------------------
# task 320 -- linear-response TDDFT (src/tddftlr.f90)
# --------------------------------------------------------------------------

def parse_tddft_response(path):
    """Return (energies, values) for any of task 320's two-block spectra:
    EPSILON_TDDFT_ij.OUT, EPSINV_TDDFT_ij.OUT, EPSM_TDDFT_ij.OUT (q = 0,
    indexed by optcomp), EPSILON_TDDFT.OUT / EPSINV_TDDFT.OUT (finite q,
    unindexed), FARADAY.OUT, KERR_TDDFT.OUT and MLD.OUT.

    src/tddftlr.f90 writes each as
        do iw=2,nwrf:  write(50,'(2G18.10)') dble(wrf(iw)), dble(...)
        write(50,*)
        do iw=2,nwrf:  write(50,'(2G18.10)') dble(wrf(iw)), aimag(...)
    -- note the loop STARTS AT 2, so a file holds nwplot-1 rows per block,
    the first being wplot(1) + one grid step and not wplot(1) itself.

    energies in Hartree. EPSILON/EPSINV/EPSM are dimensionless;
    KERR_TDDFT.OUT is in DEGREES (tddftlr multiplies by 180/pi, as
    src/moke.f90 does); FARADAY.OUT and MLD.OUT are in radians.
    """
    return _complex_from_two_blocks(path)


def tddftlr_energies(wplot, nwplot):
    """The frequency grid task 320's files actually carry, in Hartree.

    src/init3.f90 builds wrf(iw) = w1 + (w2-w1)/nwplot*(iw-1) for
    iw = 1..nwplot (w1 = wplot(1), w2 = max(wplot(2), w1)) -- note there is
    NO clipping of a negative wplot(1) to zero here, unlike task 121. The
    output loop then drops iw = 1, so this returns entries 2..nwplot.
    """
    w1 = float(wplot[0])
    w2 = max(float(wplot[1]), w1)
    step = (w2 - w1) / float(nwplot)
    return w1 + step * np.arange(1, nwplot, dtype=float)


# --------------------------------------------------------------------------
# tasks 330/331 -- spin-polarised linear response (src/tddftsplr.f90)
# --------------------------------------------------------------------------

def parse_spin_response(path):
    """Return (energies, chi) for one CHI_ij.OUT / CHI0_ij.OUT /
    CHI_T.OUT / CHI0_T.OUT file of tasks 330/331.

    Same two-block layout, but src/tddftsplr.f90 loops `do iw=1,nwrf` --
    ALL nwplot rows, unlike task 320's `do iw=2,nwrf`.

    The index pair (i, j) runs over 0..3, not 1..3: 0 is the charge
    channel and 1-3 the three Cartesian magnetisation channels, so
    CHI_00 is the density-density response dn/dv, CHI_0j the
    density-magnetisation response dn/dB_j, CHI_i0 the
    magnetisation-density response dm_i/dv and CHI_ij (i,j >= 1) the
    magnetisation-magnetisation response dm_i/dB_j whose poles are the
    magnon energies. CHI_T.OUT is the transverse combination
    m_+- = m_x +- i m_y, written only for a collinear (`.not.ncmag`)
    ground state.

    Units: omega in Hartree, chi in atomic units.
    """
    return _complex_from_two_blocks(path)


# --------------------------------------------------------------------------
# task 450 -- the laser A-field (src/genafieldt.f90)
# --------------------------------------------------------------------------

def parse_afieldt(path):
    """Return (times, afield) from AFIELDT.OUT: times shape (ntimes,) in
    atomic units of time, afield shape (ntimes, 3) in atomic units.

    src/genafieldt.f90:
        write(50,'(I8," : number of time steps")') ntimes
        do its=1,ntimes:  write(50,'(I8,4G18.10)') its, times(its),
                                                   afieldt(:,its)

    The field is a sum of Gaussian-enveloped chirped sinusoids (`pulse`),
    polynomial ramps (`ramp`) and rectangular steps (`step`); the electric
    field seen by the electrons is E(t) = -(1/c) dA/dt, so a pulse with
    zero time-integral of E leaves no net momentum transfer.
    """
    times = []
    afield = []
    with open(path) as fh:
        header = fh.readline().split()
        if not header:
            raise ValueError(f"{path} is empty; expected an 'I8 : number of time steps' header")
        ntimes = int(header[0])
        for line in fh:
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) != 5:
                raise ValueError(
                    f"{path}: expected 5 columns (step, time, Ax, Ay, Az), got {len(tokens)}"
                )
            times.append(float(tokens[1]))
            afield.append([float(t) for t in tokens[2:5]])
    if len(times) != ntimes:
        raise ValueError(f"{path} declares ntimes={ntimes} but holds {len(times)} rows")
    return np.array(times), np.array(afield)


def parse_afspt(path):
    """Return (times, afspt) from AFSPT.OUT, the spin-dependent A-field
    written only when `tafspt` is set.

    src/genafieldt.f90: `write(50,'(I8,10G18.10)') its, times(its),
    afspt(:,:,its)` with afspt(3,3) -- Fortran column-major, so the nine
    values are (a=1..3 fastest, j=1..3) and reshape order is (3, 3) in
    Fortran order: afspt[a, j] multiplies sigma_j in direction a.
    """
    times = []
    rows = []
    with open(path) as fh:
        header = fh.readline().split()
        ntimes = int(header[0])
        for line in fh:
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) != 11:
                raise ValueError(f"{path}: expected 11 columns, got {len(tokens)}")
            times.append(float(tokens[1]))
            rows.append([float(t) for t in tokens[2:11]])
    if len(times) != ntimes:
        raise ValueError(f"{path} declares ntimes={ntimes} but holds {len(times)} rows")
    # each ROW is one time step's (3,3) slice in Fortran order; reshaping the
    # whole (ntimes, 9) array at once with order="F" would interleave the time
    # axis, so each row is reshaped separately
    frames = [np.asarray(r).reshape((3, 3), order="F") for r in rows]
    return np.array(times), np.array(frames)


# --------------------------------------------------------------------------
# task 455 -- A-field power density (src/writeafpdt.f90)
# --------------------------------------------------------------------------

def parse_afpdt(path):
    """Return (times, power_density) from AFPDT.OUT.

    src/writeafpdt.f90: `write(50,'(2G18.10)') times(its), pd(its)` for all
    ntimes rows, with pd = |dA/dt|^2 / (8 pi c) -- the instantaneous laser
    power density in atomic units (its time integral is AFTED.OUT).
    """
    blocks = parse_column_blocks(path, ncols=2)
    if len(blocks) != 1:
        raise ValueError(f"expected a single block in {path}, got {len(blocks)}")
    data = blocks[0]
    return data[:, 0], data[:, 1]


def parse_afted(path):
    """Return the total energy density of the A-field (atomic units) from
    AFTED.OUT.

    src/writeafpdt.f90 writes a blank line then
    `write(50,'("Total energy density : ",G18.10)') ed`
    and a second line with the same number in J/cm^2.
    """
    with open(path) as fh:
        for line in fh:
            if "Total energy density" in line and "J/cm" not in line:
                return float(line.split(":")[1])
    raise ValueError(f"no 'Total energy density' line in {path}")


# --------------------------------------------------------------------------
# task 456 -- Fourier transform of the electric field (src/writeefieldw.f90)
# --------------------------------------------------------------------------

def parse_efieldw(path):
    """Return (energies, efield) from EFIELDW.OUT: energies shape (nw,) in
    Hartree, efield shape (nw, 3) complex in atomic units.

    src/writeefieldw.f90:
        do i=1,3
          do iw=1,nwplot:  write(50,'(3G18.10)') w(iw), ew(iw,i)
          write(50,*)
        end do
    -- THREE blank-separated blocks (one per Cartesian direction), each
    with three columns (omega, Re E, Im E), and a trailing blank line
    after the last block.

    E(w) is obtained by numerically Fourier-transforming A(t), taking the
    derivative E = -(1/c) dA/dt analytically in frequency space (a factor
    -i w / c), then applying a Lorentzian convolution of width `swidth` to
    suppress the ringing from the finite time window.
    """
    blocks = parse_column_blocks(path, ncols=3)
    if len(blocks) != 3:
        raise ValueError(
            f"expected 3 blocks (one per Cartesian direction) in {path}, got {len(blocks)}"
        )
    energies = blocks[0][:, 0]
    for block in blocks[1:]:
        if block.shape != blocks[0].shape or not np.allclose(block[:, 0], energies):
            raise ValueError(f"the three blocks in {path} are not on the same energy grid")
    values = np.stack([b[:, 1] + 1j * b[:, 2] for b in blocks], axis=1)
    return energies, values


# --------------------------------------------------------------------------
# tasks 460-463 -- real-time evolution observables (src/writetddft.f90)
# --------------------------------------------------------------------------

def parse_time_series(path, ncomponents=None):
    """Return (times, values) from one of the APPEND-mode `*_TD.OUT`
    observable files src/writetddft.f90 writes once per time step.

    All share the shape "one row per time step, first column the time":

        TOTENERGY_TD.OUT   (2G18.10) time, engytot          -> 1 component
        CHARGEIR_TD.OUT    (2G18.10) time, chgir            -> 1
        JTOTM_TD.OUT       (4G18.10) time, jtotm            -> 1 (the
                           format is wider than the item list, so only two
                           numbers are actually written)
        MOMENTM_TD.OUT     (2G18.10) time, momtotm          -> 1
        JTOT_TD.OUT        (4G18.10) time, jtot(1:3)        -> 3
        AFIND_TD.OUT       (4G18.10) time, afindt(:,0)      -> 3
        MOMENT_TD.OUT      (4G18.10) time, momtot(1:ndmag)  -> 1 or 3,
                           depending on whether the ground state is
                           collinear (ndmag=1) or non-collinear (ndmag=3)

    `ncomponents`, when given, is asserted. `values` comes back shape
    (nsteps,) for one component and (nsteps, n) otherwise.

    NOTE these files are opened with position='APPEND' and deleted only at
    itimes <= 1, so a restarted run (task 461/463) CONTINUES them rather
    than starting fresh.
    """
    blocks = parse_column_blocks(path)
    if len(blocks) != 1:
        raise ValueError(f"expected a single block in {path}, got {len(blocks)}")
    data = blocks[0]
    n = data.shape[1] - 1
    if ncomponents is not None and n != ncomponents:
        raise ValueError(f"{path}: expected {ncomponents} value columns, got {n}")
    if n == 1:
        return data[:, 0], data[:, 1]
    return data[:, 0], data[:, 1:]


def parse_chargemt_td(path):
    """Return (times, charges) from CHARGEMT_TD.OUT: the muffin-tin charge
    of every atom at every written time step.

    src/writetddft.f90 writes one blank-line-terminated block per step:
        write(50,'(G18.10)') times(itimes)
        do is, do ia:  write(50,'(2I4,G18.10)') is, ia, chgmt(ias)
        write(50,*)

    MOMENTMT_TD.OUT has the same shape with `3G18.10` of mommt(1:ndmag)
    instead, so parse_atom_time_series() handles both.
    """
    return parse_atom_time_series(path)


def parse_atom_time_series(path):
    """Return (times, values, labels) for the per-atom, per-time-step
    block files CHARGEMT_TD.OUT and MOMENTMT_TD.OUT.

    times shape (nsteps,), values shape (nsteps, natoms, ncomponents),
    labels a list of (is, ia) species/atom index pairs in file order
    (1-based, as Elk writes them). ncomponents is 1 for CHARGEMT_TD.OUT and
    ndmag (1 or 3) for MOMENTMT_TD.OUT.
    """
    times = []
    frames = []
    labels = None
    current = []
    header = None
    with open(path) as fh:
        for line in fh:
            tokens = line.split()
            if not tokens:
                if header is not None:
                    times.append(header)
                    frame_labels = [(int(r[0]), int(r[1])) for r in current]
                    if labels is None:
                        labels = frame_labels
                    elif labels != frame_labels:
                        raise ValueError(f"{path}: atom ordering changed between time steps")
                    frames.append([[float(x) for x in r[2:]] for r in current])
                    header = None
                    current = []
                continue
            if header is None:
                header = float(tokens[0])
            else:
                current.append(tokens)
    if header is not None:
        raise ValueError(f"{path}: last time-step block is not blank-line terminated")
    return np.array(times), np.array(frames), labels


# --------------------------------------------------------------------------
# tasks 480/481 -- dielectric function from the real-time current
# (src/dielectric_tdrt.f90)
# --------------------------------------------------------------------------

def parse_epsilon_tdrt(path):
    """Return (energies, epsilon) for one EPSILON_TDRT_ij.OUT file.

    src/dielectric_tdrt.f90 writes the usual two-block layout. Note it
    loops over ALL i,j = 1..3 unconditionally -- `optcomp` is NOT consulted
    -- so all nine files always appear, but only the components not
    orthogonal to the applied A-field are meaningful (the subroutine says
    so itself in its closing Info message; a component with E_j(w) ~ 0 is
    set identically to zero by its own 1e-8 guard).

    The relation used is eps_ij(w) = delta_ij
    + 4 pi i J_i(w) / [ (w + i*swidth) E_j(w) ], i.e. Ohm's law inverted
    from the real-time current response to the driving field.
    """
    return _complex_from_two_blocks(path)


def parse_jtotw(path):
    """Return (energies, current) from JTOTW.OUT: energies shape (nw,) in
    Hartree, current shape (nw, 3) complex in atomic units.

    Same three-block, three-column shape as EFIELDW.OUT
    (src/dielectric_tdrt.f90's closing write loop).
    """
    return parse_efieldw(path)
