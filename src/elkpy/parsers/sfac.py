"""Parser for Elk's structure-factor tables: SFACRHO.OUT (task 195,
src/sfacrho.f90) and SFACMAG_j.OUT (task 196, src/sfacmag.f90).

These are the direct diffraction observables -- the Fourier coefficients of
the all-electron density and of each magnetisation component,

    F(H)   = int_cell rho(r) exp(i H.r) d^3r
    F_j(H) = int_cell m_j(r) exp(i H.r) d^3r

evaluated by ``src/zftrf.f90`` over the H-vector set ``genhvec`` builds out
to |H| < ``hmaxvr``, sorted by length. Both routines multiply by the cell
volume and negate the imaginary part before printing, "in crystallography
the forward Fourier transform of real-space density is usually done with
positive phase and without 1/omega prefactor" (their own comment) -- so the
printed Re/Im pair is the crystallographic F(hkl), directly comparable to
an X-ray (SFACRHO) or polarised-neutron (SFACMAG) measurement, and the
H = 0 entry is the total number of electrons in the cell.

The (h,k,l) labels are the raw H-vector indices transformed by the input
matrix ``vhmat`` (a change of setting, e.g. primitive -> conventional
cell), which the file repeats in its own header; both branches of the
write are handled here:

    write(50,'(4I7,4G16.8)')   iv(:),mulh(ih),hc(ih),a,b,r   ! integer hkl
    write(50,'(3F7.2,I7,4G16.8)') v(:),mulh(ih),hc(ih),a,b,r ! non-integer

Both produce eight whitespace-separable fields, so one splitter covers
them; ``hkl`` comes back as floats in either case and ``multiplicity`` as
an int. ``mulh`` is the number of symmetry-equivalent H-vectors folded into
that row when ``reduceh`` is true.
"""

import numpy as np

_TABLE_HEADER = "h      k      l  multipl."


def _lines(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\n") for line in fh]


def parse_structure_factors(sfac_out_path):
    """Parse SFACRHO.OUT or SFACMAG_j.OUT.

    Returns {"vhmat": (3,3) array, "hkl": (n,3), "multiplicity": (n,) int,
    "h": (n,) |H| in Bohr^-1, "F": (n,) complex}.

    ``vhmat`` rows are the three lines written after "h k l indices
    transformed by vhmat matrix:", i.e. ``vhmat(:,1)``, ``vhmat(:,2)``,
    ``vhmat(:,3)`` -- Elk prints the matrix column by column, and the file
    is only ever a record of the input setting, so it is returned as
    printed rather than transposed.
    """
    lines = _lines(sfac_out_path)
    header = None
    vhmat = np.eye(3)
    for i, line in enumerate(lines):
        if "transformed by vhmat matrix" in line:
            vhmat = np.array([[float(x) for x in lines[i + 1 + r].split()] for r in range(3)])
        if _TABLE_HEADER in line:
            header = i
    if header is None:
        raise ValueError(f"no structure-factor table header found in {sfac_out_path}")

    hkl, multiplicity, hlen, real, imag = [], [], [], [], []
    for line in lines[header + 1 :]:
        fields = line.split()
        if len(fields) != 8:
            continue
        try:
            values = [float(x) for x in fields]
        except ValueError:
            continue
        hkl.append(values[0:3])
        multiplicity.append(int(values[3]))
        hlen.append(values[4])
        real.append(values[5])
        imag.append(values[6])
    if not hkl:
        raise ValueError(f"no structure-factor rows found in {sfac_out_path}")
    return {
        "vhmat": vhmat,
        "hkl": np.array(hkl),
        "multiplicity": np.array(multiplicity, dtype=int),
        "h": np.array(hlen),
        "F": np.array(real) + 1j * np.array(imag),
    }
