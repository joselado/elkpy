"""Parse task 130's EXPIQR.OUT (src/writeexpmat.f90) and task 135's
WFPW.OUT (src/writewfpw.f90).

Both expose a wavefunction-level primitive rather than a spectrum:

  * task 130 gives < i, k+q | e^{iq.r} | j, k >, the plane-wave (density)
    matrix elements that every response function is built from -- the
    same object src/genexpmat.f90 supplies to the RPA/BSE machinery, here
    dumped for one q-vector (`vecql`) and the k-points listed in
    `kstlist`. Note src/writeexpmat.f90 multiplies the muffin-tin phase
    factor by the cell volume (`expmt(:,:)=omega*expmt(:,:)`), so the
    printed numbers carry that factor of Omega.

  * task 135 re-expands each second-variational LAPW state in a PURE
    plane-wave basis {H+k} up to |H+k| < `hkmax` -- the natural basis for
    comparing an all-electron LAPW wavefunction with a pseudopotential
    code, or for computing an electron momentum density. It is written
    UNFORMATTED/DIRECT, one record per k-point.
"""

import numpy as np


def parse_expiqr(path):
    """Return a dict describing EXPIQR.OUT.

    {"vecql": (3,), "vecqc": (3,),
     "kpoints": [{"vkl": (3,), "vkc": (3,), "matrix": (nstsv, nstsv) complex}]}

    Format, from src/writeexpmat.f90's write statements:

        write(50,*)                                     ! blank
        write(50,'("q-vector (lattice coordinates) :")')
        write(50,'(3G18.10)') vecql
        write(50,'("q-vector (Cartesian coordinates) :")')
        write(50,'(3G18.10)') vecqc
        write(50,*)
        write(50,'(I8," : number of k-points")') nk
        write(50,'(I6," : number of states per k-point")') nstsv
        do jk=1,nk
          write(50,*)
          write(50,'(" k-point (lattice coordinates) :")')
          write(50,'(3G18.10)') vkl(:,ik)
          write(50,*)
          write(50,'(" k-point (Cartesian coordinates) :")')
          write(50,'(3G18.10)') vkc(:,ik)
          do i=1,nstsv
            write(50,*)
            write(50,'(I6," : state i; state j, <...>, |<...>|<superscript 2> below")') i
            do j=1,nstsv
              write(50,'(I6,3G18.10)') j, Re, Im, |.|^2
            end do
          end do
        end do

    matrix[i-1, j-1] is < i, k+q | e^{iq.r} | j, k > with i the ROW (the
    k+q state) and j the column (the k state), matching the loop order
    above. The file is UTF-8 (the header carries a superscript two), so it
    is decoded as such with errors replaced rather than assumed ASCII.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()

    pos = 0
    n = len(lines)

    def next_nonblank():
        nonlocal pos
        while pos < n and not lines[pos].strip():
            pos += 1
        if pos >= n:
            raise ValueError(f"unexpected end of {path}")
        line = lines[pos]
        pos += 1
        return line

    def expect(substring):
        line = next_nonblank()
        if substring not in line:
            raise ValueError(f"{path}: expected a line containing {substring!r}, got {line!r}")
        return line

    def read_vector():
        tokens = next_nonblank().split()
        if len(tokens) != 3:
            raise ValueError(f"{path}: expected a 3-vector, got {tokens}")
        return np.array([float(t) for t in tokens])

    expect("q-vector (lattice coordinates)")
    vecql = read_vector()
    expect("q-vector (Cartesian coordinates)")
    vecqc = read_vector()
    nk = int(expect("number of k-points").split()[0])
    nstsv = int(expect("number of states per k-point").split()[0])

    kpoints = []
    for _ in range(nk):
        expect("k-point (lattice coordinates)")
        vkl = read_vector()
        expect("k-point (Cartesian coordinates)")
        vkc = read_vector()
        matrix = np.zeros((nstsv, nstsv), dtype=complex)
        for i in range(nstsv):
            header = expect(": state i")
            if int(header.split()[0]) != i + 1:
                raise ValueError(f"{path}: state index out of order at row {i + 1}")
            for j in range(nstsv):
                tokens = next_nonblank().split()
                if len(tokens) != 4:
                    raise ValueError(
                        f"{path}: expected 4 columns (j, Re, Im, |.|^2), got {tokens}"
                    )
                if int(tokens[0]) != j + 1:
                    raise ValueError(f"{path}: column index out of order at ({i + 1},{j + 1})")
                matrix[i, j] = complex(float(tokens[1]), float(tokens[2]))
        kpoints.append({"vkl": vkl, "vkc": vkc, "matrix": matrix})
    return {"vecql": vecql, "vecqc": vecqc, "kpoints": kpoints}


def read_wfpw(path):
    """Read task 135's WFPW.OUT and return
    {"kpoints": (nkpt, 3), "wfpw": (nkpt, nhkmax, nspinor, nstsv) complex}.

    src/writewfpw.f90 opens the file DIRECT/UNFORMATTED with a record
    length taken from

        inquire(iolength=recl) vkl(:,1), nhkmax, nspinor, nstsv, wfpw

    and writes `write(270,rec=ik) vkl(:,ik), nhkmax, nspinor, nstsv, wfpw`
    with wfpw(nhkmax, nspinor, nstsv) complex(8).

    A direct-access record carries NO record markers, so the layout is
    simply 3 float64 (24 bytes), 3 int32 (12 bytes), then the complex
    array in Fortran (column-major) order, 16 bytes per element. The
    record is therefore self-describing: the three integers at bytes
    24..36 of record 1 give the record length itself, which is checked
    against the file size (and so against the assumption that this build
    uses 4-byte integers and byte-valued `iolength` -- gfortran's default,
    and build-config/make.inc passes neither -fdefault-integer-8 nor
    -frecord-marker).

    The coefficients are those of |psi> = sum_H c_H exp(i(H+k).r) over the
    whole cell, muffin-tin regions included -- i.e. genwfpw has already
    projected the LAPW muffin-tin expansion onto plane waves, which is why
    `hkmax` (not `rgkmax`) controls the truncation.
    """
    import os

    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(36)
        if len(head) < 36:
            raise ValueError(f"{path} is shorter than one record header")
        nhkmax, nspinor, nstsv = np.frombuffer(head[24:36], dtype="<i4")
        nhkmax, nspinor, nstsv = int(nhkmax), int(nspinor), int(nstsv)
        if min(nhkmax, nspinor, nstsv) < 1:
            raise ValueError(
                f"{path}: implausible header (nhkmax, nspinor, nstsv) = "
                f"{(nhkmax, nspinor, nstsv)}; is this really a WFPW.OUT?"
            )
        ncoeff = nhkmax * nspinor * nstsv
        recl = 36 + 16 * ncoeff
        if size % recl != 0:
            raise ValueError(
                f"{path}: file size {size} is not a multiple of the record length "
                f"{recl} implied by (nhkmax, nspinor, nstsv) = {(nhkmax, nspinor, nstsv)}"
            )
        nkpt = size // recl
        fh.seek(0)
        raw = fh.read()

    kpoints = np.zeros((nkpt, 3))
    wfpw = np.zeros((nkpt, nhkmax, nspinor, nstsv), dtype=complex)
    for ik in range(nkpt):
        base = ik * recl
        kpoints[ik] = np.frombuffer(raw[base:base + 24], dtype="<f8")
        dims = np.frombuffer(raw[base + 24:base + 36], dtype="<i4")
        if tuple(int(d) for d in dims) != (nhkmax, nspinor, nstsv):
            raise ValueError(f"{path}: record {ik + 1} disagrees on array dimensions")
        block = np.frombuffer(raw[base + 36:base + recl], dtype="<c16")
        wfpw[ik] = block.reshape((nhkmax, nspinor, nstsv), order="F")
    return {"kpoints": kpoints, "wfpw": wfpw}
