"""Parsers for Elk's GW task family (tasks 600/601/610/620/630/640).

Only the *formatted* GW output is parsed here. The self-energy itself
(``GWSEFM.OUT``), the inverse dielectric matrix (``EPSINV.OUT``) and the GW
natural orbitals/occupations (``EVECSV.OUT``/``OCCSV.OUT`` rewritten by task
640) are Fortran ``form='UNFORMATTED', access='DIRECT'`` files whose record
length is set by the compiler's ``inquire(iolength=...)`` -- see
``vendor/elk/src/putgwsefm.f90``, ``putepsinv.f90``. Those are treated as
opaque intermediates: elkpy checks that they exist and hands the directory to
the next task, rather than decoding a build-dependent binary layout.

Every format below is transcribed from the ``write`` statement that produces
it; see the citation on each function.
"""

import numpy as np


def parse_gw_fermi_energy(path):
    """The GW Fermi energy in Hartree, from ``GWEFERMI.OUT`` (task 630).

    src/writegwefm.f90::

        open(50,file='GWEFERMI.OUT',form='FORMATTED',action='WRITE')
        write(50,'(G18.10)') efermi
    """
    with open(path) as fh:
        for line in fh:
            if line.strip():
                return float(line.split()[0])
    raise ValueError(f"{path} contains no numeric value")


def parse_gw_total_spectral_function(path):
    """The Brillouin-zone-summed GW spectral function, from ``GWTSF.OUT``
    (task 610).

    src/gwspecf.f90::

        open(50,file='GWTSF.OUT',form='FORMATTED')
        do iw=1,nwplot
          write(50,'(2G18.10)') wr(iw),sft(iw)
        end do

    Returns ``(w, sf)``: the real-axis frequency grid (Hartree, spanning
    ``wplot``) and the spectral function in states/Hartree/unit cell. Note
    the Fermi energy of the GW spectral function is **undetermined** in this
    file (gwspecf.f90 says so on stdout); only the Kohn-Sham reference
    eigenvalues in the per-k files below are Fermi-referenced.
    """
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    return data[:, 0], data[:, 1]


def parse_gw_spectral_function(path):
    """One k-point's GW spectral function, from ``GWSF_Kkkkkkk.OUT``
    (task 610, filename ``write(fname,'("GWSF_K",I6.6,".OUT")') ik``).

    src/writegwsf.f90 writes the spectral function first, then a blank line,
    then a plotting-marker pair per Kohn-Sham state::

        do iw=1,nwplot
          write(50,'(2G18.10)') wr(iw),sf(iw)
        end do
        write(50,*)
        do ist=1,nstsv
          e=evalsv(ist,ik)-efermi
          write(50,'(2G18.10)') e,0.d0
          write(50,'(2G18.10)') e,1.d0/swidth
          write(50,*)
        end do

    Only the first block is the spectrum, so parsing must stop at the first
    blank line -- reading the whole file with ``loadtxt`` would silently
    append ``2*nstsv`` marker points to the curve.

    Returns ``(w, sf)``; see :func:`parse_gw_spectral_function_eigenvalues`
    for the trailing Kohn-Sham eigenvalues.
    """
    w, sf = [], []
    with open(path) as fh:
        for raw in fh:
            if not raw.strip():
                break
            fields = raw.split()
            w.append(float(fields[0]))
            sf.append(float(fields[1]))
    return np.array(w), np.array(sf)


def parse_gw_spectral_function_eigenvalues(path):
    """The Kohn-Sham eigenvalues drawn as vertical markers in the second
    half of ``GWSF_Kkkkkkk.OUT`` (src/writegwsf.f90, see above).

    Each state contributes the pair ``(e, 0)`` and ``(e, 1/swidth)``, so the
    eigenvalues are the distinct abscissae after the first blank line, in
    file (i.e. band) order. Energies are Hartree, relative to the Kohn-Sham
    Fermi energy.
    """
    seen_break = False
    energies = []
    with open(path) as fh:
        for raw in fh:
            if not raw.strip():
                seen_break = True
                continue
            if not seen_break:
                continue
            e = float(raw.split()[0])
            if not energies or energies[-1] != e:
                energies.append(e)
    return np.array(energies)


def parse_gw_band(path):
    """The GW spectral-function band structure, from ``GWBAND.OUT``
    (task 620).

    src/gwbandstr.f90::

        write(85,'(2I6," : grid size")') nwplot,npp1d
        do ip=ip0gw,npp1d
          ...
          do iw=1,nwplot
            write(85,'(3G18.10)') dpp1d(ip),wr(iw),sf(iw)
          end do
        end do

    **The header order is the reverse of the data order**: the header reads
    ``nwplot`` then ``npp1d``, while the loops run over plot points on the
    outside and frequencies on the inside, so the rows reshape as
    ``(npp1d, nwplot)``.

    Returns ``(distances, frequencies, sf)`` with `distances` shape
    ``(npp1d,)`` (distance along the k-path, Bohr^-1), `frequencies` shape
    ``(nwplot,)`` (Hartree) and `sf` shape ``(npp1d, nwplot)``.

    ``ip0gw`` (input block ``ip0gw``) restarts an interrupted run partway
    along the path *appending* to an existing file, so the number of rows
    present is taken from the file rather than from the header's ``npp1d``.
    """
    with open(path) as fh:
        header = fh.readline().split()
        nwplot = int(header[0])
        data = np.atleast_2d(np.loadtxt(fh))
    npoints = data.shape[0] // nwplot
    if npoints * nwplot != data.shape[0]:
        raise ValueError(
            f"{path}: {data.shape[0]} rows is not a whole multiple of "
            f"nwplot={nwplot} from the header (src/gwbandstr.f90)"
        )
    block = data[: npoints * nwplot].reshape(npoints, nwplot, 3)
    distances = block[:, 0, 0]
    frequencies = block[0, :, 1]
    return distances, frequencies, block[:, :, 2]
