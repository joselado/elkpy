"""Parse EIGVAL.OUT (src/writeeval.f90): the second-variational
eigenvalues and occupation numbers on the ground state's own k-mesh.

Format, straight from writeeval.f90's write statements:

    <nkpt> : nkpt
    <nstsv> : nstsv
    (blank)
    <ik> <vkl(1)> <vkl(2)> <vkl(3)> : k-point, vkl
     (state, eigenvalue and occupancy below)
    <ist> <evalsv> <occsv>       x nstsv
    (blank)
    ... repeated per k-point

These occupation numbers are the ones Elk's own post-processing tasks use
(src/dielectric.f90 reads the same array through readoccsv), so taking
band fillings from here rather than from an assumed valence-electron
count is what makes an elkpy-side sum comparable to Elk's own -- and it
avoids the pitfall documented in docs/design.md #13, where core electrons
are counted into a valence band index they are not part of at all.
"""

import numpy as np


def parse_eigval(path):
    """Return (kpoints, energies, occupations): kpoints shape (nkpt, 3) in
    fractional coordinates, energies and occupations both (nkpt, nstsv),
    energies in Hartree.
    """
    kpoints, energies, occupations = [], [], []
    current_e, current_o = None, None
    with open(path) as fh:
        header = [fh.readline(), fh.readline()]
        nkpt = int(header[0].split()[0])
        nstsv = int(header[1].split()[0])
        for line in fh:
            if "k-point, vkl" in line:
                fields = line.split(":")[0].split()
                kpoints.append([float(x) for x in fields[1:4]])
                current_e, current_o = [], []
                energies.append(current_e)
                occupations.append(current_o)
                continue
            fields = line.split()
            if len(fields) == 3 and current_e is not None:
                try:
                    ist, e, o = int(fields[0]), float(fields[1]), float(fields[2])
                except ValueError:
                    continue
                if ist == len(current_e) + 1:
                    current_e.append(e)
                    current_o.append(o)
    energies = np.array(energies)
    occupations = np.array(occupations)
    if energies.shape != (nkpt, nstsv):
        raise ValueError(
            f"{path} declares nkpt={nkpt}, nstsv={nstsv} but holds "
            f"{energies.shape} eigenvalues"
        )
    return np.array(kpoints), energies, occupations


def occupations_if_uniform(path, tol=1e-6):
    """Return the single (nstsv,) occupation vector shared by every
    k-point in EIGVAL.OUT, or raise ValueError if the occupations are
    k-dependent.

    A gapped (insulating/semiconducting) system fills the same bands at
    every k, so one vector describes the whole zone; a metal does not, and
    any sum-over-states that assumed it would silently mis-count the Fermi
    surface. Failing loud here is the point -- see
    parsers.optical.circular_absorption, whose independent-particle
    interband spectrum is only defined for the gapped case.
    """
    _, _, occupations = parse_eigval(path)
    spread = np.max(np.abs(occupations - occupations[0]), axis=0)
    if np.any(spread > tol):
        worst = int(np.argmax(spread))
        raise ValueError(
            f"occupations in {path} vary across the k-mesh (band {worst + 1} varies by "
            f"{float(spread[worst]):.3e}): this system is metallic or has partially "
            f"occupied bands, so a single k-independent occupation vector does not "
            f"describe it"
        )
    return occupations[0]
