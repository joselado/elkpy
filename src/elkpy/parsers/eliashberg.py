"""Parsers for the Eliashberg / superconductivity output files.

- ``ALPHA2F.OUT``            -- the Eliashberg spectral function alpha^2 F(w)
                               (task 250, src/alpha2f.f90).
- ``MCMILLAN.OUT``           -- lambda, <ln w>, <w^2>^(1/2), mu* and the
                               McMillan-Allen-Dynes T_c derived from it
                               (task 250, via src/mcmillan.f90).
- ``ELIASHBERG_GAP_T.OUT``   -- the gap and mass-renormalisation function at
                               the first Matsubara frequency vs temperature
                               (task 260, src/eliashberg.f90).
- ``ELIASHBERG_IA.OUT``      -- Delta(i w_n) and Z(i w_n) on the imaginary
                               axis, one ragged block per temperature.
- ``ELIASHBERG_GAP_RA.OUT`` / ``ELIASHBERG_Z_RA.OUT`` -- the same functions
                               continued to the real axis by Pade
                               approximants, one block per temperature.

All formats are transcribed from the cited ``write`` statements in
vendor/elk/src/; MCMILLAN.OUT is additionally cross-checked against the
real output file Elk ships in
vendor/elk/examples/phonons-superconductivity/Nb-DFPT/MCMILLAN.OUT.
"""

import numpy as np

from .phonon import split_blocks

# The keyword each MCMILLAN.OUT line is recognised by, and the dict key it
# becomes. Deliberately ASCII substrings of lines that also contain the
# UTF-8 characters "lambda" and "mu*" spell out in Elk's source -- matching
# on those would tie the parser to an encoding detail.
_MCMILLAN_KEYS = (
    ("Electron-phonon coupling constant", "lambda"),
    ("Logarithmic average frequency", "wlog"),
    ("RMS average frequency", "wrms"),
    ("Coulomb pseudopotential", "mustar"),
    ("(kelvin)", "tc"),
)


def parse_alpha2f(path):
    """Return (frequencies, alpha2f): both shape (nwplot,), frequencies in
    Hartree.

    src/alpha2f.f90:
        do iw=1,nwplot
          write(50,'(2G18.10)') w(iw),a2f(iw)

    The grid is NOT ``wplot`` -- alpha2f.f90 builds its own from the actual
    minimum and maximum phonon frequencies padded by 10%, so it can extend
    slightly below zero. Only ``nwplot`` (the number of points) and
    ``nswplot`` (the smoothing passes applied to a2F) are taken from the
    input; the frequency window in the ``wplot`` block is ignored here.

    alpha^2 F is dimensionless: alpha2f.f90 normalises by
    2 pi (N(E_F)/2) dw ngrkf^3.
    """
    data = np.loadtxt(path, ndmin=2)
    return data[:, 0], data[:, 1]


def parse_mcmillan(path):
    """Parse MCMILLAN.OUT into
    {"lambda", "wlog", "wrms", "mustar", "tc"}.

    src/alpha2f.f90 writes one "label : value" line per quantity, separated
    by blank lines; two of the labels contain non-ASCII characters (the
    literal Greek lambda and mu), so the file is opened as UTF-8 and matched
    on the ASCII part of each label.

    Units: `wlog` and `wrms` in Hartree (the logarithmic and root-mean-square
    averages of omega over alpha^2 F/omega), `lambda` and `mustar`
    dimensionless, `tc` in KELVIN -- src/mcmillan.f90 divides by
    ``1.2*kboltz`` and then applies Allen-Dynes' strong-coupling f1 f2
    correction factors, so this is the Allen-Dynes T_c, not bare McMillan.
    """
    values = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if ":" not in line:
                continue
            for marker, key in _MCMILLAN_KEYS:
                if marker in line:
                    values[key] = float(line.rsplit(":", 1)[1])
                    break
    missing = [key for _, key in _MCMILLAN_KEYS if key not in values]
    if missing:
        raise ValueError(f"{path} is missing entries: {missing}")
    return values


def parse_gap_vs_temperature(path):
    """Parse ELIASHBERG_GAP_T.OUT.

    src/eliashberg.f90:
        write(64,'(3G18.10)') temp,d(0),z(0)

    one line per temperature step, no blank separators. Returns
    (temperatures (nt,) in kelvin, gap (nt,) in Hartree, z (nt,)
    dimensionless) -- the gap function Delta and mass renormalisation Z
    evaluated at the LOWEST fermionic Matsubara frequency w_0 = pi k_B T,
    which is the standard proxy for the superconducting gap. T_c is where
    the gap collapses to the 1e-4 Hartree seed value, not a number Elk
    prints here.
    """
    data = np.loadtxt(path, ndmin=2)
    return data[:, 0], data[:, 1], data[:, 2]


def parse_imaginary_axis(path):
    """Parse ELIASHBERG_IA.OUT: one block per temperature, each
    ``(2 nwf + 1)`` lines of

        write(63,'(3G18.10)') wf(n),d(m),z(m)

    over n = -nwf..nwf, where ``m = n`` for n >= 0 and ``m = -n-1``
    otherwise -- so the functions are written symmetrically about zero even
    though only the non-negative half is solved for.

    Blocks are RAGGED: ``nwf = nint(wfmax/(2 pi k_B T))`` shrinks as the
    temperature rises, so this returns a list of (matsubara, gap, z)
    tuples, one per temperature, in the same order as
    parse_gap_vs_temperature.
    """
    return [(b[:, 0], b[:, 1], b[:, 2]) for b in split_blocks(path, ncols=3)]


def parse_real_axis(path):
    """Parse ELIASHBERG_GAP_RA.OUT or ELIASHBERG_Z_RA.OUT -- one block per
    temperature of

        write(65,'(3G18.10)') dble(zout(i)),a,b

    with a/b the real and imaginary parts of the Pade continuation of
    Delta (or Z) onto the real frequency axis. Returns a list of
    (frequencies, values complex), one per temperature.

    The real-axis gap edge is where Re Delta(w) = w; the imaginary part is
    the quasiparticle damping. Pade continuation of numerical Matsubara
    data is notoriously ill-conditioned, so treat the high-frequency tail
    with suspicion.
    """
    return [(b[:, 0], b[:, 1] + 1j * b[:, 2]) for b in split_blocks(path, ncols=3)]
