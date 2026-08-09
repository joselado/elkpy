"""Analytic (2l+1) x (2l+1) angular momentum matrices L_x, L_y, L_z in the
complex spherical harmonic basis (m = -l..l ascending), transcribed directly
from upstream Elk's lopzflm.f90 (vendor/elk/src/lopzflm.f90).

NOT used by the production Fortran path: src/elkpy_eigenstates.f90's
elkpy_angmomproj calls lopzflm itself for the actual matrix elements, and
this module's Fortran-vs-Python duplication is deliberate -- it exists
purely as an independent pin on lopzflm's convention, in the spirit of
parsers/berry.py's synthetic gauge-invariance tests. A Fortran-side bug that
mislabels or negates one of the three Cartesian components (e.g. swapping
the Ly/Lz output columns) would NOT be caught by Hermiticity alone -- conj()
of a Hermitian matrix is still Hermitian, so a lone sign flip on Ly is
invisible to that check (docs/design.md #19) -- so this transcription is
checked instead against the su(2) commutation algebra and the Casimir
identity on the full, untruncated (2l+1)-dimensional space, where both hold
exactly (see tests/test_parsers_angular_momentum.py); on a real diagonal-
isation's band-window-truncated output neither identity holds as a matrix
product (docs/design.md #19), so this check cannot be repeated there.
"""

import numpy as np


def angular_momentum_matrices(l):
    """Return (Lx, Ly, Lz), each a (2l+1, 2l+1) complex Hermitian matrix (m
    ascending -l..l), in the same convention as lopzflm.f90:

        (Lx + iLy) Y_lm = sqrt((l-m)(l+m+1)) Y_l,m+1
        (Lx - iLy) Y_lm = sqrt((l+m)(l-m+1)) Y_l,m-1
        Lz Y_lm = m Y_lm

    hbar = 1, so Lz has integer eigenvalues -l..l.
    """
    n = 2 * l + 1
    lx = np.zeros((n, n), dtype=complex)
    ly = np.zeros((n, n), dtype=complex)
    lz = np.zeros((n, n), dtype=complex)
    ms = np.arange(-l, l + 1)
    for k, m in enumerate(ms):
        lz[k, k] = m
        if m < l:
            t1 = 0.5 * np.sqrt((l - m) * (l + m + 1))
            lx[k + 1, k] = t1
            ly[k + 1, k] = -1j * t1
        if m > -l:
            t1 = 0.5 * np.sqrt((l + m) * (l - m + 1))
            lx[k - 1, k] = t1
            ly[k - 1, k] = 1j * t1
    return lx, ly, lz
