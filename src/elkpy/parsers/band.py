"""Parse BAND.OUT / BANDLINES.OUT from task 20 (band structure).

Verified against a real Elk 11.0.2 run (Si, examples/basic/Si, plot1d with
npp1d=200): BAND.OUT is a sequence of blank-line-separated blocks, one per
band, each block a "distance  energy" pair per k-point along the path, same
distance values repeated identically across every block.
"""

import numpy as np


def parse_bands(band_out_path):
    """Return (distances, energies) with distances shape (npoints,) and
    energies shape (nbands, npoints), both in Hartree/Bohr (atomic units,
    Fermi energy already subtracted by Elk)."""
    blocks = []
    current = []
    with open(band_out_path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                if current:
                    blocks.append(current)
                    current = []
                continue
            x, y = stripped.split()
            current.append((float(x), float(y)))
    if current:
        blocks.append(current)

    distances = np.array([p[0] for p in blocks[0]])
    energies = np.array([[p[1] for p in block] for block in blocks])
    return distances, energies


def parse_bandlines(bandlines_out_path):
    """Return the distance-along-path of each high-symmetry vertex, for
    plotting vertical gridlines alongside parse_bands' output."""
    positions = []
    with open(bandlines_out_path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            x, _y = stripped.split()
            positions.append(float(x))
    # each vertex is written as two points (x, ymin) and (x, ymax)
    return sorted(set(positions))
