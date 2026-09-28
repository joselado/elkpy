"""Parse ELKPY_ROTMOM.OUT, the orientation gradient of every self-consistent
loop (src/elkpy_rotmom.f90, patch 0027 -- docs/design.md #36).

One line per loop of the ground state: the gradient of the total energy with
respect to one global rotation of the spin frame, ``dE/dw = int B_in x m_out``
(Hartree per radian), what the stepper acts on after the turns that move
nothing are taken out, and, when the moments are being turned, the step and
the angle turned so far. The torque on the texture is ``-dE/dw``.
"""

import numpy as np


def parse_orientation(path):
    """Return a dict with one row per self-consistent loop.

    Keys:

    - "loop" (n,): Elk's loop number ``iscl``.
    - "gradient" (n, 3): ``dE/dw`` in Ha/rad, Cartesian, for the
      exchange-correlation field alone (the external fields' share is taken
      out).
    - "acting" (n,): ``|dE/dw|`` after projecting out the axis of a collinear
      texture (and the plane normal of a coplanar one unless its phase is
      relaxed) -- the number compared with the tolerance.
    - "field_bound" (n,): an upper bound, Ha/rad, on the torque the seed
      fields can still exert; no step is taken until it is below a tenth of
      the tolerance.
    - "dv" (n,): the RMS change of the Kohn-Sham potential, as in RMSDVS.OUT.
    - "step" (n, 3): the rotation vector applied after that loop, radians
      (zero when no step was taken).
    - "angle" (n,): the total angle turned so far, radians.
    - "axis" (n, 3): principal axis of the muffin-tin moment tensor
      ``sum_a m_a m_a^T`` of the loop's output magnetisation, signed along the
      total moment.
    - "normal" (n, 3): its smallest-eigenvalue axis, the plane normal of a
      coplanar texture.
    - "texture" (n,): 1 collinear, 2 coplanar, 3 non-coplanar, 0 no moment.
    - "stepped" (n,) bool.
    - "rotating" bool: whether the moments were being turned (``elkpy_rotmom``)
      or the gradient was only reported (``elkpy_torque``).
    - "parameters": dict with "tolerance", "trust", "first_step", "start"
      and "hold_phase" when rotating, else empty.
    """
    rows = []
    rotating = False
    parameters = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                if "moments turned: yes" in line:
                    rotating = True
                elif line.startswith("# parameters"):
                    numbers = [float(x) for x in line.split(":", 1)[1].split()]
                    parameters.update(zip(("tolerance", "trust", "first_step", "start"), numbers))
                elif "phase of a coplanar texture held" in line:
                    parameters["hold_phase"] = line.strip().endswith("T")
                continue
            if line.strip():
                rows.append(line.split())
    if not rows:
        empty3 = np.zeros((0, 3))
        return {
            "loop": np.zeros(0, dtype=int), "gradient": empty3, "acting": np.zeros(0),
            "field_bound": np.zeros(0), "dv": np.zeros(0), "step": empty3,
            "angle": np.zeros(0), "axis": empty3, "normal": empty3,
            "texture": np.zeros(0, dtype=int), "stepped": np.zeros(0, dtype=bool),
            "rotating": rotating, "parameters": parameters,
        }
    data = np.array([[float(x) for x in row] for row in rows])
    return {
        "loop": data[:, 0].astype(int),
        "gradient": data[:, 1:4],
        "acting": data[:, 4],
        "field_bound": data[:, 5],
        "dv": data[:, 6],
        "step": data[:, 7:10],
        "angle": data[:, 10],
        "axis": data[:, 11:14],
        "normal": data[:, 14:17],
        "texture": data[:, 17].astype(int),
        "stepped": data[:, 18].astype(bool),
        "rotating": rotating,
        "parameters": parameters,
    }
