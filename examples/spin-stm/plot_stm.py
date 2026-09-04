"""Plot ELKPY_STM2D.OUT, written by `elk` in this directory (task 9003).

The file has one header line ("n1 n2 : grid size") and then one line per plot
point: two in-plane Cartesian coordinates in Bohr followed by

    n(r,E)   m(r,E).e_tip   n(r,E) + P_tip m(r,E).e_tip,

the spin-summed vacuum LDOS, its projection onto the tip magnetisation
direction, and the Tersoff-Hamann differential conductance a tip of
polarisation P_tip would measure. The first index runs fastest.
"""

import matplotlib.pyplot as plt
import numpy as np

with open("ELKPY_STM2D.OUT") as fh:
    n1, n2 = (int(v) for v in fh.readline().split()[:2])
    data = np.loadtxt(fh)

x, y = data[:, 0].reshape(n2, n1), data[:, 1].reshape(n2, n1)
labels = [
    r"$n(\mathbf{r},E_F)$  (conventional STM)",
    r"$\mathbf{m}(\mathbf{r},E_F)\cdot\hat{\mathbf{e}}_T$  (spin contrast)",
    r"$n + P_T\,\mathbf{m}\cdot\hat{\mathbf{e}}_T$  (SP-STM)",
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for i, (ax, label) in enumerate(zip(axes, labels)):
    f = data[:, 2 + i].reshape(n2, n1)
    cmap = "bwr" if i == 1 else "inferno"
    vmax = np.abs(f).max() if i == 1 else None
    vmin = -vmax if i == 1 else None
    im = ax.pcolormesh(x, y, f, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    ax.set_title(label)
    ax.set_xlabel("x (Bohr)")
    ax.set_aspect("equal")
axes[0].set_ylabel("y (Bohr)")
fig.savefig("stm.png", dpi=150)
print("wrote stm.png")
