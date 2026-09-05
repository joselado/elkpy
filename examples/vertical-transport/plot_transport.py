"""Contract ELKPY_TRANSPORT.OUT into a transmission map and plot it.

`elk` (task 9005, run in this directory) writes the two ingredients: the
exit-plane Gram matrix S_k and the tip amplitudes psi_kn(r_p), for every state
inside the exported energy window. What is left is the energy weighting and
one quadratic form per k-point,

    T(r; E) = sum_k w_k sum_nm a_n(r) a*_m(r) S_k[n, m],
    a_n(r)  = psi_kn(r) sqrt(occmax delta_eta(E - eps_kn) / eta)

which elkpy.parsers.transport does. Setting exit_region="cell" replaces every
S_k by the identity, which is exactly the Tersoff-Hamann limit -- the local
density of states an STM tip would see at the same plane. For a single sheet
the two maps are the same picture, which is the point of the example.
"""

import matplotlib.pyplot as plt
import numpy as np

from elkpy.parsers import transport

data = transport.parse_transport("ELKPY_TRANSPORT.OUT")
n2, n1 = data["grid"]
x, y = (data["points"][:, i].reshape(n2, n1) for i in (0, 1))

# eta is the width of the energy window the leads let states through in -- the
# leads' own coupling, not a numerical smearing
flow = transport.compute_transmission(data, broadening=0.005)
local = transport.compute_transmission(data, broadening=0.005,
                                       exit_region="cell", incoherent=False)

fields = [
    (flow["transmission"][0], r"$T(\mathbf{r})$  through the sheet"),
    (local["transmission"][0], r"local density of states (Tersoff-Hamann)"),
]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
for ax, (f, label) in zip(axes, fields):
    im = ax.pcolormesh(x, y, f.reshape(n2, n1) / f.max(), shading="gouraud",
                       cmap="afmhot")
    fig.colorbar(im, ax=ax)
    ax.set(title=label, xlabel="x (Bohr)", aspect="equal")
axes[0].set_ylabel("y (Bohr)")
fig.savefig("transport.png", dpi=150)

correlation = np.corrcoef(flow["transmission"][0], local["transmission"][0])[0, 1]
print(f"open substrate channels        : {flow['channels']:.4f}")
print(f"weight off S_k's diagonal      : {flow['offdiagonal_weight']:.2e}")
print(f"correlation with the STM image : {correlation:.6f}")
print("wrote transport.png")
