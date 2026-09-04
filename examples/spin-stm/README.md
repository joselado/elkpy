# Spin-polarised STM of a non-collinear Cr monolayer

Runs directly from `elk.in`, the same way any example under
`vendor/elk/examples/` does — no Python needed:

```bash
cd examples/spin-stm
../../build/elk/src/elk        # ~1 minute
python3 plot_stm.py            # optional, writes stm.png
```

It needs an `elk` built by `build_elk.sh`, which applies
`patches/0011-spin-polarized-stm.patch` — tasks 9003/9004 and the
`elkpy_stmdir` / `elkpy_stmpol` / `elkpy_stmbias` / `elkpy_stmint` blocks are
elkpy extensions, not stock Elk.

## What it computes

Within the Tersoff-Hamann picture the current a magnetic tip draws is set by
the sample's vacuum local density of states at the tip position, projected
onto the tip magnetisation (Wortmann *et al.*, PRL **86**, 4132 (2001)):

```
dI/dV(r)  ~  n(r, E_F + eV)  +  P_T m(r, E_F + eV) . e_T
```

Upstream Elk's task 162 plots `n` alone. Task 9003 plots three fields per
point — `n`, `m . e_T`, and the combination above — so the spin-averaged and
spin-polarised images sit side by side in one file.

The system is a freestanding Cr monolayer on the triangular lattice of
Cr/Ag(111), which orders in a coplanar 120° Néel state. Its three Cr atoms
are chemically identical, so:

- `n` is the same above all three — a conventional STM sees a 1×1 lattice;
- `m . e_T` is not — one sublattice is dark and the other two have opposite
  sign for a tip along **x**, and the pattern changes when the tip is
  rotated. That is the magnetic √3×√3 superstructure.

Things worth changing:

- `elkpy_stmdir` — the tip direction, in Cartesian coordinates. Try
  `0.0 1.0 0.0` (a different sublattice goes dark) or `0.0 0.0 1.0` (the
  moments are coplanar in *xy*, so the image is identically zero).
- the third number in each `plot2d` line — the tip height, as a fraction of
  the 24 Bohr *c*-axis. The vacuum LDOS decays exponentially, and the
  magnetic superstructure decays *more slowly* than the atomic corrugation,
  so the spin contrast grows relative to it as the tip is retracted.
- `elkpy_stmbias` — the sample bias in Hartree. A positive bias probes empty
  states, so raise `nempty` to cover the window. Add `elkpy_stmint` /
  `.true.` to integrate over the whole window (constant-current mode)
  instead of sampling at one energy.

The same calculation from Python is `Calculation.get_spin_stm()`; see
`notebooks/19_spin_polarized_stm.ipynb`.
