# Fragment: named wrappers for tasks 43, 53 and 122

Doc prose for merging into `docs/design.md` (a new numbered section, or an
extension of the volumetric/plot3d discussion), `docs/physics.tex` (the MOKE
part below is written as a `\part{}` draft), `README.md`'s `FUNCTIONALITIES`
list and `CLAUDE.md`'s project status. Written by the agent that added
`get_potential()`, `get_elf()` and `get_moke()`; not part of the published
docs.

**Still outstanding when merging:** a notebook per new capability (README
`FUNCTIONALITIES` bullets link to one), per the README/notebook style rules.
A single "volumetric quantities" notebook (density, Kohn-Sham potential,
ELF, plotted as 2D slices through the Si bond) plus a MOKE spectrum notebook
(Kerr rotation and ellipticity vs photon energy for the two magnetization
directions) would cover all three.

Also update `docs/roadmap.md` Tier 3 item 4, which currently says potential
(43) and ELF (53) "are one `_run_resumed` call away using the same parser but
don't have named `get_*` methods yet" -- they now do. And `CLAUDE.md`'s
"Not implemented" list, which mentions "named `get_*` methods for
potential/ELF volumetric plots (reachable via `run_tasks()` +
`parsers.volumetric`)".

## 1. Kohn-Sham potential and ELF as 3D plots (tasks 43, 53)

Tasks 33 (density, already wrapped as `get_density()`), 43 (potential) and
53 (ELF) all end in the same writer, `vendor/elk/src/plot3d.f90`, and take
the same `plot3d` input block -- a parallelepiped given as an origin plus
three corner vectors in lattice coordinates, and a grid size. So the
Python side needs no new parser at all: `parsers/volumetric.py::parse_plot3d`
already reads the format, and the three methods now share a
`Calculation._plot3d_lines()` helper that builds the block.

Task numbers and filenames verified in `vendor/elk/src/elk.f90`'s dispatch
(`case(31,32,33) -> rhoplot`, `case(41,42,43) -> potplot`,
`case(51,52,53) -> elfplot`) and in the `open(50, file=...)` statements of
the subroutines themselves, not from the manual.

### Potential (task 43)

`potplot.f90`'s `case(43)` branch writes **two** files from a single run:

* `VCL3D.OUT` -- the electrostatic (Coulomb) potential `v_C`, i.e. the
  nuclear plus Hartree terms, from `vclmt`/`vclir`.
* `VXC3D.OUT` -- the exchange-correlation potential
  `v_xc = delta E_xc / delta n`, from `vxcmt`/`vxcir`.

Their sum is the Kohn-Sham effective potential entering
`(-1/2 grad^2 + v_C + v_xc) psi_i = eps_i psi_i`. This is the one place the
new methods do not map one-to-one onto `get_density()`'s shape: a single
task writes two distinct fields. `get_potential(..., component=)` selects
which of the two is parsed ("coulomb" by default, or "xc") and keeps
`get_density()`'s `(points, values)` return contract; asking for the other
component re-runs the task, which is cheap (no SCF beyond the resumed
ground state, just `readstate` + `plot3d`) and was judged preferable to a
method whose return arity changes with an argument.

### ELF (task 53)

`elfplot.f90` writes `ELF3D.OUT`. The (spin-averaged) electron localization
function is

    f_ELF(r) = 1 / (1 + [D(r)/D0(r)]^2)

with

    D(r)  = (1/2) ( tau(r) - (1/4)|grad n(r)|^2 / n(r) )
    D0(r) = (3/5) (6 pi^2)^(2/3) (n(r)/2)^(5/3)
    tau(r) = sum_i |grad psi_i(r)|^2

(the docstring of `elfplot.f90` itself; Becke and Edgecombe, J. Chem. Phys.
92, 5397 (1990); the reference Elk cites is Burnus, Marques and Gross, PRA
71, 010501 (2005)). `D` is the excess of the local kinetic energy density
over its von Weizsaecker (single-orbital) value, i.e. the Pauli-principle
contribution; `D0` is the same quantity for the homogeneous electron gas at
the local density. `f_ELF` is therefore dimensionless and bounded to [0, 1]
by construction: 1 means an electron pair is perfectly localized (a
covalent bond, a lone pair, a closed shell), 1/2 reproduces the homogeneous
electron gas, and 0 marks the delocalized limit.

Caveat worth repeating from Elk's own example (`examples/ELF/BN`): the ELF
depends on density gradients and is not continuous across the muffin-tin
boundaries at default cut-offs, so a plot can show a visible sphere-boundary
seam that is a basis-set artefact rather than physics. Raising `rgkmax`,
`gmaxvr`, `lmaxo`, `lmaxapw` (via `extra_blocks`) or `highq=.true.` smooths it.

### How to use in code

```python
from elkpy.structure import Structure

si = Structure([(5.13, 5.13, 0.0), (5.13, 0.0, 5.13), (0.0, 5.13, 5.13)],
               {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]})
calc = si.get_calculation("si", xc="PW", ngridk=(4, 4, 4))

# Cartesian grid points (Bohr) and the field sampled on them
points, rho = calc.get_density(grid=(20, 20, 20))
points, v_c = calc.get_potential(grid=(20, 20, 20))                  # VCL3D.OUT
points, v_xc = calc.get_potential(grid=(20, 20, 20), component="xc") # VXC3D.OUT
points, elf = calc.get_elf(grid=(20, 20, 20))

# a plane through the Si-Si bond instead of the whole cell: origin + two
# in-plane vectors + a degenerate third, in lattice coordinates
box = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)]
points, elf_slice = calc.get_elf(box=box, grid=(60, 60, 1))
```

### What was verified against a real binary

Bulk Si, `ngridk=(2,2,2)`, 4x4x4 plot grid (`tests/test_calculation_si.py`):

* `get_potential()` returns both components on the same grid as
  `get_density()`, and they are genuinely different fields (catching a
  filename/spec mix-up that returned the same file twice).
* **Sharp cross-check, no Elk-internal reference needed:** `xc="PW"` is a
  *local* density functional, so `v_xc(r)` must be a pointwise function of
  `n(r)` alone. Measured against the analytic Dirac exchange potential
  `v_x = -(3n/pi)^(1/3)` on the identical grid, `v_xc/v_x` lies in
  1.14-1.23 across this cell's density range (0.006-0.085 e/Bohr^3) -- the
  extra 14-23% being the Perdew-Wang correlation potential. This ties two
  separate Elk runs' output files (task 33's `RHO3D.OUT`, task 43's
  `VXC3D.OUT`) to an analytic formula; a shifted column, a wrong file or a
  unit error breaks it immediately, whereas a shape/sign check would not.
* ELF lies in [0, 1] everywhere (a hard bound of the formula, not a
  tolerance), and reaches 0.94 in the Si-Si bonding region -- a covalent
  crystal must depart strongly from the homogeneous-gas value of 1/2
  somewhere in the cell.

Note for anyone reading absolute values near a nucleus: the plot3d output
is an interpolation onto the requested Cartesian grid, and a grid point
landing exactly on an atom does *not* show the divergent `-Z/r` Coulomb
potential or the enormous core density (measured on the Si cell: `RHO3D`
peaks at 0.085 e/Bohr^3, in the bond, not at the nucleus). Treat these
plots as valence/bonding-scale visualizations.

## 2. Magneto-optic Kerr effect (task 122)

`get_moke()` wraps task 122 (`vendor/elk/src/moke.f90`, dispatched at
`case(122)` in `elk.f90`), returning `(energies, kerr)` -- the photon-energy
grid in Hartree and the **complex Kerr angle in degrees**, whose real part
is the Kerr rotation `theta_K` and whose imaginary part is the Kerr
ellipticity `eta_K`. Output file `KERR.OUT`, format read off `moke.f90`'s
own write statements: two blank-line-separated blocks of `(2G18.10)` pairs,
real part then imaginary part, on a shared energy grid -- parsed by the new
`parsers/moke.py`.

### The physics

Linearly polarized light reflected from a magnetized surface comes back
elliptically polarized, with its major axis rotated. The effect is
first-order in the magnetization and vanishes without spin-orbit coupling:
it needs the *off-diagonal* conductivity `sigma_xy`, which is odd under time
reversal, and only SOC ties the (time-reversal-odd) spin magnetization to
the orbital motion the light actually couples to.

`moke.f90` calls `dielectric` internally with `optcomp` fixed to the 11 and
12 components, obtaining `sigma_xx` and `sigma_xy` from the Kubo formula
implemented in `dielectric.f90` (Physica Scripta T109, 170 (2004)), and then
forms the standard polar-Kerr expression

    theta_K + i eta_K = - sigma_xy / ( sigma_xx sqrt(1 + 4 pi i sigma_xx / omega) )

(atomic units; `moke.f90` writes both parts multiplied by 180/pi, i.e. in
degrees). The square-root factor is the refractive-index denominator of the
Fresnel reflection coefficients for the two circular polarizations. Elk
returns exactly zero at `omega = 0`, where the expression is singular.

Task 122 needs the momentum matrix elements on disk (`dielectric` reads
`PMAT.OUT` via `getpmat`), so `get_moke()` runs task 120
(`writepmat.f90`) first in the same directory. That pairing is precisely
what makes this worth a named method rather than a bare `run_tasks()` call.

Note the interaction with `dielectric` (task 121, wrapped separately): task
122 *also* writes `SIGMA_11.OUT`, `SIGMA_12.OUT`, `EPSILON_11.OUT` and
`EPSILON_12.OUT` as a side effect, since it calls the same subroutine. Both
wrappers run in their own wiped `_run_resumed()` subdirectory, so they
cannot overwrite each other's output; but a caller who wants both the
conductivity tensor and the Kerr angle pays for the response function twice.

### Smearing and convergence

`swidth` (Hartree) sets the smearing whose reciprocal is the relaxation time
entering the response function. Elk's own example
(`examples/TDDFT-optics/Ni-MOKE`) raises it *after* the ground-state run to
smooth the spectrum, warning that a large smearing during the SCF cycle
suppresses the moment -- which is exactly what passing `swidth=` to
`get_moke()` does, since it runs resumed from an already-converged
`STATE.OUT` and never feeds `swidth` back into the ground state.

The Kerr angle is a Brillouin-zone integral over interband transitions and
converges slowly in `ngridk` (Elk's Ni example uses 32x32x32). At a coarse
mesh the spectrum is dominated by individual transition spikes and the
near-singular `sigma_xx` denominator, which inflates the peak values by
orders of magnitude: measured on fcc Ni, `ngridk=(4,4,4)` with default
smearing gives peaks of tens of degrees, while `ngridk=(10,10,10)` with
`swidth=0.01` brings them down to the ~0.01-0.1 degree scale of a real Ni
Kerr spectrum. Treat a coarse-mesh result as a symmetry/sign probe, not a
spectrum.

### How to use in code

```python
from elkpy.structure import Structure

# vendor/elk/examples/TDDFT-optics/Ni-MOKE: ferromagnetic fcc Ni
ni = Structure([(1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)],
               {"Ni": [(0.0, 0.0, 0.0)]}, scale=3.33)
calc = ni.get_calculation(
    "ni", xc="PW", spinpol=True, spinorb=True, ngridk=(8, 8, 8),
    extra_blocks={"bfieldc": [(0.0, 0.0, 0.01)]},  # magnetize along +z
)

# energies in Hartree, kerr complex in degrees
energies, kerr = calc.get_moke(wplot=(0.0, 0.5), nwplot=500,
                               swidth=0.01, ngridk=(16, 16, 16))
theta_K, eta_K = kerr.real, kerr.imag
```

`get_moke()` raises `ValueError` before launching anything if
`spinorb=False` or `spinpol=False`: `sigma_xy` is then identically zero by
symmetry, so the run could only ever return a flat zero.

### What was verified against a real binary

`tests/test_calculation_moke.py`, fcc Ni (Elk's own MOKE example structure)
at `ngridk=(4,4,4)`, ~50 s for the whole file:

* The guards fire without running Elk (no SOC / no magnetization).
* Shapes/dtype, the energy grid (starts at 0 -- `dielectric.f90` clips a
  negative `wplot(1)` to zero), and `kerr[0] == 0` exactly.
* **Sign of the effect:** the Kerr angle is odd under reversal of the
  magnetization. Two *independent* ground states (`bfieldc` along +z and
  -z), each with its own momentum matrix elements and conductivity tensor,
  sharing no arithmetic, give spectra that agree in magnitude to 0.6% and
  cancel to 0.9% of the peak when added:
  `max|kerr(+M) + kerr(-M)| = 0.0088 max|kerr(+M)|`. This is the check that
  distinguishes a genuine Kerr response from anything even under time
  reversal (e.g. `|sigma_xy|`), which a magnitude-only assertion would pass
  just as happily.

One convention rests on a source read rather than a runtime test: which of
`KERR.OUT`'s two blocks is the real part. `moke.f90` writes `dble(kerr)`
first and `aimag(kerr)` second, and `parsers/moke.py` follows that order --
but every check above (oddness under M reversal, `kerr[0] == 0`, magnitude
agreement) is invariant under swapping the two, so none of them would catch
a flip. Same class of source-derived convention as the Fortran conjugation
sign in the Berry-curvature work. A future denser-mesh comparison against a
published Ni Kerr spectrum would pin it empirically.

Not verified: the absolute spectrum against experiment or published DFT --
that needs a k-mesh well beyond what these tests can afford, and the coarse
mesh inflates peak magnitudes as described above.
