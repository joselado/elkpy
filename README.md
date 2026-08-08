# SUMMARY #
A Python interface to [Elk](https://elk.sourceforge.io/), an all-electron
full-potential linearized augmented-plane-wave (FP-LAPW) code that solves the
Kohn-Sham equations of density-functional theory,
$$ \Big[-\tfrac12\nabla^2+v_{\rm eff}[n](\mathbf r)\Big]\psi_{i\mathbf k}(\mathbf r)=\epsilon_i(\mathbf k)\,\psi_{i\mathbf k}(\mathbf r), \qquad v_{\rm eff}=v_{\rm ext}+v_H[n]+v_{xc}[n], $$
self-consistently in the electron density $n(\mathbf r)=\sum_{i\mathbf k}^{\rm occ}|\psi_{i\mathbf k}(\mathbf r)|^2$. elkpy wraps Elk's `elk.in`/task-number workflow in a small, `pyqula`-style object model (`Structure`, `Calculation`), and adds physics Elk itself does not provide on top: per-species spin-orbit coupling scaling, Berry curvature/Chern numbers via a Wilson-loop method, and fast eigenstate/wavefunction-overlap queries at arbitrary k-points.

```python
from elkpy.structure import Structure

avec = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
species = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}
calc = Structure(avec, species).get_calculation("run/si", xc="PW", ngridk=(4, 4, 4))

energy = calc.get_energy()                      # Hartree
(k, e) = calc.get_bands(kpath="GXWLGK")         # epsilon_i(k)
```

# INSTALLATION #

```bash
# 1. Build Elk out-of-tree (vendor/elk/ -> build/elk/, applies patches/*.patch,
#    never touches vendor/elk/ itself)
./scripts/build_elk.sh

# 2. Install elkpy (editable)
python3 -m pip install -e .        # add .[ase] for Structure.from_ase()/to_ase()
```

# FUNCTIONALITIES #
elkpy's own contribution is physics **beyond** what Elk provides upstream, added as a
small, additive Fortran patch series (`patches/`) plus Python arithmetic on top --
never a modification of `vendor/elk/` itself. The three capabilities below don't
exist in stock Elk at all.

## Per-species spin-orbit coupling scaling ##
Elk's spin-orbit term (Koelling-Harmon approximation) is
$$ \hat H_{\rm soc}(r)=f_{\rm soc}(r)\,\hat{\mathbf L}\cdot\boldsymbol\sigma $$
scaled upstream by one single global number, `socscf`, applied identically to every
atom regardless of species. elkpy promotes that to an independent scale per chemical
species, $\texttt{socscf}\to s_{is}$, so a correction fitted for one (typically heavy)
element doesn't leak onto a lighter one sharing the cell.

## Topological characterization: Berry curvature and Chern numbers ##
Elk has no Berry-curvature capability at all. elkpy adds the gauge-invariant
Wilson-loop (Fukui-Hatsugai-Suzuki) discretization of
$$ F_{12}(\mathbf k)=\partial_1A_2(\mathbf k)-\partial_2A_1(\mathbf k), \qquad c_n=\frac1{2\pi i}\int_{T^2}\!d^2k\;F_{12}(\mathbf k)\in\mathbb Z, $$
in two modes: over Elk's full periodic k-mesh (a genuine, topologically quantized
Chern number), or at an arbitrary, explicit k-point via fresh on-the-fly
diagonalization (no periodic mesh required at all) -- e.g. to resolve the curvature
of one valley of a 2D material without a mesh covering the whole zone.

## Eigenstates and wavefunction overlaps at arbitrary k-points ##
Elk can diagonalize at an arbitrary off-mesh k-point internally (e.g. along a band
path), but exposes no way to retrieve `evecsv` or a cross-k wavefunction overlap
programmatically, nor to amortize the ground-state-dependent setup cost across many
such queries. elkpy adds a persistent, interactive Elk session that returns
second-variational energies/eigenvectors and the wavefunction overlap
$$ O_{ab}(\mathbf k_a,\mathbf k_b)=\langle\psi_a(\mathbf k_a)|\psi_b(\mathbf k_b)\rangle $$
between two arbitrary k-points, on demand, staying warm across repeated queries.

## Also wraps Elk's standard DFT workflow ##
Total energy $E[n]$, band structure $\epsilon_i(\mathbf k)$, density of states,
Hellmann-Feynman forces and structural relaxation, effective mass tensor, real-space
charge density $n(\mathbf r)$, phonon dispersion/DOS via DFPT, and a `run_tasks()`
escape hatch to any other standard Elk task -- all through the same `Structure`/
`Calculation` object model as the capabilities above (see notebooks 01-03).

# EXAMPLES #
Full worked notebooks (formulas + code + real Elk output) are in [`notebooks/`](notebooks); short examples below, in the same order as FUNCTIONALITIES.

## Per-species spin-orbit coupling scaling ##
```python
from elkpy.structure import Structure

avec = [(10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)]
species = {"Bi": [(0.0, 0.0, 0.0)], "Si": [(0.5, 0.5, 0.5)]}
calc = Structure(avec, species).get_calculation(
    "run/bisi", xc="PW", spinorb=True, ngridk=(1, 1, 1),
    soc_scale={"Bi": 0.0},   # scale only Bi's spin-orbit term; Si unaffected
)
energy = calc.get_energy()
```

## K/K' valley Berry curvature of monolayer h-BN ##
Broken B/N sublattice inversion symmetry makes the K and $K'=-K$ valleys physically
inequivalent; time-reversal symmetry then requires $\Omega(K')=-\Omega(K)$ -- checked
directly at an arbitrary k-point, no periodic mesh needed:
```python
from elkpy.structure import Structure

avec = [(4.74321, 0.0, 0.0), (-2.37161, 4.10774, 0.0), (0.0, 0.0, 20.0)]
species = {"B": [(1 / 3, 2 / 3, 0.5)], "N": [(2 / 3, 1 / 3, 0.5)]}
hbn = Structure(avec, species).get_calculation("run/hbn", xc="PW", ngridk=(6, 6, 1))
hbn.get_energy()

omega_K = hbn.get_berry_curvature_path([(1/3, 1/3, 0)], 1, 4, dk=0.01)[0]["curvature"]
```
![Alt text](images/hbn_berry_curvature.png?raw=true "Berry curvature of monolayer h-BN along Gamma-K-M-Gamma")

## Eigenstates and wavefunction overlaps at arbitrary k-points ##
```python
with calc.eigenstate_session() as session:              # one warm Elk process
    state = session.get_eigenstates((0.1, 0.2, 0.05))   # H(k) c = E S(k) c
    m = session.overlap((0, 0, 0), (0.1, 0, 0), ist0=1, ist1=4)  # <psi_a(k_a)|psi_b(k_b)>
```

## Also: Elk's standard DFT workflow (band structure, DOS, charge density) ##
```python
from elkpy.structure import Structure

avec = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
species = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}
calc = Structure(avec, species).get_calculation("run/si", xc="PW", ngridk=(2, 2, 2))

(k, e) = calc.get_bands(kpath="GXWLGK")                # epsilon_i(k), Bloch's theorem
points, density = calc.get_density(grid=(24, 24, 24))  # n(r) = sum_i^occ |psi_i(r)|^2
```
![Alt text](images/si_bands.png?raw=true "Band structure of bulk silicon")
![Alt text](images/si_dos.png?raw=true "Density of states of bulk silicon")
![Alt text](images/si_density.png?raw=true "Charge density slice of bulk silicon")

# Notebooks #
Six notebooks under [`notebooks/`](notebooks), one per feature area above, each
executed end-to-end against a real compiled Elk binary and checked in with its actual
output (the DFPT phonon notebook is the exception -- left unexecuted with a note,
since a single call takes ~11-13 minutes). Listed new-physics-first, matching
FUNCTIONALITIES/EXAMPLES above; if you're new to elkpy, `01_getting_started.ipynb`
is the place to actually start:

| Notebook | Feature | Beyond Elk? |
| --- | --- | --- |
| [`04_per_species_soc_scaling.ipynb`](notebooks/04_per_species_soc_scaling.ipynb) | Per-species spin-orbit coupling scaling | yes |
| [`05_berry_curvature.ipynb`](notebooks/05_berry_curvature.ipynb) | Berry curvature/Chern number, K/K' valleys of monolayer h-BN | yes |
| [`06_eigenstate_session.ipynb`](notebooks/06_eigenstate_session.ipynb) | Eigenstates and wavefunction overlaps | yes |
| [`01_getting_started.ipynb`](notebooks/01_getting_started.ipynb) | Ground state, band structure, density of states | -- |
| [`02_relaxation_forces_and_properties.ipynb`](notebooks/02_relaxation_forces_and_properties.ipynb) | Forces, relaxation, effective mass, density, `run_tasks()` | -- |
| [`03_phonon_dispersion_and_dos.ipynb`](notebooks/03_phonon_dispersion_and_dos.ipynb) | Phonon dispersion/DOS via DFPT | -- |

New notebooks should be added here alongside any new physics capability.

# Project layout and status #

- `src/elkpy/` -- the Python package (`structure.py`, `calculation.py`, `session.py`,
  `spec.py`, `parsers/`).
- `vendor/elk/` -- vendored Elk 11.0.2 source, unmodified. `patches/` -- the additive
  Fortran patch series (SOC scaling, Berry curvature, eigenstate session) applied to
  an out-of-tree build copy, never to `vendor/elk/` itself.
- `docs/physics.tex` -- the physics writeup for each elkpy Fortran extension (formulas,
  symbols, what's verified against a real compiled binary); `docs/design.md`/
  `docs/roadmap.md` -- architecture and status.
- `tests/` -- unit tests plus integration suites against a real `elk` binary; run
  `python3 -m pytest tests/` after `./scripts/build_elk.sh`.

Roadmap Tiers 1-3 (core DFT workflow) are implemented and verified; Tiers 4-6 remain
planning-only (`docs/roadmap.md`). Not implemented: symbolic k-path's disconnected-
segment support, the classical supercell phonon method, MPI, and named `get_*` methods
for potential/ELF volumetric plots (reachable via `run_tasks()` today). `src/elkpy/`
is the source of truth -- check it directly rather than assuming the docs describe
current behavior.
