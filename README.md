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

## Ground-state electronic structure ##
- Self-consistent Kohn-Sham total energy $E[n]$, in Elk's FP-LAPW basis
- Band structure $\epsilon_i(\mathbf k)$ and density of states $g(\epsilon)$
- Hellmann-Feynman forces $\mathbf F_I=-\partial E/\partial\mathbf R_I$ and structural relaxation
- Effective mass tensor $(1/m^*)_{ab}=\hbar^{-2}\partial^2\epsilon/\partial k_a\partial k_b$
- Real-space charge density $n(\mathbf r)$
- Escape hatch (`run_tasks()`) to any other standard Elk task (e.g. $\langle\hat L\rangle,\langle\hat S\rangle,\langle\hat J\rangle$ expectation values, potential/ELF volumetric plots)

## Lattice dynamics ##
- Phonon dispersion $\omega_\nu(\mathbf q)$ and density of states via density functional perturbation theory (DFPT), from the dynamical matrix $D_{\kappa\alpha,\kappa'\beta}(\mathbf q)=(M_\kappa M_{\kappa'})^{-1/2}\,\partial^2E/\partial u^*_{\kappa\alpha}(\mathbf q)\partial u_{\kappa'\beta}(\mathbf q)$

## Per-species spin-orbit coupling scaling (elkpy Fortran extension) ##
- Independent scaling of the Koelling-Harmon spin-orbit term $\hat H_{\rm soc}=f_{\rm soc}(r)\hat{\mathbf L}\cdot\boldsymbol\sigma$ per chemical species, generalizing Elk's single global `socscf` scalar

## Topological characterization (elkpy Fortran extension) ##
- Berry curvature $F_{12}(\mathbf k)=\partial_1A_2-\partial_2A_1$ and Chern number $c_n=(2\pi i)^{-1}\int_{T^2}d^2k\,F_{12}$, via the gauge-invariant Wilson-loop (Fukui-Hatsugai-Suzuki) discretization on Elk's periodic k-mesh
- The same construction at an arbitrary, explicit k-point (no periodic mesh required), e.g. to resolve individual valleys of a 2D material

## Eigenstates and wavefunction overlaps (elkpy Fortran extension) ##
- Second-variational energies/eigenvectors at an arbitrary k-point
- Wavefunction overlaps $O_{ab}(\mathbf k_a,\mathbf k_b)=\langle\psi_a(\mathbf k_a)|\psi_b(\mathbf k_b)\rangle$ between two arbitrary k-points, kept fast over many repeated queries via a persistent interactive session

# EXAMPLES #
Full worked notebooks (formulas + code + real Elk output) are in [`notebooks/`](notebooks); short examples below.

## Band structure and density of states of bulk silicon ##
```python
from elkpy.structure import Structure

avec = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
species = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}
calc = Structure(avec, species).get_calculation("run/si", xc="PW", ngridk=(2, 2, 2))
(k, e) = calc.get_bands(kpath="GXWLGK")   # epsilon_i(k), Bloch's theorem
```
![Alt text](images/si_bands.png?raw=true "Band structure of bulk silicon")
![Alt text](images/si_dos.png?raw=true "Density of states of bulk silicon")

## Real-space charge density ##
```python
points, density = calc.get_density(grid=(24, 24, 24))   # n(r) = sum_i^occ |psi_i(r)|^2
```
![Alt text](images/si_density.png?raw=true "Charge density slice of bulk silicon")

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

## Per-species spin-orbit coupling scaling ##
```python
from elkpy.structure import Structure

avec = [(10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)]
species = {"Bi": [(0.0, 0.0, 0.0)], "Si": [(0.5, 0.5, 0.5)]}
calc = Structure(avec, species).get_calculation(
    "run/bisi", xc="PW", spinorb=True, ngridk=(1, 1, 1),
    soc_scale={"Bi": 0.0},   # scale only Bi's spin-orbit term; Si unaffected
)
e = calc.get_energy()
```

## Eigenstates and wavefunction overlaps at arbitrary k-points ##
```python
with calc.eigenstate_session() as session:              # one warm Elk process
    state = session.get_eigenstates((0.1, 0.2, 0.05))   # H(k) c = E S(k) c
    m = session.overlap((0, 0, 0), (0.1, 0, 0), ist0=1, ist1=4)  # <psi_a(k_a)|psi_b(k_b)>
```

# Notebooks #
Six notebooks under [`notebooks/`](notebooks), one per feature area above, each
executed end-to-end against a real compiled Elk binary and checked in with its actual
output (the DFPT phonon notebook is the exception -- left unexecuted with a note,
since a single call takes ~11-13 minutes):

| Notebook | Feature |
| --- | --- |
| [`01_getting_started.ipynb`](notebooks/01_getting_started.ipynb) | Ground state, band structure, density of states |
| [`02_relaxation_forces_and_properties.ipynb`](notebooks/02_relaxation_forces_and_properties.ipynb) | Forces, relaxation, effective mass, density, `run_tasks()` |
| [`03_phonon_dispersion_and_dos.ipynb`](notebooks/03_phonon_dispersion_and_dos.ipynb) | Phonon dispersion/DOS via DFPT |
| [`04_per_species_soc_scaling.ipynb`](notebooks/04_per_species_soc_scaling.ipynb) | Per-species spin-orbit coupling scaling |
| [`05_berry_curvature.ipynb`](notebooks/05_berry_curvature.ipynb) | Berry curvature/Chern number, K/K' valleys of monolayer h-BN |
| [`06_eigenstate_session.ipynb`](notebooks/06_eigenstate_session.ipynb) | Eigenstates and wavefunction overlaps |

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
