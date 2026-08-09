# SUMMARY #
A Python interface to [Elk](https://elk.sourceforge.io/), an all-electron
full-potential linearized augmented-plane-wave (FP-LAPW) code that solves the
Kohn-Sham equations of density-functional theory,
$$ \Big[-\tfrac12\nabla^2+v_{\rm eff}(\mathbf r)\Big]\psi_{i\mathbf k}(\mathbf r)=\epsilon_i(\mathbf k)\,\psi_{i\mathbf k}(\mathbf r). $$
$v_{\rm eff}=v_{\rm ext}+v_H[n]+v_{xc}[n]$ is solved self-consistently in the electron
density $n(\mathbf r)=\sum_{i\mathbf k}^{\rm occ}|\psi_{i\mathbf k}(\mathbf r)|^2$. elkpy
wraps Elk's `elk.in`/task-number workflow in a small, `pyqula`-style object model
(`Structure`, `Calculation`), and adds physics Elk itself does not provide on top:
per-species spin-orbit coupling scaling, the full quantum geometric tensor (Berry
curvature/Chern numbers and the quantum metric) via a Wilson-loop method, fast
eigenstate/wavefunction-overlap queries at arbitrary k-points, and atom-projection and
spin operators applicable to those wavefunctions.

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

## Spin-orbit coupling ##
- Per-species scaling of the spin-orbit term $\hat H_{\rm soc}(r)=f_{\rm soc}(r)\,\hat{\mathbf L}\cdot\boldsymbol\sigma$, rather than one global scale for the whole cell

## Topological characterization ##
- Berry curvature $F_{12}(\mathbf k)=\partial_1A_2-\partial_2A_1$ and Chern numbers $c_n=\frac1{2\pi i}\int_{T^2}\!d^2k\,F_{12}\in\mathbb Z$, via a gauge-invariant Wilson-loop discretization
- Berry curvature at an arbitrary k-point with no periodic mesh required, e.g. to resolve individual valleys of a 2D material

## Quantum geometry ##
- The full quantum geometric tensor $Q_{ab}=g_{ab}-\tfrac i2F_{ab}$ at an arbitrary k-point: Berry curvature $F_{ab}$ *and* the quantum metric $g_{ab}$ (Fubini-Study distance between neighbouring Bloch states), from the same wavefunction-overlap queries used for eigenstates below

## Eigenstates and wavefunction overlaps ##
- Second-variational energies and eigenvectors at an arbitrary k-point
- Wavefunction overlaps $O_{ab}(\mathbf k_a,\mathbf k_b)=\langle\psi_a(\mathbf k_a)|\psi_b(\mathbf k_b)\rangle$ between two arbitrary k-points, queried interactively
- Atom-projection operators $(P_\alpha)_{ij}=\langle\psi_i|\hat P_\alpha|\psi_j\rangle$ (muffin-tin restriction of the identity, $\sum_\alpha P_\alpha+P_{\rm interstitial}=\mathbb 1$), applicable to any wavefunction in the same band window
- Spin operators $S_x,S_y,S_z$ (eigenvalues $\pm\tfrac12$) as Hermitian matrices in a band window, applicable to any wavefunction the same way

## Ground-state electronic structure ##
- Self-consistent total energy $E[n]$, band structure $\epsilon_i(\mathbf k)$, density of states
- Hellmann-Feynman forces, structural relaxation, effective mass tensor, charge density $n(\mathbf r)$
- Phonon dispersion and density of states via density functional perturbation theory

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

## Quantum metric alongside Berry curvature, along Gamma-K-M-K'-Gamma of monolayer h-BN ##
Time-reversal symmetry requires $g_{ab}(K)=g_{ab}(K')$ even though $\Omega(K')=-\Omega(K)$ --
a genuinely new prediction about the metric, not derivable from curvature alone:
```python
result = hbn.get_quantum_geometry(path, ist0, ist1, directions=(1, 2), dk=0.01)
result[0]["g"]                # (2,2) quantum metric [[g11,g12],[g12,g22]], Bohr^2
result[0]["berry_curvature"]  # Bohr^-2, same convention as get_berry_curvature_path
```
![Alt text](images/hbn_quantum_geometry.png?raw=true "Quantum metric and Berry curvature of monolayer h-BN along Gamma-K-M-K'-Gamma")

## Eigenstates and wavefunction overlaps at arbitrary k-points ##
```python
with calc.eigenstate_session() as session:              # one warm Elk process
    state = session.get_eigenstates((0.1, 0.2, 0.05))   # H(k) c = E S(k) c
    m = session.overlap((0, 0, 0), (0.1, 0, 0), ist0=1, ist1=4)  # <psi_a(k_a)|psi_b(k_b)>
```

## Atom-projection operators: N vs. B character of monolayer h-BN ##
The more electronegative N pulls the bonding (valence-top, $\pi$) state's weight toward
itself; the antibonding (conduction-bottom, $\pi^*$) state is B-dominated instead:
```python
K = (1 / 3, 1 / 3, 0)
proj = hbn.get_atom_projection(K, ist0=ist1, ist1=ist1 + 1)  # valence top, conduction bottom
n, b = hbn.structure.atom_index("N"), hbn.structure.atom_index("B")
proj.matrices[n][0, 0].real  # N's weight on the valence-top band -- large
proj.matrices[b][1, 1].real  # B's weight on the conduction-bottom band -- large
```
![Alt text](images/hbn_atom_projection.png?raw=true "Atom-projected muffin-tin weight of monolayer h-BN's valence-top and conduction-bottom bands")

## Spin operators: spin-valley locking in monolayer WSe2 ##
Broken inversion symmetry plus strong spin-orbit coupling locks the valence-band-top spin
to the valley index: $S_z(K)=-S_z(K')$ (Xiao, Liu, Feng, Xu & Yao, PRL 108, 196802 (2012)):
```python
K, Kprime = (1 / 3, 1 / 3, 0), (-1 / 3, -1 / 3, 0)
with wse2.eigenstate_session() as session:
    sz_k = session.spin_operator(K, ist1, ist1).sz[0, 0].real          # valence-band top
    sz_kprime = session.spin_operator(Kprime, ist1, ist1).sz[0, 0].real
```
![Alt text](images/wse2_spin_valley.png?raw=true "Sz of monolayer WSe2's valence-band-top state at K and K'")

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
Nine notebooks under [`notebooks/`](notebooks), one per feature area above, each
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
| [`07_quantum_geometry.ipynb`](notebooks/07_quantum_geometry.ipynb) | Quantum metric and Berry curvature along Gamma-K-M-K'-Gamma of monolayer h-BN | yes |
| [`08_atom_projection.ipynb`](notebooks/08_atom_projection.ipynb) | Atom-projection operators, N/B character of monolayer h-BN | yes |
| [`09_spin_operators.ipynb`](notebooks/09_spin_operators.ipynb) | Spin operators, spin-valley locking in monolayer WSe2 | yes |
| [`01_getting_started.ipynb`](notebooks/01_getting_started.ipynb) | Ground state, band structure, density of states | -- |
| [`02_relaxation_forces_and_properties.ipynb`](notebooks/02_relaxation_forces_and_properties.ipynb) | Forces, relaxation, effective mass, density, `run_tasks()` | -- |
| [`03_phonon_dispersion_and_dos.ipynb`](notebooks/03_phonon_dispersion_and_dos.ipynb) | Phonon dispersion/DOS via DFPT | -- |

New notebooks should be added here alongside any new physics capability.
