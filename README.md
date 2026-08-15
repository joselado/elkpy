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
eigenstate/wavefunction-overlap queries at arbitrary k-points, atom-projection,
orbital-character (s/p/d/f), angular momentum, and spin operators applicable to those
wavefunctions, and optical (velocity) matrix elements with the circular dichroism and
Kubo-form quantum geometry built from them.

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
- Per-species scaling of the spin-orbit term $\hat H_{\rm soc}(r)=f_{\rm soc}(r)\,\hat{\mathbf L}\cdot\boldsymbol\sigma$, rather than one global scale for the whole cell [[notebook]](notebooks/04_per_species_soc_scaling.ipynb)

## Topological characterization ##
- Berry curvature $F_{12}(\mathbf k)=\partial_1A_2-\partial_2A_1$ with $A_\mu=i\langle n|\partial_\mu n\rangle$, and Chern numbers $c_n=\frac1{2\pi}\int_{T^2}\!d^2k\,F_{12}\in\mathbb Z$, via a gauge-invariant Wilson-loop discretization in the standard sign convention [[notebook]](notebooks/05_berry_curvature.ipynb)
- Berry curvature at an arbitrary k-point with no periodic mesh required, e.g. to resolve individual valleys of a 2D material [[notebook]](notebooks/05_berry_curvature.ipynb)
- The $\mathbb Z_2$ invariant $\nu\in\{0,1\}$ of a time-reversal-invariant 2D insulator, via Wannier-charge-center pumping and the non-Abelian Wilson loop $D(k_2)=\prod_iU(F_i)$, distinguishing an ordinary insulator from a quantum spin Hall insulator [[notebook]](notebooks/12_z2_invariant.ipynb)
- The full 3D strong/weak classification $(\nu_0;\nu_1\nu_2\nu_3)$ of a 3D time-reversal-invariant insulator, from the $Z_2$ invariant of each of the Brillouin zone's six time-reversal-invariant planes, distinguishing an ordinary insulator from a strong or weak topological insulator [[notebook]](notebooks/13_z2_invariant_3d.ipynb)
- The same $Z_2$ invariants from parity eigenvalues alone, $\delta_i=\prod_m\xi_{2m}(\Gamma_i)$ with $(-1)^{\nu_0}=\prod_i\delta_i$ — exact and mesh-free, needing only the 8 (3D) or 4 (2D) time-reversal-invariant momenta, for crystals with an inversion centre [[notebook]](notebooks/15_parity_invariants.ipynb)

## Quantum geometry ##
- The full quantum geometric tensor $Q_{ab}=g_{ab}-\tfrac i2F_{ab}$ at an arbitrary k-point: Berry curvature $F_{ab}$ *and* the quantum metric $g_{ab}$ (Fubini-Study distance between neighbouring Bloch states), from the same wavefunction-overlap queries used for eigenstates below [[notebook]](notebooks/07_quantum_geometry.ipynb)
- The same tensor in its Kubo (sum-over-states) form $T_{ab}=\sum_{n\in W,\,m\notin W}\langle n|v_a|m\rangle\langle m|v_b|n\rangle/(\varepsilon_n-\varepsilon_m)^2$, needing no k-derivative at all — an independent route to $g_{ab}=\mathrm{Re}\,T_{ab}$ and $F_{ab}=-2\,\mathrm{Im}\,T_{ab}$ [[notebook]](notebooks/14_optical_matrix_elements.ipynb)

## Optical response ##
- Momentum (velocity) matrix elements $p^a_{nm}=\langle\psi_n|(-i\nabla+\tfrac1{4c^2}[\vec\sigma\times\nabla V_s])_a|\psi_m\rangle$ at an arbitrary k-point — the optical dipole matrix elements, and, for a local Kohn-Sham potential in atomic units, the velocity operator $\hat{\mathbf v}=\hat{\mathbf p}$ [[notebook]](notebooks/14_optical_matrix_elements.ipynb)
- Circular dichroism $\eta=(|P_+|^2-|P_-|^2)/(|P_+|^2+|P_-|^2)$ of an interband transition, $P_\pm=p^x_{cv}\pm ip^y_{cv}$ — the valley-selective optical selection rule of a gapped honeycomb lattice [[notebook]](notebooks/14_optical_matrix_elements.ipynb)
- Band velocities $\mathbf v_n=\partial\varepsilon_n/\partial\mathbf k$ as the diagonal of the same operator, exact by Hellmann-Feynman [[notebook]](notebooks/14_optical_matrix_elements.ipynb)

## Eigenstates and wavefunction overlaps ##
- Second-variational energies and eigenvectors at an arbitrary k-point [[notebook]](notebooks/06_eigenstate_session.ipynb)
- Wavefunction overlaps $O_{ab}(\mathbf k_a,\mathbf k_b)=\langle\psi_a(\mathbf k_a)|\psi_b(\mathbf k_b)\rangle$ between two arbitrary k-points, queried interactively [[notebook]](notebooks/06_eigenstate_session.ipynb)
- Atom-projection operators $(P_\alpha)_{ij}=\langle\psi_i|\hat P_\alpha|\psi_j\rangle$ (muffin-tin restriction of the identity, $\sum_\alpha P_\alpha+P_{\rm interstitial}=\mathbb 1$), applicable to any wavefunction in the same band window [[notebook]](notebooks/08_atom_projection.ipynb)
- Orbital-character (s, p, d, f) operators $P_{\alpha,\ell}=\sum_{m,\sigma}\langle\psi_i|\hat P_{\alpha,\ell m\sigma}|\psi_j\rangle$, the atom-projection operator resolved by angular momentum $\ell=0,1,2,3$ [[notebook]](notebooks/10_orbital_projection.ipynb)
- Atomic angular momentum operators $L_x,L_y,L_z$, the same $\ell$-resolved atom-projection operator generalized from a scalar weight to the full ladder-operator matrix (reusing Elk's own `lopzflm` subroutine) [[notebook]](notebooks/11_angular_momentum.ipynb)
- Spin operators $S_x,S_y,S_z$ (eigenvalues $\pm\tfrac12$) as Hermitian matrices in a band window, applicable to any wavefunction the same way [[notebook]](notebooks/09_spin_operators.ipynb)

## Ground-state electronic structure ##
- Self-consistent total energy $E[n]$, band structure $\epsilon_i(\mathbf k)$, density of states [[notebook]](notebooks/01_getting_started.ipynb)
- Hellmann-Feynman forces, structural relaxation, effective mass tensor, charge density $n(\mathbf r)$ [[notebook]](notebooks/02_relaxation_forces_and_properties.ipynb)
- Phonon dispersion and density of states via density functional perturbation theory [[notebook]](notebooks/03_phonon_dispersion_and_dos.ipynb)

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

## Z2 invariant of monolayer graphene with enhanced intrinsic spin-orbit coupling ##
Kane & Mele's founding prediction: planar graphene with intrinsic spin-orbit coupling is
a quantum spin Hall insulator ($\nu=1$) for any nonzero coupling strength:
```python
result = graphene.get_z2_invariant(1, ist1, nkx=24, nt=13)
result["z2"]                 # 1: quantum spin Hall (0 would be an ordinary insulator)
result["wannier_centers"]    # (nt, ist1) Wannier-charge-center angles vs. pumping k
```
![Alt text](images/graphene_z2_invariant.png?raw=true "Wannier charge centers of monolayer graphene with enhanced intrinsic spin-orbit coupling, showing an odd number of crossings")

## 3D strong/weak Z2 classification of a dimerized diamond lattice ##
Fu & Kane's own minimal lattice model for introducing $(\nu_0;\nu_1\nu_2\nu_3)$ -- diamond
structure with the second basis atom displaced along [111], shortening one bond per atom:
```python
result = cs.get_z2_invariant_3d(1, ist1, nkx=12, nt=7)
result["nu0"]         # strong index (this structure: 0 -- see docs/design.md #23,
                      # which retracts an earlier unconverged nu0=1 for it)
result["nu0_by_axis"] # (1, 1, 1): the strong index agrees identically across all three axes
```
![Alt text](images/cs_dimerized_z2_invariant_3d.png?raw=true "Wannier charge centers on the k1=0 and k1=pi planes of a dimerized diamond lattice, showing an odd vs. even number of crossings")

## Quantum metric alongside Berry curvature, along Gamma-K-M-K'-Gamma of monolayer h-BN ##
Time-reversal symmetry requires $g_{ab}(K)=g_{ab}(K')$ even though $\Omega(K')=-\Omega(K)$ --
a genuinely new prediction about the metric, not derivable from curvature alone:
```python
result = hbn.get_quantum_geometry(path, ist0, ist1, directions=(1, 2), dk=0.01)
result[0]["g"]                # (2,2) quantum metric [[g11,g12],[g12,g22]], Bohr^2
result[0]["berry_curvature"]  # Bohr^2, same convention as get_berry_curvature_path
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
itself; the antibonding (conduction-bottom, $\pi^*$) state is B-dominated instead --
shown here as a full band structure colored by weight on N, not just those two bands:
```python
n = hbn.structure.atom_index("N")
with hbn.eigenstate_session() as session:
    for k in path:                                       # path through Gamma-K-M-K'-Gamma
        energies.append(session.get_eigenstates(k).energies[w0 - 1 : w1])
        n_weight.append(session.atom_projection(k, w0, w1).matrices[n].diagonal().real)
```
![Alt text](images/hbn_atom_projection.png?raw=true "Band structure of monolayer h-BN along Gamma-K-M-K'-Gamma colored by muffin-tin weight on N")

## Orbital-character operators: p vs. s character across bands of monolayer h-BN ##
The occupied valence-top ($\pi$) band at K is a nitrogen $2p_z$ state; a much deeper
bonding ($\sigma$-type) valence band is nitrogen $2s$-dominated instead -- the dominant
channel flips between the two bands of the *same* atom, visible across the whole band
structure once colored by each channel's weight:
```python
from elkpy.session import ORBITAL_LABELS  # ("s", "p", "d", "f")

n = hbn.structure.atom_index("N")
s_i, p_i = ORBITAL_LABELS.index("s"), ORBITAL_LABELS.index("p")
with hbn.eigenstate_session() as session:
    for k in path:                                       # path through Gamma-K-M-K'-Gamma
        orb = session.orbital_projection(k, w0, w1)
        s_weight.append(orb.matrices[n, s_i].diagonal().real)
        p_weight.append(orb.matrices[n, p_i].diagonal().real)
```
![Alt text](images/hbn_orbital_projection.png?raw=true "Band structure of monolayer h-BN along Gamma-K-M-K'-Gamma, colored by N's s-weight and p-weight side by side, showing the channel flip between the deep bonding band and the valence-top band")

## Angular momentum operators: orbital valley locking in monolayer WSe2 ##
The valence-band-top state at K/K' is a pure $d_{x^2-y^2}\mp id_{xy}=Y_2^{\mp2}$ state on
tungsten (Xiao, Liu, Feng, Xu & Yao, PRL 108, 196802 (2012)): $L_z(K)=-L_z(K')$, with
$|L_z^{(d)}(K)|$ exactly $2\times$ W's own d-orbital weight -- only possible for a pure
$m=\pm2$ state, not a mixture:
```python
from elkpy.session import ORBITAL_LABELS

d = ORBITAL_LABELS.index("d")
w = wse2.structure.atom_index("W")
with wse2.eigenstate_session() as session:
    for k in path:                                              # path through Gamma-K-M-K'-Gamma
        energies.append(session.get_eigenstates(k).energies[w0 - 1 : w1])
        lz.append(session.angular_momentum(k, w0, w1).lz_orbital[w, d].diagonal().real)
```
![Alt text](images/wse2_angular_momentum.png?raw=true "Band structure of monolayer WSe2 along Gamma-K-M-K'-Gamma colored by Lz on the W d-channel, showing the K/K' sign flip")

## Spin operators: spin-valley locking in monolayer WSe2 ##
Broken inversion symmetry plus strong spin-orbit coupling locks the valence-band-top spin
to the valley index: $S_z(K)=-S_z(K')$ (Xiao, Liu, Feng, Xu & Yao, PRL 108, 196802 (2012)):
```python
with wse2.eigenstate_session() as session:
    for k in path:                                       # path through Gamma-K-M-K'-Gamma
        energies.append(session.get_eigenstates(k).energies[w0 - 1 : w1])
        sz.append(session.spin_operator(k, w0, w1).sz.diagonal().real)
```
![Alt text](images/wse2_spin_valley.png?raw=true "Band structure of monolayer WSe2 along Gamma-K-M-K'-Gamma colored by Sz, showing the K/K' sign flip")

## Optical matrix elements: valley-selective circular dichroism of monolayer h-BN ##
The band-edge transition at the zone corner absorbs one circular polarization only, with
opposite handedness at the two inequivalent valleys, $\eta(K')=-\eta(K)$ (Yao, Xiao & Niu,
PRB 77, 235406 (2008); Xiao, Liu, Feng, Xu & Yao, PRL 108, 196802 (2012)):
```python
from elkpy.parsers import optical

with hbn.eigenstate_session() as session:
    for k in path:                                 # path through Gamma-K-M-K'-Gamma
        m = session.momentum(k)                    # energies AND p^a_nm, one diagonalisation
        eta.append(optical.circular_polarization(m.pmat, ist1, ist1 + 1)["eta"])
```
![Alt text](images/hbn_circular_dichroism.png?raw=true "Band structure of monolayer h-BN along Gamma-K-M-K'-Gamma with the conduction band colored by the optical selectivity eta, showing the K/K' sign flip")

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
Fifteen notebooks under [`notebooks/`](notebooks), one per feature area above, each
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
| [`10_orbital_projection.ipynb`](notebooks/10_orbital_projection.ipynb) | Orbital-character (s/p/d/f) operators, p vs. s character across bands of monolayer h-BN | yes |
| [`11_angular_momentum.ipynb`](notebooks/11_angular_momentum.ipynb) | Angular momentum operators, orbital valley locking in monolayer WSe2 | yes |
| [`12_z2_invariant.ipynb`](notebooks/12_z2_invariant.ipynb) | The 2D $\mathbb Z_2$ invariant via Wannier-charge-center pumping | yes |
| [`13_z2_invariant_3d.ipynb`](notebooks/13_z2_invariant_3d.ipynb) | The 3D strong/weak $(\nu_0;\nu_1\nu_2\nu_3)$ classification | yes |
| [`14_optical_matrix_elements.ipynb`](notebooks/14_optical_matrix_elements.ipynb) | Optical matrix elements, circular dichroism and Kubo quantum geometry of monolayer h-BN | yes |
| [`15_parity_invariants.ipynb`](notebooks/15_parity_invariants.ipynb) | Parity eigenvalues at the TRIM and the Fu-Kane symmetry-indicator $Z_2$ | yes |
| [`01_getting_started.ipynb`](notebooks/01_getting_started.ipynb) | Ground state, band structure, density of states | -- |
| [`02_relaxation_forces_and_properties.ipynb`](notebooks/02_relaxation_forces_and_properties.ipynb) | Forces, relaxation, effective mass, density, `run_tasks()` | -- |
| [`03_phonon_dispersion_and_dos.ipynb`](notebooks/03_phonon_dispersion_and_dos.ipynb) | Phonon dispersion/DOS via DFPT | -- |

New notebooks should be added here alongside any new physics capability.
