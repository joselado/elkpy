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
wavefunctions, optical (velocity) matrix elements with the circular dichroism and
Kubo-form quantum geometry built from them, and anisotropic magnetic exchange
constants (Heisenberg, Dzyaloshinskii-Moriya and Kitaev-type terms) by four-state
energy mapping.

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
./build_elk.sh

# 2. Install elkpy (editable)
python3 -m pip install -e .        # add .[ase] for Structure.from_ase()/to_ase()
```

Elk needs BLAS/LAPACK (OpenBLAS supplies both) and FFTW3 in double *and* single
precision — on Debian/Ubuntu, `gfortran libopenblas-dev libfftw3-dev`.
`build_elk.sh` link-tests the compiler flags before building and repairs them if
they do not work here: on a cluster it loads the `openblas`/`fftw` environment
modules, links with an `-rpath` so the binary still runs in a batch job that did
not load them, and compiles for a portable instruction set instead of the login
node's own. Override with `ELKPY_F90_LIB`, `ELKPY_MARCH` or `ELKPY_MODULES`
(see `docs/design.md` §8).

# FUNCTIONALITIES #

## Spin-orbit coupling ##
- Per-species scaling of the spin-orbit term $\hat H_{\rm soc}(r)=f_{\rm soc}(r)\,\hat{\mathbf L}\cdot\boldsymbol\sigma$, rather than one global scale for the whole cell [[notebook]](notebooks/04_per_species_soc_scaling.ipynb)

## Topological characterization ##
- Berry curvature $F_{12}(\mathbf k)=\partial_1A_2-\partial_2A_1$ with $A_\mu=i\langle n|\partial_\mu n\rangle$, and Chern numbers $c_n=\frac1{2\pi}\int_{T^2}\!d^2k\,F_{12}\in\mathbb Z$, via a gauge-invariant Wilson-loop discretization in the standard sign convention [[notebook]](notebooks/05_berry_curvature.ipynb)
- Berry curvature at an arbitrary k-point with no periodic mesh required, e.g. to resolve individual valleys of a 2D material [[notebook]](notebooks/05_berry_curvature.ipynb)
- The $\mathbb Z_2$ invariant $\nu\in\{0,1\}$ of a time-reversal-invariant 2D insulator, via Wannier-charge-center pumping and the non-Abelian Wilson loop $D(k_2)=\prod_iU(F_i)$, distinguishing an ordinary insulator from a quantum spin Hall insulator [[notebook]](notebooks/12_z2_invariant.ipynb)
- The full 3D strong/weak classification $(\nu_0;\nu_1\nu_2\nu_3)$ of a 3D time-reversal-invariant insulator, from the $Z_2$ invariant of each of the Brillouin zone's six time-reversal-invariant planes, distinguishing an ordinary insulator from a strong or weak topological insulator [[notebook]](notebooks/13_z2_invariant_3d.ipynb)
- The same $Z_2$ invariants from parity eigenvalues alone, $\delta_i=\prod_m\xi_{2m}(\Gamma_i)$ with $(-1)^{\nu_0}=\prod_i\delta_i$ — exact and mesh-free, needing only the 8 (3D) or 4 (2D) time-reversal-invariant momenta, for crystals with an inversion centre [[notebook]](notebooks/15_parity_invariants.ipynb)

## Magnetic exchange interactions ##
- The full anisotropic exchange tensor $J_{ij}^{\alpha\beta}$ of a magnetic pair, from $H=\sum_{i<j}\mathbf S_i\cdot\mathbf J_{ij}\cdot\mathbf S_j$, by four-state energy mapping $J_{ij}^{\alpha\beta}=(E_1+E_4-E_2-E_3)/4S^2m_{ij}$ over constrained non-collinear states [[notebook]](notebooks/18_exchange_constants.ipynb)
- Its decomposition into isotropic Heisenberg exchange, the Dzyaloshinskii-Moriya vector $D^x=\tfrac12(J^{yz}-J^{zy})$, and the symmetric anisotropy that carries the Kitaev $K$ and $\Gamma$ terms of a honeycomb magnet [[notebook]](notebooks/18_exchange_constants.ipynb)
- Single-ion anisotropy $\mathbf A_{ii}$, which the exchange formula cancels by construction and so needs configurations and formulas of its own [[notebook]](notebooks/18_exchange_constants.ipynb)

## Spin-polarized scanning tunneling microscopy ##
- Spin-polarized STM images in the Tersoff-Hamann picture, $dI/dV(\mathbf r)\propto n(\mathbf r,E_F+eV)+P_T\,\mathbf m(\mathbf r,E_F+eV)\cdot\hat{\mathbf e}_T$ — the vacuum local density of states projected onto an arbitrary Cartesian tip magnetization direction, which resolves magnetically inequivalent but chemically identical atoms [[notebook]](notebooks/19_spin_polarized_stm.ipynb)
- Both the differential-conductance map at one energy and the bias-window-integrated (constant-current) image, at any tip height [[notebook]](notebooks/19_spin_polarized_stm.ipynb)

## Vertical tunnelling transport ##
- What gets *through* a two-dimensional material rather than what an STM tip sees above it: the transmission from a point tip into a substrate plane, $T(\mathbf r;E)=\int_{\rm plane}|G(\mathbf r,\mathbf r';E)|^2 d^2r'$, set by the *nonlocal* Green's function $G(\mathbf r,\mathbf r')=\sum_{n\mathbf k}\psi_{n\mathbf k}(\mathbf r)\psi^*_{n\mathbf k}(\mathbf r')/(E-\varepsilon_{n\mathbf k}+i\eta)$ — so bands add as amplitudes, not as probabilities [[notebook]](notebooks/20_vertical_transport.ipynb)
- The interference no local picture contains: a substrate invariant under every lateral translation conserves $\mathbf k_\parallel$, leaving one Gram matrix per k-point, $S_{\mathbf k}[n,n']=\int_{\rm plane}\psi^*_{n\mathbf k}\psi_{n'\mathbf k}$, whose off-diagonal is the whole departure from a Tersoff-Hamann image [[notebook]](notebooks/20_vertical_transport.ipynb)
- A magnetic substrate, which accepts $1+P_{\rm s}\hat{\mathbf n}\cdot\boldsymbol\sigma$ inside that overlap — the spin-space sibling of a magnetic tip, but selecting which states leave rather than which are seen [[docs]](docs/design.md)

## Quantum geometry ##
- The full quantum geometric tensor $Q_{ab}=g_{ab}-\tfrac i2F_{ab}$ at an arbitrary k-point: Berry curvature $F_{ab}$ *and* the quantum metric $g_{ab}$ (Fubini-Study distance between neighbouring Bloch states), from the same wavefunction-overlap queries used for eigenstates below [[notebook]](notebooks/07_quantum_geometry.ipynb)
- The same tensor in its Kubo (sum-over-states) form $T_{ab}=\sum_{n\in W,\,m\notin W}\langle n|v_a|m\rangle\langle m|v_b|n\rangle/(\varepsilon_n-\varepsilon_m)^2$, needing no k-derivative at all — an independent route to $g_{ab}=\mathrm{Re}\,T_{ab}$ and $F_{ab}=-2\,\mathrm{Im}\,T_{ab}$ [[notebook]](notebooks/14_optical_matrix_elements.ipynb)

## Optical response ##
- Momentum (velocity) matrix elements $p^a_{nm}=\langle\psi_n|(-i\nabla+\tfrac1{4c^2}[\vec\sigma\times\nabla V_s])_a|\psi_m\rangle$ at an arbitrary k-point — the optical dipole matrix elements, and, for a local Kohn-Sham potential in atomic units, the velocity operator $\hat{\mathbf v}=\hat{\mathbf p}$ [[notebook]](notebooks/14_optical_matrix_elements.ipynb)
- Circular dichroism $\eta=(|P_+|^2-|P_-|^2)/(|P_+|^2+|P_-|^2)$ of an interband transition, $P_\pm=p^x_{cv}\pm ip^y_{cv}$ — the valley-selective optical selection rule of a gapped honeycomb lattice [[notebook]](notebooks/14_optical_matrix_elements.ipynb)
- Band velocities $\mathbf v_n=\partial\varepsilon_n/\partial\mathbf k$ as the diagonal of the same operator, exact by Hellmann-Feynman [[notebook]](notebooks/14_optical_matrix_elements.ipynb)
- Interband absorption resolved by circular polarization, $\mathrm{Im}\,\varepsilon_\pm(\omega)\propto\omega^{-2}\sum_{\mathbf k,v,c}|P_\pm|^2\delta(\omega-\varepsilon_{cv})$ — the spectrum behind valley-selective optical pumping, which stock Elk's dielectric tensor cannot resolve [[docs]](docs/design.md)
- Effective masses from the k·p sum rule $(1/m^*)^{ab}=\delta_{ab}+2\sum_m\mathrm{Re}[p^a_{nm}p^b_{mn}]/(\varepsilon_n-\varepsilon_m)$, decomposed by which interband coupling produces them [[notebook]](notebooks/16_effective_mass.ipynb)
- Spin Berry curvature and the intrinsic spin Hall conductivity, from the spin current operator $J^s_a=\tfrac12\{S_s,v_a\}$ [[notebook]](notebooks/17_spin_hall.ipynb)

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
- Kohn-Sham potential and the electron localization function as 3D fields, the dielectric tensor, and the magneto-optic Kerr effect [[docs]](docs/design.md)

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
result["nu0"]         # strong index -- 1 at THIS mesh, but not converged: it oscillates
                      # 1,0,1,0 as the mesh is refined. The exact parity indicator below
                      # gives 0 for this structure (docs/design.md #23 retracts the
                      # nu0=1 that was once read off this call)
result["nu0_by_axis"] # (1, 1, 1): the strong index agrees identically across all three
                      # axes -- an algebraic consistency check that does hold
```
![Alt text](images/cs_dimerized_z2_invariant_3d.png?raw=true "Wannier charge centers on the k1=0 and k1=pi planes of a dimerized diamond lattice; the crossing count on the k1=0 plane is mesh-dependent, see docs/design.md section 23")

## Spin-polarized STM of a non-collinear 120-degree Néel Cr monolayer ##
The three Cr atoms are chemically identical, so a conventional STM sees only the 1x1
lattice; projecting the vacuum LDOS onto a magnetic tip resolves the magnetic
superstructure instead, $dI/dV\propto n+P_T\,\mathbf m\cdot\hat{\mathbf e}_T$ (Wortmann,
Heinze, Kurz, Bihlmayer & Blügel, PRL 86, 4132 (2001)):
```python
# one run returns n, m.e_T and n + P_T m.e_T at every point of the tip plane
stm = cr.get_spin_stm(direction=(1, 0, 0), height=0.25, grid=(60, 60), swidth=0.005)

plt.pcolormesh(x, y, stm["ldos_grid"])       # conventional STM: 1x1, nearly flat
plt.pcolormesh(x, y, stm["spin_ldos_grid"])  # spin contrast: the magnetic superstructure
```
![Alt text](images/cr_spin_stm.png?raw=true "Spin-summed and spin-projected vacuum LDOS above a 120-degree Neel Cr monolayer, showing the magnetic superstructure the spin-averaged image cannot resolve")

## What gets through a sheet, rather than what a tip sees above it ##
For monolayer graphene the two are the same picture, and that is a theorem: the states at
$E_F$ are the Dirac doublet at K, the symmetry the tip/substrate geometry leaves intact
($C_{3v}$, whose mirror swaps the sublattices without flipping $z$) still acts irreducibly
on it, so by Schur's lemma $S_K$ is a multiple of the identity and there is nothing
off-diagonal for the current to interfere through:
```python
# `mono` is a monolayer-graphene Calculation built as above, with
# extra_blocks={"tshift": [False]} -- mandatory, so that the two plotting
# planes and the atoms stay in the same frame.
# The k-grid is the task's own -- independent of ngridk and reducek. A multiple
# of 3 contains K, and graphene's states at E_F are all there
geom = dict(exit_height=0.38, height=0.62, grid=(24, 24), kgrid=(6, 6, 1), broadening=0.005)
flow = mono.get_vertical_transport(**geom)                      # through the sheet
local = mono.get_vertical_transport(exit_region="cell", **geom)  # what an STM sees

flow["channels"]             # 2.0000 -- two channels the substrate cannot tell apart
flow["offdiagonal_weight"]   # 0.0    -- so nothing interferes, and the maps coincide
```
![Alt text](images/graphene_vertical_transport.png?raw=true "Vertical transmission through monolayer graphene next to the Tersoff-Hamann image at the same tip plane; the two are the same picture, correlation 1.000000")

Stack a second layer AB and the element exchanging the two zero-energy states is an in-plane
$C_2'$, which *flips* $z$ and is broken by having a tip above and a substrate below. The
states are then the non-dimer sites of two different layers, and $S_K$'s two eigenvalues
differ by a factor of 1311 instead of 1.0000017:
```python
# `bi` is the AB-stacked bilayer: layers at z1, z2 = 3.35 A apart, and `gap`
# the same standoff above and below, so only the substrate side differs
biflow = bi.get_vertical_transport(exit_height=z1 - gap, height=z2 + gap,
                                   grid=(24, 24), kgrid=(6, 6, 1), broadening=0.005)
biflow["offdiagonal_weight"]  # 0.50 -- half of S_k is off its diagonal
biflow["interference"]        # coherent minus incoherent: what no local map contains
```
![Alt text](images/bilayer_vertical_transport.png?raw=true "Vertical transmission through AB bilayer graphene next to the Tersoff-Hamann image; the current has to cross both layers and the map departs from the local one, correlation 0.81")

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
Eighteen notebooks under [`notebooks/`](notebooks), one per feature area above, each
executed end-to-end against a real compiled Elk binary and checked in with its actual
output (two are exceptions, left unexecuted with a note: the DFPT phonon notebook,
since a single call takes ~11-13 minutes, and the exchange-constants notebook,
whose full tensor is 36 constrained SCF runs). Listed new-physics-first, matching
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
| [`16_effective_mass.ipynb`](notebooks/16_effective_mass.ipynb) | Effective masses from the k·p sum rule, and which bands produce them | yes |
| [`17_spin_hall.ipynb`](notebooks/17_spin_hall.ipynb) | Spin Berry curvature, and why the two valleys agree in sign | yes |
| [`18_exchange_constants.ipynb`](notebooks/18_exchange_constants.ipynb) | Anisotropic exchange tensor by four-state energy mapping (Heisenberg, DM, Kitaev) | yes |
| [`19_spin_polarized_stm.ipynb`](notebooks/19_spin_polarized_stm.ipynb) | Spin-polarized STM image of a non-collinear 120-degree Néel Cr monolayer | yes |
| [`20_vertical_transport.ipynb`](notebooks/20_vertical_transport.ipynb) | Vertical tunnelling transport: monolayer vs AB-bilayer graphene, and where the local picture fails | yes |
| [`01_getting_started.ipynb`](notebooks/01_getting_started.ipynb) | Ground state, band structure, density of states | -- |
| [`02_relaxation_forces_and_properties.ipynb`](notebooks/02_relaxation_forces_and_properties.ipynb) | Forces, relaxation, effective mass, density, `run_tasks()` | -- |
| [`03_phonon_dispersion_and_dos.ipynb`](notebooks/03_phonon_dispersion_and_dos.ipynb) | Phonon dispersion/DOS via DFPT | -- |

New notebooks should be added here alongside any new physics capability.
