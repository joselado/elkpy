# Fragment: optical absorption (task 121 wrapper + circular-resolved spectrum)

Merge target: a new `## 24.` section in `docs/design.md`, a new `\part{}` in
`docs/physics.tex`, a "Project status" paragraph in `CLAUDE.md`, and a
`FUNCTIONALITIES` bullet + `EXAMPLES` entry in `README.md`.
Everything below is written to be moved verbatim (the two halves are marked).
The section number `24` is provisional: other fragments written in parallel
(MOKE, the k.p effective-mass sum rule) may claim it too, so renumber on merge
and update the `#24` back-references in `parsers/optical.py`,
`parsers/dielectric.py`, `calculation.py` and the two test modules.

Cross-agent merge notes (this branch was written alongside others in the same
working tree):

- `spec.py`'s `TASKS["momentum_matrix"] = 120` was added by the MOKE work;
  `get_dielectric_function()` uses the same entry, since task 121 needs
  `PMAT.OUT` for exactly the same reason task 122 does. If that work is not
  merged, the entry must come along with this one.
- `parsers/dielectric.py` imports `parse_two_blocks` from `parsers/moke.py`:
  `EPSILON_ij.OUT`, `SIGMA_ij.OUT` and `KERR.OUT` share one layout, and
  `dielectric.f90` is where it originates. If only one of the two lands, move
  that ~25-line helper into whichever module survives rather than duplicating
  it; if both land, consider making `dielectric.py` the home and having
  `moke.py` import from it, which matches the direction of the dependency in
  the Fortran (`moke.f90` calls `dielectric`).

Follow-ups NOT done here (out of scope by instruction: no README/design.md/
physics.tex/CLAUDE.md edits, no notebook):

- a `notebooks/NN_optical_absorption.ipynb` demonstrating
  `get_dielectric_function()` + `get_circular_absorption()` (spectrum plot,
  and the valley-resolved circular channels), plus its README table row and
  `[[notebook]]` link, per the README/notebook style rules.
- the `README.md` `FUNCTIONALITIES` bullet, which should go in the
  "elkpy's own physics" group (circular-resolved absorption is genuinely
  beyond stock Elk), while the task-121 wrapper belongs to the condensed
  "also wraps" mention.

---

## PART 1 -- for `docs/design.md` (as `## 24.`)

## 24. Optical absorption: Elk's dielectric function, and the circular-polarization-resolved spectrum

Two related additions, one wrapping upstream Elk and one going beyond it.

### 24.1 `get_dielectric_function()` -- wrapping task 121

`Calculation.get_dielectric_function(components=, wplot=, nwplot=, swidth=,
intraband=, ngridk=)` runs tasks 120 then 121 (`src/writepmat.f90`,
`src/dielectric.f90`) and parses `EPSILON_ij.OUT`/`SIGMA_ij.OUT`
(`parsers/dielectric.py`). Task 121 is the independent-particle
(random-phase, no local fields, no excitons) dielectric tensor, evaluated
from the Kubo-Greenwood formula of *Physica Scripta* **T109**, 170 (2004) --
the same reference `genpmatk.f90` cites for the momentum matrix elements
(§22):

$$
\sigma_{ij}(\omega)=\frac{i}{N_k\Omega}\sum_{\mathbf k}\sum_{n,m}
\frac{f_n\left(1-f_m/f_{\max}\right)}{\varepsilon_{m}-\varepsilon_{n}}
\left[\frac{p^i_{nm}\,\overline{p^j_{nm}}}{\omega-\varepsilon_{mn}+i\varsigma}
+\frac{\overline{p^i_{nm}\overline{p^j_{nm}}}}{\omega+\varepsilon_{mn}+i\varsigma}\right],
$$

$$
\epsilon_{ij}(\omega)=\delta_{ij}+\frac{4\pi i\,\sigma_{ij}(\omega)}{\omega+i\varsigma},
$$

with $\varepsilon_{mn}=\varepsilon_m-\varepsilon_n$, $\Omega$ the unit-cell
volume, $N_k$ the number of **non-reduced** $k$-points, $f_n$ the occupation
numbers, $f_{\max}$ = `occmax` (2 without spin polarization, 1 when
`nspinor`=2) and $\varsigma$ = `swidth`, whose reciprocal is the relaxation
time. `getpmat` supplies $p^i_{nm}$ at each non-reduced point by rotating
task 120's reduced-mesh `PMAT.OUT` with the crystal symmetry that maps it
there.

Two version-coupled details, both taken from the Fortran rather than the
manual and both load-bearing:

- **Task 121 needs task 120 first.** `dielectric.f90` calls `getpmat`, which
  reads `PMAT.OUT` from disk and `stop`s if it is absent or has a different
  `nstsv`. Pairing the two tasks in one `_run_resumed()` call is why this is
  a named method rather than a bare `run_tasks([121])`.
- **The output filenames carry indices** (`EPSILON_11.OUT`, one file per
  `optcomp` entry), so `spec.py` grew an `OUTPUT_FILE_TEMPLATES` dict beside
  the fixed-name `OUTPUT_FILES`. The format inside is the two-block layout
  `moke.f90` also writes: `nwplot` `(omega, value)` pairs for the real part,
  a blank line, then the same for the imaginary part. The energy grid is
  $\omega_i = \omega_1 + (\omega_2-\omega_1)(i-1)/N_\omega$ with
  $\omega_1=\max(\texttt{wplot(1)},0)$ -- non-negative, and **excluding** the
  upper endpoint.

### 24.2 `get_circular_absorption()` -- the polarization-resolved spectrum

Stock Elk gives $\sigma_{xx}$, $\sigma_{xy}$ and the rest of the Cartesian
tensor, but never $\sigma_\pm$. The circular channels are what carry the
valley physics: with the circular-basis interband matrix elements of §22,

$$P_\pm(\mathbf k)=p^x_{cv}(\mathbf k)\pm i\,p^y_{cv}(\mathbf k),$$

the absorption resolved by photon helicity is

$$
\operatorname{Im}\epsilon_\pm(\omega)=\frac{4\pi^2}{\Omega\,\omega^2}
\sum_{\mathbf k}W_{\mathbf k}\sum_{v,c}f_v\!\left(1-\frac{f_c}{f_{\max}}\right)
\tfrac12\left|P_\pm(\mathbf k)\right|^2
\delta\!\left(\omega-(\varepsilon_c-\varepsilon_v)\right),
$$

which is exactly the $\varsigma\to0$ limit of §24.1's $\operatorname{Im}
\epsilon_{ii}$ with the linear intensity $|p^i_{cv}|^2$ replaced by the
circular one. `Calculation.get_circular_absorption()` evaluates it from
elkpy's own arbitrary-$k$ momentum matrix elements (§22's task-9002
`MOMENTUM` query) over the full non-reduced mesh, with all arithmetic in
`parsers/optical.circular_absorption()`; **no new Fortran at all**, and one
`eigenstate_session()` for the whole sweep.

Three things make this the right shape:

- **The mesh must not be symmetry-reduced.** $|P_\pm|^2$ is *not* invariant
  under the operations that fold the mesh -- that non-invariance *is* the
  dichroism -- so the usual reduced-mesh-with-weights sum, correct for the
  linear components, is wrong here. `Calculation._kmesh()` reproduces Elk's
  own non-reduced grid, $\mathbf k=(\mathbf i+\texttt{vkloff})/\texttt{ngridk}$.
- **Occupations come from Elk's own `EIGVAL.OUT`** (`parsers/eigval.py`), the
  same `occsv` array `dielectric.f90` reads, and are required to be
  $k$-independent -- an independent-particle interband spectrum is a gapped
  system's object, and `occupations_if_uniform()` raises on a metal rather
  than silently mis-counting a Fermi surface. This also sidesteps §13's
  standing pitfall of inferring a band count from a valence-electron count.
- **The $\pm$ labelling follows `circular_polarization()`** (§22), i.e.
  $|P_+|^2$ is the $\sigma^+$ intensity. The relative sign between valleys is
  the published physics; the absolute handedness is a
  structure-convention pin, exactly as for $S_z(K)$ (§17) and $\eta(K)$ (§22).

### 24.3 The two lineshapes, and why the naive one is not enough

The $\delta$-function formula above is the textbook independent-particle
spectrum, but it is only the $\varsigma\to0$ limit of what task 121 actually
evaluates. Putting the real intensity $z=|e\cdot p_{cv}|^2$ into §24.1 and
taking $\operatorname{Im}\epsilon=4\pi\operatorname{Re}[\sigma/(\omega+i\varsigma)]$
gives, per transition, exactly

$$
\frac{4\pi\varsigma\,z}{\Omega\,\Delta\,(\omega^2+\varsigma^2)}
\left[\frac{2\omega-\Delta}{(\omega-\Delta)^2+\varsigma^2}
+\frac{2\omega+\Delta}{(\omega+\Delta)^2+\varsigma^2}\right],
\qquad \Delta=\varepsilon_c-\varepsilon_v,
$$

whereas the $\delta$-form with a Lorentzian of width $\varsigma$ gives
$(4\pi^2 z/\Omega\omega^2)\cdot(\varsigma/\pi)/[(\omega-\Delta)^2+\varsigma^2]$.
The two agree at resonance, $\omega=\Delta$, and their ratio elsewhere is

$$\frac{2\omega-\Delta}{\Delta}\cdot\frac{\omega^2}{\omega^2+\varsigma^2},$$

which is **first order** in $(\omega-\Delta)/\Delta$: a few percent across one
linewidth, not a small correction, and *not* a defect of either code. Two
further consequences worth stating, because both were measured rather than
anticipated:

- The $\delta$-form has a spurious $1/\omega^2$ blow-up at small $\omega$,
  where a Lorentzian's fat tail is multiplied by a diverging prefactor. On
  h-BN this produced a fake "peak" of $\sim2\times10^3$ at
  $\omega=10^{-3}\,$Ha, dwarfing the real one at $0.236\,$Ha.
- The exact form's two terms cancel *exactly* at $\omega=0$ (the resonant one
  alone even turns negative below $\omega=\Delta/2$), so keeping only the
  resonant term is a real error that no peak-region comparison would catch.

`broadening="elk"` (the default) therefore evaluates the exact finite-$\varsigma$
response; `"lorentzian"`/`"gaussian"` give the textbook $\delta$-form and are
kept because that *is* the formula the physics is usually quoted in, and
because the two agree on the integrated oscillator strength
$\int\omega^2\operatorname{Im}\epsilon\,d\omega$ at any $\varsigma$.

### 24.4 Verification

The cross-check is an algebraic identity, not a symmetry assumption:

$$|P_+|^2+|P_-|^2=2\left(|p^x_{cv}|^2+|p^y_{cv}|^2\right)
\;\Longrightarrow\;
\operatorname{Im}\epsilon_++\operatorname{Im}\epsilon_-
=\operatorname{Im}\epsilon_{xx}+\operatorname{Im}\epsilon_{yy},$$

so a task-121 run asked for the $(1,1)$ and $(2,2)$ components on the same
mesh, `nempty` and `swidth` reproduces the polarization-summed elkpy
spectrum. Nothing is shared between the two routes but the ground state:
Elk reads reduced-mesh momentum matrix elements off `PMAT.OUT` and rotates
them, sums in Fortran over `nkptnr`, and forms $\epsilon$ from $\sigma$;
elkpy re-diagonalises at every non-reduced point through the task-9002
session and does everything else in NumPy.

Against a real compiled binary, monolayer h-BN (the §22 slab; $6\times6\times1$
mesh, `nempty`=12, `swidth`=0.005 Ha, `rgkmax`=7):

- **The polarization-summed spectrum reproduces task 121 essentially exactly.**
  Over the 297 grid points where $\operatorname{Im}\epsilon$ exceeds 2% of its
  peak, the largest relative deviation is $9.3\times10^{-4}$ and the median is
  $2.0\times10^{-5}$; at the peak, 15.414871 (elkpy) against 15.414867 (Elk).
  The integrated oscillator strength
  $\int\omega^2\operatorname{Im}\epsilon\,d\omega$ agrees to $8\times10^{-5}$
  relative. The residual is at the level expected from the two subdirectories'
  independent task-1 SCF continuations and from `getpmat`'s symmetry rotation
  (Elk's own $\operatorname{Im}\epsilon_{xx}$ vs $\operatorname{Im}\epsilon_{yy}$,
  which exact symmetry makes equal, differ by $1\times10^{-4}$ of the peak).
- **The zone-integrated circular channels are equal to machine precision**:
  $\max|\operatorname{Im}\epsilon_+-\operatorname{Im}\epsilon_-|=5.6\times10^{-13}$
  against a peak of 15.4, i.e. $4\times10^{-14}$ relative -- the time-reversal
  statement above, and confirmation that nothing in the $\pm$ bookkeeping is
  biased.
- **Restricting the sum to one valley gives perfect circular selectivity**:
  at the absorption edge ($\omega=0.167$ Ha $=4.5$ eV) $\eta(K)=-0.9999$ and
  $\eta(K')=+0.9999$, with $\operatorname{Im}\epsilon_-=122.6$ against
  $\operatorname{Im}\epsilon_+=0.0044$ at $K$. This is §22's $\eta(K)=-1$
  band-edge result reappearing as a *spectrum*, and it is the output stock Elk
  cannot produce.
- **The $\delta$-lineshape's cost is measured, not assumed**: the same sum with
  `broadening="lorentzian"` at this $\varsigma$ deviates from task 121 by up to
  13.6% (median 2.2%) across the peak region, plus the $1/\omega^2$ artifact
  below the edge (a spurious $2\times10^3$ "peak" at $\omega=10^{-3}$ Ha).
  With `"gaussian"` it is worse still (median 25% in the peak region), as
  expected: Elk's broadening is Lorentzian by construction.

`tests/test_parsers_absorption.py` pins the arithmetic ahead of any Elk run,
on the massive Dirac model of §22: the $\delta$-form against its closed form
(prefactor, $1/\omega^2$, occupation weight and Lorentzian normalization at
once), the `"elk"` form against `dielectric.f90`'s expression transcribed
directly, the $\omega=0$ cancellation, the convergence of the two lineshapes
as $\varsigma$ falls, the circular-vs-linear identity above against an
independently written linear sum, $\eta$ at the peak against the already
trusted `circular_polarization()`, the $k\to-k$ cancellation and its recovery
under a mask, and oscillator-strength conservation for both the Lorentzian
and the Gaussian.

### 24.5 What this is not

The independent-particle spectrum has no electron-hole interaction, so it
misses the excitonic physics that dominates real h-BN optics: the measured
optical gap sits well below the calculated absorption onset, and the
oscillator strength is redistributed into a bound exciton peak. (Elk's own
`dielectric_bse.f90` is the route to that; not wrapped here.) The
Kohn-Sham gap itself is also LDA-underestimated. In addition, for a slab the
dielectric function is a **supercell** quantity -- $\epsilon-1$ scales like
$1/L_z$ with the vacuum thickness, so absolute values are not the 2D
material's own response. None of this affects the comparison above, where both
sides use the same cell, but all of it affects reading the numbers as h-BN's
optics.

### How to use in code

```python
hbn = Structure(HBN_AVEC, HBN_SPECIES).get_calculation(
    "hbn", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
    extra_blocks={"nempty": [12]},   # both routes must see the same nstsv
)
hbn.get_energy()

# Elk's own dielectric tensor (tasks 120 + 121)
elk = hbn.get_dielectric_function(components=((1, 1), (2, 2)), swidth=0.005)
w, eps_xx = elk["energies"], elk["epsilon"][(1, 1)]

# elkpy's circular-resolved spectrum over the same non-reduced mesh
out = hbn.get_circular_absorption(swidth=0.005)
out["eps2_total"]                      # == Im eps_xx + Im eps_yy
out["eps2_plus"], out["eps2_minus"]    # equal once summed over the zone

# the valley physics: restrict the k-sum to one valley. The returned dict
# carries everything the sum needs, so re-weighting costs no Elk time.
from elkpy.parsers import optical
v = hbn.get_circular_absorption(kpoints=[(1/3, 1/3, 0), (-1/3, -1/3, 0)],
                                swidth=0.005)
one = optical.circular_absorption(
    v["kdata"], v["omega"], v["occupations"], v["volume"],
    occmax=v["occmax"], swidth=0.005, weights=[1.0, 0.0],
)
one["eta"]     # ~ -1 across the K band edge, +1 for the K' mask
```

One caveat on `swidth` worth carrying into the docs: it is written into the
task-121 subdirectory's `elk.in`, so it also governs the smearing of the
task-1 SCF continuation that runs there, while `get_circular_absorption()`'s
own session uses whatever the `Calculation` was built with. For a gapped
system that is immaterial (both give integer occupations and identical
eigenvalues, which is why the agreement above is at the $10^{-5}$ level); for
a metal the two sides would be smeared differently and should be aligned via
`extra_blocks={"swidth": [...]}` on the `Calculation` itself. Elk's own MOKE
example raises `swidth` after the ground state for exactly this reason.

---

## PART 2 -- for `docs/physics.tex` (as a new `\part{}`)

```latex
\part{Optical absorption: dielectric function and circular dichroism}
\label{part:optics}

\section{The independent-particle dielectric function}

Linear response of a crystal to a monochromatic field of frequency
$\omega$ and polarization $\mathbf e$ is governed, in the
independent-particle (random-phase) approximation, by the
Kubo--Greenwood conductivity
%
\begin{equation}
  \sigma_{ij}(\omega)=\frac{i}{N_k\Omega}\sum_{\mathbf k}\sum_{n,m}
  \frac{f_{n\mathbf k}\left(1-f_{m\mathbf k}/f_{\max}\right)}
       {\varepsilon_{m\mathbf k}-\varepsilon_{n\mathbf k}}
  \left[\frac{p^i_{nm}\overline{p^j_{nm}}}
             {\omega-\varepsilon_{mn}+i\varsigma}
       +\frac{\overline{p^i_{nm}\overline{p^j_{nm}}}}
             {\omega+\varepsilon_{mn}+i\varsigma}\right],
  \label{eq:optics-sigma}
\end{equation}
%
where $p^i_{nm}=\langle\psi_{n\mathbf k}|\hat p_i|\psi_{m\mathbf k}\rangle$
are the momentum matrix elements of Part~\ref{part:momentum},
$\varepsilon_{mn}=\varepsilon_{m\mathbf k}-\varepsilon_{n\mathbf k}$,
$f_{n\mathbf k}$ the occupation numbers, $f_{\max}$ their maximum value
($2$ without spin polarization, $1$ for spinor states), $\Omega$ the
unit-cell volume, $N_k$ the number of $k$-points in the full (non-reduced)
Brillouin-zone sampling, and $\varsigma$ a phenomenological inverse
relaxation time that both broadens the resonances and regularizes them.
The dielectric tensor follows as
%
\begin{equation}
  \epsilon_{ij}(\omega)=\delta_{ij}
  +\frac{4\pi i\,\sigma_{ij}(\omega)}{\omega+i\varsigma}.
  \label{eq:optics-eps}
\end{equation}
%
Physically, $\operatorname{Im}\epsilon_{ij}$ counts vertical (momentum
conserving) transitions from an occupied state to an empty one, each
weighted by the squared matrix element of the velocity operator along the
field. What is neglected is the electron--hole interaction: there are no
excitons here, and no local-field corrections either, so the calculated
absorption edge lies at the Kohn--Sham gap rather than at the true optical
gap.

Taking $\varsigma\to0$ in \eqref{eq:optics-sigma}--\eqref{eq:optics-eps},
$1/(\omega-\Delta+i\varsigma)\to\mathcal P(\cdot)-i\pi\delta(\omega-\Delta)$,
leaves the familiar Fermi-golden-rule form
%
\begin{equation}
  \operatorname{Im}\epsilon_{\mathbf e}(\omega)=
  \frac{4\pi^2}{\Omega\,\omega^2}\sum_{\mathbf k}W_{\mathbf k}
  \sum_{v,c} f_v\!\left(1-\frac{f_c}{f_{\max}}\right)
  \left|\mathbf e\cdot\mathbf p_{cv}(\mathbf k)\right|^2
  \delta\!\left(\omega-(\varepsilon_c-\varepsilon_v)\right).
  \label{eq:optics-golden}
\end{equation}

\section{Circular polarization and valley selectivity}

For light propagating along $\hat z$ the two helicities correspond to
$\mathbf e_\pm\propto\hat x\pm i\hat y$, so the relevant matrix elements are
%
\begin{equation}
  P_\pm(\mathbf k)=p^x_{cv}(\mathbf k)\pm i\,p^y_{cv}(\mathbf k),
  \qquad
  \left|P_+\right|^2+\left|P_-\right|^2
  =2\left(\left|p^x_{cv}\right|^2+\left|p^y_{cv}\right|^2\right).
  \label{eq:optics-circular}
\end{equation}
%
Substituting $\tfrac12|P_\pm|^2$ for $|\mathbf e\cdot\mathbf p_{cv}|^2$ in
\eqref{eq:optics-golden} defines the helicity-resolved spectra
$\operatorname{Im}\epsilon_\pm$, whose sum is
$\operatorname{Im}\epsilon_{xx}+\operatorname{Im}\epsilon_{yy}$ by the
identity in \eqref{eq:optics-circular}.

In a gapped honeycomb crystal the three-fold rotation about the zone corner
forces the band-edge transition to be an eigenstate of $C_3$ with a definite
angular-momentum change, so only one helicity couples: $|P_-|=0$ at $K$ and
$|P_+|=0$ at $K'$ (or vice versa, depending on the structure's own
orientation conventions). This is valley-selective circular dichroism, and it
is the reason a circular-resolved spectrum carries information the Cartesian
tensor does not. Time reversal, however, relates the two valleys,
$\mathbf p_{cv}(-\mathbf k)=-\overline{\mathbf p_{cv}(\mathbf k)}$, so a sum
over a Brillouin-zone sampling closed under $\mathbf k\to-\mathbf k$ gives
$\operatorname{Im}\epsilon_+=\operatorname{Im}\epsilon_-$ identically in a
non-magnetic crystal: the dichroism is a $\mathbf k$-resolved property, and
is measured by exciting one valley selectively, not by illuminating the whole
zone.

\section{Finite broadening}

At the finite $\varsigma$ any practical calculation uses, the
$\delta$-function form \eqref{eq:optics-golden} and the exact response
\eqref{eq:optics-sigma}--\eqref{eq:optics-eps} are not the same function.
Carrying the algebra through for a single transition of energy $\Delta$ and
real intensity $z$ gives
%
\begin{equation}
  \operatorname{Im}\epsilon(\omega)=
  \frac{4\pi\varsigma z}{\Omega\,\Delta\,(\omega^2+\varsigma^2)}
  \left[\frac{2\omega-\Delta}{(\omega-\Delta)^2+\varsigma^2}
       +\frac{2\omega+\Delta}{(\omega+\Delta)^2+\varsigma^2}\right],
  \label{eq:optics-exact-line}
\end{equation}
%
against the Lorentzian-broadened $\delta$-form
$4\pi\varsigma z/[\Omega\omega^2((\omega-\Delta)^2+\varsigma^2)]$. The two
coincide at resonance and differ elsewhere by the factor
$[(2\omega-\Delta)/\Delta]\,\omega^2/(\omega^2+\varsigma^2)$, first order in
$(\omega-\Delta)/\Delta$. The second (anti-resonant) term of
\eqref{eq:optics-exact-line} is not optional: the first alone changes sign
below $\omega=\Delta/2$, and the two cancel exactly at $\omega=0$, as they
must for a system with a gap.

\section{How to use in code}

\begin{verbatim}
hbn = Structure(HBN_AVEC, HBN_SPECIES).get_calculation(
    "hbn", xc="PW", ngridk=(6, 6, 1), rgkmax=7.0,
    extra_blocks={"nempty": [12]},
)
hbn.get_energy()

# Elk's own dielectric tensor (tasks 120 then 121)
elk = hbn.get_dielectric_function(components=((1, 1), (2, 2)), swidth=0.005)

# elkpy's helicity-resolved spectrum on the same non-reduced mesh
out = hbn.get_circular_absorption(swidth=0.005)
out["eps2_total"]                    # = Im eps_xx + Im eps_yy
out["eps2_plus"], out["eps2_minus"]  # equal when summed over the whole zone

# one valley at a time: the dichroism the zone sum cancels
from elkpy.parsers import optical
optical.circular_absorption(valleys["kdata"], w, occ, volume,
                            swidth=0.005, weights=[1.0, 0.0])["eta"]
\end{verbatim}
```
