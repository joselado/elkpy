# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Roadmap Tiers 1-3 from `docs/roadmap.md` are implemented: `Structure` → `Calculation` with
`get_energy()`, `get_bands()`, `get_dos()`, `get_forces()`, `get_relaxed()`, `get_effective_mass()`,
`get_density()`, `get_phonon_dos()`/`get_phonon_dispersion()`, and a `run_tasks()` escape hatch, plus a
`spec.py` module for version-coupled task/xctype/filename knowledge. Verified end-to-end against a real
compiled Elk binary on bulk Si/Fe (`tests/test_calculation_si.py`, `tests/test_calculation_fe.py`,
`tests/test_calculation_si_phonons.py`). Also present: `vendor/elk/` (vendored Elk 11.0.2, unmodified,
tracked in git), `docs/elk_manual.pdf`/`.txt` (official manual, plain-text version for grepping), and
`docs/design.md` + `docs/roadmap.md` (architecture strategy and forward plan — read before adding code).
Not implemented: symbolic k-path's disconnected-segment support (`,` breaks), the classical supercell
phonon method (task 200, DFPT/task 205 only), scheduler-backed launchers, MPI, named `get_*` methods for
potential/ELF volumetric plots (reachable via `run_tasks()` + `parsers.volumetric`). Check `src/elkpy/`
directly rather than assuming the docs describe current code; update both as they diverge.

Also implemented, as the first real entry in the Fortran patch series described below:
`Calculation(spinorb=, soc_scale={"Fe": 1.5, ...})` — per-species scaling of the spin-orbit coupling
term, on top of upstream Elk's single global `socscf` scalar. `patches/0001-per-species-soc-scale.patch`
adds a `socscfsp(maxspecies)` array (`modmain.f90`) and an `elkpy_socscale` input block (`readinput.f90`)
overriding `socscf` per species inside `gensocfr.f90`'s existing per-atom loop — the smallest possible
hook, verified against a real compiled binary to reproduce the global-`socscf` result exactly for a
single-species cell and to scale independently per species in a two-species cell
(`tests/test_calculation_soc.py`). Physics writeup (Koelling-Harmon SOC term, why a global scale is
the wrong shape for multi-species cells): `docs/design.md` §12 and `docs/physics.tex` (Part I).

Also implemented, as the second entry in the Fortran patch series:
`Calculation.get_berry_curvature(ist0, ist1, directions=(1, 2))` — Berry curvature and Chern number via
the Wilson-loop (Fukui-Hatsugai-Suzuki) method. `patches/0002-berry-curvature-wilson-loop.patch` adds a
new task (9000) and `elkpy_berry.f90`, reusing Elk's own `genwfsvp`/`genolpq` (the machinery behind
Wannier90 export, task 550) to export mesh-neighbour wavefunction overlap matrices without needing
task 550's own neighbour-shell search (which calls the external Wannier90 library, unavailable in this
build) — the two loop directions are exact `ngridk` mesh generators, so the neighbour is constructed
directly. All Wilson-loop arithmetic (link variables, plaquette flux, Chern number, admissibility) is
done in Python (`parsers/berry.py`), independently unit-tested against synthetic overlap matrices, both
for gauge invariance and, separately, for sign -- gauge invariance alone can't catch a conjugation-sign
flip, since `conj(M)` is exactly as gauge-invariant as `M`, so the Python arithmetic's sign is pinned
directly against FHS eq. 8 with hand-built matrices of known phase (`tests/test_berry_gauge_invariance.py`).
The Fortran conjugation convention, originally resting only on a `zgemv` BLAS-semantics derivation, is
now confirmed at runtime by §22's independent Kubo cross-check -- which also found and fixed a missing
negation in the Python flux step, so elkpy now reports the standard Xiao-Chang-Niu curvature/Chern sign
everywhere via the single `parsers.berry._berry_phase()` (see §22). Against a real compiled binary, bulk Si's trivial valence manifold gives a Chern number of
floating-point zero (`tests/test_calculation_berry.py`) -- which, being zero either way, would not itself
catch a sign error.

Also implemented: `Calculation.get_berry_curvature_path(kpoints, ist0, ist1, directions=(1, 2), dk=)`
(task 9001) -- Berry curvature at an arbitrary, explicit list of k-points (e.g. a band-structure-style
path), via one small Wilson loop per point (`pyqula`'s `berry_curvature(h, k, dk=...)` convention)
rather than a periodic mesh. Also accepts `kpath="GKMG"` (pyqula/ASE-style symbolic path, same
convention as `get_bands()`/`get_phonon_dispersion()`'s `kpath=`, resolved via `_kpath_to_points()`)
instead of `kpoints=`, discretized into `npoints` points with each returned point additionally carrying
a `"distance"` entry for plotting; unlike `get_bands()`/`get_phonon_dispersion()` (which hand vertices to
Elk's own `plot1d` task, so a disconnected `,` path can't be interpolated), `get_berry_curvature_path()`
evaluates every point independently, so a disconnected `,` path (e.g. to jump straight to a specific K′
zone image) is fully supported. Needed one genuinely new piece of Fortran, not just a smaller task 9000:
`elkpy_wfcorner` expands a wavefunction at an arbitrary k-point via fresh on-the-fly diagonalisation
(`eveqnfv`/`eveqnsv`, the same pattern `src/bandstr.f90` uses for a band-structure path) rather than
`genwfsvp`'s file-backed `getevecfv`/`getevecsv`, which only knows previously-diagonalised mesh points --
so this mode needs no `ngridk` alignment and no `reducek=0` at all, and a Γ-only ground state is
sufficient (not a stopgap), since task 9001 never touches the ground state's own mesh. Trade-off: no
Chern number (needs a closed cover of the zone) and no automatic gap check (no eigenvalues exported in
this mode). Verified against a real compiled binary two ways: mesh mode and path mode agree (to ordinary
numerical tolerance) when asked to evaluate the literal same four k-points on bulk Si
(`tests/test_calculation_berry.py::test_path_and_mesh_conventions_agree`); and on monolayer h-BN (broken
sublattice inversion symmetry, unlike Si), the occupied manifold's curvature is exactly antisymmetric
between the two physically inequivalent valleys K/K′ (-8.568 vs +8.568 Bohr² in §22's
 standard convention, agreeing to 0.01%) --
required by time-reversal symmetry and a much sharper check than Si's Chern number ($0=-0$ either way).
That h-BN run also surfaced two real pitfalls worth remembering for any future band-window usage: the
occupied-band count should come from Elk's own `EIGVAL.OUT` occupation numbers, not assumed from a total
(core + valence) electron count (core electrons aren't among the valence bands `nstsv` indexes at all);
and a single band's curvature can diverge near a point where it's degenerate with a *neighbouring
occupied* band, not just the first unoccupied one, even when the requested window's own outer boundary
stays gapped -- the fix is windowing the full degenerate group together, not picking a different single
band. Physics writeup (both modes, the FHS link-variable/plaquette-flux construction, and exactly what
is/isn't verified): `docs/design.md` §13 and `docs/physics.tex` (Part II).

Also implemented, as the third entry in the Fortran patch series:
`Calculation.eigenstate_session()` (task 9002, `patches/0003-eigenstate-session.patch`,
`src/elkpy_eigenstates.f90`) — an interactive, long-lived `elk` subprocess for fast repeated
eigenstate/overlap queries, plus one-off convenience wrappers `get_eigenstates(k)` and
`get_overlap(k_a, k_b, ist0, ist1)`. Initially proposed as an f2py in-memory bridge; rejected
because the dominant per-query cost is Elk's ground-state-dependent setup, not process-spawn
overhead, and making an f2py bridge robust would mean converting Elk's pervasive bare `stop`
calls into catchable errors across most of `vendor/elk/src/` — not achievable additively. The
persistent-subprocess design gets the same "stay warm" benefit through a stdin/stdout query
loop instead, with no new build machinery. `EIGENSTATES`/`OVERLAP` queries reuse
`elkpy_wfcorner` (task 9001's fresh on-the-fly diagonalisation) and a new
`elkpy_diagonalize` (the diagonalisation-only half of the same code, kept separate rather
than extending `elkpy_wfcorner` with optional arguments, since a Fortran `optional` dummy
argument needs an explicit interface at every call site, which task 9001's existing call
site doesn't have). Verified against a real compiled binary: `evecsv` is unitary at an
arbitrary k-point; `overlap(k, k, ...)` is the identity matrix to the ~1e-3 tolerance set by
`genwfsvp`/`genolpq`'s own real-space truncation error (not a bug — observed at the same
order for a single non-degenerate band too); a session survives many repeated queries; and
`overlap(Γ, Γ+e₁, ...)` matches task 9000's mesh-exported overlap for the same pair, checked
for one non-degenerate band only, since Si's bands 2-4 are degenerate at Γ and two
independent diagonalisations are free to (and empirically do) pick different bases within
that degenerate subspace (`tests/test_calculation_eigenstates.py`). Physics writeup (the
LAPW generalized eigenproblem, why `evecfv` isn't a valid raw-overlap basis but `evecsv` is
— only within one diagonalisation — and the cross-k overlap integral): `docs/design.md` §14
and `docs/physics.tex` (Part III).

Also implemented, and — unlike the three entries above — needing no new Fortran at all:
`Calculation.get_quantum_geometry(kpoints, ist0, ist1, directions=(1, 2), dk=)` — the full
quantum geometric tensor $Q_{ab}=g_{ab}-\tfrac i2 F_{ab}$ of a band window at an arbitrary
k-point: Berry curvature $F_{ab}$ (already covered by `get_berry_curvature_path()`) *and*
the quantum metric $g_{ab}$ (Fubini-Study/Provost-Vallee metric) it had been missing. Every
overlap this needs — including each loop corner's own self-overlap
`session.overlap(k, k, ...)`, the one new ingredient curvature alone never needed a name
for — is already exposed by the task 9002 interactive session, so this is driven entirely
from Python-side `EigenstateSession.overlap()` queries (19 per k-point: a 3x3 grid of
corners centered at k, needed for the centered stencil below) plus discretization
arithmetic in `parsers/quantum_geometry.py`; no `elkpy_quantum_geometry.f90`, no new task
number. The one real subtlety: Elk's `genolpq` overlap carries a ~1e-3 real-space
truncation floor that Berry curvature is immune to (it survives only in a closed-loop phase
product, which cancels a common-mode modulus deficiency exactly) but the quantum metric,
built directly from `1 - |overlap|^2`, is not — left uncorrected, this swamps the metric's
genuine `O(dk^2)` signal with a spurious offset that *diverges* as `dk` shrinks (measured
directly: raw g11 at dk=0.05→0.002 goes 40.5, 48.2, 52.5, 62.9, 131.8, i.e. blowing up).
The fix is exact Löwdin symmetric normalization (`M -> S_a^{-1/2} M S_b^{-1/2}` using each
corner's own self-overlap `S`) before computing anything from the raw overlaps — provably
inert for curvature (`S^{-1/2}` is Hermitian positive-definite, so it cannot shift
`arg(det M)`), and confirmed to fix the metric (same dk sweep, normalized: 40.5, 47.6,
49.7, 50.5, 51.0 — converging). Verified against a real compiled binary: that dk
divergence/convergence contrast on bulk Si; that curvature from `get_quantum_geometry()`
matches `get_berry_curvature_path()` on literally identical loop corners (task 9001's
corner-1 anchor, double the step size); and on monolayer h-BN (same structure as the Berry
curvature K/K′ check above), the metric is positive semi-definite at Γ/K/M/K′, curvature
reproduces the existing K/K′ antisymmetry via an entirely independent Python code path, and
at all four points $\det g \geq (F_{12}/2)^2$ — an exact theorem ($Q_{ab}=g_{ab}-\tfrac
i2F_{ab}$ is PSD as a 2x2 Hermitian matrix, being a sum of Gram-matrix-like
$\langle\cdot|Q|\cdot\rangle$ terms, so $\det Q\geq0$), not a loose plausibility band —
holds with margin at K/K′ and trivially at Γ/M, tying the new metric's scale to the
already-trusted curvature value without dividing by a curvature that vanishes at two of
the four points (`tests/test_calculation_quantum_geometry.py`,
`tests/test_quantum_geometry_gauge_invariance.py`). Also pinned against an analytically
known case, the spin coherent state / CP¹ Fubini-Study metric (Provost & Vallée 1980's own
worked example): converges to the exact $g=\tfrac14\mathrm{diag}(1,\sin^2\theta)$ and
$F_{\theta\phi}=-\tfrac12\sin\theta$ (§22's standard convention; this pin read $+\tfrac12\sin\theta$ until that sign was derived rather than taken from the code).

The metric's off-diagonal component $g_{12}$ was originally computed from a *forward*
polarization identity ($D(v_1+v_2)-D(v_1)-D(v_2)$, the same corners curvature's own
Wilson loop needed), which carried an $O(dk)$ error — small for $g_{11}/g_{22}$ but
amplified in $g_{12}$'s difference-of-three-comparable-quantities construction; measured
directly on h-BN's K/K′ valleys, the K-vs-K′ gap in $g_{12}$ shrank by a clean factor of
~2 per dk-halving (1.83, 0.97, 0.49), the textbook $O(dk)$ signature. Fixed by switching to
the standard *centered* mixed-partial stencil (both $\pm v$ corners: $g_{11}=[D(v_1)+D(-v_1)]/(2dk_1^2)$,
$g_{12}$ from the four diagonal corners $k\pm v_1\pm v_2$), which cancels $D(v)$'s
generically-nonzero cubic-order term exactly and gives $O(dk^2)$ error instead — confirmed
on noise-free synthetic data (a skewed CP¹ reparametrization with $g_{12}\neq0$
analytically): forward-stencil error ratio ~2 per halving, centered ~4, textbook $O(dk)$
vs $O(dk^2)$. This did more than fix the convergence order: for this
time-reversal-symmetric, non-spin-orbit structure, $\psi(-k)=\psi(k)^*$ makes
$D_{-k}(v)=D_k(-v)$ exactly, and since every centered-stencil term is manifestly invariant
under negating both displacements at once, $g_{ab}(-k)$ turns out to be the *identical
arithmetic expression* as $g_{ab}(k)$ — not just closely converged but exact up to
floating-point roundoff (measured K/K′ relative differences ~1e-13 to 1e-12, with no
residual dk-trend left to see in real Elk output at all). Curvature doesn't get this same
exactness (its own Wilson loop is left forward/anchored, deliberately not centered, since
it was already $O(dk^2)$-accurate): the same conjugation maps its anchored sub-loop to a
diagonally-opposite-quadrant loop rather than the identical one, so it keeps its ordinary
$O(dk^2)$ agreement (<1%), not machine precision. This exactness needs
$\psi(-k)=\psi(k)^*$ specifically and is not expected under `spinorb=True` (§17/§19's
spin/orbital-locking checks), where that relation doesn't hold in the same simple form.
Physics writeup (the quantum geometric tensor, the Marzari-Vanderbilt/Resta
discretization, the `Tr[P∂P∂P]` derivation, the Löwdin-normalization fix, the
centered-stencil derivation and the $D_{-k}(v)=D_k(-v)$ conjugation argument):
`docs/design.md` §15 and `docs/physics.tex` (Part IV).

Also implemented, as the fourth entry in the Fortran patch series:
`Calculation.get_atom_projection(k, ist0, ist1)` (task 9002's new `PROJECTION` query,
`patches/0004-atom-projection.patch`) — the atom-projection operator $P_\alpha$ (the
muffin-tin-sphere restriction of the identity) for every atom in the cell at once, as an
`nst`$\times$`nst` Hermitian matrix in the second-variational eigenbasis of one fresh
diagonalisation at $k$. Reuses upstream `wfmtsv.f90` unchanged — the same per-atom,
$(\ell m,\sigma)$-resolved muffin-tin wavefunction expansion and `wr2cmt` radial
quadrature weight that `gendmatk.f90` already uses for the `dos`/`bandstr` tasks'
atom/lm-projected DOS and band-character output — but generalizes `gendmatk`'s
per-state diagonal occupation-matrix entry to the full off-diagonal `nst`$\times$`nst`
operator (summed over all $\ell,m$, not $\ell$-resolved), computed as one `zgemm` per
spin channel rather than `gendmatk`'s explicit per-$(\ell,m)$ loop. All `natmtot`
matrices come from the same single diagonalisation (not one query per atom), since — as
with `evecsv` generally (`docs/design.md` §14) — matrices from separate diagonalisations
aren't guaranteed to share a basis, and the operator's main correctness identity
($\sum_\alpha P_\alpha + P_\text{interstitial} = \mathbb 1$) needs every atom projected
consistently. Verified against a real compiled binary: every returned matrix is
Hermitian and positive semi-definite (immediate from `wr2cmt` being a positive
quadrature weight); summing every atom and subtracting from the identity is still
Hermitian PSD (the interstitial remainder can't be negative), with each atom's own
weight a substantial, physically reasonable fraction of bulk Si's cell; diamond Si's two
atoms — related by inversion through the bond midpoint, which sends $k\to-k$ and
therefore fixes Γ — have identical weight on band 1 at Γ (non-degenerate; bands 2-4 are
degenerate there, the same caveat already documented for `evecsv`); a diagonal entry
agrees to 5 decimal places with an entirely independent Fortran code path — upstream
`bandstr.f90` task 21's own atom-projected band-character output, same `gendmatk`/
`wfmtsv` machinery but a separate call site and reduction — catching what the
Hermitian/PSD and sum-below-identity checks alone would miss (a silent undercount, e.g.
a packing bug, that's still internally consistent); a spin-polarized run (`nspinor`=2)
stays Hermitian PSD, exercising the spin-channel accumulation the unpolarized checks
above never run twice (SOC remains untested for this feature); and on monolayer
h-BN at $K=(1/3,1/3,0)$, the occupied valence-top ($\pi$) band is N-dominated and the
unoccupied conduction-bottom ($\pi^*$) band is B-dominated — the standard qualitative
picture for h-BN's band character (the more electronegative N pulling the bonding
state's weight toward itself), a sharp sign-of-the-effect prediction rather than a
plausibility band (`tests/test_calculation_atom_projection.py`). Physics writeup (the
projection operator, the exact muffin-tin/interstitial partition identity, why it's not
gauge-comparable across separate diagonalisations): `docs/design.md` §16 and
`docs/physics.tex` (Part V).

Also implemented, and — like `get_quantum_geometry()` — needing no new Fortran at all:
`Calculation.get_spin_operator(k, ist0, ist1)` / `EigenstateSession.spin_operator(k, ist0,
ist1)` — the spin operators $S_x$, $S_y$, $S_z$ (eigenvalues $\pm\tfrac12$) as
`nst`$\times$`nst` Hermitian matrices in the second-variational eigenbasis, applicable to
any wavefunction in a band window, the spin-space sibling of the atom-projection operators
above. The reason no Fortran is needed: Elk's second-variational scheme builds the spinor
Hilbert space as a literal product basis (the same `nstfv` first-variational spatial
orbitals reused unchanged for both spin channels — `evecsv` row `i = p + (ispn-1)*nstfv`,
confirmed directly against `eveqnsv.f90`), so a spin operator ($\mathbb
1_\text{spatial}\otimes\tfrac12\sigma_a$) is block-diagonal in the spatial index with no
muffin-tin partition, radial integral, or real-space wavefunction expansion involved at
all — its matrix elements reduce to plain inner products between `evecsv`'s already-computed
spin-up/spin-down row blocks (`parsers/spin.py`, pure NumPy). This means
`EigenstateSession.spin_operator()` issues no new task-9002 query either: it's a
`get_eigenstates(k)` call (already implemented) plus that linear algebra. Requires
`spinpol=True` or `spinorb=True` (`nspinor=2`, needed for the up/down block split to
exist) — raises `ValueError` immediately otherwise, before any Fortran query. Verified
against a real compiled binary: `sx`/`sy`/`sz` are Hermitian at a generic k-point; and, on
monolayer WSe2 with `spinorb=True` — broken inversion symmetry (unlike bulk 2H stacking, a
monolayer TMD has no inversion center) plus strong spin-orbit coupling locks the
valence-band-top spin to the valley index, so $S_z(K)=-S_z(K')$ (Xiao, Liu, Feng, Xu & Yao,
*Coupled Spin and Valley Physics in Monolayers of MoS2 and Other Group-VI Dichalcogenides*,
PRL 108, 196802 (2012)) — the RELATIVE sign is that published sign-of-the-effect prediction,
the same K/K' spirit as the Berry-curvature and atom-projection checks above; the ABSOLUTE
sign ($S_z(K)$ specifically negative) is a regression pin, not itself a physics prediction —
it depends on this structure's own conventions (chalcogen z-ordering, lattice-vector
handedness) and on which `evecsv` row block (§14's `i = p + (ispn-1)*nstfv`) is physically
"up". Measured: $S_z(K)\approx-0.4997$, $S_z(K')\approx+0.4997$ — nearly maximal
polarization, consistent with WSe2's unusually strong ($\gtrsim400$ meV) valence-band SOC
splitting (`tests/test_calculation_spin.py`). That row-block labelling is itself verified,
not just derived from reading `eveqnsv.f90`: collinear ferromagnetic Fe (`spinpol=True`, no
SOC) takes `eveqnsv.f90`'s block-diagonalization branch, which zeros the off-diagonal spin
blocks so bands `1..nstfv` are EXACTLY pure spin-up and `nstfv+1..nstsv` EXACTLY pure
spin-down by construction (machine-precision $S_z=\pm0.5$, no genolpq truncation floor
involved) — and, crucially, upstream `bandstr.f90` task 23 ("spin character of band", an
entirely separate Fortran code path via `gendmatk`/`wfmtsv`, not `evecsv` arithmetic at all)
independently confirms band 1 is the state ITS OWN printed column labels "spin-up" — closing
the one thing Hermiticity/su(2)/K-K′-antisymmetry structurally cannot catch: a global
up↔down relabelling flips every one of those checks identically, so none of them alone can
tell which physical spin `evecsv` row block 1 actually is. The arithmetic itself is
separately pinned on synthetic data (`tests/test_parsers_spin.py`): Hermiticity and the
su(2) commutation relation $[S_x,S_y]=iS_z$ (and the spin-$\tfrac12$ Casimir
$S_x^2+S_y^2+S_z^2=\tfrac34\mathbb 1$) hold for a random unitary `evecsv`, not just the
identity basis — expected, since $S_a$ in any eigenbasis is a similarity transform of the
fixed physical operator, which preserves commutators. Physics writeup (the product-basis
derivation, spin-valley locking, the Fe cross-checks): `docs/design.md` §17 and
`docs/physics.tex` (Part VI).

Also implemented, as the fifth entry in the Fortran patch series:
`Calculation.get_orbital_projection(k, ist0, ist1)` (task 9002's new `ORBITAL` query,
`patches/0005-orbital-projection.patch`) — the atom-projection operator generalized in the
opposite direction from §17's spin operators: resolved by angular-momentum channel
$\ell=0,1,2,3$ (s, p, d, f — `elkpy.session.ORBITAL_LABELS`) instead of summed over every
$(\ell,m)$ up to `lmaxo`. New Fortran subroutine `elkpy_orbitalproj` computes all four
$\ell$ matrices for one atom from a SINGLE fresh diagonalisation and a SINGLE `wfmtsv`
call — the same wavefunction expansion §16's `elkpy_atomproj` uses — then applies four
separate masked reductions of that one array, mirroring `gendmatk.f90`'s own per-$\ell$
loop (the machinery behind upstream `bandstr` task 21's $\ell$-resolved band-character
output) rather than `elkpy_atomproj`'s broadcast-across-the-whole-shell weight. Two
correctness details of that masking are load-bearing and invisible to a
Hermitian/positive-semi-definite check alone (any non-negative weight preserves both,
regardless of which angular-momentum range it's nonzero on): the masked weight array must
be explicitly zeroed before each $\ell$'s fill (unlike `elkpy_atomproj`, which writes
every entry and so never needed this), and the muffin-tin-interior region only
contributes when $\ell\le$`lmaxi` (default 1, so d/f get no interior contribution at
all) — mirroring `gendmatk.f90`'s own `if (l <= lmaxi)` guard exactly; skipping either
would silently read stale memory or the wrong radial shell's coefficients while staying
perfectly Hermitian and PSD. This projects onto an angular-momentum *channel* within the
muffin-tin sphere (every radial shell sharing that $\ell$), not a specific atomic valence
orbital's own radial shape — the two coincide for light, $sp$-bonded systems with no
close-lying semicore state of that $\ell$ (the only case verified here), but are not the
same object in general; the API is named and documented for the channel it actually
computes, not the everyday "outer shell" phrasing that motivated it. Verified against a
real compiled binary: every $(\alpha,\ell)$ matrix is Hermitian and positive
semi-definite; summing s+p+d+f and subtracting from a separate `get_atom_projection()`
call's total for the same atom is still Hermitian PSD (the g/h/i remainder can't be
negative); a diagonal entry matches upstream `bandstr.f90` task 21's own $\ell$-resolved
`BAND_Sss_Aaaaa.OUT` columns to 5 decimal places with **no** `lmaxdb` override needed —
task 21's own default, `lmaxdb=3`, is exactly s,p,d,f, unlike §16's atom-total check
which needed `lmaxdb=6` to match `lmaxo` — and this same cross-check still matches on a
spin-polarized run, closing a gap §16's own spin-polarized test leaves open (Hermitian/PSD
only there, no external reference exercising the `do ispn=1,nspinor` accumulation). On
monolayer h-BN at $K=(1/3,1/3,0)$, nitrogen's occupied valence-top ($\pi$) band is
p-channel-dominated ($p\approx0.522$, s/d/f $\approx0$, consistent with a N-$2p_z$ state)
while a much deeper bonding-$\sigma$ valence band on the *same* atom is s-channel-dominated
instead ($s\approx0.534$) — the dominant channel flips between the two bands of one atom,
a sharp sign-of-the-effect prediction rather than a plausibility band
(`tests/test_calculation_orbital_projection.py`). Physics writeup (the $\ell$-resolved
operator, the masked-reduction construction, the s+p+d+f-vs-total inequality, why this
isn't a valence-shell-specific projector): `docs/design.md` §18 and `docs/physics.tex`
(Part VII).

Also implemented, as the sixth entry in the Fortran patch series, and — like the spin
operators — needing almost no new Fortran of its own:
`Calculation.get_angular_momentum(k, ist0, ist1)` (task 9002's new `ANGMOM` query,
`patches/0006-angular-momentum.patch`) — the atomic orbital angular momentum operators
$L_x,L_y,L_z$, l-resolved (s,p,d,f), the vector-operator sibling of §18's
$P_{\alpha,\ell}$: where $P_{\alpha,\ell}$ projects onto an $\ell$ channel,
$(L_a)_{\alpha,\ell}$ mixes $m$ within it (the ladder-operator structure
$L_\pm|\ell,m\rangle\propto|\ell,m\pm1\rangle$). Rather than deriving that matrix
independently, `elkpy_angmomproj` calls upstream Elk's own `lopzflm.f90` (unmodified —
the same subroutine Elk's own on-site $\hat{\bf L}\cdot\hat{\bf S}$ trace, `dmatls.f90`,
already uses) to apply the ladder operators to each radial shell's $(\ell,m)$
coefficients, then reuses §18's exact `wr2cmt`-weighted, $\ell$-masked `zgemm` reduction.
$(L_a)_{\alpha,\ell}$ is Hermitian but, unlike the projection operators, not positive
semi-definite; the su(2) commutator $[L_x,L_y]=iL_z$ and Casimir
$L_x^2+L_y^2+L_z^2=\ell(\ell+1)\mathbb 1$ hold only on the untruncated
$(2\ell{+}1)\times(2\ell{+}1)$ analytic matrices, not as matrix products on the
band-window-truncated output — verified there via a pure-Python transcription of
`lopzflm`'s formula, kept only for this synthetic pin
(`tests/test_parsers_angular_momentum.py`), including a regression pin for the one bug
class Hermiticity cannot catch ($L_y$ is purely imaginary off-diagonal, so a lone sign
flip stays perfectly Hermitian but breaks the commutator). Verified against a real
compiled binary: every returned matrix is Hermitian; the $\ell=0$ channel is identically
zero; and on monolayer WSe$_2$ with `spinorb=True` (same structure as the spin-operator
check), $\langle L_z\rangle$ on W's d channel at the valence-band top obeys
$L_z(K)=-L_z(K')$ (Xiao, Liu, Feng, Xu & Yao, PRL 108, 196802 (2012) — same paper already
cited for $S_z(K)=-S_z(K')$, whose $\mathbf k\cdot\mathbf p$ model additionally predicts
a *pure* $m=\mp2$ valence-band state at $K/K'$) — measured $|L_z^{(d)}(K)|\approx1.1380$,
exactly $2\times$ W's own d-weight ($0.56900$, from §18), only possible for a pure
$m=\pm2$ state, a quantitative confirmation rather than just a sign check
(`tests/test_calculation_angular_momentum.py`). Physics writeup (the ladder-operator
formula, the `lopzflm` reuse, the truncation caveat, the valley-locking derivation):
`docs/design.md` §19 and `docs/physics.tex` (Part VIII).

Also implemented, and — unlike every other Fortran-patch-series entry above — needing
NO new Fortran at all:
`Calculation.get_z2_invariant(ist0, ist1, loop_direction=, pump_direction=, nkx=, nt=)`
— the $\mathbb Z_2$ topological invariant of a 2D time-reversal-invariant insulator's
occupied band window, via Wannier-charge-center (WCC) pumping (Yu, Qi, Bernevig, Fang &
Dai, *Equivalent expression of $Z_2$ topological invariant for band insulators using the
non-Abelian Berry connection*, PRB 84, 075119 (2011); Soluyanov & Vanderbilt's "largest
gap" crossing-counting method, *Computing topological invariants without inversion
symmetry*, PRB 83, 235401 (2011)). Built entirely by reusing task 9000's existing
mesh-neighbour overlap export (§13's `elkpy_berry.f90`, the same one
`get_berry_curvature()` uses for Chern numbers) as a non-Abelian (multi-band) Wilson
loop instead of a single `det`-phase link variable — deliberately *not* driven from the
arbitrary-k `eigenstate_session()` (§14), since closing the Wilson loop across the
Brillouin-zone boundary would then need a fresh, never-before-exercised assumption about
Elk's arbitrary-k diagonalization reproducing the exact periodic-gauge wavefunction at
$k+G$, where task 9000's mesh export instead reuses machinery whose periodic-boundary
handling is already empirically load-bearing for §13's exact-integer Chern numbers. All
WCC/$Z_2$ arithmetic (SVD link unitarization, the Wilson-loop product, the largest-gap
reference curve, the crossing-count parity) is pure Python (`parsers/wilson.py`),
independently unit-tested on synthetic data — gauge invariance, an exact single-band
phase pin, and, for the crossing-count logic specifically, a cross-check against an
unrelated already-trusted code path: a time-reversal-symmetric two-copy Qi-Wu-Zhang
lattice model (spin-up and its complex-conjugate spin-down partner, exactly opposite
Chern numbers by construction), where $Z_2$ equals the single-spin-sector Chern number
mod 2 whenever $S_z$ is conserved (Kane & Mele, PRL 95, 146802 (2005)) — checked
directly against `parsers.berry`'s own plaquette-flux Chern number on the identical
wavefunctions, after an earlier attempt at a hand-built "partner exchange" synthetic
trajectory gave an unexpected (but, on inspection, correct) $\nu=0$ due to a degenerate
mirror-symmetric edge case, not an implementation bug (`tests/test_wilson_gauge_invariance.py`).
A first attempt at `soc_scale={"C": 100.0}` (the value initially requested for this
feature) passed `check_gap()` (a real ~15 meV gap at $K$) but gave $\nu=0$ — traced, via
a cheap `eigenstate_session()` scan through $K$ (occupied-window overlap singular values
stayed below 1 even at $\delta k$ far finer than any practical mesh spacing), to mesh
aliasing: the Dirac-point anticrossing is only $\sim10^{-3}$ wide in fractional
coordinates, far narrower than a practical `nkx`'s mesh spacing, so `check_gap()`
passing (it only checks sampled points, not the region between them) gave false
confidence — not a bug in `get_z2_invariant()`/`parsers/wilson.py` itself. Raising
`soc_scale` to 3000 (real unscaled intrinsic carbon SOC is far too weak,
$\sim1\,\mu$eV-scale, to resolve on any practical mesh at all — this is a pure numerics
knob, not a change of physics, since Kane & Mele's QSH result holds for any nonzero
coupling) opens a ~1.4 eV gap at $K$ that the same diagnostic confirms is resolvable at
a practical mesh spacing. Verified against a real compiled binary at that scale: the
occupied $\pi$ band stays gapped from $\pi^*$ by ~1.4 eV at $K$, and `get_z2_invariant()`
on the full occupied valence manifold gives $\nu=1$ — Kane & Mele's own prediction,
relying on $Z_2$'s additivity mod 2 across independently-gapped band groups (§20) to
justify using the whole valence manifold rather than hand-isolating just the $\pi/\pi^*$
complex (`tests/test_calculation_z2.py`). A second, independent physical test needs no
`soc_scale` at all: freestanding buckled-honeycomb monolayer bismuth ("bismuthene",
2 atoms/cell, P-3m1 — same motif as buckled silicene/germanene), predicted a QSH
insulator by Murakami, PRL 97, 236805 (2006), via bismuth's own large *atomic* SOC.
Structure ($a=4.34$ Å, buckling $=1.73$ Å) and gap (0.555 eV without SOC / 0.500 eV
with SOC) from Cheng, Liu, Tan, Zhang, Wei, Lv, Shi & Tang, J. Phys. Chem. C 118, 904
(2014), confirmed independently in Freitas, Rivelino, de Brito Mota, de Castilho,
Kakanakova-Georgieva & Gueorguiev, J. Phys. Chem. C 119, 23599 (2015). A real-binary
scan of `get_eigenstates()` across Γ-K-M confirmed a band-inversion mechanism genuinely
distinct from graphene's Dirac-point-at-K picture: the gap minimum (~0.6 eV) sits at Γ
(an s-p-orbital inversion, HgTe/CdTe-style), not K (>2 eV there). `get_z2_invariant()`
on the full occupied valence manifold (30 bands) gives $\nu=1$ — confirming the method
on a second, structurally and mechanistically distinct QSH system.
Physics writeup (the non-Abelian Wilson loop, Kramers pairing at the two
time-reversal-invariant pumping endpoints, the largest-gap construction, the additivity
argument): `docs/design.md` §20 and `docs/physics.tex` (Part IX).

Also implemented, and — like §15/§17/§20 — needing no new Fortran at all:
`Calculation.get_z2_invariant_3d(ist0, ist1, nkx=, nt=)` — the full 3D strong/weak
$Z_2$ classification $(\nu_0;\nu_1\nu_2\nu_3)$ (Fu, Kane & Mele, PRL 98, 106803 (2007)),
generalizing §20's 2D `get_z2_invariant()` to a 3D time-reversal-invariant insulator by
computing that same 2D invariant on each of the Brillouin zone's 6 time-reversal-invariant
(TRI) planes ($k_i=0,\pi$ for $i=1,2,3$ — each plane is itself a genuine 2D
time-reversal-invariant system) and combining the six 0/1 results via FKM's own
$\nu_0=z(k_i{=}0)\oplus z(k_i{=}\pi)$ (any axis — an algebraic identity, checked and
raising `ValueError` on disagreement) / $\nu_i=z(k_i{=}\pi)$ formulas
(`parsers.wilson.combine_3d_invariants()`, unit-tested on synthetic data). The only new
plumbing: `get_z2_invariant()` gained a `plane_offset` parameter fixing the third
(non-loop/pump) direction at an arbitrary fractional coordinate via a one-point k-mesh
offset, plus a check that `self.vkloff` is exactly 0 in the loop/pump directions
(otherwise the pumping endpoints silently drift off the true TRI momenta).
Verified directly against the minimal lattice model Fu & Kane use to *introduce*
$(\nu_0;\nu_1\nu_2\nu_3)$ in the first place (PRB 76, 045302 (2007),
arXiv:cond-mat/0611341, their eq. 4/§IV.3, confirmed against the arXiv HTML source):
diamond structure (same as this project's Si tests) with the second basis atom
displaced along the cubic body diagonal [111] by a small $\delta$ (shortening one of the
four tetrahedral bonds, lengthening the other three) — a *trigonal* distortion reducing
the space group to symmorphic $R\bar3m$. Built with cesium (a single 6$s$ valence
electron, close to FKM's single-orbital tight-binding picture; not a real crystal phase
of cesium — chosen per this project's standing rule to ask Fable about material/structure
questions, see "Development practices" below) plus `soc_scale={"Cs": 3000.0}` (real
single-band SOC is expected too weak to resolve, same reasoning as §20's graphene test).
`get_z2_invariant_3d(1, ist1, nkx=12, nt=7)` gave $\nu_0=1$, apparently matching FKM's
$\delta t_1>0$ ("dimerized") prediction — **retracted, see §23**: the exact (mesh-free)
Fu-Kane parity indicator gives $(0;000)$ robustly across six independently-gapped band
windows, and refining the WCC mesh on a disputed plane oscillates $z=1,0,1,0$ rather than
converging, so the original number carried no information. This structure is trivial and
the FKM agreement was coincidental. The *implementation* is not impugned —
`combine_3d_invariants()`'s axis-split consistency held, and WCC still agrees with parity
in 2D (graphene) — the 3D six-plane sweep was simply run at too coarse a mesh
(`tests/test_calculation_z2_3d.py`, `tests/test_calculation_parity.py`).

Two dead ends/lessons along the way, both corrected rather than silently dropped:
freestanding gray tin, uniaxially strained along **[001]** (a *tetragonal* distortion,
space group $I4_1/amd$ — *nonsymmorphic*, unlike the [111]/$R\bar3m$ case above), showed
a gap pinned to $\sim10^{-6}$ eV at the strained zone boundary across four strains tried.
Initially over-concluded as proof the material isn't gapped; consulting Fable (per the
standing rule below) corrected this — $I4_1/amd$'s band sticking groups bands into
quartets without forbidding a gap at the actual filling (Watanabe, Po, Zaletel &
Vishwanath, PRL 117, 096404 (2016)), and published DFT confirms compressive [001]-strained
$\alpha$-Sn genuinely is a gapped TI (Huang & Liu, PRB 95, 201101(R) (2017)) — so this
probe was *inconclusive* (almost certainly measured a splitting inside a stuck quartet or
the semicore manifold), not a disproof. Separately, bulk Bi$_2$Se$_3$ (rhombohedral,
$R\bar3m$, sourced from a real deposited structure — Crystallography Open Database entry
9011965 — after an earlier *hand-converted* hexagonal-to-rhombohedral attempt gave a
self-contradictory ~11 Å "bond" from otherwise-correct literature parameters, establishing
this project's standing preference for database-sourced structures below) converged with
a correct, robust gap (0.258 eV at $\Gamma$) but gave $\nu_0=0$ on all six planes — wrong
relative to the literature's $(1;000)$; a narrower band window ruled out semicore
contamination as the cause, but mesh convergence was never tested (a materially denser
mesh costs several times the ~45 minutes already spent) — left as an open, explicitly
documented question, not asserted in any test. Physics writeup (the six-plane
construction, the FKM combination formulas, both dead ends/lessons in full, the
Bi$_2$Se$_3$ structure verification): `docs/design.md` §21 and `docs/physics.tex`
(Part X).

Also implemented, as patch 0007 — the seventh and newest in the Fortran patch series:
`Calculation.get_momentum_matrix(k)` / `EigenstateSession.momentum(k)` (task 9002's new
`MOMENTUM` query, `patches/0007-momentum-matrix-elements.patch`) — the momentum matrix
elements $p^a_{nm}=\langle\psi_n|(-i\nabla+\tfrac1{4c^2}[\vec\sigma\times\nabla V_s])_a|\psi_m\rangle$
for every pair of second-variational states at an arbitrary $k$-point, plus the eigenvalues
of that same diagonalisation. This is the missing *primitive* rather than one more
observable: §16-§19 built operators at arbitrary $k$ (atom, $\ell$-channel, spin, $L$) and
§13/§15 built geometric quantities from finite-difference overlaps, and the velocity
operator is what connects them — in atomic units, for Elk's **local** Kohn-Sham potential,
$\hat{\mathbf v}=\hat{\mathbf p}$ exactly, with `genpmatk`'s
$(1/4c^2)[\vec\sigma\times\nabla V_s]$ term keeping that true under `spinorb=True`
(Rathgen & Katsnelson, Physica Scripta T109, 170 (2004)). Upstream `genpmatk.f90` is reused
unmodified — the same subroutine Elk's own task-120 `PMAT.OUT` export (`putpmat.f90`) calls;
`elkpy_momentum` only substitutes a fresh on-the-fly diagonalisation + `genwfsv` expansion
for `putpmat`'s file-backed mesh eigenvectors, the same substitution patch 0002 already
makes for task 9001. Two `genwfsv` flags differ from `elkpy_wfcorner`'s and are load-bearing:
`tsh=.true.` (spherical harmonics, since `genpmatk` applies `gradzfmt`) and `tgp=.true.`
($G+p$ coefficients, since it takes the interstitial gradient in reciprocal space); with
`tgp=.true.` the `ngridg_`/`igfft_` arguments are unused inside `wfirsv.f90`, so this file's
coarse-grid `ngdgc`/`igfc` is equivalent to `putpmat`'s fine grid. `genpmatk`'s array is
hard-dimensioned `nstsv`, so there is no band-window variant to expose (windowing is
Python-side) — which suits the use anyway, since the Kubo sums below run over states
*outside* the window. Everything built on it is pure Python (`parsers/optical.py`,
unit-testable without an Elk run): the degree of circular polarization
$\eta=(|P_+|^2-|P_-|^2)/(|P_+|^2+|P_-|^2)$, $P_\pm=p^x_{cv}\pm ip^y_{cv}$; and the quantum
geometric tensor in Kubo form,
$T_{ab}=\sum_{n\in W,m\notin W}\langle n|v_a|m\rangle\langle m|v_b|n\rangle/(\varepsilon_n-\varepsilon_m)^2$
with $g_{ab}=\mathrm{Re}\,T_{ab}$, $F_{ab}=-2\,\mathrm{Im}\,T_{ab}$ — an entirely
independent code path for §13's Wilson-loop curvature and §15's finite-difference metric.
Note `parsers.optical`'s `directions` indexes **Cartesian** axes (genpmatk's components),
unlike `get_berry_curvature()`'s identically-named reciprocal-lattice argument. Hermiticity
is deliberately NOT tested here: `genpmatk` enforces it by construction (upper triangle
computed, lower set by conjugation, diagonal forced real), so it would say nothing about
this export path — the checks with teeth are the Hellmann-Feynman identity
$\mathbf v_{nn}=\partial\varepsilon_n/\partial\mathbf k$ against finite-differenced
eigenvalues (a genuinely separate code path; note that stepping in *fractional* coordinates
gives $\mathbf v\cdot\mathbf b_i$, not $v_i$) and the geometry cross-checks. Verified against
a real compiled binary on monolayer h-BN (`tests/test_calculation_momentum.py`): the
Hellmann-Feynman identity holds; $\eta(K)=-1.000000$, $\eta(K')=+1.000000$ to six figures —
the $C_3$-enforced perfect valley-selective circular dichroism (Yao, Xiao & Niu, PRB 77,
235406 (2008); Xiao, Liu, Feng, Xu & Yao, PRL 108, 196802 (2012) — the same paper §17/§19
already cite for $S_z$/$L_z$ valley locking; Cao et al., Nat. Commun. 3, 887 (2012) for the
measurement), the relative sign being the published prediction and the absolute sign a
structure-convention-dependent regression pin; the Kubo curvature agrees with
`get_berry_curvature_path()` to 1.2% in magnitude; and the result is stable between
`nempty=12` and `nempty=20`, so the state-sum truncation is checked rather than assumed.
Synthetic pins (`tests/test_parsers_optical.py`) use the massive Dirac model, where
$\mathbf p=\partial H/\partial\mathbf k$ is exact: $\eta=\pm1$ at the valleys, curvature
against the closed form $\pm1/(2\Delta^2)$, metric against $\mathbb 1/(4\Delta^2)$.

**A sign error this cross-check found and fixed — elkpy now uses ONE Berry-phase
convention everywhere.** The standard $\mathbf A=i\langle u|\nabla_{\mathbf k}u\rangle$,
$\Omega=\nabla\times\mathbf A$ (Xiao, Chang & Niu, RMP 82, 1959 (2010)) — Wilson-loop
curvature (§13), quantum-geometry curvature (§15), Chern numbers, and the Kubo form (§22)
all agree. `parsers.berry` previously omitted the King-Smith–Vanderbilt/Resta negation
$\gamma=-\mathrm{Im}\ln\prod_j\langle u_j|u_{j+1}\rangle$ (required because
$\langle u|u+\delta\cdot\nabla u\rangle=e^{-i\mathbf A\cdot\delta}$ makes the closed
loop product $e^{-i\oint\mathbf A\cdot d\mathbf l}$), so its `curvature` was $-\Omega$ and
its `chern_number` sign was flipped. Found three ways, all agreeing: on a synthetic massive
Dirac model the Kubo route matches direct numerical differentiation to 10 significant figures
while `parsers.berry` on the identical eigenvectors gave exactly minus that; on real h-BN the
same factor of $-1$ appeared end-to-end ($+8.101$ vs $-8.194$ at $K$); and §15's
Provost-Vallée spin-1/2 pin had asserted $F_{\theta\phi}=+\tfrac12\sin\theta$ where the
derivation gives $-\tfrac12\sin\theta$ ($A_\phi=-\sin^2(\theta/2)$, integrating to
$-2\pi$, the spin-1/2 monopole charge) — that test had been calibrated to the code rather
than to the paper. It survived this long because every check was sign-blind (Si's Chern number
is $0=-0$; h-BN is a *relative* K/K′ antisymmetry; $Z_2$ is a parity; $\det g\geq(F/2)^2$ is
even in $F$). **The fix is one function**, `parsers.berry._berry_phase()`, which every
consumer routes through, so the convention is set once and cannot diverge again. $Z_2$
(§20/§21) is untouched: `parsers.wilson` builds its own Wilson loop from
`parse_berry_overlaps()`'s raw overlaps and never consumes `flux`, and a crossing parity is
invariant under reflecting the WCC curves anyway. **A positive side-finding**: because the
same $-1$ appeared in pure Python *and* end-to-end, the error was localized entirely in that
Python step, which **confirms the Fortran `moverlap`/`genolpq` conjugation convention** —
previously resting only on a `zgemv` BLAS-semantics derivation with no runtime test. **Note
for cross-project work**: pyqula's own `berry_curvature` carries the same omission
(`topologytk/overlap.py`'s `uij(wf1,wf2)[i,j] = <wf1_i|wf2_j>` is the same M(a,b) convention,
and `topology.py` takes `arctan2(Im det, Re det)/(4 dk^2)` of the identical counterclockwise
link product with no negation), so elkpy's curvature/Chern signs now differ from pyqula's —
a deliberate choice of the published convention over cross-project agreement, worth
propagating to pyqula rather than reverting here. Also corrected in passing:
`berry.py`/`quantum_geometry.py`/`calculation.py`/this file labelled Berry curvature
"Bohr⁻²" — flux is dimensionless and the plaquette area is Bohr⁻², so it is **Bohr²**; only
the labels were wrong, never the numbers. Physics writeup (the velocity
operator identity, the $C_3$ selection rule, the Kubo derivation, the sign analysis):
`docs/design.md` §22 and `docs/physics.tex` (Part XI).

Also implemented, as patch 0008 — the eighth in the Fortran patch series:
`Calculation.get_parity(k, ist0, ist1)` / `get_fu_kane_invariant(ist0, ist1, dimension=)`
(task 9002's new `PARITY` query) — the inversion operator
$P_{mn}=\langle\psi_m|\hat I|\psi_n\rangle$ at a time-reversal-invariant momentum, and the
Fu-Kane symmetry-indicator $Z_2$ built from it (Fu & Kane, PRB 76, 045302 (2007)):
$\delta_i=\prod_m\xi_{2m}(\Gamma_i)$, $(-1)^{\nu_0}=\prod_{i=1}^{8}\delta_i$,
$(-1)^{\nu_k}=\prod_{k_i=\pi}\delta_i$. The cheap counterpart to §20/§21's WCC pumping —
**8 diagonalisations instead of a mesh sweep**, and *exact* rather than convergent — at the
cost of requiring an inversion centre, so the two are complementary rather than redundant.
No new symmetry algebra: the transformation of first-variational coefficients is lifted from
upstream `getevecfv.f90` (exercised by every `reducek=1` run), and Elk makes it simpler than
expected — `findsymcrys.f90` moves inversion to symmetry element 2 **with zero translation**
whenever `tsyminv` is true (clearing the flag otherwise, having shifted the basis to put the
inversion centre at the origin), so upstream's translation branch drops out; `rotzflm` already
handles the improper rotation ($\det R<0\Rightarrow(-1)^\ell$ via `ylmrot`). The overlap needs
nothing new either: because $\hat I\mathbf k\equiv\mathbf k$ at a TRIM the rotated coefficients
live in the SAME LAPW basis, so `genwfsv`+`genolpq` at $q=0$ (§14's `OVERLAP` path, with the
muffin-tin phase set directly to unity) gives the matrix — no basis-overlap matrix required.
Three traps, all now guarded: parity eigenvalues are NOT `pmat`'s diagonal (TRIM spectra are
degenerate, so the diagonalisation returns an arbitrary basis within a multiplet — use
`parsers.symmetry.parity_eigenvalues()`); with `nspinor=2` Kramers partners share a parity
eigenvalue, so the product over ALL occupied states is identically $+1$ and carries no
information — the one-per-pair counting $\delta=(-1)^{N_-/2}$ is the entire content of the
formula; and a window boundary sitting INSIDE a band group passes every other check
(Hermitian, $\pm1$, even Kramers counts) while describing no topological group at all
(`check_window_gap()`, added after `ist0=19` on the cesium structure did exactly that and
returned a confident $\nu_0=1$). Unlike §19's su(2)/Casimir identities, $P^2=\mathbb 1$ DOES
survive the band-window truncation — $[\hat I,\hat H]=0$ makes a gapped window an invariant
subspace rather than a mere slice — so it is usable as a runtime assertion. A global sign error
in $P$ cannot propagate: every invariant is a product over an EVEN number of TRIM, observed
directly when different windows returned all eight $\delta_i$ flipped with $\nu_0$ unchanged.
Verified against a real compiled binary: graphene (`soc_scale=3000`, §20's fixture) gives
$\nu=1$ from 4 k-points, matching `get_z2_invariant()`; h-BN (no inversion centre) and a
non-TRIM k-point are both refused with the session left alive
(`tests/test_calculation_parity.py`). Requires `spinorb=True`, now enforced rather than merely
documented.

**This feature retracted §21's cesium result.** The [111]-dimerized diamond structure is
topologically TRIVIAL, $(0;000)$, not the $\nu_0=1$ §21 reported: the parity indicator gives
$(0;000)$ across six independently-gapped windows (including one dropping a semicore block
below a 69 eV gap, exercising $Z_2$ additivity mod 2), while refining the WCC mesh on a
disputed plane gives $z=1,0,1,0$ for $(n_{kx},n_t)=(12,7),(18,9),(24,13),(32,17)$ — an
oscillation, so the original $n_{kx}=12$ sample carried no information. The disagreement
localizes cleanly (all three $k_i=0$ planes differ, all three $k_i=\pi$ agree), and the plane
is robustly gapped (min direct gap 0.19 eV over a 13x13 scan), so this is NOT §20's
graphene mesh-aliasing mode — the crossing count is simply under-resolved. Physically, a Cs
diamond lattice with SOC scaled 3000x does not realize FKM's single-orbital tight-binding
phase; the earlier agreement was coincidental. The WCC *implementation* is not impugned (its
axis-split algebra held, it agrees with parity on graphene, and 2D is separately validated on
bismuthene) — the 3D six-plane sweep at a practical mesh is. This also makes
under-convergence the leading explanation for §21's other open item, Bi$_2$Se$_3$'s
$\nu_0=0$ against a literature $(1;000)$; that is settleable in minutes by parity (it is
centrosymmetric) once its structure is re-sourced from COD 9011965 as a checked-in fixture —
not done here. Physics writeup (the FKM formulas, the Kramers-pairing derivation, the
even-TRIM sign immunity, why $P^2=\mathbb 1$ survives truncation, the retraction):
`docs/design.md` §23 and `docs/physics.tex` (Part XII).

## Architecture

- `src/elkpy/structure.py` — `Structure`: lattice vectors (`avec`, Bohr) + species, each atom either a
  bare `(x,y,z)` position or a `(position, bfcmt)` pair (per-atom magnetic field, manual sec. 5.2).
  `species_files` optionally overrides the `{symbol}.in` species-filename convention. `from_ase()`/
  `to_ase()` convert Angstrom/Cartesian ASE `Atoms` (optional dependency).
- `src/elkpy/calculation.py` — `Calculation`: owns one run directory and the ground-state-defining
  parameters (`xc`, `spinpol`, `spinorb`, `soc_scale`, `rgkmax`, `ngridk`, `extra_blocks` for anything
  else — e.g. `maxscl`). `soc_scale={"Fe": 1.5}` requires `spinorb=True` and needs the
  `patches/0001-per-species-soc-scale.patch` Fortran extension applied (i.e. a binary built via
  `scripts/build_elk.sh` after the patch was added — see git log for when). `get_*` methods block and
  can be expensive — real Elk subprocesses, not in-memory work.
  `ensure_ground_state()` reuses a prior task-0 run only if a JSON manifest (`.elkpy_manifest.json`)
  shows the basis/structure/functional-defining parameters (including `extra_blocks`) and the Elk
  binary identity are unchanged; sampling parameters (e.g. a denser `ngridk` passed to `get_dos()`) are
  free to differ. Every other `get_*` runs via `_run_resumed()` in its own **wiped-clean** subdirectory,
  never in `self.workdir` — this isn't just tidiness, it's load-bearing correctness: some tasks (phonon
  DFPT, task 205) treat the mere presence of prior output files as "already done" and resume from them,
  so a stale subdirectory from an earlier killed/crashed run silently corrupts the next result instead
  of erroring (hit this for real while implementing phonons — see git log). `converged` property +
  `raise_on_nonconvergence=` control whether non-convergence raises or must be checked explicitly.
- `src/elkpy/spec.py` — version-coupled knowledge (task codes, `xctype` codes, output filenames) as
  data, each entry cross-checked against `vendor/elk/src/` (not just the manual) — an Elk version bump
  should mean editing this one file.
- `src/elkpy/inputfile.py` — generic `elk.in` block writer (block name + value lines) and `read_blocks()`
  reader (used to parse `GEOMETRY_OPT.OUT`, which Elk writes in the same block syntax).
- `src/elkpy/launcher.py` — `LocalLauncher`: `run()` is blocking local subprocess execution, used by
  every task except the eigenstate session; `start_session()` instead returns a non-blocking `Popen`
  with stdin/stdout pipes, for `Calculation.eigenstate_session()` (task 9002) to drive interactively.
  Refuses `nprocs > 1` since `build-config/make.inc` builds serial (`mpi_stub.f90`) — that combination
  would silently launch N racing copies into one directory, not parallelize.
- `src/elkpy/session.py` — `EigenstateSession`: owns the interactive task-9002 subprocess started by
  `eigenstate_session()`, sending `EIGENSTATES`/`OVERLAP`/`PROJECTION`/`ORBITAL` queries over stdin and
  parsing responses off stdout until closed (context manager) or told to `QUIT`. See `docs/design.md`
  §14 for why this is a persistent worker process rather than an f2py in-memory bridge, §16 for
  `PROJECTION` (atom-projection operators), §18 for `ORBITAL` (l-resolved s/p/d/f projectors,
  `ORBITAL_LABELS`), §22 for `MOMENTUM` (momentum/velocity matrix elements — the one query
  here that takes no band window, since `genpmatk`'s array is hard-dimensioned `nstsv`), and
  §23 for `PARITY` (the inversion operator at a TRIM, for the Fu-Kane $Z_2$ indicators).
- `src/elkpy/parsers/` — one small module per output file family (`info`, `totenergy`, `band` — reused
  for phonon dispersion, since `PHDISP.OUT` shares `BAND.OUT`'s exact layout — `dos`, reused for phonon
  DOS, `forces`, `geometry`, `effmass`, `volumetric`), each verified against real Elk output, not
  assumed from the manual. `berry` is the exception to "just a parser": it also does all of the
  Wilson-loop/Chern-number arithmetic in Python (`compute_berry_curvature()`), deliberately kept out of
  Fortran so it's unit-testable against synthetic overlap matrices (`tests/test_berry_gauge_invariance.py`)
  without an Elk run. `eigenstates` parses `EigenstateSession`'s stdout token stream (not a file) into
  energies/`evecsv`/overlap/atom-projection/orbital-projection arrays, independently unit-testable the
  same way (`tests/test_parsers_eigenstates.py`). `wilson` is the same "arithmetic, not
  just parsing" exception as `berry`, reusing `berry.parse_berry_overlaps()`'s output as
  a non-Abelian (multi-band) Wilson loop instead of a single link-variable phase, for
  `get_z2_invariant()` — see §20. `optical` is the same exception again: it turns the
  `MOMENTUM` query's raw matrix elements into circular dichroism and the Kubo-form quantum
  geometric tensor, pinned on synthetic massive-Dirac data (`tests/test_parsers_optical.py`)
  without an Elk run — see §22. `symmetry` is the same again for the `PARITY` query: parity
  eigenvalue extraction and the Fu-Kane $Z_2$ counting, with the band-window gap guard
  (`check_window_gap`) that a Kramers-parity check alone cannot supply — see §23.
- `src/elkpy/config.py` — locates the built `elk` binary (`build/elk/src/elk` by default, override via
  `ELKPY_ELK_BIN`) and the species directory (`vendor/elk/species/` by default).

## Project purpose

A Python interface to Elk, an all-electron full-potential linearized augmented-plane-wave (LAPW)
density-functional-theory (DFT) code written in Fortran. On top of the interface, this project adds
extra functionality that Elk itself does not provide.

## Development practices

Whenever the user asks for a new piece of functionality to be implemented, checking arXiv for a
relevant paper is encouraged where it plausibly helps — not limited to new physics — to ground the
implementation in an actual published source (method, formalism, convention, algorithm) rather than
guessing. This is a standing option to reach for, not something that needs to be requested each time.

This applies with the most force to new physics (a new Fortran capability, a new formula, a new
numerical scheme — not routine wrapping of an existing Elk task): checking arXiv for the relevant
method/paper is encouraged where it fits the task, to ground the implementation in the actual
published formalism (e.g. matching sign/normalization conventions, confirming which approximation a
term corresponds to) rather than guessing from the code alone.

Whenever a new formalism is added or an existing one is modified (new physics, a changed formula, a
different numerical scheme — not routine wrapping of an existing Elk task), update the documentation
describing it in both forms: the Markdown docs (`docs/design.md`/`docs/roadmap.md` or wherever the
capability is described) and the corresponding LaTeX writeup, added as a new `\part{}` (with a `\label`)
inside the single shared file `docs/physics.tex` — not a new `.tex` file per addition. Both must be
physics-focused, not just an API description: state the relevant formula(e), define each symbol, and
explain the physical meaning/approximation being made (what it captures, what it neglects, how it
relates to the underlying published method) — not merely "this function computes X". Each writeup must
also include a "how to use in code" part showing the actual elkpy call(s) (e.g. `Calculation(...)`, the
relevant `get_*()`) that exercise the formalism, so the physics and the API surface stay tied together.
Keep both in sync with the code in the same change — don't defer either to a follow-up.

When a real material's crystal `Structure` is needed (lattice vectors + atomic positions), prefer
pulling it from an actual structure database/file (a CIF from the Crystallography Open Database or
Materials Project, a published paper's POSCAR/Quantum Espresso input, etc.), loaded via ASE
(`Structure.from_ase()`) where possible, over hand-deriving it from reported lattice
parameters/Wyckoff positions. A hand conversion (e.g. hexagonal-to-rhombohedral primitive vectors from
a,c and a Wyckoff z-parameter) is an extra, error-prone derivation step even when every input number is
correct — hit for real building Bi2Se3's rhombohedral cell, where a wrong hand-derived transformation
matrix gave physically nonsensical bond lengths (~11 Å instead of ~3 Å) despite starting from correct
literature z-parameters; re-deriving it wasted significant real-DFT compute chasing a structure that
was never right. When a database/file source isn't available or a hand derivation is unavoidable,
numerically verify the result (e.g. actual computed bond lengths/layer spacing against known physical
values) before running any DFT on it, not just before trusting the final answer.

Whenever picking which real material to use for something (a demonstration, a test, choosing between
candidate structures), or trying to understand something about a material in terms of its structure
(crystal symmetry, distortion geometry, why a particular structure does or doesn't have a given
property), ask Fable (the `fable` model, e.g. via `Agent(..., model="fable")`) rather than relying
solely on your own judgement or a general-purpose research agent.

## README and notebook style

Whenever the user gives style feedback on `README.md`/`notebooks/` (or documentation
style generally) — a correction, a preference, a "make it more like X" — record it
durably in this section (or add a new section here) as part of that same change, not
just apply it to the current diff and let it lapse next time. This section is itself
the product of that process (see git log) and is the standing reference to keep
current, not a one-time writeup.

`README.md` and `notebooks/` follow [`pyqula`](https://github.com/joselado/pyqula)'s style
(the same physics-code-lineage project this one borrows its `Structure`/`Calculation`
object-model naming from) — physics-first, not an API/engineering writeup:

- **README**: pyqula's `SUMMARY`/`INSTALLATION`/`FUNCTIONALITIES`/`EXAMPLES` section
  structure (all-caps `#`-level headers). `FUNCTIONALITIES` is a short bullet list per
  category, each bullet a physics statement with its defining formula where one is
  illuminating — not a paragraph explaining how it's implemented (patch series, Fortran
  file names, caching, subprocess architecture; that belongs in `docs/design.md`, not
  the README). List elkpy's own physics — the capabilities genuinely beyond stock Elk
  (currently: per-species spin-orbit scaling, Berry curvature/Chern numbers, arbitrary-k
  eigenstates/overlaps) — before the routine wrapping of Elk's standard DFT workflow
  (energy/bands/DOS/forces/relaxation/phonons/...), which gets one condensed "also
  wraps" mention, not equal billing. `EXAMPLES` pairs a short code snippet with a real
  PNG generated from an executed notebook cell (`images/`, extracted via
  `nbformat`+`base64`, same pattern as pyqula's own `images/*.png` gallery) — never a
  hand-drawn or synthetic figure. No "Project layout"/directory-tour section — that
  reads as internal engineering documentation, not user-facing README material. Every
  `FUNCTIONALITIES` bullet ends with a `[[notebook]](notebooks/NN_name.ipynb)` link to
  the notebook that demonstrates it — same inline-link-per-bullet pattern pyqula's own
  README uses (see pyqula's `FUNCTIONALITIES` section) — so a reader goes straight from
  the one-line physics claim to the worked example, not just from a separate summary
  table at the bottom of the page.
- **Notebooks** (`notebooks/`, one per feature area, table linked from the README):
  pyqula's `jupyter-notebooks/*/main.ipynb` rhythm — a one-line "This notebook shows
  how to compute X" title cell, minimal imports, then repeating
  `[markdown: formula + one clause defining symbols] → [code: 2-6 terse lines, one #
  comment per line] → [plot]`. Cut engineering context rather than compress it into
  shorter prose; where a mechanism genuinely matters to a result (e.g. a value that's
  silently wrong if you get it from the wrong place), it becomes a `#` comment on the
  line it affects, not a markdown paragraph. Concretely, this means no standalone
  "verify the result" cell — a Hermiticity check, an identity/partition check
  (`sum_alpha P_alpha + P_interstitial = 1`), an eigenvalue-sign check — sitting between
  the formula and the headline calculation with no plot of its own: that correctness
  check already lives in `tests/` and is asserted in the prose of `docs/design.md`/
  `docs/physics.tex`, so repeating it in the notebook is exactly the kind of engineering
  context pyqula's own notebooks don't carry (see e.g. `jupyter-notebooks/08_chern_insulator/
  main.ipynb`: Hamiltonian → bands/curvature/Chern number → plot, nothing else). A
  notebook cell should either feed the next cell or feed a plot; if it does neither,
  cut it. When a quantity is naturally a function of k (or of some other physically
  meaningful axis), prefer showing it that way over a bar chart of one or two isolated
  numbers — e.g. a band structure colored by an operator's expectation value (a
  spin/orbital-texture plot), not `ax.bar(["K", "K'"], [...])` — even when the
  headline physics claim is about just two points; a categorical comparison across
  atoms/orbital channels (not indexed by k) is the one case where a grouped bar chart
  is still the right call (see [[feedback_notebook_plots_not_bar_charts]] in the
  auto-memory). Every notebook runs against a real compiled Elk binary and is checked
  in with its actual output cells — the one exception is DFPT phonons, left unexecuted
  with a note on why (~11-13 min/call) and the command to run it yourself. Add a new
  notebook (and a README table row + `FUNCTIONALITIES` link) alongside any new physics
  capability, same trigger as the `docs/physics.tex` writeup rule above.
- **LaTeX gotchas hit in practice**: matplotlib's mathtext needs braced arguments
  (`\mathbf{r}`, not `\mathbf r` — the latter raises `ParseFatalException` at render
  time, not at notebook-generation time, so it only surfaces when a cell actually
  executes); keep every inline math span's `$...$` balanced without splitting a token
  across delimiters (`$K'=-K$`, not `K$'=-$K`, which opens/closes math mid-token and
  renders garbled on both GitHub and in Jupyter); don't cram two separate relations
  into one display equation ending in a trailing comma that runs into unrelated prose
  on the next line — end a display equation cleanly and give the second relation its
  own sentence.

## Core constraint: isolate changes to vendored Elk source, don't avoid Fortran

Elk's own Fortran source will be vendored into this repository (not treated as an installed system
dependency), so that the Python interface has a fixed, buildable copy to target. Because upstream Elk
is expected to be swapped for a newer release in the future, **changes must stay isolated** — Fortran
itself is a fine implementation choice for new capability; what's constrained is how it touches the
vendored tree:

- `vendor/elk/` always stays byte-for-byte what was downloaded from upstream — never edited directly,
  including `make.inc`. All building and any Fortran changes happen from a separate out-of-tree copy
  (see `docs/design.md` §8).
- Prefer Elk's existing export tasks (matrix elements, wavefunction/Wannier90 export, `STATE.OUT`
  post-processing — see `docs/design.md` §8) for new physics when they're sufficient; this is cheaper
  and carries zero Fortran risk, not a mandate to avoid Fortran altogether.
- When new Fortran is genuinely needed, prefer additive new files (new modules/subroutines) over
  editing existing upstream files. When hooking into existing control flow is unavoidable (e.g. the
  task dispatch in `elk.f90`), keep the edit to the smallest possible footprint, clearly marked, and
  track it as one hunk in a maintained patch series applied to the build copy — never committed as a
  direct change to `vendor/elk/`.
- New functionality should live in Python or in clearly separated new Fortran files rather than being
  folded into Elk's existing modules, so the patch series stays small and easy to re-evaluate against a
  new upstream version.

## Commands

- Build Elk out-of-tree (copies `vendor/elk/` to `build/elk/`, applies `patches/*.patch` if any, drops
  in `build-config/make.inc`, builds — never touches `vendor/elk/`):
  `./scripts/build_elk.sh`. Must be run before any test/example that actually invokes Elk.
  Serial by design — `make -j` races on an implicit ordering dependency in Elk's own `src/Makefile`
  (stub files like `mpi_stub.f90`/`libxcifc_stub.f90` must compile before the modules that `use` them,
  and that isn't expressed as an explicit prerequisite upstream); this is a pre-existing upstream
  issue, not something to fix by editing `vendor/elk/`.
- Install elkpy (editable): `python3 -m pip install -e .`
- Run tests: `python3 -m pytest tests/`. Run a single test: `python3 -m pytest tests/test_calculation_si.py::test_get_bands`.
  `tests/test_calculation_*.py` are integration suites that run the real `elk` binary on bulk Si/Fe —
  they self-skip if `build/elk/src/elk` doesn't exist yet, so run `./scripts/build_elk.sh` first to
  actually exercise them. `tests/test_structure.py`'s ASE round-trip test self-skips if `ase` isn't
  installed (`pip install -e .[ase]`).
- `tests/test_calculation_si_phonons.py` (DFPT phonon dispersion/DOS) is skipped by default even with
  the binary built — confirmed ~11-13 minutes per test on a minimal 2-atom, `ngridq=(2,2,2)` grid, cost
  dominated by DFPT's per-perturbation-per-q-point work, not anything elkpy controls. Set
  `ELKPY_RUN_SLOW_TESTS=1` to actually run them.
- `build-config/make.inc` targets GNU Fortran + OpenBLAS/LAPACK + FFTW, serial (no MPI); edit it (not
  `vendor/elk/make.inc`) to change compiler/library configuration.
