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
Not implemented: symbolic k-path's disconnected-segment support (`,` breaks), scheduler-backed
launchers, MPI. (Potential and ELF volumetric plots now DO have named methods —
`get_potential()`/`get_elf()`, tasks 43/53 — as do the dielectric function and MOKE, tasks 121/122;
and the classical supercell phonon method, task 200, is no longer DFPT-only — see §32 below.)
Check `src/elkpy/` directly rather than assuming the docs describe current code; update both as
they diverge.

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
**and that explanation is now confirmed**: bulk Bi$_2$Se$_3$, §21's other open item, gives
$(1;000)$ by parity — $\delta(\Gamma)=-1$ and $+1$ at all seven other TRIM — the accepted
literature answer, on the SAME ground state §21's failed WCC sweep used (gap 0.2575 vs
0.258 eV at $\Gamma$), window-independent across seven windows including §21's own narrow
bands-61-78 one that had reproduced the wrong answer. The delta pattern matches the known
single-$\Gamma$ band-inversion mechanism (Zhang et al., Nature Physics 5, 438 (2009)), not
merely the known parity. A real material, no `soc_scale`, robust 0.26 eV gap: a sharper
confirmation of the under-convergence thesis than the cesium retraction itself. Structure
from COD 9011965 via `spglib.standardize_cell(to_primitive=True)` (a library
transformation, not a hand derivation); `tests/test_calculation_bi2se3_parity.py`, gated
behind `ELKPY_RUN_SLOW_TESTS=1`. Physics writeup (the FKM formulas, the Kramers-pairing derivation, the
even-TRIM sign immunity, why $P^2=\mathbb 1$ survives truncation, the retraction):
`docs/design.md` §23 and `docs/physics.tex` (Part XII).

Also implemented in a parallel batch, each cross-validated against an independent code
path rather than against itself — §§24-27, with `docs/physics.tex` Parts XIII-XV:

- **`get_dielectric_function()` (task 121) and `get_circular_absorption()`** (§24) —
  polarization-resolved interband absorption summed over a k-mesh from §22's `MOMENTUM`
  query. The polarization-summed spectrum reproduces Elk's own task 121 to 2e-5 median /
  9.3e-4 max on h-BN. Matching it required `dielectric.f90`'s ACTUAL finite-`swidth`
  response, not the textbook Lorentzian-δ form, which differs by a derived
  $(2\omega-\Delta)/\Delta$ factor and diverges as $1/\omega^2$ below the edge — hence
  `broadening="elk"` (the default). Zone-integrated $\varepsilon_+=\varepsilon_-$ to 4e-14
  (time reversal forces it, since the mesh is closed under $k\to-k$), so valley dichroism is
  invisible in a full-zone integral and only appears valley-restricted: $\eta(K)=-0.9999$,
  $\eta(K')=+0.9999$.
- **`get_effective_mass_sum_rule()`** (§25) — the k·p sum rule
  $(1/m^*)^{ab}=\delta_{ab}+2\sum\mathrm{Re}[p^ap^b]/(\varepsilon_n-\varepsilon_m)$, checked
  against `get_effective_mass()` (task 25, finite differences of eigenvalues): +3.1% on Si,
  converging monotonically FROM ABOVE (16.1→7.7→4.3→3.5→3.1% as retained states go
  4→85), the direction the sign argument demands since every omitted state lies above band
  $n$. NOTE the denominator is $\Delta\varepsilon$ to the FIRST power, unlike §22's Kubo
  sums — measured truncation error 36% vs 15% from identical states. Its per-band
  decomposition gives Si's $\Gamma_1$ mass exactly zero (~1e-30) from the even
  $\Gamma_{25'}$ triplet — the parity selection rule, which finite differences cannot
  show. Thomas-Reiche-Kuhn is a BZ-AVERAGED statement, not pointwise: pointwise
  $f^{ab}=\delta_{ab}-(1/m^*)^{ab}$ identically.
- **Spin Berry curvature / spin Hall** (§26, patch 0009) — `MOMENTUM` now also returns
  `evecsv`, at no extra cost (the array was already computed and discarded), which is what
  makes $J^z_a=\tfrac12\{S_z,v_a\}$ legal: §17's spin operators are built from `evecsv`,
  and combining them with §22's velocity across two diagonalisations would silently mix two
  arbitrary resolutions of a degenerate multiplet (§14) while passing every Hermiticity and
  unitarity check. Verified on graphene: $\Omega^s(K)=+\Omega^s(K')$ — the SAME sign at both
  valleys, since spin Berry curvature is EVEN under time reversal, the opposite relative
  sign to every other K/K′ check here and something a J↔v swap could not produce.
- **`get_potential()` / `get_elf()` / `get_moke()`** (§27, tasks 43/53/122) — routine
  wrapping, closing roadmap Tier 3 item 4. `v_xc` verified as a pointwise function of $n$
  against the analytic Dirac-exchange formula; the Kerr angle verified odd under
  magnetization reversal on two independent ground states.

Infrastructure from the same batch: `.github/workflows/ci.yml` (roadmap Tier 6 — unit
tests, the patch-series check §8 asks for, a build job; note `patch` exits 0 on a FUZZY
apply, so a fuzz tripwire is included, which earlier "applies cleanly" checks lacked);
a flock-based semaphore in `launcher.py` bounding concurrent `elk` processes machine-wide
(`ELKPY_MAX_CONCURRENT`, default 4) plus OpenBLAS/MKL thread pinning, since Elk links
`-lopenblas` whose threading is NOT governed by `OMP_NUM_THREADS` when built against
pthreads; and `config.py` now diagnoses a non-editable `pip install .`, which put the
module in site-packages and made every default path nonsense with no indication why.

**Open, deliberately not addressed**: the graphene `soc_scale=3000` fixture is METALLIC
(per-k occupancies 6,8,8,8,8,8,9), confirmed independently twice. The tested window is a
legitimately gapped 6-band group, so §20's and §23's WCC/parity cross-validation stands,
but §20's "spans every occupied valence band" prose overstates it. That is a physics claim
and wants its own investigation. Relatedly, the `sum(occ > 0.5)` idiom used by several
fixtures reads the count at the FIRST k-point and is only valid when the filling is
k-independent.

Also implemented, as patch 0010: **rotation-eigenvalue symmetry indicators** (§28) —
`SYMLIST` and `SYMMETRY` queries giving Elk's crystal symmetries and the operator
$\langle\psi_m|\hat O_{\rm isym}|\psi_n\rangle$ at any k-point the operation fixes,
generalizing §23's inversion (`elkpy_symop`), plus `parsers/indicators.py` with the
Benalcazar-Li-Hughes corner-charge indices (PRB 99, 245151 (2019), arXiv:1809.02142
Eqs. 3/4/14; general framework Po-Vishwanath-Watanabe, arXiv:1703.00911). Same
"skip the intermediate model" idea as §13/§20/§23, aimed at the
DFT→irvsp/vasp2trace→Bilbao pipeline. Two restrictions enforced in FORTRAN, not just
documented, since violating either returns a plausible matrix that unitarity and
$\hat O^n=\mathbb 1$ both pass: `nspinor=1` (a rotation on a spinor needs the SU(2)
factor upstream's first-variational transformation never applies — inversion escaped this
by acting trivially on spin) and zero translation (a glide/screw gives $\hat O^n$ = phase
× unity).

**Two practical findings.** `tshift=False` is MANDATORY: Elk relocates the origin by
default and, when the crystal has inversion, puts the *inversion centre* there — for a
honeycomb that is the bond midpoint, not the $C_3$ axis, so every rotation becomes
non-symmorphic and is refused. Measured on graphene: 8/24 symmorphic with the default
shift and no $C_3$ available; 24/24 with `tshift=False`. Since $Q$ is defined relative to
a chosen rotation centre, this is how the centre gets chosen, not a technicality. And
graphene is unusable for the K-point indices at all without SOC — its bands touch at K
(gap $4.3\times10^{-7}$ Ha, the Dirac point), so `check_window_gap` correctly refuses;
h-BN is the usable $C_3$ case.

**Status: the operator is verified, the corner charge is NOT, and no test asserts one.**
On h-BN: all 12 symmetries symmorphic, $C_3$ unitary with $\hat O^3=\mathbb 1$ to
~1e-3 (§14's genolpq floor), counts summing correctly, $[\Gamma_p]=0$ by construction,
measured $\Gamma[2,1,1]$, $K[1,1,2]$, $K'[1,2,1]$. But K and K′ are time-reversal
partners, so their eigenvalues are complex conjugates — exchanging the $p=2$ and $p=3$
bins that $Q^{(3)}=\frac e3[K_2^{(3)}]$ distinguishes. The same data therefore gives
$Q=0$ at K and $Q=e/3$ at K′, depending on how the generator ($R$ vs $R^{-1}$) pairs with
the corner. This is the direction ambiguity §23 could defer because inversion is its own
inverse. TWO discriminators were tried and both FAIL: $D(9)^2=D(3)$ holds under either
convention ($C_3$ is abelian, so inversion-of-argument is still a homomorphism), and the
$C_{2z}=\hat I\sigma_h$ identity against §23's parity cannot help because order-2
operations are self-inverse. The h-BN physics expectation (obstructed atomic limit, the
occupied $\pi$ band being N-centred) favours $e/3$ — but "the expected answer is nonzero
and one of our conventions is nonzero" is not evidence, and §21's cesium is exactly what
that reasoning costs. **The concrete fix, not done**: §19's $L_z$ pins the sense of
rotation, since a $C_n$ eigenvalue is $e^{-2\pi im/n}$ — cleanest with an atom ON the
axis, which h-BN lacks.

Also implemented, and — like §15/§17/§20/§21 — needing **no new Fortran at all**:
`Calculation.get_exchange_tensor(i, j, magnetic=, spin=, components=)` (§29) — the full
anisotropic magnetic exchange tensor $J_{ij}^{\alpha\beta}$ of a pair, by **four-state energy
mapping** (Šabani, Bacaksiz & Milošević, PRB 102, 014457 (2020), arXiv:2002.10861; method from
Xiang, Kan, Wei, Whangbo & Gong, PRB 84, 224429 (2011), arXiv:1106.5549), with
`parsers/exchange.py` doing the arithmetic and decomposing into isotropic Heisenberg exchange,
the Dzyaloshinskii-Moriya vector, and the symmetric anisotropy carrying the Kitaev $K$/$\Gamma$
terms. Elk has no exchange-parameter capability at all, so this is genuinely new physics on top
of the interface.

The reason no Fortran is needed: Elk already does constrained non-collinear DFT via
`fsmtype=-2` + per-atom `mommtfix` (`bfieldfsm.f90` projects the constraining field perpendicular
to the target direction, so magnitude relaxes freely and the field does no work once on target),
and — the load-bearing fact — **its reported total energy is the constrained functional with no
field contribution**: `energy.f90:226` forms `engykn = evalsum - engyvcl - engyvxc - sm` where
`sm = ∫m·B_s` uses the TOTAL effective field, into which `addbfsm.f90` has already folded the
constraining field. Three settings are load-bearing rather than cosmetic: `nosym`/`reducek=0`
(the four states break symmetry, `checkfsm.f90` hard-`stop`s on a non-invariant `mommtfix`, and
an identical k-set across the four states is what makes their systematic errors cancel);
`epsengy` (Elk's default 1e-4 Ha ≈ 2.7 meV exceeds the whole signal — elkpy defaults 1e-8); and
`bfcmt` seeding (the perpendicular constraining field can rotate a moment but not create one).
Two published traps are handled explicitly: Xiang's original DM formula carries an extra sign
valid only for an already-antisymmetric Hamiltonian, so all nine components go through one
formula and $\mathbf D$ comes from the antisymmetric part afterwards — making $\mathbf D=0$ on
an inversion-symmetric bond a real null test; and the multiplicity $m_{ij}$ (periodic images of
the flipped site) is computed by `check_supercell`, not assumed to be 1.

**Verification status — deliberately incomplete, unlike every other entry above.** The Python
arithmetic is pinned by 16 synthetic tests (`tests/test_parsers_exchange.py`): exact round trip
for all nine components, the cancellation claim tested against a model containing single-ion
anisotropy plus spectator couplings, and Šabani's correction to Xiang reproduced. Against a real
binary, only isolated and PARTIAL pieces: a NiO perpendicular two-site probe had all moments on
target to ~1° at a healthy 1.80 μB, but was interrupted after ~4 SCF cycles and never converged —
so it shows the constraint locking on fast (unlike an earlier bcc Fe probe, whose moments
collapsed to ~0.16 μB over 60 cycles), not that it holds to convergence, and `check_constraint()`
has never fired on a completed run. Elk's `bfcmt` sign convention was measured properly (a single
bcc Fe atom with `bfcmt=(0,0,+4)` converges to −2.75 μB, i.e. antiparallel).
**The nine-component NiO sweep has NOT been run** — `ELKPY_RUN_SLOW_TESTS=1 python3 -m pytest
tests/test_calculation_exchange_nio.py`, of order 60 CPU-hours at ~150 s/SCF loop × 40 loops ×
36 configurations. Its three assertions carry the real teeth: $J_2$ antiferromagnetic and ~20 meV;
the tensor **exactly isotropic with SOC off** (convention-free — a global spin rotation is a
symmetry, so any deviation measures the noise floor); and $\mathbf D=0$ on the
inversion-symmetric J₂ bond. Treat the feature as unvalidated end-to-end until that runs.

**Known limitation**: `mommtfix` constrains the SPIN moment only. For a $j_{\rm eff}=1/2$
pseudospin system such as α-RuCl₃ that is insufficient — the moment is 2/3 orbital, and Hou,
Xiang & Gong (PRB 96, 054410 (2017), arXiv:1612.00761) measure that spin and orbital directions
deviate seriously unless both are constrained. §29 documents the shape a patch would take
(Elk's `bforb` term at `eveqnsv.f90:96-105,160-168` already applies $\tfrac1{2c}\mathbf B\cdot
\hat{\mathbf L}$ per atom, so only the field source is hard-wired), including that it would need
its own total-energy correction since that term enters the second-variational Hamiltonian rather
than `bsmt`. Not implemented. Physics writeup: `docs/design.md` §29 and `docs/physics.tex`
Part XVI (note: §28's own promised Part XVI was never written — pre-existing gap, so this new
part takes that number).

Also implemented, as patch 0011 — the eleventh entry in the Fortran patch series and the first since
0002/0003 to add a new `.f90` file: **spin-polarised STM images** (§30) —
`Calculation.get_spin_stm()` / `get_spin_stm_3d()` (tasks 9003/9004, `src/elkpy_stm.f90`), the
Tersoff-Hamann differential conductance seen by a magnetic tip,
$dI/dV(\mathbf r)\propto n(\mathbf r,E_F+eV)+P_T\,\mathbf m(\mathbf r,E_F+eV)\cdot
\hat{\mathbf e}_T$ (Tersoff & Hamann, PRB 31, 805 (1985); the spin-polarised extension is
Wortmann, Heinze, Kurz, Bihlmayer & Blügel, PRL 86, 4132 (2001)), with $\hat{\mathbf e}_T$ an
arbitrary Cartesian direction. Driveable entirely from a plain `elk.in` — task 9003/9004 plus the
`elkpy_stmdir`/`elkpy_stmpol`/`elkpy_stmbias`/`elkpy_stmint` blocks — see `examples/spin-stm/`,
which needs no Python at all; that is a deliberate part of the feature, not a nicety.

The reason it needs so little Fortran: **Elk already computes $\mathbf m(\mathbf r,E)$ and
throws it away.** Upstream task 162 (`wfplot.f90`) makes its spin-summed STM image by replacing
the second-variational occupations with an energy-selecting weight and calling `rhomagv`, which
fills the global `rhomt`/`rhoir` *and* `magmt`/`magir`, symmetrises both and converts both to the
fine grids — then plots the charge density alone. `elkpy_stm.f90` is that same routine with the
magnetisation kept, forming $f_1=n$, $f_2=\mathbf m\cdot\hat{\mathbf e}_T$,
$f_3=n+P_Tf_2$ pointwise (the projection is linear, so it commutes with the muffin-tin
spherical-harmonic expansion — no re-expansion, no new radial integral) and handing all three to
Elk's generic `plot2d`/`plot3d`. $f_1,f_2$ are complete: a different $P_T$ is a recombination,
never a re-run. Two modes: a smeared delta at $E_F+eV$ (dI/dV map) or the smeared window between
$E_F$ and $E_F+eV$ (constant-current) — the latter deliberately with **no** $1/w$ factor, being a
state count rather than a density of states. `ndmag=1` (collinear) means only $\hat e_z$ can
contribute, which is the correct answer, so it prints a note rather than failing.

**An upstream bug found by the normalisation check.** `rhomagv` passes `wkpt(ik)` to `rhomagk` as
a separate argument and `rhomagk` multiplies by it, so `occsv` must be a pure occupancy exactly as
`occupy.f90` stores it. Upstream's task-162 branch folds `wkpt(ik)` in a second time — its image
carries $w_{\mathbf k}^2$; the same file's task-61/62/63 branch sets `occsv = 1/wkpt` precisely to
cancel that factor, which is what makes the double counting visible. Harmless overall scale on an
unreduced mesh, a genuine distortion with `reducek /= 0`. elkpy omits it, and that is what lets the
cell integral be checked absolutely.

**Verification** on a freestanding Cr monolayer with the triangular lattice of Cr/Ag(111) — Elk's
own `examples/magnetism/Cr-monolayer` with the vacuum opened to 24 Bohr and `reducebf` switching
the seed fields off — which orders in a coplanar 120° Néel state (Kurz, Förster, Nordström,
Bihlmayer & Blügel, PRB 69, 024415 (2004)), the system SP-STM was proposed for and has since been
simulated in detail (Palotás, Hofer & Szunyogh, PRB 84, 174428 (2011)). Three chemically identical
atoms with moments 120° apart make every check exact rather than a plausibility band
(`tests/test_calculation_spin_stm.py`, 13 tests): the spin-summed LDOS is **identical**
above all three (spread <1e-6) while the $\hat x$ projection gives ratios $0:+1:-1$ and the $\hat
y$ projection $-1:+\tfrac12:+\tfrac12$ — rotating the tip changes which sublattice is dark, the
signature by which a non-collinear structure is identified experimentally (Gao, Wulfhekel &
Kirschner, PRL 101, 267205 (2008)); the projection vanishes at the hollow site for any tip; a
$\hat z$ tip is an **exact null** (5e-7 of the charge LDOS — without SOC the seed's $m_z=0$
subspace is invariant under the Kohn-Sham flow), which is the check that the Cartesian component
requested is the one returned rather than $|\mathbf m|$; a $(1,1,1)$ tip reproduces
$(\mathbf m\!\cdot\!\hat x+\mathbf m\!\cdot\!\hat y+\mathbf m\!\cdot\!\hat z)/\sqrt3$
from three separate runs to 2.5e-7; and $|\mathbf m\cdot\hat{\mathbf e}|\le n$ pointwise, which a
$\boldsymbol\sigma$-vs-$\mathbf S$ factor-of-2 slip would break (Elk's $\mathbf m$ is
$n_\uparrow-n_\downarrow$, twice §17's $\langle\mathbf S\rangle$). The **absolute
normalisation** — the one thing every check above, being a ratio or a null, leaves free — comes
from `src/occupy.f90`'s `FERMIDOS.OUT`, the DOS at $E_F$ computed as a sum over eigenvalues with no
density and no plotting machinery at all: agreement of order 1e-5 (measured 1.2e-5 and 3.8e-5 on
two cells), the same order as the density representation's own total-charge error. That is the check that caught the `wkpt` double counting,
as an exact factor of 2. Finally, against upstream task 162 on an unreduced mesh the pointwise
ratio is constant to 1e-6 *and equals $N_{\mathbf k}$ exactly* — a quantitative confirmation that
the whole difference between the two routines is that one weight. Physics writeup: `docs/design.md`
§30 and `docs/physics.tex` Part XVII.

Also implemented, as patch 0012 — the twelfth entry in the Fortran patch series:
**vertical tunnelling transport through a two-dimensional material** (§31) —
`Calculation.get_vertical_transport()` (task 9005, `src/elkpy_transport.f90`), which computes
what gets *through* the sheet rather than what an STM tip sees above it (§30). A point tip at
$\mathbf r$ above the material, an infinite featureless metallic substrate plane below it: the
current is then set by the **nonlocal** Green's function between the two, since a point tip makes
$\Gamma_{\rm t}$ rank one and collapses the Landauer trace exactly to
$T(\mathbf r;E)=\int_{\rm plane}|G(\mathbf r,\mathbf r';E)|^2d^2r'$. The band sum happens *before*
the modulus: different bands are different routes through the material and they add as amplitudes.
Driveable entirely from a plain `elk.in` (task 9005 plus the `elkpy_transport_exit`/`_window`/
`_kgrid`/`_koffset`/`_sdir`/`_spol` blocks and the usual `plot2d`) — see
`examples/vertical-transport/`.

Because the substrate is invariant under every lateral lattice translation it conserves
$\mathbf k_\parallel$, so the k-sum is **incoherent** and only bands at the same $\mathbf k$
interfere. What survives is one Hermitian Gram matrix per k-point,
$S_{\mathbf k}[n,n']=\int_{\rm plane}\psi^*_{n\mathbf k}\hat P_{\rm s}\psi_{n'\mathbf k}$, and one
quadratic form. In the interstitial (both planes are in vacuum) Elk's wavefunction is an exact
plane-wave sum, so **$S_{\mathbf k}$ is computed in closed form** — the in-plane G-vector
orthogonality collapse, one gather and one `zgemm` per k-point, nothing sampled and nothing
converging. The k-mesh is the task's own (`elkpy_transport_wf` diagonalises fresh, the
`bandstr.f90` route), so it is independent of `ngridk` **and of `reducek`** — which also avoids a
failure mode a reduced wedge would carry here, the reduction using $z$-flipping operations whose
Gram matrix is the overlap on the *mirror-image* plane.

**Two conventions are taken from `defumat`'s measurements rather than rediscovered.** The literal
resolvent denominator **cannot be evaluated by a truncated band sum** — the far-from-$E$ states
build the barrier's evanescent decay entirely by cancellation (measured cancellation ratio 349 on
a completely diagonalised cell) — so the amplitude is the on-shell one,
$a=\psi\sqrt{f_{\rm occ}\delta_\eta/\eta}$, which is *exactly* the resolvent's modulus and keeps
the interference between bands degenerate at the tip energy. And $G$ carries $\psi$ conjugated in
the **exit** variable, so the contraction is $\sum a_na^*_mS[n,m]$, not $\sum a^*_na_mS[n,m]$; the
transposed version is real, non-negative, blind to a degenerate rotation and exactly right in the
Tersoff-Hamann limit, so only a literal $\int|G|^2$ quadrature separates them.

All arithmetic is Python (`parsers/transport.py`), like `berry`/`wilson`/`optical` — which is what
makes an energy sweep free (neither the wavefunctions nor $S_{\mathbf k}$ depend on $E$) and the
formula unit-testable. This is the one deliberate asymmetry with §30: `examples/spin-stm/` needs
no Python at all, `examples/vertical-transport/` needs a dozen lines of it, because putting the
contraction in Fortran too would put one formula in two languages.

**Verification.** No code computes this quantity for comparison (QE's `PWCOND` is a Landauer
transmission between two semi-infinite *crystalline leads* — no point contact, therefore no map),
so the checks close inside the package. The **Tersoff-Hamann limit is exact rather than
approximate**: `exit_region="cell"` makes every $S_{\mathbf k}$ the identity and $T$ becomes the
tunnelling density of states — agreeing with `get_spin_stm()`'s (§30, via `rhomagv`) to **1.6e-5 of
the peak with NO factor**, which is the only check a wrong overall normalisation could not hide in,
every other one here being a ratio or a null. Setting the tip plane *equal to* the exit plane makes
the exported amplitudes sample the exit plane itself, so the closed-form $S_{\mathbf k}$ can be
checked against a literal rectangle rule (**7e-15**) and the contraction against a literal
$\int|G|^2$ (**8e-15**, where the transposed convention differs). **The physics is an exact
symmetry statement**, sharpened by Fable: the group that matters is not K's little group but its
subgroup of **$z$-preserving** elements, since anything flipping $z$ maps the substrate plane onto
the tip side. For **monolayer graphene** that is $C_{3v}$, whose $\sigma_v$ exchanges the two
sublattices without flipping $z$ — the Dirac doublet is still the 2D irrep $E$, so by Schur
$S_K\propto\mathbb 1$: measured eigenvalues 0.05242094 and 0.05242103, **equal to 8.5e-7**, exactly
2.000000 open channels, zero off-diagonal weight, correlation with the tunnelling image >0.999.
(`defumat`, a plane-wave pseudopotential code, gets $\mathrm{diag}(0.05405086,0.05405084)$ for the
same object — cross-code agreement in the *value*.) For the **AB bilayer** the only element
exchanging the pair is an in-plane $C_2'$, which flips $z$; the intact group $C_3$ is abelian and
Schur gives nothing, the states being the non-dimer sites of two *different* layers. Measured with
equal standoffs: $S_K$ eigenvalues 3.33e-5 and 4.36e-2, a **ratio of 1311**, half of
$S_{\mathbf k}$ off-diagonal, correlation down to 0.81, incoherent map 74x above the coherent one.

**Traps, all guarded in Fortran rather than documented**: a plane cutting a muffin-tin sphere (the
plane-wave part is not the wavefunction there and returns a plausible number — note the radius that
matters is `checkmt`'s, which shrinks carbon from 1.80 to 1.32 Bohr); `tshift=False` is **mandatory**,
the same trap as §28, since Elk otherwise moves the origin onto the inversion centre while both
plotting planes stay in the input frame (measured: the sheet moves from $z$=0.5 to 0 and both planes
end up on the same side); the material must lie *between* the planes. Two limitations are documented
rather than guarded: `rgkmax` bounds how far into the vacuum tail the plane-wave representation is
meaningful ($S_{\mathbf k}$ stays Hermitian/PSD while becoming meaningless), and a k-mesh without K
gives graphene a map that is **identically zero, not small**. Not implemented, deliberately: the
resolvent method, a finite contact patch, a tilted exit plane, plane-to-plane geometry, and any
absolute conductance (the two lead couplings are unfixed prefactors — what this delivers is the map
and its contrast). The **magnetic substrate** ($\hat P_{\rm s}=1+P_{\rm s}\hat{\mathbf n}\cdot
\boldsymbol\sigma$, §30's normalisation so $P_{\rm s}=0$ gives the plain spin-summed Gram matrix)
is implemented and structurally exercised, but has **no end-to-end physics test**: the natural one
is spin-layer locking in a 2H WSe2 bilayer or an A-type AFM bilayer, which is 10-30x the graphene
cost. Physics writeup: `docs/design.md` §31 and `docs/physics.tex` Part XVIII.

Also implemented, and unlike every entry above **not new physics at all** — it makes the physics
Elk *already has* reachable by name (`docs/design.md` §32, no `physics.tex` part and no notebook,
per the routine-wrapping rule below): six task-family mixins under `src/elkpy/tasks/`, composed in
`tasks/__init__.py`'s `ALL_MIXINS` and unpacked by `class Calculation(*ALL_MIXINS)`. **143 of the
146 live task codes `vendor/elk/src/elk.f90` dispatches on now sit behind a named method (97.9%)**,
up from about twenty; `Calculation` exposes 115 `get_*` methods. Coverage is measured against the
dispatch, not claimed — a code counts only when a named method actually puts it in a task list,
since `run_tasks()` could always reach all of them, which is exactly what this improves on. The
three exceptions: task 2 (`geomopt` from atomic densities — the capability is wrapped, `get_relaxed()`
emitting task 3, the same subroutine reading `STATE.OUT`) and tasks 201/271, restarts that read
files a *previous* run of the same task left behind, which the wiped-subdirectory invariant (§4)
cannot supply. New coverage includes: stress/strain and the piezoelectric and magnetoelectric
tensors; the electric field gradient and Mössbauer hyperfine parameters; X-ray and magnetic
structure factors; Hartree-Fock, RDMFT and DFT+U tensor moments; molecular dynamics; the complete
1D/2D/3D plotting triples (magnetisation, B_xc and its divergence, the m×B_xc torque density, the
electrostatic field, paramagnetic current, meta-GGA W_xc, wavefunctions); Fermi surfaces in all
five Elk representations plus the nesting function; l-, (l,m)-, spin- and moment-resolved band
character and partial DOS; second-harmonic generation, the Bethe-Salpeter chain, linear-response
and real-time TDDFT, and the spin response whose transverse poles are the magnon energies; the
**classical supercell phonon method (task 200 — the only route for a magnetic cell**, since
`phonon.f90` hard-stops on `spinpol`), Born effective charges, LO-TO splitting, electron-phonon
coupling, the Eliashberg function and gap equations; magnetic anisotropy energy, spin spirals by
both the supercell and generalised-Bloch routes, GW, Wannier90 export and the ultra-long-range
family. Alongside it, `src/elkpy/params.py` records all 315 `case(...)` branches of
`readinput.f90` as data — Fortran variable, type, value shape, default, the branch's own range
checks, and a description — with a validator, a renderer, and a browsable surface
(`describe`/`search`/27 categories) re-exported from `elkpy` directly; its completeness test
re-parses `readinput.f90` and fails in **both** directions, so a version bump reports exactly which
blocks appeared or vanished. `spec.py` grew from 23 to 152 task codes, 27 to 190 output filenames
and 3 to 22 filename templates. **Verification is uneven and deliberately labelled per method** in
`docs/design.md` §32 and in each docstring: some methods were exercised end-to-end against the real
binary, many are format-derived (task ordering, blocks and output layouts transcribed from
`vendor/elk/src/`, parsers unit-tested against fixtures built from the cited `write` statements,
but never run), and a few are structurally unrunnable in this build (meta-GGA `W_xc` needs libxc,
which `build-config/make.inc` stubs out; Wannier90's `.amn`/`.mmn` need `libwannier`). Treat a
format-derived method as untested until its first real run.

## Architecture

- `src/elkpy/structure.py` — `Structure`: lattice vectors (`avec`, Bohr) + species, each atom either a
  bare `(x,y,z)` position or a `(position, bfcmt)` pair (per-atom magnetic field, manual sec. 5.2).
  `species_files` optionally overrides the `{symbol}.in` species-filename convention. `from_ase()`/
  `to_ase()` convert Angstrom/Cartesian ASE `Atoms` (optional dependency).
- `src/elkpy/calculation.py` — `Calculation`: owns one run directory and the ground-state-defining
  parameters (`xc`, `spinpol`, `spinorb`, `soc_scale`, `rgkmax`, `ngridk`, `extra_blocks` for anything
  else — e.g. `maxscl`). `soc_scale={"Fe": 1.5}` requires `spinorb=True` and needs the
  `patches/0001-per-species-soc-scale.patch` Fortran extension applied (i.e. a binary built via
  `build_elk.sh` after the patch was added — see git log for when). `get_*` methods block and
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
  should mean editing this one file. 152 task codes, 190 output filenames, 22 filename templates.
- `src/elkpy/tasks/` — one mixin module per Elk task family (`groundstate`, `spectra`, `optics`,
  `phonons`, `magnetism_manybody`, plus `params`), composed in `tasks/__init__.py`'s `ALL_MIXINS`
  and unpacked by `class Calculation(*ALL_MIXINS)` — that tuple is the single place the composition
  is written down. No mixin holds state, defines `__init__`, or imports `..calculation`, and
  `Calculation`'s own methods come first in the MRO, so nothing a mixin defines can shadow the
  existing surface. `_ndmag()` (Elk's number of magnetisation components, from `init0.f90`) and its
  `_block_floats`/`_block_flag` helpers live on `Calculation` itself, NOT in a mixin: several
  families need it and two independent transcriptions of one Fortran rule drift apart. See
  `docs/design.md` §32, including the three non-wiping run mechanisms and what is only
  format-derived.
- `src/elkpy/params.py` — every `elk.in` input block `readinput.f90` accepts (all 315 `case(...)`
  branches) as a data table: Fortran variable, declared type and module, value shape, default, the
  branch's own range checks, and a description. Plus a validator, an exact-text renderer (working
  around two `inputfile.py` bugs — see its module docstring) and a browsable surface
  (`describe`/`search`/`categories`), re-exported from `elkpy` directly. Its completeness test
  re-parses `readinput.f90` and fails in BOTH directions, so a version bump names the delta.
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
  §23 for `PARITY` (the inversion operator at a TRIM, for the Fu-Kane $Z_2$ indicators),
  and §33 for `LAPW` (`lapw_problem(k)` — every ingredient of the first-variational LAPW
  eigenproblem at one k-point: the $\bf G+k$ set, `apwalm`, the derivative matrices $D$,
  $H$ and $O$ with their interstitial parts separated, and Elk's own `evalfv`/`evecfv`;
  an export for the JAX port, and the only route to `apwalm`, which nothing upstream
  writes).
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
  without an Elk run — see §22. `exchange` is the same exception again: the four-state formula, the
  isotropic/DM/symmetric decomposition, the Kitaev parametrisation and the supercell
  multiplicity bookkeeping, all pinned on synthetic data without an Elk run
  (`tests/test_parsers_exchange.py`) — see §29. `stm` is a plain parser again, for the spin-polarised STM task's own
  `ELKPY_STMDOS.OUT` cell integrals — the images themselves go through Elk's generic
  `plot2d`/`plot3d` writers and so through `volumetric` (`parse_plot2d` added there, whose
  grid ORDER is the load-bearing detail: `plotpt2d.f90` runs the first plotting vector's index
  fastest, so a column reshapes as `(n2, n1)` and getting it backwards transposes every image
  without changing a number). `symmetry` is the same again for the `PARITY` query: parity
  eigenvalue extraction and the Fu-Kane $Z_2$ counting, with the band-window gap guard
  (`check_window_gap`) that a Kramers-parity check alone cannot supply — see §23.
- `src/elkpy/exchange.py` — orchestration for four-state energy mapping: builds each
  configuration's `mommtfix` block and seeded `bfcmt`, runs each as its own `Calculation` in
  its own subdirectory (so the ground-state manifest cache applies per configuration and an
  interrupted sweep resumes), threads the sweep over configurations, and — critically —
  `check_constraint()` re-reads each converged run's muffin-tin moments and refuses any that
  drifted off target, since `fsmtype=-2` lets a magnitude collapse and the resulting energy
  looks perfectly plausible. See `docs/design.md` §29.
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

## JAX port (Workstream B): status, and the memory discipline it runs under

`docs/jax_port.md` (1,623 lines) is the design study, `docs/continue_here.md` §3 the cold-start
summary, `docs/jax_port_phase0.md` the running log of what Phase 0 measured,
`docs/jax_port_phase1.md` the same for Phase 1 (through §1m), and
`docs/jax_port_phase2.md` for Phase 2, which is under way (§§2a-2h). §2a is the LDA
exchange-correlation functional: `src/elkjax/xc.py` transcribes `xc_pwca.f90`, with
`jax.grad` reproducing Elk's hand-coded $v_{xc}$ at machine precision against the study's
stated $10^{-10}$, exchange exact against Dirac and its spin scaling, correlation
anchored on the Gell-Mann–Brueckner high-density limit, Elk's own $v_{xc}$ on a real grid
at 3e-5 median — limited by a nonlinear functional not commuting with either of Elk's
representations of a real-space function, not by the transcription — and the study's
named `NaN` hazard turned into an assertion: Elk's `rho < 1e-20` guard written as one
`jnp.where` gives the correct value and a `NaN` gradient at exactly zero density.
Patch **0016** adds a `GROUNDSTATE` query — the density and potentials on Elk's own
grids, $k$-independent — which makes that comparison **exact**: Elk evaluates the
functional pointwise on the FFT grid, so `vxcir` is literally the transcription applied
to `rhoir`, to 4.4e-16, once `potks`'s own `trimrfg` low-pass at $|G|>2k_{\max}$ is
reproduced with it (`elkjax.grid.trim`; without it the same comparison stops at 2.5e-5
and looks like a mediocre transcription). §2b is **PBE**, and it is the port's first
real demonstration of its own premise: only the ENERGY densities are transcribed —
exact against Elk's `exir`/`ecir` at 4e-16, which unlike `vxcir` are not trimmed — and
`jax.grad` of the discretised energy supplies the functional derivative
$-\nabla\cdot(\partial(\rho\varepsilon)/\partial\nabla\rho)$ that Elk gets from
Perdew's hand-derived expression needing $\nabla^2\rho$ and
$(\nabla\rho)\cdot(\nabla|\nabla\rho|)$ as extra inputs. **Nothing here computes a
Laplacian.** It agrees with Elk to 2.4e-5 median — not machine precision, and the reason
matters for Phase 3: **Elk discretises the exact continuum functional derivative; AD
returns the exact derivative of the discretised energy**, and those differ because
$|\nabla\rho|$ is not band-limited even when $\nabla\rho$ is. The prediction is
asserted rather than described — the residual tracks the reduced gradient $s$ (5.8e-6 in
its lowest quarter, 1.25e-5 in its highest) — and for scale the gradient terms are 11% of
$v_{xc}$, so the disagreement is ~1% of what AD reproduces from nothing. §2c is the cell
integral and inner product (`rfint`/`rfinp`): $\int\rho$ gives the electron count to
**1.1e-14** against the study's 1e-8 criterion (note `rhomt` INCLUDES the core density —
assuming valence misses by 20 electrons, not by a tolerance), and $E_x$/$E_c$ match Elk's
INFO.OUT to its print width). §2d transcribes the muffin-tin angular transform
(`rbsht`/`rfsht`) and found a real property of Elk with it: `potxc.f90:55-58` calls
`symrfmt` on `vxcmt` and `bxcmt` and **not** on `exmt`/`ecmt`, so inside a muffin tin
$v_{xc}=\hat S\,v_{xc}[\rho]$ while $\varepsilon_{xc}=\varepsilon_{xc}[\rho]$ — the
projection is not the identity even on an already-symmetric $\rho$, because $v_{xc}[\rho]$
is not band-limited when $\rho$ is and the SHT round trip leaks weight into the forbidden
harmonics (Elk holds 1e-20 at Si's $l=1,2,5$; the pointwise potential holds 1e-3). On a
`symtype=0` ground state the same code reproduces `vxcmt` to 1.4e-14. Two consequences:
reproducing Elk's SCF on a symmetric cell needs `symlatc`/`lsplsymc`/`ieqatom` exported and
`rotrfmt` transcribed (one more patch, not a research problem), or a `symtype=0` run; and
**Elk's own $v_{xc}$ is not the functional derivative of its own $E_{xc}$ there**, so a
force or total-energy check better than $\sim10^{-4}$ relative would be evidence of a
mistake rather than of success. §2e is the **Weinert Poisson solve** (patch **0017**,
`src/elkjax/poisson.py`), which closes the last ingredient no export supplies as a
function of the density: `vclir` to 1.6e-15 relative and `vclmt` to 4e-20 ($l=0$, of order
$10^7$ since it carries the nucleus) and 7e-14 ($l>0$), on bulk Si and monolayer h-BN.
Patch 0017 exports only `wprmt`, `vcln`, `npsd`/`lnpsd` and `atposc`; $r^l$, $R^l$ and
$4\pi/G^2$ come from the mesh and `gc`, and `ylmg`/`sfacg`/`jlgrmt` are `genylmv`/
`gensfacgp`/`sbessel`, already pinned element-wise by patch 0013 — exporting `ylmg` alone
would be 38 MB of text. Two checks owe Elk's `vclmt` nothing: the monopole identity
$\sqrt{4\pi}q_{00}=N_{\rm MT}-Z$ recovers $Z=14.000000,5.000000,7.000000$ with $N_{\rm MT}$
from `elkjax.integrate`, and two mutation tests remove one thing Elk does each (the nuclear
term *before* the multipoles are read; the outer region's own spline weights for
$l>l_{\max}^{\rm i}$) and assert the answer moves — both mutants smooth, of the right order
and wrong. Nothing there is differentiated, deliberately: Poisson is linear in $\rho$, so
its linearisation is itself. §2f assembles the **total energy** (`src/elkjax/energy.py`),
which is the first thing in Phase 2 to use more than one of its own pieces at once — none
of the checks above says the functional, the quadrature and the Poisson solve are
consistent WITH EACH OTHER. Every density-functional term of `energy.f90` matches Elk's
own exported scalars to $<10^{-13}$ relative on two structures, asserted term by term
rather than through the total (`engykn` is $+579$ against `engyen`'s $-1219$, so an error
of $10^{-3}$ in either would leave `engytot` looking fine at $10^{-6}$). `evalsum`,
`engyts` and `engynn` are **imported** — they need the second-variational step, a zone sum
and the lattice — so this is the density-functional half, not a transcription of
`energy.f90`. Patch 0017 exports Elk's thirteen converged scalars precisely so the
comparison is term-by-term at full precision rather than at `INFO.OUT`'s print width.
**And it found §2d's prediction to be wrong**: §2d predicted $E_{v_{xc}}$ would inherit the
symmetrisation gap at $10^{-4}$, and it agrees at $2.7\times10^{-16}$, because $\hat S$ is
a group average — an orthogonal projection — and $\rho$ is already in its range, so
$\langle\rho,\hat Sv\rangle=\langle\rho,v\rangle$ identically and the leak lives entirely
in harmonics $\rho$ does not have (measured: 5.3e-3 pointwise, 1e-16 relative against
$\rho$). What survives of §2d is that an SCF iteration compares potentials *pointwise* and
still needs $\hat S$. **A prediction derived from a verified finding is not itself
verified.** §2g closes §2d's remaining consequence with patch **0018**, and the design
choice is the content: `symrfmt`'s operator is **exported rather than transcribed**,
because `rotrflm`'s Euler-angle and Wigner-$D$ construction has no consumer inside Elk
but `symrfmt` itself — so a re-derivation would have no independent check except
agreement with what it replaces, and `ieqatom`/`tfeqat`/the inverse lattice rotation
would have to come with it. `elkpy_gsexport` calls `symrfmt` on basis vectors, giving one
$l_{\max}^{\rm o}$-square matrix per ordered atom pair (a rotation is diagonal in the
radial index and does not mix $l$, so the inner region uses its top-left block), and
applying it takes the pointwise `vxcmt` gap from 5.3e-3 to **6.4e-14**. One measurement
worth keeping: the operator is idempotent to 1e-16 on a CUBIC lattice and only 1.2e-11 on
a hexagonal one, growing with $l$ — Elk's own `roteuler`, whose inverse trigonometry is
exact when the Cartesian `symlatc` entries are $0$ and $\pm1$ and is not otherwise. That
bounds how idempotent `symrfmt` can be, not its accuracy in use. §2h turns the chain
into a circle with patch **0019**: every other Phase 2 section goes from a density to an
energy, and `elkjax.density` goes back, reproducing Elk's valence density
from `evecfv` in BOTH regions to **9e-16** (9.0e-16 and 7.7e-16 in the muffin tins,
7.5e-16 in the interstitial) on an unreduced mesh — exact, since in the interstitial
an LAPW state is a plain plane-wave sum and the only truncation is the basis's own. Two of
its three results are scope statements and both are asserted rather than written down: on a
symmetry-REDUCED mesh it is **16% off**, because `rhomagv` calls `symrf` afterwards and this
does not (§2g's story again, in the interstitial, where the operator is `symrfir`); and the
residual on the unreduced mesh is `rhonorm`'s UNIFORM shift, identified by measurement —
switching `trhonorm` off takes it from 2.82e-05 to 2.8e-18. The muffin-tin half needed only ONE routine
(`wfmtsv`) rather than five, because patch 0019 loops Elk's own `rhomagk` into a LOCAL
array and exports THAT — the density before `rhomagsh`, `symrf`, `rfmtctof` and
`rhocore`, none of which is therefore transcribed (patch 0018's design again). Two traps,
one hit: `evecfv` carries $n_{\rm mat}=n_{gk}+n_{\rm lotot}$ coefficients and exporting
only $n_{gk}$ left the interstitial EXACT (local orbitals vanish there) while the muffin
tin came out smooth, positive, correctly scaled and 100% wrong; and `wfmtsv`'s outer
region restarts its radial stride one step PAST the inner boundary rather than continuing
it. Patch 0015's finding also recurred with a sharper consequence — the reference was
built from the previous iteration's radial functions, so the query's answer depended on
whether `LAPW` had been asked for first (1.2e-10 against 9e-16); fixed with `genapwlofr`
in the Fortran rather than documented around, since **a query whose answer depends on
which query ran before it is a trap**. Patch **0020** then closes the chain: `rhomagsh` (back to
harmonics) at 8.7e-16 and `rfmtctof` (coarse radial mesh to fine) at 1.0e-15, each
against its own exported intermediate rather than through their composition. `rfmtctof`
is exported as a MATRIX for the same reason `symrfmt` is — `rfinterp`'s spline weights
come from `wspline` and depend only on the mesh — with TWO per species, since it
interpolates the whole radial range below $l_{\max}^{\rm i}$ and the outer region alone
above it; using the wrong one reads the inner region's zeros as data, which is smooth,
finite and wrong. **With `symtype=0` those three stages are the whole of `rhomagv`**, so
the chain from first-variational eigenvectors to the valence density on the fine mesh is
closed and exact. Still not done: `rhocore` and the core states (an input at fixed
potential, so an export would do), `symrfir` for a symmetry-reduced mesh, `rhonorm`'s one
constant, and the magnetic branches. §§2i-2j then put the pieces together: the Kohn-Sham
potential COMPOSES pointwise ($v_{\rm cl}+\hat Sv_{xc}$ from three separate modules,
<1e-14 in the muffin tin and <1e-13 against Elk's own `vsir`, which `potks` forms after
trimming `vxcir` and not `vclir` — a mutation test pins that trimming the Coulomb term
too is smooth, correctly-integrating and wrong by only $10^{-12}$); and
`density_from_potential` closes the loop, going potential → $H,O$ at every $k$ →
eigensolve → density with **nothing in the path reading an eigenvector**, at 5e-11. The
missing link was building the interstitial blocks from `vsig`/`cfunig` in $G$-space
instead of recovering $\tilde v_s$ as a matrix in one $k$-point's own basis; against
Elk's own matrices at $\Gamma$, $H$ agrees to 2.7e-15 and $O$ to 5.6e-16. **The 5e-11 is
measured, not excused**: it is Elk's own two exports of `evecfv` disagreeing by 8.5e-9 —
`elkpy_lapwexport` diagonalises fresh after `genapwlofr` while `elkpy_denskexport` reads
the stored ones — and on the gauge-invariant occupied projector this solve matches the
fresh `evecfv` at 1.6e-14 while stored-vs-fresh is 3.7e-11. Patch 0015's finding for the
third time. What is NOT yet done is iterating: that needs `rhocore`, `rhonorm`, a
zone-summed Fermi level and `symrfir`, none a research problem, and the open question is
whether the iteration is stable. **§2k finally differentiates Phase 2** — every section
before it checks a *value* — and found a defect doing so:
`elkjax.integrate.cell_inner_product` called `np.asarray` on its muffin-tin argument and
could not be traced at all, silently, since §2c. Fixed. Then
$\delta E_{xc}/\delta\rho=v_{xc}[\rho]$ holds to **5.7e-17** with AD running through both
`xc_pwca` and the quadrature, and the 1.25e-5 gap to Elk's `vxcir` is entirely `trimrfg`
— asserted as an EQUALITY with `grid.trim`, not an order-of-magnitude coincidence. **The
electrostatic half does not close**: 0.10 Ha absolute, which is 33% of $v_{\rm cl}$'s own
range and 4% of the $v_H$ (2.17) and $v_{\rm nuc}$ (2.48) that nearly cancel to make it
(0.31) — quoting either number alone misleads. Pinned two-sidedly so a later fix fails
the test instead of passing it quietly; the candidate to test first is whether the
discretised Coulomb kernel's reciprocity is only as accurate as the pseudocharge
construction, making it a property of the Weinert method rather than a bug. Verdict, in one line: **a research project justified by
differentiability, not by the GPU** — SIRIUS already does FP-LAPW on CUDA/ROCm with Elk as its
reference, and Elk's hot spots are already near-peak BLAS-3. Nothing about the port is a plan of
record; **Phase 0 (§6 of the study) is designed to kill it, not to start it**, and that is what
is being worked on.

Code lives in `src/elkjax/` — a **sibling package** to `elkpy`, deliberately not `elkpy.jax`.
Two reasons, both load-bearing: Phase 0 is explicitly "no Elk code", and elkpy's fast unit tests
must not acquire a `jax` dependency at all (measured: 0.2 s of import time, plus a 40-thread XLA
pool on the first array operation — see below). Install with
`python3 -m pip install -e .[jax]`. Tests are `tests/test_jax_*.py` and self-skip when `jax` is
unimportable, the same pattern `tests/test_structure.py` uses for ASE. They add ~22 s to the
default suite (each Phase 0a test converges a real fixed point), so
`-k "not calculation_ and not jax"` still gets elkpy's own 440 in ~1 s; the heavier sweeps are
behind `ELKPY_RUN_SLOW_TESTS=1` and take ~3 min. The one test needing BOTH jax and the Elk
binary is `tests/test_calculation_lapw_assembly.py` — named `calculation_`, not `jax_`, so
the fast-suite filter above already excludes it; it converges three ground states, ~30 s.

### Memory and CPU discipline — read before running any JAX in this repository

This box: 12 cores, 39 GB RAM (~28 GB available, ~4 GB of swap already in use), **no CUDA
jaxlib** (`jax.devices()` is `[CpuDevice(id=0)]`). The rules below exist because the study's own
Phase 0e asks for production shapes that this machine cannot hold.

- **Never allocate production shapes here.** At the study's own production figures
  ($n_{\rm mat}\approx3000$, $n_{\bf k}\approx100$, complex128) a single k-point's $H$ or $S$ is
  $3000^2\times16$ B $=144$ MB, so $H+S$ over the k-set is **26.8 GiB before the
  eigenvectors** (another 13.4 GiB) and before any eigensolver workspace, against ~28 GiB
  available — it does not fit, and swapping a 39 GB box is how a workstation is lost for an hour. Execute at $n\le1500$, $n_{\bf k}\le4$ and measure the scaling exponent instead.
- **For Phase 0e, do not execute at all — lower and compile.**
  `jax.jit(step).lower(*jax.ShapeDtypeStruct(...)).compile()` gives both numbers the item asks
  for without allocating a byte of the shapes: wall time for compile, and
  `.memory_analysis()` (verified present in JAX 0.7.1) for
  `temp_size_in_bytes`/`argument_size_in_bytes`/`output_size_in_bytes`. Extrapolation from an
  executed small case is a fallback, not the method.
- **Cap every JAX script and test** with `elkjax.memory.limit_address_space()` (a
  `resource.setrlimit(RLIMIT_AS, ...)` wrapper, default 16 GB) so a runaway allocation raises
  `MemoryError`/`XlaRuntimeError` immediately instead of driving the machine into swap. Verified
  to leave a CPU `eigh` at $n=800$ untouched while turning a 25 TB allocation into a prompt
  error. `tests/test_jax_projector.py` calls it at module level; do the same in anything new.
- **`lax.map`/`lax.scan` over the k-axis is the memory default; `vmap(eigh)` is opt-in.** `vmap`
  materialises every k-point's matrix simultaneously — exactly the 26.8 GiB above. Phase 0d is the
  design fork that would justify `vmap`, and it **requires a GPU this machine does not have**;
  it is deferred, not answered. A CPU ratio (1.03x, measured in the study) does not settle it.
- **`.claude/settings.json`'s `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` pins do NOT govern XLA.**
  Measured here: a single 1200x1200 `jnp` matmul under `OMP_NUM_THREADS=1` spawns **40 threads**;
  `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` changes nothing (still 40). What works is
  affinity: `taskset -c 0-3 python3 ...` gives 16 threads confined to 4 cores. **Wrap every JAX
  invocation in `taskset` while the four-core budget stands** — `launcher.py`'s flock semaphore
  only covers `elk` subprocesses and sees none of this.
- **`jax_enable_x64` must be set before the first array exists**, and doubles every figure above.
  `import elkjax` does it (`src/elkjax/__init__.py` calls `jax.config.update`), so import that
  first and never set it mid-module; `JAX_ENABLE_X64=1` in the environment works too. All of
  this work is float64/complex128: an all-electron spectrum spans ~2500 Ha, so float32 is not an
  option (study §1).

### Phase 0 — prove or kill (study §6)

| Item | What it settles | Status |
|---|---|---|
| 0b | safe-$K$ projector rule: does the $(f_i-f_j)/(\lambda_i-\lambda_j)$ `custom_jvp` fix the reassembly jitter, and what happens in the padding block | **done, and now closed against REAL matrices** by Phase 1f — the rule is exercised on Elk's own Cholesky-reduced $\tilde H$ at a symmetry multiplet, with $\kappa(O)\approx5\times10^3$ measured per run rather than prescribed |
| 0a | reverse-mode implicit diff (`custom_vjp` + GMRES) through an SCF fixed point whose matvec passes through `eigh` at a multiplet | **done — it works**, ≤1e-14 against a dense IFT reference on four spectra including an exactly degenerate one; the naive rule fails on the same machinery |
| 0a′ | the same at second order — decides the full port over §9.2's hybrid | **done — it works**, via `projector.sign_projector` (matrix sign by Newton-Schulz: no eigensolve, so differentiable to any order); `grad(grad)` through the fixed point agrees with central FD to 1.2e-9 where the `eigh`-based rule gives `NaN`. Use `grad(grad)`, never `jax.hessian` — a `custom_vjp` cannot be forward-differentiated |
| 0c | `jax.jvp(match)` against `dmatch.f90`'s analytic $d(\texttt{apwalm})/dr$ | **done — exact to 7e-16** in both modes (`src/elkjax/lapw.py` transcribes `match`, `gengkvec`, `gensfacgp`, `genylmv`, `sbessel`), **and the forward half is now closed against Elk's own `apwalm`** by patch 0013 (§33): 1.9e-15 in `match`'s `omax==1` division branch and 8.0e-13 in its general linear-solve branch, the latter reachable only through a generated `apword=2` species file since every species file Elk ships sets `apword=1` |
| 0d | `vmap(eigh)` vs `lax.map` at $n=1000$ on a real GPU | **timing deferred (no GPU); memory settled by 0e** — at production shapes a `lax.scan` accumulator holds 0.411 GiB of temporaries and `vmap` holds 40.2 GiB, so `vmap` over the k-axis does not fit on a 40 GB device whatever the timing says |
| 0e | `jit` compile time and peak memory for one traced SCF step at production shapes | **done — `docs/jax_port_phase0.md`.** Compile time is FLAT in the shapes (0.46 s at both $(200,4)$ and $(3000,100)$) and **superlinear (exponent ≈1.85) in HLO op count** — isolated with the corrector, which is linear in its pass count, since Gram-Schmidt's own op count is quadratic in `n_lo` — while a `lax.scan` over 4x more radial points costs nothing. Design rule: `scan` repeated structure, unroll only what must be. Differentiating the step adds only ~1.2x |

### Phase 1 — one k-point, one species, no SCF (study §6)

**`hmlfv`/`olpfv` are done, forward.** `src/elkjax/hamiltonian.py` transcribes the
muffin-tin half of the first-variational LAPW eigenproblem —
$O^{\rm MT}=A^\dagger A$ and $H^{\rm MT}=A^\dagger ZA$ with
$Z=\sum_{\ell_2m_2}\langle Y_{\ell_1m_1}|R_{\ell_2m_2}|Y_{\ell_3m_3}\rangle\,
h_{\ell_2m_2}$ — and each of its **six blocks is compared separately** against Elk's
own (`tests/test_calculation_lapw_assembly.py`), so a failure names one upstream
routine rather than "$H$ is wrong". Machine precision on bulk Si at `apword` 1 and 2 and
on monolayer h-BN; the assembled pair reproduces Elk's `evalfv` to 9e-15 Ha. Patch
**0014** supplies what 0013 did not: the radial integrals `oalo`/`ololo`/`haa`/`hloa`/
`hlolo`, the local-orbital bookkeeping and the complex Gaunt array `gntyry`. The
**interstitial blocks are taken from the export, not built** — $H^{\rm I}$ needs the
interstitial Kohn-Sham potential $V_s$, which is Phase 2 — so `cfunig`/`vsig` are
deliberately not exported.

**The trap, and only one of the three fixtures can see it.** `hmlrad.f90` builds
`hlolo`'s $\ell_2=0$ element as the unsymmetrised
$\int u^{\rm lo}_i(\hat H u^{\rm lo}_j)r^2dr$ — no averaging over the two orderings and
no kinetic surface term, unlike `haa`, whose counterpart is explicitly averaged and whose
transpose is explicitly assigned. `hmllolo` therefore evaluates each local-orbital pair
in ONE order and Hermitises the rest, and a consumer must do the same. Using both halves
gives a Hermitian, positive-definite, plausible, **wrong** matrix: measured on h-BN's
nitrogen (two $\ell=0$ local orbitals), the orderings differ by 1.3e-2 Ha, reaching $H$
as 3.7e-3 Ha and `evalfv` as 4.3e-7 Ha. Silicon cannot see it — one s and one p, no
repeated $\ell$ — so the h-BN premise is **asserted** by its own test rather than
assumed. `apword=2` is equally load-bearing: at `apword=1` the APW-order axes of `haa`
and `hloa` are length 1, so an $i_o\leftrightarrow\ell$ swap is a no-op rather than a
detected error. Mutation-tested one error at a time.

**The residual is Elk's own guard, measured not assumed.** `hmlaa`/`hmlalo` skip any
Gaunt-contracted $z_1$ below 1e-12 (the `zaxpy` guard); reproducing it takes h-BN's
`hmlalo` from 3.3e-14 to 3.0e-17. The dense version here is the more accurate of the
two, so the guard is documented rather than copied — and any element-wise comparison
against Elk has a ~1e-12 floor because of it.

**The assembly is now a function of $k$, and it has been differentiated.** Two
observations remove the need for any new Fortran: $O$'s interstitial block IS the
characteristic function and is $k$-independent, and $H$'s is that plus an explicit
kinetic term — so $V_s$, the one Phase 2 ingredient involved, is recovered as a MATRIX
by subtracting the two exported blocks elementwise, with no Fourier index mapping.
`apwalm` comes from `elkjax.lapw.match`, closed with a Cholesky-reduced `eigvalsh`.
Back at the exported $k$ it reproduces Elk's $H$, $O$ and `evalfv` (9.8e-15 / 2.7e-15 /
5.0e-15 on Si), and `jax.grad` agrees with central FD of the same function to 1e-9.

**And it found that $d\varepsilon/dk \neq \langle p\rangle$ in a finite LAPW basis.**
Against `genpmatk` (§22's `MOMENTUM`, an independent Fortran path) the two agree in sign
and to 0.2-1.4%, NOT to machine precision — and the gap is **flat in `rgkmax`** (2.336e-3,
2.339e-3, 2.340e-3 at 7, 8, 9 for band 0) while the eigenvalue converges, so it is not
the plane-wave cutoff. It is the **muffin-tin linearisation**: `apword` 1→2 (augmenting
with $\dot u$ as well as $u$) cuts it up to 4x, most for the high band where it was
largest. Hellmann-Feynman needs a $k$-INDEPENDENT basis and LAPW's is not one, so AD
returns $v^\dagger(\partial_kH-\varepsilon\,\partial_kO)v$ while `genpmatk` returns
$\langle\psi|-i\nabla|\psi\rangle$. **Two consequences**: `genpmatk` is not a
machine-precision reference for a band velocity — which is the mechanism behind
`tests/test_calculation_momentum.py`'s own Hellmann-Feynman check needing `rel=2e-2` —
and any remaining gradient criterion must finite-difference the *same* code path. The
test asserts the direction of the effect rather than a tolerance, and requires the two
NOT to agree exactly so its premise cannot lapse silently. Separately: `apword=2`'s
matrices inherit `match`'s ill-conditioned general branch (1.3e-12 vs 9.8e-15) but the
SPECTRUM does not (3.8e-15 either way) — a near-null-space rotation inside the APW order
space, which eigenvalues are blind to.

**The eigensolve is now wired to the safe-$K$ projector rule, and 0b is closed against
real matrices** (§1f, `src/elkjax/phase1_projector.py`,
`tests/test_calculation_lapw_projector.py`). `hamiltonian.py` gained `cholesky_reduce`,
`projector_tolerance` ($\epsilon\,\kappa(O)\,\lVert\tilde H\rVert$, measured per run
from a dense `eigvalsh` — §8b's Cholesky estimate is uninformative), `occupied_window`
(the host-side refusal) and `occupied_projector`. Bulk Si at $\Gamma$ supplies the
disputed configuration by SYMMETRY rather than by construction: the $\Gamma_{25'}$
triplet inside a window whose boundary is open by 0.093 Ha. Measured there, the naive
route is wrong by 3.9e-1 in forward mode and returns `NaN` in reverse, against 2.2e-11
for the safe one; at a generic $k$ on the same ground state both agree to 2e-13, which
is what makes it a measurement of the rule rather than of the fixture. The naive error
tracks $1/\delta\lambda$ across three fixtures spanning ten decades of splitting
(3.8e-13 at 2.6e-2 Ha, 1.1e-10 at 1.9e-5 Ha, total failure at 5.1e-15 Ha). The
projector reproduces Elk's own occupied subspace — as $YY^\dagger$ with
$Y=L^\dagger C$, a projector comparison because `evecfv` is arbitrary inside the
triplet — to 5e-14, six orders inside the study's own $10^{-8}$ criterion.

**Three findings from doing it.** (i) The tolerance is a *resolution* floor, not a
symmetry statement: Elk's matrices split the $\Gamma_{25'}$ triplet **unevenly**,
5.1e-15 Ha for one pair and 3.53e-9 Ha for the other, the second 68x ABOVE the
5.21e-11 Ha tolerance, so cutting the triplet at `nocc=3` is refused while cutting the
SAME triplet at `nocc=2` is accepted and returns a finite derivative of a subspace that
is not physically separable. The refusal is necessary, not sufficient — window the whole
degenerate group, as §13 already does for Berry curvature. Tightening `epspot` 1e-6 →
1e-9 leaves that 3.53e-9 identical to twelve digits, so it is not SCF convergence;
source open. (ii) **`soc_scale` cannot move the first-variational spectrum at all** —
`socfr` enters only `eveqnsv`, with zero occurrences in `hmlfv`/`olpfv`/`hmlaa`/
`hmlalo`/`hmllolo`/`olpaa`/`olpalo`/`olplolo`/`eveqnfv`/`hmlrad`/`olprad`
(grep-verified) — so the study's adversarial sweep is "refuse always", not a threshold
crossing, and is withdrawn as written; its content is delivered by cutting a real
multiplet instead. (iii) **The $k$-tangent is `NaN` at $\Gamma$** while the value there
is exact — see the next paragraph.

**That third finding was a real blocker, and it is fixed (§1g).** At any basis function
with $\mathbf G+\mathbf k$ on the $z$-axis, $Y_{\ell m}(\hat v)$ has no derivative (the
direction is undefined) and $\lvert\mathbf G+\mathbf k\rvert$ is $\sqrt\cdot$ at zero —
**two independent poles, and fixing only the first leaves the second, which is invisible
until it is**. $\mathbf G=0$ is in every basis, so this was every reciprocal-lattice
point, and in a slab cell $\mathbf G=(0,0,\pm2\pi/c)$ is too, so it was the entire
$k_z=0$ plane — all of a 2D material's physics, the $K$ point included — which combined
with (i) left the safe-$K$ rule and the $k$-derivative usable in DISJOINT places, since
multiplets live at high-symmetry points. The product is smooth even though its factors
are not, so `match` now regroups it as a **regular solid harmonic** $r^\ell Y_{\ell m}$
(`elkjax.lapw.solid_harmonics` — `spherical_harmonics`' own recursion with
$\cos\theta\to z$, $\sin\theta e^{i\phi}\to x+iy$, $\beta\to\beta r^2$, hence a
polynomial in the Cartesian components) times $j_\ell^{(i_o)}(x)x^{i_o-\ell}$
(`spherical_bessel_scaled` — even in $x$, hence a function of
$x^2=R^2(\mathbf G+\mathbf k)\cdot(\mathbf G+\mathbf k)$ with its own small-$x$ series),
and forms neither $\hat g$ nor $\lvert g\rvert$; `gkc` is therefore **no longer an
argument of `match`**, since leaving it in the signature would let a call site
reintroduce the second pole. `spherical_harmonics`/`spherical_bessel` are untouched and
remain what item 0c checks. The identity is exact, so forward values are unchanged:
Elk's `apwalm` element-wise at generic $k$ **and now at $\Gamma$** (a real check, since
Elk handles $\mathbf G+\mathbf k=0$ its own way), the `dmatch` identity, and all six
assembly blocks are still green; $dP/dk$ at $\Gamma$ through the multiplet goes from
`NaN` to 1.4e-14. The new route is also *more* accurate near the origin than the old
one, which loses 100% at $\ell=6,i_o=2,x=10^{-6}$ by forming $j_6''\sim10^{-27}$ and
dividing by $x^4$. **The two fixes are independent**: with the poles gone, the *naive*
projector's $k$-derivative at $\Gamma$ is still `NaN`, which its own test asserts so
that "we fixed the pole" cannot be mistaken for "the $k$-derivative is fine".

**The study's negative test is done too (§1h), and it corrected the study's own fixture
suggestion.** The criterion is that AD, central FD and one-sided FD of an INDIVIDUAL
eigenvalue must *disagree* at an exact degeneracy while the multiplet trace agrees. The
study names "h-BN at $\Gamma$", and that cannot work — nor can Si at $\Gamma$: at any
time-reversal-invariant momentum every branch is EVEN in $\mathbf k$, so the sorted
branches never exchange between $+t$ and $-t$ and AD and central FD both correctly return
zero (measured $10^{-17}$–$10^{-11}$ on Si's $\Gamma_{25'}$ triplet). Degeneracy is not
enough; the branches must cross LINEARLY. On graphene at $K$ (2 atoms, `rgkmax=6`, under
two minutes including the ground state) the Dirac pair gives AD $\pm0.184$ (a
basis-dependent number that means nothing on its own — it is the diagonal of
$v^\dagger\delta Hv$ in whichever basis the eigensolver picked, so only its *disagreement*
with the other two is the result), central FD $\pm0.0009$ (the branch average, since the
branches exchange) and one-sided $\mp0.376$ (the extreme branch), while their trace agrees
across all three to
$5\times10^{-7}$ of that scale — with the $\sigma$ doublet at the same $K$, equally
degenerate but not linearly split, as the in-fixture control where AD and central FD do
agree. So `first_variational_eigenvalues` is safe for a trace and unsafe for an
individual band inside a multiplet, asserted rather than documented.

**Smeared occupations are done too (§1i), and they corrected their own premise.** A hard
integer window cannot exercise the kernel's near-degenerate branch at all — both branches
of a same-side pair are identically zero — so §1f's `tol` plateau said nothing about it;
Fermi-Dirac occupations make the branch value $f'$, and the threshold becomes
load-bearing. What was *expected* was that a metal is where it bites. What was measured is
that **whether the branch fires is set by the assembly's roundoff, not by the physics**:
graphene at $K$ — the metal, with the Fermi level exactly on the Dirac degeneracy
(confirmed by the single-$k$ $\mu$ reproducing Elk's own zone-integrated `EFERMI.OUT` to
$3.4\times10^{-9}$ Ha) — is split $3.4\times10^{-7}$ Ha and does **not** fire, while gapped
Si at $\Gamma$ is split $1.1\times10^{-15}$ Ha inside $\Gamma_{25'}$ and does. Both were
needed, because each shows only one of two distinct failures: on graphene the direct
quotient is exact to $10^{-13}$ and it is JAX's own eigenvector rule that fails,
**proportionally in the smearing width** (3.9e-9, 3.9e-8, 2.8e-7 across $w=10^{-3}$ to
$10^{-1}$, since its absolute error is set by the splitting while the quantity it is
measured against is $f'=1/4w$ — broader smearing is not gentler); on Si that rule returns
`NaN` outright while the quotient survives but is wrong by up to 3.7e-3, and at Elk's
default `swidth` is **exactly zero** against a true kernel of $-3.2\times10^{-2}$, the two
occupations being bitwise equal. The oracle for all of this is analytic, not either
candidate: `reference.fermi_divided_difference_kernel` writes the logistic difference
quotient as $-\frac1{4w}\,\mathrm{sinhc}(z_{ij})/(\cosh u_i\cosh u_j)$, which contains no
subtraction and so arbitrates between the branches. The **self-consistent Fermi level**
came with it: `mu` is now a differentiable primal of `smeared_projector`, so
`fixed_number_projector` is just the composition with `projector.fermi_level` (a
`custom_jvp` over a never-differentiated bisection), and study §8(b)'s
$d\mu=\sum_jf'_jA_{jj}/\sum_jf'_j$ is tested for the first time — gauge-invariant at a
multiplet (inside a degenerate group $f'$ is constant, so the sum is a trace), and
**dominant rather than corrective**: dropping it is wrong by 69x at a half-filled level.
Its denominator vanishing is physics, not numerics — in a gap nothing responds and $\mu$ is
undetermined, which `check_fermi_level_determined` refuses. Two limits left: §8(b)'s
k-point weights cancel at a single $k$ and remain untested (a zone-summed Fermi level is
Phase 2), and the safe rule's floor on Si is $2\times10^{-9}$ rather than $10^{-13}$
because the *other* pair sits $67\times$ ABOVE the tolerance and therefore takes the
cancellation-prone quotient — the tolerance is a cliff, and beating that needs the stable
kernel inside the JVP rather than a better threshold.

**The tolerance was then removed rather than tuned, and second derivatives followed
(§1i second pass, §1j).** The stable form of the kernel is *analytically exact at every
splitting*, so it belongs inside the JVP rather than beside it: `projector.fermi_kernel`
now carries it and `smeared_projector` uses it, which takes Si's floor from
$2\times10^{-9}$ to $7.4\times10^{-12}$/$3.1\times10^{-12}$/$4.8\times10^{-14}$ across the
three widths and makes `tol` **inert** for smeared occupations — asserted as *equality*
across fourteen decades of it, which is a stronger and different statement from study
§8(b)'s "flat over two decades". (`tol` stays load-bearing for `hard_window_projector`,
whose straddling-pair `NaN` is a claim about a derivative that does not exist.) What is
left is **not the kernel**: the residual shrinks as the smearing *widens*, which is
backwards, and chasing that showed LAPACK and XLA disagreeing about the eigenvalues by
$2.1\times10^{-14}$ Ha, amplified by the kernel's $f''/f'\sim1/w$ relative sensitivity —
rerunning the identical closed form on XLA's own decomposition gives 7.9e-14 / 5.5e-15 /
2.6e-15, and the difference between the two references *is* the residual to two digits
(`phase1_smearing.eigensolver_floor`). Knock-on worth remembering: the reference and the
implementation now share a *formula*, so the arbiter where it matters is central FD
(legitimate here, since $P=f(H)$ is smooth) and `direct_quotient_projector`, which keeps
the literal quotient. **Second order (§1j)**: the safe-$K$ rule is first-order by
construction — its JVP body calls `eigh` — and on bulk Si at $\Gamma$ with the
$\Gamma_{25'}$ triplet enclosed, `grad(grad)` through it returns `NaN`, as does the naive
route, while `sign_projector` (matrix sign by Newton-Schulz, no eigensolve) returns a
finite value agreeing with a central difference of the safe rule's own first derivative to
$3\times10^{-11}$–$2\times10^{-10}$. The control is in the same ground state: at a generic
$k$ all three agree to $10^{-10}$, so the failure is the multiplet, not the order.
Differentiated twice in $k$ through the whole assembly too, where refining the FD step
gives 1.44e-4, 1.29e-5, 1.44e-6, 1.29e-7 — textbook $O(h^2)$, i.e. FD converging *onto*
AD. Cost, which is the half a toy could not supply: the Newton-Schulz count is set by the
top of the basis (17.9 Ha) and not by the 0.35 Ha valence manifold, so the ratio is 190
and the predicted count 13 — 10 steps is not converged (error 1.1, not a projector at
all), 20 reaches $3\times10^{-14}$ against both the safe rule and Elk's own occupied
subspace, in 42 ms at $n=177$. Use `grad(grad)`, never `jax.hessian`.

**The radial integrals are no longer inputs (§1k), and the chain closes without any
Phase 2 ingredient.** The plan put this in Phase 2 because it needs the muffin-tin
potential — it does, but a converged potential is an *input* that can be exported and
held fixed exactly as `STATE.OUT` already is. Patch **0015** exports `vsmt` in Elk's own
packing, the radial mesh `rlmt` with its `wr2mt` quadrature weights, the linearisation
energies, and `apwfr`/`apwdfr`/`lofr` in full; `src/elkjax/radial.py` transcribes
`hmlrad`/`olprad` and `src/elkjax/radial_functions.py` transcribes
`rschrodint`/`genapwfr`/`genlofr`, so **vsmt → apwfr/lofr → radial integrals → H, O →
evalfv** is closed and differentiable. Element-wise against Elk on the three fixtures:
`oalo`/`ololo` 2.4e-16, `haa`/`hloa`/`hlolo` 2.0e-16, `apwfr`/`apwdfr` 1.3e-14, `lofr`
7e-15. Elk's predictor-corrector is transcribed rather than replaced — a Runge-Kutta step
would converge to the same continuum solution and disagree at the mesh's own truncation
error, four decades above what is checked; its first three points are unrolled (their
stencil windows overlap the $r\to0$ boundary values, and the current iterate sits INSIDE
the four-point window it is integrated over) and `lax.scan` starts at the fourth.

**The finding, and it is structural: the two channels of the potential are exactly
complementary.** Frozen-basis vs full AD on bulk Si over the occupied window — a purely
SPHERICAL perturbation gives a frozen-basis derivative of **exactly zero**; a purely
NON-SPHERICAL one moves the basis not at all (7.0e-16). `hmlrad`'s $\ell_2=0$ element is
$\langle u|\hat Hu\rangle$ and `genapwfr` has already applied $\hat H$ — the radial
functions ARE that operator's solutions, so the radial equation has **eliminated** the
explicit $\int u\,v_{\rm sph}\,u$ integral, which is the LAPW construction itself; and
`genapwfr`/`genlofr` integrate in the spherical part alone, so the non-spherical potential
cannot move the basis. **So "full minus frozen" is NOT the basis relaxation** — it is the
Hellmann-Feynman term Elk's bookkeeping hides *plus* the relaxation, and conflating them
was a real error corrected here. `phase1_potential.hellmann_feynman` computes
$\sum_n\langle\psi_n|\delta V|\psi_n\rangle$ explicitly (the same integrals with the
$\ell_2=0$ slice filled by the potential integral rather than zeroed), and the honest
decomposition is HF + relaxation. **The relaxation depends on the SHAPE of the
perturbation by a factor of 100**: 29% of the derivative for white noise reaching the
nuclear cusp, where a basis at fixed linearisation energy cannot follow it, and **0.30%**
for a smooth valence-region bump — roughly what an SCF update does. Quoting the first
alone misrepresents the method. The Phase 2 warning that survives is narrower and
sharper: a chain producing a perfectly correct $\delta v_s$ and feeding it to a basis
frozen in Elk's own $\ell_2=0$ sense returns **zero** for the spherical channel while
passing the study's pointwise $v_{xc}$ check. **The relaxation itself has two routes**, and
the first version of this measurement missed one: a perturbed potential reaches $H$ and
$O$ both inside the radial integrals and through the matrix $D$ of radial derivatives at
$R_{\rm MT}$ that `match` inverts, so rebuilding `apwfr` without rebuilding $D$ (and hence
`apwalm`) freezes the basis at the sphere boundary while its interior moves. Nothing in
the gradient checks could see it — AD and FD then differentiate the same truncated
function and agree to 4e-10, and both structural zeros survive — which is Phase 0's own
"a green gradient test does not validate a transcription" recurring verbatim; the check
that exposes it is forward, `elkjax.radial_functions.derivative_matrices` against the
exported `dmat`. It was worth 21% of the full derivative. The frozen branch is pinned by
a closed form, not
by finite differences (at fixed basis the overlap does not respond, so first-order
perturbation theory collapses to $\sum_n c_n^\dagger\delta Hc_n$ with Elk's own
`evecfv`): 1.2e-15. Getting that reference right needs one non-obvious fact — **the map
from the potential to the radial integrals is AFFINE, not linear**, its constant part
being that same $\ell_2=0$ block, and carrying it into $\delta H$ flips the sign
(-2.27e-1 against a true +2.63e-2) rather than merely degrading it. AD vs central FD of
the same function: 3.9e-10 at $h=10^{-4}$, degrading as $h$ shrinks — the $1/h$ roundoff
signature, not a wrong gradient.

**Patch 0015 also had to fix an inconsistency in the export itself.** `gndstate` calls
`genapwlofr` at the top of an SCF iteration and `potks`/`mixerifc` at the bottom, so on
exit the radial functions and integrals belong to the PREVIOUS iteration's potential;
0015 calls `genapwlofr` before exporting. Found by splitting the comparison into
potential-free and potential-carrying integrals (2.6e-16 vs 3e-10) — an aggregate number
would have read as an indexing bug. Knock-on: that regeneration moves Elk's own matrices
by ~3e-10 and retuned one over-fitted constant in the smearing suite by 13x, while the
Dirac splitting moved only in its eighth digit; measured with the call off and on rather
than inferred, and the test now asserts the mechanism rather than a fixed factor.

**The position derivative is half done (§1l)**, and the half that exists is pinned by an
exact identity rather than a finite difference. At frozen potential (rigid muffin tin)
positions enter only through `match`'s structure factor, and **a rigid translation of
every atom cannot move the spectrum** — the matrix transforms by the diagonal unitary
$U=\mathrm{diag}(e^{i(\mathbf G_i+\mathbf k)\cdot\boldsymbol\delta})$, the muffin-tin
blocks getting that right on their own. $\tilde\Theta$ is BUILT too — closed-form
geometry (`hamiltonian.characteristic_function_matrix`, `gencfun`+`genffacgp`), matching
Elk's own $O^{\rm I}$ element-wise to 1e-16 and closing the overlap half of item 1c — so
the only response supplied by hand is the interstitial Kohn-Sham potential's,
$\tilde v(\mathbf G)\to\tilde v(\mathbf G)e^{-i\mathbf G\cdot\boldsymbol\delta}$.
Measured 1.8e-15 Ha (Si) and 3.8e-15 (h-BN) with it, 2.6e-5 and 1.4e-3 without. Building
$\tilde\Theta$ does NOT monotonically shrink the residual (on h-BN it grows, the two
omissions having partly cancelled), so a smaller residual is not evidence of a better
assembly; it does move the single-atom derivative by 7.5%/35%. **The FORWARD form of that null is sharper than the
gradient form**: without the interstitial response the error is $O(\delta^2)$ on Si and
$O(\delta)$ on h-BN, so the wrong assembly satisfies the *gradient* null identically on
silicon. Third distinct instance in this port of "a green gradient test does not validate
a transcription" (after 0c's $4\pi(-i)^\ell$ and §1k's frozen `apwalm`) — every time the
check with teeth was forward. **It is NOT a force**: the muffin-tin and interstitial
Kohn-Sham potentials' own response to the displacement is Phase 2, and is now the only
thing missing. Also open: smeared occupations **at second order**, which `sign_projector` does not cover
(it is hard-window only; that needs a Chebyshev expansion of the Fermi function, and §1i
removed the tolerance from the smeared first derivative, not the `eigh` from its JVP); and
the unrolled Newton-Schulz tape, which is **now fixed** (§1m): `lax.scan` is the default,
worth 230x the HLO instructions and 142x the compile time at 80 steps and second order
(0.21 s against 30.0 s), with a count flat in the tape. Two qualifications — the
differentiation order multiplies the ratio rather than adding to it, so the order this
routine exists for is the one that pays most; and `scan` saves the GRAPH, not the memory
(10%, since its backward pass stores one residual per iteration exactly as the unrolled
tape does). It is also not bitwise (6e-16): calling the same function the same number of
times does not fix the arithmetic, XLA fusing the two shapes into different regions.

**$\kappa(O)$ for a real LAPW overlap is measured, and the cheap estimate is
useless.** Patch 0013 (§33) supplies real $H$ and $O$; `python3 -m elkjax.phase0b_overlap`
reproduces the table in `docs/jax_port_phase0.md` §0b(ii). At the standard `rgkmax=7`,
$\kappa(O)\approx5\times10^3$, so the safe-$K$ tolerance
$\epsilon\,\kappa\,\lVert\tilde H\rVert$ is $\approx1.6\times10^{-11}$ Ha on Si and
$6.5\times10^{-11}$ Ha on h-BN. Three things to carry forward. It is set by the **cutoff,
not the matrix size** — at `rgkmax=8`, Si's $227\times227$ overlap and h-BN's
$2118\times2118$ agree to within the spread across $k$-points, while `rgkmax`
$7\to8\to9$ takes Si from $5\times10^3$ to $2.6\times10^4$ to $2.1\times10^5$ (h-BN
$7\to8$: $5.0\times10^3\to3.5\times10^4$), so the tolerance must be recomputed per run
rather than hard-coded. The study's §8b Cholesky-diagonal estimate is worse
than "a lower bound, 140x low": it is **uninformative**, moving only 8.05→9.25 across a
74-fold range of $\kappa$, so it must not be used to set a threshold. And the norm in
that formula should be $\lVert L^{-1}HL^{-\dagger}\rVert$, not $\lVert H\rVert$ — 3x
larger here. Knock-on: the study's Phase 1 adversarial `soc_scale` sweep 3000→3 stops
three to six orders of magnitude above the gap at which its own refusal criterion is
meant to fire (the spread is the unknown curvature of the gap in the scale), so it has to
be extended below `soc_scale=1`.

**Two mixers, one caveat about Elk's own.** Unrolling the SCF instead of differentiating
it implicitly is not merely inaccurate: measured, unrolled *Anderson* reaches a forward
value good to 1.8e-13 while its gradient is wrong by $10^{17}$–$10^{32}$ relative, across
a five-decade sweep of the mixer's internal ridge, where unrolled *linear* mixing
converges normally. Elk's default `mixtype=3` is a Broyden scheme of the same shape. The
implicit route is indifferent by construction — which also means "implicit agrees between
mixers" proves nothing on its own, since its backward pass only ever sees $(\theta,v^*)$.

**Two traps that a badly chosen test walks straight past**, both measured here rather
than reasoned about. A *direction* that is a single real diagonal entry makes the naive
projector rule look correct in reverse mode (§0b), and a *perturbation that respects the
symmetry protecting a degeneracy* makes it look correct everywhere — error 1.5e-14
symmetric versus 4.7e-1 symmetry-broken on the same Hamiltonian (§0a), because neither
the perturbation nor the observable then has a matrix element between the partners.
Test along general directions, and break the symmetry; and always compare forward-mode
against reverse-mode, which for a scalar-in scalar-out function must agree exactly.

**A green gradient test does not validate a transcription.** Measured on 0c: dropping
`genylmv`'s $4\pi(-i)^l$ prefactor multiplies each $\ell$ block by a fixed complex
number, and the result still passes the exact `dmatch` identity to 7e-16 — a constant
factor commutes with $\partial/\partial\mathbf r_\alpha$. Every AD check here needs a
forward check beside it, and the strongest available without Elk is the quantity's own
defining equation (for `match`, the matching condition $DA=b$ rebuilt independently),
not a comparison of its pieces.

**Use an analytic reference, not finite differences, wherever a degeneracy is in play.** The
study's own §8(b) measures FD failing at a multiplet — central FD of the *sorted* spectrum
returns the branch average, so it cannot detect a wrong individual-eigenvalue gradient at all,
and near a degeneracy it is noise. For the occupied projector the closed form is available and
costs nothing:
$dP=\sum_{i\in W,\,j\notin W}\big(|i\rangle\langle i|\,dH\,|j\rangle\langle j| + \text{h.c.}\big)/(\lambda_i-\lambda_j)$,
gauge-invariant, exact, and valid at any $n$ — that is the reference `docs/continue_here.md` §3
says is missing, and it is what decides whether the custom rule is needed for a hard integer
window at all. **Measured, and the answer is yes**: over 3 assemblies x 21 Hermitian directions
on the disputed spectrum the naive route is wrong by $1.1\times10^{1}$ (forward) and
$3.4\times10^{0}$ (reverse) relative, against $2.9\times10^{-14}$ with the rule, while central
FD agrees with the closed form to $3.7\times10^{-8}$ — so FD was *reliable* here and the earlier
check's one-in-three disagreement was AD error, not FD noise. It saw agreement because it probed
a single real diagonal direction in reverse mode only; that same direction in forward mode is
already wrong at $1.3\times10^{-2}$. **Always check forward against reverse** — for a
scalar-in, scalar-out function they are the same number, so disagreement is proof on its own and
costs nothing. Also measured: which failure mode appears is the *eigensolver's* choice, since
LAPACK and XLA split the same engineered pair differently and XLA returns it bitwise equal at
$n=1000$ (finite garbage at $n=400$, `NaN` at $n=1000$, same code). The mechanism, for the
record: the two divergent terms are exact negatives and would cancel bitwise *if*
$A=v^\dagger\,\delta H\,v$ were bitwise Hermitian, and JAX's `_eigh_jvp_rule` forms it with no
symmetrisation — so $\|A-A^\dagger\|/\delta\lambda\approx0.2$–$0.5$ survives, which is the
size of the observed failure.

`docs/continue_here.md` is current as of §1k: both workstreams are on `master`, and its
§3 marks patches 0013/0014/0015, the κ(S) measurement, the projector rule at a real
multiplet, the `match` pole removal, the negative test, the smeared occupations, second
derivatives, the radial integrals/potential derivative and the frozen-potential position
derivative all done. The `ELKPY_F90_LIB` override it documents is still what builds Elk
here.


## Commands

- Build Elk out-of-tree (copies `vendor/elk/` to `build/elk/`, applies `patches/*.patch` if any, drops
  in `build-config/make.inc`, builds — never touches `vendor/elk/`):
  `./build_elk.sh`. Must be run before any test/example that actually invokes Elk.
  Serial by design — `make -j` races on an implicit ordering dependency in Elk's own `src/Makefile`
  (stub files like `mpi_stub.f90`/`libxcifc_stub.f90` must compile before the modules that `use` them,
  and that isn't expressed as an explicit prerequisite upstream); this is a pre-existing upstream
  issue, not something to fix by editing `vendor/elk/`.
- Install elkpy (editable): `python3 -m pip install -e .`
- Run tests: `python3 -m pytest tests/`. Run a single test: `python3 -m pytest tests/test_calculation_si.py::test_get_bands`.
  `tests/test_calculation_*.py` are integration suites that run the real `elk` binary on bulk Si/Fe —
  they self-skip if `build/elk/src/elk` doesn't exist yet, so run `./build_elk.sh` first to
  actually exercise them. `tests/test_structure.py`'s ASE round-trip test self-skips if `ase` isn't
  installed (`pip install -e .[ase]`).
- `tests/test_calculation_si_phonons.py` (DFPT phonon dispersion/DOS) is skipped by default even with
  the binary built — confirmed ~11-13 minutes per test on a minimal 2-atom, `ngridq=(2,2,2)` grid, cost
  dominated by DFPT's per-perturbation-per-q-point work, not anything elkpy controls. Set
  `ELKPY_RUN_SLOW_TESTS=1` to actually run them.
- `build-config/make.inc` targets GNU Fortran + OpenBLAS (which bundles LAPACK — do NOT re-add
  `-llapack`, it is absent in Spack-built OpenBLAS and redundant where it exists) + FFTW3 double and
  single precision, serial (no MPI); edit it (not `vendor/elk/make.inc`) to change compiler/library
  configuration. Those are the workstation defaults: `build_elk.sh` link-tests them and, when they
  fail, loads `${ELKPY_MODULES:-openblas fftw}`, adds `-Wl,-rpath` for every `LIBRARY_PATH` entry (so
  the binary still runs in a batch job with no modules loaded), and swaps `-march=native` for
  `-march=haswell -mtune=generic` whenever an environment-module system is present — a login node is
  routinely newer than the compute nodes, so `native` builds fine and then `SIGILL`s. Overrides:
  `ELKPY_F90_LIB` (verbatim, and a link failure is fatal), `ELKPY_MARCH`, `ELKPY_MODULES`. The
  resolved values are appended to the copied `build/elk/make.inc`, which is therefore a
  self-contained record of what was built. See `docs/design.md` §8.
