# Feature status ledger (Workstream A)

The per-feature record of what elkpy adds on top of Elk, and — the part that matters
most — **how far each one is actually verified**. This was the "Project status" section
of `CLAUDE.md` until it outgrew a file that is loaded into every session; `CLAUDE.md`
now carries the summary table and the traps, and this file carries the narrative.

Read the matching `docs/design.md §N` for the design reasoning and `docs/physics.tex`
Part for the physics writeup. Check `src/elkpy/` directly rather than assuming this
file describes current code; update both as they diverge.

---


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

Also implemented, as the first real entry in the Fortran patch series (`CLAUDE.md`, "Core constraint"):
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
questions, see `CLAUDE.md`, "Development practices") plus `soc_scale={"Cs": 3000.0}` (real
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
standing rule in `CLAUDE.md`, "Development practices") corrected this — $I4_1/amd$'s band sticking groups bands into
quartets without forbidding a gap at the actual filling (Watanabe, Po, Zaletel &
Vishwanath, PRL 117, 096404 (2016)), and published DFT confirms compressive [001]-strained
$\alpha$-Sn genuinely is a gapped TI (Huang & Liu, PRB 95, 201101(R) (2017)) — so this
probe was *inconclusive* (almost certainly measured a splitting inside a stuck quartet or
the semicore manifold), not a disproof. Separately, bulk Bi$_2$Se$_3$ (rhombohedral,
$R\bar3m$, sourced from a real deposited structure — Crystallography Open Database entry
9011965 — after an earlier *hand-converted* hexagonal-to-rhombohedral attempt gave a
self-contradictory ~11 Å "bond" from otherwise-correct literature parameters, establishing
this project's standing preference for database-sourced structures, `CLAUDE.md` "Development practices") converged with
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
`berry.py`/`quantum_geometry.py`/`calculation.py`/`CLAUDE.md` labelled Berry curvature
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

**The strongest evidence §31 has is not in `tests/`.** A separate session ran a converged
45-atom NiBr2 monolayer spin spiral (15 Ni per magnetic period, in-plane helix, `spinorb`)
through *both* task 9003 and task 9005 and got **agreement to 0.2% on every well-sampled
harmonic** — `rhomagv` in Fortran against the Python Gram-matrix contraction, two code paths
sharing nothing, on a system an order of magnitude past the graphene fixture. That is an
end-to-end check of §31 and of §30 at once. It is **not** vendored: the export is 1.85 GB, and a
naive decimation would break the check rather than shrink it, since the agreement is partly a
statement about k-summation *and* the comparison is only meaningful on a transverse sample count
sharing the supercell's period (see the aliasing note below). Capturing it as a fixture means
cutting to a couple of k-points on a safe grid and **re-establishing the 0.2% on the cut**, and
keeping both sides — the 9003 maps are 1.4 MB each; a fixture with only the 9005 side captures
half the check. Data at `/scratch/work/ladovj1/calculations/NiBr2_elk_stm/`, the user's to keep
or drop.

That run also found **four traps on the Python side, all now closed or documented** (report and
per-item triage: `docs/field_report_nibr2.md`; substance: `docs/design.md` §31). Two were
silent-failure bugs and are fixed with tests: `compute_transmission(energies=)` is **absolute**
while `get_vertical_transport(energies=)` is **relative to $E_F$**, and passing the relative one
to the parser returned a plausible small map at the wrong energy instead of an error (now a raise
against the exported window); and `amplitude_weights`' `occmax` defaulted to 2.0, a silent factor
of two for every spin-orbit run (now required). Two are documented: Elk 11's `ramdisk .true.`
default means a *separately* converged ground state leaves no `EVEC*.OUT` for a later 9003 —
elkpy's own path is immune, since `_run_resumed()` runs task 1 and 9003 in one process — and a
`plot2d` transverse average kills only the components with $p\equiv0\ (\mathrm{mod}\ n_2)$, so a
count not sharing the supercell's period **aliases the atomic lattice into a low harmonic**
(measured: $3.4\times10^{-3}$ read where the true value is $2.9\times10^{-6}$). Two items are open
and are decisions rather than fixes: promoting the reporter's validated `spin_ldos()`/`tip_image()`
(staged at `docs/field_report_nibr2_spin_ldos.py`, self-test passing) into
`parsers/transport.py`, and an `adopt_ground_state()` for post-processing an externally converged
run — both in `docs/continue_here.md` §4.

Also implemented, and unlike every entry above **not new physics at all** — it makes the physics
Elk *already has* reachable by name (`docs/design.md` §32, no `physics.tex` part and no notebook,
per `CLAUDE.md`'s routine-wrapping rule): six task-family mixins under `src/elkpy/tasks/`, composed in
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

---

## §34 — `STATE.OUT`: the format written down, and a fixture that pins it

**Verified against a committed run, no binary needed to check it.**

elkpy still does not read `STATE.OUT`; Elk's own `readstate` does, and nothing here
changes that. What is new is that the *conventions* inside the file are written down
in `docs/design.md` §34 with a file-and-line citation for each, and a committed
fixture makes them executable rather than remembered. Written because a sibling
project reads the format directly, and every convention in it has a plausible wrong
answer that produces a smooth, believable, incorrect density.

Three small additions, all cheap:

- `spec.ELK_VERSION = (11, 0, 2)`. It is also `STATE.OUT`'s first record, so a binary
  reader can assert the layout it was checked against. `tests/test_spec.py` asserts it
  in **both** directions — against `vendor/elk/src/modmain.f90`'s
  `version(3)` parameter, and against the fixture's first record — so an Elk bump
  fails loudly rather than silently invalidating `spec.py`.
- `parsers.info.parse_charges()`. The `Charges :` block of `INFO.OUT`
  (`src/writechg.f90`), one per SCF iteration with `index=-1` the converged one.
  Per-atom `chgmt` and core leakage come back in Elk's own `ias` order, which is
  `STATE.OUT`'s atom order, so the two index together.
- `tests/fixtures/h_sc/`: simple cubic hydrogen, one atom, $a = 3.0$ Bohr,
  unpolarised, $4\times4\times4$, everything else at Elk defaults, `tshift = .false.`
  pinned explicitly. Tasks 0 and 33 in one `elk.in`, so `STATE.OUT` and `RHO3D.OUT`
  are one consistent set — which `get_density()` cannot give, since it runs task 33
  in its own wiped subdirectory. Seconds on one core; `regenerate.sh` rebuilds it and
  prunes back to the six committed files.

Hydrogen because it has no core, so all-electron and valence are the same function
and `rhocore` confounds nothing.

**What the fixture actually establishes** (`tests/test_state_fixture.py`, 7 tests):
the header record order of `writestate.f90` read back end to end; $r_{\rm MT}$ =
1.4 recovered as the last radial mesh point, post-`autormt`; the density record
holding **both** `rhomt` and `rhoir` in one Fortran record, asserted by its exact
byte length; the inner region genuinely zero beyond `lmmaxi` = 4 for the first 129
of 197 radial points, and non-zero after, so `nrmti` is recoverable from a file that
never writes it; `efermi` present in the header; `ngvec` < `ngtot`.

The sharpest one is a single number. `rfpts` clamps $r$ up to $r_{\rm sp}(1)$ at the
nucleus, so `RHO3D.OUT` at the origin is exactly $\rho_{00}(r_1)y_{00}$ out of
`STATE.OUT` — 0.2859303012 both ways. That one equality pins the record layout, the
$lm$-fastest reshape, the $y_{00}$ factor, and $\rho$-not-$r^2\rho$ at once. It is
also physics: finite at the nucleus (a cusp, not a pole), just under the free-atom
$1/\pi$.

**And the trap the fixture caught**, which was written down wrong first and then
measured: `rhonorm` (`src/rhonorm.f90`, called from `rhomag.f90:24`, on by default)
adds a uniform constant to `rhoir` and to the $l=0$ channel of `rhomt` so the total
charge is right. It updates `chgmt`/`chgmttot` and sets `chgir = chgtot - chgmttot`;
it does **not** update `chgcalc`. Therefore:

- `chgmt` per atom is post-shift and does describe the `rhomt` in the file —
  reintegrating the $l=0$ channel recovers 0.6125761996 to 9e-6, which is Simpson
  against Elk's spline weights and nothing else. It is the clean integrated check.
- the printed `error` of 7.4e-4 is what `rhonorm` **corrected**, not a floor under a
  reintegration test. Read the other way round it sends someone hunting a 1e-3
  discrepancy that is not in their code.
- `chgir` is still not comparable to a sharp-boundary sum: before `rhonorm` it is a
  smooth-`cfunir` integral, after it a residual defined to close the total. Measured
  here: 0.38466 sharp against 0.38742 printed, 0.7% apart — nearly four times the
  printed error.
