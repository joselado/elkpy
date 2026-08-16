# Caveat found on the graphene + soc_scale=3000 fixture

Surfaced by the spin-Hall agent, then confirmed directly.

## Measurement

Ground state: 2-atom graphene, ngridk=(6,6,1), rgkmax=7.0, spinorb=True,
soc_scale={"C": 3000.0} -- the fixture shared by tests/test_calculation_z2.py
(docs/design.md #20) and tests/test_calculation_parity.py (#23).

EIGVAL.OUT occupation numbers, per k-point: n(occ>0.5) = 6, 8, 8, 8, 8, ...
No partial occupancies anywhere (every state is 0.0000 or 1.0000).

At the first k-point, states 5-6 sit at +10.72 eV and states 7-8 at +20.62 eV
(relative to state 1), so [1,6] is gapped by ~10 eV THERE. At other k-points
states 7-8 are occupied.

## Consequence

`ist1 = sum(o > 0.5 for o in occ)` evaluated on EIGVAL.OUT's FIRST k-point --
the idiom used by both fixtures -- therefore returns 6, and 6 is not the
filling at other k-points. The band count at fixed filling is not k-independent
here, i.e. this system is not a filled-band insulator at 8 states with SOC
scaled 3000x.

## What this does and does not affect

DOES NOT affect the method cross-check. get_z2_invariant() (WCC pumping) and
get_fu_kane_invariant() (parity) were both run on the SAME window and both give
1. Two independent routes agreeing on one gapped band group is exactly what
that comparison was for, and it stands.

DOES weaken the physical reading. "Graphene with intrinsic SOC is a QSH
insulator, Z2 = 1" (Kane & Mele) is a statement about the OCCUPIED manifold.
If the window used is a gapped band group that is not the occupied manifold at
every k, then Z2 = 1 is a true statement about that group but not automatically
the Kane-Mele claim. docs/design.md #20 and #23 currently present it as the
latter.

## Follow-up needed (not done)

1. Determine whether the system is genuinely gapped at fixed filling -- scan
   the direct gap above state 6 and above state 8 across the zone.
2. If it is not, either lower soc_scale to a value that keeps a clean gap at
   fixed filling while staying mesh-resolvable (#20 records that 100x was too
   small to resolve and 3000x was chosen for that reason -- there may be a
   usable window in between), or restate #20/#23's claim as being about the
   isolated [1,6] group rather than about the valence manifold.
3. The `sum(occ > 0.5)` idiom appears in several fixtures across the suite. It
   is correct only when the filling is k-independent; it should read the count
   from a k-point where the group is gapped, or better, verify k-independence.

No claim here that #20's Z2 = 1 is wrong -- it was not retested.
