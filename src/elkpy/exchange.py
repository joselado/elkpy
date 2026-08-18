"""Orchestration for four-state energy mapping: turning a magnetic pair and a
choice of tensor components into constrained non-collinear Elk runs, and
their total energies back into an exchange tensor.

The arithmetic lives in parsers/exchange.py (unit-testable without Elk); this
module owns the expensive half -- building each configuration's `mommtfix`
block, running it in its own directory, and collecting the energies.

Why this needs no new Fortran (docs/design.md #29): Elk already constrains
per-atom moment DIRECTIONS natively. `fsmtype=-2` plus a per-atom `mommtfix`
adds a Lagrange-style field updated each SCF cycle (vendor/elk/src/
bfieldfsm.f90), and for the negative variant that field is projected
perpendicular to the target direction (`r3vo`), so the magnitude is free to
relax and the field does no work on the moment once it is on target.
Crucially, Elk's reported total energy is the CONSTRAINED energy functional
with no field contribution: energy.f90 forms the kinetic energy as
`engykn = evalsum - engyvcl - engyvxc - sm` where `sm = int m . B_s` uses the
TOTAL effective field, into which addbfsm.f90 has already folded the
constraining field. Its work is therefore removed exactly, for any fsmtype.

Two settings below are load-bearing rather than cosmetic:

* `nosym`/`reducek=0`. The four configurations of a component deliberately
  break the crystal symmetry, and checkfsm.f90 hard-`stop`s if `mommtfix` is
  not invariant under the symmetry group findsym.f90 found (which inspects
  `bfcmt` but NOT `mommtfix`). Beyond dodging that stop, an identical k-set
  across all four states is what lets their systematic errors cancel in the
  difference -- symmetry-reduced meshes would differ between configurations.

* `epsengy`. Elk's default is 1e-4 Ha ~ 2.7 meV, larger than the entire
  signal. The formula sums four total energies, so the noise floor is about
  four times the per-run convergence; the literature standard is 1e-5 eV
  (Sabani et al.; Hou et al.), and we default an order tighter still.
"""

import concurrent.futures
import numpy as np

from .parsers import exchange as ex
from .parsers.totenergy import parse_final_energy
from .structure import Structure

# Elk's own "do not fix this moment" sentinel is any component >= 1000
# (bfieldfsm.f90), so unconstrained atoms are simply left out of the block.
DEFAULT_EPSENGY = 1.0e-8
DEFAULT_TAUFSM = 0.005


def species_atom_numbers(structure, index):
    """Elk's 1-based (species number, atom number) for a 0-based global atom
    index, in the same ordering Structure.atom_index() produces."""
    running = 0
    for ispecies, (symbol, atoms) in enumerate(structure.species.items(), start=1):
        if index < running + len(atoms):
            return ispecies, index - running + 1, symbol
        running += len(atoms)
    raise ValueError(f"atom index {index} out of range ({running} atoms in structure)")


def atom_cartesian(structure, index):
    """Cartesian position (Bohr) of a 0-based global atom index."""
    avec = np.array(structure.avec, dtype=float) * structure.scale
    running = 0
    for atoms in structure.species.values():
        if index < running + len(atoms):
            frac = np.array(atoms[index - running][0], dtype=float)
            return frac @ avec
        running += len(atoms)
    raise ValueError(f"atom index {index} out of range ({running} atoms in structure)")


def n_atoms(structure):
    return sum(len(atoms) for atoms in structure.species.values())


def magnetic_indices(structure, symbols):
    """0-based global indices of every atom whose species is in `symbols`."""
    symbols = {symbols} if isinstance(symbols, str) else set(symbols)
    unknown = symbols - set(structure.species)
    if unknown:
        raise ValueError(
            f"species {sorted(unknown)} not in structure (known: {sorted(structure.species)})"
        )
    return [
        index
        for index in range(n_atoms(structure))
        if species_atom_numbers(structure, index)[2] in symbols
    ]


def configuration_moments(structure, magnetic, i, j, dir_i, dir_j, dir_spectator):
    """The moment direction of every magnetic atom in one of the four states:
    the pair (i, j) along their own axes, every other magnetic atom along the
    spectator axis -- identical in all four states, which is what makes the
    spectator terms cancel."""
    if i == j:
        raise ValueError("a four-state exchange pair needs two distinct atoms")
    for index in (i, j):
        if index not in magnetic:
            raise ValueError(f"atom {index} is not among the magnetic atoms {magnetic}")
    moments = {index: np.asarray(dir_spectator, dtype=float) for index in magnetic}
    moments[i] = np.asarray(dir_i, dtype=float)
    moments[j] = np.asarray(dir_j, dtype=float)
    return moments


def mommtfix_block(structure, moments, magnitude):
    """`mommtfix` lines: species number, atom number, and the target moment
    vector. With fsmtype=-2 only the DIRECTION of this vector is used (the
    constraining field is projected perpendicular to it), so `magnitude` sets
    a scale, not a constraint."""
    lines = []
    for index in sorted(moments):
        ispecies, iatom, _symbol = species_atom_numbers(structure, index)
        direction = np.asarray(moments[index], dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError(f"zero moment direction for atom {index}")
        vector = direction / norm * magnitude
        lines.append((ispecies, iatom) + tuple(float(v) for v in vector))
    return lines


def seeded_structure(structure, moments, field):
    """A copy of `structure` with per-atom bfcmt seeded along each atom's
    target direction.

    This is doing two jobs. It gives the SCF a starting magnetic
    configuration close to the constrained one (the constraint field alone,
    being perpendicular to the target, cannot create a moment from nothing --
    it can only rotate one that already exists). And it is what actually
    reduces the symmetry group: findsym.f90 inspects `bfcmt0`, so seeding it
    keeps Elk's own symmetry analysis consistent with the magnetic
    configuration we are imposing.

    Elk's convention is that the moment aligns ANTIPARALLEL to bfcmt (the
    field couples to the electron spin, which is antiparallel to its magnetic
    moment), so the seed is applied with a minus sign.
    """
    species = {}
    running = 0
    for symbol, atoms in structure.species.items():
        seeded = []
        for offset, (position, bfcmt) in enumerate(atoms):
            index = running + offset
            if index in moments:
                direction = np.asarray(moments[index], dtype=float)
                direction = direction / np.linalg.norm(direction)
                seeded.append((position, tuple(-field * v for v in direction)))
            else:
                seeded.append((position, bfcmt))
        species[symbol] = seeded
        running += len(atoms)
    return Structure(
        structure.avec,
        species,
        sppath=structure.sppath,
        scale=structure.scale,
        species_files=structure.species_files,
    )


# ---------------------------------------------------------------------------
# Running the configurations
# ---------------------------------------------------------------------------


def _configuration_calculation(calc, workdir, structure, moments, magnitude,
                               fsmtype, taufsm, epsengy, seed_field, extra_blocks):
    from .calculation import Calculation

    blocks = dict(calc.extra_blocks)
    blocks.update({
        "fsmtype": [fsmtype],
        "mommtfix": mommtfix_block(structure, moments, magnitude),
        "taufsm": [taufsm],
        "epsengy": [epsengy],
        # see the module docstring: identical k-set across all four states,
        # and checkfsm.f90's symmetry stop sidestepped
        "nosym": [True],
        "reducek": [0],
    })
    if seed_field:
        blocks.setdefault("reducebf", [0.8])
    blocks.update(extra_blocks or {})
    return Calculation(
        seeded_structure(structure, moments, seed_field) if seed_field else structure,
        workdir,
        xc=calc.xc,
        spinpol=True,
        spinorb=calc.spinorb,
        soc_scale=calc.soc_scale or None,
        rgkmax=calc.rgkmax,
        ngridk=calc.ngridk,
        vkloff=calc.vkloff,
        sppath=calc.sppath,
        launcher=calc.launcher,
        extra_blocks=blocks,
        raise_on_nonconvergence=False,
    )


def check_constraint(calc, moments, tol_degrees=10.0):
    """Verify that a finished configuration's muffin-tin moments actually
    point along their targets.

    This is the check the whole method rests on: `fsmtype=-2` constrains the
    direction only, so the magnitude is free to collapse, and a collapsed
    moment is not held by anything. A configuration that silently drifted off
    its target still produces a perfectly plausible total energy, so nothing
    downstream can detect it -- it has to be caught here.

    Returns a list of (atom_index, angle_degrees, magnitude) and raises if
    any constrained atom is off target by more than `tol_degrees`.
    """
    info = (calc.workdir / "INFO.OUT").read_text()
    if "Moments :" not in info:
        raise RuntimeError(f"{calc.workdir}/INFO.OUT has no moment output")
    block = info.rsplit("Moments :", 1)[1]
    measured = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("atom "):
            parts = stripped.split(":")[1].split()
            measured.append(np.array([float(p) for p in parts[:3]]))
        elif measured and ("total in muffin-tins" in stripped or "FSM" in stripped):
            break
    report, bad = [], []
    for index in sorted(moments):
        if index >= len(measured):
            raise RuntimeError(
                f"{calc.workdir}/INFO.OUT reports {len(measured)} muffin-tin moments, "
                f"too few for atom index {index}"
            )
        vector = measured[index]
        magnitude = float(np.linalg.norm(vector))
        target = np.asarray(moments[index], dtype=float)
        target = target / np.linalg.norm(target)
        if magnitude < 1e-6:
            angle = 180.0
        else:
            angle = float(np.degrees(np.arccos(np.clip(vector @ target / magnitude, -1, 1))))
        report.append((index, angle, magnitude))
        if angle > tol_degrees:
            bad.append((index, angle, magnitude))
    if bad:
        raise RuntimeError(
            f"constrained moments did not reach their targets in {calc.workdir}: "
            + ", ".join(f"atom {i}: {a:.1f} deg off, |m|={m:.3f}" for i, a, m in bad)
            + ". The four-state mapping assumes these directions exactly; see "
            "docs/design.md #29 on fsmtype=-2 letting the magnitude collapse."
        )
    return report


def run_configurations(calc, configurations, label="exchange", magnitude=1.0,
                       fsmtype=-2, taufsm=DEFAULT_TAUFSM, epsengy=DEFAULT_EPSENGY,
                       seed_field=2.0, extra_blocks=None, workers=1,
                       tol_degrees=10.0):
    """Run a {name: {atom_index: direction}} mapping of configurations and
    return {name: total energy in Hartree}.

    Each configuration is a separate Calculation in its own subdirectory, so
    Elk's own ground-state manifest caching applies per configuration: a
    re-run after a crash reuses whatever already converged instead of
    repeating the whole sweep.

    `workers` runs configurations concurrently. The work is embarrassingly
    parallel over configurations and each Elk process is serial, so threading
    the sweep beats threading one run; LocalLauncher's flock semaphore
    (ELKPY_MAX_CONCURRENT) still bounds how many actually run at once.
    """
    root = calc.workdir / label
    root.mkdir(parents=True, exist_ok=True)

    def run_one(item):
        name, moments = item
        sub = _configuration_calculation(
            calc, root / name, calc.structure, moments, magnitude,
            fsmtype, taufsm, epsengy, seed_field, extra_blocks,
        )
        sub.ensure_ground_state()
        if not sub.converged:
            raise RuntimeError(
                f"configuration {name} did not converge; the four-state difference needs "
                f"every one of its states converged (a single bad state invalidates the "
                f"whole component, unlike a least-squares fit over many configurations)"
            )
        check_constraint(sub, moments, tol_degrees=tol_degrees)
        return name, parse_final_energy(sub.workdir / "TOTENERGY.OUT")

    items = list(configurations.items())
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_one, items))
    else:
        results = [run_one(item) for item in items]
    return dict(results)


def _component_list(components):
    if components == "all":
        return [(a, b) for a in range(3) for b in range(3)]
    if components == "symmetric":
        return [(a, b) for a in range(3) for b in range(a, 3)]
    if components == "diagonal":
        return [(a, a) for a in range(3)]
    return [(ex.axis_index(a), ex.axis_index(b)) for a, b in components]


def exchange_tensor(calc, i, j, magnetic, spin=0.5, components="all",
                    label=None, workers=1, **run_kwargs):
    """The full four-state exchange tensor of the pair (i, j), in meV.

    `magnetic` is the list of 0-based global atom indices to constrain (every
    magnetic atom -- the spectators have to be held too, or they are not
    spectators). `components` is "all" (9 components), "symmetric" (6, then
    mirrored -- valid only when the bond midpoint is an inversion centre, so
    that D = 0 by Moriya's theorem), "diagonal", or an explicit list.

    The multiplicity is computed from the structure, not assumed: see
    parsers.exchange.check_supercell.
    """
    structure = calc.structure
    avec = np.array(structure.avec, dtype=float) * structure.scale
    r_i, r_j = atom_cartesian(structure, i), atom_cartesian(structure, j)
    distances = ex.image_distances(avec, r_i, r_j)
    multiplicity = ex.bond_multiplicity(avec, r_i, r_j)
    pairs = _component_list(components)

    configurations = {}
    index = {}
    for a, b in pairs:
        # spin=1.0 here: these vectors only carry DIRECTIONS (mommtfix_block
        # renormalizes them), while the physical S enters the formula below
        for state, directions in enumerate(ex.four_state_directions(a, b, spin=1.0), start=1):
            moments = configuration_moments(
                structure, magnetic, i, j,
                directions["i"], directions["j"], directions["spectator"],
            )
            name = f"{ex.AXES[a]}{ex.AXES[b]}_{state}"
            configurations[name] = moments
            index[(a, b, state)] = name

    label = label or f"exchange_{i}_{j}"
    energies = run_configurations(calc, configurations, label=label, workers=workers,
                                  **run_kwargs)
    quads = {
        (a, b): tuple(energies[index[(a, b, state)]] for state in (1, 2, 3, 4))
        for a, b in pairs
    }
    tensor = ex.assemble_tensor(quads, spin=spin, multiplicity=multiplicity)
    if components == "symmetric":
        tensor = np.where(np.isnan(tensor), tensor.T, tensor)
    return {
        "tensor": tensor,
        "multiplicity": multiplicity,
        "distance": float(distances[0]),
        "image_distances": distances[:6],
        "energies": energies,
        **ex.decompose(tensor),
    }
