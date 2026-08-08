"""Calculation: the pyqula-style central object, get_* methods delegating to
per-observable parsing (docs/design.md #3).

Unlike pyqula's Hamiltonian, a get_* call here runs a real subprocess and can
take anywhere from seconds to hours -- it is not a cheap in-memory operation,
despite the familiar naming.
"""

import hashlib
import json
import shutil
from pathlib import Path

from . import config, spec
from .inputfile import InputFile
from .launcher import LocalLauncher
from .parsers import band, dos, effmass, forces, geometry, info, totenergy, volumetric

MANIFEST_NAME = ".elkpy_manifest.json"


class Calculation:
    def __init__(
        self,
        structure,
        workdir,
        xc="PW",
        spinpol=False,
        rgkmax=7.0,
        ngridk=(4, 4, 4),
        vkloff=(0.0, 0.0, 0.0),
        sppath=None,
        launcher=None,
        extra_blocks=None,
        raise_on_nonconvergence=True,
    ):
        """
        extra_blocks: any elk.in block not covered by a named parameter above
        (e.g. {"spinorb": [True], "maxscl": [200], "dft+u": [...]})  --
        applied to every task this Calculation runs (ground state and every
        get_*/run_tasks call), and included in the ground-state cache
        signature (docs/roadmap.md Tier 1 #1).

        raise_on_nonconvergence: if True (default), ensure_ground_state()
        raises RuntimeError on non-convergence -- the safe default, since a
        bare energy/bands/dos value from a non-converged run is easy to
        mistake for a real result. Set False for sweep-style usage over many
        structures, where a non-converged point should be recorded (via the
        `converged` property) and skipped rather than crashing the whole
        sweep (docs/roadmap.md Tier 1 #2).
        """
        self.structure = structure
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.xc = xc
        self.spinpol = spinpol
        self.rgkmax = rgkmax
        self.ngridk = tuple(ngridk)
        self.vkloff = tuple(vkloff)
        self.sppath = Path(sppath) if sppath else (structure.sppath or config.resolve_species_path())
        self.launcher = launcher or LocalLauncher()
        self.extra_blocks = dict(extra_blocks or {})
        self.raise_on_nonconvergence = raise_on_nonconvergence
        self._manifest_path = self.workdir / MANIFEST_NAME
        self._converged = None

    def _xctype_code(self):
        if isinstance(self.xc, int):
            return self.xc
        try:
            return spec.XC_CODES[self.xc]
        except KeyError:
            raise ValueError(
                f"Unknown xc '{self.xc}'; use one of {sorted(spec.XC_CODES)} or an explicit "
                f"Elk xctype integer (see docs/elk_manual.txt sec. 5.150)"
            )

    def _add_base_blocks(self, input_file, ngridk=None):
        sppath_str = str(self.sppath)
        if not sppath_str.endswith("/"):
            sppath_str += "/"
        input_file.add_block("sppath", [sppath_str])
        if self.structure.scale != 1.0:
            input_file.add_block("scale", [self.structure.scale])
        input_file.add_block("avec", self.structure.avec)

        atoms_lines = [len(self.structure.species)]
        for symbol, atoms in self.structure.species.items():
            atoms_lines.append(self.structure.species_filename(symbol))
            atoms_lines.append(len(atoms))
            for position, bfcmt in atoms:
                atoms_lines.append(tuple(position) + tuple(bfcmt))
        input_file.add_block("atoms", atoms_lines)

        input_file.add_block("xctype", [self._xctype_code()])
        input_file.add_block("spinpol", [self.spinpol])
        input_file.add_block("rgkmax", [self.rgkmax])
        input_file.add_block("ngridk", [ngridk or self.ngridk])
        input_file.add_block("vkloff", [self.vkloff])

        for name, lines in self.extra_blocks.items():
            input_file.add_block(name, lines)

    def _basis_signature(self):
        return {
            "avec": self.structure.avec,
            "species": {k: list(v) for k, v in self.structure.species.items()},
            "species_files": dict(self.structure.species_files),
            "scale": self.structure.scale,
            "xctype": self._xctype_code(),
            "spinpol": self.spinpol,
            "rgkmax": self.rgkmax,
            "sppath": str(self.sppath),
            "ngridk": list(self.ngridk),
            "extra_blocks": self.extra_blocks,
            "elk_binary": f"{self.launcher.elk_binary}:{self.launcher.elk_binary.stat().st_mtime}",
        }

    def _load_valid_manifest(self):
        """Return the saved manifest dict if STATE.OUT exists and its
        recorded basis signature matches the current one, else None.

        Deliberately doesn't consider convergence -- a non-converged cached
        run is still a "valid" (matching) cache entry in the sense that its
        STATE.OUT/manifest don't need to be regenerated; ensure_ground_state
        is what decides whether that non-convergence should raise, via
        raise_on_nonconvergence.
        """
        if not self._manifest_path.exists() or not (self.workdir / spec.OUTPUT_FILES["state"]).exists():
            return None
        with open(self._manifest_path) as fh:
            saved = json.load(fh)
        # normalize through JSON (fresh signature has tuples, saved has lists
        # after the round trip -- compare their JSON forms, not Python equality)
        saved_sig = json.dumps(saved.get("basis_signature"), sort_keys=True)
        fresh_sig = json.dumps(self._basis_signature(), sort_keys=True)
        if saved_sig != fresh_sig:
            return None
        return saved

    @property
    def converged(self):
        """True/False once the ground state has actually run (this call or
        a cached prior one), None if ensure_ground_state() hasn't run yet.
        See `raise_on_nonconvergence` for whether non-convergence raises
        immediately instead of just being recorded here."""
        return self._converged

    def _raise_if_not_converged(self):
        if not self._converged and self.raise_on_nonconvergence:
            raise RuntimeError(
                f"Ground state did not converge in {self.workdir} (hit maxscl) -- see "
                f"INFO.OUT, or construct with raise_on_nonconvergence=False and check "
                f"calc.converged instead of raising"
            )

    def ensure_ground_state(self):
        """Run task 0 if no valid ground state exists yet in this directory
        (docs/design.md #4: basis-defining parameters must match exactly for
        STATE.OUT to be considered reusable). Does not re-run merely because
        a cached run didn't converge -- that's still a "valid" (matching)
        ground state in the cache sense; see `converged`/
        `raise_on_nonconvergence` for how non-convergence is surfaced, on
        both this path and the fresh-run path below."""
        manifest = self._load_valid_manifest()
        if manifest is not None:
            self._converged = manifest.get("converged")
            self._raise_if_not_converged()
            return
        f = InputFile()
        f.add_block("tasks", [spec.TASKS["ground_state"]])
        self._add_base_blocks(f)
        f.write(self.workdir / "elk.in")
        self.launcher.run(self.workdir)
        converged = info.parse_convergence(self.workdir / spec.OUTPUT_FILES["info"])
        with open(self._manifest_path, "w") as fh:
            json.dump({"basis_signature": self._basis_signature(), "converged": converged}, fh)
        self._converged = converged
        self._raise_if_not_converged()

    def _run_resumed(self, subdir_name, tasks, extra_blocks=None, ngridk=None):
        """Run task(s) resumed from the ground state's STATE.OUT, in an
        isolated subdirectory rather than self.workdir.

        This is deliberate, not just tidiness: task 1 rewrites STATE.OUT,
        TOTENERGY.OUT and INFO.OUT wherever it runs (src/gndstate.f90). If it
        ran directly in self.workdir, a get_dos(ngridk=...) call at a denser
        sampling mesh than the ground state would silently overwrite the
        ground state's own STATE.OUT/TOTENERGY.OUT while the manifest still
        recorded the original ngridk -- a later get_energy() would then
        report a cache hit and return the wrong (denser-mesh) energy. Running
        in a copy avoids this: self.workdir's STATE.OUT and manifest are
        never touched by anything other than ensure_ground_state().

        The subdirectory is wiped before each run, not just created if
        missing: some tasks (phonon DFPT in particular, task 205) treat the
        mere presence of prior output files (DYN*.OUT/DVS*.OUT) as "this
        part of the calculation is already done" and resume from them.
        Reusing a subdirectory that holds partial output from an earlier
        killed/crashed run -- e.g. this process got interrupted mid-run --
        silently corrupts the next run's result instead of erroring; a
        clean directory every time avoids that at the cost of not getting a
        free resume if a run is deliberately interrupted and retried later.
        """
        self.ensure_ground_state()
        subdir = self.workdir / subdir_name
        shutil.rmtree(subdir, ignore_errors=True)
        subdir.mkdir(parents=True)
        shutil.copyfile(
            self.workdir / spec.OUTPUT_FILES["state"], subdir / spec.OUTPUT_FILES["state"]
        )
        f = InputFile()
        f.add_block("tasks", [spec.TASKS["ground_state_resume"]] + list(tasks))
        self._add_base_blocks(f, ngridk=ngridk)
        for name, lines in (extra_blocks or {}).items():
            f.add_block(name, lines)
        f.write(subdir / "elk.in")
        self.launcher.run(subdir)
        return subdir

    def _default_label(self, tasks, blocks):
        label = f"tasks_{'_'.join(map(str, tasks))}"
        if blocks:
            digest = hashlib.sha1(
                json.dumps(blocks, sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
            label += f"_{digest}"
        return label

    def _resolve_vertices(self, vertices, kpath):
        if (vertices is None) == (kpath is None):
            raise ValueError("pass exactly one of vertices= or kpath=")
        if vertices is not None:
            return vertices
        return self._kpath_to_vertices(kpath)

    def _kpath_to_vertices(self, kpath):
        """Resolve a symbolic k-path (e.g. "GXWLGK") to plot1d vertices via
        ASE's Bravais-lattice-aware special points (optional dependency;
        docs/roadmap.md Tier 1 #6). Both Elk's vlvp1d and ASE's special
        points are fractional reciprocal-lattice coordinates ("lattice
        coordinates", manual sec. 5.94), so no extra conversion is needed
        beyond avec's Bohr -> Angstrom for ASE's Cell."""
        try:
            from ase.cell import Cell
            from ase.dft.kpoints import parse_path_string
        except ImportError as exc:
            raise ImportError(
                "kpath= requires the optional 'ase' dependency "
                "(pip install elkpy[ase]); pass vertices= directly to avoid it"
            ) from exc
        from .structure import BOHR_PER_ANGSTROM

        cell = Cell([[c / BOHR_PER_ANGSTROM for c in row] for row in self.structure.avec])
        special_points = cell.bandpath().special_points
        segments = parse_path_string(kpath)
        if len(segments) != 1:
            raise ValueError(
                f"kpath '{kpath}' has disconnected segments (a ',' break); only a single "
                f"connected path is supported here -- pass vertices= directly for that case"
            )
        try:
            return [tuple(special_points[label]) for label in segments[0]]
        except KeyError as exc:
            raise ValueError(
                f"unknown k-point label {exc} in kpath '{kpath}'; available: "
                f"{sorted(special_points)}"
            ) from exc

    def get_energy(self):
        """Total energy in Hartree (task 0/1, TOTENERGY.OUT)."""
        self.ensure_ground_state()
        return totenergy.parse_final_energy(self.workdir / spec.OUTPUT_FILES["totenergy"])

    def get_bands(self, vertices=None, kpath=None, npoints=200):
        """Band structure along a path (task 20, BAND.OUT).

        Pass exactly one of:
        - `vertices`: high-symmetry points in lattice (fractional
          reciprocal) coordinates, e.g. [(0,0,0), (0.5,0,0), (0.5,0.5,0)] --
          matches Elk's plot1d vlvp1d directly.
        - `kpath`: a symbolic path string, e.g. "GXWLGK", resolved via ASE's
          Bravais-lattice-aware special points (optional dependency; only a
          single connected path, no ',' breaks, is supported).

        Returns (distances, energies) with energies shape (nbands, npoints),
        Hartree, Fermi energy already subtracted by Elk.
        """
        vertices = self._resolve_vertices(vertices, kpath)
        blocks = {"plot1d": [(len(vertices), npoints)] + [tuple(v) for v in vertices]}
        subdir = self._run_resumed("bands", [spec.TASKS["bands"]], blocks)
        return band.parse_bands(subdir / spec.OUTPUT_FILES["band"])

    def get_dos(self, ngridk=None):
        """Total density of states (task 10, TDOS.OUT).

        `ngridk` optionally overrides the calculation's k-mesh for this call
        only (a sampling-only parameter, safe to differ from the ground
        state's ngridk -- see docs/design.md #4). Returns (energies, dos),
        Hartree / states-per-Hartree-per-unit-cell.
        """
        subdir = self._run_resumed(
            "dos", [spec.TASKS["dos"]], ngridk=tuple(ngridk) if ngridk else None
        )
        return dos.parse_dos(subdir / spec.OUTPUT_FILES["tdos"])

    def get_forces(self):
        """Total force on each atom, Hartree/Bohr (tforce enabled for this
        resumed run only; task 1, INFO.OUT "Forces :" section --
        src/writeforces.f90). Returns an (natoms, 3) array, atom order
        matching the atoms block (species in order, then atoms within each
        species in order)."""
        subdir = self._run_resumed("forces", [], extra_blocks={"tforce": [True]})
        return forces.parse_forces(subdir / spec.OUTPUT_FILES["info"])

    def get_relaxed(self, workdir=None):
        """Relax atomic positions (task 3, resumed from the ground state's
        STATE.OUT with the current positions as the starting point -- manual
        sec. 5.127). Only atomic positions are relaxed (lattice vectors
        fixed); pass `latvopt` via extra_blocks for cell relaxation.

        Returns a new Calculation (in `workdir`, default
        self.workdir/"relaxed") built from a new Structure with the final
        positions parsed from GEOMETRY_OPT.OUT, same parameters as this
        Calculation.

        Deliberately reconverges its own ground state from atomic densities
        rather than reusing this run's converged density: the relaxed
        Structure has different positions, so its _basis_signature()
        legitimately differs from this Calculation's, and seeding a
        "pre-validated" manifest for a directory whose STATE.OUT was never
        actually produced by a task-0 run at those exact positions would be
        the same class of stale-cache bug already fixed once for
        get_dos/get_bands (see _run_resumed's docstring). Correct-but-slower
        over fast-but-fragile.
        """
        from .structure import Structure

        subdir = self._run_resumed("relax", [spec.TASKS["relax_resume"]])
        # species_order: writegeom.f90 echoes species back in their original
        # declared order but only as the species *filename* -- if
        # species_files overrides that filename away from "{symbol}.in", the
        # filename alone can't recover the real element, so pass the
        # original order explicitly instead of guessing from the filename.
        avec, species = geometry.parse_last_geometry(
            subdir / spec.OUTPUT_FILES["geometry_opt"],
            species_order=list(self.structure.species.keys()),
        )
        relaxed_structure = Structure(
            avec,
            species,
            sppath=self.structure.sppath,
            # scale=1.0, not self.structure.scale: writegeom.f90 always
            # hardcodes scale to 1.0 and bakes any scaling directly into the
            # avec it writes (see parsers/geometry.py), so avec here is
            # already fully scaled -- reapplying self.structure.scale would
            # scale the lattice a second time.
            scale=1.0,
            species_files=self.structure.species_files,
        )
        new_workdir = Path(workdir) if workdir else self.workdir / "relaxed"
        return relaxed_structure.get_calculation(
            new_workdir,
            xc=self.xc,
            spinpol=self.spinpol,
            rgkmax=self.rgkmax,
            ngridk=self.ngridk,
            vkloff=self.vkloff,
            sppath=self.sppath,
            launcher=self.launcher,
            extra_blocks=self.extra_blocks,
            raise_on_nonconvergence=self.raise_on_nonconvergence,
        )

    def get_effective_mass(self, kpoint):
        """Effective mass tensor at a k-point (lattice coordinates), task 25
        (src/effmass.f90). Returns a list of per-state dicts: {"state",
        "eigenvalue", "tensor" (3x3, Cartesian), "trace"}."""
        subdir = self._run_resumed(
            "effmass", [spec.TASKS["effective_mass"]], extra_blocks={"vklem": [tuple(kpoint)]}
        )
        return effmass.parse_effective_mass(subdir / spec.OUTPUT_FILES["effmass"])

    def get_density(self, box=None, grid=(20, 20, 20)):
        """3D charge density on a grid (task 33, src/rhoplot.f90).

        `box` is the plot3d parallelepiped [origin, v1, v2, v3] in lattice
        coordinates (default: the unit cell). Returns (points, density):
        points shape (N,3) in Cartesian Bohr, density shape (N,).

        Potential (task 43, writes both VCL3D.OUT and VXC3D.OUT) and ELF
        (task 53) plots share this exact plot3d block/output format but
        aren't wrapped as named methods yet -- reachable via
        `run_tasks([43], blocks={"plot3d": [...]})` (or `[53]`) plus
        `parsers.volumetric.parse_plot3d` directly on the resulting file.
        """
        box = box or [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        blocks = {"plot3d": [tuple(box[0]), tuple(box[1]), tuple(box[2]), tuple(box[3]), tuple(grid)]}
        subdir = self._run_resumed("density", [spec.TASKS["density_3d"]], blocks)
        return volumetric.parse_plot3d(subdir / spec.OUTPUT_FILES["density_3d"])

    def get_phonon_dos(self, ngridq, nrmtscf=4, lmaxi=2):
        """Phonon density of states via DFPT (tasks 205 then 210,
        src/dyntask.f90 / src/phdos.f90).

        Only the DFPT method is wrapped here (single-shot within one
        invocation, per examples/phonons-superconductivity/*-DFPT). The
        classical supercell method (task 200) needs coordinated multi-run
        bookkeeping across displacements/machines and is reachable via
        run_tasks() instead if needed.

        nrmtscf/lmaxi default to the values Elk's own DFPT phonon examples
        use for accurate gradients; override if you know better values.
        Returns (frequencies, dos), same shape/units convention as get_dos.
        """
        blocks = {"ngridq": [tuple(ngridq)], "nrmtscf": [nrmtscf], "lmaxi": [lmaxi]}
        subdir = self._run_resumed(
            "phonon_dos", [spec.TASKS["phonon_dfpt"], spec.TASKS["phonon_dos"]], blocks
        )
        return dos.parse_dos(subdir / spec.OUTPUT_FILES["phdos"])

    def get_phonon_dispersion(
        self, vertices=None, kpath=None, ngridq=(4, 4, 4), npoints=200, nrmtscf=4, lmaxi=2
    ):
        """Phonon dispersion via DFPT (tasks 205 then 220). Same
        vertices/kpath interface as get_bands(). Returns (distances,
        frequencies) with frequencies shape (nbranches, npoints) -- reuses
        parsers/band.py, since src/phdisp.f90 writes the identical
        blank-line-separated block layout as BAND.OUT."""
        vertices = self._resolve_vertices(vertices, kpath)
        blocks = {
            "ngridq": [tuple(ngridq)],
            "nrmtscf": [nrmtscf],
            "lmaxi": [lmaxi],
            "plot1d": [(len(vertices), npoints)] + [tuple(v) for v in vertices],
        }
        subdir = self._run_resumed(
            "phonon_dispersion", [spec.TASKS["phonon_dfpt"], spec.TASKS["phonon_dispersion"]], blocks
        )
        return band.parse_bands(subdir / spec.OUTPUT_FILES["phdisp"])

    def run_tasks(self, tasks, blocks=None, resume=True, label=None):
        """Escape hatch for any Elk task not covered by a named get_*
        method (docs/design.md #3).

        If `resume` (default), runs task(s) resumed from STATE.OUT in an
        isolated subdirectory, same as get_bands/get_dos -- see
        _run_resumed's docstring for why. If not `resume`, runs a fresh task
        0 + `tasks` from atomic densities in its own subdirectory, also
        never touching self.workdir's ground state. Does not parse output --
        the caller reads whatever files the task(s) produced from the
        returned directory.

        The default `label` folds in a hash of `blocks` (not just `tasks`),
        so two calls with the same task numbers but different blocks don't
        silently overwrite each other's output directory -- pass an
        explicit `label` for a human-readable directory name instead.
        """
        label = label or self._default_label(tasks, blocks)
        if resume:
            return self._run_resumed(label, tasks, blocks)
        subdir = self.workdir / label
        shutil.rmtree(subdir, ignore_errors=True)
        subdir.mkdir(parents=True)
        f = InputFile()
        f.add_block("tasks", [spec.TASKS["ground_state"]] + list(tasks))
        self._add_base_blocks(f)
        for name, lines in (blocks or {}).items():
            f.add_block(name, lines)
        f.write(subdir / "elk.in")
        self.launcher.run(subdir)
        return subdir
