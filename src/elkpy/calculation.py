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

import numpy as np

from . import config, spec
from .inputfile import InputFile
from .launcher import LocalLauncher
from .parsers import band, berry, dos, effmass, forces, geometry, info, quantum_geometry, totenergy, volumetric
from .session import EigenstateSession

MANIFEST_NAME = ".elkpy_manifest.json"


class Calculation:
    def __init__(
        self,
        structure,
        workdir,
        xc="PW",
        spinpol=False,
        spinorb=False,
        soc_scale=None,
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
        (e.g. {"maxscl": [200], "dft+u": [...]})  -- applied to every task
        this Calculation runs (ground state and every get_*/run_tasks call),
        and included in the ground-state cache signature (docs/roadmap.md
        Tier 1 #1).

        spinorb: enable spin-orbit coupling (elk.in `spinorb`, manual
        sec. 5.129) -- adds a sigma.L term to the second-variational
        Hamiltonian.

        soc_scale: optional {symbol: factor} overriding the strength of the
        spin-orbit coupling term for specific species (elkpy Fortran
        extension, `elkpy_socscale` block -- vendor/elk carries no per-
        species SOC control upstream, only the single global `socscf`
        scalar; see patches/0001-per-species-soc-scale.patch, which patches
        vendor/elk/src/gensocfr.f90 to look up a per-species scale before
        falling back to the global one). Species not listed keep Elk's
        default global scale (`socscf`, 1.0 unless overridden via
        extra_blocks). Requires spinorb=True -- otherwise SOC is off
        entirely and a scale has nothing to act on. Every key must be a
        species symbol present in `structure`.

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
        self.spinorb = spinorb
        self.soc_scale = dict(soc_scale or {})
        if self.soc_scale and not self.spinorb:
            raise ValueError(
                "soc_scale given but spinorb=False -- spin-orbit coupling is off, so a "
                "per-species scale has no effect; pass spinorb=True"
            )
        unknown = set(self.soc_scale) - set(structure.species)
        if unknown:
            raise ValueError(
                f"soc_scale species {sorted(unknown)} not in structure "
                f"(known species: {sorted(structure.species)})"
            )
        negative = {k: v for k, v in self.soc_scale.items() if v < 0}
        if negative:
            raise ValueError(f"soc_scale must be >= 0, got {negative}")
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
        input_file.add_block("spinorb", [self.spinorb])
        if self.soc_scale:
            species_index = {symbol: i + 1 for i, symbol in enumerate(self.structure.species)}
            input_file.add_block(
                "elkpy_socscale",
                [(species_index[symbol], scale) for symbol, scale in self.soc_scale.items()],
            )
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
            "spinorb": self.spinorb,
            "soc_scale": self.soc_scale,
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

    def _kpath_to_points(self, kpath, npoints):
        """Resolve a symbolic k-path (e.g. "GKMG" or "GKM,K'G") to a dense
        list of (kx, ky, kz) fractional-coordinate points plus their
        cumulative Cartesian distance along the path, via ASE's
        Bravais-lattice-aware special points and Cell.bandpath()
        discretization (optional 'ase' dependency, same as
        _kpath_to_vertices()).

        Unlike _kpath_to_vertices() (used by get_bands()/
        get_phonon_dispersion(), which hand raw *vertices* to Elk's own
        plot1d task for discretization -- and which therefore can't support
        a disconnected ',' path, since plot1d interpolates one continuous
        line), get_berry_curvature_path() evaluates one independent small
        Wilson loop per point rather than relying on Elk's path machinery at
        all, so the discretization happens here in Python and a disconnected
        ',' path is fine: each point is independent, so there's no
        interpolation across the break to get wrong.
        """
        try:
            from ase.cell import Cell
        except ImportError as exc:
            raise ImportError(
                "kpath= requires the optional 'ase' dependency "
                "(pip install elkpy[ase]); pass kpoints= directly to avoid it"
            ) from exc
        from .structure import BOHR_PER_ANGSTROM

        cell = Cell([[c / BOHR_PER_ANGSTROM for c in row] for row in self.structure.avec])
        bandpath = cell.bandpath(kpath, npoints=npoints)
        distances, _special_x, _special_labels = bandpath.get_linear_kpoint_axis()
        return [tuple(k) for k in bandpath.kpts], list(distances)

    def _reciprocal_vectors(self):
        """Cartesian reciprocal lattice vectors (rows, Bohr^-1), computed
        directly from self.structure.avec by the identical formula Elk's
        own src/reciplat.f90 uses (b1=2pi(a2xa3)/s, b2=2pi(a3xa1)/s,
        b3=2pi(a1xa2)/s, s=a1.(a2xa3), dividing by the signed s, not
        abs(s)) -- kept in Python rather than exported from a run so
        get_quantum_geometry() needs no new Fortran at all (unlike
        parsers.berry, which reads bvec back out of an Elk-written file)."""
        avec = np.array(self.structure.avec)
        a1, a2, a3 = avec
        s = np.dot(a1, np.cross(a2, a3))
        twopi = 2 * np.pi
        return np.array([twopi * np.cross(a2, a3) / s, twopi * np.cross(a3, a1) / s, twopi * np.cross(a1, a2) / s])

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
            spinorb=self.spinorb,
            soc_scale=self.soc_scale,
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

    def get_berry_curvature(self, ist0, ist1, directions=(1, 2), ngridk=None, gap_tol=1e-4):
        """Berry curvature via the Wilson-loop / Fukui-Hatsugai-Suzuki (FHS)
        method (task 9000, elkpy Fortran extension --
        patches/0002-berry-curvature-wilson-loop.patch, src/elkpy_berry.f90;
        not upstream Elk). See docs/design.md #13 / docs/physics.tex Part II
        for the physics (link variables, plaquette flux, admissibility) this
        implements, and cond-mat/0503172 for the original method.

        `ist0`/`ist1`: the contiguous, 1-indexed second-variational band
        window (e.g. the occupied valence bands) the non-Abelian Wilson loop
        is built from -- this window must stay gapped from the rest of the
        spectrum at every k-point on the mesh, checked automatically (raises
        ValueError otherwise; disable via `run_tasks` + `parsers.berry`
        directly if you understand the risk).

        `directions`: the two `ngridk` grid axes (1, 2 or 3, must differ)
        spanning the Wilson-loop plane -- e.g. (1, 2) computes curvature on
        the k1-k2 planes, with k3 as the free (slice) direction.

        `ngridk` optionally overrides the calculation's k-mesh for this call
        only, same as get_dos() -- a sampling-only parameter (docs/design.md
        #4), though a denser mesh is generally needed here than for DOS: see
        the returned "max_flux" diagnostic (the FHS admissibility condition,
        cond-mat/0503172 eq. 14) to judge convergence.

        Requires reducek=0 (set automatically) so that eigenvectors are
        available on the full, non-reduced ngridk mesh the Wilson loop walks.

        Returns a dict: {"flux": (n1,n2,n3) array (dimensionless plaquette
        flux, eq. 8), "chern_number": (n_free,) array (one Chern number per
        slice along the direction not in `directions`), "max_flux": float
        (admissibility diagnostic -- keep well under pi)}.
        """
        blocks = {
            "elkpy_berry": [(directions[0], directions[1], ist0, ist1)],
            "reducek": [0],
        }
        subdir = self._run_resumed(
            "berry", [spec.TASKS["berry_curvature"]], blocks, ngridk=tuple(ngridk) if ngridk else None
        )
        return berry.parse_berry_curvature(subdir / spec.OUTPUT_FILES["berry"], gap_tol=gap_tol)

    def get_berry_curvature_path(
        self,
        kpoints=None,
        ist0=None,
        ist1=None,
        directions=(1, 2),
        dk=0.005,
        kpath=None,
        npoints=100,
        label="berry_path",
    ):
        """Berry curvature at an arbitrary, explicit list of k-points (task
        9001, elkpy Fortran extension -- patches/0002-berry-curvature-wilson-loop.patch,
        src/elkpy_berry.f90). Unlike get_berry_curvature(), which evaluates
        the Fukui-Hatsugai-Suzuki construction over a full periodic mesh to
        get a Chern number, this evaluates one small Wilson loop per
        requested point via fresh on-the-fly diagonalisation from the
        converged potential -- the same construction pyqula's
        `berry_curvature(h, k, dk=...)` uses -- so the requested points need
        not lie on any mesh (e.g. an arbitrary band-structure-style path), at
        the cost of not producing a Chern number (that requires closing the
        loop over the whole Brillouin zone, which a handful of path points
        don't do).

        Pass exactly one of:
        - `kpoints`: explicit iterable of (kx, ky, kz) in fractional
          (lattice) coordinates.
        - `kpath`: a symbolic path string, e.g. "GKMG" (pyqula/ASE style),
          resolved via ASE's Bravais-lattice-aware special points and
          discretized into `npoints` points along the path (density
          proportional to each segment's Cartesian length -- ASE's
          Cell.bandpath() convention; optional 'ase' dependency). Unlike
          get_bands()/get_phonon_dispersion(), a disconnected ',' path (e.g.
          "GKM,K'G" to skip straight from M to a specific K' image) IS
          supported here -- see _kpath_to_points()'s docstring for why. Each
          returned point also carries a "distance" entry (cumulative
          Cartesian distance along the path, for plotting against the same
          x-axis as get_bands()).

        `ist0`/`ist1`: contiguous, 1-indexed band window, as in
        get_berry_curvature() -- but note this method does *not* check that
        the window stays gapped at every point (no eigenvalues are exported
        for this mode); a closing gap shows up as a near-singular overlap
        matrix, raising ValueError from parsers.berry, not a clean check.

        `directions`: the two reciprocal-lattice directions (1, 2 or 3, must
        differ) the small loop is built from, as in get_berry_curvature().

        `dk`: fractional-coordinate half-width of the loop around each
        point. This is the accuracy knob, with a floor: too large smears out
        sharply-peaked curvature (e.g. near a Dirac-like gap), too small and
        the overlap between two nearly-identical LAPW diagonalizations is
        dominated by basis-truncation noise. There's no single correct
        default -- check stability by evaluating one point of interest at a
        few `dk` values and looking for where the resulting curvature
        plateaus before trusting a full path.

        `npoints`: only used with `kpath`, total points distributed across
        the path (same meaning as get_bands()'s `npoints`).

        Returns a list of dicts, one per requested k-point, in the order
        given: {"k": (kx,ky,kz), "flux": dimensionless plaquette phase in
        (-pi,pi], "curvature": flux / loop area (1/Bohr^2)}, plus a
        "distance" entry when `kpath` was used.
        """
        if (kpoints is None) == (kpath is None):
            raise ValueError("pass exactly one of kpoints= or kpath=")
        if ist0 is None or ist1 is None:
            raise ValueError("ist0 and ist1 are required")
        distances = None
        if kpath is not None:
            kpoints, distances = self._kpath_to_points(kpath, npoints)
        kpoints = [tuple(k) for k in kpoints]
        blocks = {
            "elkpy_berry_path": [
                (directions[0], directions[1], dk, ist0, ist1),
                (len(kpoints),),
            ]
            + kpoints
        }
        subdir = self._run_resumed(label, [spec.TASKS["berry_curvature_path"]], blocks)
        result = berry.parse_berry_curvature_path(subdir / spec.OUTPUT_FILES["berry_path"])
        if distances is not None:
            for point, distance in zip(result, distances):
                point["distance"] = float(distance)
        return result

    def get_quantum_geometry(
        self,
        kpoints=None,
        ist0=None,
        ist1=None,
        directions=(1, 2),
        dk=0.005,
        kpath=None,
        npoints=100,
        label="quantum_geometry",
    ):
        """Quantum geometric tensor (Berry curvature *and* quantum metric)
        at an arbitrary, explicit list of k-points -- the metric-carrying
        companion to get_berry_curvature_path(), built from the same small
        loop of neighbouring k-points but entirely in Python, driven by
        EigenstateSession.overlap() queries (task 9002) rather than a new
        Fortran task: the loop's corner-to-corner overlaps
        get_berry_curvature_path() already needs for curvature, plus each
        corner's own self-overlap <psi(k)|psi(k)> (needed to
        Loewdin-normalize the metric -- see
        parsers.quantum_geometry._normalize_overlap()), are already exposed
        by the existing eigenstate/overlap session. See docs/design.md #15
        and docs/physics.tex Part IV for the physics (quantum metric,
        Fubini-Study distance, why the metric needs normalizing but
        curvature doesn't) this implements.

        Same kpoints=/kpath=/ist0/ist1/directions/dk/npoints conventions as
        get_berry_curvature_path() -- see its docstring for what each means
        (including the dk convergence-plateau caveat, which applies here
        too). `ist0`/`ist1` is not gap-checked here either, for the same
        reason (no eigenvalues exported by an overlap-only query).

        Unlike get_berry_curvature_path() (one Elk subprocess launch per
        point, via task 9001's fresh corner diagonalisation), this opens
        ONE eigenstate_session() and issues 19 overlap queries per point (9
        self-overlaps + 8 cross overlaps from k0, walking a 3x3 grid of
        corners centered at k0, + 2 more cross overlaps for curvature's own
        forward sub-loop) -- reusing the persistent session's "stay warm"
        setup (docs/design.md #14) rather than re-paying it 19 times per
        point. The centered grid (rather than just the forward quadrant) is
        what gives the metric's off-diagonal component g12 its correct
        k -> -k symmetry -- see parsers.quantum_geometry.compute_quantum_geometry's
        docstring and docs/design.md #15 for why.

        Returns a list of dicts, one per requested k-point, in the order
        given: {"k": (kx,ky,kz) fractional, "g": (2,2) array
        [[g11,g12],[g12,g22]] (quantum metric, Bohr^2), "berry_curvature":
        float (Bohr^-2, identical convention/value to
        get_berry_curvature_path()'s "curvature" -- see
        test_curvature_matches_berry_curvature_path_on_identical_corners), "Q": (2,2)
        complex array, Q = g - (i/2)*berry_curvature*[[0,1],[-1,0]]}, plus a
        "distance" entry when `kpath` was used.
        """
        if (kpoints is None) == (kpath is None):
            raise ValueError("pass exactly one of kpoints= or kpath=")
        if ist0 is None or ist1 is None:
            raise ValueError("ist0 and ist1 are required")
        distances = None
        if kpath is not None:
            kpoints, distances = self._kpath_to_points(kpath, npoints)
        kpoints = [tuple(k) for k in kpoints]

        bvec = self._reciprocal_vectors()
        v1 = dk * bvec[directions[0] - 1]
        v2 = dk * bvec[directions[1] - 1]
        d1, d2 = directions

        def displaced(k, n1, n2):
            k = list(k)
            k[d1 - 1] += n1 * dk
            k[d2 - 1] += n2 * dk
            return tuple(k)

        results = []
        with self.eigenstate_session(label=label) as session:
            for k0 in kpoints:
                k1p, k1m = displaced(k0, 1, 0), displaced(k0, -1, 0)
                k2p, k2m = displaced(k0, 0, 1), displaced(k0, 0, -1)
                k12pp, k12pm = displaced(k0, 1, 1), displaced(k0, 1, -1)
                k12mp, k12mm = displaced(k0, -1, 1), displaced(k0, -1, -1)
                overlaps = {
                    "s0": session.overlap(k0, k0, ist0, ist1),
                    "s1p": session.overlap(k1p, k1p, ist0, ist1),
                    "s1m": session.overlap(k1m, k1m, ist0, ist1),
                    "s2p": session.overlap(k2p, k2p, ist0, ist1),
                    "s2m": session.overlap(k2m, k2m, ist0, ist1),
                    "s12pp": session.overlap(k12pp, k12pp, ist0, ist1),
                    "s12pm": session.overlap(k12pm, k12pm, ist0, ist1),
                    "s12mp": session.overlap(k12mp, k12mp, ist0, ist1),
                    "s12mm": session.overlap(k12mm, k12mm, ist0, ist1),
                    "m1p": session.overlap(k0, k1p, ist0, ist1),
                    "m1m": session.overlap(k0, k1m, ist0, ist1),
                    "m2p": session.overlap(k0, k2p, ist0, ist1),
                    "m2m": session.overlap(k0, k2m, ist0, ist1),
                    "m12pp": session.overlap(k0, k12pp, ist0, ist1),
                    "m12pm": session.overlap(k0, k12pm, ist0, ist1),
                    "m12mp": session.overlap(k0, k12mp, ist0, ist1),
                    "m12mm": session.overlap(k0, k12mm, ist0, ist1),
                    "edge_b": session.overlap(k1p, k12pp, ist0, ist1),
                    "edge_c": session.overlap(k2p, k12pp, ist0, ist1),
                }
                result = quantum_geometry.compute_quantum_geometry(overlaps, v1, v2)
                results.append({"k": k0, **result})
        if distances is not None:
            for point, distance in zip(results, distances):
                point["distance"] = float(distance)
        return results

    def eigenstate_session(self, label="eigenstates"):
        """Start an interactive eigenstate/overlap query session (task
        9002, elkpy Fortran extension -- patches/0003-eigenstate-session.patch,
        src/elkpy_eigenstates.f90; not upstream Elk). See docs/design.md
        #14 for the physics (why evecfv isn't a valid raw-overlap basis but
        evecsv is, only within one diagonalisation; why cross-k overlaps
        need the genwfsvp/genolpq route) and design (why this is a
        persistent worker process rather than an f2py in-memory bridge)
        this implements.

        Returns an EigenstateSession kept alive across many queries,
        avoiding Elk's ground-state-dependent setup cost (readstate,
        genvsig, linengy, genapwlofr, gensocfr) being repeated per query --
        see EigenstateSession's docstring. Use as a context manager:

            with calc.eigenstate_session() as session:
                e = session.get_eigenstates((0, 0, 0))
                m = session.overlap((0, 0, 0), (0.1, 0, 0), ist0=1, ist1=4)

        For a single one-off query, get_eigenstates()/get_overlap() are
        thinner wrappers around a short-lived session.
        """
        self.ensure_ground_state()
        subdir = self.workdir / label
        shutil.rmtree(subdir, ignore_errors=True)
        subdir.mkdir(parents=True)
        shutil.copyfile(
            self.workdir / spec.OUTPUT_FILES["state"], subdir / spec.OUTPUT_FILES["state"]
        )
        f = InputFile()
        f.add_block(
            "tasks", [spec.TASKS["ground_state_resume"], spec.TASKS["eigenstate_session"]]
        )
        self._add_base_blocks(f)
        f.write(subdir / "elk.in")
        proc = self.launcher.start_session(subdir)
        # nspinor=2 whenever spinpol or spinorb is set -- spinorb forces
        # spinpol internally in Elk's own init0.f90 regardless of the
        # elk.in spinpol value, so it alone is enough to imply nspinor=2.
        nspinor = 2 if (self.spinpol or self.spinorb) else 1
        return EigenstateSession(proc, subdir, nspinor=nspinor)

    def get_eigenstates(self, k):
        """Second-variational energies (Hartree) and eigenvectors (evecsv)
        at a single k-point (fractional lattice coordinates), via fresh
        on-the-fly diagonalisation. A one-off convenience wrapper around
        eigenstate_session() -- opens and closes its own session, so prefer
        calling eigenstate_session() directly and reusing it for
        repeated/adaptive queries (each call here re-pays the
        ground-state-dependent setup cost eigenstate_session() is designed
        to amortize).

        Returns an Eigenstates(k, energies, evecsv) namedtuple -- see
        EigenstateSession.get_eigenstates().
        """
        with self.eigenstate_session() as session:
            return session.get_eigenstates(k)

    def get_overlap(self, k_a, k_b, ist0, ist1):
        """<psi_a(k_a)|psi_b(k_b)> for the contiguous band window
        [ist0, ist1]. A one-off convenience wrapper around
        eigenstate_session() -- see get_eigenstates()'s docstring about
        preferring eigenstate_session() directly for repeated queries, and
        EigenstateSession.overlap() for what this computes and why it's the
        only valid way to compare eigenstates across k-points.
        """
        with self.eigenstate_session() as session:
            return session.overlap(k_a, k_b, ist0, ist1)

    def get_atom_projection(self, k, ist0, ist1):
        """The atom-projection operator P_alpha for every atom alpha in the
        cell, restricted to the contiguous band window [ist0, ist1], at a
        single k-point. A one-off convenience wrapper around
        eigenstate_session() -- see get_eigenstates()'s docstring about
        preferring eigenstate_session() directly for repeated queries, and
        EigenstateSession.atom_projection() for what this computes, its
        (natmtot, nst, nst) return shape/atom ordering, and its gauge
        caveat.

        `self.structure.atom_index(symbol, index)` maps a (species, index)
        pair to this array's first axis.
        """
        with self.eigenstate_session() as session:
            return session.atom_projection(k, ist0, ist1)

    def get_orbital_projection(self, k, ist0, ist1):
        """The l-resolved atom-projection operators P_{alpha,l} for l=0,1,2,3
        (s, p, d, f) for every atom alpha in the cell, restricted to the
        contiguous band window [ist0, ist1], at a single k-point. A one-off
        convenience wrapper around eigenstate_session() -- see
        get_eigenstates()'s docstring about preferring eigenstate_session()
        directly for repeated queries, and EigenstateSession.orbital_projection()
        for what this computes, its (natmtot, 4, nst, nst) return shape/atom
        and l ordering, and its gauge caveat.

        `self.structure.atom_index(symbol, index)` maps a (species, index)
        pair to this array's first axis; `elkpy.session.ORBITAL_LABELS`
        gives the second axis's l order (s, p, d, f).
        """
        with self.eigenstate_session() as session:
            return session.orbital_projection(k, ist0, ist1)

    def get_spin_operator(self, k, ist0, ist1):
        """The spin operators S_x, S_y, S_z for the contiguous band window
        [ist0, ist1], at a single k-point. A one-off convenience wrapper
        around eigenstate_session() -- see get_eigenstates()'s docstring
        about preferring eigenstate_session() directly for repeated
        queries, and EigenstateSession.spin_operator() for what this
        computes, its return shape, and why it needs spinpol=True or
        spinorb=True.
        """
        with self.eigenstate_session() as session:
            return session.spin_operator(k, ist0, ist1)

    def get_angular_momentum(self, k, ist0, ist1):
        """The (orbital) angular momentum operators L_x, L_y, L_z,
        l-resolved for l=0,1,2,3 (s, p, d, f), for every atom alpha in the
        cell, restricted to the contiguous band window [ist0, ist1], at a
        single k-point. A one-off convenience wrapper around
        eigenstate_session() -- see get_eigenstates()'s docstring about
        preferring eigenstate_session() directly for repeated queries, and
        EigenstateSession.angular_momentum() for what this computes, its
        return shape, and why the su(2)/Casimir identities don't hold
        exactly on the returned (band-window-truncated) matrices.

        `self.structure.atom_index(symbol, index)` maps a (species, index)
        pair to this array's first axis; `elkpy.session.ORBITAL_LABELS`
        gives the l order (s, p, d, f) of the *_orbital fields.
        """
        with self.eigenstate_session() as session:
            return session.angular_momentum(k, ist0, ist1)

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
