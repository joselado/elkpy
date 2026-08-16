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
from .parsers import (
    band,
    berry,
    dielectric,
    dos,
    effmass,
    eigval,
    forces,
    geometry,
    info,
    moke,
    optical,
    quantum_geometry,
    symmetry,
    totenergy,
    volumetric,
    wilson,
)
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

    def _add_base_blocks(self, input_file, ngridk=None, vkloff=None):
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
        input_file.add_block("vkloff", [vkloff or self.vkloff])

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

    def _run_resumed(self, subdir_name, tasks, extra_blocks=None, ngridk=None, vkloff=None):
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

        `ngridk`/`vkloff` optionally override the calculation's own mesh/
        offset for this call only, same sampling-only-parameter reasoning as
        `ngridk` alone previously (docs/design.md #4) -- e.g.
        get_z2_invariant()'s `plane_offset` needs a k-mesh offset to 0.5 in
        one direction to sample the k_i=pi TRI plane, not just k_i=0.
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
        self._add_base_blocks(f, ngridk=ngridk, vkloff=vkloff)
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

    def get_effective_mass_sum_rule(
        self, k, ist, degeneracy_tol=1e-4, nstates=None, decompose=False,
        label="effmass_kp",
    ):
        """The same effective mass tensor get_effective_mass() computes,
        by an entirely different route: the k.p sum rule

            (1/m*)^{ab}_n = delta_ab
                + 2 sum_{m != n} Re[p^a_nm p^b_mn] / (eps_n - eps_m),

        i.e. from the momentum matrix elements of ONE diagonalisation at
        `k` (task 9002's MOMENTUM query, docs/design.md #22) rather than
        from a polynomial fit to eigenvalues on a small k-mesh around it
        (Elk task 25). See parsers.optical.effective_mass_tensor for the
        derivation, the units, the degeneracy guard and -- importantly --
        the truncation caveats: the sum's 1/(eps_n - eps_m) weight decays
        only as the FIRST power of the energy denominator, so it converges
        markedly more slowly in `nempty` than the 1/(delta eps)^2 Kubo
        geometric sums, and omitted core states are not negligible.

        `k` is a single k-point in fractional (lattice) coordinates and
        `ist` a 1-based band index -- unlike get_effective_mass(), which
        returns every state at once, since one diagonalisation's whole
        pmat is windowed here in Python. `nstates` truncates the m-sum to
        the lowest N states, which is how to trace the nempty convergence
        out of a single run; `decompose=True` additionally returns each
        intermediate band's own contribution, the interband decomposition
        of the mass that a finite-difference curvature cannot provide.

        Returns parsers.optical.effective_mass_tensor's dict, plus "k".
        Compare "inverse_mass" against get_effective_mass()'s
        "derivative_tensor" (its raw matrix of eigenvalue derivatives) and
        "mass" against its "tensor" (that matrix inverted), both Cartesian
        and in atomic units.
        """
        with self.eigenstate_session(label=label) as session:
            m = session.momentum(tuple(k))
        result = optical.effective_mass_tensor(
            m.energies, m.pmat, ist, degeneracy_tol=degeneracy_tol,
            nstates=nstates, decompose=decompose,
        )
        return {"k": tuple(k), **result}

    def get_density(self, box=None, grid=(20, 20, 20)):
        """3D charge density on a grid (task 33, src/rhoplot.f90).

        `box` is the plot3d parallelepiped [origin, v1, v2, v3] in lattice
        coordinates (default: the unit cell). Returns (points, density):
        points shape (N,3) in Cartesian Bohr, density shape (N,).

        The Kohn-Sham potential (task 43) and the ELF (task 53) are written
        by the same plot3d writer (src/plot3d.f90) in the same format -- see
        get_potential() and get_elf().
        """
        blocks = {"plot3d": self._plot3d_lines(box, grid)}
        subdir = self._run_resumed("density", [spec.TASKS["density_3d"]], blocks)
        return volumetric.parse_plot3d(subdir / spec.OUTPUT_FILES["density_3d"])

    @staticmethod
    def _plot3d_lines(box, grid):
        """The `plot3d` block's five lines: the parallelepiped's origin and
        three corner vectors in lattice coordinates (default: the unit
        cell), then the grid size (src/readinput.f90 case('plot3d'))."""
        box = box or [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        return [tuple(box[0]), tuple(box[1]), tuple(box[2]), tuple(box[3]), tuple(grid)]

    def get_potential(self, box=None, grid=(20, 20, 20), component="coulomb"):
        """3D Kohn-Sham potential on a grid (task 43, src/potplot.f90).

        Same `box`/`grid` convention and same (points, values) return as
        get_density(): points shape (N,3) in Cartesian Bohr, values shape
        (N,) in Hartree.

        The Kohn-Sham effective potential is the sum of two pieces, and
        task 43 writes each to its own file in a single run -- `component`
        only selects which one is parsed:

        - "coulomb" (VCL3D.OUT): the electrostatic potential v_C, i.e. the
          nuclear (-Z/r) plus Hartree terms. Strongly negative and
          divergent at each nucleus, so a grid point landing on an atom
          dominates the value range.
        - "xc" (VXC3D.OUT): the exchange-correlation potential v_xc, the
          functional derivative delta E_xc / delta n of the chosen `xc`.

        Asking for the other component re-runs the task (cheap for a small
        grid: no SCF beyond the resumed ground state, just readstate +
        plot3d).
        """
        try:
            filename = {
                "coulomb": spec.OUTPUT_FILES["potential_coulomb_3d"],
                "xc": spec.OUTPUT_FILES["potential_xc_3d"],
            }[component]
        except KeyError:
            raise ValueError(
                f"unknown potential component '{component}'; use 'coulomb' (VCL3D.OUT) "
                f"or 'xc' (VXC3D.OUT)"
            ) from None
        blocks = {"plot3d": self._plot3d_lines(box, grid)}
        subdir = self._run_resumed("potential", [spec.TASKS["potential_3d"]], blocks)
        return volumetric.parse_plot3d(subdir / filename)

    def get_elf(self, box=None, grid=(20, 20, 20)):
        """Electron localization function on a 3D grid (task 53,
        src/elfplot.f90).

        Same `box`/`grid` convention and same (points, values) return as
        get_density(). The (spin-averaged) ELF is

            f_ELF(r) = 1 / (1 + [D(r)/D0(r)]^2),

        with D(r) = (tau(r) - |grad n|^2/(4n))/2 the excess local kinetic
        energy density over its von Weizsaecker (single-orbital) value and
        D0(r) = (3/5)(6 pi^2)^(2/3) (n/2)^(5/3) the same quantity for the
        homogeneous electron gas at the local density (Becke and Edgecombe,
        J. Chem. Phys. 92, 5397 (1990); Burnus, Marques and Gross, PRA 71,
        010501 (2005), the reference elfplot.f90 itself cites). Values are
        dimensionless and bounded to [0, 1] by construction: 1 means
        perfect localization (a covalent bond or lone pair), 1/2 the
        homogeneous-electron-gas reference.

        Note (Elk's own ELF example, examples/ELF/BN): the ELF depends on
        density gradients and is not continuous at the muffin-tin
        boundaries unless the cut-offs are raised (rgkmax, gmaxvr, lmaxo,
        lmaxapw -- via extra_blocks -- or highq=.true.), so a plot at
        default settings can show a visible sphere-boundary seam that is a
        basis-set artefact, not physics.
        """
        blocks = {"plot3d": self._plot3d_lines(box, grid)}
        subdir = self._run_resumed("elf", [spec.TASKS["elf_3d"]], blocks)
        return volumetric.parse_plot3d(subdir / spec.OUTPUT_FILES["elf_3d"])

    def get_moke(self, wplot=(0.0, 0.5), nwplot=500, swidth=None, ngridk=None):
        """Complex magneto-optic Kerr angle (tasks 120 then 122,
        src/writepmat.f90 / src/moke.f90, KERR.OUT).

        Returns (energies, kerr): the photon energy grid (nw,) in Hartree
        and the complex Kerr angle (nw,) in DEGREES -- real part the Kerr
        rotation theta_K, imaginary part the ellipticity. From the optical
        conductivity tensor sigma, moke.f90 forms

            theta_K + i eta_K = -sigma_xy / (sigma_xx sqrt(1 + 4 pi i
                                             sigma_xx / omega)),

        the standard polar-Kerr expression; sigma_xx and sigma_xy come from
        src/dielectric.f90's Kubo formula (Physica Scripta T109, 170
        (2004)), which moke.f90 calls internally with optcomp fixed to the
        11 and 12 components. The effect is odd under time reversal: the
        off-diagonal sigma_xy vanishes without BOTH a net magnetization and
        spin-orbit coupling, so a nonmagnetic or SOC-free run returns
        identically zero rather than failing.

        Task 122 needs the momentum matrix elements (PMAT.OUT) on disk, so
        task 120 is run first in the same directory -- that pairing is the
        reason this is a named method rather than a bare run_tasks() call.

        - `wplot`/`nwplot`: the photon-energy window (Hartree) and number of
          grid points. dielectric.f90 clips a negative lower bound to zero,
          and moke.f90 returns exactly zero at omega = 0.
        - `swidth`: the smearing width (Hartree), whose reciprocal is the
          relaxation time entering the response function. Elk's own MOKE
          example (examples/TDDFT-optics/Ni-MOKE) raises it AFTER the
          ground-state run to smooth the spectrum -- which is exactly what
          passing it here does, since this runs resumed from an already
          converged STATE.OUT and never feeds `swidth` back into the ground
          state (a large smearing during the SCF cycle would suppress the
          moment).
        - `ngridk`: optionally overrides the k-mesh for this call only. The
          Kerr angle is a Brillouin-zone integral of interband transitions
          and converges slowly -- Elk's Ni example uses 32x32x32.
        """
        if not self.spinorb:
            raise ValueError(
                "get_moke() requires spinorb=True: the Kerr angle is driven by the "
                "off-diagonal conductivity sigma_xy, which vanishes identically without "
                "spin-orbit coupling to tie the magnetization to the orbital motion"
            )
        if not self.spinpol:
            raise ValueError(
                "get_moke() requires a magnetic ground state (spinpol=True, plus a "
                "symmetry-breaking field such as bfieldc or per-atom bfcmt): sigma_xy is "
                "odd under time reversal, so an unmagnetized cell gives zero Kerr angle"
            )
        blocks = {"wplot": [(nwplot, 100, 1), (wplot[0], wplot[1])]}
        if swidth is not None:
            blocks["swidth"] = [swidth]
        subdir = self._run_resumed(
            "moke",
            [spec.TASKS["momentum_matrix"], spec.TASKS["moke"]],
            blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        return moke.parse_kerr(subdir / spec.OUTPUT_FILES["kerr"])

    def get_dielectric_function(
        self, components=((1, 1),), wplot=(0.0, 0.5), nwplot=500, swidth=0.001,
        intraband=False, ngridk=None, label="dielectric",
    ):
        """Dielectric tensor and optical conductivity in the
        independent-particle (random-phase, no local fields, no excitons)
        approximation -- tasks 120 then 121, src/writepmat.f90 /
        src/dielectric.f90, EPSILON_ij.OUT / SIGMA_ij.OUT.

        dielectric.f90 evaluates the Kubo-Greenwood formula (Physica
        Scripta T109, 170 (2004)) over the NON-reduced k-mesh,

            sigma_ij(w) = (i / (N_k Omega)) sum_k sum_{n,m}
                f_n (1 - f_m/f_max) / (e_m - e_n)
                [ p^i_nm conj(p^j_nm) / (w - (e_m - e_n) + i s)
                + conj(p^i_nm conj(p^j_nm)) / (w + (e_m - e_n) + i s) ],

            eps_ij(w) = delta_ij + 4 pi i sigma_ij(w) / (w + i s),

        with s = `swidth`, Omega the cell volume, N_k the number of
        non-reduced k-points, f_n the occupations and p the momentum
        matrix elements task 120 writes to PMAT.OUT (getpmat rotates them
        from the reduced mesh to each non-reduced point). Im eps_ii is the
        interband absorption spectrum: taking s -> 0 turns the resonant
        denominator into -i pi delta(w - (e_m - e_n)) and leaves

            Im eps_ii(w) = (4 pi^2 / (N_k Omega w^2)) sum_k sum_{v,c}
                f_v (1 - f_c/f_max) |p^i_cv|^2 delta(w - (e_c - e_v)),

        which is exactly what get_circular_absorption() re-derives from
        elkpy's own arbitrary-k momentum matrix elements, resolved by
        circular polarization -- see docs/design.md #24.

        - `components`: which tensor components (i, j), 1-based Cartesian
          (1, 2, 3 = x, y, z), to compute; Elk's `optcomp` block, one run
          per call covering all of them. Default the xx component alone.
        - `wplot`/`nwplot`: photon-energy window (Hartree) and number of
          grid points. dielectric.f90 clips a negative lower bound to zero
          and its grid is w1 + (w2 - w1)*(iw - 1)/nwplot, i.e. it EXCLUDES
          the upper endpoint.
        - `swidth`: the Lorentzian broadening (Hartree) -- i/swidth is the
          complex relaxation time entering the denominators above. Elk's
          own default (0.001, readinput.f90) is stated explicitly here
          rather than left implicit, because any comparison against an
          independently computed spectrum has to use the same one. Note
          that `swidth` is also Elk's occupation smearing, so it applies to
          the task-1 continuation this run starts with -- immaterial for a
          gapped system, but for a metal set it on the Calculation
          (`extra_blocks`) so every route is smeared alike.
        - `intraband`: add the Drude term and write PLASMA_ij.OUT.
          Irrelevant (and zero) for a gapped system.
        - `ngridk`: optionally overrides the k-mesh for this call only. The
          spectrum is a Brillouin-zone integral and converges slowly in it.

        The number of empty states summed over is Elk's `nempty`, which
        belongs to the Calculation (`extra_blocks={"nempty": [...]}`) so
        that this and eigenstate_session()-driven sums see the same
        `nstsv` -- Elk's default leaves very few empty states and truncates
        the spectrum at a low energy.

        Returns {"energies": (nw,) Hartree, "epsilon": {(i, j): (nw,)
        complex}, "sigma": {(i, j): (nw,) complex}, "swidth": float,
        "plasma": {(i, j): float}} -- the last only when intraband=True.
        """
        components = [tuple(int(x) for x in c) for c in components]
        for i, j in components:
            if not (1 <= i <= 3 and 1 <= j <= 3):
                raise ValueError(f"optcomp entries must be Cartesian axes in 1..3, got {(i, j)}")
        blocks = {
            "wplot": [(nwplot, 100, 1), (wplot[0], wplot[1])],
            "swidth": [swidth],
            "optcomp": [(i, j) for i, j in components],
            "intraband": [intraband],
        }
        subdir = self._run_resumed(
            label,
            [spec.TASKS["momentum_matrix"], spec.TASKS["dielectric"]],
            blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        result = {"epsilon": {}, "sigma": {}, "swidth": float(swidth)}
        energies = None
        for i, j in components:
            names = {
                key: spec.OUTPUT_FILE_TEMPLATES[key].format(i=i, j=j)
                for key in ("epsilon", "sigma")
            }
            energies, eps = dielectric.parse_epsilon(subdir / names["epsilon"])
            result["epsilon"][(i, j)] = eps
            _, sig = dielectric.parse_sigma(subdir / names["sigma"])
            result["sigma"][(i, j)] = sig
        result["energies"] = energies
        if intraband:
            result["plasma"] = {}
            for i, j in components:
                if i != j:
                    continue
                path = subdir / spec.OUTPUT_FILE_TEMPLATES["plasma"].format(i=i, j=j)
                result["plasma"][(i, j)] = float(path.read_text().split()[0])
        return result

    def _kmesh(self, ngridk=None):
        """The FULL non-reduced k-mesh Elk itself generates, in fractional
        coordinates: vkl = (i + vkloff)/ngridk for i = 0..n-1 in each
        direction (src/init1.f90's boxl offset plus src/genppts.f90's
        [0, 1) mapping)."""
        ngridk = tuple(ngridk) if ngridk else self.ngridk
        offset = self.vkloff
        return [
            (
                (i1 + offset[0]) / ngridk[0],
                (i2 + offset[1]) / ngridk[1],
                (i3 + offset[2]) / ngridk[2],
            )
            for i1 in range(ngridk[0])
            for i2 in range(ngridk[1])
            for i3 in range(ngridk[2])
        ]

    def get_circular_absorption(
        self, wplot=(0.0, 0.5), nwplot=500, swidth=0.001, directions=(1, 2),
        ngridk=None, kpoints=None, weights=None, broadening="elk",
        label="absorption",
    ):
        """Polarization-resolved interband absorption, Im eps_+(w) and
        Im eps_-(w) for left- and right-circularly polarized light --
        elkpy's own beyond-stock-Elk optical spectrum, built from the
        arbitrary-k momentum matrix elements of the task-9002 MOMENTUM
        query (docs/design.md #22) rather than from a mesh-bound Fortran
        task.

        See parsers.optical.circular_absorption() for the formula, the
        conventions and the k -> -k argument for why the ZONE-INTEGRATED
        channels coincide in a non-magnetic crystal while the k-resolved
        dichroism does not. Stock Elk (task 121, get_dielectric_function())
        gives sigma_xx and sigma_xy but never sigma_+/sigma_-; the
        polarization-SUMMED total here equals Im eps_aa + Im eps_bb, which
        is exactly what that task computes, so the two are directly
        comparable (docs/design.md #24).

        One elk process for the whole sweep: this opens a single
        eigenstate_session() and queries every mesh point through it.

        - `kpoints`: explicit list of k-points to sum over (fractional
          coordinates); default is the full NON-reduced `ngridk` mesh Elk
          itself would generate. Symmetry reduction is not applicable here
          -- |P_pm|^2 is not invariant under the operations that fold the
          mesh, which is the physical content of the dichroism.
        - `weights`: per-k-point weights, default uniform. Passing a mask
          restricts the sum to a region of the zone (e.g. one valley),
          which is what separates the two circular channels.
        - `wplot`/`nwplot`: photon-energy window (Hartree) and number of
          grid points, on the same grid convention as
          get_dielectric_function() (upper endpoint excluded), so the two
          spectra can be compared point by point.
        - `swidth`: broadening (Hartree), Elk's own `swidth`; matching
          task 121 means matching it on both sides.
        - `broadening`: "elk" (default) evaluates src/dielectric.f90's own
          finite-`swidth` response, which is what makes the two spectra
          comparable point by point. "lorentzian"/"gaussian" instead use
          the textbook delta-function form, which is the same physics only
          as swidth -> 0: at a finite width the two differ by a factor
          (2w - Delta)/Delta, a few percent across a linewidth, and the
          delta form additionally blows up as 1/w^2 below the absorption
          edge, where a Lorentzian's tail is multiplied by a diverging
          prefactor.

        Occupations come from the ground state's own EIGVAL.OUT (Elk's
        occsv, the same array src/dielectric.f90 reads), and must be
        k-independent -- this independent-particle interband spectrum is
        defined for a gapped system, and a metal raises rather than being
        silently mis-counted.

        Returns parsers.optical.circular_absorption()'s dict plus
        "kpoints" and everything needed to redo the sum without re-running
        Elk -- "kdata" (the per-k-point momentum matrices and energies),
        "occupations", "volume" and "occmax". Feeding those back into
        parsers.optical.circular_absorption() with different `weights` (a
        valley mask) or a different `broadening` costs no Elk time at all.
        """
        self.ensure_ground_state()
        occupations = eigval.occupations_if_uniform(
            self.workdir / spec.OUTPUT_FILES["eigval"]
        )
        if kpoints is None:
            kpoints = self._kmesh(ngridk)
        kdata = self.get_momentum_matrix(kpoints=kpoints, label=label)
        w1 = max(wplot[0], 0.0)
        w2 = max(wplot[1], w1)
        omega = w1 + (w2 - w1) * np.arange(nwplot) / nwplot
        volume = abs(np.linalg.det(np.array(self.structure.avec))) * self.structure.scale**3
        occmax = 1.0 if (self.spinpol or self.spinorb) else 2.0
        result = optical.circular_absorption(
            kdata, omega, occupations, volume, occmax=occmax, swidth=swidth,
            directions=directions, weights=weights, broadening=broadening,
        )
        result["kdata"] = kdata
        result["kpoints"] = [tuple(kp) for kp in kpoints]
        result["occupations"] = occupations
        result["volume"] = volume
        result["occmax"] = occmax
        return result

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

        Sign convention: the standard A = i<u|grad_k u>, Omega = curl A
        (Xiao, Chang & Niu, RMP 82, 1959 (2010)), set in one place --
        parsers.berry._berry_phase() -- and shared by get_quantum_geometry()
        and parsers.optical's independent Kubo route. See docs/design.md #22.

        Returns a dict: {"flux": (n1,n2,n3) array (dimensionless plaquette
        Berry phase, minus the argument of eq. 8's link product),
        "chern_number": (n_free,) array (one Chern number per
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
        (-pi,pi], "curvature": flux / loop area (Bohr^2)}, plus a
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
        float (Bohr^2, identical convention/value to
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

    def get_z2_invariant(
        self,
        ist0,
        ist1,
        loop_direction=1,
        pump_direction=2,
        plane_offset=0.0,
        nkx=41,
        nt=21,
        gap_tol=1e-4,
        label="z2",
    ):
        """Z2 topological invariant of a 2D time-reversal-invariant
        insulator's occupied band window [ist0, ist1], via Wannier-charge-
        center (WCC) pumping (Yu, Qi, Bernevig, Fang, Dai, PRB 84, 075119
        (2011), arXiv:1101.2011; the "largest gap" crossing-counting method
        of Soluyanov & Vanderbilt, PRB 83, 235401 (2011), arXiv:1102.5600).
        See docs/design.md #20 and docs/physics.tex Part IX for the physics
        (why only half the Brillouin zone needs to be pumped, Kramers
        pairing at the two ends, additivity of Z2 across independently
        gapped band groups) this implements.

        Needs NO new Fortran: reuses task 9000's existing mesh-neighbour
        overlap export (elkpy Fortran extension, patches/0002-berry-
        curvature-wilson-loop.patch, src/elkpy_berry.f90 -- the same export
        get_berry_curvature() uses for its Chern-number arithmetic), just
        read here as a non-Abelian Wilson loop instead of an Abelian
        plaquette flux. All WCC/Z2 arithmetic happens in pure Python
        (parsers/wilson.py), independently unit-tested against synthetic
        overlap matrices (tests/test_wilson_gauge_invariance.py) -- this
        method's own job is only building the right mesh and slicing the
        Fortran-exported overlaps into per-pumping-step closed loops.

        `loop_direction` (default 1): the reciprocal-lattice direction
        (1, 2 or 3) the non-Abelian Wilson loop is closed over -- spans the
        FULL Brillouin zone (nkx points), relying on the SAME periodic mesh
        machinery already trusted for get_berry_curvature()'s Chern-number
        boundary closure.

        `pump_direction` (default 2, must differ from `loop_direction`):
        the pumping direction, swept only over the time-reversal-invariant
        HALF Brillouin zone [0, 0.5] (nt points, both endpoints included --
        k_pump=0 and k_pump=0.5 are the two time-reversal-invariant momenta
        where the WCC spectrum is Kramers-paired, arXiv:1101.2011 Sec. II).
        Internally requested as a full-Brillouin-zone mesh of
        2*(nt-1) points along this direction (so 0 and 0.5 both land
        exactly on mesh points), of which only the first nt (covering
        [0, 0.5]) are used.

        `plane_offset` (default 0.0, the only value a 2D calculation needs):
        the fractional (lattice) coordinate the THIRD direction -- the one
        that is neither `loop_direction` nor `pump_direction` -- is fixed
        at, via a one-point k-mesh offset in that direction only (`loop_direction`/
        `pump_direction`'s own offsets are always 0, enforced below). For a
        genuinely 2D system this third direction is just the vacuum
        direction and 0.0 is the only sensible value; get_z2_invariant_3d()
        uses 0.0 and 0.5 -- the two time-reversal-invariant PLANE positions,
        k_i=0 and k_i=pi -- to compute the six per-plane invariants a 3D
        strong/weak classification needs (docs/design.md #21).

        Raises ValueError if `self.vkloff` is nonzero in `loop_direction` or
        `pump_direction`: a nonzero offset there would silently shift the
        pumping direction's two sampled endpoints off the true
        time-reversal-invariant momenta, giving a wrong Z2 with no error --
        the same class of silent-wrong-answer already hit once with mesh
        aliasing (see the resolvability warning below).

        `ist0`/`ist1`: the contiguous, 1-indexed band window, as in
        get_berry_curvature() -- checked gapped at every mesh point via
        berry.check_gap() (same as get_berry_curvature(), same `gap_tol`),
        both a correctness guard (Z2 is only defined for a window gapped
        from the rest of the spectrum everywhere) and, for a window chosen
        as "all occupied valence bands" (the standard ab initio convention
        -- see docs/design.md #20 for why this gives the same Z2 as
        isolating just the topologically-relevant bands), a check that no
        band reordering by energy silently swapped in an unintended state
        anywhere on the mesh.

        `nkx`: mesh points spanning the closed Wilson-loop direction (the
        resolution of the loop integral itself). `nt`: pumping steps across
        the half Brillouin zone. Real DFT wavefunctions (unlike a toy
        tight-binding model) make each additional mesh point relatively
        expensive -- start modest (e.g. the defaults) and increase if
        `wannier_centers` looks under-resolved (jagged rather than smooth
        as a function of the pumping coordinate) or if a near-singular link
        overlap raises ValueError from parsers.wilson._unitarize.

        A gap passing `check_gap()` is NOT by itself evidence that `nkx`
        resolves it: `check_gap()` only checks the eigenvalue gap AT the
        sampled mesh points, not how narrow, in k, a small-gap feature
        (e.g. a weakly-split Dirac point) is between them -- a real case
        hit while testing this method (docs/design.md #20): graphene with
        `soc_scale=100` passed `check_gap()` (a genuine ~15 meV gap at K)
        but gave a wrong (aliased) Z2, because the anticrossing region was
        far narrower in k than the mesh spacing. Before trusting a result
        near a small, sharply-localized gap, check resolvability directly
        with a cheap `eigenstate_session().overlap()` scan across the
        narrow-gap point: singular values of the occupied-window overlap
        between neighbouring mesh points should stay close to 1, not drop
        well below it.

        Returns a dict: {"z2": 0 or 1, "wannier_centers": (nt, nst) array
        of WCC angles (radians, sorted, one row per pumping step),
        "pump": (nt,) array of the pumping-direction k-values sampled (in
        [0, 0.5], fractional lattice coordinates)}.
        """
        if loop_direction == pump_direction:
            raise ValueError("loop_direction and pump_direction must differ")
        if nt < 2:
            raise ValueError("nt must be at least 2 (both time-reversal-invariant endpoints)")
        if self.vkloff[loop_direction - 1] != 0.0 or self.vkloff[pump_direction - 1] != 0.0:
            raise ValueError(
                f"self.vkloff must be 0 in loop_direction/pump_direction (got "
                f"{self.vkloff}) -- a nonzero offset there moves the pumping "
                f"direction's endpoints off the time-reversal-invariant momenta and "
                f"silently gives a wrong Z2"
            )

        nky_full = 2 * (nt - 1)
        normal_direction = 6 - loop_direction - pump_direction  # the direction in {1,2,3} not used above
        directions = (loop_direction, pump_direction)
        blocks = {
            "elkpy_berry": [(directions[0], directions[1], ist0, ist1)],
            "reducek": [0],
        }
        ngridk = [1, 1, 1]
        ngridk[loop_direction - 1] = nkx
        ngridk[pump_direction - 1] = nky_full
        vkloff = list(self.vkloff)
        vkloff[normal_direction - 1] = plane_offset
        subdir = self._run_resumed(
            label, [spec.TASKS["berry_curvature"]], blocks,
            ngridk=tuple(ngridk), vkloff=tuple(vkloff),
        )
        parsed = berry.parse_berry_overlaps(subdir / spec.OUTPUT_FILES["berry"])
        berry.check_gap(parsed, tol=gap_tol)

        theta_by_step = []
        pump_values = []
        for j in range(nt):
            link_overlaps = []
            for i in range(nkx):
                key = [0, 0, 0]
                key[loop_direction - 1] = i
                key[pump_direction - 1] = j
                link_overlaps.append(parsed["overlaps"][tuple(key)][0])
            theta_by_step.append(wilson.wilson_loop_wannier_centers(link_overlaps))
            pump_values.append(j / nky_full)

        z2 = wilson.z2_from_wannier_centers(theta_by_step)
        return {
            "z2": z2,
            "wannier_centers": np.array(theta_by_step),
            "pump": np.array(pump_values),
        }

    def get_z2_invariant_3d(self, ist0, ist1, nkx=41, nt=21, gap_tol=1e-4, label="z2_3d"):
        """The full 3D strong/weak Z2 classification (nu0; nu1, nu2, nu3) of
        a 3D time-reversal-invariant insulator's occupied band window
        [ist0, ist1] (Fu, Kane & Mele, PRL 98, 106803 (2007),
        arXiv:cond-mat/0607699). See docs/design.md #21 and
        docs/physics.tex Part X for the physics (why the 6 TRI planes each
        reduce to a genuine 2D problem, why nu0 is basis-independent but
        (nu1,nu2,nu3) are not) this implements.

        Needs no new machinery beyond get_z2_invariant() itself: this calls
        it six times, once per (axis, plane_offset) pair -- axis=i fixed at
        k_i=0 and at k_i=0.5 (the two TRI values), for i=1,2,3, using the
        OTHER two reciprocal directions as that call's loop_direction/
        pump_direction (cyclically: axis 1 fixed -> loop/pump=(2,3); axis 2
        -> (3,1); axis 3 -> (1,2)) -- then combines the six 0/1 results via
        wilson.combine_3d_invariants() (FKM eqs. 2-3), which also checks
        that the strong index nu0 agrees across all three axis choices (an
        algebraic guarantee per FKM, so disagreement raises ValueError as a
        bug, not a physical result -- see combine_3d_invariants()'s
        docstring).

        This is SIX FULL runs of get_z2_invariant()'s own mesh (six
        separate task-1-resumed Elk subprocesses, not just six overlap
        exports) -- six times the cost of a single 2D get_z2_invariant()
        call at the same nkx/nt. Same `ist0`/`ist1`/`nkx`/`nt`/`gap_tol`
        meaning as get_z2_invariant() (including its resolvability warning
        -- a gap passing check_gap() is not evidence nkx resolves it),
        applied independently to each of the six planes.

        Returns a dict: {"nu0": 0 or 1, "nu": (nu1, nu2, nu3), "nu0_by_axis":
        (nu0 via axis 1, via axis 2, via axis 3) -- see
        combine_3d_invariants(), "planes": {(axis, offset):
        get_z2_invariant()'s own full return dict} for every one of the six
        planes, keyed by (1|2|3, 0.0|0.5)}.
        """
        axis_loop_pump = {1: (2, 3), 2: (3, 1), 3: (1, 2)}
        planes = {}
        for axis, (loop, pump) in axis_loop_pump.items():
            for offset in (0.0, 0.5):
                planes[(axis, offset)] = self.get_z2_invariant(
                    ist0, ist1, loop_direction=loop, pump_direction=pump,
                    plane_offset=offset, nkx=nkx, nt=nt, gap_tol=gap_tol,
                    label=f"{label}_axis{axis}_{'0' if offset == 0.0 else 'pi'}",
                )
        z_by_axis_offset = {key: result["z2"] for key, result in planes.items()}
        combined = wilson.combine_3d_invariants(z_by_axis_offset)
        combined["planes"] = planes
        return combined

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

    def _session_query_path(
        self, session_method, k, kpoints, kpath, ist0, ist1, npoints, label,
        require_window=True,
    ):
        """Shared plumbing for the five eigenstate_session() query wrappers
        below (get_atom_projection/get_orbital_projection/
        get_spin_operator/get_angular_momentum/get_momentum_matrix): a
        single `k=` call opens
        (and tears down) its own session and returns the underlying
        EigenstateSession method's namedtuple unchanged, same as before --
        for `kpoints=`/`kpath=`, opens ONE session and reuses it across
        every point (same kpoints=/kpath=/npoints convention as
        get_quantum_geometry(), including why re-opening a session per
        point would repay Elk's ground-state-dependent setup cost each
        time -- see its docstring), returning a list of dicts (one per
        point: "k" plus the namedtuple's own fields, plus "distance" when
        `kpath` was used) -- the same list-of-dicts shape
        get_berry_curvature_path()/get_quantum_geometry() already return.

        `require_window=False` is for get_momentum_matrix(), whose band
        window is optional (its underlying query has no window at all --
        see EigenstateSession.momentum()).
        """
        if sum(x is not None for x in (k, kpoints, kpath)) != 1:
            raise ValueError("pass exactly one of k=, kpoints=, or kpath=")
        if require_window and (ist0 is None or ist1 is None):
            raise ValueError("ist0 and ist1 are required")
        if k is not None:
            with self.eigenstate_session() as session:
                return getattr(session, session_method)(k, ist0, ist1)
        distances = None
        if kpath is not None:
            kpoints, distances = self._kpath_to_points(kpath, npoints)
        kpoints = [tuple(kp) for kp in kpoints]
        results = []
        with self.eigenstate_session(label=label) as session:
            for kp in kpoints:
                results.append(getattr(session, session_method)(kp, ist0, ist1)._asdict())
        if distances is not None:
            for point, distance in zip(results, distances):
                point["distance"] = float(distance)
        return results

    def get_atom_projection(
        self, k=None, ist0=None, ist1=None, kpoints=None, kpath=None, npoints=100,
        label="atom_projection",
    ):
        """The atom-projection operator P_alpha for every atom alpha in the
        cell, restricted to the contiguous band window [ist0, ist1]. A
        one-off convenience wrapper around eigenstate_session() -- see
        get_eigenstates()'s docstring about preferring eigenstate_session()
        directly for repeated queries, and EigenstateSession.atom_projection()
        for what this computes, its (natmtot, nst, nst) return shape/atom
        ordering, and its gauge caveat.

        Pass exactly one of `k` (single point, returns an AtomProjection
        namedtuple as before), `kpoints` (iterable of points), or `kpath`
        (symbolic path string, e.g. "GKMG") -- the latter two open ONE
        session and reuse it across every point, returning a list of dicts
        (see _session_query_path()'s docstring, and get_quantum_geometry()'s
        for the kpath=/npoints= convention).

        `self.structure.atom_index(symbol, index)` maps a (species, index)
        pair to this array's first axis.
        """
        return self._session_query_path(
            "atom_projection", k, kpoints, kpath, ist0, ist1, npoints, label
        )

    def get_orbital_projection(
        self, k=None, ist0=None, ist1=None, kpoints=None, kpath=None, npoints=100,
        label="orbital_projection",
    ):
        """The l-resolved atom-projection operators P_{alpha,l} for l=0,1,2,3
        (s, p, d, f) for every atom alpha in the cell, restricted to the
        contiguous band window [ist0, ist1]. A one-off convenience wrapper
        around eigenstate_session() -- see get_eigenstates()'s docstring
        about preferring eigenstate_session() directly for repeated
        queries, and EigenstateSession.orbital_projection() for what this
        computes, its (natmtot, 4, nst, nst) return shape/atom and l
        ordering, and its gauge caveat.

        Pass exactly one of `k` (single point, returns an
        OrbitalProjection namedtuple as before), `kpoints`, or `kpath` --
        see get_atom_projection()'s docstring for what the latter two do.

        `self.structure.atom_index(symbol, index)` maps a (species, index)
        pair to this array's first axis; `elkpy.session.ORBITAL_LABELS`
        gives the second axis's l order (s, p, d, f).
        """
        return self._session_query_path(
            "orbital_projection", k, kpoints, kpath, ist0, ist1, npoints, label
        )

    def get_spin_operator(
        self, k=None, ist0=None, ist1=None, kpoints=None, kpath=None, npoints=100,
        label="spin_operator",
    ):
        """The spin operators S_x, S_y, S_z for the contiguous band window
        [ist0, ist1]. A one-off convenience wrapper around
        eigenstate_session() -- see get_eigenstates()'s docstring about
        preferring eigenstate_session() directly for repeated queries, and
        EigenstateSession.spin_operator() for what this computes, its
        return shape, and why it needs spinpol=True or spinorb=True.

        Pass exactly one of `k` (single point, returns a SpinOperator
        namedtuple as before), `kpoints`, or `kpath` -- see
        get_atom_projection()'s docstring for what the latter two do.
        """
        return self._session_query_path(
            "spin_operator", k, kpoints, kpath, ist0, ist1, npoints, label
        )

    def get_angular_momentum(
        self, k=None, ist0=None, ist1=None, kpoints=None, kpath=None, npoints=100,
        label="angular_momentum",
    ):
        """The (orbital) angular momentum operators L_x, L_y, L_z,
        l-resolved for l=0,1,2,3 (s, p, d, f), for every atom alpha in the
        cell, restricted to the contiguous band window [ist0, ist1]. A
        one-off convenience wrapper around eigenstate_session() -- see
        get_eigenstates()'s docstring about preferring eigenstate_session()
        directly for repeated queries, and EigenstateSession.angular_momentum()
        for what this computes, its return shape, and why the su(2)/Casimir
        identities don't hold exactly on the returned (band-window-truncated)
        matrices.

        Pass exactly one of `k` (single point, returns an AngularMomentum
        namedtuple as before), `kpoints`, or `kpath` -- see
        get_atom_projection()'s docstring for what the latter two do.

        `self.structure.atom_index(symbol, index)` maps a (species, index)
        pair to this array's first axis; `elkpy.session.ORBITAL_LABELS`
        gives the l order (s, p, d, f) of the *_orbital fields.
        """
        return self._session_query_path(
            "angular_momentum", k, kpoints, kpath, ist0, ist1, npoints, label
        )

    def get_momentum_matrix(
        self, k=None, ist0=None, ist1=None, kpoints=None, kpath=None, npoints=100,
        label="momentum",
    ):
        """The momentum (equivalently, in atomic units for a local
        Kohn-Sham potential, velocity) matrix elements p^a_nm for all
        second-variational states, plus the eigenvalues of the same
        diagonalisation. A one-off convenience wrapper around
        eigenstate_session() -- see get_eigenstates()'s docstring about
        preferring eigenstate_session() directly for repeated queries, and
        EigenstateSession.momentum() for what this computes, its
        (3, nstsv, nstsv) return shape and Cartesian component order, and
        why the energies must travel with the matrix elements.

        Pass exactly one of `k` (single point, returns a Momentum
        namedtuple), `kpoints`, or `kpath` -- see get_atom_projection()'s
        docstring for what the latter two do.

        Unlike the other session wrappers, `ist0`/`ist1` are OPTIONAL and
        default to all states: the Kubo-form quantities in
        `elkpy.parsers.optical` (kubo_berry_curvature/kubo_quantum_metric,
        an independent code path for get_berry_curvature_path()/
        get_quantum_geometry()) sum over states outside a window, so they
        take the window themselves and need an unwindowed pmat here.
        `elkpy.parsers.optical.circular_polarization` turns the same pmat
        into the valley-selective circular dichroism of an interband
        transition. See docs/design.md #22.

        The number of empty states in the sum is set by Elk's `nempty`
        (Calculation(extra_blocks={"nempty": ...})); the default is low,
        and every Kubo quantity built from this converges from below as it
        rises.
        """
        return self._session_query_path(
            "momentum", k, kpoints, kpath, ist0, ist1, npoints, label,
            require_window=False,
        )

    def get_parity(
        self, k=None, ist0=None, ist1=None, kpoints=None, kpath=None, npoints=100,
        label="parity",
    ):
        """The inversion (parity) operator P_mn = <psi_m|I|psi_n> over the
        contiguous band window [ist0, ist1], plus the eigenvalues of the
        same diagonalisation. A one-off convenience wrapper around
        eigenstate_session() -- see EigenstateSession.parity() for what
        this computes, why it is defined only at a time-reversal-invariant
        momentum, and why the eigenvalues should be taken via
        parsers.symmetry.parity_eigenvalues() rather than from pmat's
        diagonal.

        Pass exactly one of `k`, `kpoints` or `kpath` -- see
        get_atom_projection()'s docstring for the latter two. For the
        Fu-Kane invariants use get_fu_kane_invariant() instead, which
        visits the right k-points for you.
        """
        return self._session_query_path(
            "parity", k, kpoints, kpath, ist0, ist1, npoints, label
        )

    def get_fu_kane_invariant(self, ist0, ist1, dimension=3, label="fu_kane"):
        """The Z2 topological invariant(s) from parity eigenvalues at the
        time-reversal-invariant momenta -- the symmetry-indicator route of
        Fu & Kane, PRB 76, 045302 (2007), requiring only 8 (3D) or 4 (2D)
        k-points instead of the Wannier-charge-center mesh sweep of
        get_z2_invariant()/get_z2_invariant_3d() (docs/design.md #20/#21).

        Valid only for a crystal with an inversion centre; without one this
        raises (Elk's `tsyminv`), and the WCC method remains the general
        fallback. Requires nspinor=2 (spinorb=True): with spin-orbit
        coupling absent, spin-SU(2) forces the two spin sectors to have
        opposite Chern numbers and Z2 is trivially 0, so the invariant
        carries no information.

        [ist0, ist1] must enclose the occupied manifold as a group gapped
        from the states above it, with both Kramers partners of every level
        inside -- a boundary that splits a pair raises rather than silently
        returning a wrong parity (parsers.symmetry.trim_delta).

        Returns, for dimension=3, {"nu0": int, "nu": (nu1, nu2, nu3),
        "deltas": {k: +-1}}; for dimension=2, {"nu": int, "deltas": {...}}.
        See docs/design.md #23.
        """
        if dimension not in (2, 3):
            raise ValueError(f"dimension must be 2 or 3, got {dimension}")
        if not (self.spinorb or self.spinpol):
            raise ValueError(
                "the Fu-Kane Z2 invariant requires a spinor calculation (spinorb=True): "
                "the delta_i product runs over one member of each KRAMERS PAIR, which "
                "needs nspinor=2 to exist. Without spin-orbit coupling, spin-SU(2) forces "
                "the two spin sectors to opposite Chern numbers and Z2 is trivially 0, so "
                "the invariant would carry no information even if it could be formed"
            )
        if self.spinpol and not self.spinorb:
            raise ValueError(
                "the Fu-Kane Z2 invariant requires time-reversal symmetry, which "
                "spinpol=True without spinorb=True breaks (a collinear magnet has no "
                "Kramers degeneracy); the formula does not apply"
            )
        trims = symmetry.TRIM_3D if dimension == 3 else symmetry.TRIM_2D
        deltas = {}
        with self.eigenstate_session(label=label) as session:
            for kpoint in trims:
                result = session.parity(kpoint, ist0, ist1)
                # the window must be a gapped band group at EVERY TRIM, not
                # merely Kramers-consistent -- see check_window_gap
                symmetry.check_window_gap(result.energies, ist0, ist1)
                deltas[kpoint] = symmetry.trim_delta(result.pmat)
        if dimension == 2:
            return {"nu": symmetry.fu_kane_z2_2d(deltas), "deltas": deltas}
        out = symmetry.fu_kane_z2_3d(deltas)
        out["deltas"] = deltas
        return out

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
