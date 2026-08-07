"""Calculation: the pyqula-style central object, get_* methods delegating to
per-observable parsing (docs/design.md #3).

Unlike pyqula's Hamiltonian, a get_* call here runs a real subprocess and can
take anywhere from seconds to hours -- it is not a cheap in-memory operation,
despite the familiar naming.
"""

import json
from pathlib import Path

from . import config
from .inputfile import InputFile
from .launcher import LocalLauncher
from .parsers import band, dos, info, totenergy

XC_CODES = {
    "PZ": 2,      # LDA, Perdew-Zunger/Ceperley-Alder
    "PW": 3,      # LSDA, Perdew-Wang/Ceperley-Alder (Elk's default)
    "Xalpha": 4,  # LDA, X-alpha
    "vBH": 5,     # LSDA, von Barth-Hedin
    "PBE": 20,    # GGA, Perdew-Burke-Ernzerhof
    "RPBE": 21,   # GGA, revised PBE
    "PBEsol": 22, # GGA, PBEsol
    "WC06": 26,   # GGA, Wu-Cohen
    "AM05": 30,   # GGA, Armiento-Mattsson
}

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
    ):
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
        self._manifest_path = self.workdir / MANIFEST_NAME

    def _xctype_code(self):
        if isinstance(self.xc, int):
            return self.xc
        try:
            return XC_CODES[self.xc]
        except KeyError:
            raise ValueError(
                f"Unknown xc '{self.xc}'; use one of {sorted(XC_CODES)} or an explicit "
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
        for symbol, positions in self.structure.species.items():
            atoms_lines.append(f"{symbol}.in")
            atoms_lines.append(len(positions))
            for pos in positions:
                atoms_lines.append(tuple(pos) + (0.0, 0.0, 0.0))
        input_file.add_block("atoms", atoms_lines)

        input_file.add_block("xctype", [self._xctype_code()])
        input_file.add_block("spinpol", [self.spinpol])
        input_file.add_block("rgkmax", [self.rgkmax])
        input_file.add_block("ngridk", [ngridk or self.ngridk])
        input_file.add_block("vkloff", [self.vkloff])

    def _basis_signature(self):
        return {
            "avec": self.structure.avec,
            "species": {k: list(v) for k, v in self.structure.species.items()},
            "scale": self.structure.scale,
            "xctype": self._xctype_code(),
            "spinpol": self.spinpol,
            "rgkmax": self.rgkmax,
            "sppath": str(self.sppath),
            "ngridk": list(self.ngridk),
            "elk_binary": f"{self.launcher.elk_binary}:{self.launcher.elk_binary.stat().st_mtime}",
        }

    def _ground_state_valid(self):
        if not self._manifest_path.exists() or not (self.workdir / "STATE.OUT").exists():
            return False
        with open(self._manifest_path) as fh:
            saved = json.load(fh)
        # normalize through JSON (fresh signature has tuples, saved has lists
        # after the round trip -- compare their JSON forms, not Python equality)
        saved_sig = json.dumps(saved.get("basis_signature"), sort_keys=True)
        fresh_sig = json.dumps(self._basis_signature(), sort_keys=True)
        return saved_sig == fresh_sig and saved.get("converged")

    def ensure_ground_state(self):
        """Run task 0 if no valid converged ground state exists yet in this
        directory (docs/design.md #4: basis-defining parameters must match
        exactly for STATE.OUT to be considered reusable)."""
        if self._ground_state_valid():
            return
        f = InputFile()
        f.add_block("tasks", [0])
        self._add_base_blocks(f)
        f.write(self.workdir / "elk.in")
        self.launcher.run(self.workdir)
        converged = info.parse_convergence(self.workdir / "INFO.OUT")
        with open(self._manifest_path, "w") as fh:
            json.dump({"basis_signature": self._basis_signature(), "converged": converged}, fh)
        if not converged:
            raise RuntimeError(
                f"Ground state did not converge in {self.workdir} "
                f"(hit maxscl) -- see INFO.OUT"
            )

    def get_energy(self):
        """Total energy in Hartree (task 0/1, TOTENERGY.OUT)."""
        self.ensure_ground_state()
        return totenergy.parse_final_energy(self.workdir / "TOTENERGY.OUT")

    def get_bands(self, vertices, npoints=200):
        """Band structure along a path (task 20, BAND.OUT).

        `vertices` is a list of high-symmetry points in lattice (fractional
        reciprocal) coordinates, e.g. [(0,0,0), (0.5,0,0), (0.5,0.5,0)] --
        matches Elk's plot1d vlvp1d directly. Symbolic k-path labels
        (e.g. "GXWLGK") are not resolved automatically yet; that needs
        Bravais-lattice-aware special-point logic (spglib/seekpath-level),
        deferred beyond this first slice.

        Returns (distances, energies) with energies shape (nbands, npoints),
        Hartree, Fermi energy already subtracted by Elk.
        """
        self.ensure_ground_state()
        f = InputFile()
        f.add_block("tasks", [1, 20])
        self._add_base_blocks(f)
        f.add_block("plot1d", [(len(vertices), npoints)] + [tuple(v) for v in vertices])
        f.write(self.workdir / "elk.in")
        self.launcher.run(self.workdir)
        return band.parse_bands(self.workdir / "BAND.OUT")

    def get_dos(self, ngridk=None):
        """Total density of states (task 10, TDOS.OUT).

        `ngridk` optionally overrides the calculation's k-mesh for this call
        only (a sampling-only parameter, safe to differ from the ground
        state's ngridk -- see docs/design.md #4). Returns (energies, dos),
        Hartree / states-per-Hartree-per-unit-cell.
        """
        self.ensure_ground_state()
        f = InputFile()
        f.add_block("tasks", [1, 10])
        self._add_base_blocks(f, ngridk=tuple(ngridk) if ngridk else None)
        f.write(self.workdir / "elk.in")
        self.launcher.run(self.workdir)
        return dos.parse_dos(self.workdir / "TDOS.OUT")

    def run_tasks(self, tasks, blocks=None, resume=True):
        """Escape hatch for any Elk task not covered by a named get_*
        method (docs/design.md #3). Writes and runs an elk.in with the given
        raw task codes and extra input blocks; does not parse output -- the
        caller reads whatever files the task(s) produced from self.workdir.
        """
        if resume:
            self.ensure_ground_state()
        f = InputFile()
        f.add_block("tasks", ([1] if resume else [0]) + list(tasks))
        self._add_base_blocks(f)
        for name, lines in (blocks or {}).items():
            f.add_block(name, lines)
        f.write(self.workdir / "elk.in")
        self.launcher.run(self.workdir)
        return self.workdir
