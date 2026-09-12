"""Spectra, band-structure variants, Fermi surfaces and the real-space
plotting family (``docs/design.md`` #3's ``get_*`` pattern).

This mixin closes three systematic gaps in elkpy's coverage of Elk's task
list.

**The 1D and 2D members of every plotting triple.** Elk's real-space
plotting tasks come in triples whose last digit selects dimensionality --
x1 a line, x2 a plane, x3 a parallelepiped -- and every member routes
through the same three writers (``src/plot1d.f90``, ``plot2d.f90``,
``plot3d.f90``). elkpy previously exposed only the 3D member of the
density/potential/ELF triples. Added here: the 1D and 2D density (31/32),
potential (41/42) and ELF (51/52) plots, plus all three dimensions of the
wavefunction modulus (61/62/63), magnetisation (71/72/73), exchange-
correlation field (81/82/83), electric field (141/142/143), m x B_xc
(151/152/153), div B_xc (91/92/93), meta-GGA W_xc (341/342/343) and
paramagnetic current density (371/372/373), the 1D static density (471),
the upstream spin-summed STM image (162) and the core wavefunctions (65).

**Fermi surfaces**, which elkpy lacked entirely: tasks 100/101/103/104
(``src/fermisurf.f90``) and 102 (``src/fermisurfbxsf.f90``, the ``.bxsf``
format XCrySDen reads), plus the Fermi-surface nesting function (105).

**Band character and expectation values**: the atom/l/m/spin-resolved band
structures of tasks 21-24 (``src/bandstr.f90``, the independent Fortran
code path ``docs/design.md`` sections 16/18/19 already lean on as a
cross-check), the partial and interstitial DOS that task 10 writes
alongside TDOS.OUT, muffin-tin L/S/J expectation values (15/16), the free-
atom Dirac eigenvalues (150), ELNES (140), the electron momentum density
for Compton scattering (170 then 171/172/173) and the smearing functions
themselves (14).

TWO TRAPS found in the Fortran and guarded here rather than documented:

1. ``src/init1.f90`` lines 130-134 REPLACE ``ngridk`` with ``np3d`` for
   tasks 100-104, and (except for 102) replace the k-point box with the
   ``plot3d`` box::

       if (any(task == [100,101,102,103,104])) then
         ngridk(:)=np3d(:)
         if (task /= 102) boxl(:,:)=vclp3d(:,:)
       end if

   So for a Fermi surface the ``plot3d`` block alone sets BOTH the plotting
   grid and the k-mesh, and passing ``ngridk`` does nothing at all. That is
   also why the write loop in ``fermisurf.f90`` runs over ``ngridk`` while
   the coordinates come from ``plotpt3d``/``np3d``: after the override they
   are the same grid. ``get_fermi_surface()`` therefore takes ``grid``, not
   ``ngridk``, and says so.

2. ``src/vecplot.f90``'s spin-polarisation guard covers tasks 72, 73, 82
   and 83 but NOT 71 or 81, which would read the unallocated ``magmt`` /
   ``bxcmt``. The 1D magnetisation and 1D B_xc are refused here instead.

Task codes and filenames are looked up through :mod:`elkpy.spec` first, so
that module stays the single version-coupled registry, with the literal in
this file as the fallback that keeps this module importable before those
entries are merged.
"""

import shutil
from pathlib import Path

import numpy as np

from .. import spec
from ..parsers import atomicstates, character, fermisurface, plots, volumetric


def _task(key, code):
    """The task code for `key`, from spec.TASKS if present."""
    return spec.TASKS.get(key, code)


def _file(key, name):
    """The output filename for `key`, from spec.OUTPUT_FILES if present."""
    return spec.OUTPUT_FILES.get(key, name)


def _template(key, tmpl):
    """The indexed output filename template for `key`."""
    return spec.OUTPUT_FILE_TEMPLATES.get(key, tmpl)


#: `kind` -> task code for get_fermi_surface(), src/fermisurf.f90
FERMI_SURFACE_KINDS = {
    "product": 100,
    "bands": 101,
    "delta": 103,
    "band_deltas": 104,
}

#: `kind` -> task code for get_band_character(), src/bandstr.f90
BAND_CHARACTER_KINDS = {"l": 21, "lm": 22, "spin": 23, "moment": 24}

_DEFAULT_LINE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
_DEFAULT_PLANE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]


class SpectraTasks:
    """``get_*`` methods for Elk's spectra / band / Fermi-surface / plotting
    tasks. Mixed into :class:`elkpy.calculation.Calculation`."""

    # ------------------------------------------------------------------
    # shared plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _plot1d_lines(vertices, npoints):
        """The ``plot1d`` block: ``nvp1d npp1d`` followed by one line per
        vertex in lattice coordinates (src/readinput.f90 case('plot1d')).

        ``src/plotpt1d.f90`` distributes ``npp1d`` points over the segments
        joining the vertices in proportion to each segment's Cartesian
        length, so the sampling is uniform in distance rather than per
        segment. For a real-space plot the vertices are fractional
        coordinates of ``avec``; for the reciprocal-space plots
        (get_electron_momentum_density(dim=1)) they are fractional
        coordinates of ``bvec``.
        """
        vertices = [tuple(float(x) for x in v) for v in vertices]
        if len(vertices) < 2:
            raise ValueError("a 1D plot needs at least two vertices")
        if npoints < len(vertices):
            raise ValueError(
                f"npoints={npoints} is fewer than the {len(vertices)} vertices; "
                f"src/plotpt1d.f90 refuses npp1d < nvp1d"
            )
        return [(len(vertices), int(npoints))] + vertices

    # ``_ndmag()`` (Elk's number of magnetisation components, from
    # src/init0.f90) lives on Calculation itself: this family and the
    # magnetism/ULR family both need it, and two transcriptions of one
    # Fortran rule drift apart. See Calculation._ndmag.

    def _atom_filenames(self, template):
        """Map (species symbol, 1-based atom index) -> the per-atom filename
        Elk builds as ``"...S",I2.2,"_A",I4.4,".OUT"`` from the species
        index and the atom index within that species (src/bandstr.f90,
        src/dos.f90, src/wfcrplot.f90 all share the convention).

        The species index is the position in the ``atoms`` block, which
        ``Structure.species`` preserves.
        """
        names = {}
        for i, (symbol, atoms) in enumerate(self.structure.species.items(), start=1):
            for ia in range(1, len(atoms) + 1):
                names[(symbol, ia)] = template.format(species=i, atom=ia)
        return names

    def _run_plot(
        self,
        label,
        tasks,
        dim,
        filename,
        nf=1,
        lines_filename=None,
        line=None,
        npoints=200,
        plane=None,
        box=None,
        grid=None,
        extra_blocks=None,
        ngridk=None,
    ):
        """Run one member of a plotting triple and parse its output.

        `dim` picks which of the ``plot1d``/``plot2d``/``plot3d`` blocks is
        written and which parser is used; `filename` is that task's output
        file for this dimension; `nf` the number of function columns the
        task writes (1 for a scalar field, 3 for a vector field).
        """
        blocks = dict(extra_blocks or {})
        if dim == 1:
            blocks["plot1d"] = self._plot1d_lines(line or _DEFAULT_LINE, npoints)
        elif dim == 2:
            blocks["plot2d"] = self._plot2d_lines(plane or _DEFAULT_PLANE, grid or (40, 40))
        elif dim == 3:
            blocks["plot3d"] = self._plot3d_lines(box, grid or (20, 20, 20))
        else:
            raise ValueError(f"dim must be 1, 2 or 3, got {dim}")
        subdir = self._run_resumed(
            label, tasks, blocks, ngridk=tuple(ngridk) if ngridk else None
        )
        return self._parse_plot(subdir, dim, filename, nf, lines_filename)

    @staticmethod
    def _parse_plot(subdir, dim, filename, nf, lines_filename=None):
        """Parse one plotting-family output file into this module's uniform
        dict return (see :meth:`get_density_1d` for the key list)."""
        if dim == 1:
            distances, values = plots.parse_plot1d(subdir / filename, nf=nf)
            result = {"distance": distances, "values": values}
            if lines_filename is not None and (subdir / lines_filename).exists():
                result["vertices"] = plots.parse_plot_lines(subdir / lines_filename)
            return result
        if dim == 2:
            points, values, grid = volumetric.parse_plot2d(subdir / filename, nf=nf)
            result = {"points": points, "grid": grid, "values": values}
            if nf == 1:
                result["values_grid"] = plots.reshape_plot2d(values, grid)
            else:
                result["values_grid"] = np.stack(
                    [plots.reshape_plot2d(values[:, i], grid) for i in range(nf)]
                )
            return result
        points, values, grid = plots.parse_plot3d(subdir / filename, nf=nf)
        result = {"points": points, "grid": grid, "values": values}
        if nf == 1:
            result["values_grid"] = plots.reshape_plot3d(values, grid)
        else:
            result["values_grid"] = np.stack(
                [plots.reshape_plot3d(values[:, i], grid) for i in range(nf)]
            )
        return result

    def _require_spinpol(self, what):
        if not (self.spinpol or self.spinorb):
            raise ValueError(
                f"{what} requires a spin-polarised calculation (spinpol=True and/or "
                f"spinorb=True): without it Elk stores no magnetisation or "
                f"exchange-correlation magnetic field at all"
            )

    # ------------------------------------------------------------------
    # charge density, potential, ELF -- the 1D and 2D siblings of the
    # existing get_density()/get_potential()/get_elf()
    # ------------------------------------------------------------------

    def get_density_1d(self, line=None, npoints=200, label="density_1d"):
        """Charge density along a line (task 31, src/rhoplot.f90 ->
        RHO1D.OUT, RHOLINES.OUT).

        `line` is a list of at least two vertices in LATTICE coordinates
        (fractional coordinates of ``avec``); the default runs from the
        origin to the end of the first lattice vector. ``npoints`` points
        are distributed along the segments in proportion to their Cartesian
        length (src/plotpt1d.f90).

        Returns a dict:

        - "distance" (N,): cumulative Cartesian distance along the path,
          Bohr.
        - "values" (N,): the density in electrons/Bohr^3.
        - "vertices": the distance at each vertex, for drawing gridlines
          (from RHOLINES.OUT).

        Every 1D member of this family returns the same three keys; the 2D
        and 3D members return "points"/"grid"/"values"/"values_grid"
        instead.
        """
        return self._run_plot(
            label,
            [_task("density_1d", 31)],
            1,
            _file("density_1d", "RHO1D.OUT"),
            lines_filename=_file("density_lines", "RHOLINES.OUT"),
            line=line,
            npoints=npoints,
        )

    def get_density_2d(self, plane=None, grid=(40, 40), label="density_2d"):
        """Charge density on a plane (task 32, src/rhoplot.f90 ->
        RHO2D.OUT).

        `plane` is the ``plot2d`` parallelogram ``[origin, v1, v2]`` in
        lattice coordinates -- the two corners are absolute VERTICES, not
        edge vectors, so the parallelogram is spanned by (v1 - origin) and
        (v2 - origin). The default is the a1-a2 face of the unit cell.

        Returns "points" (N, 2) in the plotting plane's own Cartesian frame
        (Bohr, src/plotpt2d.f90's ``vppc``), "grid" (n1, n2), "values" (N,)
        and "values_grid" (n2, n1) -- the first plotting vector's index runs
        fastest, so the reshape is (n2, n1) and NOT (n1, n2).
        """
        return self._run_plot(
            label,
            [_task("density_2d", 32)],
            2,
            _file("density_2d", "RHO2D.OUT"),
            plane=plane,
            grid=grid,
        )

    def get_potential_1d(self, component="coulomb", line=None, npoints=200,
                         label="potential_1d"):
        """Kohn-Sham potential along a line (task 41, src/potplot.f90).

        Task 41 writes BOTH components in one run -- VCL1D.OUT (the
        electrostatic potential, nuclear plus Hartree) and VXC1D.OUT (the
        exchange-correlation potential) -- and `component` ("coulomb" or
        "xc") only selects which is parsed, same convention as the existing
        get_potential(). Both share the vertex file VLINES.OUT.

        Same return keys as get_density_1d(); values in Hartree.
        """
        filename = {
            "coulomb": _file("potential_coulomb_1d", "VCL1D.OUT"),
            "xc": _file("potential_xc_1d", "VXC1D.OUT"),
        }.get(component)
        if filename is None:
            raise ValueError(
                f"unknown potential component '{component}'; use 'coulomb' or 'xc'"
            )
        return self._run_plot(
            label,
            [_task("potential_1d", 41)],
            1,
            filename,
            lines_filename=_file("potential_lines", "VLINES.OUT"),
            line=line,
            npoints=npoints,
        )

    def get_potential_2d(self, component="coulomb", plane=None, grid=(40, 40),
                         label="potential_2d"):
        """Kohn-Sham potential on a plane (task 42, src/potplot.f90 ->
        VCL2D.OUT and VXC2D.OUT). Same `component` convention as
        get_potential_1d(), same return keys as get_density_2d()."""
        filename = {
            "coulomb": _file("potential_coulomb_2d", "VCL2D.OUT"),
            "xc": _file("potential_xc_2d", "VXC2D.OUT"),
        }.get(component)
        if filename is None:
            raise ValueError(
                f"unknown potential component '{component}'; use 'coulomb' or 'xc'"
            )
        return self._run_plot(
            label,
            [_task("potential_2d", 42)],
            2,
            filename,
            plane=plane,
            grid=grid,
        )

    def get_elf_1d(self, line=None, npoints=200, label="elf_1d"):
        """Electron localization function along a line (task 51,
        src/elfplot.f90 -> ELF1D.OUT, ELFLINES.OUT).

        Same definition and same basis-set caveat as the existing
        get_elf(): the ELF depends on density gradients and is discontinuous
        at the muffin-tin boundary unless the cut-offs are raised, so a line
        crossing a sphere edge can show a step that is a basis artefact.
        Values are dimensionless and bounded to [0, 1]. Same return keys as
        get_density_1d()."""
        return self._run_plot(
            label,
            [_task("elf_1d", 51)],
            1,
            _file("elf_1d", "ELF1D.OUT"),
            lines_filename=_file("elf_lines", "ELFLINES.OUT"),
            line=line,
            npoints=npoints,
        )

    def get_elf_2d(self, plane=None, grid=(40, 40), label="elf_2d"):
        """Electron localization function on a plane (task 52,
        src/elfplot.f90 -> ELF2D.OUT). Same return keys as
        get_density_2d()."""
        return self._run_plot(
            label,
            [_task("elf_2d", 52)],
            2,
            _file("elf_2d", "ELF2D.OUT"),
            plane=plane,
            grid=grid,
        )

    # ------------------------------------------------------------------
    # wavefunctions and STM
    # ------------------------------------------------------------------

    def get_wavefunction(self, ik=1, ist=1, dim=3, line=None, npoints=200,
                         plane=None, box=None, grid=None, label=None):
        """Modulus squared of one Kohn-Sham wavefunction (tasks 61/62/63,
        src/wfplot.f90 -> WF1D.OUT + WFLINES.OUT, WF2D.OUT, WF3D.OUT).

        ``|psi_{ik,ist}(r)|^2`` for the state selected by the ``kstlist``
        block. Elk builds it by zeroing every occupation number except
        ``occsv(ist,ik) = 1/wkpt(ik)`` and then calling ``rhomagv``, i.e.
        it reuses the charge-density machinery with a one-state occupancy;
        symmetrisation is switched off (``nsymcrys = 1``) so the plot is the
        genuine single-state density, not its site-symmetrised average.
        Normalisation: the unit-cell integral is 1 (spinor-summed).

        `ik` indexes the GROUND STATE's k-point list as Elk stores it, so it
        depends on ``ngridk``, ``vkloff`` and on symmetry reduction; there
        is no way to ask for an arbitrary k-point here (use
        get_eigenstates() for that). `ist` is a second-variational state
        index.

        `dim` selects the line (1), plane (2) or box (3) form and picks
        which of `line`/`plane`/`box` and `grid` apply, exactly as in the
        density methods above. Returns the same keys as the corresponding
        get_density_* method.
        """
        tasks_by_dim = {1: 61, 2: 62, 3: 63}
        files_by_dim = {
            1: _file("wavefunction_1d", "WF1D.OUT"),
            2: _file("wavefunction_2d", "WF2D.OUT"),
            3: _file("wavefunction_3d", "WF3D.OUT"),
        }
        if dim not in tasks_by_dim:
            raise ValueError(f"dim must be 1, 2 or 3, got {dim}")
        return self._run_plot(
            label or f"wavefunction_{dim}d",
            [_task(f"wavefunction_{dim}d", tasks_by_dim[dim])],
            dim,
            files_by_dim[dim],
            lines_filename=_file("wavefunction_lines", "WFLINES.OUT"),
            line=line,
            npoints=npoints,
            plane=plane,
            box=box,
            grid=grid,
            extra_blocks={"kstlist": [(int(ik), int(ist))]},
        )

    def get_stm(self, height=0.25, plane=None, grid=(40, 40), swidth=None,
                ngridk=None, label="stm"):
        """Upstream Elk's spin-SUMMED STM image (task 162, src/wfplot.f90 ->
        STM2D.OUT).

        The Tersoff-Hamann constant-current picture (Tersoff and Hamann,
        PRB 31, 805 (1985)): the tunnel current tracks the sample's local
        density of states in the vacuum at the tip position, which Elk gets
        by replacing every occupation number with a smeared delta at the
        Fermi energy,

            occsv(ist,ik) = occmax w_k sdelta((E_F - e_{ist,k})/swidth)
                            / swidth,

        and then plotting the resulting charge density on the ``plot2d``
        plane. This is the spin-summed n(r, E_F) only; for the magnetic-tip
        projection ``n + P_T m . e_T``, an arbitrary bias, and the
        constant-current window, use get_spin_stm() (docs/design.md #30),
        which also fixes a ``wkpt`` double-counting present in this upstream
        branch.

        - `height`: tip height as a fractional coordinate along the third
          lattice vector, used only to build the default plane.
        - `plane`: the ``plot2d`` parallelogram ``[origin, v1, v2]`` in
          lattice coordinates, overriding `height`.
        - `swidth`: the energy-selection smearing width (Hartree) for this
          call only.
        - `ngridk`: k-mesh override for this call only; a vacuum LDOS at
          E_F wants a denser mesh than a total energy does.

        Returns the same keys as get_density_2d(); "values" is in
        electrons/Bohr^3 (the smeared-delta weight makes it an LDOS up to
        the width factor, hence the caveat above about the upstream
        normalisation).
        """
        if plane is None:
            plane = [(0, 0, height), (1, 0, height), (0, 1, height)]
        blocks = {}
        if swidth is not None:
            blocks["swidth"] = [float(swidth)]
        return self._run_plot(
            label,
            [_task("stm", 162)],
            2,
            _file("stm_2d", "STM2D.OUT"),
            plane=plane,
            grid=grid,
            extra_blocks=blocks,
            ngridk=ngridk,
        )

    def get_core_wavefunctions(self, label="core_wavefunctions"):
        """Radial core-state wavefunctions (task 65, src/wfcrplot.f90 ->
        WFCORE_Sss_Aaaaa.OUT).

        Elk treats the core states fully relativistically and solves them in
        the spherical part of the muffin-tin potential on each species'
        radial mesh (``gencore``), separately from the valence LAPW problem
        -- which is why core electrons are NOT among the ``nstsv`` valence
        bands the eigenstate methods index (a point docs/design.md #13
        records the hard way). This task writes those radial functions out.

        Returns a dict keyed by (species symbol, 1-based atom index), each
        value ``(r, u)`` with `r` the species radial mesh in Bohr and `u`
        shape (ncore, nr). Elk stores ``u(r) = r R(r)``, so the physical
        radial function is ``u/r`` and the normalisation is
        ``int |u|^2 dr = 1`` with no extra r^2 Jacobian. Only states flagged
        ``spcore`` in the species file appear, in species-file order.
        """
        subdir = self._run_resumed(label, [_task("core_wavefunctions", 65)])
        template = _template("core_wavefunction", "WFCORE_S{species:02d}_A{atom:04d}.OUT")
        result = {}
        for key, name in self._atom_filenames(template).items():
            result[key] = atomicstates.parse_wfcore(subdir / name)
        return result

    # ------------------------------------------------------------------
    # vector fields: magnetisation, B_xc, E field, m x B_xc, div B_xc
    # ------------------------------------------------------------------

    def _vector_plot(self, label, name, tasks_by_dim, files_by_dim, dim, nf=3, **kwargs):
        if dim not in tasks_by_dim:
            raise ValueError(f"dim must be 1, 2 or 3, got {dim}")
        return self._run_plot(
            label or f"{name}_{dim}d",
            [tasks_by_dim[dim]],
            dim,
            files_by_dim[dim],
            nf=nf,
            lines_filename=files_by_dim.get("lines"),
            **kwargs,
        )

    def get_magnetisation(self, dim=3, line=None, npoints=200, plane=None,
                          box=None, grid=None, label=None):
        """Magnetisation vector field m(r) (tasks 71/72/73,
        src/vecplot.f90 -> MAG1D/MAG2D/MAG3D.OUT).

        m(r) is the vector part of the spin density matrix,

            rho_ab(r) = (1/2) [ n(r) delta_ab + sigma . m(r) ]_ab,

        so ``|m|`` is ``n_up - n_dn`` in the collinear case -- TWICE the
        spin expectation value ``<S>`` of docs/design.md #17, the same
        factor-of-two convention #30 records for the spin-polarised STM.
        Units: electrons/Bohr^3.

        For a COLLINEAR run (``ndmag == 1``) Elk stores a single component
        and vecplot writes it as (0, 0, m_z); a non-collinear run
        (``spinorb=True``, or an x/y ``bfcmt``/``bfieldc`` anywhere) stores
        all three.

        The 2D form is written by ``plot2d(tproj=.true., ...)``, which
        PROJECTS the 3D vector onto the plotting plane's own two axes before
        writing -- so at dim=2 the three columns are the in-plane
        components in the ``vppc`` frame plus the out-of-plane remainder,
        not Cartesian x/y/z. The 1D and 3D forms are unprojected Cartesian.

        Returns the usual per-dimension keys with "values" shape (N, 3) and
        "values_grid" shape (3, ...) -- component axis first.
        """
        self._require_spinpol("get_magnetisation()")
        return self._vector_plot(
            label,
            "magnetisation",
            {1: _task("magnetisation_1d", 71), 2: _task("magnetisation_2d", 72),
             3: _task("magnetisation_3d", 73)},
            {1: _file("magnetisation_1d", "MAG1D.OUT"),
             2: _file("magnetisation_2d", "MAG2D.OUT"),
             3: _file("magnetisation_3d", "MAG3D.OUT"),
             "lines": _file("magnetisation_lines", "MAGLINES.OUT")},
            dim, line=line, npoints=npoints, plane=plane, box=box, grid=grid,
        )

    def get_bxc(self, dim=3, line=None, npoints=200, plane=None, box=None,
                grid=None, label=None):
        """Exchange-correlation magnetic field B_xc(r) (tasks 81/82/83,
        src/vecplot.f90 -> BXC1D/BXC2D/BXC3D.OUT).

        The functional derivative ``delta E_xc / delta m(r)``, i.e. the
        field that appears in the Kohn-Sham equation alongside the scalar
        v_xc and that produces the spin splitting; in Hartree (atomic
        units, so a Bohr magneton is 1/2). Within LSDA it is antiparallel
        to m(r) wherever the local functional is, which is why
        get_bxc_divergence() is a meaningful probe of non-collinearity.

        Same `dim`/return convention -- and the same plane-projection at
        dim=2 -- as get_magnetisation().
        """
        self._require_spinpol("get_bxc()")
        return self._vector_plot(
            label,
            "bxc",
            {1: _task("bxc_1d", 81), 2: _task("bxc_2d", 82), 3: _task("bxc_3d", 83)},
            {1: _file("bxc_1d", "BXC1D.OUT"), 2: _file("bxc_2d", "BXC2D.OUT"),
             3: _file("bxc_3d", "BXC3D.OUT"),
             "lines": _file("bxc_lines", "BXCLINES.OUT")},
            dim, line=line, npoints=npoints, plane=plane, box=box, grid=grid,
        )

    def get_electric_field(self, dim=3, line=None, npoints=200, plane=None,
                           box=None, grid=None, label=None):
        """Electric field E(r) = -grad v_C(r) (tasks 141/142/143,
        src/vecplot.f90 -> EF1D/EF2D/EF3D.OUT).

        The gradient of the ELECTROSTATIC (nuclear plus Hartree) potential
        only -- the exchange-correlation part is excluded, so this is the
        classical field of the charge distribution, not the full Kohn-Sham
        force field. Elk computes it with ``gradrf(vclmt, vclir, ...)`` and
        negates. Units: Hartree/Bohr (atomic units of field).

        It diverges as ``Z/r^2`` at every nucleus, so a plotting point
        landing on an atom dominates the range. Needs no spin polarisation.
        Same `dim`/return convention as get_magnetisation(), including the
        dim=2 plane projection.
        """
        return self._vector_plot(
            label,
            "electric_field",
            {1: _task("electric_field_1d", 141), 2: _task("electric_field_2d", 142),
             3: _task("electric_field_3d", 143)},
            {1: _file("electric_field_1d", "EF1D.OUT"),
             2: _file("electric_field_2d", "EF2D.OUT"),
             3: _file("electric_field_3d", "EF3D.OUT"),
             "lines": _file("electric_field_lines", "EFLINES.OUT")},
            dim, line=line, npoints=npoints, plane=plane, box=box, grid=grid,
        )

    def get_magnetic_torque(self, dim=3, line=None, npoints=200, plane=None,
                            box=None, grid=None, label=None):
        """The local magnetic torque density m(r) x B_xc(r) (tasks
        151/152/153, src/vecplot.f90 -> MCBXC1D/MCBXC2D/MCBXC3D.OUT).

        The torque the exchange-correlation field exerts on the local
        magnetisation. Within a LOCAL spin-density functional B_xc is
        strictly (anti)parallel to m pointwise, so this cross product would
        vanish identically; it is nonzero only where the functional is
        non-local in the spin direction or where the muffin-tin/interstitial
        representation of the two fields differs -- which makes it a direct
        map of where the calculation's non-collinearity actually lives. Its
        cell integral is the total spin torque, zero at self-consistency
        for a converged non-collinear state.

        This is the torque DENSITY. Its cell integral -- the total spin
        torque, the zero-torque-theorem diagnostic -- is a separate Elk
        task (160) exposed as :meth:`get_total_magnetic_torque`.

        Requires a NON-COLLINEAR run: ``src/vecplot.f90`` refuses
        ``ndmag == 1`` outright, since a collinear m x B_xc is zero by
        construction. Same `dim`/return convention as get_magnetisation().
        """
        if self._ndmag() != 3:
            raise ValueError(
                "get_magnetic_torque() requires a NON-COLLINEAR calculation "
                "(ndmag = 3): set spinorb=True, or give some atom an x/y bfcmt "
                "component. src/vecplot.f90 refuses the collinear case because "
                "m(r) x B_xc(r) is then identically zero"
            )
        return self._vector_plot(
            label,
            "magnetic_torque",
            {1: _task("magnetic_torque_1d", 151), 2: _task("magnetic_torque_2d", 152),
             3: _task("magnetic_torque_3d", 153)},
            {1: _file("magnetic_torque_1d", "MCBXC1D.OUT"),
             2: _file("magnetic_torque_2d", "MCBXC2D.OUT"),
             3: _file("magnetic_torque_3d", "MCBXC3D.OUT"),
             "lines": _file("magnetic_torque_lines", "MCBXCLINES.OUT")},
            dim, line=line, npoints=npoints, plane=plane, box=box, grid=grid,
        )

    def get_bxc_divergence(self, dim=3, line=None, npoints=200, plane=None,
                           box=None, grid=None, label=None):
        """Divergence of the exchange-correlation magnetic field,
        ``div B_xc(r)`` (tasks 91/92/93, src/dbxcplot.f90 ->
        DBXC1D/DBXC2D/DBXC3D.OUT).

        A SCALAR field (one column, unlike the vector plots above). This is
        the quantity Elk's ``nosource`` option drives to zero: a physical
        magnetic field is source-free, but the exchange-correlation field of
        a local spin-density functional is not, and its divergence measures
        that unphysical magnetic-monopole density. Units: Hartree/Bohr.

        Requires spin polarisation. Same `dim`/return convention as
        get_density_*, with scalar "values" and "values_grid".
        """
        self._require_spinpol("get_bxc_divergence()")
        return self._vector_plot(
            label,
            "bxc_divergence",
            {1: _task("bxc_divergence_1d", 91), 2: _task("bxc_divergence_2d", 92),
             3: _task("bxc_divergence_3d", 93)},
            {1: _file("bxc_divergence_1d", "DBXC1D.OUT"),
             2: _file("bxc_divergence_2d", "DBXC2D.OUT"),
             3: _file("bxc_divergence_3d", "DBXC3D.OUT"),
             "lines": _file("bxc_divergence_lines", "DBXCLINES.OUT")},
            dim, nf=1, line=line, npoints=npoints, plane=plane, box=box, grid=grid,
        )

    def get_paramagnetic_current(self, dim=3, line=None, npoints=200, plane=None,
                                 box=None, grid=None, label=None):
        """Paramagnetic current density j_p(r) (tasks 371/372/373,
        src/jprplot.f90 -> JPR1D/JPR2D/JPR3D.OUT).

        The occupied-state expectation value

            j_p(r) = -(1/2) sum_i f_i Im[ psi_i^*(r) grad psi_i(r)
                                          - psi_i(r) grad psi_i^*(r) ],

        i.e. the current WITHOUT the diamagnetic ``-n(r) A(r)/c`` term, so
        it is gauge dependent and vanishes for a time-reversal-symmetric
        ground state with no applied vector potential. It is the piece Elk
        needs for the current-density response (``tjr``) and for the static
        density of task 471. Units: electrons/(Bohr^2 x atomic unit of
        time).

        Unlike the other vector plots this one recomputes wavefunctions
        (``genjpr`` after ``genapwfr``/``genlofr``/``readoccsv``), so it
        costs a pass over the ground-state k-mesh rather than just reading
        STATE.OUT. Same `dim`/return convention as get_magnetisation(),
        including the dim=2 plane projection.
        """
        return self._vector_plot(
            label,
            "paramagnetic_current",
            {1: _task("paramagnetic_current_1d", 371),
             2: _task("paramagnetic_current_2d", 372),
             3: _task("paramagnetic_current_3d", 373)},
            {1: _file("paramagnetic_current_1d", "JPR1D.OUT"),
             2: _file("paramagnetic_current_2d", "JPR2D.OUT"),
             3: _file("paramagnetic_current_3d", "JPR3D.OUT"),
             "lines": _file("paramagnetic_current_lines", "JPRLINES.OUT")},
            dim, line=line, npoints=npoints, plane=plane, box=box, grid=grid,
        )

    def get_wxc(self, dim=3, line=None, npoints=200, plane=None, box=None,
                grid=None, label=None):
        """Meta-GGA exchange-correlation potential W_xc(r) (tasks
        341/342/343, src/wxcplot.f90 -> WXC1D/WXC2D/WXC3D.OUT).

        A meta-GGA's energy density depends on the kinetic-energy density
        tau(r) as well as n and grad n, so its functional derivative has an
        extra piece ``W_xc = delta E_xc / delta tau`` acting on the kinetic
        term of the Kohn-Sham Hamiltonian, alongside the ordinary
        multiplicative v_xc. This plots that piece; it is dimensionless
        (tau has the dimensions of an energy density, so W_xc is a
        dimensionless weight on the kinetic operator).

        BUILD REQUIREMENT: ``src/wxcplot.f90`` refuses any run whose
        ``xcgrad`` is not one of 3, 4, 5, 6, and in Elk 11.0.2 those values
        are set ONLY by ``xcdata_libxc`` (src/modxcifc.f90 line 494) -- so a
        meta-GGA needs a libxc-linked build, which this project's
        ``build-config/make.inc`` does not provide (it uses
        ``libxcifc_stub.f90``). The wrapper is here so it works the moment
        libxc is linked; with the stub build Elk will stop with
        "meta-GGA not in use". Scalar field, same return keys as
        get_bxc_divergence().
        """
        return self._vector_plot(
            label,
            "wxc",
            {1: _task("wxc_1d", 341), 2: _task("wxc_2d", 342), 3: _task("wxc_3d", 343)},
            {1: _file("wxc_1d", "WXC1D.OUT"), 2: _file("wxc_2d", "WXC2D.OUT"),
             3: _file("wxc_3d", "WXC3D.OUT"),
             "lines": _file("wxc_lines", "WLINES.OUT")},
            dim, nf=1, line=line, npoints=npoints, plane=plane, box=box, grid=grid,
        )

    def get_static_density(self, line=None, npoints=200, label="static_density"):
        """Static (screened) charge density along a line (task 471,
        src/rhosplot.f90 -> RHOS1D.OUT, RHOSLINES.OUT).

        The density that responds to a STATIC vector potential, used by
        Elk's real-time TDDFT machinery to separate the material's static
        screening from its dynamical response. ``src/rhostatic.f90`` forms

            rho_s^{(i)}(r) = n(r) + (c/A_0) [ j_p^{(0)}(r) - j_p^{(i)}(r) ],

        i.e. the ground-state density corrected by the change in the
        PARAMAGNETIC current when a constant A-field of magnitude
        ``A_0 = (3/4)c`` is applied along Cartesian direction i. The core
        density is then subtracted, so what is plotted is the valence
        static density.

        The three written columns are therefore the three APPLIED FIELD
        DIRECTIONS x, y, z -- not the components of a vector field. 1D only:
        upstream provides no 2D or 3D member of this task.

        COST: ``rhostatic`` runs FOUR ground-state calculations (one at zero
        field plus one per direction), each a single SCF iteration resumed
        from STATE.OUT, so this is much more expensive than the other
        plotting tasks. Returns the usual 1D keys with "values" shape
        (N, 3).
        """
        return self._run_plot(
            label,
            [_task("static_density_1d", 471)],
            1,
            _file("static_density_1d", "RHOS1D.OUT"),
            nf=3,
            lines_filename=_file("static_density_lines", "RHOSLINES.OUT"),
            line=line,
            npoints=npoints,
        )

    # ------------------------------------------------------------------
    # band structure variants
    # ------------------------------------------------------------------

    def get_band_character(self, kind="l", vertices=None, kpath=None, npoints=200,
                           lmaxdb=None, lmirep=None, sqaxis=None, label=None):
        """Band structure resolved by atom and by angular momentum, spin or
        magnetic moment (tasks 21/22/23/24, src/bandstr.f90 ->
        BAND_Sss_Aaaaa.OUT).

        Elk diagonalises along the same ``plot1d`` path get_bands() uses,
        then for every atom and every second-variational state forms the
        muffin-tin density matrix with ``gendmatk`` -- the same
        ``wfmtsv``/``wr2cmt`` expansion docs/design.md sections 16/18/19
        build the atom, l-channel and angular-momentum operators from -- and
        writes its diagonal as the "band character", the weight of that band
        inside that atom's sphere resolved by:

        - ``kind="l"`` (task 21): the total muffin-tin weight of the atom
          plus its s, p, d, f, ... decomposition up to `lmaxdb`.
        - ``kind="lm"`` (task 22): every (l, m) channel separately, in Elk's
          lm ordering. With `lmirep` (Elk's default, True) the (l, m) basis
          is first rotated into the irreducible representations of the site
          symmetry group, and ELMIREP.OUT is written alongside.
        - ``kind="spin"`` (task 23): the spin-up and spin-down weights,
          quantised along `sqaxis`. Requires spin polarisation.
        - ``kind="moment"`` (task 24): the ``ndmag`` components of the local
          moment carried by that band on that atom. Requires spin
          polarisation.

        Because the weight is a muffin-tin integral, the characters of all
        atoms sum to LESS than one -- the interstitial remainder is not
        attributed to any atom. This is the same partition
        docs/design.md #16 makes exact, and this task is the independent
        Fortran code path against which #16's and #18's operators are
        cross-checked.

        Pass exactly one of `vertices` (lattice coordinates) or `kpath` (a
        symbolic path string), as for get_bands().

        Returns a dict keyed by (species symbol, 1-based atom index), each
        value the dict character.parse_band_character() returns
        ("distances", "energies", "characters", and for kind="l" also
        "total" and "l"), plus the key "vertices" holding the
        high-symmetry-point distances from BANDLINES.OUT.
        """
        if kind not in BAND_CHARACTER_KINDS:
            raise ValueError(
                f"unknown band-character kind '{kind}'; use one of "
                f"{sorted(BAND_CHARACTER_KINDS)}"
            )
        if kind in ("spin", "moment"):
            self._require_spinpol(f"get_band_character(kind='{kind}')")
        vertices = self._resolve_vertices(vertices, kpath)
        blocks = {
            "plot1d": [(len(vertices), npoints)] + [tuple(v) for v in vertices],
        }
        if lmaxdb is not None:
            blocks["lmaxdb"] = [int(lmaxdb)]
        if lmirep is not None:
            blocks["lmirep"] = [bool(lmirep)]
        if sqaxis is not None:
            blocks["sqaxis"] = [tuple(float(x) for x in sqaxis)]
        subdir = self._run_resumed(
            label or f"band_character_{kind}",
            [_task(f"band_character_{kind}", BAND_CHARACTER_KINDS[kind])],
            blocks,
        )
        template = _template("band_character", "BAND_S{species:02d}_A{atom:04d}.OUT")
        result = {}
        for key, name in self._atom_filenames(template).items():
            result[key] = character.parse_band_character(subdir / name, kind=kind)
        bandlines = subdir / _file("bandlines", "BANDLINES.OUT")
        if bandlines.exists():
            result["vertices"] = plots.parse_plot_lines(bandlines)
        return result

    def get_partial_dos(self, ngridk=None, nwplot=500, wplot=(-0.5, 0.5),
                        dosmsum=True, dosssum=False, lmaxdb=None, swidth=None,
                        label="partial_dos"):
        """Partial (atom- and l- or lm-resolved) and interstitial density of
        states (task 10, src/writedos.f90 -> src/dos.f90 -> TDOS.OUT,
        PDOS_Sss_Aaaaa.OUT, IDOS.OUT).

        Task 10 already writes these alongside the total DOS that
        get_dos() parses; this method only asks for them and reads them
        back. Same construction as the band character above -- the
        ``gendmatk`` muffin-tin density matrix, here Brillouin-zone
        integrated by ``brzint`` onto the ``wplot`` energy grid instead of
        being followed along a path.

        - `dosmsum` (default True here, False upstream): sum each l channel
          over m, giving s/p/d/f columns instead of every (l, m). The
          upstream default writes ``(lmaxdb+1)^2`` columns per atom, which
          is rarely what is wanted.
        - `dosssum`: sum over spin. Without it a spin-polarised run writes
          both channels and Elk NEGATES the spin-down one (``sps(2) = -1``,
          a plotting convention); the parser undoes that sign.
        - `wplot`/`nwplot`: the energy window (Hartree, relative to E_F) and
          number of points.
        - `ngridk`: k-mesh override for this call only -- a DOS wants a
          denser mesh than a total energy does (sampling-only parameter,
          docs/design.md #4).

        Returns a dict:

        - "energies" (nw,): Hartree, Fermi energy at zero.
        - "total" (nspin, nw): TDOS.OUT.
        - "interstitial" (nspin, nw): IDOS.OUT -- the TOTAL MINUS every
          muffin-tin channel, since dos.f90 subtracts each partial
          contribution from the running total as it goes. Not a second copy
          of the total.
        - "partial": dict keyed by (species symbol, 1-based atom index),
          each value (nspin, nchannels, nw).
        - "nspin", "channels": the spin count actually written and either
          "l" or "lm" depending on `dosmsum`.
        """
        nspin = 1 if (dosssum or not (self.spinpol or self.spinorb)) else 2
        blocks = {
            "wplot": [(int(nwplot), 100, 1), (float(wplot[0]), float(wplot[1]))],
            "tpdos": [True],
            "dosmsum": [bool(dosmsum)],
            "dosssum": [bool(dosssum)],
        }
        if lmaxdb is not None:
            blocks["lmaxdb"] = [int(lmaxdb)]
        if swidth is not None:
            blocks["swidth"] = [float(swidth)]
        subdir = self._run_resumed(
            label,
            [_task("dos", 10)],
            blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        energies, total = character.parse_dos_blocks(subdir / _file("tdos", "TDOS.OUT"))
        _e, interstitial = character.parse_dos_blocks(subdir / _file("idos", "IDOS.OUT"))
        template = _template("partial_dos", "PDOS_S{species:02d}_A{atom:04d}.OUT")
        partial = {}
        for key, name in self._atom_filenames(template).items():
            _e, dos = character.parse_partial_dos(subdir / name, nspin=nspin)
            partial[key] = dos
        if nspin == 2:
            total = total.copy()
            total[1] = -total[1]
            interstitial = interstitial.copy()
            interstitial[1] = -interstitial[1]
        return {
            "energies": energies,
            "total": total,
            "interstitial": interstitial,
            "partial": partial,
            "nspin": nspin,
            "channels": "l" if dosmsum else "lm",
        }

    # ------------------------------------------------------------------
    # Fermi surfaces and nesting
    # ------------------------------------------------------------------

    def get_fermi_surface(self, kind="product", grid=(20, 20, 20), box=None,
                          swidth=None, label="fermi_surface"):
        """3D Fermi surface data on a reciprocal-space grid (tasks
        100/101/103/104, src/fermisurf.f90).

        Four representations of the same information, all on the grid the
        ``plot3d`` block defines:

        - ``kind="product"`` (task 100): one scalar per point,
          ``prod_n (e_n(k) - E_F)`` over only the bands that cross E_F. Its
          ZERO ISOSURFACE is the Fermi surface, all sheets at once -- the
          cheapest thing to hand to an isosurface renderer.
        - ``kind="bands"`` (task 101): one column per crossing band,
          ``e_n(k) - E_F``, so each sheet can be drawn separately (and
          coloured by band index).
        - ``kind="delta"`` (task 103): one scalar per point,
          ``sum_n sdelta((e_n(k)-E_F)/swidth)/swidth`` over ALL bands -- a
          smeared spectral density at the Fermi level rather than a sharp
          surface, so a volume rendering shows how "thick" the surface is.
        - ``kind="band_deltas"`` (task 104): the same smeared delta,
          resolved per band.

        THE ``ngridk`` TRAP. ``src/init1.f90`` lines 130-134 replace
        ``ngridk`` with ``np3d`` and (for every kind but the bxsf one) the
        k-point box with the ``plot3d`` box, for exactly these tasks. So
        `grid` sets the k-mesh as well as the plotting grid -- the
        calculation's own ``ngridk`` is ignored -- and there is no way for
        the two to disagree. That also fixes the cost: a (20, 20, 20) grid
        means diagonalising at 8000 k-points before symmetry reduction.

        `box` is the ``plot3d`` parallelepiped in RECIPROCAL lattice
        coordinates, default the reciprocal unit cell; Elk's own examples
        use a 2x2x2 box to show several zones at once.

        With COLLINEAR magnetism (``ndmag == 1``) fermisurf forces
        ``reducek = 0`` and writes two files, splitting the second-
        variational states at ``nstfv``; otherwise one.

        Returns a dict of the parsed files keyed "total", or "up" and "dn"
        for the collinear-magnetic case -- see
        parsers.fermisurface.parse_fermisurf() for each value's keys --
        plus "paths", the files on disk.
        """
        if kind not in FERMI_SURFACE_KINDS:
            raise ValueError(
                f"unknown Fermi-surface kind '{kind}'; use one of "
                f"{sorted(FERMI_SURFACE_KINDS)}"
            )
        blocks = {"plot3d": self._plot3d_lines(box, grid)}
        if swidth is not None:
            blocks["swidth"] = [float(swidth)]
        subdir = self._run_resumed(
            label, [_task(f"fermi_surface_{kind}", FERMI_SURFACE_KINDS[kind])], blocks
        )
        candidates = {
            "total": _file("fermisurf", "FERMISURF.OUT"),
            "up": _file("fermisurf_up", "FERMISURF_UP.OUT"),
            "dn": _file("fermisurf_dn", "FERMISURF_DN.OUT"),
        }
        result = {"paths": {}}
        for key, name in candidates.items():
            path = subdir / name
            if path.exists():
                result[key] = fermisurface.parse_fermisurf(path)
                result["paths"][key] = path
        if len(result["paths"]) == 0:
            raise RuntimeError(
                f"task {FERMI_SURFACE_KINDS[kind]} wrote no FERMISURF file in {subdir}"
            )
        return result

    def get_fermi_surface_bxsf(self, grid=(20, 20, 20), copy_to=None,
                               label="fermi_surface_bxsf"):
        """Fermi surface in XCrySDen's ``.bxsf`` band-grid format (task 102,
        src/fermisurfbxsf.f90 -> FERMISURF.bxsf, or FERMISURF_UP.bxsf /
        FERMISURF_DN.bxsf for collinear magnetism).

        The format the rest of the community's Fermi-surface tooling reads
        (``xcrysden --bxsf FERMISURF.bxsf``, and FermiSurfer via a
        converter): a header giving the reciprocal lattice vectors and a
        band grid of ``ngridk + 1`` points per direction -- the extra plane
        repeats the first, which is what a periodic isosurface needs -- with
        the Fermi energy shifted to exactly zero. Only the bands that cross
        E_F are written.

        As for get_fermi_surface(), ``src/init1.f90`` sets ``ngridk`` from
        the ``plot3d`` block's ``np3d``, so `grid` is the k-mesh; unlike the
        other kinds, task 102 keeps the STANDARD reciprocal unit cell as the
        k-box and forces ``vkloff = 0``, so no `box` argument is offered
        (the plot3d corners would be ignored).

        `copy_to` optionally copies the ``.bxsf`` files to a directory of
        your choice, since the run subdirectory is wiped by the next call
        that reuses the same `label`.

        Returns a dict keyed "total" (or "up"/"dn"), each value the dict
        parsers.fermisurface.parse_bxsf() returns, plus "paths".
        """
        blocks = {"plot3d": self._plot3d_lines(None, grid)}
        subdir = self._run_resumed(label, [_task("fermi_surface_bxsf", 102)], blocks)
        candidates = {
            "total": _file("fermisurf_bxsf", "FERMISURF.bxsf"),
            "up": _file("fermisurf_bxsf_up", "FERMISURF_UP.bxsf"),
            "dn": _file("fermisurf_bxsf_dn", "FERMISURF_DN.bxsf"),
        }
        result = {"paths": {}}
        for key, name in candidates.items():
            path = subdir / name
            if not path.exists():
                continue
            if copy_to is not None:
                destination = Path(copy_to)
                destination.mkdir(parents=True, exist_ok=True)
                path = Path(shutil.copy(path, destination / name))
            result[key] = fermisurface.parse_bxsf(path)
            result["paths"][key] = path
        if len(result["paths"]) == 0:
            raise RuntimeError(f"task 102 wrote no .bxsf file in {subdir}")
        return result

    def get_nesting(self, ngridk=None, swidth=None, label="nesting"):
        """Fermi-surface nesting function N(q) (task 105, src/nesting.f90 ->
        NEST3D.OUT, NESTING.OUT).

        The joint density of Fermi-level states connected by q,

            N(q) = (occmax Omega_BZ / N_k) sum_k
                   [ sum_n delta(E_F - e_n(k)) ]
                   [ sum_m delta(E_F - e_m(k+q)) ],

        with the delta functions smeared by ``swidth``. A peak at some q
        means large parallel patches of Fermi surface are separated by that
        vector, the geometric precondition for a charge- or spin-density-
        wave instability (and for a phonon Kohn anomaly) at that q. It is
        identically zero for an insulator, whose Fermi-level DOS vanishes.

        ``src/init2.f90`` FORCES ``ngridq = ngridk`` for this task, so the
        q-grid is the k-grid and there is nothing separate to set; pass
        `ngridk` to control the resolution of both (sampling only -- see
        docs/design.md #4). Because the sum is over pairs of mesh points the
        cost grows as the SQUARE of the mesh size.

        Returns the dict parsers.fermisurface.parse_nesting() builds:
        "points" (N, 3) Cartesian Bohr^-1, "grid", "values" (N,) and
        "total", the Brillouin-zone integral per unit volume.
        """
        blocks = {}
        if swidth is not None:
            blocks["swidth"] = [float(swidth)]
        subdir = self._run_resumed(
            label,
            [_task("nesting", 105)],
            blocks or None,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        return fermisurface.parse_nesting(
            subdir / _file("nest_3d", "NEST3D.OUT"),
            subdir / _file("nesting", "NESTING.OUT"),
        )

    # ------------------------------------------------------------------
    # expectation values, spectra and atomic data
    # ------------------------------------------------------------------

    def get_lsj(self, label="lsj"):
        """Muffin-tin L, S and J expectation values per atom (task 15,
        src/writelsj.f90 -> LSJ.OUT).

        The occupation-weighted traces

            <L_a> = Tr[ L_a rho ],   <S_a> = Tr[ S_a rho ],   J = L + S,

        over the muffin-tin density matrix ``rho`` that ``gendmat`` builds
        by summing ``occsv``-weighted ``wfmtsv`` expansions over the whole
        k-mesh -- the ground-state, Brillouin-zone-integrated counterpart of
        the per-k, per-band operators of docs/design.md sections 17 and 19,
        and computed through the same ``dmatls``/``lopzflm`` ladder-operator
        code #19 reuses.

        Muffin-tin only: the interstitial contribution is not included, so
        these are on-site atomic moments, not the cell totals. In a
        non-magnetic, non-spin-orbit crystal all three vanish by symmetry;
        the interesting case is a magnetic 3d or 4f site under
        ``spinorb=True``, where <L> measures the unquenched orbital moment
        and the sign of <L> . <S> follows Hund's third rule (antiparallel
        below half filling, parallel above).

        Returns the list parsers.atomicstates.parse_lsj() builds: one dict
        per atom with "species", "symbol", "atom" and the Cartesian vectors
        "L", "S", "J" in units of hbar.
        """
        subdir = self._run_resumed(label, [_task("lsj", 15)])
        return atomicstates.parse_lsj(subdir / _file("lsj", "LSJ.OUT"))

    def get_lsj_states(self, kstlist=((1, 1),), label="lsj_states"):
        """Muffin-tin L, S and J for individual (k-point, band) states
        (task 16, src/writelsj.f90 -> LSJ_KST.OUT).

        The same operators as get_lsj(), but with the occupancies replaced
        by a delta on one (k, state) pair at a time -- and with
        symmetrisation switched off (``nsymcrys = 1``,
        ``eqatoms = .false.``), so symmetry-equivalent atoms are reported
        separately rather than averaged. This is what makes it a per-STATE
        quantity comparable with the band character of
        get_band_character(kind="spin"), rather than a ground-state average.

        `kstlist` is a sequence of (ik, ist) pairs, both 1-based: `ik`
        indexes the ground state's own k-point list (so it depends on
        ``ngridk``, ``vkloff`` and symmetry reduction) and `ist` a
        second-variational band. Unlike get_wavefunction(), which uses only
        the first pair, task 16 loops over the whole list.

        Returns the list parsers.atomicstates.parse_lsj_kst() builds: one
        dict per (k-point, state, atom) with "ik", "k" (lattice
        coordinates), "ist", "species", "symbol", "atom" and "L", "S", "J".
        """
        pairs = [(int(ik), int(ist)) for ik, ist in kstlist]
        if not pairs:
            raise ValueError("kstlist must contain at least one (ik, ist) pair")
        subdir = self._run_resumed(
            label, [_task("lsj_states", 16)], {"kstlist": pairs}
        )
        return atomicstates.parse_lsj_kst(subdir / _file("lsj_kst", "LSJ_KST.OUT"))

    def get_smearing_functions(self, nwplot=500, swidth=None, stype=None,
                               label="smearing"):
        """The smooth Dirac delta and Heaviside functions Elk uses for
        occupation numbers (task 14, src/writesf.f90 -> SDELTA.OUT,
        STHETA.OUT).

        Every Fermi-level quantity in this bucket -- the DOS, the smeared
        Fermi surface (kinds "delta"/"band_deltas"), the nesting function,
        the STM image -- weights states by ``sdelta((E_F - e)/swidth)`` or
        occupies them by ``stheta``, so this task plots the very kernel
        those use, on a window of +-10 ``swidth``. ``stype`` selects the
        broadening scheme (0 Gaussian, 1 Methfessel-Paxton order 1, 2
        Methfessel-Paxton order 2, 3 Fermi-Dirac -- Elk's default, and the
        one docs/design.md #31 matches its on-shell amplitude to); the
        Methfessel-Paxton kernels go NEGATIVE in their tails, which is why a
        smeared DOS built from them can dip below zero.

        No ground state is needed at all -- ``writesf`` reads only
        ``swidth``, ``stype`` and ``nwplot`` -- but this still runs through
        the usual resumed subdirectory for uniformity.

        Returns a dict with "energy" (nw,) in Hartree, "delta" (nw,) in
        1/Hartree and "theta" (nw,) dimensionless.
        """
        blocks = {"wplot": [(int(nwplot), 100, 1), (-0.5, 0.5)]}
        if swidth is not None:
            blocks["swidth"] = [float(swidth)]
        if stype is not None:
            blocks["stype"] = [int(stype)]
        subdir = self._run_resumed(label, [_task("smearing_functions", 14)], blocks)
        energy, delta = atomicstates.parse_two_column(
            subdir / _file("sdelta", "SDELTA.OUT")
        )
        _e, theta = atomicstates.parse_two_column(
            subdir / _file("stheta", "STHETA.OUT")
        )
        return {"energy": energy, "delta": delta, "theta": theta}

    def get_atomic_eigenvalues(self, label="atomic_eigenvalues"):
        """Free-atom Kohn-Sham-Dirac eigenvalues for every species
        (task 150, src/writeevsp.f90 -> EVALSP.OUT).

        Elk solves the fully relativistic radial Dirac-Kohn-Sham equation
        for each isolated species in ``init0``, both to build the starting
        density and to decide which states are treated as core. This task
        writes those atomic eigenvalues out; they are the reference against
        which "is this state core or valence?" and "how deep is the semicore
        block?" are answered -- questions docs/design.md #13 records
        stumbling over when picking band windows.

        Each state carries Elk's ``ksp = |kappa| = j + 1/2`` (a magnitude,
        not the signed Dirac quantum number -- ``src/readspecies.f90``
        refuses ``ksp < 1`` and the shell degeneracy is ``2*ksp``), so an
        l > 0 shell appears TWICE, as k = l for j = l - 1/2 and k = l + 1
        for j = l + 1/2, and the splitting between the pair is the
        free-atom spin-orbit splitting -- a useful sanity check on the
        scale ``soc_scale`` is multiplying (docs/design.md #12).

        Needs no ground state and no k-mesh: ``writeevsp`` calls only
        ``init0``. Returns the list parsers.atomicstates.parse_evalsp()
        builds: one dict per species with "species", "symbol" and "states",
        each state a dict of "n", "l", "k" (kappa) and "energy" in Hartree.
        """
        subdir = self._run_resumed(label, [_task("atomic_eigenvalues", 150)])
        return atomicstates.parse_evalsp(subdir / _file("evalsp", "EVALSP.OUT"))

    def get_elnes(self, q, wplot=(-0.5, 0.5), nwplot=500,
                  emax=None, swidth=None, ngridk=None, label="elnes"):
        """Electron energy loss near-edge structure (task 140,
        src/elnes.f90 -> ELNES.OUT).

        The double differential scattering cross-section for an electron
        that transfers momentum q and energy w to the crystal,

            d^2 sigma / (dOmega dw) ~ (1/q^4) sum_{k,i,j}
                |<i, k+q| e^{iq.r} |j, k>|^2 f_j (1 - f_i)
                delta(w - e_i(k+q) + e_j(k)),

        i.e. the dynamic structure factor in the first Born approximation,
        Brillouin-zone integrated by ``brzint``. Elk builds the matrix
        elements with ``genexpmat``, the full ``e^{iq.r}`` operator rather
        than its dipole limit, so this is valid at finite q where the
        dipole selection rules break down.

        **q = 0 is refused, and not because of the 1/q^4 factor.** Elk does
        drop the Rutherford factor there (``elnes.f90``: ``if (q > epslat)``),
        but the matrix element collapses first:
        ``genexpmat.f90:30-38`` tests the global ``vecql`` and returns the
        IDENTITY outright, which is just the statement that
        :math:`\\langle i,{\\bf k}|j,{\\bf k}\\rangle=\\delta_{ij}`.
        ``elnes.f90`` then builds
        :math:`f_{ij}=|M_{ij}|^2 f_j(f_{\\max}-f_i)`, which for
        :math:`i=j` is :math:`f_i(f_{\\max}-f_i)` -- zero for every fully
        occupied and every empty state -- while the energy transfer
        :math:`\\varepsilon_i({\\bf k})-\\varepsilon_i({\\bf k})` is zero
        as well. So the cross-section is identically zero at q = 0, and no
        window contains an edge. Measured on fcc Al: exactly 0.0 at every one
        of 100 grid points with the default ``emaxelnes``, and, once `emax`
        is raised past :math:`E_F` so that the metal's PARTIALLY occupied
        states are admitted, a spike at zero energy loss and nothing else.
        Use get_dielectric_function() for an optical spectrum; q -> 0 here is
        not its limit, it is a different (and empty) quantity.

        - `q`: the momentum transfer in LATTICE (fractional reciprocal)
          coordinates, required and nonzero. It must be COMMENSURATE with
          the k-mesh -- ``elnes.f90`` checks ``ngridk * q`` is integral and
          stops otherwise -- because the k+q state has to be another mesh
          point.
        - `emax`: only initial states below this energy contribute
          (Elk's ``emaxelnes``, default -1.2 Ha), which is how the CORE-loss
          edge is selected: raise it towards E_F to include valence
          excitations too.
        - `wplot`/`nwplot`: the energy-loss window (Hartree) and grid.
        - `ngridk`: k-mesh override for this call only.

        Returns (energies, cross_section): the energy-loss grid (nw,) in
        Hartree and the cross-section in atomic units.
        """
        blocks = {
            "vecql": [tuple(float(x) for x in q)],
            "wplot": [(int(nwplot), 100, 1), (float(wplot[0]), float(wplot[1]))],
        }
        if emax is not None:
            blocks["emaxelnes"] = [float(emax)]
        if swidth is not None:
            blocks["swidth"] = [float(swidth)]
        if all(abs(float(qi)) < 1e-8 for qi in q):
            raise ValueError(
                "q=0 gives an identically zero ELNES cross-section, not an "
                "optical limit: genexpmat.f90:30-38 returns the identity for a "
                "zero vecql, so elnes.f90's occupation factor f_j (f_max - f_i) "
                "vanishes for every fully occupied and every empty state and "
                "the energy transfer is zero for the rest. Pass a nonzero q "
                "commensurate with the k-mesh, or use "
                "get_dielectric_function() for an optical spectrum."
            )
        mesh = tuple(ngridk) if ngridk else self.ngridk
        for i, qi in enumerate(q):
            if abs(mesh[i] * qi - round(mesh[i] * qi)) > 1e-6:
                raise ValueError(
                    f"q={tuple(q)} is incommensurate with the k-mesh {tuple(mesh)}: "
                    f"src/elnes.f90 requires ngridk*q to be integral, since the "
                    f"k+q state has to be another mesh point"
                )
        subdir = self._run_resumed(
            label, [_task("elnes", 140)], blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        return atomicstates.parse_two_column(subdir / _file("elnes", "ELNES.OUT"))

    def get_electron_momentum_density(self, dim=3, line=None, npoints=200,
                                      plane=None, box=None, grid=None,
                                      hkmax=12.0, label=None):
        """Electron momentum density rho(p) (task 170 then 171/172/173,
        src/writeemd.f90 and src/emdplot.f90 -> EMD1D/EMD2D/EMD3D.OUT).

        The momentum-space counterpart of the charge density,

            rho(p) = sum_{n,k,sigma} f_{nk} |<p | psi_{nk sigma}>|^2,

        with p = H + k running over reciprocal lattice vectors H out to
        `hkmax`, so -- unlike n(r) -- it is NOT periodic and extends over
        many Brillouin zones. Its projection along a direction is the
        Compton profile measured by inelastic X-ray scattering, and the
        breaks in it at the Fermi-surface calipers are the standard way to
        read a Fermi surface out of a Compton experiment.

        Task 170 (``writeemd``) computes ``rho(p)`` at every H+k by Fourier
        transforming the LAPW wavefunctions (``genwfpw``) and writes the
        UNFORMATTED direct-access file EMD.OUT; tasks 171/172/173 then
        interpolate that onto a line, plane or box. Both are run here in one
        go, so the binary intermediate never needs handling. What each plot
        contains differs by dimension, and this is not merely a slice:

        - dim=1 (EMD1D.OUT): rho INTEGRATED over the two directions
          perpendicular to the line -- i.e. the Compton profile itself,
          not a cut.
        - dim=2 (EMD2D.OUT): rho integrated along the plane's normal.
        - dim=3 (EMD3D.OUT): rho itself, no integration.

        The plotting vertices/corners are in RECIPROCAL lattice coordinates
        here (``emdplot`` passes ``bvec`` to plotpt1d/plotpt2d), unlike
        every real-space member of this family, and they routinely span
        SEVERAL zones -- Elk's own Compton-scattering example plots a
        +-2 reciprocal-lattice-unit plane. Only the FIRST segment of a 1D
        path is used: ``src/emdplot1d.f90`` calls ``plotpt1d`` with ``nv=2``
        whatever ``nvp1d`` says.

        Three prerequisites, all read from the Fortran:

        - ``src/emdplot.f90`` refuses a nonzero ``vkloff``, so the ground
          state must be built with an unshifted mesh; checked here before
          anything runs.
        - `hkmax` has to be the SAME for tasks 170 and 171/172/173:
          ``src/reademd.f90`` compares the stored ``nhk`` and k-vector of
          every record against the current ones and stops on any
          disagreement. Running both in one go, as this does, makes that
          automatic -- which is the reason not to expose task 170 on its
          own.
        - Symmetry reduction is FINE: ``src/rfhkintp.f90`` maps each
          interpolation corner back to its reduced k-point with
          ``findkpt`` and rotates the H+k vector by the corresponding
          lattice symmetry, so ``reducek=0`` is not required (unlike the
          Fermi-surface tasks, which force it themselves for collinear
          magnetism).

        `hkmax` is a hard cut-off, not a smooth one: ``rfhkintp`` returns
        exactly zero for any point with |H+p| > hkmax, so the plot is
        identically zero outside that sphere. It also sets the cost --
        the 1D and 2D forms integrate on a grid of
        ``2 max_i(nh_i ngridk_i)`` points per direction, so the 1D form is
        that number SQUARED of interpolations per plotted point.

        Returns the usual per-dimension keys, with "values" the momentum
        density in electrons per atomic unit of momentum cubed (or its
        1D/2D integral).
        """
        if any(abs(float(x)) > 1e-10 for x in self.vkloff):
            raise ValueError(
                f"get_electron_momentum_density() requires vkloff=(0,0,0) for the "
                f"ground state (this Calculation has {tuple(self.vkloff)}): "
                f"src/emdplot.f90 stops on any offset mesh"
            )
        tasks_by_dim = {1: 171, 2: 172, 3: 173}
        files_by_dim = {
            1: _file("emd_1d", "EMD1D.OUT"),
            2: _file("emd_2d", "EMD2D.OUT"),
            3: _file("emd_3d", "EMD3D.OUT"),
        }
        if dim not in tasks_by_dim:
            raise ValueError(f"dim must be 1, 2 or 3, got {dim}")
        return self._run_plot(
            label or f"emd_{dim}d",
            [_task("momentum_density", 170), _task(f"emd_{dim}d", tasks_by_dim[dim])],
            dim,
            files_by_dim[dim],
            line=line,
            npoints=npoints,
            plane=plane,
            box=box,
            grid=grid,
            extra_blocks={"hkmax": [float(hkmax)]},
        )
