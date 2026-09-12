"""Ground-state, geometry and mechanical/electric-response tasks.

This mixin covers the part of Elk's task dispatch (``vendor/elk/src/elk.f90``)
that asks "what does this crystal do when you push on it, put it in a
field, or look at a nucleus":

===== ====================== =============================================
task  subroutine             quantity
===== ====================== =============================================
5     ``hartfock``           Hartree-Fock / hybrid ground state
68    ``rdstatus``           RAM-disk report (diagnostic, stdout only)
110   ``mossbauer``          contact charge density, hyperfine field
115   ``writeefg``           electric field gradient tensor
190   ``geomplot``           XCrySDen / V_Sim geometry files
195   ``sfacrho``            X-ray structure factors
196   ``sfacmag``            magnetic structure factors
300   ``rdmft``              reduced-density-matrix functional theory
380   ``piezoelt``           piezoelectric tensor
390   ``magnetoelt``         magnetoelectric tensor
400   ``writetm``            DFT+U tensor-moment decomposition
420   ``moldyn``             Born-Oppenheimer molecular dynamics
421   ``moldyn``             ... restarted from a previous run
430   ``writestrain``        the strain-tensor basis
440   ``writestress``        stress (total-energy strain derivatives)
===== ====================== =============================================

Tasks 0/1/2/3 (``gndstate``/``geomopt``) already have named methods on
``Calculation`` -- ``get_energy()``, ``get_forces()``, ``get_relaxed()`` --
and are not repeated here. Everything below composes with them: the
``ensure_ground_state()`` cache is what supplies STATE.OUT to the resumed
tasks, and ``get_stress()`` is the natural partner of ``get_relaxed()``,
which relaxes atomic positions at fixed cell.

**Two run shapes.** Most tasks here read a converged STATE.OUT and go
through ``Calculation._run_resumed`` like every other ``get_*`` method.
Tasks 380, 390, 420/421 and 440 instead drive their own sequence of
ground-state calculations internally -- ``genstress`` sets
``trdstate=.false.`` and starts from atomic densities, ``piezoelt`` and
``magnetoelt`` override ``tshift``/``ngridk``/``maxscl``, ``moldyn``
overrides ``tshift``/``tforce``. Prefixing a ground state to those would
be work Elk immediately throws away, so they use ``_run_standalone()``
below, which is ``_run_resumed`` minus the task-1 prefix and the STATE.OUT
copy. They are correspondingly expensive: a stress calculation is
``nstrain + 1`` full SCF cycles, a piezoelectric one the same again with a
Berry-phase polarisation (itself a fine-mesh ground state per direction,
``src/polar.f90``) at every step.
"""

import shutil
from pathlib import Path

import numpy as np

from .. import spec
from ..inputfile import InputFile
from ..parsers import diagnostics, geomfile, hyperfine, moldyn, sfac, stress, tensmom, totenergy

# Task codes and output filenames for this family, each read off the
# ``select case(task)`` dispatch in vendor/elk/src/elk.f90 and the
# `open(unit,file=...)` statement in the named subroutine. These are also
# reported for addition to spec.py, which is where version-coupled
# knowledge belongs (CLAUDE.md, "Architecture"); the lookups below prefer
# spec.py's entry as soon as it exists and fall back to this local mirror
# until then, so the module works either way and never diverges silently.
_TASKS = {
    "hartree_fock": 5,
    "ramdisk_status": 68,
    "mossbauer": 110,
    "efg": 115,
    "geometry_plot": 190,
    "structure_factor_rho": 195,
    "structure_factor_mag": 196,
    "rdmft": 300,
    "piezoelectric": 380,
    "magnetoelectric": 390,
    "tensor_moments": 400,
    "molecular_dynamics": 420,
    "molecular_dynamics_resume": 421,
    "strain": 430,
    "stress": 440,
    "test_check": 500,
}

_FILES = {
    "strain": "STRAIN.OUT",
    "stress": "STRESS.OUT",
    "piezoelectric": "PIEZOELT.OUT",
    "magnetoelectric": "MAGNETOELT.OUT",
    "efg": "EFG.OUT",
    "mossbauer": "MOSSBAUER.OUT",
    "structure_factor_rho": "SFACRHO.OUT",
    "tensor_moments": "TENSMOM.OUT",
    "geometry_xsf": "crystal.xsf",
    "geometry_ascii": "crystal.ascii",
    "geometry_axsf": "crystal.axsf",
    "hf_info": "HF_INFO.OUT",
    "band_gap": "GAP.OUT",
    "dtotenergy": "DTOTENERGY.OUT",
    "moment": "MOMENT.OUT",
    "moment_magnitude": "MOMENTM.OUT",
    "rdm_info": "RDM_INFO.OUT",
    "rdm_energy": "RDM_ENERGY.OUT",
    # NB these three names are the CANONICAL spec keys: writetdengy.f90 /
    # writemomtd.f90 write the same files for molecular dynamics (420/421)
    # and for real-time TDDFT (460-463), so spec.py carries one entry each.
    "totenergy_td": "TOTENERGY_TD.OUT",
    "md_forcetot": "FORCETOT_TD.OUT",
    "md_forcemax": "FORCEMAX_TD.OUT",
    "md_displacement_lattice": "ATDISPL_TD.OUT",
    "md_displacement_cartesian": "ATDISPC_TD.OUT",
    "moment_td": "MOMENT_TD.OUT",
    "momentm_td": "MOMENTM_TD.OUT",
    "md_moment_mt": "MOMENTMT_TD.OUT",
    "md_moment_ir": "MOMENTIR_TD.OUT",
    "md_timestep": "TIMESTEP.OUT",
    "md_restart": "ATDVC.OUT",
}

_FILE_TEMPLATES = {
    "structure_factor_mag": "SFACMAG_{j}.OUT",
    "test": "TEST_{id}.OUT",
}


def _task(key):
    return spec.TASKS.get(key, _TASKS[key])


def _file(key):
    return spec.OUTPUT_FILES.get(key, _FILES[key])


def _template(key):
    return spec.OUTPUT_FILE_TEMPLATES.get(key, _FILE_TEMPLATES[key])


class GroundStateResponseTasks:
    """``get_*`` methods for Elk tasks 5, 68, 110, 115, 190, 195/196, 300,
    380, 390, 400, 420/421, 430 and 440. Mixed into ``Calculation``."""

    # ------------------------------------------------------------------
    # runner for the tasks that manage their own ground-state sequence
    # ------------------------------------------------------------------
    def _run_standalone(self, subdir_name, tasks, extra_blocks=None, ngridk=None, vkloff=None):
        """Run `tasks` alone in a wiped-clean subdirectory, with NO task-0
        or task-1 ground state prefixed and no STATE.OUT copied in.

        ``Calculation._run_resumed`` prefixes task 1 and
        ``run_tasks(resume=False)`` prefixes task 0, which is right for a
        task that reads a converged density. It is wrong for tasks 380,
        390, 420/421 and 440: each calls ``gndstate`` itself with
        ``trdstate=.false.`` (``genstress``/``moldyn``) or after
        overriding ``tshift``/``ngridk``/``maxscl``
        (``piezoelt``/``magnetoelt``), so a prefixed ground state is
        discarded work and -- worse -- would be a ground state computed
        under different settings than the ones the task then uses.

        Same wiped-clean-subdirectory discipline as ``_run_resumed``
        (docs/design.md #4): never runs in ``self.workdir``, so nothing
        here can touch the cached ground state or its manifest.
        """
        subdir = self.workdir / subdir_name
        shutil.rmtree(subdir, ignore_errors=True)
        subdir.mkdir(parents=True)
        f = InputFile()
        f.add_block("tasks", list(tasks))
        self._add_base_blocks(f, ngridk=ngridk, vkloff=vkloff)
        for name, lines in (extra_blocks or {}).items():
            f.add_block(name, lines)
        f.write(subdir / "elk.in")
        self.launcher.run(subdir)
        return subdir

    @staticmethod
    def _raise_on_elk_error(log, subdir):
        """Turn a bare Fortran ``stop`` into a Python exception.

        Elk reports most internal failures as ``write(*,...)`` followed by an
        unadorned ``stop``, which exits **0** -- so ``launcher.run()``, whose
        only test is the return code, returns normally. That is harmless when
        the run directory was wiped first (the parse then fails on a missing
        file), and it is not harmless on the one non-wiping path here: the
        previous run's output is still sitting there, ready to be parsed and
        returned as if it were new.
        """
        try:
            text = Path(log).read_text(errors="replace")
        except OSError:
            return
        for line in text.splitlines():
            if line.lstrip().startswith("Error("):
                raise RuntimeError(
                    f"elk reported {line.strip()!r} in {subdir} and then "
                    f"stopped with exit status 0; see {log}"
                )

    # ------------------------------------------------------------------
    # 430 / 440 -- strain basis and stress
    # ------------------------------------------------------------------
    def get_strain_tensors(self, label="strain"):
        """The orthonormal basis of strain tensors Elk differentiates
        along (task 430, src/writestrain.f90, STRAIN.OUT).

        Elk does not parametrise a deformation by the six Voigt components.
        ``src/genstrain.f90`` instead builds a set {e_k} of symmetry-adapted,
        mutually orthonormal 3x3 tensors: e_1 is the isotropic
        ``avec/||avec||_F``, and each subsequent candidate delta_ij is
        symmetrised over the crystal's own point group (``symmat``),
        transformed to mixed Cartesian-lattice coordinates, projected
        orthogonal to everything already accepted, and kept only if what
        remains has non-negligible norm. The number that survive,
        ``nstrain``, is therefore a property of the *symmetry*: 1 for a
        cubic crystal (every anisotropic distortion averages to a multiple
        of the identity, which e_1 already spans), more as the symmetry
        drops.

        Every strain-differentiated quantity in this module -- stress (440)
        and the piezoelectric tensor (380) -- is indexed by that same k,
        so this method is how you find out what those indices mean. The
        deformation applied for component k is
        ``A -> A + deltast * e_k`` (``src/strainabg.f90``) with A the 3x3
        lattice-vector matrix.

        Costs no self-consistent field cycle at all: ``writestrain`` is
        ``init0`` plus ``genstrain``.

        Returns a list of (3,3) arrays. Row j of each is the Cartesian
        displacement added to lattice vector j, matching elkpy's row-vector
        ``Structure.avec`` layout directly.
        """
        subdir = self._run_standalone(label, [_task("strain")])
        return stress.parse_strain(subdir / _file("strain"))

    def get_stress(self, deltast=None, label="stress"):
        """Stress: the total-energy derivative along each strain basis
        tensor (task 440, src/writestress.f90 -> src/genstress.f90,
        STRESS.OUT).

        For each e_k from ``get_strain_tensors()``, Elk deforms the cell by
        ``A -> A + deltast * e_k``, re-converges the ground state, and
        forms the forward difference

            sigma_k = [E(deltast) - E(0)] / deltast,

        in Hartree per unit of the dimensionless strain parameter. This is
        the missing half of ``get_relaxed()``: task 2/3 relaxes atomic
        positions at fixed cell, and the stress is what tells you whether
        the cell itself is at equilibrium. All components zero (to within
        ``deltast``-scale numerical noise) is the variable-cell equilibrium
        condition; feeding that information back is what ``latvopt`` does
        inside a geometry optimisation.

        Because it is a finite difference of *converged total energies*,
        the cost is ``nstrain + 1`` full SCF calculations, the first of
        them from atomic densities (``genstress`` sets ``trdstate=.false.``
        and only then reuses the previous potential). The accuracy is set
        by how well those energies cancel, so a tight ``epsengy`` matters
        more here than for a single-point energy.

        `deltast` overrides Elk's strain step (default 0.005). Smaller is
        a better derivative but a worse cancellation; Elk's own test
        tolerance for the stress components is 5e-2.

        A caveat that is physics rather than plumbing: the basis is
        defined by ``rgkmax`` times the muffin-tin radius, which does not
        change when the cell does, so an incompletely converged basis
        gives a spurious volume dependence -- Pulay stress. Measured on
        bulk Si at rgkmax = 5 (a deliberately cheap setting) the isotropic
        component comes out at about +4 GPa at the experimental lattice
        constant, where a converged LDA calculation wants to contract.
        Converge ``rgkmax`` before reading a stress as physics.

        Returns {"strain": [(3,3), ...], "stress": (nstrain,) array,
        "pressure": float}. ``pressure`` is the hydrostatic pressure in
        Hartree/Bohr^3 obtained from the isotropic component alone -- see
        ``parsers.stress.pressure_from_stress`` for the derivation -- and
        is positive when the cell wants to expand. It is meaningful only
        because ``genstrain`` guarantees e_1 is the isotropic tensor for
        this task (its ``latvopt`` branch applies to tasks 2/3 only);
        this method checks that e_1 really is proportional to ``avec``
        and returns ``pressure=None`` if it somehow is not.

        The volume and norm are taken from ``scale * avec``, the vectors
        Elk actually works with (``readinput.f90:2275`` scales them before
        any physics), so the pressure is independent of how a given cell is
        split between ``Structure.avec`` and ``Structure.scale``. Getting
        that wrong is a factor of ``scale**2`` -- 105x for silicon written
        the conventional way -- and it is invisible to the isotropy check
        above, ``avec/||avec||`` being scale-invariant.
        """
        blocks = {}
        if deltast is not None:
            blocks["deltast"] = [float(deltast)]
        subdir = self._run_standalone(label, [_task("stress")], blocks)
        result = stress.parse_stress(subdir / _file("stress"))

        # readinput.f90:2275 does avec(:,:)=sc*avec(:,:) BEFORE any physics,
        # so genstrain built e_1 from the scaled vectors and genstress
        # differentiated in that frame. Handing pressure_from_stress the
        # unscaled ones uses ||A||/sc and V/sc^3 and returns the pressure
        # times sc^2 -- 105x for Si at scale=10.26. The guard below cannot
        # catch it, since avec/||avec|| is itself scale-invariant.
        avec = np.asarray(self.structure.avec, dtype=float) * self.structure.scale
        isotropic = avec / np.linalg.norm(avec)
        if np.allclose(result["strain"][0], isotropic, atol=1e-6):
            result["pressure"] = stress.pressure_from_stress(result["stress"], avec)
        else:
            result["pressure"] = None
        return result

    # ------------------------------------------------------------------
    # 380 / 390 -- polarisation derivatives
    # ------------------------------------------------------------------
    def get_piezoelectric_tensor(self, deltast=None, nkspolar=None, label="piezoelectric"):
        """Piezoelectric tensor: the change in electric polarisation per
        unit strain (task 380, src/piezoelt.f90, PIEZOELT.OUT).

        For each strain basis tensor e_k (see ``get_strain_tensors()``),

            d_k,i = dP_i / dt   with   A -> A + t e_k,

        computed as a finite difference of the Berry-phase polarisation
        (``src/polar.f90``, King-Smith and Vanderbilt, PRB 47, 1651(R)
        (1993)) between the unstrained and strained cells. The polarisation
        itself is only defined modulo a quantum, so ``piezoelt`` first
        folds both endpoints into [0, 2pi) and then picks the branch that
        makes the difference smallest -- which silently gives the wrong
        answer if ``deltast`` is large enough to move the polarisation by
        more than half a quantum, the standard trap of the Berry-phase
        method.

        Piezoelectricity requires a non-centrosymmetric crystal: with an
        inversion centre every component vanishes identically by symmetry,
        which makes such a cell a genuine null test rather than a wasted
        run. It also requires an insulator -- the Berry phase of a metal is
        not defined.

        Expensive: one ground state per strain component, each followed by
        a polarisation evaluation that itself re-runs one SCF loop on a
        k-mesh refined by ``nkspolar`` (default 4) in each of the three
        directions in turn.

        - `deltast`: strain step (default 0.005), same variable as
          ``get_stress()``.
        - `nkspolar`: k-mesh refinement factor for the polarisation Berry
          phase (default 4). The phase is a product of overlaps along a
          string of k-points, so this controls the discretisation error of
          the polarisation, not of the strain derivative.

        Returns a list of dicts, one per strain component:
        {"strain": (3,3), "lattice": (3,), "cartesian": (3,), "length":
        float} -- the derivative in lattice and in Cartesian coordinates,
        atomic units.
        """
        blocks = {}
        if deltast is not None:
            blocks["deltast"] = [float(deltast)]
        if nkspolar is not None:
            blocks["nkspolar"] = [int(nkspolar)]
        subdir = self._run_standalone(label, [_task("piezoelectric")], blocks)
        return stress.parse_piezoelectric(subdir / _file("piezoelectric"))

    def get_magnetoelectric_tensor(self, deltabf=None, nkspolar=None, label="magnetoelectric"):
        """Magnetoelectric tensor: the electric polarisation induced by an
        applied magnetic field (task 390, src/magnetoelt.f90,
        MAGNETOELT.OUT).

            alpha_ji = dP_i / dB_j,

        computed as a central difference of the Berry-phase polarisation
        with the global field ``bfieldc`` displaced by +-deltabf/2 along
        each Cartesian direction in turn (six ground states in all).

        This is the linear magnetoelectric effect -- a magnetic field
        producing an electric dipole. It is odd under BOTH time reversal
        and spatial inversion, so it vanishes identically unless the
        crystal breaks both; and because the mechanism that ties spin to
        charge is spin-orbit coupling, a ``spinorb=False`` run gives zero
        for a reason that is physics rather than numerics.
        ``magnetoelt`` forces ``spinpol=.true.`` and ``reducebf=1``
        internally, so this method does not require ``spinpol`` on the
        ``Calculation``; ``spinorb=True`` is the setting that actually
        matters and is warned about below.

        - `deltabf`: field step in atomic units (default 0.5). Large,
          deliberately: the induced polarisation is tiny and the Berry
          phase is noisy, so the derivative needs a finite lever arm.
        - `nkspolar`: k-mesh refinement for the polarisation Berry phase
          (default 4), as in ``get_piezoelectric_tensor()``.

        Returns {"lattice": (3,3), "cartesian": (3,3), "length": (3,)},
        each matrix indexed [j, i] so that ``cartesian[j, i] = dP_i/dB_j``.
        """
        if not self.spinorb:
            raise ValueError(
                "get_magnetoelectric_tensor() requires spinorb=True: without spin-orbit "
                "coupling the magnetic field couples only to spin, which is decoupled "
                "from the orbital motion, so dP/dB vanishes identically. Elk would run "
                "six ground states and return zeros."
            )
        blocks = {}
        if deltabf is not None:
            blocks["deltabf"] = [float(deltabf)]
        if nkspolar is not None:
            blocks["nkspolar"] = [int(nkspolar)]
        subdir = self._run_standalone(label, [_task("magnetoelectric")], blocks)
        return stress.parse_magnetoelectric(subdir / _file("magnetoelectric"))

    # ------------------------------------------------------------------
    # 115 / 110 -- nuclear-site observables
    # ------------------------------------------------------------------
    def get_efg(self, lmaxi=2, label="efg"):
        """Electric field gradient at every nucleus (task 115,
        src/writeefg.f90, EFG.OUT).

            V^alpha_ij = d^2 V'_C(r) / dr_i dr_j |_{r = r_alpha},

        the second derivative of the Coulomb potential with the l=m=0
        component removed inside each muffin tin (so the spherical nuclear
        and Hartree part, which carries no gradient information, does not
        swamp it). Evaluated analytically from the muffin-tin
        spherical-harmonic expansion by applying ``gradrfmt`` twice, in
        Cartesian coordinates and Hartree/Bohr^2.

        This is a directly measurable quantity: the nuclear quadrupole
        coupling constant is ``e Q V_zz / h`` and shows up as the quadrupole
        splitting in NQR, NMR and Moessbauer spectra. It is also an
        exacting test of the density near the nucleus, which is why it is
        an all-electron speciality -- a pseudopotential code cannot compute
        it without reconstructing the core region.

        Symmetry does most of the work: the EFG is a traceless symmetric
        rank-2 tensor, so it vanishes identically at any site with cubic
        or higher point symmetry (both silicon sites in diamond, for
        instance) and is nonzero only where the local environment is
        anisotropic.

        `lmaxi` is Elk's inner-muffin-tin angular momentum cutoff. Its
        default of 1 makes the EFG unrepresentable -- an l=2 object cannot
        live in an l<=1 expansion -- and ``writeefg`` hard-``stop``s rather
        than returning zeros. This method therefore raises it to 2 for the
        resumed run, whose task-1 ground state re-converges the density
        with the extra channels present before the EFG is taken.

        Returns a list of dicts, one per atom in Elk's atom order:
        {"species", "symbol", "atom", "tensor" (3,3), "trace",
        "eigenvalues" (3,)}. The eigenvalues come from LAPACK ``dsyev``
        and so are in ascending order, not the crystallographic
        |V_zz| >= |V_yy| >= |V_xx| convention.
        """
        if lmaxi < 2:
            raise ValueError(
                f"get_efg() needs lmaxi >= 2 (got {lmaxi}): the electric field gradient "
                f"is the l=2 part of the muffin-tin potential and cannot be represented "
                f"in an l<=1 inner-region expansion; src/writeefg.f90 stops outright"
            )
        subdir = self._run_resumed(label, [_task("efg")], extra_blocks={"lmaxi": [int(lmaxi)]})
        return hyperfine.parse_efg(subdir / _file("efg"))

    def get_mossbauer(self, tbdip=None, label="mossbauer"):
        """Moessbauer parameters: contact charge density and magnetic
        hyperfine field at every nucleus (task 110, src/mossbauer.f90,
        MOSSBAUER.OUT; Bluegel, Akai, Zeller and Dederichs, PRB 35, 3271
        (1987)).

        Two quantities, both l=m=0 muffin-tin quantities sampled at the
        nucleus rather than at the muffin-tin boundary:

        - the **contact charge density** rho(0), reported at the nuclear
          centre, at the nuclear surface, averaged over the nuclear
          volume, and the same pair at the Thomson radius. The Moessbauer
          isomer shift is proportional to the difference in this quantity
          between two chemical environments, so it is the absolute
          calibration-free observable of the technique.
        - the **Fermi contact hyperfine field**, ``B = (8 pi/3) mu_B
          <m(0)>`` in Elk's units, printed in atomic units and in tesla.
          Written only for a spin-polarised run; a non-magnetic cell gets
          the density section alone, and this parser omits the keys rather
          than inventing zeros.

        With ``tbdip`` set, the spin (and, with ``tjr``, orbital) dipolar
        field is added as a further section. Elk itself notes that the
        contact term is then implicitly contained in the dipole field, so
        the two sections are not independent.

        Runs resumed: ``mossbauer`` needs the eigenvalues and occupation
        numbers on disk (``readevalsv``/``readoccsv``) before it can
        rebuild the density, which the task-1 prefix supplies.

        `tbdip` optionally switches on the dipole terms for this call.

        Returns a list of dicts, one per atom -- see
        ``parsers.hyperfine.parse_mossbauer`` for the exact keys.
        """
        blocks = {}
        if tbdip is not None:
            blocks["tbdip"] = [bool(tbdip)]
        subdir = self._run_resumed(label, [_task("mossbauer")], extra_blocks=blocks or None)
        return hyperfine.parse_mossbauer(subdir / _file("mossbauer"))

    # ------------------------------------------------------------------
    # 195 / 196 -- structure factors
    # ------------------------------------------------------------------
    def get_structure_factors(
        self, magnetic=False, hmaxvr=None, reduceh=None, vhmat=None, wsfac=None,
        label="structure_factors",
    ):
        """X-ray (task 195) or magnetic (task 196) structure factors --
        src/sfacrho.f90 / src/sfacmag.f90, SFACRHO.OUT / SFACMAG_j.OUT.

            F(H)   = int_cell rho(r) e^{i H.r} d^3r
            F_j(H) = int_cell m_j(r) e^{i H.r} d^3r

        the Fourier coefficients of the all-electron charge density and of
        each Cartesian magnetisation component, over the reciprocal-lattice
        vectors H with |H| < ``hmaxvr``. These are the diffraction
        observables themselves: F(H) is what an X-ray experiment measures
        (as |F|^2, up to the atomic form factor conventions the
        crystallographic phase choice here already matches), and F_j(H) is
        what polarised neutron diffraction measures. Being all-electron,
        the core contribution is included exactly rather than through a
        frozen atomic form factor.

        Elk prints ``omega * F`` with the imaginary part negated, i.e. the
        crystallographic convention (positive phase, no 1/Omega prefactor),
        so the H = 0 coefficient of SFACRHO.OUT is exactly the number of
        electrons in the cell -- a hard normalisation check.

        - `magnetic`: return the three magnetisation components instead of
          the charge density. Requires ``spinpol`` (``sfacmag`` returns
          immediately otherwise, writing no file); only ``ndmag``
          components exist, one for a collinear run and three for a
          non-collinear one.
        - `hmaxvr`: the H-vector cutoff in Bohr^-1 (Elk default 20, which
          gives thousands of reflections).
        - `reduceh`: fold symmetry-equivalent H-vectors into one row with
          a multiplicity (Elk default true).
        - `vhmat`: 3x3 matrix transforming the printed (h,k,l) labels, for
          reporting indices in a different (e.g. conventional) setting.
          It is written and read ROW by row (``readinput.f90:1414-1417``)
          and applied that way (``sfacrho.f90:50-52``), but the output file
          echoes it COLUMN by column (``sfacrho.f90:43-45``). The returned
          ``vhmat`` is transposed back, so what comes out is the matrix that
          went in; a diagonal one, which is the default, hides the
          difference entirely.
        - `wsfac`: (emin, emax) energy window in Hartree restricting which
          states contribute -- this is how a *valence-only* or
          *core-only* structure factor is obtained. Leaving it at the
          default uses the density straight from STATE.OUT with no
          recomputation.

        Returns the dict from ``parsers.sfac.parse_structure_factors`` for
        the charge density, or a list of such dicts (one per magnetisation
        component) when ``magnetic``.
        """
        if magnetic and not (self.spinpol or self.spinorb):
            raise ValueError(
                "get_structure_factors(magnetic=True) needs a spin-polarized calculation: "
                "src/sfacmag.f90 returns immediately when spinpol is false and writes no "
                "file at all"
            )
        blocks = {}
        if hmaxvr is not None:
            blocks["hmaxvr"] = [float(hmaxvr)]
        if reduceh is not None:
            blocks["reduceh"] = [bool(reduceh)]
        if vhmat is not None:
            blocks["vhmat"] = [tuple(row) for row in vhmat]
        if wsfac is not None:
            blocks["wsfac"] = [tuple(wsfac)]
        task = _task("structure_factor_mag" if magnetic else "structure_factor_rho")
        subdir = self._run_resumed(label, [task], extra_blocks=blocks or None)
        if not magnetic:
            return sfac.parse_structure_factors(subdir / _file("structure_factor_rho"))
        template = _template("structure_factor_mag")
        results = []
        for j in (1, 2, 3):
            path = subdir / template.format(j=j)
            if not path.exists():
                break
            results.append(sfac.parse_structure_factors(path))
        if not results:
            raise RuntimeError(
                f"task {task} wrote no SFACMAG_j.OUT in {subdir}; see elk.out"
            )
        return results

    # ------------------------------------------------------------------
    # 5 / 300 -- alternative ground states
    # ------------------------------------------------------------------
    def get_hartree_fock(self, maxscl=None, hybrid=None, hybridc=None, label="hartree_fock"):
        """Hartree-Fock (or hybrid-functional) ground state (task 5,
        src/hartfock.f90).

        Replaces the local exchange-correlation potential by the nonlocal
        Fock exchange operator,

            Sigma_x(r,r') = -sum_{occ} psi_i(r) psi_i*(r') / |r - r'|,

        built in the second-variational basis by ``src/eveqnhf.f90`` and
        iterated to self-consistency exactly as the Kohn-Sham loop is. With
        ``hybrid=.true.`` the Fock term is instead mixed into the
        semi-local functional with weight ``hybridc`` (0.25 gives PBE0),
        which is what makes this the entry point for hybrid calculations
        rather than pure HF.

        Physically: HF has no correlation at all and no self-interaction
        error, so it overestimates band gaps roughly as badly as LDA/GGA
        underestimates them -- the two bracket experiment, and a hybrid
        interpolates. The starting point is the converged Kohn-Sham density
        (``readstate``), which the task-1 prefix supplies.

        Costly in a way ordinary DFT is not: the exchange operator couples
        every pair of k-points, so the work grows as the *square* of the
        k-mesh, and the Coulomb kernel needs the q-point machinery
        (``init2``). Treat a dense mesh as out of reach.

        - `maxscl`: number of HF iterations (Elk's global default, 200, is
          far more than usually needed here).
        - `hybrid` / `hybridc`: switch to a hybrid functional and set the
          exact-exchange fraction.

        Returns {"energy": float, "energy_history": (n,) array,
        "band_gap": (n,) array, "directory": Path}: the final Hartree-Fock
        total energy in Hartree, its iteration history from the
        subdirectory's TOTENERGY.OUT (which ``hartfock`` reopens from
        record 1, so only HF iterations appear), the estimated indirect gap
        per iteration from GAP.OUT, and the run directory so HF_INFO.OUT
        can be read for the full energy decomposition.
        """
        blocks = {}
        if maxscl is not None:
            blocks["maxscl"] = [int(maxscl)]
        if hybrid is not None:
            blocks["hybrid"] = [bool(hybrid)]
        if hybridc is not None:
            blocks["hybridc"] = [float(hybridc)]
        subdir = self._run_resumed(label, [_task("hartree_fock")], extra_blocks=blocks or None)
        totenergy_path = subdir / spec.OUTPUT_FILES["totenergy"]
        energies = np.atleast_1d(np.loadtxt(totenergy_path, ndmin=1))
        gap_path = subdir / _file("band_gap")
        gaps = np.loadtxt(gap_path, ndmin=1) if gap_path.exists() else np.array([])
        return {
            "energy": totenergy.parse_final_energy(totenergy_path),
            "energy_history": energies,
            "band_gap": np.atleast_1d(gaps),
            "directory": subdir,
        }

    def get_rdmft(self, rdmxctype=None, rdmmaxscl=None, maxitn=None, maxitc=None,
                  rdmtemp=None, label="rdmft"):
        """Reduced density matrix functional theory ground state (task
        300, src/rdmft.f90).

        RDMFT replaces the density with the full one-body density matrix
        as the basic variable, so the natural-orbital occupation numbers
        n_i become variational parameters in [0, 1] rather than being
        pinned to 0 or 1 by the aufbau principle. The total energy is

            E = T[gamma] + V_ext[rho] + E_H[rho] + E_xc[{n_i}, {phi_i}],

        with the kinetic energy exact for a given gamma and only the
        exchange-correlation piece approximated -- ``rdmxctype`` selects
        which approximation (2, the default, is the power functional
        E_x = -1/2 sum_ij (n_i n_j)^alpha <ij|ji> with alpha =
        ``rdmalpha``). The minimisation alternates an inner loop over the
        natural orbitals (``rdmminc``) with one over the occupations
        (``rdmminn``).

        Its point is fractional occupations: static correlation and
        Mott-type physics that no single-determinant method can represent.
        The price is the same nonlocal-exchange cost as Hartree-Fock, once
        per outer iteration, so this is only practical on very small
        cells and coarse meshes.

        Runs resumed -- ``rdmft`` starts from the converged Kohn-Sham
        density and orbitals.

        - `rdmxctype`: exchange-correlation functional of the density
          matrix.
        - `rdmmaxscl`: number of outer self-consistency loops (default 2).
        - `maxitn` / `maxitc`: inner iteration counts for the occupation
          and natural-orbital minimisations.
        - `rdmtemp`: electronic temperature (Hartree) adding an entropy
          term to the functional.

        Returns {"energy": float, "energy_history": (n,) array,
        "directory": Path}, in Hartree, read from RDM_ENERGY.OUT
        (``write(65,'(G18.10)') engytot`` once per outer loop);
        RDM_INFO.OUT in the returned directory carries the full
        decomposition and the converged occupation numbers.
        """
        blocks = {}
        for name, value, cast in (
            ("rdmxctype", rdmxctype, int),
            ("rdmmaxscl", rdmmaxscl, int),
            ("maxitn", maxitn, int),
            ("maxitc", maxitc, int),
            ("rdmtemp", rdmtemp, float),
        ):
            if value is not None:
                blocks[name] = [cast(value)]
        subdir = self._run_resumed(label, [_task("rdmft")], extra_blocks=blocks or None)
        energies = np.atleast_1d(np.loadtxt(subdir / _file("rdm_energy"), ndmin=1))
        return {
            "energy": float(energies[-1]),
            "energy_history": energies,
            "directory": subdir,
        }

    # ------------------------------------------------------------------
    # 400 -- DFT+U tensor moments
    # ------------------------------------------------------------------
    def get_tensor_moments(self, tm3type=None, label="tensor_moments"):
        """Coupled tensor-moment decomposition of the DFT+U density matrix
        (task 400, src/writetm.f90 -> src/writetm3.f90, TENSMOM.OUT;
        Bultmark, Cricchio, Granas and Nordstrom, PRB 80, 035121 (2009)).

        The +U density matrix n_{m sigma, m' sigma'} of an (atom, l) shell
        is re-expanded in irreducible spherical tensors w^{kpr}_t obtained
        by coupling an orbital rank k = 0..2l to a spin rank p = 0 or 1,
        giving total rank r = |k-p| .. k+p and components t = -r..r. The
        low ranks are the familiar quantities -- (0,0,0) the shell
        occupation, (0,1,1) the spin moment, (1,0,1) the orbital moment --
        and the higher ones are the charge and spin multipoles that
        distinguish, for instance, competing orbital orderings that share
        the same moments. ``writetm3`` also reports each moment's
        contribution to the muffin-tin Hartree + exchange energy, so the
        decomposition is simultaneously an energy decomposition of the +U
        term.

        Requires a DFT+U calculation: the routine stops immediately if
        ``dftu = 0``, and it reads the density matrix back from
        DMATMT.OUT, which only a DFT+U ground state writes
        (``src/writedftu.f90``, called from ``gndstate``). Set the
        ``dft+u`` block through the ``Calculation``'s ``extra_blocks`` so
        that the cached ground state -- and the task-1 prefix of this run
        -- are themselves DFT+U.

        `tm3type` selects the output convention: 0 real, corresponding to
        Hermitian Gamma matrices (default); 1 the complex van der Laan
        convention (J. Phys.: Condens. Matter 7, 9947 (1995)); 2 real
        tesseral. Only type 1 gives complex components.

        Returns a list of dicts, one per (atom, l, k, p, r) moment -- see
        ``parsers.tensmom.parse_tensor_moments``.
        """
        if "dft+u" not in self.extra_blocks:
            raise ValueError(
                "get_tensor_moments() needs a DFT+U calculation: src/writetm.f90 stops "
                "with 'dftu = 0' otherwise, and the density matrix it decomposes "
                "(DMATMT.OUT) is only written by a DFT+U ground state. Construct the "
                "Calculation with extra_blocks={'dft+u': [(dftu, inpdftu), (is, l, U, J)]} "
                "-- see docs/elk_manual.txt sec. 5.42."
            )
        blocks = {}
        if tm3type is not None:
            blocks["tm3type"] = [int(tm3type)]
        subdir = self._run_resumed(label, [_task("tensor_moments")], extra_blocks=blocks or None)
        return tensmom.parse_tensor_moments(subdir / _file("tensor_moments"))

    # ------------------------------------------------------------------
    # 420 / 421 -- molecular dynamics
    # ------------------------------------------------------------------
    def get_molecular_dynamics(
        self, tstime=None, dtimes=None, ntsforce=None, atdfc=None, restart=False,
        label="moldyn",
    ):
        """Born-Oppenheimer molecular dynamics (tasks 420/421,
        src/moldyn.f90).

        At each force step the electrons are converged self-consistently at
        the current nuclear positions, the Hellmann-Feynman forces are
        taken, and the nuclei are advanced classically by
        ``src/atptstep.f90``:

            M_alpha d^2 R_alpha / dt^2 = F_alpha[{R}],

        i.e. the adiabatic approximation -- the electrons follow the nuclei
        instantaneously and stay in their ground state, with no electronic
        dynamics or non-adiabatic transitions (those are the TDDFT tasks,
        460+). Times are in Hartree atomic units (1 a.u. = 0.0241888 fs).

        The cost is one full SCF per force step, and the number of force
        steps is ``tstime / (dtimes * ntsforce)`` -- with Elk's defaults
        (1000, 0.1, 100) that is 100 ground states, so choose these
        deliberately rather than accepting the defaults.

        - `tstime`: total simulated time (a.u.).
        - `dtimes`: the underlying time-grid spacing (a.u.).
        - `ntsforce`: how many grid steps pass between force evaluations;
          the nuclei are propagated on the coarse ``ntsforce * dtimes``
          step.
        - `atdfc`: a friction coefficient damping the nuclear velocities,
          ``v -> (1 - atdfc*dt) v`` in ``src/atptstep.f90``. Zero by
          default, i.e. energy-conserving dynamics; a positive value turns
          the run into a damped relaxation towards the force-free geometry
          rather than a trajectory, which is the cheap way to use this as
          a structural optimiser with inertia.
        - `restart`: continue a previous run in the same subdirectory
          (task 421) instead of starting at t = 0 (task 420). This is the
          one deliberate exception to elkpy's wiped-clean-subdirectory
          rule: ``moldyn`` reads TIMESTEP.OUT and ATDVC.OUT back from the
          run directory to recover where it stopped and with what
          velocities, so wiping would destroy exactly the state being
          resumed. The directory must already hold a completed task-420
          run.

        Returns a dict of trajectories:
        {"time": (nsteps,), "energy": (nsteps,), "force_max": (nsteps,),
        "displacement_lattice"/"displacement_cartesian"/"force":
        (nsteps, natoms, 3), "species"/"atom": (natoms,),
        "directory": Path}. Cartesian displacements and forces are in
        atomic units (Bohr, Hartree/Bohr). ``moment`` and
        ``moment_magnitude`` are added for a spin-polarised run, and
        ``final_state`` (the ATDVC.OUT restart state: per-atom
        displacement and velocity) whenever it exists -- ``atptstep``
        returns without writing it on the last force step, so a run of a
        single step leaves no restart file at all.
        """
        blocks = {}
        if tstime is not None:
            blocks["tstime"] = [float(tstime)]
        if dtimes is not None:
            blocks["dtimes"] = [float(dtimes)]
        if ntsforce is not None:
            blocks["ntsforce"] = [int(ntsforce)]
        if atdfc is not None:
            blocks["atdfc"] = [float(atdfc)]

        if restart:
            subdir = self.workdir / label
            # ATDVC.OUT, not TIMESTEP.OUT. moldyn.f90:36-43 sets trdatdv and
            # calls readatdvc, which is the file that must exist; TIMESTEP.OUT
            # is written unconditionally by writetimes and so proves nothing.
            # A single-force-step run leaves the former and not the latter:
            # atptstep.f90:17-18 returns on itimes+ntsforce > ntimes, before
            # the writeatdvc at its line 34.
            for key, why in (
                ("md_restart", "moldyn.f90:36-43 reads it through readatdvc"),
                ("md_timestep", "readtimes.f90 reads the time grid back from it"),
            ):
                if not (subdir / _file(key)).exists():
                    raise FileNotFoundError(
                        f"restart=True needs {_file(key)} in {subdir} ({why}), "
                        "and it is not there. A run short enough that "
                        "atptstep never reached a force step writes TIMESTEP.OUT "
                        "but no ATDVC.OUT, and so cannot be restarted; run "
                        "get_molecular_dynamics() for longer than ntsforce steps."
                    )
            before = (subdir / _file("md_timestep")).stat().st_mtime_ns
            f = InputFile()
            f.add_block("tasks", [_task("molecular_dynamics_resume")])
            self._add_base_blocks(f)
            for name, lines in blocks.items():
                f.add_block(name, lines)
            f.write(subdir / "elk.in")
            log = self.launcher.run(subdir)
            # readatdvc and readtimes both end in a BARE Fortran `stop` after
            # printing, which exits 0 -- so the launcher sees success while the
            # directory still holds the PREVIOUS run's *_TD.OUT files, which
            # moldyn deletes only on task 420. Without this the caller gets the
            # old trajectory back as if it were new. Changing dtimes between
            # runs takes the same path (readtimes.f90:32-40), which no
            # existence check can catch, so the test is on the run's own
            # output: an error line, or a TIMESTEP.OUT that never moved.
            self._raise_on_elk_error(log, subdir)
            if (subdir / _file("md_timestep")).stat().st_mtime_ns == before:
                raise RuntimeError(
                    f"the restart left {_file('md_timestep')} in {subdir} "
                    "untouched, so it wrote no new time step and the "
                    "trajectory files there are still the previous run's; see "
                    f"{log}"
                )
        else:
            subdir = self._run_standalone(
                label, [_task("molecular_dynamics")], blocks or None
            )

        result = {"directory": subdir}
        energy = moldyn.parse_scalar_series(subdir / _file("totenergy_td"))
        result["time"] = energy[:, 0]
        result["energy"] = energy[:, 1]
        result["force_max"] = moldyn.parse_scalar_series(subdir / _file("md_forcemax"))[:, 1]
        for key, name in (
            ("displacement_lattice", "md_displacement_lattice"),
            ("displacement_cartesian", "md_displacement_cartesian"),
            ("force", "md_forcetot"),
        ):
            series = moldyn.parse_atom_series(subdir / _file(name))
            result[key] = series["values"]
            result.setdefault("species", series["species"])
            result.setdefault("atom", series["atom"])
            result.setdefault("step", series["step"])
        for key, name in (("moment", "moment_td"), ("moment_magnitude", "momentm_td")):
            path = subdir / _file(name)
            if path.exists():
                result[key] = moldyn.parse_scalar_series(path)[:, 1:]
        restart_path = subdir / _file("md_restart")
        if restart_path.exists():
            result["final_state"] = moldyn.parse_atdvc(restart_path)
        return result

    # ------------------------------------------------------------------
    # 190 -- geometry export
    # ------------------------------------------------------------------
    def get_geometry_files(self, label="geometry"):
        """Write and read back Elk's own view of the crystal geometry
        (task 190, src/geomplot.f90, ``crystal.xsf`` and
        ``crystal.ascii``).

        No physics is computed -- ``geomplot`` is ``init0`` and two file
        writes, so this costs no SCF cycle -- but the result is not simply
        the input echoed back. ``init0`` runs the symmetry analysis first,
        and with Elk's default ``tshift=.true.`` the atomic basis is
        translated so that a centrosymmetric crystal has its inversion
        centre at the origin (``src/findsymcrys.f90``). Diamond silicon
        entered as (0,0,0) and (1/4,1/4,1/4) comes back as
        +-(3/8,3/8,3/8) -- measured, not predicted: the diamond lattice
        has several inversion centres and which one Elk lands on is its
        own choice. That shift is invisible in every energy or band
        structure but is load-bearing wherever an origin is physically
        meaningful (elkpy hits it in the rotation-eigenvalue indicators of
        docs/design.md #28 and the transport geometry of #31), so reading
        these files back is a cheap way to see the frame Elk is actually
        using. The lattice vectors are unaffected.

        Returns {"xsf": {...}, "ascii": {...}, "xsf_path": Path,
        "ascii_path": Path, "directory": Path}. The XSF dictionary
        (``parsers.geomfile.parse_xsf``) holds ``avec``, ``symbols`` and
        Cartesian ``positions`` in ANGSTROM -- ``geomplot`` multiplies by
        the Bohr radius on that path only; the V_Sim ``ascii`` dictionary
        is in Bohr and in a rotated frame (first lattice vector along x).
        """
        subdir = self._run_standalone(label, [_task("geometry_plot")])
        xsf_path = subdir / _file("geometry_xsf")
        ascii_path = subdir / _file("geometry_ascii")
        return {
            "xsf": geomfile.parse_xsf(xsf_path),
            "ascii": geomfile.parse_vsim_ascii(ascii_path),
            "xsf_path": xsf_path,
            "ascii_path": ascii_path,
            "directory": subdir,
        }

    # ------------------------------------------------------------------
    # 68 / 500 -- diagnostics
    # ------------------------------------------------------------------
    def get_ramdisk_status(self, label="ramdisk"):
        """Report on Elk's in-memory record store (task 68,
        ``rdstatus`` in src/modramdisk.f90).

        With ``ramdisk=.true.`` (Elk's default) the eigenvector,
        eigenvalue and occupation records that would otherwise be
        EVECFV.OUT/EVECSV.OUT/EVALSV.OUT/OCCSV.OUT are held in memory
        instead. That is normally invisible, and it is also the thing that
        decides whether a large run fits: the eigenvector store scales as
        ``nkpt * nstsv * (nmatmax + nstsv)`` complex numbers.

        Not a physical observable, and the only task in this family whose
        output goes to standard output rather than to a file -- so this
        parses elkpy's captured ``elk.out``. Run resumed (task 1 first) so
        that the store has something in it; a bare task 68 in a fresh
        directory correctly reports "RAM disk not initialised", which is
        returned as ``initialised=False`` rather than raised.

        Returns {"initialised": bool, "files": [{"filename", "records",
        "bytes"}], "nfiles": int, "bytes": int}.
        """
        subdir = self._run_resumed(label, [_task("ramdisk_status")])
        return diagnostics.parse_ramdisk_status(subdir / "elk.out")

    def get_test_values(self, tasks, blocks=None, label=None):
        """Run `tasks` with Elk's ``test`` flag set and return the
        ``TEST_nnn.OUT`` files they drop (src/modtest.f90's ``writetest``).

        Many Elk subroutines end with a ``call writetest(id, descr, ...)``
        that dumps the single characteristic array of the task -- the
        stress components for 440, the EFG for 115, the structure factors
        for 195, the piezoelectric tensor for 380 -- in a fixed
        machine-readable form, together with the tolerance the Elk
        developers consider meaningful for that quantity. That tolerance
        is the useful part: it is upstream's own statement of how
        reproducible the number is, which is otherwise nowhere in the
        output.

        Returns {task_id: {"description", "type", "tolerance", "values"}},
        keyed by the ``writetest`` id (which equals the task number for
        every task in this family).
        """
        blocks = dict(blocks or {})
        blocks["test"] = [True]
        label = label or self._default_label(list(tasks) + ["test"], blocks)
        subdir = self._run_resumed(label, list(tasks), extra_blocks=blocks)
        results = {}
        for path in sorted(subdir.glob("TEST_*.OUT")):
            identifier = int(path.name[len("TEST_") : -len(".OUT")])
            results[identifier] = diagnostics.parse_test_file(path)
        if not results:
            raise RuntimeError(
                f"tasks {list(tasks)} wrote no TEST_*.OUT files in {subdir}; not every "
                f"Elk subroutine calls writetest"
            )
        return results

    def run_regression_check(self, reference_dir, tasks, blocks=None, label="testcheck"):
        """Compare this calculation's ``TEST_nnn.OUT`` output against a
        directory of reference files (task 500, src/testcheck.f90).

        ``testcheck`` is Elk's own regression harness: for every
        ``TEST_nnn.OUT_`` reference file present it reads the matching
        ``TEST_nnn.OUT`` produced by this run, checks the variable type and
        count agree, and compares value by value against the tolerance
        stored in the reference. Any mismatch is a Fortran ``error stop``,
        which surfaces here as a ``RuntimeError`` from the launcher.

        This is the one way to check a build or a patch series against
        upstream's own expected numbers -- Elk ships such reference files
        in ``vendor/elk/tests/`` -- rather than against elkpy's judgement
        of what is reasonable. It is a development tool, not physics.

        `reference_dir` must hold the ``TEST_nnn.OUT_`` files (note the
        trailing underscore); they are copied into the run directory,
        `tasks` are run with ``test=.true.`` to generate the fresh
        ``TEST_nnn.OUT`` files, and task 500 diffs them.

        Returns {"passed": bool, "values": {...}, "reference": {...},
        "directory": Path, "error": str or None}. ``values`` and
        ``reference`` are the parsed fresh and reference files, so a
        failure can be inspected rather than merely reported.
        """
        reference_dir = Path(reference_dir)
        references = sorted(reference_dir.glob("TEST_*.OUT_"))
        if not references:
            raise FileNotFoundError(f"no TEST_*.OUT_ reference files in {reference_dir}")

        blocks = dict(blocks or {})
        blocks["test"] = [True]
        subdir = self._run_resumed(label, list(tasks), extra_blocks=blocks)
        for path in references:
            shutil.copyfile(path, subdir / path.name)

        f = InputFile()
        f.add_block("tasks", [_task("test_check")])
        self._add_base_blocks(f)
        f.write(subdir / "elk.in")
        error = None
        try:
            self.launcher.run(subdir, log_name="testcheck.out")
        except RuntimeError as exc:
            error = str(exc)

        values, reference = {}, {}
        for path in sorted(subdir.glob("TEST_*.OUT")):
            identifier = int(path.name[len("TEST_") : -len(".OUT")])
            values[identifier] = diagnostics.parse_test_file(path)
        for path in references:
            identifier = int(path.name[len("TEST_") : -len(".OUT_")])
            reference[identifier] = diagnostics.parse_test_file(subdir / path.name)
        return {
            "passed": error is None,
            "values": values,
            "reference": reference,
            "directory": subdir,
            "error": error,
        }
