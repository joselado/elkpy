"""Phonons, electron-phonon coupling and superconductivity.

The mixin here covers Elk's lattice-dynamics family beyond the three tasks
``Calculation`` already wraps directly (205 DFPT, 210 phonon DOS, 220 phonon
dispersion):

    200/201/202  phononsc   -- dynamical matrices by finite displacements in
                              a supercell (the only route for a MAGNETIC
                              cell, see get_phonons_supercell)
    208/209      bornechg   -- static Born effective charges
    230          writephn   -- phonon frequencies and eigenvectors at an
                              explicit list of q-points
    240/241      ephcouple  -- electron-phonon matrix elements, phonon
                              linewidths and mode couplings
    245          phlwidth   -- phonon linewidths along a q-path
    250          alpha2f    -- Eliashberg spectral function, lambda, T_c
    260          eliashberg -- isotropic Eliashberg equations
    270/271      gndsteph   -- self-consistent electron-phonon Bogoliubov
    280          ephdos     -- renormalised electronic DOS
    478          bornecdyn  -- dynamical (frequency-dependent) Born charges

PHYSICS SUMMARY. Within the harmonic approximation the lattice is described
by the dynamical matrix

    D_{ka,k'b}(q) = (1/sqrt(M_k M_k')) sum_R e^{i q.R}
                    d^2 E / du_{ka}(0) du_{k'b}(R),

whose eigenvalues are omega_{q nu}^2 and whose eigenvectors are the
polarisation vectors e_{q nu}. Elk builds it either by density functional
perturbation theory (task 205, one linear-response calculation per
(q, atom, direction) perturbation) or by finite differences of the
Hellmann-Feynman forces in a supercell (task 200). Everything else in this
family is post-processing of the resulting DYN files, plus -- for the
electron-phonon part -- the self-consistent first-order change in the
Kohn-Sham potential that DFPT writes alongside them (the DVS files).

The electron-phonon vertex is

    g^{q nu}_{mn}(k) = sqrt(1/(2 omega_{q nu}))
                       <psi_{m k+q}| dV_KS/du_{q nu} |psi_{n k}>,

from which task 240 forms the phonon linewidth (the Allen formula)

    gamma_{q nu} = 2 pi omega_{q nu} sum_{k,m,n} |g^{q nu}_{mn}(k)|^2
                   delta(e_{n k} - e_F) delta(e_{m k+q} - e_F),

the mode-resolved coupling lambda_{q nu} = gamma_{q nu} / (pi N(e_F)
omega_{q nu}^2), the Eliashberg spectral function (task 250)

    alpha^2 F(w) = (1/(2 pi N(e_F))) sum_{q nu} (gamma_{q nu}/omega_{q nu})
                   delta(w - omega_{q nu}),

and finally lambda = 2 int alpha^2 F(w)/w dw with the Allen-Dynes T_c, or a
full solution of the isotropic Eliashberg equations on the Matsubara axis
(task 260).

COST. This is the most expensive family in Elk by a wide margin: a single
DFPT run on a minimal 2-atom cell with ngridq=(2,2,2) takes 11-13 minutes on
the machine this project was developed on, and a converged electron-phonon
calculation for a real superconductor is hours to days. Every method below
that needs dynamical matrices re-runs the DFPT/supercell step from scratch
in its own wiped subdirectory, because that is what makes the results
trustworthy: Elk's ``dyntask``/``bectask`` treat an EXISTING DYN/BEC file as
"this piece is already done" and silently skip it, so a directory holding
partial output from a killed run corrupts the next result instead of
erroring (the hazard docs/design.md #4 records for task 205). Use
:meth:`PhononTasks.get_superconductivity` to pay that cost ONCE and get
every downstream quantity from the same set of dynamical matrices, rather
than calling the focused methods one after another.

VERIFICATION STATUS. ``get_born_charges`` is binary-verified (bulk Si, where
the diamond structure forces Z* = 0 on every atom -- see
tests/test_calculation_phonons.py). Everything else here is
FORMAT-DERIVED: the task ordering, input blocks and output formats are
transcribed from vendor/elk/src/, and the parsers are unit-tested against
fixtures built from those write statements, but no end-to-end run has been
made.
"""

import numpy as np

from .. import spec
from ..parsers import band, dos, eliashberg as parsers_eliashberg
from ..parsers import phonon as parsers_phonon

# Task codes for this family. These mirror the entries handed to the
# integrator for spec.TASKS; the existing three (205/210/220) plus 120/121
# are taken from spec.TASKS itself rather than duplicated.
PHONON_TASKS = {
    "phonons_supercell": 200,          # src/phononsc.f90
    "phonons_supercell_resume": 201,   # src/phononsc.f90, trdstate=.true.
    "phonons_supercell_dryrun": 202,   # src/phononsc.f90, empty DYN files only
    "born_charges": 208,               # src/bornechg.f90
    "born_charges_dryrun": 209,        # src/bornechg.f90, empty BEC files only
    "phonon_modes": 230,               # src/writephn.f90
    "ephcouple": 240,                  # src/ephcouple.f90
    "ephcouple_write": 241,            # src/ephcouple.f90 + putephmat
    "phonon_linewidths": 245,          # src/phlwidth.f90
    "alpha2f": 250,                    # src/alpha2f.f90
    "eliashberg": 260,                 # src/eliashberg.f90
    "gndsteph": 270,                   # src/gndsteph.f90
    "gndsteph_resume": 271,            # src/gndsteph.f90, restart from EVEC/EVAL UV
    "ephdos": 280,                     # src/ephdos.f90
    "born_charges_dynamical": 478,     # src/bornecdyn.f90
}

PHONON_OUTPUT_FILES = {
    "phonon_modes": "PHONON.OUT",        # src/writephn.f90
    "gammaq": "GAMMAQ.OUT",              # src/writegamma.f90
    "lambdaq": "LAMBDAQ.OUT",            # src/writelambda.f90
    "phlwidth": "PHLWIDTH.OUT",          # src/phlwidth.f90
    "phlwlines": "PHLWLINES.OUT",        # src/phlwidth.f90
    "alpha2f": "ALPHA2F.OUT",            # src/alpha2f.f90
    "mcmillan": "MCMILLAN.OUT",          # src/alpha2f.f90
    "eliashberg_info": "ELIASHBERG.OUT",             # src/eliashberg.f90
    "eliashberg_ia": "ELIASHBERG_IA.OUT",            # src/eliashberg.f90
    "eliashberg_gap_t": "ELIASHBERG_GAP_T.OUT",      # src/eliashberg.f90
    "eliashberg_gap_ra": "ELIASHBERG_GAP_RA.OUT",    # src/eliashberg.f90
    "eliashberg_z_ra": "ELIASHBERG_Z_RA.OUT",        # src/eliashberg.f90
    "tdos_eph": "TDOS_EPH.OUT",          # src/ephdos.f90
    "faceeh": "FACEEH.OUT",              # src/ephdos.f90
    "ephgap": "EPHGAP.OUT",              # src/gndsteph.f90
    "eph_info": "EPH_INFO.OUT",          # src/gndsteph.f90
}

# BEC_Sss_Aaaa_Pp.OUT is indexed, so it belongs in OUTPUT_FILE_TEMPLATES;
# the formatting itself lives in parsers.phonon.born_charge_filename.
BORN_CHARGE_TEMPLATE = "BEC_S{s:02d}_A{a:03d}_P{p:d}.OUT"  # src/becfext.f90


class PhononTasks:
    """``get_*`` methods for Elk's phonon / electron-phonon / superconductivity
    tasks. Mixed into :class:`elkpy.calculation.Calculation`."""

    # ------------------------------------------------------------------
    # shared plumbing
    # ------------------------------------------------------------------

    def _species_counts(self):
        """Atoms per species, in the order the `atoms` block is written --
        which is what Elk's own (is, ia) indices count over."""
        return [len(atoms) for atoms in self.structure.species.values()]

    def _species_symbols(self):
        return list(self.structure.species)

    def _dyn_task(self, method):
        """The task code that generates the dynamical matrices."""
        if method == "dfpt":
            if self.spinpol or self.spinorb:
                raise ValueError(
                    "DFPT phonons (task 205) are refused by Elk for a spin-polarised "
                    "calculation -- src/phonon.f90 hard-stops with 'spin-polarised "
                    "phonons not yet available'. spinorb counts as spin-polarised "
                    "here: src/init0.f90:126 sets spinpol=.true. whenever spinorb is "
                    "on. Use method='supercell' (task 200), "
                    "which stores and restores bfcmt0/mommtfix and is the only route "
                    "to phonons in a magnetic cell."
                )
            return spec.TASKS["phonon_dfpt"]
        if method == "supercell":
            return PHONON_TASKS["phonons_supercell"]
        raise ValueError(f"method must be 'dfpt' or 'supercell', got {method!r}")

    def _phonon_blocks(
        self,
        ngridq,
        nrmtscf=4,
        lmaxi=2,
        deltaph=None,
        swidth=None,
        stype=None,
        reduceq=None,
        radkpt=None,
    ):
        """The input blocks every dynamical-matrix run needs.

        `lmaxi` >= 2 is not a suggestion for DFPT: src/phonon.f90 hard-stops
        below it ("lmaxi too small for calculating DFPT phonons"), because
        the muffin-tin force derivatives are taken on the inner
        angular-momentum expansion. src/phononsc.f90 has no such check --
        Elk's own Al-supercell example runs at the default lmaxi=1 -- but
        the supercell forces need the same accuracy for the same reason, so
        the floor is applied to both paths here rather than left to bite
        silently. `nrmtscf` scales up the radial mesh for the same reason;
        Elk's own phonon examples use 1.5 to 12 depending on the species.
        """
        lmaxi = int(lmaxi)
        if lmaxi < 2:
            raise ValueError(
                f"lmaxi must be >= 2 for phonon calculations: src/phonon.f90 stops "
                f"outright below it, and although src/phononsc.f90 has no such check "
                f"its force differences need the same inner-muffin-tin accuracy. "
                f"Got {lmaxi}"
            )
        blocks = {
            "ngridq": [tuple(int(x) for x in ngridq)],
            "nrmtscf": [nrmtscf],
            "lmaxi": [lmaxi],
        }
        if deltaph is not None:
            blocks["deltaph"] = [float(deltaph)]
        if swidth is not None:
            blocks["swidth"] = [float(swidth)]
        if stype is not None:
            blocks["stype"] = [int(stype)]
        if reduceq is not None:
            blocks["reduceq"] = [int(reduceq)]
        if radkpt is not None:
            blocks["radkpt"] = [float(radkpt)]
        return blocks

    def _check_eph_commensurate(self, ngridq, ngridk=None):
        """src/ephcouple.f90 forms k+q for every k on the non-reduced mesh
        and calls ``findkpt``, which hard-stops if the result is not itself
        a mesh point. That requires the q-mesh to be a subgrid of the
        k-mesh in every direction. (The ``vkloff`` offset cancels in k+q, so
        only the divisions matter.)
        """
        mesh = tuple(int(x) for x in (ngridk or self.ngridk))
        q = tuple(int(x) for x in ngridq)
        bad = [i for i in range(3) if q[i] == 0 or mesh[i] % q[i] != 0]
        if bad:
            raise ValueError(
                f"the k-mesh {mesh} must be commensurate with the q-mesh {q} for the "
                f"electron-phonon tasks: src/ephcouple.f90 looks k+q up on the k-mesh "
                f"with findkpt, which stops if it is off-mesh (directions {bad} fail)"
            )

    def _eliashberg_blocks(self, nwplot, ngrkf, nswplot, wplot, mustar=None, ntemp=None):
        """The `wplot` block is a single Elk block carrying FOUR numbers:
        ``nwplot ngrkf nswplot`` on the first line and the frequency range
        on the second (src/readinput.f90 case('wplot')).

        Which of them matter depends on the consumer, and it is not uniform:
        src/phdos.f90 and src/alpha2f.f90 IGNORE the range entirely and
        build their own grid from the actual phonon bandwidth, using only
        `nwplot` (points), `ngrkf` (the fine q-mesh they Fourier-interpolate
        onto, ngrkf^3 dynamical-matrix diagonalisations -- 10^6 at Elk's
        default of 100) and `nswplot` (smoothing passes). src/ephdos.f90 and
        src/bornecdyn.f90 do use the range.
        """
        blocks = {"wplot": [(int(nwplot), int(ngrkf), int(nswplot)), tuple(wplot)]}
        if mustar is not None:
            blocks["mustar"] = [float(mustar)]
        if ntemp is not None:
            blocks["ntemp"] = [int(ntemp)]
        return blocks

    # LAMBDAQ.OUT holds HALF the Allen mode coupling, and the factor is
    # exactly 2 rather than a convention to be settled by inspection:
    #
    #   occupy.f90:94    fermidos = fermidos*occmax*t0   -> the TOTAL DOS,
    #                    both spins (occmax = 2 unpolarised, and a spinor run
    #                    counts both channels among its nstsv states anyway)
    #   ephcouple.f90:136  t1 = pi*wkptnr*occmax          -> GAMMAQ carries the
    #                    spin sum, i.e. it is the full Allen linewidth
    #   writelambda.f90:25 t1 = pi*fermidos*wphq**2       -> divides by pi*N_total
    #   alpha2f.f90:99     t1 = twopi*(fermidos/2)*...    -> the standard
    #                    normalisation, 2*pi*N_per-spin
    #
    # so the file's value is gamma/(pi N_total w^2) while Allen's
    # lambda = gamma/(pi N_per-spin w^2) is twice that -- and twice is also
    # what alpha2f, and therefore get_eliashberg_function()'s `lambda`, uses.
    # Elk's own shipped Nb example settles which side is standard:
    # examples/phonons-superconductivity/Nb-DFPT/MCMILLAN.OUT gives 1.0534,
    # the accepted value, from the alpha2f side.
    EPH_LAMBDAQ_TO_ALLEN = 2.0

    def _parse_eph_tables(self, subdir):
        """GAMMAQ.OUT + LAMBDAQ.OUT, the two q-resolved tables tasks
        240/241 always write together.

        ``couplings`` is the Allen mode coupling, i.e. LAMBDAQ.OUT's column
        times :data:`EPH_LAMBDAQ_TO_ALLEN`. Returning the file's own numbers
        beside a ``lambda`` normalised the other way is a trap: both are
        called a coupling, they differ by exactly 2, and using the smaller
        one in McMillan/Allen-Dynes moves T_c by about an order of magnitude
        while looking entirely plausible.
        """
        gamma = parsers_phonon.parse_qpoint_table(subdir / PHONON_OUTPUT_FILES["gammaq"])
        lam = parsers_phonon.parse_qpoint_table(subdir / PHONON_OUTPUT_FILES["lambdaq"])
        return {
            "natoms": gamma["natoms"],
            "qpoints": gamma["qpoints"],
            "qpoints_cartesian": gamma["qpoints_cartesian"],
            "linewidths": gamma["values"],
            "couplings": self.EPH_LAMBDAQ_TO_ALLEN * lam["values"],
            "couplings_as_written": lam["values"],
        }

    # ------------------------------------------------------------------
    # Born effective charges
    # ------------------------------------------------------------------

    def get_born_charges(
        self,
        deltaph=0.01,
        nkspolar=4,
        ngridk=None,
        dry_run=False,
        label="born_charges",
    ):
        """Static Born effective charge tensors (task 208,
        src/bornechg.f90).

        The Born (or transverse) effective charge of atom kappa is the
        mixed second derivative

            Z*_{kappa, a b} = Omega dP_a / du_{kappa b}
                            = d F_{kappa a} / d E_b,

        the macroscopic polarisation induced by displacing one sublattice
        (equivalently, the force induced by a macroscopic field). It is the
        ingredient that makes an infrared spectrum and the LO-TO splitting
        possible: a longitudinal optic mode carries a macroscopic
        depolarising field whose restoring force is proportional to
        Z*^2/eps_inf, so without Z* the LO and TO branches are degenerate at
        Gamma.

        Elk computes it by the King-Smith-Vanderbilt Berry-phase route
        (PRB 47, 1651(R) (1993)): displace one atom by +/- deltaph/2 along
        one Cartesian direction, converge the ground state for each, and
        take the difference of the electronic polarisations

            P_l = sum_k Im ln det <u_{i,k+dk_l} | u_{j,k}>,

        with dk_l = B_l / (nkspolar * ngridk_l). The core and nuclear charge
        (``chgcr + spzn``) are added to the diagonal element, so what comes
        back is the full charge, not the electronic part alone.

        - `deltaph`: the finite displacement in Bohr (Elk's default 0.01).
          Too small and the polarisation difference drowns in SCF noise; too
          large and anharmonicity contaminates it.
        - `nkspolar`: the k-mesh refinement factor along the direction being
          integrated, per Berry phase. Elk's default is 4; the convergence
          of Z* in it should be checked, since a Berry phase needs a fine
          string of k-points.
        - `ngridk`: overrides the k-mesh for this call only.
        - `dry_run`: run task 209 instead, which creates the empty BEC files
          and stops. Returns the subdirectory path; useful only for
          inspecting how many perturbations the cell implies, since elkpy
          wipes the directory on the next call and so cannot support Elk's
          own "farm the empty files out to many machines" protocol.

        COST: per (atom, direction), TWO full ground states (the -/+
        displacements) and SIX single-iteration ones -- src/polar.f90 runs
        one ``maxscl=1`` gndstate per reciprocal-lattice direction, and it
        is called after each of the two displacements. That is 6 * natoms
        full ground states and 18 * natoms cheap ones for the whole cell:
        Elk does NOT use symmetry to skip equivalent atoms here.

        Note that task 208 leaves the calculation's own STATE.OUT alone:
        ``bornechg`` hands the global ``filext`` to ``bectask``, so every
        file its internal ground states write carries the perturbation
        suffix (``STATE_S01_A001_P1.OUT`` and so on). That is why Elk's own
        LO-TO examples can chain 208 straight into 205 -- and why
        get_phonon_dispersion_loto() below does the same.

        Returns ``{"charges": {(ispecies, iatom): (3, 3) array},
        "symbols": {(ispecies, iatom): str}, "acoustic_sum": (3, 3) array}``,
        all 1-based keys in `atoms`-block order. ``charges[key][i][j]`` is
        indexed [displacement, polarisation] -- Elk's convention, the
        transpose of the usual Z*_{ab} = dP_a/du_b. ``acoustic_sum`` is the
        sum over every atom, which the acoustic sum rule requires to vanish;
        its size is the honest error bar on the calculation.
        """
        task = PHONON_TASKS["born_charges_dryrun" if dry_run else "born_charges"]
        blocks = {"deltaph": [float(deltaph)], "nkspolar": [int(nkspolar)]}
        subdir = self._run_resumed(
            label, [task], blocks, ngridk=tuple(ngridk) if ngridk else None
        )
        if dry_run:
            return subdir
        counts = self._species_counts()
        charges = parsers_phonon.parse_born_charges(subdir, counts)
        symbols = self._species_symbols()
        return {
            "charges": charges,
            "symbols": {
                (s + 1, a + 1): symbols[s]
                for s, n in enumerate(counts)
                for a in range(n)
            },
            "acoustic_sum": sum(charges.values()),
        }

    def get_born_charges_dynamical(
        self,
        deltaph=0.01,
        wplot=(0.0, 5.0),
        nwplot=20000,
        tstime=800.0,
        dtimes=0.1,
        swidth=None,
        label="born_charges_dynamical",
    ):
        """Frequency-dependent (dynamical) Born effective charges (task 478,
        src/bornecdyn.f90).

        The static Z* above is the omega -> 0 limit of

            Z*_{kappa, a b}(omega) = (1/(deltaph cos(tdphi)))
                                     FT[ J_a(t) ](omega),

        the Fourier transform of the macroscopic current that flows after a
        small static vector potential is applied to a cell with one atom
        displaced. Elk gets J(t) by running real-time TDDFT with Ehrenfest
        dynamics (it sets task=462 internally), so this measures the
        electronic response at finite frequency -- the object that governs
        infrared absorption away from the static limit rather than at it
        (C.-Yu Wang et al., PRB 106, L180303 (2022), the reference Elk's own
        examples/Born-effective-charge/hBN-dynBEC cites).

        - `wplot`/`nwplot`: the output frequency window (Hartree) and number
          of points. Unlike the phonon-DOS tasks this one DOES use the
          range.
        - `tstime`/`dtimes`: total propagation time and time step in atomic
          units. The frequency resolution is set by `tstime`, the stability
          by `dtimes`; Elk's hBN example uses 800 and 0.1.
        - `swidth`: doubles here as the exponential damping applied to J(t)
          before transforming, which is what sets the linewidth of the
          spectrum.

        COST: one ground state plus a full real-time propagation
        (tstime/dtimes steps) per (atom, direction) -- the most expensive
        single method in this module by a large factor.

        Returns ``{"frequencies": (nw,) Hartree,
        "charges": {(ispecies, iatom): (nw, 3, 3) complex},
        "symbols": {...}}`` with the same [displacement, polarisation] index
        convention as get_born_charges().

        UNTESTED: format-derived from bornecdyn.f90's write statements
        only. Note also that task 478 writes into the SAME
        ``BEC_Sss_Aaaa_Pp.OUT`` filenames as task 208 but with a completely
        different layout, so its directory must never be fed to a
        ``tphnat=.true.`` phonon run; elkpy keeps them in separate
        subdirectories.
        """
        blocks = {
            "deltaph": [float(deltaph)],
            "wplot": [(int(nwplot), 100, 1), tuple(wplot)],
            "tstime": [float(tstime)],
            "dtimes": [float(dtimes)],
        }
        if swidth is not None:
            blocks["swidth"] = [float(swidth)]
        subdir = self._run_resumed(label, [PHONON_TASKS["born_charges_dynamical"]], blocks)
        counts = self._species_counts()
        symbols = self._species_symbols()
        charges = {}
        frequencies = None
        for ispecies, natoms in enumerate(counts, start=1):
            for iatom in range(1, natoms + 1):
                rows = []
                for ip in (1, 2, 3):
                    name = parsers_phonon.born_charge_filename(ispecies, iatom, ip)
                    frequencies, z = parsers_phonon.parse_born_charge_dynamical(subdir / name)
                    rows.append(z)
                charges[(ispecies, iatom)] = np.stack(rows, axis=1)
        return {
            "frequencies": frequencies,
            "charges": charges,
            "symbols": {
                (s + 1, a + 1): symbols[s]
                for s, n in enumerate(counts)
                for a in range(n)
            },
        }

    # ------------------------------------------------------------------
    # phonon modes at explicit q-points
    # ------------------------------------------------------------------

    def get_phonon_modes(
        self,
        qpoints=((0.0, 0.0, 0.0),),
        ngridq=(2, 2, 2),
        method="dfpt",
        nrmtscf=4,
        lmaxi=2,
        label="phonon_modes",
    ):
        """Phonon frequencies AND eigenvectors at an explicit list of
        q-points (tasks 205 then 230, src/writephn.f90).

        The complement of get_phonon_dispersion(), which gives frequencies
        along a path but throws the eigenvectors away. Having the
        eigenvectors is what lets you say which atoms move in a given mode,
        classify a mode by symmetry, build an infrared or Raman intensity
        (with get_born_charges()), or freeze a soft mode into the structure.

        The q-points need not lie on the `ngridq` mesh: src/dynrtoq.f90
        Fourier-interpolates the real-space force constants to any q, which
        is the same interpolation the dispersion plot uses. What `ngridq`
        controls is the real-space range of the force constants actually
        computed, and therefore how faithful that interpolation is.

        - `qpoints`: list of q in lattice coordinates (Elk's `phwrite`
          block).
        - `method`: "dfpt" (task 205) or "supercell" (task 200).

        Returns a list of dicts, one per q-point, with keys "index",
        "qpoint", "frequencies" (nbph,) in Hartree and "eigenvectors"
        (nbph, nbph) complex. ``eigenvectors[j]`` is mode j; its components
        run over (species, atom, Cartesian) in `atoms`-block order, fastest
        in the Cartesian index. They diagonalise the MASS-WEIGHTED dynamical
        matrix (src/dynev.f90), so a physical displacement pattern is
        ``eigenvectors[j][i] / sqrt(M_i)``; an unstable mode appears as a
        NEGATIVE frequency, since dynev stores sign(sqrt(|w^2|), w^2).

        FORMAT-DERIVED: parser pinned against writephn.f90's write
        statements, not against a real run.
        """
        qpoints = [tuple(float(x) for x in q) for q in qpoints]
        blocks = self._phonon_blocks(ngridq, nrmtscf=nrmtscf, lmaxi=lmaxi)
        blocks["phwrite"] = [len(qpoints)] + qpoints
        subdir = self._run_resumed(
            label, [self._dyn_task(method), PHONON_TASKS["phonon_modes"]], blocks
        )
        return parsers_phonon.parse_phonon_modes(subdir / PHONON_OUTPUT_FILES["phonon_modes"])

    # ------------------------------------------------------------------
    # LO-TO split dispersion
    # ------------------------------------------------------------------

    def get_phonon_dispersion_loto(
        self,
        vertices=None,
        kpath=None,
        ngridq=(2, 2, 2),
        npoints=200,
        method="dfpt",
        nrmtscf=4,
        lmaxi=2,
        deltaph=0.01,
        nkspolar=4,
        wplot=(0.0, 0.5),
        nwplot=500,
        swidth=None,
        ngridk=None,
        label="phonon_dispersion_loto",
    ):
        """Phonon dispersion INCLUDING the non-analytic LO-TO term (tasks
        120, 121, 208, 205, 220 in one run, with ``tphnat=.true.``).

        In a polar insulator the q -> 0 limit of the dynamical matrix is
        direction-dependent: a longitudinal optic mode sets up a macroscopic
        electric field that stiffens it above the transverse modes. Elk adds
        the standard non-analytic correction (src/dynqnat.f90, reached from
        ``dynrtoq``/``dynqtor`` only when ``tphnat`` is set)

            D^NA_{ka,k'b}(q) = (4 pi / Omega)
                (q.Z*_k)_a (q.Z*_k')_b / (q . eps_inf . q) / sqrt(M_k M_k'),

        which needs two extra ingredients beyond the force constants: the
        Born effective charges Z* (task 208) and the electronic dielectric
        tensor eps_inf at zero frequency (task 121, whose first frequency
        point Elk reads via src/readepsw0.f90). This method runs all of it
        in one directory, which is what the LO-TO recipe in Elk's own
        examples/phonons-superconductivity/GaAs-LO-TO does.

        DELIBERATE COMPROMISE, stated rather than hidden: Elk's example
        splits this across two invocations precisely so that the dielectric
        tensor can use a dense k-mesh (16^3 for GaAs) and the phonon part a
        coarse one (4^3). A single invocation has ONE `ngridk`, so both
        stages share it. Setting `ngridk` high enough for eps_inf makes the
        Born charges and DFPT very expensive; leaving it low leaves eps_inf
        under-converged and therefore the LO-TO splitting quantitatively
        wrong (it enters as 1/eps_inf). Check the convergence of the split
        in `ngridk` explicitly, or compute eps_inf separately with
        get_dielectric_function() and compare.

        `wplot[0]` must be <= 0: src/readepsw0.f90 refuses an EPSILON file
        whose first frequency is not zero, and src/dielectric.f90 builds its
        grid as ``max(wplot(1), 0) + ...``, so a negative or zero lower
        bound is what puts a point exactly at omega = 0.

        All nine tensor components are requested (Elk's `optcomp` block, one
        row per component, which sets `noptcomp` implicitly) because
        readepsw0 reads all nine files and symmetrises.

        Returns ``{"distances": (npoints,), "frequencies": (nbph, npoints)
        Hartree, "vertices": [...], "born_charges": {...},
        "acoustic_sum": (3,3)}``.

        FORMAT-DERIVED for the chain as a whole; the Born-charge half is
        binary-verified separately (see get_born_charges).
        """
        if wplot[0] > 0.0:
            raise ValueError(
                f"wplot[0] must be <= 0 so that the dielectric grid has a point at "
                f"omega = 0; src/readepsw0.f90 stops otherwise. Got {wplot[0]}"
            )
        vertices = self._resolve_vertices(vertices, kpath)
        blocks = self._phonon_blocks(
            ngridq, nrmtscf=nrmtscf, lmaxi=lmaxi, deltaph=deltaph, swidth=swidth
        )
        blocks.update(
            {
                "nkspolar": [int(nkspolar)],
                "tphnat": [True],
                "wplot": [(int(nwplot), 100, 1), tuple(wplot)],
                "optcomp": [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)],
                "plot1d": [(len(vertices), npoints)] + [tuple(v) for v in vertices],
            }
        )
        tasks = [
            spec.TASKS["momentum_matrix"],
            spec.TASKS["dielectric"],
            PHONON_TASKS["born_charges"],
            self._dyn_task(method),
            spec.TASKS["phonon_dispersion"],
        ]
        subdir = self._run_resumed(
            label, tasks, blocks, ngridk=tuple(ngridk) if ngridk else None
        )
        distances, frequencies = band.parse_bands(subdir / spec.OUTPUT_FILES["phdisp"])
        counts = self._species_counts()
        charges = parsers_phonon.parse_born_charges(subdir, counts)
        return {
            "distances": distances,
            "frequencies": frequencies,
            "vertices": band.parse_bandlines(subdir / spec.OUTPUT_FILES["phdlines"]),
            "born_charges": charges,
            "acoustic_sum": sum(charges.values()),
        }

    # ------------------------------------------------------------------
    # electron-phonon coupling
    # ------------------------------------------------------------------

    def get_electron_phonon_coupling(
        self,
        ngridq=(2, 2, 2),
        method="dfpt",
        nrmtscf=4,
        lmaxi=2,
        swidth=0.005,
        stype=1,
        full_window=False,
        write_matrix_elements=False,
        ngridk=None,
        label="ephcouple",
    ):
        """Phonon linewidths and mode-resolved electron-phonon couplings on
        the q-mesh (tasks 205 then 240, or 241; src/ephcouple.f90).

        For each q-point and branch nu, ephcouple evaluates the Allen
        formula for the phonon linewidth

            gamma_{q nu} = 2 pi omega_{q nu} sum_{k m n}
                w_k |g^{q nu}_{mn}(k)|^2
                delta(e_{n k} - e_F) delta(e_{m k+q} + omega_{q nu} - e_{n k+q}),

        writing it to GAMMAQ.OUT, and the dimensionless mode coupling

            lambda_{q nu} = gamma_{q nu} / (pi N(e_F) omega_{q nu}^2)

        to LAMBDAQ.OUT, with N(e_F) the PER-SPIN density of states at the
        Fermi level -- which is the one place Elk's two routines disagree
        with each other, see below. The vertex g uses the self-consistent first-order
        change in the Kohn-Sham potential that task 205 saved in the DVS
        files, PLUS the rigid-ion term ``-grad V_s`` that ephcouple adds
        back in (the bare shift of the potential following the nucleus).

        Two implementation details of ephcouple worth knowing, both from the
        Fortran rather than the manual: it re-diagonalises the whole problem
        with the speed of light multiplied by 100 (i.e. NON-relativistically,
        into ``*_EPH.OUT`` files) before computing the matrix elements, and
        it determines the Fermi energy with swidth forced to 1e-5 so that
        N(e_F) is precise. It also raises the inner muffin-tin
        angular-momentum cut-off to at least 4 on its own.

        - `ngridq`: the phonon q-mesh. Must divide the k-mesh in every
          direction -- src/ephcouple.f90 looks k+q up with ``findkpt``,
          which stops on an off-mesh vector. Checked here before anything
          runs.
        - `swidth`/`stype`: the smearing entering the two delta functions.
          These are the dominant convergence parameters of the whole
          calculation together with the k-mesh; Elk's own examples use
          swidth=0.005 with stype=1 for the phonon stage -- which is
          Methfessel-Paxton order 1, not Gaussian (that is stype=0);
          src/sdelta.f90 lists the codes.
        - `full_window`: task 241 instead of 240. Task 240 restricts the
          matrix elements to states within 4*swidth of e_F (which is all the
          double delta function can see); task 241 computes the full
          nstsv x nstsv window AND writes EPHMAT.OUT, the direct-access file
          the Bogoliubov tasks 270/280 need. Setting
          `write_matrix_elements` implies this.
        - `ngridk`: overrides the k-mesh for this call only. A converged
          lambda typically needs a MUCH denser k-mesh than the ground state
          (Elk's Nb example: 12^3 for the phonons, 24^3 here), which is
          exactly what this argument is for.

        Returns ``{"qpoints": (nq, 3) lattice, "qpoints_cartesian": (nq, 3),
        "linewidths": (nq, nbph) Hartree, "couplings": (nq, nbph)
        dimensionless, "couplings_as_written": the same before the factor
        below, "natoms": int}``.

        **`couplings` is TWICE what LAMBDAQ.OUT holds, and the factor is
        exactly 2** -- settled from the Fortran, not left to the reader.
        ``occupy.f90:94`` builds ``fermidos`` as the TOTAL (both-spin)
        density of states, and ``ephcouple.f90:136`` multiplies the
        linewidth accumulator by ``occmax``, so GAMMAQ.OUT is the full Allen
        linewidth. ``writelambda.f90:25`` then divides it by
        ``pi*fermidos*w^2`` -- by the total DOS -- where Allen's formula, and
        ``alpha2f.f90:99``'s ``twopi*(fermidos/2)``, use the per-spin one. So
        the file's column is half the standard mode coupling, and half the
        ``lambda`` that get_eliashberg_function() returns beside it. Elk's own
        Nb example settles which side is standard: its shipped MCMILLAN.OUT
        gives 1.0534, the accepted value, from the alpha2f side.

        Read that as a DERIVATION, verified line by line, not as a measured
        ratio. Elk ships MCMILLAN.OUT for Nb and no LAMBDAQ.OUT beside it, so
        there is no q-resolved file here to sum and compare against 1.0534 --
        and doing it from a run of one's own would test the q-mesh
        convergence of alpha2f's own interpolation at least as much as the
        factor. What is checked is each of the four Fortran lines above.
        Feeding the
        file's own number to McMillan/Allen-Dynes with mu* = 0.15 drops T_c by
        roughly an order of magnitude and looks entirely plausible, which is
        why `couplings` is corrected here and the raw column is kept under a
        name that cannot be mistaken for a coupling.

        FORMAT-DERIVED apart from that factor.
        """
        self._check_eph_commensurate(ngridq, ngridk)
        task = PHONON_TASKS[
            "ephcouple_write" if (full_window or write_matrix_elements) else "ephcouple"
        ]
        blocks = self._phonon_blocks(
            ngridq, nrmtscf=nrmtscf, lmaxi=lmaxi, swidth=swidth, stype=stype
        )
        subdir = self._run_resumed(
            label,
            [self._dyn_task(method), task],
            blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        return self._parse_eph_tables(subdir)

    def get_phonon_linewidths(
        self,
        vertices=None,
        kpath=None,
        ngridq=(2, 2, 2),
        npoints=200,
        method="dfpt",
        nrmtscf=4,
        lmaxi=2,
        swidth=0.005,
        stype=1,
        ngridk=None,
        label="phlwidth",
    ):
        """Phonon linewidths along a q-path (tasks 205, 240 then 245;
        src/phlwidth.f90).

        Same relation to get_electron_phonon_coupling() as
        get_phonon_dispersion() has to the raw dynamical matrices: the
        linewidths gamma_{q nu} are known only on the `ngridq` mesh, so
        phlwidth builds a Hermitian "gamma matrix" whose eigenvalues squared
        are the linewidths, Fourier-interpolates THAT to real space, and
        diagonalises it simultaneously with the dynamical matrix at each
        point of the path (src/dynevs.f90). Interpolating the matrix rather
        than the eigenvalues is what keeps branches from being scrambled
        where they cross.

        Physically gamma_{q nu} is the inverse lifetime of a phonon decaying
        into an electron-hole pair, so it is nonzero only where the
        double-Fermi-surface condition can be met -- it vanishes identically
        in a gapped system, and its q-dependence maps Fermi-surface nesting.

        Same arguments as get_electron_phonon_coupling(), plus the
        `vertices`/`kpath`/`npoints` path interface get_bands() uses.

        Returns ``{"distances": (npoints,), "linewidths": (nbph, npoints)
        Hartree, "vertices": [...], "qpoints": (nq, 3),
        "linewidths_q": (nq, nbph), "couplings_q": (nq, nbph)}`` -- both the
        interpolated path and the raw mesh values it came from.

        FORMAT-DERIVED. PHLWIDTH.OUT shares BAND.OUT's exact block layout,
        so parsers/band.py is reused unchanged.
        """
        self._check_eph_commensurate(ngridq, ngridk)
        vertices = self._resolve_vertices(vertices, kpath)
        blocks = self._phonon_blocks(
            ngridq, nrmtscf=nrmtscf, lmaxi=lmaxi, swidth=swidth, stype=stype
        )
        blocks["plot1d"] = [(len(vertices), npoints)] + [tuple(v) for v in vertices]
        tasks = [
            self._dyn_task(method),
            PHONON_TASKS["ephcouple"],
            PHONON_TASKS["phonon_linewidths"],
        ]
        subdir = self._run_resumed(
            label, tasks, blocks, ngridk=tuple(ngridk) if ngridk else None
        )
        distances, linewidths = band.parse_bands(subdir / PHONON_OUTPUT_FILES["phlwidth"])
        tables = self._parse_eph_tables(subdir)
        return {
            "distances": distances,
            "linewidths": linewidths,
            "vertices": band.parse_bandlines(subdir / PHONON_OUTPUT_FILES["phlwlines"]),
            "qpoints": tables["qpoints"],
            "linewidths_q": tables["linewidths"],
            "couplings_q": tables["couplings"],
        }

    # ------------------------------------------------------------------
    # Eliashberg spectral function and superconductivity
    # ------------------------------------------------------------------

    def get_eliashberg_function(
        self,
        ngridq=(2, 2, 2),
        method="dfpt",
        nrmtscf=4,
        lmaxi=2,
        swidth=0.005,
        stype=1,
        mustar=0.15,
        nwplot=500,
        ngrkf=100,
        nswplot=1,
        ngridk=None,
        label="alpha2f",
    ):
        """Eliashberg spectral function alpha^2 F(omega), the coupling
        constant lambda and the McMillan-Allen-Dynes T_c (tasks 205, 240
        then 250; src/alpha2f.f90).

        alpha^2 F is the phonon density of states weighted by how strongly
        each mode scatters electrons at the Fermi surface,

            alpha^2 F(w) = (1/(2 pi N(e_F)))
                           sum_{q nu} (gamma_{q nu}/omega_{q nu})
                           delta(w - omega_{q nu}),

        and it is the single function that isotropic Eliashberg theory needs
        to know about the lattice. From it src/mcmillan.f90 forms

            lambda   = 2 int alpha^2F(w)/w dw,
            w_log    = exp[(2/lambda) int alpha^2F(w) ln(w)/w dw],
            <w^2>^.5 = [(2/lambda) int alpha^2F(w) w dw]^(1/2),

        and the Allen-Dynes critical temperature -- the McMillan exponential

            T_c = (w_log / 1.2 k_B) exp[-1.04(1+lambda)
                                        / (lambda - mu*(1 + 0.62 lambda))]

        multiplied by Allen and Dynes' strong-coupling and
        shape-of-spectrum correction factors f_1 f_2 (PRB 12, 905 (1975),
        the paper Elk's own output cites).

        Like phonon DOS and the linewidth path, alpha2f interpolates onto a
        FINE uniform q-mesh of `ngrkf`^3 points -- 10^6 dynamical-matrix
        diagonalisations at Elk's default of 100, which is the dominant cost
        of the post-processing (not of the DFPT) and is worth lowering for a
        cell with many branches. `nswplot` is the number of smoothing passes
        applied to alpha^2F, and the frequency window is NOT taken from the
        wplot range: alpha2f.f90 builds its own from the actual phonon
        bandwidth.

        - `mustar`: the Coulomb pseudopotential mu*, an EMPIRICAL parameter
          this method does not compute (0.1-0.15 for most metals). It enters
          only T_c, never alpha^2F or lambda.

        Returns ``{"a2f_frequencies": (nw,) Hartree, "alpha2f": (nw,),
        "lambda", "wlog", "wrms", "mustar", "tc" (kelvin), plus the
        q-resolved "qpoints"/"linewidths"/"couplings" tables}``. The
        frequency key is prefixed because get_superconductivity() returns
        this alongside a phonon dispersion on a completely different grid.

        ``lambda`` (from MCMILLAN.OUT, via alpha^2F) and the q-resolved
        ``couplings`` are on the SAME normalisation here, which needs saying
        because in Elk's own files they are not: LAMBDAQ.OUT divides by the
        total density of states where alpha2f.f90 divides by the per-spin
        one, so the file's column is half. get_electron_phonon_coupling()
        documents the factor and applies it; ``couplings_as_written`` is the
        uncorrected column if you want to compare against LAMBDAQ.OUT itself.
        That the two keys agree is derived from the Fortran rather than
        measured against a shipped file -- see get_electron_phonon_coupling()
        for why there is no such file to measure against.

        FORMAT-DERIVED, except that the MCMILLAN.OUT parser is additionally
        checked against the real file Elk ships in
        examples/phonons-superconductivity/Nb-DFPT/.
        """
        self._check_eph_commensurate(ngridq, ngridk)
        blocks = self._phonon_blocks(
            ngridq, nrmtscf=nrmtscf, lmaxi=lmaxi, swidth=swidth, stype=stype
        )
        blocks.update(
            self._eliashberg_blocks(nwplot, ngrkf, nswplot, (0.0, 0.5), mustar=mustar)
        )
        tasks = [
            self._dyn_task(method),
            PHONON_TASKS["ephcouple"],
            PHONON_TASKS["alpha2f"],
        ]
        subdir = self._run_resumed(
            label, tasks, blocks, ngridk=tuple(ngridk) if ngridk else None
        )
        return self._parse_superconductivity(subdir, gap=False)

    def get_eliashberg_gap(
        self,
        ngridq=(2, 2, 2),
        method="dfpt",
        nrmtscf=4,
        lmaxi=2,
        swidth=0.005,
        stype=1,
        mustar=0.15,
        ntemp=40,
        nwplot=500,
        ngrkf=100,
        nswplot=1,
        ngridk=None,
        label="eliashberg",
    ):
        """Superconducting gap from the isotropic Eliashberg equations
        (tasks 205, 240, 250 then 260; src/eliashberg.f90).

        Solves the coupled Matsubara-axis equations for the gap function
        Delta(i w_n) and the mass renormalisation Z(i w_n) at each of
        `ntemp` temperatures,

            Z(i w_n) = 1 + (pi T / w_n) sum_m [lambda(n-m) - lambda(n+m+1)]
                       w_m Z(i w_m) / R_m,
            Delta(i w_n) Z(i w_n) = pi T sum_m
                       [lambda(n-m) + lambda(n+m+1) - 2 mu*]
                       Delta(i w_m) Z(i w_m) / R_m,

        with w_n = pi T (2n+1), R_m = sqrt((w_m^2 + Delta_m^2) Z_m^2) and
        the phonon kernel lambda(n) = 2 int w alpha^2F(w)/(w^2 + (2 pi T
        n)^2) dw taken from the alpha^2F computed in the same run. This is
        the ISOTROPIC, flat-density-of-states approximation -- no
        Fermi-surface anisotropy, no energy dependence of N(e), and mu* is
        an input rather than a computed Coulomb repulsion.

        Elk chooses the temperature grid itself from the McMillan T_c:
        `ntemp` steps from T_c/6 (or 0.1 K) to 5 T_c, so T_c is read off as
        the temperature where the gap collapses to its 1e-4 Hartree seed,
        not printed directly.

        Returns ``{"temperatures": (nt,) kelvin, "gap": (nt,) Hartree,
        "z": (nt,), "imaginary_axis": [(w_n, Delta, Z), ...],
        "gap_real_axis": [(w, Delta(w) complex), ...],
        "z_real_axis": [...], plus every key get_eliashberg_function()
        returns}``. The per-temperature lists are RAGGED -- the number of
        Matsubara frequencies shrinks as T rises -- so they are lists, not
        arrays.

        FORMAT-DERIVED.
        """
        self._check_eph_commensurate(ngridq, ngridk)
        blocks = self._phonon_blocks(
            ngridq, nrmtscf=nrmtscf, lmaxi=lmaxi, swidth=swidth, stype=stype
        )
        blocks.update(
            self._eliashberg_blocks(
                nwplot, ngrkf, nswplot, (0.0, 0.5), mustar=mustar, ntemp=ntemp
            )
        )
        tasks = [
            self._dyn_task(method),
            PHONON_TASKS["ephcouple"],
            PHONON_TASKS["alpha2f"],
            PHONON_TASKS["eliashberg"],
        ]
        subdir = self._run_resumed(
            label, tasks, blocks, ngridk=tuple(ngridk) if ngridk else None
        )
        return self._parse_superconductivity(subdir, gap=True)

    def get_superconductivity(
        self,
        vertices=None,
        kpath=None,
        ngridq=(2, 2, 2),
        npoints=200,
        method="dfpt",
        nrmtscf=4,
        lmaxi=2,
        swidth=0.005,
        stype=1,
        mustar=0.15,
        ntemp=40,
        nwplot=500,
        ngrkf=100,
        nswplot=1,
        ngridk=None,
        label="superconductivity",
    ):
        """The WHOLE chain in one run: dynamical matrices, phonon DOS,
        phonon dispersion, electron-phonon coupling, phonon linewidths,
        alpha^2F/T_c and the Eliashberg gap (tasks 205, 210, 220, 240, 245,
        250, 260).

        This is the method to reach for. Each of the focused methods above
        re-runs the dynamical-matrix step from scratch in its own wiped
        subdirectory -- the deliberate price of never resuming from
        possibly-corrupt DYN files on disk -- so calling four of them costs
        four DFPT runs, while this costs one and reads every downstream file
        out of the same directory. That is exactly the ordering Elk's own
        examples/phonons-superconductivity/Nb-DFPT recommends in its
        commented second stage.

        Arguments are the union of get_phonon_dispersion(),
        get_phonon_linewidths() and get_eliashberg_gap(); see those for what
        each means and which are the real convergence knobs (`ngridq`,
        `ngridk`, `swidth`).

        Returns a dict combining all of them: "phonon_dos" (frequencies,
        dos), "distances"/"frequencies"/"vertices" for the dispersion,
        "linewidths" along the same path, the q-resolved tables, alpha^2F,
        the McMillan/Allen-Dynes numbers and the Eliashberg gap.

        FORMAT-DERIVED.
        """
        self._check_eph_commensurate(ngridq, ngridk)
        vertices = self._resolve_vertices(vertices, kpath)
        blocks = self._phonon_blocks(
            ngridq, nrmtscf=nrmtscf, lmaxi=lmaxi, swidth=swidth, stype=stype
        )
        blocks.update(
            self._eliashberg_blocks(
                nwplot, ngrkf, nswplot, (0.0, 0.5), mustar=mustar, ntemp=ntemp
            )
        )
        blocks["plot1d"] = [(len(vertices), npoints)] + [tuple(v) for v in vertices]
        tasks = [
            self._dyn_task(method),
            spec.TASKS["phonon_dos"],
            spec.TASKS["phonon_dispersion"],
            PHONON_TASKS["ephcouple"],
            PHONON_TASKS["phonon_linewidths"],
            PHONON_TASKS["alpha2f"],
            PHONON_TASKS["eliashberg"],
        ]
        subdir = self._run_resumed(
            label, tasks, blocks, ngridk=tuple(ngridk) if ngridk else None
        )
        result = self._parse_superconductivity(subdir, gap=True)
        result["phonon_dos"] = dos.parse_dos(subdir / spec.OUTPUT_FILES["phdos"])
        distances, frequencies = band.parse_bands(subdir / spec.OUTPUT_FILES["phdisp"])
        result["distances"] = distances
        result["frequencies"] = frequencies
        result["vertices"] = band.parse_bandlines(subdir / spec.OUTPUT_FILES["phdlines"])
        lw_distances, linewidths = band.parse_bands(subdir / PHONON_OUTPUT_FILES["phlwidth"])
        result["linewidth_distances"] = lw_distances
        result["linewidths"] = linewidths
        return result

    def _parse_superconductivity(self, subdir, gap):
        """Read every alpha2f/Eliashberg output present in `subdir`."""
        frequencies, a2f = parsers_eliashberg.parse_alpha2f(
            subdir / PHONON_OUTPUT_FILES["alpha2f"]
        )
        result = {"a2f_frequencies": frequencies, "alpha2f": a2f}
        result.update(parsers_eliashberg.parse_mcmillan(subdir / PHONON_OUTPUT_FILES["mcmillan"]))
        result.update(self._parse_eph_tables(subdir))
        if gap:
            temperatures, gap_t, z_t = parsers_eliashberg.parse_gap_vs_temperature(
                subdir / PHONON_OUTPUT_FILES["eliashberg_gap_t"]
            )
            result["temperatures"] = temperatures
            result["gap"] = gap_t
            result["z"] = z_t
            result["imaginary_axis"] = parsers_eliashberg.parse_imaginary_axis(
                subdir / PHONON_OUTPUT_FILES["eliashberg_ia"]
            )
            result["gap_real_axis"] = parsers_eliashberg.parse_real_axis(
                subdir / PHONON_OUTPUT_FILES["eliashberg_gap_ra"]
            )
            result["z_real_axis"] = parsers_eliashberg.parse_real_axis(
                subdir / PHONON_OUTPUT_FILES["eliashberg_z_ra"]
            )
        return result

    # ------------------------------------------------------------------
    # supercell phonons (the only magnetic route)
    # ------------------------------------------------------------------

    def get_phonons_supercell(
        self,
        vertices=None,
        kpath=None,
        ngridq=(2, 2, 2),
        npoints=200,
        deltaph=0.01,
        radkpt=40.0,
        nrmtscf=4,
        lmaxi=2,
        dry_run=False,
        label="phonons_supercell",
    ):
        """Phonon dispersion by the CLASSICAL frozen-phonon supercell method
        (tasks 200 then 220; src/phononsc.f90).

        Instead of linear response, this displaces each atom by
        +/- deltaph/2 in a supercell commensurate with the q-point, runs a
        ground state for each, and Fourier-transforms the resulting
        Hellmann-Feynman force differences into a column of the dynamical
        matrix:

            D_{ka,k'b}(q) = -(1/(N_sc deltaph)) sum_i e^{-i q.R_i}
                            [F_{k'b}(R_i; +u) - F_{k'b}(R_i; -u)],

        with the cos- and sin-like displacement patterns (p = 0, 1) giving
        the real and imaginary parts. Elk drives the whole loop over
        (q, species, atom, direction) in ONE invocation, using the same
        "which DYN file is missing" bookkeeping as DFPT, so this is a
        single-call method despite the multi-run flavour of Elk's own
        example (that example is about running the same job on several
        machines at once for speed, not about needing several runs).

        WHY IT EXISTS ALONGSIDE DFPT: src/phonon.f90 hard-stops on
        ``spinpol``, so for a magnetic cell the supercell method is the only
        route Elk offers -- phononsc stores and restores ``bfcmt0`` and
        ``mommtfix`` precisely so that the displaced supercells keep their
        magnetic configuration (Elk's examples/phonons-superconductivity/
        Ni-supercell).

        COST, and it is severe: the supercell has ``nqptnr`` times the atoms
        of the primitive cell, so a 4x4x4 q-mesh means a 64x supercell run
        twice for every atom and direction. A 2x2x2 mesh (8x) is the
        realistic ceiling for anything but a one-atom cell. There is no
        symmetry reduction of the displacements.

        - `radkpt`: task 200 forces ``autokpt=.true.``, so the k-mesh of the
          DISPLACED supercells is chosen from this radius rather than from
          `ngridk` (which is restored afterwards for any downstream task).
          Elk's Al example uses 40.
        - `deltaph`: displacement amplitude in Bohr.
        - `dry_run`: task 202 -- create the empty DYN files and stop,
          returning the subdirectory path.

        Returns ``(distances, frequencies)`` like get_phonon_dispersion().

        UNTESTED: format-derived from phononsc.f90; the dispersion half
        reuses the already-verified task-220 path.
        """
        if dry_run:
            blocks = self._phonon_blocks(
                ngridq, nrmtscf=nrmtscf, lmaxi=lmaxi, deltaph=deltaph, radkpt=radkpt
            )
            return self._run_resumed(
                label, [PHONON_TASKS["phonons_supercell_dryrun"]], blocks
            )
        vertices = self._resolve_vertices(vertices, kpath)
        blocks = self._phonon_blocks(
            ngridq, nrmtscf=nrmtscf, lmaxi=lmaxi, deltaph=deltaph, radkpt=radkpt
        )
        blocks["plot1d"] = [(len(vertices), npoints)] + [tuple(v) for v in vertices]
        subdir = self._run_resumed(
            label,
            [PHONON_TASKS["phonons_supercell"], spec.TASKS["phonon_dispersion"]],
            blocks,
        )
        return band.parse_bands(subdir / spec.OUTPUT_FILES["phdisp"])

    # ------------------------------------------------------------------
    # electron-phonon Bogoliubov (renormalised electronic structure)
    # ------------------------------------------------------------------

    def get_electron_phonon_bogoliubov(
        self,
        ngridq=(2, 2, 2),
        method="dfpt",
        nrmtscf=4,
        lmaxi=2,
        swidth=0.005,
        stype=1,
        wplot=(-0.05, 0.05),
        nwplot=10000,
        ngrkf=300,
        nswplot=8,
        wphcut=1e-6,
        ephscf=(8.0, 0.02),
        maxscl=None,
        ngridk=None,
        label="eph_bogoliubov",
    ):
        """Self-consistent coupled electron-phonon Bogoliubov equations and
        the renormalised electronic DOS (tasks 205, 241, 270 then 280;
        src/gndsteph.f90 / src/ephdos.f90).

        This is the ab-initio superconductivity route that does NOT go
        through the isotropic Eliashberg approximation: instead of a single
        alpha^2F, it diagonalises the coupled fermionic and bosonic
        Bogoliubov-de Gennes problem self-consistently in the full
        (state, k) and (branch, q) space, mixing the electron density matrix
        (u, v) and the phonon density matrix (w, x) together
        (C.-Yu Wang, J. K. Dewhurst, S. Sharma and E. K. U. Gross,
        PRB 105, 174509 (2022)). The anomalous part of the converged density
        matrix is the superconducting order parameter; task 280 then gives
        the quasiparticle DOS with its gap, and the "fermionic anomalous
        correlation entropy" per state, which measures how strongly each
        state participates in the condensate.

        - `wphcut`: modes below this frequency have their coupling zeroed
          (initeph.f90), which removes the acoustic divergence at q -> 0.
        - `ephscf`: (initial scale factor, per-iteration step) with which
          the electron-phonon term is switched on gradually --
          ``s <- (1-step) s + step``, so a large first element means the
          coupling starts strongly over-scaled and relaxes towards 1.
        - `maxscl`: iteration limit. Elk's own examples use 500-8000 here,
          far above the default.

        COST, and this one is not negotiable: task 241 writes EPHMAT.OUT,
        the full ``nstsv x nstsv x nbph`` vertex for every (k, q) pair, and
        initeph.f90 then holds ALL of it in memory as
        ``ephmkq(nstsv, nstsv, nbph, nkptnr, nqpt)`` -- 16 bytes times that
        product, which is hundreds of gigabytes for anything resembling a
        converged calculation. Elk's own examples say plainly that this
        "requires hundreds of cores to run in a reasonable time", and this
        build is serial (docs/design.md #8). It is wrapped here so the
        surface is complete and the input blocks are recorded correctly, not
        because it is expected to run on one machine.

        Returns ``{"dos": (frequencies, dos) for the renormalised
        electronic DOS in states/Hartree/cell with e_F at zero,
        "face_energies", "face_entropy", "face_kpoints", "gap_history"}``.

        UNTESTED, and expected to be impractical serially. Task 271 (restart
        from a previous solution) is deliberately NOT wrapped: it resumes
        from EVALUV/EVECUV files that only a previous 270 run leaves behind,
        which the wiped-subdirectory model cannot supply.
        """
        self._check_eph_commensurate(ngridq, ngridk)
        blocks = self._phonon_blocks(
            ngridq, nrmtscf=nrmtscf, lmaxi=lmaxi, swidth=swidth, stype=stype
        )
        blocks.update(self._eliashberg_blocks(nwplot, ngrkf, nswplot, wplot))
        blocks["wphcut"] = [float(wphcut)]
        blocks["ephscf"] = [tuple(float(x) for x in ephscf)]
        if maxscl is not None:
            blocks["maxscl"] = [int(maxscl)]
        tasks = [
            self._dyn_task(method),
            PHONON_TASKS["ephcouple_write"],
            PHONON_TASKS["gndsteph"],
            PHONON_TASKS["ephdos"],
        ]
        subdir = self._run_resumed(
            label, tasks, blocks, ngridk=tuple(ngridk) if ngridk else None
        )
        energies, entropy, kpoints = parsers_phonon.parse_face_histogram(
            subdir / PHONON_OUTPUT_FILES["faceeh"]
        )
        gap_path = subdir / PHONON_OUTPUT_FILES["ephgap"]
        return {
            "dos": dos.parse_dos(subdir / PHONON_OUTPUT_FILES["tdos_eph"]),
            "face_energies": energies,
            "face_entropy": entropy,
            "face_kpoints": kpoints,
            "gap_history": np.loadtxt(gap_path, ndmin=1) if gap_path.exists() else None,
        }
