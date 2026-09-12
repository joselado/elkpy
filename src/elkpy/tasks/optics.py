"""Optical, dielectric, TDDFT and Bethe-Salpeter task wrappers.

Mixin for :class:`elkpy.calculation.Calculation`, covering Elk's optics /
response family (see ``vendor/elk/src/elk.f90``'s dispatch):

    125  nonlinopt         second-harmonic generation, chi(-2w; w, w)
    130  writeexpmat       < i,k+q | e^{iq.r} | j,k > matrix elements
    135  writewfpw         wavefunctions in a pure plane-wave basis
    180  writeepsinv       inverse RPA dielectric matrix (screened Coulomb)
    185  writehmlbse       Bethe-Salpeter Hamiltonian
    186  writeevbse        its eigenvalues/eigenvectors (exciton energies)
    187  dielectric_bse    excitonic dielectric function
    285  aceplot           anomalous correlation entropy
    320  tddftlr           linear-response TDDFT dielectric function
    330  tddftsplr         spin-polarised linear response (magnons)
    331  tddftsplr         the same, also dumping chi(G,G')
    450  genafieldt        the laser vector potential A(t)
    455  writeafpdt        its power density
    456  writeefieldw      E(w) = FT of -(1/c) dA/dt
    460  tddft             real-time evolution from t = 0
    461  tddft             ... restart
    462  tddft             ... with Ehrenfest nuclear dynamics
    463  tddft             ... Ehrenfest restart
    480  dielectric_tdrt   eps(w) from the real-time current, A(t) given
    481  dielectric_tdrt   ... assuming E(t) is a delta function at t = 0

Tasks 121 (`get_dielectric_function`) and 122 (`get_moke`) already live in
Calculation itself and are not repeated here.

PREREQUISITE CHAINS are the defining feature of this family, and every one
below is established from the Fortran rather than the manual. Elk's task
dispatch is a LOOP over the `tasks` block, so a chain is expressed simply
by listing its members in order in a single `_run_resumed` call, all
sharing one directory:

  * 125, 187      need PMAT.OUT           (task 120, src/writepmat.f90)
  * 185           needs EPSINV.OUT        (task 180) whenever `hdbse` is
                  true -- which is Elk's DEFAULT (readinput.f90:270), and
                  src/hmldbsek.f90:148 reads it via getcfgq
  * 186           needs HMLBSE.OUT        (task 185)
  * 187           needs EVBSE.OUT         (task 186) and PMAT.OUT
  * 455, 456      need AFIELDT.OUT        (task 450, via readafieldt)
  * 460-463       need AFIELDT.OUT        (src/init0.f90:337 reads it for
                  exactly tasks 460,461,462,463,480,481,485)
  * 480/481       need AFIELDT.OUT and JTOT_TD.OUT (task 460-463)
  * 285           needs EVALUV/EVECUV.OUT (task 270/271), which need the
                  electron-phonon matrix elements (240/241) and the DFPT
                  phonons (205) -- hours of compute, see
                  get_anomalous_entropy()

No `ngridq` block is needed for any of these: src/init2.f90:44-47 FORCES
ngridq = ngridk for tasks 105, 180, 185, 320, 330, 331, 670 and 680, which
is exactly what makes src/hmldbsek.f90's getcfgq lookup of EPSINV.OUT at
q = k - k' resolvable. Task 285's own chain is the exception -- the DFPT
phonons (205) are not on that list, so get_anomalous_entropy() takes
`ngridq` explicitly.

FREQUENCY-GRID TRAPS (each transcribed from the source that builds it):

  * task 125 and 187 build w(iw) = wplot(2)/nwplot * (iw-1): they start at
    ZERO and IGNORE wplot(1) entirely, unlike task 121. The methods below
    therefore take `wmax` rather than a `wplot` pair, so there is no
    silently-discarded argument.
  * task 320/330/331 use src/init3.f90's wrf grid, which does honour
    wplot(1) and does NOT clip a negative one to zero. tddftlr additionally
    writes only iw = 2..nwplot, so its files hold nwplot-1 rows.
  * task 456/480/481 clip: w1 = max(wplot(1), 0).
"""

from pathlib import Path

import numpy as np

from .. import spec
from ..inputfile import InputFile
from ..parsers import bse as parsers_bse
from ..parsers import expmat as parsers_expmat
from ..parsers import nonlinopt as parsers_nonlinopt
from ..parsers import tddft as parsers_tddft


#: `fxctype` codes reachable in this build, i.e. those src/genvfxc.f90
#: accepts AND src/modfxcifc.f90 can evaluate without libxc (which
#: build-config/make.inc stubs out via libxcifc_stub.f90).
FXC_KERNELS = {
    "RPA": 0,        # f_xc = 0; the response is pure Hartree screening
    "ALDA": 3,       # adiabatic LDA, Perdew-Wang-Ceperley-Alder f_xc
    "LRC": 200,      # long-range correction -(alpha + beta w^2)/(4 pi q^2)
    "bootstrap": 210,        # self-consistent bootstrap kernel
    "bootstrap1": 211,       # single-iteration bootstrap
}


def _cartesian_indices(component, arity):
    """Validate an `optcomp` entry of `arity` 1-based Cartesian indices."""
    component = tuple(int(x) for x in component)
    if len(component) != arity:
        raise ValueError(
            f"expected {arity} Cartesian indices per component, got {component}"
        )
    if any(not (1 <= x <= 3) for x in component):
        raise ValueError(f"Cartesian indices must be 1, 2 or 3 (x, y, z), got {component}")
    return component


class OpticsTasks:
    """get_* methods for Elk's optical/dielectric/TDDFT/BSE task family."""

    # ------------------------------------------------------------------
    # task 125 -- second-harmonic generation
    # ------------------------------------------------------------------

    def get_shg(
        self, components=((1, 1, 1),), wmax=0.5, nwplot=500, swidth=0.001,
        ngridk=None, label="shg",
    ):
        """Second-order optical susceptibility chi^{abc}(-2w; w, w), the
        second-harmonic-generation tensor -- tasks 120 then 125,
        src/writepmat.f90 / src/nonlinopt.f90.

        nonlinopt.f90 follows Sipe and Ghahramani, PRB 48, 11705 (1993)
        and Hughes and Sipe, PRB 53, 10751 (1996) (the papers it names in
        its own header), evaluating the length-gauge expression from the
        position matrix elements r_nm = p_nm / (i (e_m - e_n)) built out
        of task 120's PMAT.OUT. It reports the total

            chi^{abc} = chi_II^{abc} + eta_II^{abc}
                        + (i/2w) sigma_II^{abc}

        and each contribution separately: `chi_II` is the pure interband
        (virtual-electron/virtual-hole) term, `eta_II` collects the
        intraband modulation of the interband polarisation, and
        `sigma_II` the intraband (Delta = v_mm - v_nn) term. Both the 2w
        (resonant at half the gap) and w denominators appear, which is why
        SHG probes states a linear spectrum cannot.

        chi vanishes identically in a centrosymmetric crystal: under
        inversion chi -> -chi while the crystal is unchanged. That makes a
        diamond-structure test a NULL test, not a magnitude test.

        - `components`: (a, b, c) Cartesian TRIPLES, 1-based (1, 2, 3 =
          x, y, z) -- Elk's `optcomp` block, which for this task takes
          three indices per line rather than two. Passing a 2-tuple here
          is rejected: readinput.f90 pads a short line with " 1 1", so a
          two-index line would silently become c = 1.
        - `wmax`: upper end of the photon-energy window in Hartree. There
          is deliberately no lower bound: src/nonlinopt.f90 builds its grid
          as w(iw) = wmax/nwplot * (iw-1), always starting at zero and
          ignoring wplot(1).
        - `swidth`: the broadening (Hartree). It does double duty here --
          besides entering the denominators as i/tau, it is also the
          threshold below which nonlinopt DISCARDS a transition
          (`if (abs(t1) > swidth)` guards on e(m,n) and on the
          e(l,n)-e(m,l) resonance), so a very large value silently removes
          near-degenerate contributions.
        - `ngridk`: optionally overrides the k-mesh for this call only.
          A triple state sum plus a zone integral: this converges slowly.

        The number of empty states is Elk's `nempty` and belongs to the
        Calculation (`extra_blocks={"nempty": [...]}`); the l-summation
        runs over all `nstsv`, so a small `nempty` truncates the virtual
        states the two-photon resonance needs.

        Returns {"energies": (nw,) Hartree,
                 "chi": {(a,b,c): (nw,) complex},
                 "chi_ii": {...}, "eta_ii": {...}, "sigma_ii": {...},
                 "workdir": Path} -- all in atomic units.
        """
        components = [_cartesian_indices(c, 3) for c in components]
        blocks = {
            "wplot": [(nwplot, 100, 1), (0.0, float(wmax))],
            "swidth": [float(swidth)],
            "optcomp": list(components),
        }
        subdir = self._run_resumed(
            label,
            [spec.TASKS["momentum_matrix"], spec.TASKS["nonlinear_optics"]],
            blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        keys = {
            "chi": "shg_chi",
            "chi_ii": "shg_chi_ii",
            "eta_ii": "shg_eta_ii",
            "sigma_ii": "shg_sigma_ii",
        }
        result = {name: {} for name in keys}
        energies = None
        for a, b, c in components:
            for name, template_key in keys.items():
                filename = spec.OUTPUT_FILE_TEMPLATES[template_key].format(a=a, b=b, c=c)
                energies, values = parsers_nonlinopt.parse_chi2(subdir / filename)
                result[name][(a, b, c)] = values
        result["energies"] = energies
        result["workdir"] = subdir
        return result

    # ------------------------------------------------------------------
    # task 130 -- plane-wave (density) matrix elements
    # ------------------------------------------------------------------

    def get_expiqr(self, vecql=(0.0, 0.0, 0.0), kpoint_indices=None, label="expiqr"):
        """Matrix elements < i, k+q | e^{iq.r} | j, k > (task 130,
        src/writeexpmat.f90, EXPIQR.OUT).

        These are the building block of every density response function:
        chi0(q, w) is a sum of |< c,k+q | e^{iq.r} | v,k >|^2 over
        occupied-empty pairs. Exposing them directly lets a custom
        response function (a plasmon dispersion, an EELS loss spectrum, a
        local-field-corrected chi0) be assembled in Python.

        - `vecql`: the q-vector in lattice (fractional reciprocal)
          coordinates, Elk's `vecql` block. src/genexpmat.f90 needs k+q to
          be a point of the k-mesh, so q should be commensurate with
          `ngridk` -- checked here rather than left to fail inside Elk.
        - `kpoint_indices`: 1-based indices into Elk's REDUCED k-point
          list (KPOINTS.OUT order) to write out; Elk's `kstlist` block.
          Default is k-point 1 alone (Elk's own default). Pass a list to
          get several. Passing indices at all is much cheaper than the
          whole mesh: the file holds nstsv^2 complex numbers per k-point.

        Note src/writeexpmat.f90 scales the muffin-tin phase factor by the
        cell volume (`expmt(:,:)=omega*expmt(:,:)`), so every returned
        matrix element carries a factor of Omega; the completeness bound is
        therefore sum_j |M_ij|^2 <= Omega^2, approached from below as the
        state count grows.

        Returns {"vecql": (3,), "vecqc": (3,) Cartesian a.u.,
                 "kpoints": [{"vkl", "vkc", "matrix"}], "workdir": Path}
        with matrix[i, j] = < i, k+q | e^{iq.r} | j, k >.
        """
        vecql = tuple(float(x) for x in vecql)
        self._check_commensurate_q(vecql)
        if kpoint_indices is None:
            kstlist = [(1, 1)]
        else:
            kstlist = [(int(ik), 1) for ik in kpoint_indices]
            if any(ik < 1 for ik, _ in kstlist):
                raise ValueError("kpoint_indices are 1-based indices into Elk's k-point list")
        blocks = {"vecql": [vecql], "kstlist": kstlist}
        subdir = self._run_resumed(label, [spec.TASKS["expmat"]], blocks)
        result = parsers_expmat.parse_expiqr(subdir / spec.OUTPUT_FILES["expiqr"])
        result["workdir"] = subdir
        return result

    def _check_commensurate_q(self, vecql, ngridk=None):
        """Raise unless q is a vector of the k-mesh's own reciprocal
        lattice, i.e. ngridk * q is integral.

        src/tddftlr.f90 and src/tddftsplr.f90 both hard-`stop` on this
        ("q-vector incommensurate with k-point grid"); src/genexpmat.f90
        needs the same thing to find k+q in the mesh. Checking in Python
        turns a lost run into an immediate ValueError.
        """
        grid = np.array(ngridk if ngridk else self.ngridk, dtype=float)
        v = grid * np.array(vecql, dtype=float)
        if np.any(np.abs(v - np.rint(v)) > 1e-6):
            raise ValueError(
                f"vecql={tuple(vecql)} is incommensurate with ngridk="
                f"{tuple(int(g) for g in grid)}: ngridk*q must be integral, got "
                f"{tuple(v)}. Elk hard-stops on this."
            )

    # ------------------------------------------------------------------
    # task 135 -- plane-wave wavefunctions
    # ------------------------------------------------------------------

    def get_plane_wave_wavefunctions(self, hkmax=None, label="wfpw"):
        """Second-variational wavefunctions re-expanded in a pure plane-wave
        basis (task 135, src/writewfpw.f90, WFPW.OUT).

        genwfpw projects the full LAPW state -- muffin-tin spherical-
        harmonic expansion included -- onto plane waves e^{i(H+k).r} with
        |H+k| < `hkmax`, so the result is the same wavefunction an
        (ultra-hard) pseudopotential code would produce. Useful for
        computing the electron momentum density, for comparing against a
        plane-wave code, or for feeding an external post-processor.

        Convergence is visible in the data itself: sum_H |c_H|^2 for a
        state approaches 1 from below as `hkmax` grows (the muffin-tin
        cusps are what a plane-wave basis struggles with).

        - `hkmax`: the plane-wave cut-off in atomic units, Elk's `hkmax`
          block (default 12.0, readinput.f90:280). The array is
          (nhkmax, nspinor, nstsv) complex per k-point, and nhkmax grows
          as hkmax^3, so raising it is expensive in BOTH time and disk.

        Returns {"kpoints": (nkpt, 3) lattice coordinates,
                 "wfpw": (nkpt, nhkmax, nspinor, nstsv) complex,
                 "workdir": Path}. Note the k-points are Elk's REDUCED set.
        """
        blocks = {}
        if hkmax is not None:
            blocks["hkmax"] = [float(hkmax)]
        subdir = self._run_resumed(label, [spec.TASKS["wfpw"]], blocks)
        result = parsers_expmat.read_wfpw(subdir / spec.OUTPUT_FILES["wfpw"])
        result["workdir"] = subdir
        return result

    # ------------------------------------------------------------------
    # task 180 -- inverse dielectric matrix / screened interaction
    # ------------------------------------------------------------------

    def get_inverse_dielectric_matrix(
        self, gmaxrf=None, swidth=None, ngridk=None, label="epsinv",
    ):
        """The inverse RPA dielectric matrix eps^-1(G, G'; q, w) (task 180,
        src/writeepsinv.f90 -> src/epsinv.f90, EPSINV.OUT).

        This is the screened Coulomb interaction W = eps^-1 v: the RPA
        polarisability v^1/2 chi0 v^1/2 is accumulated over the
        non-reduced k-mesh (src/genvchi0.f90), 1 - v^1/2 chi0 v^1/2 is
        formed and inverted in the G-vector basis, once per q-point of the
        `ngridq` mesh. It is the input the BSE's DIRECT (screened
        electron-hole attraction) term needs -- src/hmldbsek.f90:148 reads
        it back -- and the same object a GW self-energy would use.

        STATIC ONLY, and that is a property of the task rather than a
        choice here: src/init3.f90 sets nwrf = nwplot only for tasks 320,
        330 and 331, and otherwise leaves nwrf = 1 with
        wrf(1) = i*swidth. So task 180 evaluates eps^-1 at w = 0.

        - `gmaxrf`: the |G| cut-off for the response-function basis
          (Elk's `gmaxrf`, default 3.0). This sets ngrf and hence the
          ngrf x ngrf matrix that is inverted per q-point -- the dominant
          cost and the dominant convergence parameter.
        - `swidth`: the imaginary part of the frequency (Hartree).
        - `ngridk`: optionally overrides the k-mesh for this call only.

        EPSINV.OUT is written UNFORMATTED/DIRECT with one record per
        q-point holding (vql, ngrf, nwrf, epsi) (src/putepsinv.f90), and
        its record length depends on ngrf, which depends on the run's own
        G-vector set. It is therefore treated as an opaque intermediate:
        this returns the directory holding it, for a subsequent BSE call
        or for an external reader. Use get_bse_dielectric(), which runs
        task 180 in the same directory as 185-187 and so never has to move
        the file.

        Returns {"workdir": Path, "path": Path to EPSINV.OUT}.
        """
        blocks = {}
        if gmaxrf is not None:
            blocks["gmaxrf"] = [float(gmaxrf)]
        if swidth is not None:
            blocks["swidth"] = [float(swidth)]
        subdir = self._run_resumed(
            label, [spec.TASKS["epsinv"]], blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        return {"workdir": subdir, "path": subdir / spec.OUTPUT_FILES["epsinv"]}

    # ------------------------------------------------------------------
    # tasks 185/186/187 -- the Bethe-Salpeter chain
    # ------------------------------------------------------------------

    def _bse_blocks(
        self, nvbse, ncbse, bsefull, exchange, direct, gmaxrf, swidth,
        wmax, nwplot, components,
    ):
        blocks = {
            "nvbse": [int(nvbse)],
            "ncbse": [int(ncbse)],
            "bsefull": [bool(bsefull)],
            "hxbse": [bool(exchange)],
            "hdbse": [bool(direct)],
            "swidth": [float(swidth)],
            "wplot": [(int(nwplot), 100, 1), (0.0, float(wmax))],
        }
        if gmaxrf is not None:
            blocks["gmaxrf"] = [float(gmaxrf)]
        if components is not None:
            blocks["optcomp"] = list(components)
        return blocks

    def _bse_tasks(self, direct, final_task):
        """The task chain ending at `final_task`.

        Task 180 is included only when the DIRECT term is switched on,
        since that is the only consumer of EPSINV.OUT
        (src/hmldbsek.f90) -- with hdbse=.false. the Hamiltonian is
        diagonal-plus-exchange and needs no screening. `hdbse` defaults to
        .true. in Elk (readinput.f90:270), so the default chain does
        include it.

        No `ngridq` is emitted: src/init2.f90:44-47 forces
        ngridq = ngridk for tasks 180 and 185, which is what makes
        src/hmldbsek.f90's EPSINV.OUT lookup at q = k - k' resolvable.
        """
        tasks = [spec.TASKS["momentum_matrix"]]
        if direct:
            tasks.append(spec.TASKS["epsinv"])
        tasks.append(spec.TASKS["bse_hamiltonian"])
        tasks.append(spec.TASKS["bse_eigenvectors"])
        if final_task is not None:
            tasks.append(final_task)
        return tasks

    def get_bse_excitations(
        self, nvbse=2, ncbse=3, bsefull=False, exchange=True, direct=True,
        gmaxrf=None, swidth=0.001, ngridk=None, label="bse_excitations",
    ):
        """Bethe-Salpeter excitation energies (tasks 120, 180, 185, 186 --
        src/writehmlbse.f90 then src/writeevbse.f90, EIGVAL_BSE.OUT).

        The BSE Hamiltonian acts on electron-hole pair states
        |v, c, k> and reads, in the Tamm-Dancoff (resonant) block,

            H[(vck), (v'c'k')] = (e_ck - e_vk) delta
                                 + 2 hxbse * K^x - hdbse * K^d,

        with K^x the (unscreened, repulsive) exchange kernel -- the local
        fields -- and K^d the SCREENED direct kernel, the attractive
        electron-hole interaction that binds excitons. Its eigenvalues are
        the correlated excitation energies; an eigenvalue below the
        independent-particle gap is a bound exciton, and the gap-minus-
        eigenvalue difference is its binding energy.

        - `nvbse`/`ncbse`: valence and conduction states per k-point
          entering the pair basis (Elk's `nvbse`/`ncbse`, defaults 2 and
          3). The matrix size is nmbse = nvbse * ncbse * nkptnr (doubled
          when `bsefull`), and the diagonalisation is DENSE -- cost grows
          as nmbse^3 and memory as nmbse^2. This is the expensive member
          of the family.
        - `bsefull`: solve the full non-Hermitian Hamiltonian including
          the anti-resonant block and the coupling between the two,
          instead of the Tamm-Dancoff Hermitian block alone.
        - `exchange`/`direct`: Elk's `hxbse`/`hdbse`. Switching `direct`
          off drops task 180 from the chain and removes the excitonic
          binding altogether.
        - `gmaxrf`: the response-function |G| cut-off used by task 180 for
          the screening.
        - `ngridk`: optionally overrides the k-mesh for this call only.
          Exciton binding energies converge notoriously slowly in it.

        src/genidxbse.f90 hard-stops with "not enough conduction states"
        unless nstsv exceeds the topmost occupied band by at least ncbse,
        so `nempty` (on the Calculation, via `extra_blocks`) must be at
        least ncbse; that is checked here rather than left to Elk. The
        check is CONSERVATIVE, not exact: Elk's `nempty` is per atom
        (`nempty0`, scaled up by the atom count in src/init0.f90), so a
        multi-atom cell passing the check can still be short of states in
        principle -- and passing it is no guarantee, only failing it is a
        guarantee of failure.
        src/writehmlbse.f90 additionally warns that the BSE may fail for a
        metal.

        Returns {"energies": (nmbse,) complex Hartree, "workdir": Path}.
        The imaginary part is nonzero only when `bsefull` is set.
        """
        self._check_bse_states(ncbse)
        blocks = self._bse_blocks(
            nvbse, ncbse, bsefull, exchange, direct, gmaxrf, swidth,
            wmax=1.0, nwplot=500, components=None,
        )
        subdir = self._run_resumed(
            label, self._bse_tasks(direct, None), blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        energies = parsers_bse.parse_bse_eigenvalues(
            subdir / spec.OUTPUT_FILES["eigval_bse"]
        )
        return {"energies": energies, "workdir": subdir}

    def get_bse_dielectric(
        self, components=((1, 1),), wmax=0.5, nwplot=500, nvbse=2, ncbse=3,
        bsefull=False, exchange=True, direct=True, gmaxrf=None, swidth=0.001,
        ngridk=None, label="bse",
    ):
        """Excitonic dielectric function from the Bethe-Salpeter equation
        (tasks 120, 180, 185, 186, 187 -- src/dielectric_bse.f90,
        EPSILON_BSE_ij.OUT), plus the excitation energies of
        get_bse_excitations().

        Where task 121 (`get_dielectric_function`) sums independent
        electron-hole pairs, this sums BSE eigenstates: each excitation a
        contributes an oscillator strength built from the coherent
        superposition of its pair amplitudes,

            d_a = sum_{vck} (E_a / (e_ck - e_vk)) A^a_{vck} p_{vc,k},
            sigma_ij(w) = i occmax w_k / Omega sum_a
                          [ d^i_a conj(d^j_a) / E_a ]
                          [ 1/(w - E_a + i s) + c.c.-form ],
            eps_ij = delta_ij + 4 pi i sigma_ij / (w + i s).

        Beyond the independent-particle picture this adds two effects at
        once: oscillator strength is redistributed toward bound excitons
        below the gap, and the above-gap continuum is reshaped (the
        Sommerfeld enhancement).

        - `components`: (i, j) Cartesian PAIRS, 1-based -- Elk's `optcomp`,
          two indices here, unlike get_shg()'s three.
        - `wmax`: upper end of the photon-energy window (Hartree). As with
          task 125, src/dielectric_bse.f90 builds w(iw) =
          wmax/nwplot * (iw-1) starting at zero and ignoring wplot(1).
        - every other argument is get_bse_excitations()'s.

        Returns {"energies": (nw,) Hartree,
                 "epsilon": {(i,j): (nw,) complex},
                 "excitations": (nmbse,) complex Hartree,
                 "workdir": Path}.
        """
        components = [_cartesian_indices(c, 2) for c in components]
        self._check_bse_states(ncbse)
        blocks = self._bse_blocks(
            nvbse, ncbse, bsefull, exchange, direct, gmaxrf, swidth,
            wmax, nwplot, components,
        )
        subdir = self._run_resumed(
            label, self._bse_tasks(direct, spec.TASKS["bse_dielectric"]), blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        epsilon = {}
        energies = None
        for i, j in components:
            filename = spec.OUTPUT_FILE_TEMPLATES["epsilon_bse"].format(i=i, j=j)
            energies, values = parsers_bse.parse_epsilon_bse(subdir / filename)
            epsilon[(i, j)] = values
        excitations = parsers_bse.parse_bse_eigenvalues(
            subdir / spec.OUTPUT_FILES["eigval_bse"]
        )
        return {
            "energies": energies,
            "epsilon": epsilon,
            "excitations": excitations,
            "workdir": subdir,
        }

    def _check_bse_states(self, ncbse):
        nempty = self.extra_blocks.get("nempty")
        if nempty is None:
            raise ValueError(
                "the BSE needs empty states: set nempty on the Calculation "
                "(extra_blocks={'nempty': [n]}) with n >= ncbse. Elk's default "
                "(nempty0 = 4, readinput.f90:99) leaves src/genidxbse.f90 to "
                "hard-stop with 'not enough conduction states' on most cells"
            )
        if float(nempty[0]) < ncbse:
            raise ValueError(
                f"nempty={nempty[0]} is smaller than ncbse={ncbse}; "
                f"src/genidxbse.f90 needs at least ncbse states above the "
                f"topmost occupied band at every k-point"
            )

    # ------------------------------------------------------------------
    # task 320 -- linear-response TDDFT
    # ------------------------------------------------------------------

    def get_tddft_dielectric(
        self, components=((1, 1),), vecql=(0.0, 0.0, 0.0), fxc="RPA",
        wplot=(0.0, 0.5), nwplot=500, swidth=0.001, gmaxrf=None,
        fxclrc=None, ngridk=None, label="tddftlr",
    ):
        """Dielectric function from linear-response TDDFT (task 320,
        src/tddftlr.f90).

        Solves the Dyson equation for the density response including
        LOCAL FIELDS -- i.e. the full G, G' matrix structure task 121
        throws away:

            eps^-1 = 1 + v^1/2 chi v^1/2,
            chi    = [1 - v^1/2 chi0 v^1/2 - v^-1/2 f_xc v^-1/2 v chi0]^-1
                     chi0,

        with chi0 the Kohn-Sham (independent-particle) polarisability and
        f_xc the exchange-correlation kernel. Setting f_xc = 0 gives the
        RPA; an f_xc with the right long-range 1/q^2 behaviour is what
        makes TDDFT reproduce excitonic absorption at all.

        At q = 0 the head of the matrix is 3x3 (the three Cartesian
        directions of the q -> 0 limit) and the wings are 3 x ngrf, so
        the macroscopic dielectric tensor -- the measurable one -- is
        obtained by inverting the 3x3 head of eps^-1 alone, which is what
        EPSM_TDDFT_ij.OUT holds. At q = 0 the run additionally produces
        the Faraday rotation, a TDDFT Kerr angle and a magnetic linear
        dichroism spectrum.

        - `vecql`: the momentum transfer in lattice coordinates. Must be
          commensurate with `ngridk` (Elk hard-stops otherwise; checked
          here). q = 0 selects the optical limit and the indexed output
          files; a finite q gives the unindexed EPSILON_TDDFT.OUT and no
          macroscopic/magneto-optic output, since the head is then a
          scalar.
        - `fxc`: kernel name from FXC_KERNELS ("RPA", "ALDA", "LRC",
          "bootstrap", "bootstrap1") or a raw integer `fxctype` code.
          Elk's own default (-1) is not a valid kernel and makes
          src/genvfxc.f90 stop, so a choice is mandatory -- "RPA" here.
          libxc kernels (fxctype 100) are unavailable in this build
          (build-config/make.inc uses libxcifc_stub.f90).
        - `fxclrc`: (alpha, beta) for the LRC kernel, Elk's `fxclrc`.
        - `wplot`/`nwplot`: the frequency window. Unlike tasks 125/187
          this DOES honour wplot(1), and does not clip a negative one; but
          the output loop starts at iw = 2, so the returned grid has
          nwplot-1 points beginning one step above wplot(1).
        - `swidth`: the imaginary part added to every frequency.
        - `gmaxrf`: the |G| cut-off setting ngrf, i.e. how many local-field
          components are kept. The convergence parameter of the method.

        Returns, for q = 0,
        {"energies", "epsilon": {(i,j): ...}, "epsinv": {...},
         "epsm": {...}, "faraday", "kerr", "mld", "workdir"};
        for finite q, {"energies", "epsilon", "epsinv", "workdir"} with
        the two spectra as bare arrays rather than per-component dicts.
        All spectra complex; "kerr" is in DEGREES, "faraday"/"mld" in
        radians.
        """
        components = [_cartesian_indices(c, 2) for c in components]
        vecql = tuple(float(x) for x in vecql)
        self._check_commensurate_q(vecql, ngridk)
        blocks = {
            "vecql": [vecql],
            "fxctype": [self._fxctype_code(fxc)],
            "wplot": [(int(nwplot), 100, 1), (float(wplot[0]), float(wplot[1]))],
            "swidth": [float(swidth)],
            "optcomp": list(components),
        }
        if fxclrc is not None:
            blocks["fxclrc"] = [tuple(float(x) for x in fxclrc)]
        if gmaxrf is not None:
            blocks["gmaxrf"] = [float(gmaxrf)]
        subdir = self._run_resumed(
            label, [spec.TASKS["tddft_linear_response"]], blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        tq0 = all(abs(x) < 1e-8 for x in vecql)
        result = {"workdir": subdir}
        if not tq0:
            energies, eps = parsers_tddft.parse_tddft_response(
                subdir / spec.OUTPUT_FILES["epsilon_tddft_q"]
            )
            _, epsinv = parsers_tddft.parse_tddft_response(
                subdir / spec.OUTPUT_FILES["epsinv_tddft_q"]
            )
            result.update({"energies": energies, "epsilon": eps, "epsinv": epsinv})
            return result
        energies = None
        for key, template in (
            ("epsilon", "epsilon_tddft"),
            ("epsinv", "epsinv_tddft"),
            ("epsm", "epsm_tddft"),
        ):
            result[key] = {}
            for i, j in components:
                filename = spec.OUTPUT_FILE_TEMPLATES[template].format(i=i, j=j)
                energies, values = parsers_tddft.parse_tddft_response(subdir / filename)
                result[key][(i, j)] = values
        for key, filekey in (
            ("faraday", "faraday"), ("kerr", "kerr_tddft"), ("mld", "mld"),
        ):
            energies, values = parsers_tddft.parse_tddft_response(
                subdir / spec.OUTPUT_FILES[filekey]
            )
            result[key] = values
        result["energies"] = energies
        return result

    @staticmethod
    def _fxctype_code(fxc):
        if isinstance(fxc, int):
            return fxc
        try:
            return FXC_KERNELS[fxc]
        except KeyError:
            raise ValueError(
                f"unknown xc kernel {fxc!r}; use one of {sorted(FXC_KERNELS)} or a raw "
                f"integer fxctype code (src/genvfxc.f90's select case)"
            ) from None

    # ------------------------------------------------------------------
    # tasks 330/331 -- spin-polarised linear response
    # ------------------------------------------------------------------

    def get_spin_response(
        self, vecql=(0.0, 0.0, 0.0), fxc="ALDA", wplot=(0.0, 0.1), nwplot=500,
        swidth=0.001, gmaxrf=None, write_full=False, ngridk=None,
        label="tddftsplr",
    ):
        """Spin-dependent linear response chi_ij(G=G'=0; q, w) (tasks 330 /
        331, src/tddftsplr.f90) -- the magnon spectrum.

        chi is a 4x4 matrix in the (charge, m_x, m_y, m_z) basis:

            chi_00      density-density,        dn/dv
            chi_0j      density-magnetisation,  dn/dB_j
            chi_i0      magnetisation-density,  dm_i/dv
            chi_ij      magnetisation-magnetisation, dm_i/dB_j

        obtained from the Kohn-Sham chi0 by the Dyson equation
        chi = [1 - chi0 f_Hxc]^-1 chi0 with f_Hxc the Hartree plus
        spin-dependent xc kernel. The poles of the TRANSVERSE component
        chi_+- (m_+- = m_x +- i m_y), returned here as "chi_transverse",
        are the magnon energies at wavevector q; scanning q gives the
        magnon dispersion, the dynamical counterpart of the static
        exchange constants get_exchange_tensor() extracts.

        Requires a spin-polarised ground state -- src/tddftsplr.f90's very
        first statement stops on `.not.spinpol` -- which is enforced here
        before any run. The transverse channel exists only for a COLLINEAR
        magnet (`.not.ncmag`); for a non-collinear one Elk writes no
        CHI_T.OUT and this returns None for it.

        - `vecql`: magnon wavevector in lattice coordinates; must be
          commensurate with `ngridk`. q = 0 gives the uniform (Goldstone)
          mode, whose vanishing at zero energy is the standard check that
          the kernel and the ground state are consistent.
        - `fxc`: kernel name/code as in get_tddft_dielectric(); "ALDA" is
          the meaningful default here, since "RPA" (f_xc = 0) removes the
          Stoner exchange splitting that binds the magnon. Only "RPA" and
          "ALDA" are usable for this task: src/tddftsplr.f90 builds its
          kernel through src/genspfxcg.f90 -> src/genspfxcr.f90 ->
          fxcifc, whose `select case(abs(fxctype(1)))` accepts only 0/1,
          3 and 100 (libxc, stubbed out in this build) -- the "LRC" and
          "bootstrap" kernels exist only inside src/genvfxc.f90, which
          task 320 uses and this one does not.
        - `write_full`: use task 331 instead of 330, which additionally
          dumps the complete chi(G, G') to the binary CHI.OUT.
        - `wplot`/`nwplot`/`swidth`/`gmaxrf`: as in
          get_tddft_dielectric(). Note magnon energies are tens to
          hundreds of meV, i.e. a much narrower window than an optical
          spectrum -- hence the small default wplot.

        Unlike task 320, src/tddftsplr.f90 writes ALL nwplot frequencies
        (`do iw=1,nwrf`), so no row is dropped.

        Returns {"energies": (nwplot,) Hartree,
                 "chi": {(i,j): (nwplot,) complex} for i,j in 0..3,
                 "chi0": {(i,j): ...},
                 "chi_transverse", "chi0_transverse": (nwplot,) complex or
                 None, "workdir": Path}.
        """
        if not self.spinpol:
            raise ValueError(
                "get_spin_response() requires spinpol=True: src/tddftsplr.f90 stops "
                "immediately on a spin-unpolarised calculation, since a magnetisation "
                "response needs a magnetisation to respond"
            )
        vecql = tuple(float(x) for x in vecql)
        self._check_commensurate_q(vecql, ngridk)
        blocks = {
            "vecql": [vecql],
            "fxctype": [self._fxctype_code(fxc)],
            "wplot": [(int(nwplot), 100, 1), (float(wplot[0]), float(wplot[1]))],
            "swidth": [float(swidth)],
        }
        if gmaxrf is not None:
            blocks["gmaxrf"] = [float(gmaxrf)]
        task = spec.TASKS[
            "tddft_spin_response_full" if write_full else "tddft_spin_response"
        ]
        subdir = self._run_resumed(
            label, [task], blocks, ngridk=tuple(ngridk) if ngridk else None,
        )
        result = {"workdir": subdir, "chi": {}, "chi0": {}}
        energies = None
        for key, template in (("chi", "chi_spin"), ("chi0", "chi0_spin")):
            for i in range(4):
                for j in range(4):
                    filename = spec.OUTPUT_FILE_TEMPLATES[template].format(i=i, j=j)
                    energies, values = parsers_tddft.parse_spin_response(subdir / filename)
                    result[key][(i, j)] = values
        for key, filekey in (
            ("chi_transverse", "chi_transverse"),
            ("chi0_transverse", "chi0_transverse"),
        ):
            path = subdir / spec.OUTPUT_FILES[filekey]
            if path.is_file():
                energies, values = parsers_tddft.parse_spin_response(path)
                result[key] = values
            else:
                result[key] = None
        result["energies"] = energies
        return result

    # ------------------------------------------------------------------
    # tasks 450/455/456 -- the laser field
    # ------------------------------------------------------------------

    @staticmethod
    def _afield_blocks(tstime, dtimes, pulses, ramps, steps):
        """The `tstime`/`dtimes`/`pulse`/`ramp`/`step` blocks
        (src/readinput.f90 case('pulse'/'ramp'/'step'),
        src/genafieldt.f90 for the meaning of each number).

        Each block is a count followed by one line per entry. A short line
        is padded by readinput with ' 1.0 0.0 0.0 0.0', which supplies the
        default spin components (sigma_0 weight 1, no sigma_x/y/z), so an
        8-number pulse line is complete for a spin-independent field.
        """
        blocks = {"tstime": [float(tstime)], "dtimes": [float(dtimes)]}
        for name, entries, width, what in (
            ("pulse", pulses, (8, 12), "A0x A0y A0z omega phase chirp t0 fwhm"),
            ("ramp", ramps, (8, 12), "A0x A0y A0z t0 c1 c2 c3 c4"),
            ("step", steps, (5, 9), "A0x A0y A0z t_start t_stop"),
        ):
            entries = [tuple(float(x) for x in e) for e in (entries or ())]
            if not entries:
                continue
            for entry in entries:
                if len(entry) not in width:
                    raise ValueError(
                        f"each `{name}` entry needs {width[0]} numbers "
                        f"({what}), or {width[1]} to add the spin components "
                        f"(sigma_0, sigma_x, sigma_y, sigma_z weights); got {len(entry)}"
                    )
            blocks[name] = [len(entries)] + entries
        if not any(k in blocks for k in ("pulse", "ramp", "step")):
            raise ValueError(
                "no A-field specified: pass at least one of pulses=, ramps= or "
                "steps=. With npulse = nramp = nstep = 0 (Elk's defaults) "
                "src/genafieldt.f90 writes an identically zero A(t)"
            )
        return blocks

    def get_afield(
        self, pulses=(), ramps=(), steps=(), tstime=1000.0, dtimes=0.1,
        wplot=(0.0, 0.5), nwplot=500, swidth=0.001, label="afield",
    ):
        """The time-dependent laser vector potential A(t), its power
        density and the electric field spectrum E(w) -- tasks 450, 455 and
        456 (src/genafieldt.f90, src/writeafpdt.f90,
        src/writeefieldw.f90).

        A(t) is built as a superposition of Gaussian-enveloped, optionally
        chirped sinusoids, polynomial ramps and rectangular steps
        (src/genafieldt.f90's own docstring):

            A(t) = sum_i A0_i exp[-(t - t0_i)^2 / 2 sigma_i^2]
                          sin[w_i (t - t0_i) + phi_i + rc_i t^2 / 2],
            sigma = FWHM / (2 sqrt(2 ln 2)).

        The electrons see E(t) = -(1/c) dA/dt, so the peak power density
        is |A0|^2 w^2 / (8 pi c) -- returned time-resolved as
        "power_density" and integrated as "energy_density". A pulse whose
        A(t) returns to zero delivers no net momentum; one that does not
        (a ramp or a step) is a DC field.

        This is a pure post-processing chain: it needs the ground state
        only because every Elk task does, and costs milliseconds. Running
        it first is the cheap way to check that a pulse has the intended
        frequency content before committing to a real-time evolution.

        - `pulses`: sequence of (A0x, A0y, A0z, omega, phase_deg,
          chirp_rate, t0, fwhm), optionally extended by four spin
          components (sigma_0, sigma_x, sigma_y, sigma_z weights) to make
          the field spin-dependent -- which additionally requires
          `tafspt` on the Calculation.
        - `ramps`: (A0x, A0y, A0z, t_start, c1, c2, c3, c4), the field
          growing as c1 t + c2 t^2 + c3 t^3 + c4 t^4 after t_start.
        - `steps`: (A0x, A0y, A0z, t_start, t_stop), a constant A between
          the two times -- the delta-function E(t) used by task 481.
        - `tstime`/`dtimes`: total simulation time and time step, atomic
          units of time (1 a.u. = 24.189 as). ntimes = nint(tstime/dtimes)
          + 1 (src/gentimes.f90).
        - `wplot`/`nwplot`/`swidth`: the frequency window for E(w). A
          negative lower bound is clipped to zero
          (src/writeefieldw.f90), and `swidth` is the width of the
          Lorentzian convolution that suppresses the ringing from the
          finite time window.

        Returns {"times": (ntimes,), "afield": (ntimes, 3),
                 "power_density": (ntimes,), "energy_density": float,
                 "energies": (nwplot,), "efield_w": (nwplot, 3) complex,
                 "workdir": Path, "blocks": dict} -- all atomic units.
        """
        blocks = self._afield_blocks(tstime, dtimes, pulses, ramps, steps)
        blocks["wplot"] = [(int(nwplot), 100, 1), (float(wplot[0]), float(wplot[1]))]
        blocks["swidth"] = [float(swidth)]
        subdir = self._run_resumed(
            label,
            [spec.TASKS["afield"], spec.TASKS["afield_power"],
             spec.TASKS["efield_fourier"]],
            blocks,
        )
        times, afield = parsers_tddft.parse_afieldt(subdir / spec.OUTPUT_FILES["afieldt"])
        _, power = parsers_tddft.parse_afpdt(subdir / spec.OUTPUT_FILES["afpdt"])
        energy = parsers_tddft.parse_afted(subdir / spec.OUTPUT_FILES["afted"])
        energies, efield_w = parsers_tddft.parse_efieldw(
            subdir / spec.OUTPUT_FILES["efieldw"]
        )
        return {
            "times": times,
            "afield": afield,
            "power_density": power,
            "energy_density": energy,
            "energies": energies,
            "efield_w": efield_w,
            "workdir": subdir,
            "blocks": blocks,
        }

    # ------------------------------------------------------------------
    # tasks 460-463 -- real-time TDDFT
    # ------------------------------------------------------------------

    def _check_tddft_ready(self):
        """src/tddft.f90's very first statement is

            if (tshift) then ... 'use tshift = .false. for the
            ground-state run' ... stop

        `tshift` is Elk's automatic relocation of the origin to the
        crystal's inversion/symmetry centre; the time-dependent code needs
        the ground state and the A-field to share one frame, so it must be
        off for the ENTIRE Calculation, not just this call -- which is why
        it is checked on `extra_blocks` rather than accepted as an
        argument.
        """
        tshift = self.extra_blocks.get("tshift")
        if tshift is None or bool(tshift[0]):
            raise ValueError(
                "real-time TDDFT requires tshift=.false. on the Calculation itself "
                "(extra_blocks={'tshift': [False]}), because the ground state must "
                "have been computed in the unshifted frame -- src/tddft.f90 stops on "
                "the first line otherwise"
            )

    def get_tddft_evolution(
        self, pulses=(), ramps=(), steps=(), tstime=100.0, dtimes=0.1,
        ntswrite=None, ehrenfest=False, ngridk=None, label="tddft",
    ):
        """Real-time TDDFT: propagate the Kohn-Sham orbitals under a laser
        A-field and record the observables (tasks 450 then 460, or 462 for
        Ehrenfest nuclear dynamics -- src/tddft.f90, src/writetddft.f90).

        Each time step evolves |psi_j(t)> with the time-dependent
        Kohn-Sham Hamiltonian built from the instantaneous density and
        magnetisation and from the vector potential entering through the
        minimal coupling (p + A/c)^2 / 2; the density is rebuilt, the
        potential recomputed, and the total energy and the total current

            J(t) = -(1/Omega) integral j(r, t) dr

        written every step. J(t) is the observable everything else is
        derived from: its Fourier transform against E(w) gives the
        dielectric function (get_tddft_rt_dielectric()), and its harmonics
        are the high-harmonic spectrum.

        THIS IS EXPENSIVE. Cost is ntimes = nint(tstime/dtimes) + 1
        propagation steps, each comparable to one SCF iteration, with no
        convergence shortcut -- a realistic pulse is thousands of steps.
        Start with a short `tstime` and check the energy is conserved when
        the field is off before committing.

        - `pulses`/`ramps`/`steps`/`tstime`/`dtimes`: as in get_afield();
          task 450 runs first in the same directory so its AFIELDT.OUT is
          what src/init0.f90 reads back.
        - `ntswrite`: (every, from) -- write the optional per-step
          quantities every `every` steps starting at step `from`, Elk's
          `ntswrite` block (default 500, 1). The always-written scalars
          (energy, current, moments) are unaffected.
        - `ehrenfest`: use task 462 instead of 460, letting the nuclei
          move under the forces of the evolving electron density
          (Ehrenfest dynamics). Elk zeroes the displacements/velocities at
          the start and reads back the forces of the PREVIOUS run
          (src/tdinit.f90's readforcet), so a meaningful Ehrenfest run is
          the second of a pair.
        - `ngridk`: optionally overrides the k-mesh for this call only.
          Note src/tdinit.f90 re-reduces the mesh with only those
          symmetries that leave A(t) invariant, so a polarised field
          effectively uses more k-points than the ground state did.

        Requires tshift=.false. on the Calculation (see
        _check_tddft_ready).

        Returns {"times", "energy", "current" (ntimes-1, 3),
                 "current_magnitude", "interstitial_charge",
                 "muffin_tin_charge", "moment", "induced_afield",
                 "afield", "workdir", "blocks", "restart_task"} -- entries
        that the run did not produce (moments without spinpol, the induced
        field without `tafindt`) come back None.
        """
        self._check_tddft_ready()
        blocks = self._afield_blocks(tstime, dtimes, pulses, ramps, steps)
        if ntswrite is not None:
            blocks["ntswrite"] = [tuple(int(x) for x in ntswrite)]
        task = spec.TASKS["tddft_ehrenfest" if ehrenfest else "tddft"]
        subdir = self._run_resumed(
            label, [spec.TASKS["afield"], task], blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        result = self._read_tddft_observables(subdir)
        result["workdir"] = subdir
        result["blocks"] = blocks
        result["ngridk"] = tuple(ngridk) if ngridk else None
        result["restart_task"] = spec.TASKS[
            "tddft_ehrenfest_restart" if ehrenfest else "tddft_restart"
        ]
        return result

    @staticmethod
    def _read_tddft_observables(subdir):
        subdir = Path(subdir)

        def series(filekey, ncomponents=None):
            path = subdir / spec.OUTPUT_FILES[filekey]
            if not path.is_file():
                return None, None
            return parsers_tddft.parse_time_series(path, ncomponents)

        times, energy = series("totenergy_td", 1)
        _, current = series("jtot_td", 3)
        _, current_magnitude = series("jtotm_td", 1)
        _, chgir = series("chargeir_td", 1)
        _, moment = series("moment_td")
        _, afind = series("afind_td", 3)
        result = {
            "times": times,
            "energy": energy,
            "current": current,
            "current_magnitude": current_magnitude,
            "interstitial_charge": chgir,
            "moment": moment,
            "induced_afield": afind,
        }
        chgmt_path = subdir / spec.OUTPUT_FILES["chargemt_td"]
        if chgmt_path.is_file():
            mt_times, mt_values, mt_labels = parsers_tddft.parse_atom_time_series(chgmt_path)
            result["muffin_tin_charge"] = {
                "times": mt_times, "values": mt_values, "atoms": mt_labels,
            }
        else:
            result["muffin_tin_charge"] = None
        afieldt_path = subdir / spec.OUTPUT_FILES["afieldt"]
        if afieldt_path.is_file():
            a_times, afield = parsers_tddft.parse_afieldt(afieldt_path)
            result["afield"] = {"times": a_times, "afield": afield}
        else:
            result["afield"] = None
        return result

    def continue_tddft_evolution(self, previous, tstime=None):
        """Restart a real-time run (task 461, or 463 for Ehrenfest) in the
        directory it already occupies.

        This is the one method here that does NOT go through
        `_run_resumed`: that helper WIPES its subdirectory before running,
        which would destroy exactly the files a restart needs -- the
        `_TD.OUT` eigenvectors, TIMESTEP.OUT (read by src/tdrestart.f90's
        readtimes to find the last completed step) and the APPEND-mode
        observable files. It therefore rewrites elk.in in place and calls
        the launcher directly.

        Task 450 is re-run first: src/readafieldt.f90 does
        `ntimes = min(ntimes, ntimes_)`, so extending `tstime` without
        regenerating AFIELDT.OUT would silently truncate the evolution
        back to the original length instead of extending it.

        - `previous`: the dict returned by get_tddft_evolution() (its
          "workdir", "blocks" and "restart_task" entries are used).
        - `tstime`: a new total simulation time; the run picks up at the
          step after the last one already written.

        Returns the same shape as get_tddft_evolution(). Note the
        observable files are opened with position='APPEND' and deleted
        only at itimes <= 1 (src/writetddft.f90), so the returned series
        span the whole evolution, original run included.
        """
        subdir = Path(previous["workdir"])
        if not (subdir / spec.OUTPUT_FILES["state"]).is_file():
            raise ValueError(
                f"{subdir} does not look like a completed TDDFT run directory "
                f"(no STATE.OUT); pass the dict get_tddft_evolution() returned"
            )
        blocks = dict(previous["blocks"])
        if tstime is not None:
            blocks["tstime"] = [float(tstime)]
        f = InputFile()
        f.add_block("tasks", [spec.TASKS["afield"], previous["restart_task"]])
        # the original run's k-mesh override must be reproduced exactly:
        # src/tdrestart.f90 picks up the `_TD.OUT` eigenvectors written for
        # that k-set, and silently reverting to self.ngridk would restart
        # from wavefunctions belonging to a different mesh
        self._add_base_blocks(f, ngridk=previous.get("ngridk"))
        for name, lines in blocks.items():
            f.add_block(name, lines)
        f.write(subdir / "elk.in")
        self.launcher.run(subdir)
        result = self._read_tddft_observables(subdir)
        result["workdir"] = subdir
        result["blocks"] = blocks
        result["ngridk"] = previous.get("ngridk")
        result["restart_task"] = previous["restart_task"]
        return result

    # ------------------------------------------------------------------
    # tasks 480/481 -- dielectric function from the real-time current
    # ------------------------------------------------------------------

    def get_tddft_rt_dielectric(
        self, pulses=(), ramps=(), steps=(), tstime=100.0, dtimes=0.1,
        wplot=(0.0, 0.5), nwplot=500, swidth=0.001, delta_kick=False,
        remove_drude=False, ehrenfest=False, ngridk=None, label="tddft_rt",
    ):
        """Dielectric tensor from a real-time evolution (tasks 450, 460
        then 480 or 481 -- src/dielectric_tdrt.f90).

        The full non-linear time propagation is run once and then linearly
        analysed: Ohm's law in frequency space,

            eps_ij(w) = delta_ij + 4 pi i J_i(w) / [(w + i s) E_j(w)],

        with J(w) the Fourier transform of the total current
        get_tddft_evolution() records and E(w) that of
        -(1/c) dA/dt. Because it comes from a real propagation it contains
        the whole response -- local fields, the xc kernel actually used and
        any non-linearity the pulse excites -- not just an
        independent-particle sum. Only components not orthogonal to the
        applied field are meaningful; src/dielectric_tdrt.f90 zeroes a
        component whose E_j(w) is below 1e-8 rather than dividing by it.

        - `delta_kick`: use task 481 instead of 480. Task 481 assumes E(t)
          is a SINGLE delta function at t = 0 and takes E(w) = -A(0)/c, a
          constant, without transforming A(t) at all -- the standard
          "kick" protocol. That requires the field to be a `step` starting
          at t = 0 and running to the END of the simulation (t_stop >=
          tstime): a step that switches off partway through is two kicks
          of opposite sign, which task 481 has no way to know about and
          silently mis-analyses. Task 480 instead Fourier-transforms A(t)
          numerically, which is what a finite pulse needs.
        - `remove_drude`: Elk's `jtconst0`. Subtracts the time-averaged
          constant part of J(t), which removes the Drude (free-carrier)
          term and leaves the interband response.
        - every other argument as in get_tddft_evolution()/get_afield().

        Note that all NINE tensor components are always written --
        src/dielectric_tdrt.f90 loops i, j = 1..3 unconditionally and
        never consults `optcomp`, unlike every other member of this
        family.

        Returns {"energies": (nwplot,) Hartree,
                 "epsilon": {(i,j): (nwplot,) complex} for all 9 (i,j),
                 "current_w": (nwplot, 3) complex,
                 "evolution": the get_tddft_evolution() dict,
                 "workdir": Path}.
        """
        self._check_tddft_ready()
        blocks = self._afield_blocks(tstime, dtimes, pulses, ramps, steps)
        blocks["wplot"] = [(int(nwplot), 100, 1), (float(wplot[0]), float(wplot[1]))]
        blocks["swidth"] = [float(swidth)]
        if remove_drude:
            blocks["jtconst0"] = [True]
        evolve = spec.TASKS["tddft_ehrenfest" if ehrenfest else "tddft"]
        analyse = spec.TASKS[
            "dielectric_tdrt_delta" if delta_kick else "dielectric_tdrt"
        ]
        subdir = self._run_resumed(
            label, [spec.TASKS["afield"], evolve, analyse], blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        epsilon = {}
        energies = None
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                filename = spec.OUTPUT_FILE_TEMPLATES["epsilon_tdrt"].format(i=i, j=j)
                energies, values = parsers_tddft.parse_epsilon_tdrt(subdir / filename)
                epsilon[(i, j)] = values
        _, current_w = parsers_tddft.parse_jtotw(subdir / spec.OUTPUT_FILES["jtotw"])
        evolution = self._read_tddft_observables(subdir)
        evolution["workdir"] = subdir
        evolution["blocks"] = blocks
        evolution["ngridk"] = tuple(ngridk) if ngridk else None
        evolution["restart_task"] = spec.TASKS[
            "tddft_ehrenfest_restart" if ehrenfest else "tddft_restart"
        ]
        return {
            "energies": energies,
            "epsilon": epsilon,
            "current_w": current_w,
            "evolution": evolution,
            "workdir": subdir,
        }

    # ------------------------------------------------------------------
    # task 285 -- anomalous correlation entropy
    # ------------------------------------------------------------------

    def get_anomalous_entropy(
        self, ngridq, prerequisite_tasks=None, nrmtscf=4, lmaxi=2,
        extra_blocks=None, label="aceplot",
    ):
        """Fermionic and bosonic anomalous correlation entropy on the k-
        and q-meshes (task 285, src/aceplot.f90, FACE3D.OUT/BACE3D.OUT).

        For a superconducting (Bogoliubov) state the occupation of a
        quasiparticle level is no longer 0 or 1, and the entropy-like
        measure

            S_k = -occmax sum_n [ v_n ln v_n + (1 - v_n) ln(1 - v_n) ]

        (with v_n the anomalous norm from the Bogoliubov transformation)
        quantifies how strongly that k-point participates in the pairing;
        the bosonic counterpart does the same for the phonon squeezing,
        S_q = -sum_i [ x_i ln x_i - (1 + x_i) ln(1 + x_i) ].

        THIS IS THE EXPENSIVE OUTLIER OF THIS MODULE and is wrapped, not
        validated. src/initeph.f90 reads EVALUV.OUT/EVECUV.OUT via
        getevaluv/getevecuv, which only the electron-phonon
        superconductivity ground state (task 270/271, src/gndsteph.f90)
        writes; that in turn needs the electron-phonon matrix elements and
        those need the DFPT phonons (task 205). The default
        `prerequisite_tasks` is that chain -- (205, 241, 270) -- and running
        it is hours to days of compute on anything but a toy cell; the
        phonon step alone is measured at 11-13 minutes for 2-atom Si on a
        2x2x2 q-mesh (see CLAUDE.md).

        All four tasks run in ONE invocation, and `prerequisite_tasks` is
        there to change the chain -- task 200 in place of 205 for supercell
        rather than DFPT phonons, say -- not to shorten it. It cannot pick up
        where an earlier call left off: like every `get_*` here this one runs
        in a subdirectory that is wiped first (docs/design.md #4), so a
        shortened chain finds none of the files it means to reuse and aborts
        inside the Fortran exactly as a missing EPHMAT.OUT does.

        The coupling step is **241, not 240**, and the difference is not the
        band window the two task numbers suggest: src/ephcouple.f90:131 is
        `if (task == 241) call putephmat(iq,ik,ephmat)` and is the only
        putephmat call site in the tree, so task 240 writes no EPHMAT.OUT at
        all. src/initeph.f90:92-100 then calls getephmat for every (q,k),
        which opens EPHMAT.OUT as an UNFORMATTED DIRECT file -- created empty
        if absent -- and reads a record, i.e. a Fortran end-of-file abort
        after the whole phonon run has already been paid for.

        `lmaxi` >= 2 is likewise a prerequisite of task 205 rather than an
        accuracy knob: src/phonon.f90:34 stops with "lmaxi too small for
        calculating DFPT phonons", and readinput.f90:84's default is 1, so
        the blocks come from the phonon family's own `_phonon_blocks()`
        rather than from a second copy of the same rule here.

        `ngridq` is REQUIRED rather than defaulted: task 205 is not on
        src/init2.f90:44-47's list of tasks that force ngridq = ngridk, so
        Elk's own default would run Gamma-point-only phonons and produce a
        bosonic entropy that means nothing. It must divide `ngridk`
        (src/init2.f90:58 checks mod(ngridk, ngridq) and stops otherwise),
        the same constraint get_phonon_dos() carries.

        Returns {"kpoints": (nk, 3) Cartesian, "fermionic": (nk,),
                 "qpoints": (nq, 3) Cartesian, "bosonic": (nq,),
                 "kgrid": (3,), "qgrid": (3,), "workdir": Path}.
        """
        ngridq = tuple(int(n) for n in ngridq)
        self._check_eph_commensurate(ngridq)
        if prerequisite_tasks is None:
            prerequisite_tasks = (
                spec.TASKS["phonon_dfpt"],
                spec.TASKS["ephcouple_write"],
                spec.TASKS["gndsteph"],
            )
        blocks = self._phonon_blocks(ngridq, nrmtscf=nrmtscf, lmaxi=lmaxi)
        blocks.update(extra_blocks or {})
        tasks = list(prerequisite_tasks) + [spec.TASKS["anomalous_entropy"]]
        subdir = self._run_resumed(label, tasks, blocks)
        kgrid, kpoints, fermionic = _parse_ace(subdir / spec.OUTPUT_FILES["face_3d"])
        qgrid, qpoints, bosonic = _parse_ace(subdir / spec.OUTPUT_FILES["bace_3d"])
        return {
            "kpoints": kpoints, "fermionic": fermionic, "kgrid": kgrid,
            "qpoints": qpoints, "bosonic": bosonic, "qgrid": qgrid,
            "workdir": subdir,
        }


def _parse_ace(path):
    """FACE3D.OUT / BACE3D.OUT (src/aceplot.f90):

        write(50,'(3I6," : grid size")') ngridk(:)
        do i3, i2, i1:  write(50,'(4G18.10)') vkc(:,ik), ace

    with i1 running fastest. Returns (grid, points, values).
    """
    grid = None
    points = []
    values = []
    with open(path) as fh:
        for line in fh:
            tokens = line.split()
            if not tokens:
                continue
            if grid is None:
                grid = tuple(int(t) for t in tokens[:3])
                continue
            points.append([float(t) for t in tokens[:3]])
            values.append(float(tokens[3]))
    return grid, np.array(points), np.array(values)
