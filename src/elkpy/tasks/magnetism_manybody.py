"""Magnetism, many-body GW, Wannier90 export and the ultra-long-range family.

This mixin covers the Elk task codes

* **28/29** -- magnetic anisotropy energy (``vendor/elk/src/mae.f90``)
* **160** -- exchange-correlation torque (``vendor/elk/src/torque.f90``)
* **350/351/352** -- supercell spin spirals (``vendor/elk/src/spiralsc.f90``),
  plus the *generalised Bloch* spin spiral, which is not a task at all but the
  ``spinsprl``/``vqlss`` input pair on an ordinary ground state
* **550** -- Wannier90 export (``vendor/elk/src/writew90.f90``)
* **600/601/610/620/630/640** -- the GW family (``gwsefm``, ``gwspecf``,
  ``gwbandstr``, ``writegwefm``, ``gwdmat``)
* **700/701/710/720/725/731-733/741-743/771-773** -- the ultra-long-range
  (ULR) family (``gndstulr``, ``writedosu``, ``bandstrulr``, ``rhouplot``,
  ``potuplot``, ``maguplot``)

Two structural notes, both consequences of reading the Fortran rather than
the manual.

**Chaining without re-running the expensive step.**
``Calculation._run_resumed`` wipes its subdirectory and prepends task 1, which
is exactly right for *producing* ``GWSEFM.OUT`` or ``STATE_ULR.OUT`` but wrong
for *consuming* them: task 610 reads the self-energy task 600 spent hours
writing, and tasks 710/720/731... read the ultracell state task 700 converged.
Those dependents do their own ``init0``/``init1`` and read what is already on
disk, so they are run **in place** by :meth:`_run_dependent` -- same directory,
no wipe, no task 1 -- with the producing run's own input blocks replayed from
a small JSON sidecar so the k-set and ultracell definition cannot silently
drift between the two invocations.

**Everything here is expensive.**  A GW self-energy is hours to days on a real
system, ``mae`` runs one full SCF cycle *per magnetisation direction*, and
``spiralsc`` runs one full SCF cycle per q-point on a supercell. None of these
are the seconds-to-minutes that a ``get_*`` name might suggest.

Every method taking a real-frequency window writes Elk's ``wplot`` block,
whose first line is ``nwplot, ngrkf, nswplot`` (src/readinput.f90
``case('wplot','dos')``) and second line the window itself: `nwplot` grid
points, `ngrkf` the fine k-grid used for Brillouin-zone interpolation
(``nsk = max(ngrkf/ngridk, 1)``) and `nswplot` the number of smoothing passes
applied afterwards. Elk's own GW examples use ``800 100 0`` -- no smoothing,
since the analytic continuation has already broadened everything -- which is
the default here for the GW methods; the ULR DOS keeps Elk's default of one
smoothing pass.
"""

import json
import shutil
import warnings
from pathlib import Path

import numpy as np

from ..inputfile import InputFile
from ..parsers import gw as parsers_gw
from ..parsers import magnetism as parsers_magnetism
from ..parsers import ulr as parsers_ulr
from ..parsers import volumetric
from ..parsers import wannier90 as parsers_w90

# --- task codes -------------------------------------------------------------
# Duplicated as module constants rather than read from elkpy.spec so that this
# module imports cleanly before the integrator adds the matching spec.TASKS
# entries. The same values are returned as `spec_tasks` for that addition.
TASK_MAE = 28  # start from atomic densities
TASK_MAE_RESUME = 29  # read STATE.OUT (mae.f90: trdstate = (task == 29))
TASK_TORQUE = 160
TASK_SPIRAL_SUPERCELL = 350
TASK_SPIRAL_SUPERCELL_RESUME = 351
TASK_SPIRAL_SUPERCELL_DRYRUN = 352
TASK_WANNIER90 = 550
TASK_GW_SELF_ENERGY = 600
TASK_GW_SELF_ENERGY_KEEP_EPSINV = 601
TASK_GW_SPECTRAL_FUNCTION = 610
TASK_GW_BAND_STRUCTURE = 620
TASK_GW_FERMI_ENERGY = 630
TASK_GW_DENSITY_MATRIX = 640
# Blocks that EPSINV.OUT is built with, and so cannot change when task 601
# reuses it. `wmaxgw`/`tempk` set the number of Matsubara frequencies
# (genwgw.f90: nwgw = 2*nint(wmaxgw/(pi*kB*tempk)), then nwrf = nwbs+1) and
# `gmaxrf` the number of response G-vectors (init3.f90's ngrf loop); both are
# record dimensions, and getcfgq.f90:54-71 stops with "differing ng"/"differing
# m". `ngridq` changes the q-set getcfgq indexes records by, caught the same way
# ("differing vectors"). `nempty` is the dangerous one: it changes the states
# epsinv is BUILT from without changing any dimension, so Elk reads the stale
# file happily and returns a wrong self-energy.
GW_EPSINV_PINNED = ("wmaxgw", "tempk", "gmaxrf", "nempty", "ngridq")

# The ULR state file holds the Q-space density and potential of ONE ultracell.
# readstulr.f90 checks only unit-cell shapes (natmtot, npcmtmax, ngtc, ngtot,
# ndmag, fsmtype), so a changed ultracell is read silently and means nothing.
# `ngridq` is deliberately NOT here: readstulr.f90:114-127 maps the file's own
# Q-vectors onto the new grid and zeroes the rest, so restarting on a larger
# Q-grid is a supported use, not a mismatch.
ULR_STATE_PINNED = ("avecu", "scaleu")

TASK_ULR_GROUND_STATE = 700
TASK_ULR_GROUND_STATE_RESUME = 701
TASK_ULR_DOS = 710
TASK_ULR_BANDS = 720  # kappa = 0 only
TASK_ULR_BANDS_ALL_KAPPA = 725
TASK_ULR_DENSITY = {1: 731, 2: 732, 3: 733}
TASK_ULR_POTENTIAL = {1: 741, 2: 742, 3: 743}
TASK_ULR_MAGNETISATION = {1: 771, 2: 772, 3: 773}

# --- output filenames -------------------------------------------------------
FILE_MAE = "MAE.OUT"
FILE_MAE_PER_VOLUME = "MAEPUV.OUT"
FILE_MAE_INFO = "MAE_INFO.OUT"
FILE_GW_SELF_ENERGY = "GWSEFM.OUT"
FILE_GW_EPSINV = "EPSINV.OUT"
FILE_GW_TOTAL_SPECTRAL_FUNCTION = "GWTSF.OUT"
FILE_GW_BAND = "GWBAND.OUT"
FILE_GW_FERMI_ENERGY = "GWEFERMI.OUT"
FILE_ULR_STATE = "STATE_ULR.OUT"
FILE_ULR_INFO = "ULR_INFO.OUT"
FILE_ULR_RMSDVS = "RMSDVS.OUT"
FILE_ULR_TDOS = "TDOSULR.OUT"
FILE_ULR_BAND = "BANDULR.OUT"
FILE_ULR_BAND_SPECTRAL = "BANDSFU.OUT"
FILE_ULR_DENSITY = {1: "RHOU1D.OUT", 2: "RHOU2D.OUT", 3: "RHOU3D.OUT"}
FILE_ULR_DENSITY_LINES = "RHOULINES.OUT"
FILE_ULR_POTENTIAL = {1: "VSU1D.OUT", 2: "VSU2D.OUT", 3: "VSU3D.OUT"}
FILE_ULR_POTENTIAL_LINES = "VSULINES.OUT"
FILE_ULR_MAGNETISATION = {1: "MAGU1D.OUT", 2: "MAGU2D.OUT", 3: "MAGU3D.OUT"}
FILE_ULR_MAGNETISATION_LINES = "MAGULINES.OUT"

GWSF_TEMPLATE = "GWSF_K{ik:06d}.OUT"

# vendor/elk/src/w90_stub.f90's own message, printed immediately before its
# `error stop`. Matched verbatim so that a genuine task-550 failure is not
# mistaken for the expected missing-library abort.
W90_STUB_MESSAGE = "Error(wannier_setup): libwannier not or improperly installed"

_STAGE_MANIFEST = ".elkpy_stage.json"


class _RawToken:
    """A value that :func:`elkpy.inputfile._format_value` must NOT quote.

    ``InputFile`` wraps every ``str`` in single quotes, which is right for
    ``sppath``/``seedname`` (Elk list-direct-reads those into a character
    variable) but wrong for the handful of blocks Elk reads as a *raw line*
    and then re-parses itself -- ``wann_bands``/``idxw90`` (``read(50,'(A)')``
    then ``numlist``) and ``xlwin`` (verbatim text copied into the ``.win``
    file). ``_format_value`` falls through to ``str(v)`` for any other type,
    so this passes the text through untouched.
    """

    __slots__ = ("text",)

    def __init__(self, text):
        self.text = str(text)

    def __str__(self):
        return self.text

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"_RawToken({self.text!r})"


class MagnetismManyBodyTasks:
    """``get_*`` methods for tasks 28-29, 160, 350-352, 550, 600-640 and
    700-773. Mixed into :class:`elkpy.calculation.Calculation`."""

    # -----------------------------------------------------------------
    # shared plumbing
    # -----------------------------------------------------------------

    def _run_dependent(self, subdir, tasks, extra_blocks=None):
        """Run further task(s) **inside an existing** run directory.

        Unlike ``_run_resumed`` this neither wipes the directory nor prepends
        task 1: the whole point is to consume files a previous, expensive task
        left there (``GWSEFM.OUT``, ``STATE_ULR.OUT``, the per-k ``EVALU``
        files). Every one of these dependent tasks calls ``init0``/``init1``
        itself and then reads what it needs from disk.

        The producing run's input blocks are replayed from the JSON sidecar
        written by :meth:`_write_stage_manifest`, so the k-set, ``ngridq``,
        ``avecu`` and friends are guaranteed identical between the two
        invocations -- getting those wrong does not error, it silently reads
        a mismatched file.
        """
        subdir = Path(subdir)
        if not subdir.is_dir():
            raise FileNotFoundError(f"no such run directory: {subdir}")
        stage = self._read_stage_manifest(subdir)
        blocks = dict(stage.get("blocks", {}))
        blocks.update(extra_blocks or {})
        f = InputFile()
        f.add_block("tasks", list(tasks))
        self._add_base_blocks(
            f,
            ngridk=tuple(stage["ngridk"]) if stage.get("ngridk") else None,
            vkloff=tuple(stage["vkloff"]) if stage.get("vkloff") else None,
        )
        for name, lines in blocks.items():
            f.add_block(name, lines)
        f.write(subdir / "elk.in")
        self.launcher.run(subdir)
        return subdir

    @staticmethod
    def _write_stage_manifest(subdir, blocks, ngridk=None, vkloff=None):
        payload = {
            "blocks": {k: v for k, v in (blocks or {}).items()},
            "ngridk": list(ngridk) if ngridk else None,
            "vkloff": list(vkloff) if vkloff else None,
        }
        with open(Path(subdir) / _STAGE_MANIFEST, "w") as fh:
            json.dump(payload, fh, default=str)

    @staticmethod
    def _read_stage_manifest(subdir):
        path = Path(subdir) / _STAGE_MANIFEST
        if not path.exists():
            return {}
        with open(path) as fh:
            return json.load(fh)

    @staticmethod
    def _normalise_block(value):
        """A block's value in a form two calls can be compared in.

        The stage sidecar is JSON, so every tuple in it comes back a list and
        anything exotic came back through ``default=str``. Round-tripping the
        in-memory value the same way is the only comparison that cannot report
        a spurious mismatch between ``(2, 2, 2)`` and ``[2, 2, 2]``.
        """
        return json.loads(json.dumps(value, default=str))

    def _run_reusing(self, label, tasks, blocks, required, pinned, ngridk=None):
        """Run task(s) **in** an existing run directory, reusing a file in it.

        The restart-style tasks -- 601 (skip ``epsinv``, read the existing
        ``EPSINV.OUT``) and 701 (restart from ``STATE_ULR.OUT``) -- exist to
        READ a file a previous, expensive call left in this same directory.
        Dispatching them through :meth:`Calculation._run_resumed` cannot work
        and did: its ``shutil.rmtree`` is unconditional, so the file was
        deleted before ``elk.in`` was written. The task was therefore
        unreachable under ANY label -- a fresh label gives an empty directory
        and the identical Fortran abort -- and, under the label it was given,
        additionally destroyed hours of prior work.

        `required` is the filename the task exists to read: a precondition
        here, so the failure is a Python ``ValueError`` naming the file rather
        than a Fortran end-of-file abort inside ``getcfgq``/``readstulr``.

        `pinned` names the blocks that must be IDENTICAL to the producing
        run's, because the file was built with them. Elk catches some of these
        itself (``getcfgq`` stops on a differing ``ng``/``m``) and misses
        others entirely -- so the check is here, on every one of them, and
        names the block that moved. Blocks outside `pinned` are free to
        change; that is the whole point of a restart.

        Returns ``(subdir, merged_blocks, ngridk)``. The merged blocks and the
        producing run's mesh are what the caller must record in the stage
        sidecar, so that a later dependent task (610, 710, ...) replays what
        actually ran rather than this call's half of it.
        """
        subdir = self.workdir / label
        if not (subdir / required).is_file():
            raise ValueError(
                f"no {required} in {subdir}, which this task exists to read. "
                f"Run the producing task first, under the same label "
                f"(label={label!r}), and do not delete its directory -- unlike "
                "every other get_* here, this one runs IN that directory "
                "rather than in a wiped copy, because wiping it would delete "
                f"{required}."
            )
        stage = self._read_stage_manifest(subdir)
        old = stage.get("blocks", {})
        for name in pinned:
            was, now = old.get(name), (blocks or {}).get(name)
            if self._normalise_block(was) != self._normalise_block(now):
                raise ValueError(
                    f"{required} in {subdir} was built with {name}={was!r}, "
                    f"but this call asks for {name}={now!r}. The file is only "
                    f"meaningful for the {name} it was built with, so it "
                    "cannot be reused here: either pass the original value, "
                    "or re-run the producing task from scratch under a "
                    "different label."
                )
        stage_ngridk = tuple(stage["ngridk"]) if stage.get("ngridk") else None
        if ngridk is not None and tuple(ngridk) != stage_ngridk:
            raise ValueError(
                f"{required} in {subdir} was built on ngridk={stage_ngridk}, "
                f"but this call asks for ngridk={tuple(ngridk)}. The k-set is "
                "part of the file; _run_dependent replays the producing run's "
                "mesh and would silently ignore this one."
            )
        merged = dict(old)
        merged.update(blocks or {})
        self._run_dependent(subdir, tasks, blocks)
        return subdir, merged, stage_ngridk

    def _fresh_subdir(self, label, keep=False):
        subdir = self.workdir / label
        if not keep:
            shutil.rmtree(subdir, ignore_errors=True)
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    # ``_block_floats()``, ``_block_flag()`` and ``_ndmag()`` were moved
    # to Calculation during integration: the spectra family transcribed
    # the same src/init0.f90 rule independently, so it now has one home.
    # See Calculation._ndmag.

    # -----------------------------------------------------------------
    # 28/29 -- magnetic anisotropy energy
    # -----------------------------------------------------------------

    def get_mae(
        self,
        npmae=-1,
        from_scratch=False,
        epspot=1e-8,
        epsengy=1e-6,
        socscf=None,
        ngridk=None,
        extra_blocks=None,
        label="mae",
    ):
        """Magnetic anisotropy energy (tasks 28/29, src/mae.f90).

        The MAE is the energy cost of rotating the magnetisation away from
        its easy axis. ``mae.f90`` estimates it by brute force: it fixes the
        moment direction at each of ``npmae`` points, converges a **complete
        ground state at each one**, and reports the spread

        .. math:: \\Delta E = \\max_i E(\\hat m_i) - \\min_i E(\\hat m_i).

        The implementation detail worth knowing is that it does not rotate
        the moment at all -- it rotates the *lattice* by the inverse rotation
        (``axangrot``/``r3mm`` on ``avec``) while holding the magnetisation
        along ``+z`` with ``cmagz=.true.``, which keeps the collinear
        machinery valid and avoids re-deriving the symmetry group for a
        tilted moment. Spin-orbit coupling is what couples the moment to the
        lattice in the first place, so ``mae.f90`` forces ``spinorb=.true.``
        regardless of this ``Calculation``'s own setting, and any fixed-spin-
        moment constraint is switched off (``fsmtype=0``). It also starts
        from a large ``bfieldc = (0,0,-1)`` damped by ``reducebf=0.85`` each
        SCF loop, to break the symmetry cleanly at every direction.

        **Cost**: ``npmae`` full self-consistent ground states, each with SOC
        and a symmetry-reduced (i.e. small) point group. This is the single
        most expensive method in this mixin apart from GW.

        Parameters
        ----------
        npmae
            How the direction set is built (src/gentpmae.f90, and Elk's
            ``npmae`` block):

            * ``>= 4`` -- that many points spread evenly over the sphere
              (``sphcover``);
            * ``-1`` to ``-4`` -- the symmetry-reduced cardinal directions of
              the ``(2|npmae|+1)^3`` integer lattice vectors, the cheapest
              physically meaningful choice and Elk's own default (-1);
            * ``2`` -- x and z only; ``3`` -- x, y and z only.

            0, 1 and values below -4 are rejected by ``gentpmae`` with a hard
            stop, so they are rejected here first.
        from_scratch
            ``True`` runs task 28 (each direction started from atomic
            densities) after a fresh task 0; the default ``False`` runs task
            29, which resumes from this ``Calculation``'s cached ground state
            -- much cheaper, and the reason the cache exists.
        epspot, epsengy
            Convergence targets. The defaults are tightened well past Elk's
            own (1e-6 and 1e-4 Ha) because the MAE of a 3d magnet is
            micro-Hartree scale: Elk's default ``epsengy`` alone is roughly
            two orders of magnitude larger than the signal. Elk's
            ``examples/magnetism/FeCo-MAE`` uses exactly these values.
        socscf
            Optional global spin-orbit scale factor. ``mae.f90`` records it
            at the top of ``MAE_INFO.OUT`` precisely because the MAE is the
            quantity this knob is normally fitted against (see
            ``docs/design.md`` #12; per-species scaling is available through
            ``Calculation(soc_scale=...)``).

        Returns
        -------
        dict
            ``mae`` (Hartree), ``mae_per_volume`` (Hartree/Bohr^3),
            ``directions`` (one dict per sampled direction: spherical and
            Cartesian direction, converged total moment and total energy),
            ``min_point``/``max_point`` and ``directory``.
        """
        if not self.spinpol:
            raise ValueError(
                "get_mae() requires a magnetic ground state (spinpol=True): the "
                "magnetic anisotropy energy is the direction dependence of the "
                "magnetisation's energy, which is identically zero without one. "
                "mae.f90 itself never checks -- it relies on init0.f90 forcing "
                "spinpol once it sets spinorb -- so the check is made here."
            )
        if npmae in (0, 1) or npmae < -4:
            raise ValueError(
                f"invalid npmae {npmae}: src/gentpmae.f90 accepts >= 4 (evenly "
                "covered sphere), -1..-4 (symmetry-reduced cardinal directions), "
                "2 (x,z) or 3 (x,y,z), and stops on anything else"
            )
        blocks = {
            "npmae": [int(npmae)],
            "epspot": [float(epspot)],
            "epsengy": [float(epsengy)],
        }
        if socscf is not None:
            blocks["socscf"] = [float(socscf)]
        blocks.update(extra_blocks or {})

        if from_scratch:
            subdir = self._fresh_subdir(label)
            f = InputFile()
            f.add_block("tasks", [0, TASK_MAE])
            self._add_base_blocks(f, ngridk=tuple(ngridk) if ngridk else None)
            for name, lines in blocks.items():
                f.add_block(name, lines)
            f.write(subdir / "elk.in")
            self.launcher.run(subdir)
        else:
            subdir = self._run_resumed(
                label,
                [TASK_MAE_RESUME],
                blocks,
                ngridk=tuple(ngridk) if ngridk else None,
            )

        result = parsers_magnetism.parse_mae_info(subdir / FILE_MAE_INFO)
        result["mae"] = parsers_magnetism.parse_mae(subdir / FILE_MAE)
        result["mae_per_volume"] = parsers_magnetism.parse_mae_per_volume(
            subdir / FILE_MAE_PER_VOLUME
        )
        result["directory"] = subdir
        return result

    # -----------------------------------------------------------------
    # 160 -- exchange-correlation torque
    # -----------------------------------------------------------------

    def get_total_magnetic_torque(self, label="torque"):
        """Total torque exerted by the exchange-correlation field on the
        magnetisation (task 160, src/torque.f90).

        .. math::
            \\boldsymbol\\tau = \\int \\mathbf m(\\mathbf r) \\times
            \\mathbf B_{\\rm xc}(\\mathbf r)\\, d^3r

        computed by ``rvfcross`` on the converged ``magmt``/``magir`` and
        ``bxcmt``/``bxcir`` from ``STATE.OUT``, then integrated with
        ``rfint``. This is a *diagnostic of the functional*, not of the
        material: an exact exchange-correlation functional obeys the zero-
        torque theorem (the xc energy is invariant under a global spin
        rotation, so :math:`\\boldsymbol\\tau` must vanish), while the local
        approximations Elk implements do not, and the residual measures how
        badly. It is also the driving term of magnetisation dynamics, so a
        non-collinear state that is not stationary shows up here.

        **Returns zeros by construction on a collinear calculation.**
        ``torque.f90`` short-circuits with ``if (.not.ncmag) torq = 0`` before
        reading anything: with :math:`\\mathbf m \\parallel \\mathbf B_{\\rm
        xc}` everywhere the cross product vanishes identically. ``ncmag``
        means ``ndmag == 3``, which init0.f90 sets when ``spinorb`` is on,
        when a transverse ``bfieldc``/``bfcmt`` component is present, or for
        source-free/spin-spiral runs -- and *unsets* whenever ``cmagz`` is
        true. That decision is deterministic given ``elk.in``, so
        :meth:`_ndmag` reproduces it and this method **warns** when the run
        will return structural zeros rather than a measurement. It warns
        rather than raises because zeros are still the correct answer for a
        collinear state, just an uninformative one.

        Task 160 writes **no output file**; the three numbers exist only on
        standard output, which the launcher captures into ``elk.out``.

        Returns a ``(3,)`` array of Cartesian components in atomic units.
        """
        if not (self.spinpol or self.spinorb):
            raise ValueError(
                "get_total_magnetic_torque() requires a spin-polarised calculation "
                "(spinpol=True and/or spinorb=True): without a magnetisation "
                "there is nothing for B_xc to exert a torque on"
            )
        if self._ndmag() != 3:
            warnings.warn(
                "this calculation is collinear (ndmag != 3 by init0.f90's rule), "
                "so torque.f90 short-circuits to exactly [0, 0, 0] without "
                "reading anything -- m is parallel to B_xc everywhere. Add a "
                "transverse bfcmt/bfieldc component, or spinorb=True, for a "
                "non-collinear state with a torque to measure",
                RuntimeWarning,
                stacklevel=2,
            )
        subdir = self._run_resumed(label, [TASK_TORQUE])
        return parsers_magnetism.parse_torque(subdir / "elk.out")

    # -----------------------------------------------------------------
    # spin spirals: generalised Bloch theorem (no task code)
    # -----------------------------------------------------------------

    def _spiral_child(self, q, bfieldc, label, ngridk, extra_blocks):
        blocks = dict(self.extra_blocks)
        blocks.update(
            {
                "spinsprl": [True],
                "vqlss": [tuple(float(x) for x in q)],
            }
        )
        if bfieldc is not None:
            blocks["bfieldc"] = [tuple(float(x) for x in bfieldc)]
        blocks.update(extra_blocks or {})
        return type(self)(
            self.structure,
            workdir=self.workdir / label,
            xc=self.xc,
            spinpol=True,
            spinorb=False,
            rgkmax=self.rgkmax,
            ngridk=tuple(ngridk) if ngridk else self.ngridk,
            vkloff=self.vkloff,
            sppath=self.sppath,
            launcher=self.launcher,
            extra_blocks=blocks,
            raise_on_nonconvergence=self.raise_on_nonconvergence,
        )

    def get_spin_spiral_energy(
        self, q, bfieldc=(0.05, 0.0, 0.0), ngridk=None, extra_blocks=None, label=None
    ):
        """Total energy of a spin spiral of wavevector `q`, via the
        **generalised Bloch theorem** (``spinsprl``/``vqlss``; no task code).

        A flat spin spiral

        .. math::
            \\mathbf m(\\mathbf r + \\mathbf R) = \\bigl(
            m_\\perp\\cos(\\mathbf q\\!\\cdot\\!\\mathbf R + \\phi),\\;
            m_\\perp\\sin(\\mathbf q\\!\\cdot\\!\\mathbf R + \\phi),\\;
            m_z \\bigr)

        is generally incommensurate with the lattice, so it has no finite
        supercell. The generalised Bloch theorem removes the need for one:
        the spiral is a symmetry of the Hamiltonian combining a translation
        by :math:`\\mathbf R` with a spin rotation by :math:`\\mathbf q\\cdot
        \\mathbf R` about z, so the two spinor components can be given
        different Bloch vectors, :math:`\\mathbf k \\mp \\mathbf q/2`, and the
        whole spiral is computed in the **chemical unit cell** at the cost of
        two first-variational problems per k-point (``nspnfv = 2``).

        The price is exact: the theorem holds only when spin-orbit coupling
        is absent, because SOC ties the spin rotation to the lattice and the
        combined operation stops being a symmetry. ``spinorb`` is therefore
        forced off for this run regardless of the parent's setting, and
        ``spinsprl`` forces non-collinear magnetism (init0.f90 sets
        ``ndmag=3``, ``cmagz=.false.``).

        This runs a **fresh, independent ground state** in its own
        subdirectory -- the spiral changes the Hamiltonian, so the parent's
        ``STATE.OUT`` is not a valid starting point and is deliberately not
        reused.

        Parameters
        ----------
        q
            Spiral wavevector in lattice (fractional reciprocal)
            coordinates.
        bfieldc
            Global external field seeding the transverse moment. A spiral
            needs a nonzero perpendicular component to nucleate; with
            ``bfieldc = 0`` the calculation converges to whatever collinear
            state it started in. The default matches Elk's own
            ``examples/magnetism/Fe-spiral``. Pass ``None`` to omit the block.

        Returns the converged total energy in Hartree.
        """
        q = tuple(float(x) for x in q)
        if label is None:
            label = "spiral_q_" + "_".join(f"{x:.4f}".replace("-", "m") for x in q)
        child = self._spiral_child(q, bfieldc, label, ngridk, extra_blocks)
        return child.get_energy()

    def get_spin_spiral_dispersion(
        self, qpoints, bfieldc=(0.05, 0.0, 0.0), ngridk=None, extra_blocks=None,
        label="spiral_dispersion", reference=None,
    ):
        """Frozen-magnon energy dispersion :math:`E(\\mathbf q)` over a list
        of spiral wavevectors (generalised Bloch theorem, see
        :meth:`get_spin_spiral_energy`).

        This is the standard route from DFT to the magnetic exchange
        interaction without a supercell: mapping onto a classical Heisenberg
        model :math:`H = -\\sum_{ij} J_{ij}\\,\\hat{\\mathbf m}_i\\cdot
        \\hat{\\mathbf m}_j`, a spiral of wavevector :math:`\\mathbf q` has
        energy

        .. math::
            E(\\mathbf q) - E(0) = -\\bigl[J(\\mathbf q) - J(0)\\bigr],
            \\qquad J(\\mathbf q) = \\sum_{\\mathbf R} J(\\mathbf R)\\,
            e^{i\\mathbf q\\cdot\\mathbf R},

        so :math:`E(\\mathbf q)` **is** the Fourier transform of the exchange
        constants (up to sign and normalisation), and its curvature at
        :math:`\\mathbf q = 0` gives the spin-wave stiffness. Compare
        ``Calculation.get_exchange_tensor`` (docs/design.md #29), which gets
        the same physics one bond at a time from four constrained supercell
        energies -- real space rather than reciprocal space, but able to
        resolve the anisotropic and Dzyaloshinskii-Moriya parts that the
        SOC-free spiral ansatz cannot.

        Each q-point is a separate self-consistent calculation in its own
        subdirectory, so an interrupted sweep resumes for free (each child's
        ground-state manifest is checked independently).

        Returns ``(qpoints, energies)`` with `energies` in Hartree; if
        `reference` is given (a q-vector, typically ``(0,0,0)``), the
        energies are returned relative to that point's own run, which cancels
        most of the basis-set error.
        """
        qpoints = [tuple(float(x) for x in q) for q in qpoints]
        energies = []
        for i, q in enumerate(qpoints):
            child = self._spiral_child(
                q, bfieldc, f"{label}/q{i:03d}", ngridk, extra_blocks
            )
            energies.append(child.get_energy())
        energies = np.array(energies)
        if reference is not None:
            e0 = self.get_spin_spiral_energy(
                reference,
                bfieldc=bfieldc,
                ngridk=ngridk,
                extra_blocks=extra_blocks,
                label=f"{label}/reference",
            )
            energies = energies - e0
        return np.array(qpoints), energies

    # -----------------------------------------------------------------
    # 350/351/352 -- supercell spin spirals
    # -----------------------------------------------------------------

    def get_spin_spiral_supercell(
        self,
        ngridq,
        dry_run=False,
        from_state=False,
        radkpt=None,
        resume=False,
        extra_blocks=None,
        label="spiral_supercell",
    ):
        """Spin spirals by explicit **supercells**, one per q-point
        (tasks 350/351/352, src/spiralsc.f90).

        The complement of :meth:`get_spin_spiral_energy`. Where the
        generalised Bloch theorem needs no supercell but forbids spin-orbit
        coupling, this route builds, for each q on the ``ngridq`` mesh, the
        smallest supercell that makes the spiral commensurate
        (``findscq``/``genscss``: the supercell lattice vectors are chosen so
        that :math:`\\mathbf q\\cdot\\mathbf R` is an integer multiple of
        :math:`2\\pi`, and each replicated atom's ``bfcmt``/``mommtfix``
        vector is rotated by :math:`\\mathbf q\\cdot\\mathbf r` in the xy
        plane), then runs an ordinary ground state on it. Nothing about the
        Hamiltonian is approximated, so this works with SOC and with
        functionals that do not preserve the spiral ansatz -- at the cost of
        ``nscss`` times as many atoms.

        Elk's own note applies: **automatic k-point generation is always
        enabled for this task**. ``spiralsc.f90`` sets ``autokpt=.true.``
        before each supercell run, so ``ngridk`` is ignored and the k-point
        density is controlled by ``radkpt`` instead (each supercell then gets
        a mesh matched to its own size, which is the point).

        The task is *restartable by file existence*: ``sstask.f90`` walks the
        q-points and claims the first one whose ``SS_Q...OUT`` file does not
        yet exist, which is how Elk lets several processes share one
        directory. Two consequences:

        * ``dry_run=True`` (task 352) creates every SS file **empty** and
          computes nothing -- it exists to enumerate the work. Running task
          350 afterwards in the same directory is then a **no-op**, because
          every q-point already looks claimed. This method therefore wipes
          its directory by default; pass ``resume=True`` only to continue an
          interrupted *real* sweep.
        * An interrupted run leaves the in-progress q-point's file empty;
          those are reported separately rather than being parsed as zeros.

        Parameters
        ----------
        ngridq
            The q-point mesh the spirals are computed on. Note this is the
            *same input block* the phonon tasks use.
        from_state
            Task 351 instead of 350. **This is not "resume from this
            Calculation's ground state".** ``sstask`` sets the global
            ``filext`` to that q-point's own ``_Q..._..._....OUT`` suffix
            *before* ``gndstate`` runs, so ``readstate`` opens
            ``STATE_Q..._..._....OUT`` -- the supercell's own state file
            from a previous pass -- and never the unit cell's ``STATE.OUT``
            (which would fail ``readstate``'s ``differing natoms`` check
            anyway, since the supercell has ``nscss`` times as many atoms).
            Use it only to re-converge q-points that already have their
            ``STATE_Q*.OUT`` files in the directory, having removed the
            ``SS_Q*.OUT`` files you want recomputed; it therefore requires
            ``resume=True`` and will raise otherwise.
        radkpt
            k-point density (Bohr) for the automatically generated meshes.

        Returns a dict with ``results`` (one parsed record per finished
        q-point, sorted by \\|q\\|, each carrying ``energy`` **per unit
        cell**, the q-vector in lattice and Cartesian coordinates and the
        number of unit cells in its supercell), ``pending`` (filenames
        claimed but not finished, including every file after a dry run) and
        ``directory``.
        """
        if dry_run:
            task = TASK_SPIRAL_SUPERCELL_DRYRUN
        elif from_state:
            if not resume:
                raise ValueError(
                    "from_state=True (task 351) reads each q-point's own "
                    "STATE_Q..._..._....OUT, written by a previous task-350 pass "
                    "in the same directory (sstask.f90 sets filext before "
                    "gndstate runs). Wiping the directory would delete exactly "
                    "those files, so pass resume=True -- and remove the "
                    "SS_Q*.OUT files of the q-points you want recomputed, since "
                    "sstask skips every q-point whose SS file already exists."
                )
            task = TASK_SPIRAL_SUPERCELL_RESUME
        else:
            task = TASK_SPIRAL_SUPERCELL
        blocks = {"ngridq": [tuple(int(n) for n in ngridq)]}
        if radkpt is not None:
            blocks["radkpt"] = [float(radkpt)]
        blocks.update(extra_blocks or {})

        subdir = self._fresh_subdir(label, keep=resume)
        f = InputFile()
        f.add_block("tasks", [task])
        self._add_base_blocks(f)
        for name, lines in blocks.items():
            f.add_block(name, lines)
        f.write(subdir / "elk.in")
        self.launcher.run(subdir)

        results, pending = parsers_magnetism.collect_spin_spirals(subdir)
        return {"results": results, "pending": pending, "directory": subdir}

    # -----------------------------------------------------------------
    # 550 -- Wannier90 export
    # -----------------------------------------------------------------

    def get_wannier90_input(
        self,
        seedname="wannier",
        num_wann=None,
        bands=None,
        projections=None,
        num_iter=None,
        dis_num_iter=None,
        extra_win_lines=None,
        write_unk=False,
        ngridk=None,
        extra_blocks=None,
        label="wannier90",
    ):
        """Wannier90 export (task 550, src/writew90.f90).

        Elk's Wannier90 interface writes the five files Wannier90 consumes:
        the input template ``<seedname>.win``, the eigenvalues
        ``<seedname>.eig``, the projection overlaps ``<seedname>.amn``
        :math:`A^{(\\mathbf k)}_{mn} = \\langle\\psi_{m\\mathbf k}|g_n\\rangle`,
        the neighbour overlaps ``<seedname>.mmn``
        :math:`M^{(\\mathbf k,\\mathbf b)}_{mn} = \\langle u_{m\\mathbf k}|
        u_{n,\\mathbf k+\\mathbf b}\\rangle` and, for spinors, the spin matrix
        elements ``<seedname>.spn``.

        **The last three cannot be produced by this build.** ``writew90``
        calls ``setupw90`` between ``.eig`` and ``.amn``, and ``setupw90``
        calls ``wannier_setup`` -- a routine *of the Wannier90 library*, whose
        job is to return the :math:`\\mathbf b`-vector shells. This build
        links ``w90_stub.f90`` (``SRC_W90S`` in ``build-config/make.inc``),
        which prints ``Error(wannier_setup): libwannier not or improperly
        installed`` and executes ``error stop``. That abort is *expected* and
        is caught here rather than raised: ``.win`` and ``.eig`` are written
        before it and are complete and usable.

        This is also the reason ``patches/0002-berry-curvature-wilson-loop.patch``
        exists at all -- elkpy's Berry-curvature task reimplements the
        ``.mmn``-style overlap export (``genwfsvp``/``genolpq``) without the
        neighbour-shell search that needs the library (docs/design.md #13).

        Parameters
        ----------
        num_wann
            Number of Wannier functions. Elk's convention (initw90.f90) is
            that a **non-positive** value is added to ``num_bands``, so the
            default ``None`` leaves Elk's own default of 0, i.e. one Wannier
            function per band.
        bands
            Band subset as a Wannier90-style range string, e.g. ``"1-4"`` or
            ``"1-4, 7"`` (Elk's ``wann_bands``/``idxw90`` block, parsed by
            ``numlist``). Written unquoted, since Elk reads the whole line as
            text and re-parses it.
        projections
            ``{symbol: [l, ...]}`` initial projections by angular momentum
            (Elk's ``lprojw90``, which also switches on ``projw90`` and then
            *derives* ``num_wann`` from the projector count). Species are
            indexed by their order in ``structure.species``, matching Elk's
            own species numbering.
        extra_win_lines
            Verbatim extra lines for the ``.win`` file (Elk's
            ``xlwin``/``wannierExtra``), e.g. ``["dis_win_max = 20.0"]``.

        Returns a dict with ``directory``, ``win``/``eig`` paths, the parsed
        ``eigenvalues`` (eV, Fermi-referenced, shape ``(nkpt, num_bands)``),
        the parsed ``win`` contents, ``wannier90_available`` (False whenever
        the stub aborted) and ``missing`` (the files the abort prevented).
        """
        blocks = {"seedname": [str(seedname)]}
        if num_wann is not None:
            blocks["num_wann"] = [int(num_wann)]
        if bands is not None:
            blocks["wann_bands"] = [_RawToken(bands)]
        if num_iter is not None:
            blocks["num_iter"] = [int(num_iter)]
        if dis_num_iter is not None:
            blocks["dis_num_iter"] = [int(dis_num_iter)]
        if write_unk:
            blocks["wrtunk"] = [True]
        if projections:
            species = list(self.structure.species)
            lines = []
            for symbol, ls in projections.items():
                if symbol not in species:
                    raise ValueError(
                        f"projections species {symbol!r} not in structure "
                        f"(known: {species})"
                    )
                padded = list(ls) + [-1] * (4 - len(ls))
                lines.append(tuple([species.index(symbol) + 1] + padded[:4]))
            blocks["projw90"] = [True]
            blocks["lprojw90"] = lines
        if extra_win_lines:
            blocks["xlwin"] = [_RawToken(line) for line in extra_win_lines]
        blocks.update(extra_blocks or {})

        # Converge (or reuse) the ground state BEFORE the try, so that a
        # non-convergence RuntimeError from ensure_ground_state is not
        # mistaken below for the expected libwannier abort.
        self.ensure_ground_state()
        available = True
        try:
            subdir = self._run_resumed(
                label,
                [TASK_WANNIER90],
                blocks,
                ngridk=tuple(ngridk) if ngridk else None,
            )
        except RuntimeError:
            # w90_stub.f90's `error stop` (gfortran: exit status 1) -- expected
            # in a build without libwannier. _run_resumed has already created
            # and populated the directory, so the .win/.eig written before the
            # abort survive. Confirm it really was the stub before swallowing
            # the failure: any other non-zero exit must still surface.
            subdir = self.workdir / label
            log = subdir / "elk.out"
            if not log.exists() or W90_STUB_MESSAGE not in log.read_text():
                raise
            available = False

        win_path = subdir / f"{seedname}.win"
        eig_path = subdir / f"{seedname}.eig"
        if not win_path.exists() or not eig_path.exists():
            raise RuntimeError(
                f"task 550 produced neither {win_path.name} nor {eig_path.name} in "
                f"{subdir}; see {subdir / 'elk.out'} -- this is a real failure, not "
                "the expected missing-libwannier abort (which happens only after "
                "both files are written; src/writew90.f90)"
            )
        missing = [
            name
            for name in (f"{seedname}.amn", f"{seedname}.mmn", f"{seedname}.spn")
            if not (subdir / name).exists()
        ]
        energies, nkpt, num_bands = parsers_w90.parse_w90_eig(eig_path)
        return {
            "directory": subdir,
            "win": win_path,
            "eig": eig_path,
            "eigenvalues": energies,
            "nkpt": nkpt,
            "num_bands": num_bands,
            "win_contents": parsers_w90.parse_w90_win(win_path),
            "wannier90_available": available and not missing,
            "missing": missing,
        }

    # -----------------------------------------------------------------
    # 600-640 -- GW
    # -----------------------------------------------------------------

    def _gw_blocks(
        self, wmaxgw, tempk, nempty, gmaxrf, actype, npole, nspade, tsediag,
        ngridq, extra_blocks,
    ):
        blocks = {}
        if wmaxgw is not None:
            blocks["wmaxgw"] = [float(wmaxgw)]
        if tempk is not None:
            blocks["tempk"] = [float(tempk)]
        if nempty is not None:
            blocks["nempty"] = [int(nempty)]
        if gmaxrf is not None:
            blocks["gmaxrf"] = [float(gmaxrf)]
        if actype is not None:
            blocks["actype"] = [int(actype)]
        if npole is not None:
            blocks["npole"] = [int(npole)]
        if nspade is not None:
            blocks["nspade"] = [int(nspade)]
        if tsediag:
            blocks["tsediag"] = [True]
        if ngridq is not None:
            blocks["ngridq"] = [tuple(int(n) for n in ngridq)]
        blocks.update(extra_blocks or {})
        return blocks

    def get_gw_self_energy(
        self,
        wmaxgw=5.0,
        tempk=1500.0,
        nempty=20,
        gmaxrf=3.0,
        actype=10,
        npole=None,
        nspade=None,
        tsediag=False,
        ngridq=None,
        ngridk=None,
        reuse_epsinv=False,
        extra_blocks=None,
        label="gw",
    ):
        """GW self-energy on the Matsubara axis (tasks 600/601,
        src/gwsefm.f90). **The expensive step of every GW workflow.**

        Elk's GW is the finite-temperature :math:`G_0W_0` self-energy
        evaluated entirely on the imaginary axis,

        .. math::
            \\Sigma_{nm}(\\mathbf k, i\\omega_j) = -\\frac{1}{\\beta}
            \\sum_{\\mathbf q, i\\nu_l} \\sum_{p}
            \\langle nm|W(\\mathbf q, i\\nu_l)|p\\rangle\\,
            G_p(\\mathbf k-\\mathbf q, i\\omega_j - i\\nu_l),

        with the screened interaction :math:`W = \\varepsilon^{-1} v` built
        from the RPA inverse dielectric matrix. Working on the Matsubara axis
        keeps every quantity smooth (no :math:`i\\eta` prescription and no
        pole structure to resolve), which is why the real-axis spectral
        function comes later and by analytic continuation (task 610).

        The imaginary-axis grid is not specified directly: the fermionic
        frequencies are :math:`\\omega_j = (2j+1)\\pi/\\beta` with
        :math:`\\beta = 1/(k_B T)` set by ``tempk``, truncated at
        ``wmaxgw``. So ``tempk`` sets the *spacing* and ``wmaxgw`` the
        *extent*, and neither is a physical temperature in the usual sense --
        Elk's own examples use 1500-2000 K purely to keep the number of
        frequencies tractable.

        What this task does, in order (src/gwsefm.f90): read ``STATE.OUT``,
        regenerate the radial functions, read the second-variational
        eigenvalues/occupations from file, call ``genpmat`` (momentum matrix
        elements, the same machinery as task 120), call ``epsinv`` to build
        and write the RPA inverse dielectric matrix ``EPSINV.OUT``, form the
        matrix elements of :math:`-V_{\\rm xc}` and :math:`-B_{\\rm xc}`
        (which the GW self-energy replaces), then loop over k-points writing
        :math:`\\Sigma` to ``GWSEFM.OUT``.

        ``reuse_epsinv=True`` selects **task 601**, whose only difference is
        that it skips the ``epsinv`` call and reads the existing
        ``EPSINV.OUT`` in the SAME directory -- so it runs in place there
        rather than in a wiped copy, and raises if the file is absent.

        Note what it is *not* good for. ``EPSINV.OUT``'s records are
        dimensioned by ``ngrf`` and ``nwrf``, and ``genwgw.f90`` builds the
        Matsubara count from ``wmaxgw`` and ``tempk``
        (:math:`n_{\\rm wgw}=2\\,{\\rm nint}[w_{\\max}/(\\pi k_B T)]`,
        then ``nwrf = nwbs + 1``), so changing either makes
        ``getcfgq`` stop with "differing m" -- the screening is not
        independent of the frequency grid the way re-running "at a different
        ``wmaxgw`` with the same screening" would need. The legitimate use is
        re-entering the k-loop with the screening already built: a run killed
        partway through, or a change of ``tsediag`` (``gwsefmk.f90:209``,
        which touches only the self-energy's own matrix structure) or of the
        continuation settings ``actype``/``npole``/``nspade``, which tasks
        610/620 consume later. Every block ``EPSINV.OUT`` depends on is
        checked against the producing run's (``GW_EPSINV_PINNED``) and a
        mismatch raises here, naming the block -- including ``nempty``, which
        changes the file's contents without changing its shape and which Elk
        therefore cannot catch.

        Both ``GWSEFM.OUT`` and ``EPSINV.OUT`` are unformatted direct-access
        files whose record length is compiler-dependent, so elkpy treats them
        as opaque: this method returns the **directory**, and the dependent
        tasks (610/630/640) consume it in place through
        :meth:`_run_dependent` rather than re-running this step.

        Parameters
        ----------
        nempty
            Number of empty states. GW converges notoriously slowly in this
            parameter -- Elk's Si example uses 20, its band-structure example
            40, and neither claims convergence.
        gmaxrf
            G-vector cut-off for the response function. The dominant cost
            knob after ``nempty`` and the k-mesh.
        actype, npole, nspade
            Analytic-continuation settings used later, by tasks 610/620;
            recorded here so the whole chain shares one setting.
        ngridq
            q-mesh for the screened interaction; defaults (init2.f90) to
            ``ngridk``.
        """
        blocks = self._gw_blocks(
            wmaxgw, tempk, nempty, gmaxrf, actype, npole, nspade, tsediag,
            ngridq, extra_blocks,
        )
        ngridk = tuple(ngridk) if ngridk else None
        if reuse_epsinv:
            task = TASK_GW_SELF_ENERGY_KEEP_EPSINV
            subdir, blocks, ngridk = self._run_reusing(
                label, [task], blocks, FILE_GW_EPSINV, GW_EPSINV_PINNED,
                ngridk=ngridk,
            )
        else:
            task = TASK_GW_SELF_ENERGY
            subdir = self._run_resumed(label, [task], blocks, ngridk=ngridk)
        if not (subdir / FILE_GW_SELF_ENERGY).exists():
            raise RuntimeError(
                f"task {task} did not produce {FILE_GW_SELF_ENERGY} in {subdir}; "
                f"see {subdir / 'elk.out'}"
            )
        self._write_stage_manifest(subdir, blocks, ngridk=ngridk, vkloff=None)
        return subdir

    def get_gw_spectral_function(
        self,
        gw_dir=None,
        wplot=(-0.8, 0.5),
        nwplot=800,
        ngrkf=100,
        nswplot=0,
        kpoint=None,
        label="gw",
        **self_energy_kwargs,
    ):
        """GW spectral function on the real axis (task 610, src/gwspecf.f90).

        Solves the Dyson equation

        .. math::
            G(\\mathbf k, \\omega) = \\bigl[\\omega - \\varepsilon^{\\rm KS}
            _{\\mathbf k} - \\Sigma(\\mathbf k, \\omega) + V_{\\rm xc}
            \\bigr]^{-1}, \\qquad
            A(\\mathbf k,\\omega) = -\\tfrac1\\pi\\,{\\rm Im}\\,{\\rm Tr}\\,G,

        after analytically continuing the Matsubara self-energy written by
        task 600 to the real axis (``dysonr``; the continuation method is the
        ``actype`` block -- 10 is a multi-shift Pade average controlled by
        ``nspade``, lower values fit ``npole`` poles). The k-resolved spectra
        go to ``GWSF_Kkkkkkk.OUT``, the Brillouin-zone sum to ``GWTSF.OUT``.

        If `gw_dir` is a directory produced by :meth:`get_gw_self_energy`,
        the task runs **in place** there and reuses that self-energy. If it
        is ``None``, the whole chain (task 1 + 600 + 610) runs in one
        invocation -- correct, but it recomputes the expensive part, so pass
        `gw_dir` when iterating on the plotting window.

        The quasiparticle peak positions in ``GWTSF.OUT`` are *not*
        Fermi-referenced: gwspecf.f90 states plainly that "Fermi energy for
        the GW spectral function is undetermined" -- see
        :meth:`get_gw_fermi_energy` (task 630) for that.

        Returns ``(w, sf)`` in Hartree and states/Hartree/unit cell, or, if
        `kpoint` (1-based reduced k-point index) is given, that k-point's own
        ``(w, sf)`` from ``GWSF_K......OUT``.
        """
        blocks = {"wplot": [(int(nwplot), int(ngrkf), int(nswplot)), tuple(wplot)]}
        if gw_dir is None:
            gw_dir = self.get_gw_self_energy(label=label, **self_energy_kwargs)
        subdir = self._run_dependent(gw_dir, [TASK_GW_SPECTRAL_FUNCTION], blocks)
        if kpoint is not None:
            return parsers_gw.parse_gw_spectral_function(
                subdir / GWSF_TEMPLATE.format(ik=int(kpoint))
            )
        return parsers_gw.parse_gw_total_spectral_function(
            subdir / FILE_GW_TOTAL_SPECTRAL_FUNCTION
        )

    def get_gw_fermi_energy(self, gw_dir=None, label="gw", **self_energy_kwargs):
        """The GW Fermi energy (task 630, src/writegwefm.f90).

        The Kohn-Sham Fermi energy is not the interacting one: the
        self-energy shifts every quasiparticle level, so the chemical
        potential must be re-found from the *interacting* particle number.
        ``gwefermi.f90`` bisects :math:`\\mu` until the Matsubara sum of the
        interacting Green's function,

        .. math:: N(\\mu) = \\frac{1}{\\beta}\\sum_{\\mathbf k, i\\omega_j}
                  {\\rm Tr}\\, G(\\mathbf k, i\\omega_j)\\, e^{i\\omega_j 0^+},

        matches the valence charge (``gwchgk`` per k-point, with
        ``gwtails`` supplying the high-frequency tail correction). Because
        ``gwchgk`` calls ``getgwsefm``, this task **requires ``GWSEFM.OUT``**
        from task 600 -- it is not a cheap standalone post-processing step,
        despite writing only a single number.

        Returns the GW Fermi energy in Hartree (``GWEFERMI.OUT``).
        """
        if gw_dir is None:
            gw_dir = self.get_gw_self_energy(label=label, **self_energy_kwargs)
        subdir = self._run_dependent(gw_dir, [TASK_GW_FERMI_ENERGY])
        return parsers_gw.parse_gw_fermi_energy(subdir / FILE_GW_FERMI_ENERGY)

    def get_gw_density_matrix(self, gw_dir=None, label="gw", **self_energy_kwargs):
        """GW density matrix, natural orbitals and occupations (task 640,
        src/gwdmat.f90).

        For each k-point, ``gwdmatk`` sums the interacting Green's function
        over Matsubara frequencies to build the one-particle density matrix

        .. math::
            D_{nm}(\\mathbf k) = \\frac{1}{\\beta}\\sum_{i\\omega_j}
            G_{nm}(\\mathbf k, i\\omega_j)\\,e^{i\\omega_j 0^+},

        then diagonalises it. Its eigenvectors are the **natural orbitals**
        and its eigenvalues the occupation numbers -- which, unlike
        Kohn-Sham occupations, are fractional even in a band insulator,
        because correlation scatters weight above the Fermi level. Like task
        630 this needs ``GWSEFM.OUT``, and it also needs the GW Fermi energy,
        which it computes internally.

        The results **overwrite** ``EVECSV.OUT`` and ``OCCSV.OUT`` in the run
        directory, replacing the Kohn-Sham eigenvectors and occupations with
        the GW natural orbitals and occupations, so that any subsequent task
        reading those files sees the correlated quantities. Both are
        unformatted direct-access files, so this method returns the
        **directory** rather than parsed arrays; the natural place to consume
        them is another Elk task run in that directory.

        Returns the run directory.
        """
        if gw_dir is None:
            gw_dir = self.get_gw_self_energy(label=label, **self_energy_kwargs)
        return self._run_dependent(gw_dir, [TASK_GW_DENSITY_MATRIX])

    def get_gw_band_structure(
        self,
        vertices=None,
        kpath=None,
        npoints=200,
        wplot=(-1.5, 2.0),
        nwplot=800,
        ngrkf=100,
        nswplot=0,
        start_point=None,
        ngridk=None,
        label="gw_bands",
        **gw_kwargs
    ):
        """GW spectral-function band structure along a k-path (task 620,
        src/gwbandstr.f90).

        Not a band structure in the Kohn-Sham sense: an interacting system
        has no eigenvalue at each k, only a spectral function
        :math:`A(\\mathbf k, \\omega)` whose peaks are quasiparticles with a
        finite width :math:`\\propto {\\rm Im}\\,\\Sigma`. The output is
        therefore an intensity map over (path distance, energy), and its
        ridges are the quasiparticle bands.

        Unlike every other GW task here this one is **self-contained**: for
        each of the ``npp1d`` points on the path it shifts the k-mesh offset
        so the path point becomes the mesh's first k-point (``vkloff =
        vplp1d * ngridk``), re-solves the Kohn-Sham problem on that shifted
        mesh, recomputes the momentum matrix elements *and the entire inverse
        dielectric matrix*, and only then evaluates the self-energy at that
        one k-point. So it costs roughly ``npp1d`` times a full task-600 run,
        does not consume ``GWSEFM.OUT``, and cannot be resumed from
        :meth:`get_gw_self_energy`'s directory. Elk's own example
        (``examples/GW/Si-GW-band-structure``) puts it at "about 3 days on
        200 CPU cores".

        ``start_point`` maps to Elk's ``ip0gw`` block: the run appends to
        ``GWBAND.OUT`` from that path point onwards, so an interrupted job
        can be continued rather than restarted.

        Returns ``(distances, frequencies, sf)`` with `sf` of shape
        ``(npoints, nwplot)``.
        """
        vertices = self._resolve_vertices(vertices, kpath)
        blocks = self._gw_blocks(
            gw_kwargs.pop("wmaxgw", 5.0),
            gw_kwargs.pop("tempk", 2000.0),
            gw_kwargs.pop("nempty", 40),
            gw_kwargs.pop("gmaxrf", 3.0),
            gw_kwargs.pop("actype", 10),
            gw_kwargs.pop("npole", None),
            gw_kwargs.pop("nspade", None),
            gw_kwargs.pop("tsediag", False),
            gw_kwargs.pop("ngridq", None),
            gw_kwargs.pop("extra_blocks", None),
        )
        if gw_kwargs:
            raise TypeError(
                f"get_gw_band_structure() got unexpected keyword arguments "
                f"{sorted(gw_kwargs)}"
            )
        blocks["plot1d"] = [(len(vertices), int(npoints))] + [
            tuple(v) for v in vertices
        ]
        blocks["wplot"] = [(int(nwplot), int(ngrkf), int(nswplot)), tuple(wplot)]
        if start_point is not None:
            blocks["ip0gw"] = [int(start_point)]
        subdir = self._run_resumed(
            label,
            [TASK_GW_BAND_STRUCTURE],
            blocks,
            ngridk=tuple(ngridk) if ngridk else None,
        )
        return parsers_gw.parse_gw_band(subdir / FILE_GW_BAND)

    # -----------------------------------------------------------------
    # 700-773 -- ultra-long-range
    # -----------------------------------------------------------------

    def get_ulr_ground_state(
        self,
        avecu,
        ngridq,
        scaleu=None,
        ngridkpa=None,
        q0cut=None,
        rndbfcu=None,
        bfieldcu=None,
        efieldcu=None,
        maxscl=None,
        beta0=None,
        epspot=None,
        from_state=False,
        ngridk=None,
        extra_blocks=None,
        label="ulr",
    ):
        """Ultra-long-range (ultracell) ground state (tasks 700/701,
        src/gndstulr.f90).

        The ULR method (T. Mueller, S. Sharma, E. K. U. Gross and
        J. K. Dewhurst, PRL **125**, 256402 (2020), arXiv:2008.12573) targets
        order that is *incommensurate* with the chemical unit cell -- a spin
        density wave, a charge density wave, a long-wavelength modulation --
        without paying for a supercell calculation. The trick is to keep the
        fast (unit-cell) and slow (ultracell) degrees of freedom in different
        representations:

        * the Kohn-Sham orbitals stay in the ordinary unit-cell LAPW basis, at
          k-points of the unit-cell Brillouin zone shifted by a small set of
          ultracell **kappa-points**;
        * the density, magnetisation and potential acquire an extra,
          *slow* dependence expanded in a handful of ultracell reciprocal
          vectors :math:`\\mathbf Q` (``ngridq``), so
          :math:`n(\\mathbf r, \\mathbf R) = \\sum_{\\mathbf Q}
          n_{\\mathbf Q}(\\mathbf r)\\, e^{i\\mathbf Q\\cdot\\mathbf R}`
          with :math:`\\mathbf r` inside the unit cell and :math:`\\mathbf R`
          running over the ultracell's lattice vectors.

        The ULR Hamiltonian is then block-structured in :math:`\\mathbf Q`
        and diagonalised as ``nstulr = nstsv * nkpa`` states per original
        k-point (``eveqnulr``), so the cost grows with the number of
        Q-points, not with the ultracell's atom count. ``ngridkpa`` defaults
        (genkpakq.f90) to half the Q-grid, ``(ngridq+1)/2``, and is capped
        there.

        **Prerequisites and cost.** ``gndstulr`` calls ``readstate``, so an
        ordinary converged ``STATE.OUT`` must exist -- which is what
        ``_run_resumed``'s prepended task 1 supplies. It forces
        ``reducek=0``. It is a full self-consistent loop of its own, with its
        own ``ULR_INFO.OUT``/``RMSDVS.OUT``, and Elk's own example warns that
        a "very small mixing parameter [is] required", so ``maxscl`` in the
        thousands is normal.

        ``from_state=True`` selects **task 701**, which calls ``readstulr``
        instead of ``potuinit`` -- it restarts from the ``STATE_ULR.OUT`` a
        previous pass left in the SAME directory, so it runs in place there
        rather than in a wiped copy, and raises if that file is absent. This
        is how a ULR calculation is actually converged: ``maxscl`` in the
        thousands at a mixing parameter of 0.001 is several restarts, not one
        call. ``ngridq`` may legitimately change between passes --
        ``readstulr.f90:114-127`` maps the file's own Q-vectors onto the new
        grid and zeroes the rest, so a restart on a finer Q-grid keeps what it
        already has -- but the ultracell itself may not, and ``avecu``/
        ``scaleu`` are checked against the producing run
        (``ULR_STATE_PINNED``). ``readstulr`` verifies only unit-cell shapes
        (``natmtot``, ``npcmtmax``, ``ngtc``, ``ngtot``, ``ndmag``,
        ``fsmtype``), so a changed ultracell is read without complaint and
        means nothing.

        Parameters
        ----------
        avecu
            Ultracell lattice vectors (3x3), in units of the unit-cell
            lattice vectors' own scale -- multiplied by ``scaleu`` exactly as
            ``avec`` is by ``scale``.
        ngridq
            Ultracell Q-point grid. **The same block name the phonon and
            spin-spiral tasks use**, with a completely different meaning
            here.
        rndbfcu
            Amplitude of a random ultracell magnetic field used to seed
            symmetry breaking (Elk's Cr spin-density-wave example uses 1.0
            with ``reducebf`` damping it away).
        q0cut
            Cut-off below which small nonzero Q-vectors are treated as zero
            in the Coulomb Green's function -- the ultracell's own
            :math:`\\mathbf Q\\to 0` divergence.

        Returns the run directory (the ULR state is a *stage*, consumed by
        :meth:`get_ulr_dos`, :meth:`get_ulr_bands` and the plotting methods
        via their ``ulr_dir=`` argument).
        """
        avecu = [tuple(float(x) for x in row) for row in avecu]
        if len(avecu) != 3 or any(len(row) != 3 for row in avecu):
            raise ValueError("avecu must be three 3-vectors")
        blocks = {
            "avecu": avecu,
            "ngridq": [tuple(int(n) for n in ngridq)],
        }
        if scaleu is not None:
            blocks["scaleu"] = [float(scaleu)]
        if ngridkpa is not None:
            blocks["ngridkpa"] = [tuple(int(n) for n in ngridkpa)]
        if q0cut is not None:
            blocks["q0cut"] = [float(q0cut)]
        if rndbfcu is not None:
            blocks["rndbfcu"] = [float(rndbfcu)]
        if bfieldcu is not None:
            blocks["bfieldcu"] = [tuple(float(x) for x in bfieldcu)]
        if efieldcu is not None:
            blocks["efieldcu"] = [tuple(float(x) for x in efieldcu)]
        if maxscl is not None:
            blocks["maxscl"] = [int(maxscl)]
        if beta0 is not None:
            blocks["beta0"] = [float(beta0)]
        if epspot is not None:
            blocks["epspot"] = [float(epspot)]
        blocks.update(extra_blocks or {})

        task = (
            TASK_ULR_GROUND_STATE_RESUME if from_state else TASK_ULR_GROUND_STATE
        )
        ngridk = tuple(ngridk) if ngridk else None
        if from_state:
            # The restart runs in place, so the file it reads is also the file
            # it rewrites: its mere existence afterwards proves nothing, and
            # the RuntimeError below would pass on the PREVIOUS pass's state.
            # Its timestamp is what says this pass got as far as writing one.
            # ...taken before the run but tolerant of the file's absence, so
            # that a missing file is _run_reusing's ValueError below and not a
            # bare FileNotFoundError from this stat().
            state = self.workdir / label / FILE_ULR_STATE
            before = state.stat().st_mtime_ns if state.is_file() else None
            subdir, blocks, ngridk = self._run_reusing(
                label, [task], blocks, FILE_ULR_STATE, ULR_STATE_PINNED,
                ngridk=ngridk,
            )
            if (subdir / FILE_ULR_STATE).stat().st_mtime_ns == before:
                raise RuntimeError(
                    f"task {task} left {FILE_ULR_STATE} in {subdir} untouched, "
                    "so this restart wrote no new ultracell state and the file "
                    "there is still the previous pass's; see "
                    f"{subdir / 'elk.out'} and {FILE_ULR_INFO}"
                )
        else:
            subdir = self._run_resumed(label, [task], blocks, ngridk=ngridk)
        if not (subdir / FILE_ULR_STATE).exists():
            raise RuntimeError(
                f"task {task} did not write {FILE_ULR_STATE} in {subdir} -- "
                "gndstulr.f90 only writes it when maxscl > 1 or a WRITE file "
                f"triggers it; see {subdir / 'elk.out'} and {FILE_ULR_INFO}"
            )
        self._write_stage_manifest(subdir, blocks, ngridk=ngridk, vkloff=None)
        return subdir

    def get_ulr_dos(
        self, ulr_dir, wplot=(-0.5, 0.5), nwplot=500, ngrkf=100, nswplot=1
    ):
        """Ultra-long-range total density of states (task 710,
        src/writedosu.f90).

        The ULR eigenvalues ``evalu`` (``nstulr`` per original k-point,
        i.e. ``nstsv * nkpa`` -- every unit-cell band replicated once per
        kappa-point) are Fermi-referenced and Brillouin-zone-integrated with
        the same ``brzint`` tetrahedron/smearing machinery as the ordinary
        DOS, then divided by ``nkpa`` so the result is normalised **per unit
        cell** rather than per ultracell. Comparing it with the ordinary
        ``get_dos()`` therefore shows directly what the long-range order did
        to the spectrum -- a gap opening at the nesting vector, for instance.

        Runs in place in `ulr_dir` (from :meth:`get_ulr_ground_state`),
        reading ``STATE_ULR.OUT`` and the per-k ``EVALU`` files.

        Returns ``(energies, dos)`` in Hartree and states/Hartree/unit cell.
        """
        blocks = {"wplot": [(int(nwplot), int(ngrkf), int(nswplot)), tuple(wplot)]}
        subdir = self._run_dependent(ulr_dir, [TASK_ULR_DOS], blocks)
        return parsers_ulr.parse_ulr_dos(subdir / FILE_ULR_TDOS)

    def get_ulr_bands(
        self,
        ulr_dir,
        vertices=None,
        kpath=None,
        npoints=200,
        all_kappa=False,
        nkappa=None,
        wplot=(-0.5, 0.5),
        nwplot=500,
        ngrkf=100,
        nswplot=0,
    ):
        """Ultra-long-range band structure (tasks 720/725,
        src/bandstrulr.f90).

        The ultracell's Brillouin zone is smaller than the unit cell's, so a
        ULR band structure is the unit-cell one **folded** by the
        kappa-points. ``bandstrulr`` unfolds it again: for each kappa it
        shifts the plotting path by :math:`-\\boldsymbol\\kappa`, solves the
        ULR eigenvalue equation, and computes each state's *kappa-point
        character* (``charkpa``) -- the weight with which the ULR eigenvector
        sits in that kappa block. Plotting energy against path distance with
        the character as intensity recovers a band structure in the unit
        cell's own zone, with the long-range order visible as avoided
        crossings and shadow bands.

        `all_kappa=False` (task 720) plots :math:`\\boldsymbol\\kappa = 0`
        only -- one pass, ``nstulr`` bands. `all_kappa=True` (task 725) loops
        over every kappa-point and **appends**, giving ``nkpa`` consecutive
        groups. ``BANDULR.OUT`` carries no record of ``nkpa``, so it is
        derived from the ultracell run's own ``ngridq``/``ngridkpa`` by
        :meth:`_ulr_nkappa`; pass `nkappa` explicitly to override that (and
        see the caveat in that method's docstring).

        Returns ``(distances, energies, characters, spectral)`` where
        `spectral` is the ``(distances, frequencies, sf)`` triple from
        ``BANDSFU.OUT`` -- the same information broadened into an intensity
        map.
        """
        vertices = self._resolve_vertices(vertices, kpath)
        blocks = {
            "plot1d": [(len(vertices), int(npoints))] + [tuple(v) for v in vertices],
            "wplot": [(int(nwplot), int(ngrkf), int(nswplot)), tuple(wplot)],
        }
        task = TASK_ULR_BANDS_ALL_KAPPA if all_kappa else TASK_ULR_BANDS
        subdir = self._run_dependent(ulr_dir, [task], blocks)
        if nkappa is None:
            nkappa = self._ulr_nkappa(ulr_dir) if all_kappa else 1
        distances, energies, characters = parsers_ulr.parse_ulr_bands(
            subdir / FILE_ULR_BAND, nkappa=int(nkappa)
        )
        spectral = parsers_ulr.parse_ulr_band_spectral(
            subdir / FILE_ULR_BAND_SPECTRAL
        )
        return distances, energies, characters, spectral

    def _ulr_nkappa(self, ulr_dir):
        """Number of kappa-points ``nkpa``, derived from the ultracell run's
        input blocks.

        src/genkpakq.f90::

            do i=1,3
              nk=(ngridq(i)+1)/2
              if (ngridkpa(i) > 0) then
                ngridkpa(i)=min(ngridkpa(i),nk)
              else
                ngridkpa(i)=nk
              end if
            end do
            nkpa=ngridkpa(1)*ngridkpa(2)*ngridkpa(3)

        **Caveat**: ``genkpakq`` first calls ``nfftifc(npfftq,3,ngridq)``,
        which rounds ``ngridq`` *up* to the next FFT-friendly size in place,
        so a grid Elk had to adjust gives a larger ``nkpa`` than the input
        block implies. That is why ``get_ulr_bands`` takes an explicit
        ``nkappa=`` escape hatch; a wrong value is caught by
        ``parse_ulr_bands`` (the block count would not divide) rather than
        silently mis-shaping the result.
        """
        stage = self._read_stage_manifest(ulr_dir)
        blocks = stage.get("blocks", {})
        if "ngridq" not in blocks:
            return 1
        ngridq = [int(n) for n in blocks["ngridq"][0]]
        if "ngridkpa" in blocks:
            requested = [int(n) for n in blocks["ngridkpa"][0]]
        else:
            requested = [-1, -1, -1]
        nkpa = 1
        for i in range(3):
            nk = (ngridq[i] + 1) // 2
            nkpa *= min(requested[i], nk) if requested[i] > 0 else nk
        return nkpa

    def _ulr_plot_blocks(self, dimension, box, plane, vertices, kpath, npoints, grid):
        if dimension == 3:
            return {"plot3d": self._plot3d_lines(box, grid or (20, 20, 20))}
        if dimension == 2:
            if plane is None:
                raise ValueError("dimension=2 needs plane=[origin, corner1, corner2]")
            return {"plot2d": self._plot2d_lines(plane, grid or (40, 40))}
        if dimension == 1:
            vertices = self._resolve_vertices(vertices, kpath)
            return {
                "plot1d": [(len(vertices), int(npoints))]
                + [tuple(v) for v in vertices]
            }
        raise ValueError(f"dimension must be 1, 2 or 3, got {dimension}")

    def _ulr_plot(self, ulr_dir, tasks, filenames, dimension, nf, blocks):
        subdir = self._run_dependent(ulr_dir, [tasks[dimension]], blocks)
        path = subdir / filenames[dimension]
        if dimension == 3:
            return volumetric.parse_plot3d(path, nf=nf)
        if dimension == 2:
            return volumetric.parse_plot2d(path, nf=nf)
        return parsers_ulr.parse_plotu1d(path, nf=nf)

    def get_ulr_density(
        self,
        ulr_dir,
        dimension=3,
        box=None,
        plane=None,
        vertices=None,
        kpath=None,
        npoints=500,
        grid=None,
    ):
        """Ultra-long-range charge density plot (tasks 731/732/733,
        src/rhouplot.f90).

        Evaluates :math:`n(\\mathbf r)` of the *ultracell* -- the Q-expanded
        density read back from ``STATE_ULR.OUT`` and resummed on the plotting
        points by ``plotulr`` -- so the plotting path may (and normally does)
        run over **many unit cells**: Elk's own Cr example plots from
        ``(0,0,0)`` to ``(21,0,0)`` in unit-cell lattice coordinates, across
        the full 21-cell ultracell, to show one period of the density wave.
        The plotting vectors are interpreted in unit-cell lattice
        coordinates (``plotu1d``/``plotu3d`` pass ``avec``, not ``avecu``),
        which is exactly what makes coordinates above 1 meaningful.

        `dimension` picks 1D (731, a path through ``vertices``/``kpath``), 2D
        (732, a parallelogram through ``plane``) or 3D (733, a
        parallelepiped through ``box``) -- the same last-digit convention as
        Elk's ordinary plotting tasks.

        Returns the same shapes as the corresponding non-ULR plot: 1D
        ``(distances, values)``, 2D ``(points, values, grid)``, 3D
        ``(points, values)``.
        """
        blocks = self._ulr_plot_blocks(
            dimension, box, plane, vertices, kpath, npoints, grid
        )
        return self._ulr_plot(
            ulr_dir, TASK_ULR_DENSITY, FILE_ULR_DENSITY, dimension, 1, blocks
        )

    def get_ulr_potential(
        self,
        ulr_dir,
        dimension=3,
        box=None,
        plane=None,
        vertices=None,
        kpath=None,
        npoints=500,
        grid=None,
    ):
        """Ultra-long-range Kohn-Sham potential plot (tasks 741/742/743,
        src/potuplot.f90). Same geometry arguments as
        :meth:`get_ulr_density`; the field plotted is
        :math:`v_s(\\mathbf r)` of the ultracell, converted back from the
        Q-representation.

        Upstream note: ``potuplot.f90``'s ``case(742)`` branch calls
        ``open(50)`` where it plainly means ``close(50)``. Re-opening an
        already-connected unit with no ``file=`` is legal Fortran and merely
        changes connection properties, so ``VSU2D.OUT`` is still written and
        flushed when the program exits -- but the file is left open for the
        rest of the run. This is upstream's bug and is left untouched
        (``vendor/elk/`` is byte-for-byte upstream).
        """
        blocks = self._ulr_plot_blocks(
            dimension, box, plane, vertices, kpath, npoints, grid
        )
        return self._ulr_plot(
            ulr_dir, TASK_ULR_POTENTIAL, FILE_ULR_POTENTIAL, dimension, 1, blocks
        )

    def get_ulr_magnetisation(
        self,
        ulr_dir,
        dimension=3,
        box=None,
        plane=None,
        vertices=None,
        kpath=None,
        npoints=500,
        grid=None,
        ndmag=None,
    ):
        """Ultra-long-range magnetisation plot (tasks 771/772/773,
        src/maguplot.f90). Same geometry arguments as
        :meth:`get_ulr_density`.

        This is the headline output of the ULR method: a spin density wave,
        whose whole point is that its period is incommensurate with the unit
        cell, appears here directly as :math:`\\mathbf m(\\mathbf r)` over
        the ultracell.

        The number of columns is ``ndmag``: **1** for a collinear
        calculation (magnetisation along z only) and **3** for a
        non-collinear one. ``maguplot.f90`` stops outright on a
        spin-unpolarised run, so that is refused here first. Getting the
        column count wrong is a *silent* error -- ``parse_plotu1d`` with
        ``nf=1`` on a three-column file happily returns the first component
        as if it were the whole answer -- so the default is not a guess: it
        comes from :meth:`_ndmag`, which reproduces init0.f90's own rule
        (transverse ``bfieldc``/``bfcmt``, ``spinorb``, ``nosource``,
        ``spinsprl``, ``cmagz``) exactly. Pass `ndmag` to override.

        For ``dimension=2`` with ``ndmag == 3``, ``plotu2d`` is called with
        ``tproj=.true.``, so the three components returned are the
        magnetisation **locally projected onto the plotting plane's own
        axes**, not Cartesian x/y/z.
        """
        if not (self.spinpol or self.spinorb):
            raise ValueError(
                "get_ulr_magnetisation() requires a spin-polarised calculation: "
                "maguplot.f90 stops with 'Error(maguplot): spin-unpolarised "
                "calculation' otherwise"
            )
        if ndmag is None:
            ndmag = self._ndmag()
        blocks = self._ulr_plot_blocks(
            dimension, box, plane, vertices, kpath, npoints, grid
        )
        return self._ulr_plot(
            ulr_dir,
            TASK_ULR_MAGNETISATION,
            FILE_ULR_MAGNETISATION,
            dimension,
            int(ndmag),
            blocks,
        )
