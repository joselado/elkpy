"""The complete elk.in input-parameter surface, as data plus a validator,
a renderer and a small discovery API.

Elk reads its input from `elk.in`, a sequence of named blocks
(`docs/elk_manual.txt` sec. 4.4).  `vendor/elk/src/readinput.f90` dispatches
on the block name in one big `select case(trim(block))`; every branch there
is one entry in :data:`BLOCKS` below.  `Calculation` types about a dozen of
those as named Python arguments and passes everything else through its
untyped `extra_blocks` dict, where a misspelt name or a wrong arity only
surfaces as a Fortran error deep inside a subprocess -- often as the
generic "Error(readinput): error reading from elk.in".  This module closes
that gap: it knows every block, its Fortran variable, its type, its shape,
its default and what it does, so a mistake is a Python exception naming the
block and what was expected.

Provenance of each field (this is version-coupled knowledge, in the same
spirit as `spec.py` -- an Elk version bump should mean regenerating this
one table):

* name/aliases/shape/arity  -- the `case(...)` branch and its `read`
  statements in `vendor/elk/src/readinput.f90`; `src` records the line
  number of the branch so any entry can be checked in seconds.
* variable/type/module      -- the Fortran declaration in
  `vendor/elk/src/mod*.f90` (or, for the handful of blocks that set a local
  variable of `readinput` itself, `module="readinput (local)"`).
* default                   -- readinput.f90's own "default values" section
  (lines 48-405 of this release), not the manual, which is occasionally
  stale.
* lo/hi                     -- the range checks the branch itself performs
  before `stop`ping, transcribed so the same rejection happens in Python.
* description               -- `docs/elk_manual.txt` sec. 5 where the block
  is documented there (`dsrc="manual"`), otherwise the comment above the
  variable's declaration in `mod*.f90` (`dsrc="fortran"`).

Coverage is asserted, not assumed: `tests/test_params.py` re-parses
`readinput.f90` at test time and fails if this table and that file disagree
in either direction, so the next Elk version bump reports exactly which
blocks appeared or vanished.

Shapes
------
``scalar``   one line, one value (``opt`` = optional extra values on the
             same line -- Elk's "derivative" tail, e.g. ``rgkmax 7.0 0.1``)
``vector``   one line, ``n`` values (``nmin`` < ``n`` when the branch pads
             the line before reading it, e.g. ``xctype 20`` is legal)
``block``    a fixed sequence of lines, ``lines`` giving (type, n, opt) each
``list``     any number of like-shaped lines, terminated by a blank line
``counted``  a header line (whose ``count_at``-th value is the row count),
             then that many rows
``raw``      lines copied through verbatim (``notes``, Wannier90 extras)
``special``  irregular multi-line layout (``atoms``, ``DFT+U``, ``tm3fix``)
``opaque``   the branch hands the open unit to another reader (``species``)
``deprecated`` the branch only prints "no longer used"

Usage
-----
    >>> from elkpy import params
    >>> params.describe("rgkmax").default
    7.0
    >>> [b.name for b in params.search("smear")]
    ['autoswidth', 'stype', 'swidth']
    >>> params.render_blocks({"maxscl": 40, "ngridk": (4, 4, 4)})
    {'maxscl': [40], 'ngridk': [(4, 4, 4)]}
"""

import difflib
import re
import warnings

__all__ = [
    "BLOCKS", "EXTENSION_BLOCKS", "CATEGORIES", "CALCULATION_OWNED",
    "ParameterError", "ParamBlock", "FortranReal", "Verbatim",
    "describe", "search", "categories", "in_category", "known", "is_known",
    "validate_blocks", "render_blocks", "fortran_real",
]


class ParameterError(ValueError):
    """An elk.in block name or value that Elk would reject."""


# ---------------------------------------------------------------------------
# rendering tokens
# ---------------------------------------------------------------------------
# Both of these exist to slip the right text past inputfile._format_value,
# which renders bool -> .true./.false., float -> "%.10f", str -> 'quoted',
# and anything else -> str(v).  Neither type may be edited into
# inputfile.py from here (that file is shared), so the shim lives here.

def fortran_real(x):
    """Format a real the way Elk's list-directed `read` wants it.

    inputfile._format_value renders a plain float as ``f"{v:.10f}"``, which
    silently turns every tolerance below 1e-10 into ``0.0000000000`` --
    `epsband` (1e-12) and `epsdmat` (1e-8) live exactly there.  This keeps
    full round-trip precision and forces the ``1.0E-12`` form, whose
    mantissa always carries a decimal point (a bare ``1e-12`` is accepted by
    gfortran but is not portable Fortran list-directed input).
    """
    x = float(x)
    s = repr(x)
    if "e" in s or "E" in s:
        mant, exp = re.split("[eE]", s)
        if "." not in mant:
            mant += ".0"
        return "%sE%s" % (mant, exp)
    if "." not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s


class FortranReal(float):
    """A float that renders as a Fortran literal instead of ``%.10f``.

    Subclasses ``float`` so that it is still JSON-serialisable with its real
    value -- `Calculation._basis_signature` json-dumps `extra_blocks` into
    the ground-state cache manifest, so a token that serialised to a
    constant would make two different inputs share a cache entry.
    """

    __slots__ = ()

    def __str__(self):
        return fortran_real(self)

    def __format__(self, spec):
        return fortran_real(self)

    def __repr__(self):
        return fortran_real(self)


class Verbatim(dict):
    """A line copied into elk.in exactly as given, with no quoting.

    Used for the blocks Elk reads with ``read(unit,'(A)')`` rather than
    list-directed: `notes`, `xlwin` (verbatim Wannier90 .win lines) and
    `idxw90` (Elk's own ``1-4,7`` number-list syntax).  A plain `str` would
    come out of inputfile._format_value wrapped in single quotes, which
    ends up inside the note or the .win file.  Subclasses ``dict`` so it is
    neither bool/float/str (hence rendered via ``str(v)``) yet still
    json-serialisable, distinctly per text, for the cache manifest.
    """

    __slots__ = ()

    def __init__(self, text):
        super(Verbatim, self).__init__(verbatim=str(text))

    def __str__(self):
        return self["verbatim"]

    def __repr__(self):
        return "Verbatim(%r)" % self["verbatim"]


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

class ParamBlock(object):
    """One elk.in input block."""

    __slots__ = ("name", "aliases", "type", "shape", "n", "nmin", "opt",
                 "lines", "line", "linemin", "header", "count_at", "maxrows",
                 "nlines", "default", "lo", "lo_ex", "hi", "hi_ex", "var",
                 "module", "cat", "src", "status", "desc", "dsrc", "note",
                 "source")

    def __init__(self, name, type=None, shape="scalar", n=None, nmin=None,
                 opt=0, lines=None, line=None, linemin=None, header=None,
                 count_at=None, maxrows=None, nlines=None, default=None,
                 lo=None, lo_ex=False, hi=None, hi_ex=False, var="",
                 module="", cat="misc", src=0, aliases=(), status="active",
                 desc="", dsrc="fortran", note="", source="elk"):
        self.name = name
        self.aliases = tuple(aliases)
        self.type = type
        self.shape = shape
        self.n = n
        self.nmin = nmin if nmin is not None else n
        self.opt = opt
        self.lines = lines
        self.line = line
        self.linemin = linemin
        self.header = header
        self.count_at = count_at
        self.maxrows = maxrows
        self.nlines = nlines
        self.default = default
        self.lo = lo
        self.lo_ex = lo_ex
        self.hi = hi
        self.hi_ex = hi_ex
        self.var = var
        self.module = module
        self.cat = cat
        self.src = src
        self.status = status
        self.desc = desc
        self.dsrc = dsrc
        self.note = note
        self.source = source

    @property
    def names(self):
        """Every spelling Elk accepts for this block, canonical name first."""
        return (self.name,) + self.aliases

    def signature(self):
        """A one-line description of what this block's value must look like."""
        if self.shape == "scalar":
            s = "one %s" % self.type
            if self.opt:
                s += " (plus up to %d optional trailing %s)" % (self.opt, self.type)
            return s
        if self.shape == "vector":
            if isinstance(self.type, tuple):
                return "one line: '" + " ".join(self.type) + "'"
            s = "%d %ss on one line" % (self.n, self.type)
            if self.nmin != self.n:
                s = "%d to %d %ss on one line (Elk pads the rest)" % (
                    self.nmin, self.n, self.type)
            if self.opt:
                s += " (plus up to %d optional trailing values)" % self.opt
            return s
        if self.shape == "block":
            return "%d lines: %s" % (
                len(self.lines),
                ", ".join("%d %s" % (ln[1], ln[0]) for ln in self.lines))
        if self.shape == "list":
            return "any number of lines of %s, blank-line terminated" % (
                _line_desc(self.line, self.linemin),)
        if self.shape == "counted":
            return ("a header line of %d %s, then that many lines of %s"
                    % (self.header[1], self.header[0],
                       _line_desc(self.line, self.linemin)))
        if self.shape == "raw":
            return "verbatim text line%s" % ("" if self.nlines == 1 else "s")
        if self.shape == "special":
            return "an irregular multi-line block -- see note"
        if self.shape == "opaque":
            return "handed straight to another Fortran reader -- see note"
        return "no value (this block is %s)" % self.status

    def __repr__(self):
        return "<ParamBlock %s (%s, %s)>" % (self.name, self.cat, self.shape)


def _line_desc(line, linemin):
    t, n = line[0], line[1]
    if isinstance(t, tuple):
        return "'" + " ".join(t) + "'"
    if linemin is not None and linemin != n:
        return "%d to %d %ss" % (linemin, n, t)
    return "%d %s%s" % (n, t, "s" if n != 1 else "")


def _b(*args, **kw):
    return ParamBlock(*args, **kw)


#: Every input block of the vendored Elk release, keyed by canonical name.
#: Generated from vendor/elk/src/readinput.f90 (see the module docstring for
#: where each field comes from) and checked by tests/test_params.py.
_UPSTREAM = [
    _b("tasks", shape='list', line=('int', 1), maxrows=40, var='tasks', module='modmain',
       cat='control', src=421, desc='the list of tasks to perform, one integer per line',
       dsrc='fortran'),
    _b("species", shape='opaque', var='-', cat='structure', src=448,
       desc="generate a species file for each 'Z symbol name Rmin Rmt Rmax nrmt' line and stop (see src/genspecies.f90)",
       dsrc='fortran',
       note='calls genspecies(50), which reads its own multi-line species-generation format from the same unit; not typed here'),
    _b("fspecies", shape='list', line=(('real', 'str'), 2), var='-', cat='structure', src=451,
       desc="generate fractional-nuclear-charge species files from 'Z symbol' lines (Z < 0), then stop",
       dsrc='fortran', note="each line is 'Z symbol' with Z<0; calls genfspecies"),
    _b("avec", shape='block', lines=(('real', 3, 3), ('real', 3, 3), ('real', 3, 3)),
       default=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), var='avec',
       module='modmain', cat='structure', src=476,
       desc='the three lattice vectors, one per line, in atomic units (Bohr)', dsrc='fortran'),
    _b("scale", 'real', default=1.0, var='sc', module='readinput (local)', cat='structure',
       src=482, desc='lattice vector scaling factor', dsrc='manual'),
    _b("scale1", 'real', default=1.0, var='sc1', module='readinput (local)', cat='structure',
       src=484, desc='scaling factor for the first lattice vector', dsrc='fortran'),
    _b("scale2", 'real', default=1.0, var='sc2', module='readinput (local)', cat='structure',
       src=486, desc='scaling factor for the second lattice vector', dsrc='fortran'),
    _b("scale3", 'real', default=1.0, var='sc3', module='readinput (local)', cat='structure',
       src=488, desc='scaling factor for the third lattice vector', dsrc='fortran'),
    _b("scalex", 'real', default=1.0, var='scx', module='readinput (local)', cat='structure',
       src=490, desc='scaling factor for the x-component of every lattice vector',
       dsrc='fortran'),
    _b("scaley", 'real', default=1.0, var='scy', module='readinput (local)', cat='structure',
       src=492, desc='scaling factor for the y-component of every lattice vector',
       dsrc='fortran'),
    _b("scalez", 'real', default=1.0, var='scz', module='readinput (local)', cat='structure',
       src=494, desc='scaling factor for the z-component of every lattice vector',
       dsrc='fortran'),
    _b("epslat", 'real', default=1e-06, lo=0.0, lo_ex=True, var='epslat', module='modmain',
       cat='structure', src=496,
       desc='vectors with lengths less than this are considered zero', dsrc='manual'),
    _b("primcell", 'bool', default=False, var='primcell', module='modmain', cat='structure',
       src=504, desc='.true. if the primitive unit cell should be found', dsrc='manual'),
    _b("tshift", 'bool', default=True, var='tshift', module='modmain', cat='structure',
       src=506,
       desc='set to .true. if the crystal can be shifted so that the atom closest to the origin is exactly at the origin',
       dsrc='manual'),
    _b("autokpt", 'bool', default=False, var='autokpt', module='modmain', cat='kpoints',
       src=508, desc='.true. if the k-point set is to be determined automatically',
       dsrc='manual'),
    _b("radkpt", 'real', default=40.0, lo=0.0, lo_ex=True, var='radkpt', module='modmain',
       cat='kpoints', src=510, desc='radius of sphere used to determine k-point density',
       dsrc='manual'),
    _b("ngridk", 'int', shape='vector', n=3, opt=3, default=(1, 1, 1), lo=1, var='ngridk',
       module='modmain', cat='kpoints', src=518, desc='the k-point mesh sizes', dsrc='manual'),
    _b("vkloff", 'real', shape='vector', n=3, default=(0.0, 0.0, 0.0), lo=0.0, hi=1.0,
       hi_ex=True, var='vkloff', module='modmain', cat='kpoints', src=529,
       desc='the k-point offset vector in lattice coordinates', dsrc='manual'),
    _b("reducek", 'int', default=1, var='reducek', module='modmain', cat='kpoints', src=538,
       desc='type of reduction of the k-point set', dsrc='manual'),
    _b("ngridq", 'int', shape='vector', n=3, default=(-1, -1, -1), lo=1, var='ngridq',
       module='modmain', cat='kpoints', src=540, desc='the phonon q-point mesh sizes',
       dsrc='manual',
       note='the default (-1,-1,-1) is a sentinel meaning "use ngridk"; it is '
            'not itself a legal input, since readinput.f90:540 refuses any '
            'component below 1'),
    _b("reduceq", 'int', default=1, var='reduceq', module='modmain', cat='kpoints', src=548,
       desc='type of reduction of the q-point set', dsrc='manual'),
    _b("rgkmax", 'real', opt=1, default=7.0, lo=0.0, lo_ex=True, var='rgkmax',
       module='modmain', cat='basis', src=550,
       desc='R_MT^min x max{|G+k|} -- the APW basis cut-off, gkmax = rgkmax/R_MT^min',
       dsrc='fortran'),
    _b("gmaxvr", 'real', opt=1, default=12.0, var='gmaxvr', module='modmain', cat='basis',
       src=560,
       desc='maximum length of |G| for expanding the interstitial density and potential',
       dsrc='manual'),
    _b("lmaxapw", 'int', opt=1, default=8, lo=0, var='lmaxapw', module='modmain', cat='basis',
       src=564, desc='angular momentum cut-off for the APW functions', dsrc='manual'),
    _b("lmaxo", 'int', opt=1, default=6, lo=3, var='lmaxo', module='modmain', cat='basis',
       src=581, aliases=('lmaxvr',),
       desc='angular momentum cut-off for the muffin-tin density and potential',
       dsrc='manual'),
    _b("lmaxi", 'int', default=1, lo=1, var='lmaxi', module='modmain', cat='basis', src=591,
       aliases=('lmaxinr',),
       desc='angular momentum cut-off for the muffin-tin density and potential on the inner part of the muffin-tin',
       dsrc='manual'),
    _b("lmaxmat", shape='deprecated', var='-', cat='deprecated', src=599, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("fracinr", 'real', default=0.01, var='fracinr', module='modmain', cat='basis', src=602,
       desc='fraction of the muffin-tin radius up to which lmaxi is used as the real 0.01 angular momentum cut-off p',
       dsrc='manual'),
    _b("trhonorm", 'bool', default=True, var='trhonorm', module='modmain', cat='basis',
       src=604,
       desc='trhonorm is .true. if the density is to be normalised after every iteration',
       dsrc='fortran'),
    _b("spinpol", 'bool', default=False, var='spinpol', module='modmain', cat='magnetism',
       src=606, desc='set to .true. if a spin-polarised calculation is required',
       dsrc='manual'),
    _b("spinorb", 'bool', default=False, var='spinorb', module='modmain', cat='magnetism',
       src=608, desc='set to .true. if a spin-orbit coupling is required', dsrc='manual'),
    _b("socscf", 'real', default=1.0, lo=0.0, var='socscf', module='modmain', cat='magnetism',
       src=610, desc='scaling factor for the spin-orbit coupling term in the Hamiltonian',
       dsrc='manual'),
    _b("bforb", 'bool', default=False, var='bforb', module='modmain', cat='magnetism',
       src=618, desc='.true. if the external B-field-orbit coupling term should be included',
       dsrc='manual'),
    _b("bfdmag", 'bool', default=False, var='bfdmag', module='modmain', cat='magnetism',
       src=620,
       desc='.true. if the external B-field diamagnetic coupling term should be included',
       dsrc='manual'),
    _b("xctype", 'int', shape='vector', n=3, nmin=1, default=(3, 0, 0), var='xctype',
       module='modmain', cat='xc', src=622,
       desc='integers defining the type of exchange-correlation functional to be used',
       dsrc='manual'),
    _b("xctsp", 'int', shape='vector', n=3, nmin=1, default=(3, 0, 0), var='xctsp',
       module='modmain', cat='xc', src=626,
       desc='exchange-correlation type for atomic species (the converged ground-state of the crystal does not depend on this choice)',
       dsrc='fortran'),
    _b("ktype", 'int', shape='vector', n=3, nmin=1, default=(52, 0, 0), var='ktype',
       module='modmain', cat='xc', src=630, desc='kinetic energy density functional type',
       dsrc='fortran'),
    _b("stype", 'int', default=3, var='stype', module='modmain', cat='occupation', src=640,
       desc='integer defining the type of smearing to be used', dsrc='manual'),
    _b("swidth", 'real', default=0.001, lo=1e-09, var='swidth', module='modmain',
       cat='occupation', src=642,
       desc='width of the smooth approximation to the Dirac delta function', dsrc='manual'),
    _b("autoswidth", 'bool', default=False, var='autoswidth', module='modmain',
       cat='occupation', src=651,
       desc='.true. if the smearing parameter swidth should be determined automatically',
       dsrc='manual'),
    _b("mstar", 'real', default=10.0, lo=0.0, lo_ex=True, var='mstar', module='modmain',
       cat='occupation', src=653,
       desc='value of the effective mass parameter used for adaptive determination of swidth',
       dsrc='manual'),
    _b("epsocc", 'real', default=1e-10, lo=0.0, lo_ex=True, var='epsocc', module='modmain',
       cat='convergence', src=661,
       desc='smallest occupancy for which a state will contribute to the density',
       dsrc='manual'),
    _b("epschg", 'real', default=0.001, lo=0.0, lo_ex=True, var='epschg', module='modmain',
       cat='convergence', src=669,
       desc='maximum allowed error in the calculated total charge beyond which a warning message will be issued',
       dsrc='manual'),
    _b("nempty", 'real', opt=1, default=4.0, lo=0.0, lo_ex=True, var='nempty0',
       module='modmain', cat='basis', src=677, aliases=('nempty0',),
       desc='the number of empty states per atom and spin', dsrc='manual'),
    _b("mixtype", 'int', default=3, var='mixtype', module='modmain', cat='convergence',
       src=687, desc='type of mixing required for the potential', dsrc='manual'),
    _b("mixsave", 'bool', default=False, var='mixsave', module='modmain', cat='convergence',
       src=689,
       desc='.true. if the mixer work array is to be saved during a ground-state run',
       dsrc='manual'),
    # readinput.f90:691 is the one branch whose behaviour depends on WHICH of
    # its names was used: 'amixpm' reads both elements, 'beta0' and 'betamax'
    # read one each, so they are three entries here rather than one with two
    # aliases.  Note also that Elk range-checks the two elements differently
    # -- amixpm(1) only has to be >= 0, amixpm(2) must be in [0,1] -- so the
    # vector form carries the weaker bound and betamax the stronger one.
    _b("amixpm", 'real', shape='vector', n=2, default=(0.05, 1.0), lo=0.0,
       var='amixpm', module='modmain', cat='convergence', src=691,
       desc='adaptive mixing parameters: the initial and maximum mixing '
            'weights (beta0, betamax)', dsrc='manual',
       note='readinput.f90:699,705 checks amixpm(1) >= 0 and amixpm(2) in '
            '[0,1]; see the beta0 and betamax blocks, which set one each'),
    _b("beta0", 'real', default=0.05, lo=0.0, var='amixpm(1)', module='modmain',
       cat='convergence', src=691, desc='initial adaptive mixing weight',
       dsrc='manual',
       note='the same readinput.f90 branch as amixpm, but taking a SINGLE '
            'value: it sets amixpm(1) alone'),
    _b("betamax", 'real', default=1.0, lo=0.0, hi=1.0, var='amixpm(2)',
       module='modmain', cat='convergence', src=691,
       desc='maximum adaptive mixing weight', dsrc='manual',
       note='the same readinput.f90 branch as amixpm, but taking a SINGLE '
            'value: it sets amixpm(2) alone'),
    _b("mixsdb", 'int', default=5, lo=2, var='mixsdb', module='modmain', cat='convergence',
       src=712, desc='subspace dimension for Broyden mixing', dsrc='manual'),
    _b("broydpm", 'real', shape='vector', n=2, default=(0.4, 0.15), lo=0.0, hi=1.0,
       var='broydpm', module='modmain', cat='convergence', src=720,
       desc='Broyden mixing parameters alpha and w0', dsrc='manual'),
    _b("mixrho", 'bool', default=False, var='mixrho', module='modmain', cat='convergence',
       src=730,
       desc='if mixrho is .true. then the (density, magnetisation) is mixed, otherwise the (potential, magnetic field)',
       dsrc='fortran'),
    _b("maxscl", 'int', default=200, lo=0, var='maxscl', module='modmain', cat='convergence',
       src=732, desc='maximum number of self-consistent loops allowed', dsrc='manual'),
    _b("epspot", 'real', default=1e-06, var='epspot', module='modmain', cat='convergence',
       src=740, desc='convergence criterion for the Kohn-Sham potential and field',
       dsrc='manual'),
    _b("epsengy", 'real', default=0.0001, var='epsengy', module='modmain', cat='convergence',
       src=742, desc='convergence criterion for the total energy', dsrc='manual'),
    _b("epsforce", 'real', default=0.005, var='epsforce', module='modmain', cat='convergence',
       src=744,
       desc='convergence tolerance for the forces during a geometry optimisation run',
       dsrc='manual'),
    _b("epsstress", 'real', default=0.002, var='epsstress', module='modmain',
       cat='convergence', src=746,
       desc='convergence tolerance for the stress tensor during a geometry optimisation run with lattice vector relaxation',
       dsrc='manual'),
    _b("sppath", 'str', default="", var='sppath', module='modmain', cat='structure', src=748,
       desc='path where the species files can be found', dsrc='manual'),
    _b("scrpath", 'str', default="", var='scrpath', module='modmain', cat='structure',
       src=751, desc='scratch space path', dsrc='manual'),
    _b("molecule", 'bool', default=False, var='molecule', module='modmain', cat='structure',
       src=753, desc='.true. if the system is an isolated molecule', dsrc='manual'),
    _b("atoms", shape='special', lo=1, var='nspecies/spfname/natoms/atposl/bfcmt0',
       module='modmain', cat='structure', src=755,
       desc='the atomic species, positions and muffin-tin magnetic fields', dsrc='fortran',
       note="nested: nspecies, then per species a filename line, a count line and that many 'x y z [bx by bz [dx dy dz]]' lines; normally built by Structure/Calculation"),
    _b("plot1d", shape='counted', line=('real', 3), header=('int', 2), count_at=0,
       default=((2, 200), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), lo=1, var='nvp1d/npp1d/vvlp1d',
       module='modmain', cat='plotting', src=795,
       desc="'nvp1d npp1d' then nvp1d vertices, in lattice coordinates, of the line along which quantities are plotted",
       dsrc='fortran'),
    _b("ip0gw", 'int', default=1, lo=1, var='ip0gw', module='modgw', cat='plotting', src=814,
       aliases=('ip01d',), desc='starting point for GW band structure', dsrc='fortran'),
    _b("plot2d", shape='block',
       lines=(('real', 3, 0), ('real', 3, 0), ('real', 3, 0), ('int', 2, 0)),
       default=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (40, 40)),
       var='vclp2d/np2d', module='modmain', cat='plotting', src=822,
       desc='the plot plane: origin, first corner, second corner (lattice coordinates), then the two grid sizes',
       dsrc='fortran'),
    _b("plot3d", shape='block',
       lines=(('real', 3, 0), ('real', 3, 0), ('real', 3, 0), ('real', 3, 0), ('int', 3, 0)),
       default=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (20, 20, 20)),
       var='vclp3d/np3d', module='modmain', cat='plotting', src=833,
       desc='the plot box: origin and three corners (lattice coordinates), then the three grid sizes',
       dsrc='fortran'),
    _b("wplot", shape='block', lines=(('int', 3, 0), ('real', 2, 0)),
       default=((500, 100, 1), (-0.5, 0.5)), lo=2, var='nwplot/ngrkf/nswplot/wplot',
       module='modmain', cat='dos', src=845, aliases=('dos',),
       desc="'nwplot ngrkf nswplot' then the frequency/energy window 'wmin wmax' for DOS and optics plots",
       dsrc='fortran',
       note="the first line is 'nwplot ngrkf nswplot', the second 'wmin wmax'"),
    _b("dosocc", 'bool', default=False, var='dosocc', module='modmain', cat='dos', src=872,
       desc='dosocc is .true. if the DOS is to be weighted by the occupancy', dsrc='fortran'),
    _b("tpdos", 'bool', default=True, var='tpdos', module='modmain', cat='dos', src=874,
       desc='tpdos is .true. if the partial DOS should be calculated', dsrc='fortran'),
    _b("dosmsum", 'bool', default=False, var='dosmsum', module='modmain', cat='dos', src=876,
       desc='.true. if the partial DOS is to be summed over m', dsrc='manual'),
    _b("dosssum", 'bool', default=False, var='dosssum', module='modmain', cat='dos', src=878,
       desc='.true. if the partial DOS is to be summed over spin', dsrc='manual'),
    _b("lmirep", 'bool', default=True, var='lmirep', module='modmain', cat='dos', src=880,
       desc='.true. if the Ylm basis is to be transformed into the basis of irreducible representations of the site symmetries for DOS plotting',
       dsrc='manual'),
    _b("atpopt", 'int', default=1, var='atpopt', module='modmain', cat='geometry', src=882,
       desc='atomic position optimisation type 0 : no optimisation 1 : unconstrained optimisation',
       dsrc='fortran'),
    _b("maxatpstp", 'int', default=200, lo=1, var='maxatpstp', module='modmain',
       cat='geometry', src=884, aliases=('maxatmstp',),
       desc='maximum number of atomic position optimisation steps', dsrc='fortran'),
    _b("tau0atp", 'real', default=0.2, var='tau0atp', module='modmain', cat='geometry',
       src=892, aliases=('tau0atm',),
       desc='the step size to be used for atomic position optimisation', dsrc='manual'),
    _b("deltast", 'real', default=0.005, lo=0.0, lo_ex=True, var='deltast', module='modmain',
       cat='geometry', src=894,
       desc='size of the change in lattice vectors used for calculating the stress tensor',
       dsrc='manual'),
    _b("avecref", shape='block', lines=(('real', 3, 0), ('real', 3, 0), ('real', 3, 0)),
       default=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), var='avecref',
       module='modmain', cat='structure', src=902,
       desc='first reference lattice vector, etc.', dsrc='manual'),
    _b("latvopt", 'int', default=0, var='latvopt', module='modmain', cat='geometry', src=906,
       desc='type of lattice vector optimisation to be performed during structural relaxation',
       dsrc='manual'),
    _b("maxlatvstp", 'int', default=30, lo=1, var='maxlatvstp', module='modmain',
       cat='geometry', src=908, desc='maximum number of lattice vector optimisation steps',
       dsrc='fortran'),
    _b("tau0latv", 'real', default=0.2, var='tau0latv', module='modmain', cat='geometry',
       src=916, desc='the step size to be used for lattice vector optimisation',
       dsrc='manual'),
    _b("nstfsp", shape='deprecated', var='-', cat='deprecated', src=918, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("lradstp", 'int', default=4, lo=1, var='lradstp', module='modmain', cat='basis',
       src=921, desc='radial step length for determining coarse radial mesh', dsrc='manual'),
    _b("chgexs", 'real', opt=1, default=0.0, var='chgexs', module='modmain', cat='occupation',
       src=929, desc='excess electronic charge', dsrc='manual'),
    _b("nprad", shape='deprecated', var='-', cat='deprecated', src=933, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("scissor", 'real', default=0.0, var='scissor', module='modmain', cat='dos', src=936,
       desc='the scissor correction', dsrc='manual'),
    _b("noptcomp", 'int', default=1, lo=1, hi=27, var='noptcomp', module='modmain',
       cat='optics', src=938, desc='number of optical matrix components required',
       dsrc='fortran'),
    _b("optcomp", shape='list', line=('int', 3), linemin=1, maxrows=27, var='optcomp',
       module='modmain', cat='optics', src=947,
       desc="the components of the optical tensor to calculate, one 'i j k' line each",
       dsrc='fortran'),
    _b("intraband", 'bool', default=False, var='intraband', module='modmain', cat='optics',
       src=980,
       desc='.true. if the intraband (Drude-like) contribution is to be added to the dieletric tensor',
       dsrc='manual'),
    _b("evaltol", shape='deprecated', var='-', cat='deprecated', src=982, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("deband", shape='deprecated', var='-', cat='deprecated', src=985, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("epsband", 'real', default=1e-12, lo=0.0, lo_ex=True, var='epsband', module='modmain',
       cat='basis', src=988, desc='convergence tolerance for determining band energies',
       dsrc='manual'),
    _b("demaxbnd", 'real', default=2.5, lo=0.0, lo_ex=True, var='demaxbnd', module='modmain',
       cat='basis', src=996,
       desc='maximum allowed change in energy during band energy search; enforced only if default energy is less than zero',
       dsrc='fortran'),
    _b("autolinengy", 'bool', default=False, var='autolinengy', module='modmain', cat='basis',
       src=1004,
       desc='.true. if the fixed linearisation energies are to be determined automatically',
       dsrc='manual'),
    _b("dlefe", 'real', default=-0.1, var='dlefe', module='modmain', cat='basis', src=1006,
       desc='difference between the fixed linearisation energy and the Fermi energy',
       dsrc='manual'),
    _b("autodlefe", 'bool', default=True, var='autodlefe', module='modmain', cat='basis',
       src=1008,
       desc='.true. if the difference between the fixed linearisation energies and Fermi energy should be found automatically',
       dsrc='manual'),
    _b("deapw", 'real', default=0.2, var='deapw', module='modmain', cat='basis', src=1010,
       desc='energy step size used for APW numerical derivatives', dsrc='fortran'),
    _b("delorb", 'real', default=0.05, var='delorb', module='modmain', cat='basis', src=1018,
       desc='energy step size used for local-orbital numerical derivatives', dsrc='fortran'),
    _b("bfieldc", 'real', shape='vector', n=3, opt=3, default=(0.0, 0.0, 0.0), var='bfieldc0',
       module='modmain', cat='magnetism', src=1026,
       desc='global external magnetic field in Cartesian coordinates', dsrc='manual'),
    _b("efieldc", 'real', shape='vector', n=3, default=(0.0, 0.0, 0.0), var='efieldc',
       module='modmain', cat='fields', src=1030,
       desc='electric field vector in Cartesian coordinates', dsrc='fortran'),
    _b("dmaxefc", 'real', default=1000000.0, lo=0, var='dmaxefc', module='modmain',
       cat='magnetism', src=1032,
       desc='maximum distance over which the electric field is applied', dsrc='fortran'),
    _b("afieldc", 'real', shape='vector', n=3, opt=3, default=(0.0, 0.0, 0.0), var='afieldc',
       module='modmain', cat='fields', src=1040,
       desc='vector potential A-field in Cartesian coordinates which couples to the paramagnetic current',
       dsrc='fortran'),
    _b("afspc", shape='block', lines=(('real', 3, 3), ('real', 3, 3), ('real', 3, 3)),
       default=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), var='afspc',
       module='modmain', cat='fields', src=1044,
       desc='spin-dependent vector potential (3 x 3 tensor) in Cartesian coordinates',
       dsrc='fortran'),
    _b("fsmtype", 'int', default=0, var='fsmtype', module='modmain', cat='magnetism',
       src=1050, aliases=('fixspin',), desc='fixed spin moment (FSM) type', dsrc='manual'),
    _b("momfix", 'real', shape='vector', n=3, opt=3, default=(0.0, 0.0, 0.0), var='momfix',
       module='modmain', cat='magnetism', src=1052,
       desc='the desired total moment for a FSM calculation', dsrc='manual'),
    _b("momfixm", 'real', default=0.0, lo=0.0, var='momfixm', module='modmain',
       cat='magnetism', src=1056,
       desc='the desired total moment magnitude for a FSM calculation', dsrc='manual'),
    _b("mommtfix", shape='list', line=(('int', 'int', 'real', 'real', 'real'), 5),
       var='mommtfix', module='modmain', cat='magnetism', src=1064,
       desc="the fixed muffin-tin spin moments: one 'is ia mx my mz' line per constrained atom, terminated by a blank line",
       dsrc='fortran'),
    _b("mommtfixm", shape='list', line=(('int', 'int', 'real'), 3), var='mommtfixm',
       module='modmain', cat='magnetism', src=1078,
       desc="the fixed muffin-tin spin moment magnitudes: one 'is ia m' line per constrained atom, terminated by a blank line",
       dsrc='fortran'),
    _b("taufsm", 'real', default=0.01, lo=0.0, var='taufsm', module='modmain',
       cat='magnetism', src=1092,
       desc='the step size to be used when finding the effective magnetic field in fixed spin moment calculations',
       dsrc='manual'),
    _b("autormt", shape='deprecated', var='-', cat='deprecated', src=1100,
       status='deprecated', desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("rmtdelta", 'real', default=0.05, lo=0.0, var='rmtdelta', module='modmain',
       cat='basis', src=1103, desc='minimum allowed distance between muffin-tin surfaces',
       dsrc='manual'),
    _b("isgkmax", 'int', default=-1, var='isgkmax', module='modmain', cat='basis', src=1109,
       desc='species for which the muffin-tin radius will be used for calculating gkmax',
       dsrc='manual'),
    _b("nosym", 'bool', default=False, var='symtype', module='modmain', cat='kpoints',
       src=1111, desc='.true. to switch off all crystal symmetry (equivalent to symtype=0)',
       dsrc='fortran', note='.true. sets symtype=0 (no symmetry)'),
    _b("symtype", 'int', default=1, lo=0, hi=2, var='symtype', module='modmain',
       cat='kpoints', src=1114,
       desc='type of symmetry allowed for the crystal 0 : only the identity element is used 1 : full symmetry group is used 2 : only symmorphic symmetries are allowed',
       dsrc='fortran'),
    _b("deltaph", 'real', default=0.01, lo=0.0, lo_ex=True, var='deltaph', module='modphonon',
       cat='phonons', src=1122,
       desc='size of the atomic displacement used for calculating dynamical matrices',
       dsrc='manual'),
    _b("phwrite", shape='counted', line=('real', 3), header=('int', 1), count_at=0, lo=1,
       var='nphwrt/vqlwrt', module='modphonon', cat='phonons', src=1130,
       desc='number of q-points for which phonon modes are to be found', dsrc='manual'),
    _b("notes", shape='raw', var='notes', module='modmain', cat='io', src=1143,
       desc='the ith line of the notes', dsrc='manual'),
    _b("tforce", 'bool', default=False, var='tforce', module='modmain', cat='geometry',
       src=1153,
       desc='set to .true. if the force should be calculated at the end of the self-consistent cycle',
       dsrc='manual'),
    _b("tfibs", shape='deprecated', var='-', cat='deprecated', src=1155, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("maxitoep", 'int', default=400, lo=1, var='maxitoep', module='modmain', cat='xc',
       src=1158,
       desc='maximum number of iterations when solving the exact exchange integral equations',
       dsrc='manual'),
    _b("tauoep", shape='deprecated', var='-', cat='deprecated', src=1166, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("tau0oep", 'real', default=0.1, lo=0.0, var='tau0oep', module='modmain', cat='xc',
       src=1169, desc='initial step length for the OEP iterative solver', dsrc='manual'),
    _b("kstlist", shape='list', line=('int', 3), linemin=2, maxrows=20, var='kstlist',
       module='modmain', cat='kpoints', src=1177,
       desc="the k-point and state list: one 'ik ist [jst]' line each", dsrc='fortran'),
    _b("vklem", 'real', shape='vector', n=3, default=(0.0, 0.0, 0.0), var='vklem',
       module='modmain', cat='kpoints', src=1204,
       desc='the k-point in lattice coordinates at which to compute the effective mass tensors',
       dsrc='manual'),
    _b("deltaem", 'real', default=0.025, lo=0.0, lo_ex=True, var='deltaem', module='modmain',
       cat='kpoints', src=1206,
       desc='the size of the k-vector displacement used when calculating numerical derivatives for the effective mass tensor',
       dsrc='manual'),
    _b("ndspem", 'int', default=1, lo=1, hi=4, var='ndspem', module='modmain', cat='kpoints',
       src=1214,
       desc='the number of k-vector displacements in each direction around vklem when computing the numerical derivatives for the effective mass tensor',
       dsrc='manual'),
    _b("nosource", 'bool', default=False, var='nosource', module='modmain', cat='magnetism',
       src=1222,
       desc='when set to .true., source fields are projected out of the exchange-correlation magnetic field',
       dsrc='manual'),
    _b("spinsprl", 'bool', default=False, var='spinsprl', module='modmain', cat='magnetism',
       src=1224, desc='set to .true. if a spin-spiral calculation is required', dsrc='manual'),
    _b("ssdph", 'bool', default=True, var='ssdph', module='modmain', cat='magnetism',
       src=1226,
       desc='set to .true. if a complex de-phasing factor is to be used in spin-spiral calculations',
       dsrc='manual'),
    _b("vqlss", 'real', shape='vector', n=3, opt=3, default=(0.0, 0.0, 0.0), var='vqlss',
       module='modmain', cat='magnetism', src=1228,
       desc='the q-vector of the spin-spiral state in lattice coordinates', dsrc='manual'),
    _b("nwrite", 'int', default=0, var='nwrite', module='modmain', cat='convergence',
       src=1232,
       desc='number of self-consistent loops after which STATE.OUT is to be written',
       dsrc='manual'),
    _b("DFT+U", shape='special', var='dftu/inpdftu/ujdu/fdu/edu/lamdu/udufix',
       module='moddftu', cat='dftu', src=1234, aliases=('dft+u', 'lda+u'),
       desc='type of DFT+U calculation', dsrc='manual',
       note="header line 'dftu inpdftu', then one 'is l <parameters>' line per correlated species, terminated by a blank line; the parameter count depends on inpdftu"),
    _b("tmwrite", 'bool', default=False, var='tmwrite', module='moddftu', cat='magnetism',
       src=1304, aliases=('tmomlu',),
       desc='set to .true. if the tensor moments and the corresponding decomposition of DFT+U energy should be calculated at every loop of the self-consistent cycle',
       dsrc='manual'),
    _b("readadu", shape='deprecated', var='-', cat='deprecated', src=1306,
       aliases=('readalu',), status='deprecated',
       desc='set to .true. if the interpolation constant for DFT+U should be read from file rather than calculated',
       dsrc='manual'),
    _b("rdmxctype", 'int', default=2, var='rdmxctype', module='modrdm', cat='rdmft', src=1309,
       desc='xc functional', dsrc='fortran'),
    _b("rdmmaxscl", 'int', default=2, lo=0, var='rdmmaxscl', module='modrdm', cat='rdmft',
       src=1311, desc='maximum number of self-consistent loops', dsrc='fortran'),
    _b("maxitn", 'int', default=200, var='maxitn', module='modrdm', cat='rdmft', src=1318,
       desc='maximum number of iterations for occupation number optimisation', dsrc='fortran'),
    _b("maxitc", 'int', default=0, var='maxitc', module='modrdm', cat='rdmft', src=1320,
       desc='maximum number of iteration for natural orbital optimisation', dsrc='fortran'),
    _b("taurdmn", 'real', default=0.5, lo=0.0, var='taurdmn', module='modrdm', cat='rdmft',
       src=1322, desc='step size for occupation numbers', dsrc='fortran'),
    _b("taurdmc", 'real', default=0.25, lo=0.0, var='taurdmc', module='modrdm', cat='rdmft',
       src=1330, desc='step size for natural orbital coefficients', dsrc='fortran'),
    _b("rdmalpha", 'real', default=0.656, lo=0.0, lo_ex=True, hi=1.0, hi_ex=True,
       var='rdmalpha', module='modrdm', cat='rdmft', src=1338,
       desc='exponent for the Power and hybrid functionals', dsrc='fortran'),
    _b("rdmtemp", 'real', default=0.0, lo=0.0, var='rdmtemp', module='modrdm', cat='rdmft',
       src=1346, desc='temperature', dsrc='fortran'),
    _b("reducebf", 'real', default=1.0, lo=0.5, hi=1.0, var='reducebf', module='modmain',
       cat='magnetism', src=1354, desc='reduction factor for the external magnetic fields',
       dsrc='manual'),
    _b("ptnucl", 'bool', default=True, var='ptnucl', module='modmain', cat='basis', src=1362,
       desc='ptnucl is .true. if the nuclei are to be treated as point charges, if .false. the nuclei have a finite spherical distribution',
       dsrc='fortran'),
    _b("tefvr", 'bool', default=True, var='tefvr', module='modmain', cat='basis', src=1364,
       aliases=('tseqr',),
       desc='set to .true. if a real symmetric eigenvalue solver should be used for crystals which have inversion symmetry',
       dsrc='manual'),
    _b("tefvs", shape='deprecated', var='-', cat='deprecated', src=1366, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("mefvs", 'int', default=-1, var='mefvs', module='modmain', cat='basis', src=1369,
       desc='parameter determining the size of the subspace used for the firstvariational eigenvalue problem',
       dsrc='manual'),
    _b("tefvit", shape='deprecated', var='-', cat='deprecated', src=1371, aliases=('tseqit',),
       status='deprecated', desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("minitefv", shape='deprecated', var='-', cat='deprecated', src=1374,
       aliases=('minseqit',), status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("nefvit", shape='deprecated', var='-', cat='deprecated', src=1377,
       aliases=('maxitefv', 'maxseqit', 'nseqit'), status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("befvit", shape='deprecated', var='-', cat='deprecated', src=1380, aliases=('bseqit',),
       status='deprecated', desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("epsefvit", shape='deprecated', var='-', cat='deprecated', src=1383,
       aliases=('epsseqit',), status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("tauseq", shape='deprecated', var='-', cat='deprecated', src=1386, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("vecql", 'real', shape='vector', n=3, default=(0.0, 0.0, 0.0), var='vecql',
       module='modmain', cat='kpoints', src=1389,
       desc='q-vector in lattice and Cartesian coordinates for calculating the matrix elements <i,k+q| exp(iq.r) |j,k>',
       dsrc='fortran'),
    _b("mustar", 'real', default=0.15, var='mustar', module='modphonon', cat='phonons',
       src=1391,
       desc='Coulomb pseudopotential, mu* , used in the McMillan-Allen-Dynes equation',
       dsrc='manual'),
    _b("sqaxis", 'real', shape='vector', n=3, default=(0.0, 0.0, 1.0), var='sqaxis',
       module='modmain', cat='magnetism', src=1393, aliases=('sqados',),
       desc='spin-quantisation axis in Cartesian coordinates used when plotting the spin-resolved DOS and band structure (z-axis by default)',
       dsrc='fortran'),
    _b("test", 'bool', default=False, var='test', module='modtest', cat='testing', src=1395,
       desc='if test is .true. then the test variables are written to file', dsrc='fortran'),
    _b("frozencr", shape='deprecated', var='-', cat='deprecated', src=1397,
       status='deprecated', desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("spincore", 'bool', default=False, var='spincore', module='modmain', cat='magnetism',
       src=1400, desc='set to .true. if the core should be spin-polarised', dsrc='manual'),
    _b("solscf", 'real', default=1.0, lo=0.0, var='solscf', module='readinput (local)',
       cat='basis', src=1402,
       desc='scaling factor for the speed of light (solsc = sol*solscf); the non-relativistic limit is solscf -> infinity',
       dsrc='fortran'),
    _b("emaxelnes", 'real', default=-1.2, var='emaxelnes', module='modmain', cat='response',
       src=1410, desc='maximum allowed initial-state eigenvalue for ELNES calculations',
       dsrc='manual'),
    _b("wsfac", 'real', shape='vector', n=2, default=(-1100000.0, 1100000.0), var='wsfac',
       module='modmain', cat='response', src=1412,
       desc='energy window to be used when calculating density or magnetic structure factors',
       dsrc='manual'),
    _b("vhmat", shape='block', lines=(('real', 3, 0), ('real', 3, 0), ('real', 3, 0)),
       default=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), var='vhmat',
       module='modpw', cat='response', src=1414,
       desc='the 3x3 matrix applied to the reciprocal H-vectors, one row per line',
       dsrc='fortran'),
    _b("reduceh", 'bool', default=True, var='reduceh', module='modpw', cat='response',
       src=1418,
       desc='set to .true. if the reciprocal H-vectors should be reduced by the symmorphic crystal symmetries',
       dsrc='manual'),
    _b("hybrid", 'bool', default=False, var='hybrid0', module='modmain', cat='xc', src=1420,
       desc='.true if a hybrid functional is to be used when running a Hartree-Fock calculation',
       dsrc='manual'),
    _b("hybridc", 'real', default=1.0, lo=0.0, hi=1.0, var='hybridc', module='modmain',
       cat='xc', src=1422, aliases=('hybmix',), desc='hybrid functional mixing coefficient',
       dsrc='manual'),
    _b("ecvcut", 'real', default=-3.5, var='ecvcut', module='modmain', cat='basis', src=1430,
       desc='core-valence cut-off energy for species file generation', dsrc='fortran'),
    _b("esccut", 'real', default=-0.4, var='esccut', module='modmain', cat='basis', src=1432,
       desc='semi-core-valence cut-off energy for species file generation', dsrc='fortran'),
    _b("nvbse", 'int', default=2, lo=0, var='nvbse0', module='modmain', cat='response',
       src=1434, desc='number of valence states to be used for BSE calculations',
       dsrc='manual'),
    _b("ncbse", 'int', default=3, lo=0, var='ncbse0', module='modmain', cat='response',
       src=1442, desc='number of conduction states to be used for BSE calculations',
       dsrc='manual'),
    _b("istxbse", shape='list', line=('int', 1), maxrows=20, var='istxbse', module='modmain',
       cat='response', src=1450,
       desc='extra valence states to include in the BSE Hamiltonian, one index per line',
       dsrc='fortran'),
    _b("jstxbse", shape='list', line=('int', 1), maxrows=20, var='jstxbse', module='modmain',
       cat='response', src=1476,
       desc='extra conduction states to include in the BSE Hamiltonian, one index per line',
       dsrc='fortran'),
    _b("bsefull", 'bool', default=False, var='bsefull', module='modmain', cat='response',
       src=1502,
       desc='if bsefull is .true. then the full BSE Hamiltonian is calculated, otherwise only the Hermitian block',
       dsrc='fortran'),
    _b("hxbse", 'bool', default=True, var='hxbse', module='modmain', cat='response', src=1504,
       desc='.true. if the exchange term is to be included in the BSE Hamiltonian',
       dsrc='fortran'),
    _b("hdbse", 'bool', default=True, var='hdbse', module='modmain', cat='response', src=1506,
       desc='.true. if the direct term is to be included in the BSE Hamiltonian',
       dsrc='manual'),
    _b("gmaxrf", 'real', default=3.0, lo=0.0, var='gmaxrf', module='modmain', cat='response',
       src=1508, aliases=('gmaxrpa',),
       desc='maximum length of |G| for computing response functions', dsrc='manual'),
    _b("mbwgrf", 'int', default=-1, var='mbwgrf', module='modmain', cat='response', src=1516,
       desc='matrix bandwidth of response functions in the G-vector basis', dsrc='manual'),
    _b("emaxrf", 'real', default=1000000.0, lo=0.0, var='emaxrf', module='modmain',
       cat='response', src=1518,
       desc='energy cut-off used when calculating Kohn-Sham response functions',
       dsrc='manual'),
    _b("fxctype", 'int', shape='vector', n=3, nmin=1, default=(-1, -1, -1), var='fxctype',
       module='modtddft', cat='response', src=1526,
       desc='integer defining the type of exchange-correlation kernel fxc', dsrc='manual'),
    _b("fxclrc", 'real', shape='vector', n=2, nmin=1, default=(0.0, 0.0), var='fxclrc',
       module='modtddft', cat='response', src=1530,
       desc='parameters for the dynamical long-range contribution (LRC) to the TDDFT exchange-correlation kernel',
       dsrc='manual'),
    _b("ntemp", 'int', default=40, lo=1, var='ntemp', module='modphonon', cat='response',
       src=1534, desc='number of temperature steps', dsrc='manual'),
    _b("trimvg", shape='deprecated', var='-', cat='deprecated', src=1542, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("rndstate", 'int', default=799047353, var='rndstate', cat='random', src=1545,
       aliases=('rndseed',), desc="seed for Elk's random number generator (|value| is taken)",
       dsrc='fortran'),
    _b("rndatposc", 'real', default=0.0, var='rndatposc', module='modmain', cat='structure',
       src=1548, desc='magnitude of random displacements added to the atomic positions',
       dsrc='fortran'),
    _b("rndbfcmt", 'real', default=0.0, var='rndbfcmt', module='modmain', cat='structure',
       src=1550, desc='magnitude of random vectors added to muffin-tin fields',
       dsrc='fortran'),
    _b("rndavec", 'real', default=0.0, var='rndavec', module='readinput (local)',
       cat='structure', src=1552, desc='lattice vector randomisation amplitude',
       dsrc='manual'),
    _b("c_tb09", 'real', default=0.0, var='c_tb09', module='modmain', cat='xc', src=1554,
       desc="Tran-Blaha '09 constant c [Phys. Rev. Lett. 102, 226401 (2009)]", dsrc='fortran'),
    _b("lowq", 'bool', default=False, var='-', cat='basis', src=1558,
       aliases=('highq', 'vhighq', 'uhighq'),
       desc='.true. selects a whole quality preset; lowq/highq/vhighq/uhighq are four separate blocks setting rgkmax, gmaxvr, lmaxapw, nrmtscf, radkpt, ... to progressively better values',
       dsrc='fortran',
       note='each of lowq/highq/vhighq/uhighq is a separate block; .true. sets a whole quality preset (rgkmax, gmaxvr, lmaxapw, nrmtscf, radkpt, ...)'),
    _b("hmaxvr", 'real', default=20.0, lo=0.0, var='hmaxvr', module='modpw', cat='basis',
       src=1662, desc='maximum length of H-vectors', dsrc='manual'),
    _b("hkmax", 'real', default=12.0, lo=0.0, lo_ex=True, var='hkmax', module='modpw',
       cat='basis', src=1670, desc='maximum |H+k| cut-off for plane wave', dsrc='fortran'),
    _b("lorbcnd", 'bool', default=False, var='lorbcnd', module='modmain', cat='basis',
       src=1678,
       desc='.true. if conduction state local-orbitals are to be automatically added to the basis',
       dsrc='manual'),
    _b("lorbordc", 'int', default=3, lo=2, var='lorbordc', module='modmain', cat='basis',
       src=1680, desc='the order of the conduction state local-orbitals', dsrc='manual'),
    _b("nrmtscf", 'real', opt=1, default=1.0, lo=0.5, var='nrmtscf', module='modmain',
       cat='basis', src=1695, desc='scale factor for number of muffin-tin points',
       dsrc='fortran'),
    _b("lmaxdb", 'int', default=3, lo=0, var='lmaxdb', module='modmain', cat='dos', src=1705,
       aliases=('lmaxdos',), desc='angular momentum cut-off for the partial DOS plot',
       dsrc='manual'),
    _b("epsdev", 'real', default=0.0025, lo=0.0, lo_ex=True, var='epsdev', module='modphonon',
       cat='convergence', src=1713,
       desc='smallest allowed perturbation theory denominator for eigenvector derivatives',
       dsrc='fortran'),
    _b("msmooth", shape='deprecated', var='-', cat='deprecated', src=1721,
       status='deprecated', desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("npmae", 'int', default=-1, var='npmae0', module='modmain', cat='magnetism', src=1724,
       desc='number or distribution of directions for MAE calculations', dsrc='manual'),
    _b("wrtvars", 'bool', default=False, var='wrtvars', module='modvars', cat='io', src=1726,
       desc='if wrtvars is .true. then variables are written to VARIABLES.OUT',
       dsrc='fortran'),
    _b("ftmtype", 'int', default=0, var='ftmtype', module='moddftu', cat='magnetism',
       src=1728, desc='1 to enable a fixed tensor moment (FTM) calculation, 0 otherwise',
       dsrc='manual'),
    _b("tmomfix", shape='deprecated', var='-', cat='deprecated', src=1730, status='removed',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("tm3fix", shape='special', lo=1, var='ntmfix/itmfix/wkprfix', module='moddftu',
       cat='magnetism', src=1736,
       desc='the fixed 3-index tensor moments: a count line then three lines per entry',
       dsrc='fortran',
       note="count line, then THREE lines per entry: 'is ia l', 'k p r t', and the tensor component"),
    _b("tauftm", 'real', default=0.1, lo=0.0, var='tauftm', module='moddftu', cat='magnetism',
       src=1765, desc='fixed tensor moment step size', dsrc='fortran'),
    _b("ftmstep", shape='deprecated', var='-', cat='deprecated', src=1773,
       status='deprecated', desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("cmagz", 'bool', default=False, var='cmagz', module='modmain', cat='magnetism',
       src=1776, aliases=('forcecmag',),
       desc='.true. if z-axis collinear magnetism is to be enforced', dsrc='manual'),
    _b("rotavec", 'real', shape='vector', n=4, default=(0.0, 0.0, 0.0, 0.0), var='axang',
       module='readinput (local)', cat='structure', src=1778,
       desc='axis-angle representation of lattice vector rotation', dsrc='manual',
       note='axis-angle (x, y, z, angle in degrees) rotation applied to avec'),
    _b("tstime", 'real', default=1000.0, lo=0.0, lo_ex=True, var='tstime', module='modtddft',
       cat='tddft', src=1780, desc='total simulation time of time evolution run',
       dsrc='manual'),
    _b("dtimes", 'real', default=0.1, lo=0.0, lo_ex=True, var='dtimes', module='modtddft',
       cat='tddft', src=1788, desc='time step used in time evolution run', dsrc='manual'),
    _b("pulse", shape='counted', line=('real', 12), linemin=8, header=('int', 1), count_at=0,
       lo=1, var='npulse/pulse', module='modtddft', cat='tddft', src=1796,
       desc='number of pulses', dsrc='manual'),
    _b("ramp", shape='counted', line=('real', 12), linemin=8, header=('int', 1), count_at=0,
       lo=1, var='nramp/ramp', module='modtddft', cat='tddft', src=1811,
       desc='number of ramps', dsrc='manual'),
    _b("step", shape='counted', line=('real', 9), linemin=5, header=('int', 1), count_at=0,
       lo=1, var='nstep/step', module='modtddft', cat='tddft', src=1826,
       desc='number of A-field steps', dsrc='fortran'),
    _b("ncgga", shape='deprecated', var='-', cat='deprecated', src=1841, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("dncgga", 'real', default=1e-08, lo=0.0, var='dncgga', module='modmain', cat='xc',
       src=1844, desc='small constant used to stabilise non-collinear GGA', dsrc='manual'),
    _b("ntswrite", 'int', shape='vector', n=2, nmin=1, default=(500, 1), var='ntswrite',
       module='modtddft', cat='tddft', src=1852,
       desc='observables are written to file every ntswrite(1) time steps; this begins at or after time step ntswrite(2); writing occurs at the first time step irrespective of ntswrite',
       dsrc='fortran'),
    _b("nxoapwlo", 'int', default=0, lo=0, var='nxoapwlo', module='modmain', cat='basis',
       src=1856, aliases=('nxapwlo',),
       desc='extra order of radial functions to be added to the existing APW and local-orbital set',
       dsrc='manual'),
    _b("nxlo", 'int', default=0, lo=0, var='nxlo', module='modmain', cat='basis', src=1864,
       desc='excess local orbitals', dsrc='fortran'),
    _b("tdrho1d", 'bool', default=False, var='tdrho1d', module='modtddft', cat='tddft',
       src=1872,
       desc='.true. if the density is to be written on the plot1d line every ntswrite time steps of a TDDFT run',
       dsrc='fortran'),
    _b("tdrho2d", 'bool', default=False, var='tdrho2d', module='modtddft', cat='tddft',
       src=1874,
       desc='.true. if the density is to be written on the plot2d plane every ntswrite time steps of a TDDFT run',
       dsrc='fortran'),
    _b("tdrho3d", 'bool', default=False, var='tdrho3d', module='modtddft', cat='tddft',
       src=1876,
       desc='.true. if the density is to be written on the plot3d box every ntswrite time steps of a TDDFT run',
       dsrc='fortran'),
    _b("tdmag1d", 'bool', default=False, var='tdmag1d', module='modtddft', cat='tddft',
       src=1878,
       desc='.true. if the magnetisation is to be written on the plot1d line at each time step of a TDDFT run',
       dsrc='fortran'),
    _b("tdmag2d", 'bool', default=False, var='tdmag2d', module='modtddft', cat='tddft',
       src=1880,
       desc='.true. if the magnetisation is to be written on the plot2d plane at each time step of a TDDFT run',
       dsrc='fortran'),
    _b("tdmag3d", 'bool', default=False, var='tdmag3d', module='modtddft', cat='tddft',
       src=1882,
       desc='.true. if the magnetisation is to be written on the plot3d box at each time step of a TDDFT run',
       dsrc='fortran'),
    _b("tdjr1d", 'bool', default=False, var='tdjr1d', module='modtddft', cat='tddft',
       src=1884, aliases=('tdcd1d',),
       desc='.true. if the current density is to be written on the plot1d line at each time step of a TDDFT run',
       dsrc='fortran'),
    _b("tdjr2d", 'bool', default=False, var='tdjr2d', module='modtddft', cat='tddft',
       src=1886, aliases=('tdcd2d',),
       desc='.true. if the current density is to be written on the plot2d plane at each time step of a TDDFT run',
       dsrc='fortran'),
    _b("tdjr3d", 'bool', default=False, var='tdjr3d', module='modtddft', cat='tddft',
       src=1888, aliases=('tdcd3d',),
       desc='.true. if the current density is to be written on the plot3d box at each time step of a TDDFT run',
       dsrc='fortran'),
    _b("tddos", 'bool', default=False, var='tddos', module='modtddft', cat='tddft', src=1890,
       desc='.true. if the time-dependent density of states is to be written at each time step of a TDDFT run',
       dsrc='fortran'),
    _b("tdlsj", 'bool', default=False, var='tdlsj', module='modtddft', cat='tddft', src=1892,
       desc='.true. if the on-site L, S and J moments are to be written at each time step of a TDDFT run',
       dsrc='fortran'),
    _b("tdjtk", 'bool', default=False, var='tdjtk', module='modtddft', cat='tddft', src=1894,
       desc='.true. if the total current as a function of time is to be written during a TDDFT run',
       dsrc='fortran'),
    _b("tdxrmk", 'bool', default=False, var='tdxrmk', module='modtddft', cat='tddft',
       src=1896,
       desc='.true. if the x-ray magnetic scattering amplitudes are to be written during a TDDFT run',
       dsrc='fortran'),
    _b("epseph", shape='deprecated', var='-', cat='deprecated', src=1898, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("rndevt0", 'real', default=0.0, var='rndevt0', module='modtddft', cat='tddft',
       src=1901, desc='magnitude of complex numbers added to initial eigenvectors',
       dsrc='fortran'),
    _b("sxcscf", 'real', opt=1, default=1.0, var='sxcscf', module='modmain', cat='xc',
       src=1903, aliases=('ssxc', 'rstsf'), desc='spin exchange-correlation scaling factor',
       dsrc='fortran'),
    _b("tempk", 'real', lo=0.0, lo_ex=True, var='tempk', module='modmain', cat='occupation',
       src=1907, desc='temperature T of the electronic system in kelvin', dsrc='manual',
       note="a shortcut: sets swidth = kB*tempk (Elk has no default temperature; swidth's own default applies unless this block is given)"),
    _b("avecu", shape='block', lines=(('real', 3, 0), ('real', 3, 0), ('real', 3, 0)),
       default=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), var='avecu',
       module='modulr', cat='ultralr', src=1919,
       desc='ultracell lattice vectors stored column-wise', dsrc='fortran'),
    _b("scaleu", 'real', default=1.0, var='scu', module='readinput (local)', cat='ultralr',
       src=1923, desc='scaling factor for all three ultra-long-range lattice vectors avecu',
       dsrc='fortran'),
    _b("scaleu1", 'real', default=1.0, var='scu1', module='readinput (local)', cat='ultralr',
       src=1925, desc='scaling factor for the first ultra-long-range lattice vector',
       dsrc='fortran'),
    _b("scaleu2", 'real', default=1.0, var='scu2', module='readinput (local)', cat='ultralr',
       src=1927, desc='scaling factor for the second ultra-long-range lattice vector',
       dsrc='fortran'),
    _b("scaleu3", 'real', default=1.0, var='scu3', module='readinput (local)', cat='ultralr',
       src=1929, desc='scaling factor for the third ultra-long-range lattice vector',
       dsrc='fortran'),
    _b("q0cut", 'real', default=0.0, var='q0cut', module='modulr', cat='response', src=1931,
       desc="Q-vector cut-off for the ultra long-range Coulomb Green's function",
       dsrc='manual'),
    _b("ngridkpa", 'int', shape='vector', n=3, default=(-1, -1, -1), var='ngridkpa',
       module='modulr', cat='kpoints', src=1933, desc='kappa-point grid sizes',
       dsrc='fortran'),
    _b("rndbfcu", 'real', default=0.0, var='rndbfcu', module='modulr', cat='magnetism',
       src=1935, desc='random amplitude used for initialising the long-range magnetic field',
       dsrc='fortran'),
    _b("bfieldcu", 'real', shape='vector', n=3, default=(0.0, 0.0, 0.0), var='bfieldcu',
       module='modulr', cat='magnetism', src=1937, aliases=('bfielduc',),
       desc='global external magnetic field in Cartesian coordinates', dsrc='manual'),
    _b("efieldcu", 'real', shape='vector', n=3, default=(0.0, 0.0, 0.0), var='efieldcu',
       module='modulr', cat='fields', src=1939, aliases=('efielduc',),
       desc='electric field vector in Cartesian coordinates', dsrc='fortran'),
    _b("tplotq0", 'bool', default=True, var='tplotq0', module='modulr', cat='response',
       src=1941,
       desc='if tplotq0 is .true. then the Q = 0 term is included when generating plots',
       dsrc='fortran'),
    _b("trdvclr", 'bool', default=False, var='trdvclr', module='modulr', cat='response',
       src=1943,
       desc='set to .true. if the ultra long-range, real-space external Coulomb potential should be read in from VCLR.OUT',
       dsrc='manual'),
    _b("trdbfcr", 'bool', default=False, var='trdbfcr', module='modulr', cat='magnetism',
       src=1945,
       desc='set to .true. if the ultra long-range, real-space external magnetic field in Cartesian coordinates should be read in from BFCR.OUT',
       dsrc='manual'),
    _b("evtype", shape='deprecated', var='-', cat='deprecated', src=1947, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("wmaxgw", 'real', default=-10.0, var='wmaxgw', module='modgw', cat='gw', src=1950,
       desc='maximum Matsubara frequency for GW calculations', dsrc='manual'),
    _b("twdiag", shape='deprecated', var='-', cat='deprecated', src=1952, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("tsediag", 'bool', default=False, var='tsediag', module='modgw', cat='gw', src=1955,
       desc='set to .true. if the self-energy matrix should be treated as diagonal',
       dsrc='manual'),
    _b("actype", 'int', default=10, var='actype', module='modgw', cat='gw', src=1957,
       desc='analytic continuation type', dsrc='manual'),
    _b("npole", 'int', default=3, lo=1, var='npole', module='modgw', cat='gw', src=1959,
       desc='number of poles used for fitting the self-energy matrix elements',
       dsrc='fortran'),
    _b("nspade", 'int', default=100, lo=1, var='nspade', module='modgw', cat='gw', src=1967,
       desc='number of complex shifts used in averaging the Pade approximant for the analytic continuation of the self-energy to the real axis',
       dsrc='fortran'),
    _b("tfav0", 'bool', default=True, var='tfav0', module='modmain', cat='phonons', src=1975,
       desc='tfav0 is .true. if the average force should be zero in order to prevent translation of the atomic basis',
       dsrc='fortran'),
    _b("rmtscf", 'real', default=1.0, lo=0.0, lo_ex=True, var='rmtscf', module='modmain',
       cat='basis', src=1977, desc='muffin-tin radius scaling factor', dsrc='manual'),
    _b("mrmtav", 'int', default=0, var='mrmtav', module='modmain', cat='basis', src=1985,
       desc='order of averaging applied to the muffin-tin radii', dsrc='manual'),
    _b("rmtall", 'real', default=-1.0, var='rmtall', module='modmain', cat='basis', src=1987,
       desc='muffin-tin radius for all species', dsrc='manual'),
    _b("maxthd", 'int', default=0, var='maxthd', module='modomp', cat='parallel', src=1989,
       aliases=('omp_num_threads', 'OMP_NUM_THREADS'),
       desc='maximum number of OpenMP threads available', dsrc='fortran'),
    _b("maxthd1", 'int', default=0, var='maxthd1', module='modomp', cat='parallel', src=1991,
       desc='maximum number of OpenMP threads for the first nesting level', dsrc='fortran'),
    _b("maxthdmkl", 'int', default=0, var='maxthdmkl', module='modomp', cat='parallel',
       src=1993, desc='maximum number of threads available to MKL', dsrc='fortran'),
    _b("maxlvl", 'int', default=4, lo=1, var='maxlvl', module='modomp', cat='parallel',
       src=1995, aliases=('omp_max_active_levels', 'OMP_MAX_ACTIVE_LEVELS'),
       desc='maximum OpenMP nesting level', dsrc='fortran'),
    _b("stable", 'bool', default=False, var='-', cat='basis', src=2003,
       desc='.true. selects a numerically more stable (and more expensive) parameter set: autolinengy, mrmtav, lmaxapw, gmaxvr, msmgmt',
       dsrc='fortran',
       note='.true. raises autolinengy/mrmtav/lmaxapw/gmaxvr/msmgmt to a numerically safer set'),
    _b("metagga", 'bool', default=False, var='-', cat='xc', src=2021,
       desc='.true. selects the parameter set required for meta-GGA functionals: lmaxi, gmaxvr, nrmtscf, msmgmt, epspot, epsengy',
       dsrc='fortran',
       note='.true. raises lmaxi/gmaxvr/nrmtscf/msmgmt and tightens epspot/epsengy for meta-GGA functionals'),
    _b("t0tdlr", shape='deprecated', var='-', cat='deprecated', src=2041, status='deprecated',
       desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("tdphi", 'real', default=0.0, var='tdphi', module='modtddft', cat='tddft', src=2044,
       desc='phase defining complex direction of time evolution', dsrc='fortran'),
    _b("thetamld", 'real', default=0.7853981633974483, var='thetamld', module='modtddft',
       cat='tddft', src=2048,
       desc='magnetic linear dichroism (MLD) angle between the electric and magnetic fields',
       dsrc='fortran', note='read in degrees, stored in radians'),
    _b("ntsbackup", 'int', default=0, var='ntsbackup', module='modtddft', cat='tddft',
       src=2052,
       desc='number of time steps after which the time-dependent eigenvectors are backed up',
       dsrc='fortran'),
    _b("seedname", 'str', default="wannier", var='seedname', module='modw90', cat='io',
       src=2054, desc='seedname for all Wannier90 files', dsrc='fortran'),
    _b("num_wann", 'int', default=0, var='num_wann', module='modw90', cat='wannier', src=2057,
       desc='number of Wannier functions to calculate', dsrc='fortran'),
    _b("idxw90", shape='raw', nlines=1, var='idxw90', module='modw90', cat='wannier',
       src=2059, aliases=('wann_bands',), desc='index to bands', dsrc='fortran',
       note="a single line in Elk's number-list syntax, e.g. '1-4,7,9' (parsed by src/numlist.f90)"),
    _b("projw90", 'bool', default=False, var='projw90', module='modw90', cat='wannier',
       src=2065, desc='.true. if the s, p, d and f projectors should be calculated',
       dsrc='manual'),
    _b("lprojw90", shape='list', line=('int', 5), var='lprojw90', module='modw90',
       cat='wannier', src=2067,
       desc="per-species angular-momentum projectors for Wannier90: one 'is l1 l2 l3 l4' line per species",
       dsrc='fortran',
       note="'is l1 l2 l3 l4' -- species index then its four angular-momentum projection flags"),
    _b("num_iter", 'int', default=500, var='num_iter', module='modw90', cat='wannier',
       src=2087, desc='number of iterations for the minimisation of omega', dsrc='fortran'),
    _b("dis_num_iter", 'int', default=500, var='dis_num_iter', module='modw90', cat='wannier',
       src=2089, desc='number of iterations for disentanglement', dsrc='fortran'),
    _b("trial_step", 'real', default=0.001, var='trial_step', module='modw90', cat='wannier',
       src=2091, desc='trial step for the line search minimisation', dsrc='fortran'),
    _b("xlwin", shape='raw', var='xlwin', module='modw90', cat='wannier', src=2093,
       aliases=('wannierExtra',), desc='extra lines to write to .win file', dsrc='fortran',
       note='verbatim extra lines appended to the Wannier90 .win file'),
    _b("wrtunk", 'bool', default=False, var='wrtunk', module='modw90', cat='wannier',
       src=2103,
       desc='wrtunk is .true. if the UNKkkkkk.s files are to be written in order to enable real-space wavefunction plotting',
       dsrc='fortran'),
    _b("tbdip", 'bool', default=False, var='tbdip', module='modmain', cat='fields', src=2105,
       desc='tbdip is .true. if the spin and current dipole fields are to be added to the Kohn-Sham magnetic field',
       dsrc='fortran'),
    _b("tjr", 'bool', default=False, var='tjr', module='modmain', cat='fields', src=2107,
       aliases=('tcden',),
       desc='tjr is .true. if the current density j(r) is to be calculated', dsrc='fortran'),
    _b("tauefm", 'real', default=0.01, var='tauefm', module='modbog', cat='bogoliubov',
       src=2109, desc='Fermi energy adjustment step size', dsrc='fortran'),
    _b("epsefm", 'real', default=1e-06, var='epsefm', module='modbog', cat='bogoliubov',
       src=2111, desc='Fermi energy convergence tolerance', dsrc='fortran'),
    _b("t0gclq0", shape='deprecated', var='-', cat='deprecated', src=2113,
       status='deprecated', desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("tafindt", 'bool', default=False, var='tafindt', module='modtddft', cat='fields',
       src=2116,
       desc="if tafindt is .true. then the induced A-field is determined from Maxwell's equation and added to the total",
       dsrc='fortran'),
    _b("afindscf", shape='deprecated', var='-', cat='deprecated', src=2118,
       status='deprecated', desc='no longer used by this version of Elk', dsrc='fortran'),
    _b("afindpm", 'real', shape='vector', n=3, default=(0.0, 0.0, 1.0), var='afindpm',
       module='modtddft', cat='fields', src=2121, desc='induced A-field parameters',
       dsrc='fortran'),
    _b("nkspolar", 'int', default=4, lo=1, var='nkspolar', module='modmain', cat='response',
       src=2129,
       desc='number of k-points subdivision used for calculating the polarisation phase',
       dsrc='fortran'),
    _b("ntsforce", 'int', default=100, lo=1, var='ntsforce', module='modtddft',
       cat='geometry', src=2137, desc='number of time steps between force calculations',
       dsrc='fortran'),
    _b("wphcut", 'real', default=1e-06, lo=0.0, lo_ex=True, var='wphcut', module='modphonon',
       cat='phonons', src=2145,
       desc='phonon frequency cut-off below which modes are neglected', dsrc='fortran'),
    _b("ephscf", 'real', shape='vector', n=2, default=(8.0, 0.02), var='ephscf',
       module='modphonon', cat='phonons', src=2153,
       desc='scale factor of the electron-phonon term and its mixing parameter',
       dsrc='fortran'),
    _b("anomalous", 'bool', default=False, var='anomalous', module='modphonon', cat='phonons',
       src=2155,
       desc='anomalous is .true. if only the anomalous density matrix is to be used in the construction of the electron-phonon Hamiltonian',
       dsrc='fortran'),
    _b("tephde", 'bool', default=False, var='tephde', module='modphonon', cat='phonons',
       src=2157, desc='tephde is .true. if D = D0 + E, otherwise D = D0', dsrc='fortran'),
    _b("bdiag", 'bool', default=False, var='bdiag', module='modbog', cat='bogoliubov',
       src=2159, desc='bdiag is .true. if the matrix B is taken to be diagonal',
       dsrc='fortran'),
    _b("ecutb", 'real', default=0.001, lo=0.0, lo_ex=True, var='ecutb', module='modbog',
       cat='bogoliubov', src=2161,
       desc='cut-off energy for matrix B (elements outside this window are set to zero)',
       dsrc='fortran'),
    _b("ediag", 'bool', default=False, var='ediag', module='modbog', cat='bogoliubov',
       src=2169, desc='ediag is .true. if the matrix E is taken to be diagonal',
       dsrc='fortran'),
    _b("pwxpsn", 'int', default=2, lo=1, var='pwxpsn', module='modbog', cat='bogoliubov',
       src=2171, desc='power used in formula for (W,X) pseudo-normalisation (see article)',
       dsrc='fortran'),
    _b("ramdisk", 'bool', default=True, var='ramdisk', module='modramdisk', cat='io',
       src=2179, desc='ramdisk is .true. if the RAM disk should be used', dsrc='fortran'),
    _b("wrtdisk", 'bool', default=True, var='wrtdisk', module='modramdisk', cat='io',
       src=2181, aliases=('wrtdsk',),
       desc='wrtdisk is .true. if files should also be written to disk', dsrc='fortran'),
    _b("epsdmat", 'real', default=1e-08, var='epsdmat', module='moddftu', cat='magnetism',
       src=2183,
       desc='tolerance for checking invariance of density matrix under symmetry operations',
       dsrc='fortran'),
    _b("tm3type", 'int', default=0, lo=0, hi=2, var='tm3type', module='moddftu',
       cat='magnetism', src=2185,
       desc='3-index tensor moment type 0 : real, corresponding to Hermitian Gamma matrices 1 : complex, see Appendix A of G. van der Laan and B. T. Thole, J. Phys.: Condens. Matter 7, 9947 (1995) 2 : real, corresponding to tesseral spherical harmonics',
       dsrc='fortran'),
    _b("tm3vdl", 'bool', default=False, var='-', cat='magnetism', src=2193,
       aliases=('tm3old',),
       desc='.true. to use the older van der Laan convention for the 3-index tensor moments',
       dsrc='fortran'),
    _b("batch", 'bool', default=False, var='batch', module='modvars', cat='control', src=2196,
       desc='if batch is .true. then Elk will run in batch mode', dsrc='fortran'),
    _b("tafspt", 'bool', default=False, var='tafspt', module='modtddft', cat='geometry',
       src=2198, desc='tafspt is .true. if the A-field is spin- and time-dependent',
       dsrc='fortran'),
    _b("tbaspat", 'bool', default=False, var='tbaspat', module='modtddft', cat='geometry',
       src=2200,
       desc='tbaspat is .true. if the effective magnetic field B = 1/c^2((As)tA).sigma should be added to the time-dependent Hamiltonian',
       dsrc='fortran'),
    _b("trdatdv", 'bool', default=False, var='trdatdv', module='modmain', cat='geometry',
       src=2202,
       desc='trdatdv is .true. if the atomic displacements and velocities are to be read from file',
       dsrc='fortran'),
    _b("atdfc", 'real', default=0.0, lo=0.0, var='atdfc', module='modmain', cat='geometry',
       src=2204, desc='atomic damping force coefficient', dsrc='fortran'),
    _b("maxforce", 'real', default=-1.0, var='maxforce', module='modmain', cat='convergence',
       src=2212,
       desc='maximum allowed force magnitude; if this force is reached for any atom then all forces are rescaled so that the maximum force magnitude is this value',
       dsrc='fortran'),
    _b("msmgmt", 'int', default=0, var='msmgmt', module='modmain', cat='basis', src=2214,
       aliases=('msmg2mt',),
       desc='smoothing order used when calculating gradients in the muffin-tin',
       dsrc='fortran'),
    _b("ntsorth", 'int', default=1000, var='ntsorth', module='modtddft', cat='tddft',
       src=2216,
       desc='number of time steps after which the time-dependent Kohn-Sham orbitals are made strictly orthogonal using a singular value decomposition',
       dsrc='fortran'),
    _b("deltabf", 'real', default=0.5, lo=0.0, lo_ex=True, var='deltabf', module='modmain',
       cat='magnetism', src=2218,
       desc='small change in magnetic field used for calculating the magnetoelectric tensor',
       dsrc='fortran'),
    _b("jtconst0", 'bool', default=False, var='jtconst0', module='modtddft', cat='tddft',
       src=2226,
       desc='jtconst0 is .true. if the constant part of J(t) should be set to zero when calculating the dielectric function; this effectively removes the Drude term',
       dsrc='fortran'),
    _b("trmt0", 'bool', default=True, var='trmt0', module='modmain', cat='basis', src=2228,
       desc='trmt0 is .true. if the original muffin-tin radii rmt0 are to be retained between tasks',
       dsrc='fortran'),
    _b("ksgwrho", 'bool', default=False, var='ksgwrho', module='modgw', cat='gw', src=2230,
       desc='ksgwrho is .true. if the GW density is to be used in the self-consistent Kohn-Sham calculation',
       dsrc='fortran'),
    _b("npfftg", 'int', default=4, var='npfftg', module='modmain', cat='basis', src=2232,
       desc='number of prime factors for the G-vector FFT', dsrc='fortran'),
    _b("npfftgc", 'int', default=4, var='npfftgc', module='modmain', cat='basis', src=2234,
       desc='number of prime factors for the coarse G-vector FFT', dsrc='fortran'),
    _b("npfftq", 'int', default=4, var='npfftq', module='modmain', cat='basis', src=2236,
       desc='number of prime factors for the q-vector FFT', dsrc='fortran'),
    _b("npfftw", 'int', default=4, var='npfftw', module='modgw', cat='basis', src=2238,
       desc='number of prime factors for the Matsubara frequency FFT', dsrc='fortran'),
    _b("tphnat", 'bool', default=False, var='tphnat', module='modphonon', cat='phonons',
       src=2240,
       desc='tphnat is .true. if the non-analytic term is to be added to the phonon dispersion; this requires the Born effective charges and dielectric function',
       dsrc='fortran'),
    _b("ecutthc", 'real', default=0.01, lo=0.0, lo_ex=True, var='ecutthc', module='modtdhfc',
       cat='xc', src=2242,
       desc='energy window cut-off for TDHFC states around the Fermi energy', dsrc='fortran'),
    _b("tbdipu", 'bool', default=False, var='tbdipu', module='modulr', cat='fields', src=2250,
       desc='if tbdipu is .true. then the spin dipole-dipole interaction is included',
       dsrc='fortran'),
    _b("bdipscf", 'real', default=1.0, var='bdipscf', module='modmain', cat='fields',
       src=2252, desc='dipole magnetic field scaling factor (default 1)', dsrc='fortran'),
]

#: Blocks added to the BUILD copy of readinput.f90 by this project's own
#: Fortran patch series (patches/*.patch); they do not exist in
#: vendor/elk/src/readinput.f90 and are only accepted by a binary built via
#: build_elk.sh.  Kept separate so the completeness test against the vendored
#: tree stays exact in both directions.
_EXTENSIONS = [
    _b("elkpy_socscale", shape="list", line=(("int", "real"), 2),
       var="socscfsp", module="modmain", cat="magnetism", src=0, source="elkpy",
       desc="per-species scale factor for the spin-orbit coupling term: one "
            "'is scale' line per species, terminated by a blank line",
       note="patches/0001-per-species-soc-scale.patch; requires spinorb"),
    _b("elkpy_berry", "int", shape="vector", n=4, var="elkpy_berry_*",
       module="modmain", cat="response", src=0, source="elkpy",
       desc="Berry curvature (task 9000) parameters: 'dir1 dir2 ist0 ist1' -- "
            "the two k-mesh directions spanning the Wilson-loop plane and the "
            "second-variational band window",
       note="patches/0002-berry-curvature-wilson-loop.patch"),
    _b("elkpy_berry_path", shape="special", var="elkpy_berrypath_*",
       module="modmain", cat="response", src=0, source="elkpy",
       desc="Berry curvature at an explicit k-point list (task 9001): "
            "'dir1 dir2 dk ist0 ist1', then the number of k-points, then one "
            "'k1 k2 k3' line each",
       note="patches/0002-berry-curvature-wilson-loop.patch"),
    _b("elkpy_stmdir", "real", shape="vector", n=3, default=(0.0, 0.0, 1.0),
       var="elkpy_stmdir", module="modmain", cat="magnetism", src=0,
       source="elkpy",
       desc="tip magnetisation direction of the spin-polarised STM task "
            "(tasks 9003/9004), in Cartesian coordinates; normalised internally",
       note="patches/0011-spin-polarized-stm.patch"),
    _b("elkpy_stmpol", "real", default=1.0, lo=-1.0, hi=1.0, var="elkpy_stmpol",
       module="modmain", cat="magnetism", src=0, source="elkpy",
       desc="effective spin polarisation of the STM tip, in [-1, 1]",
       note="patches/0011-spin-polarized-stm.patch"),
    _b("elkpy_stmbias", "real", default=0.0, var="elkpy_stmbias",
       module="modmain", cat="magnetism", src=0, source="elkpy",
       desc="sample bias in Hartree: the LDOS is sampled at efermi + bias",
       note="patches/0011-spin-polarized-stm.patch"),
    _b("elkpy_stmint", "bool", default=False, var="elkpy_stmint",
       module="modmain", cat="magnetism", src=0, source="elkpy",
       desc=".true. to integrate the LDOS over the bias window "
            "(constant-current mode) instead of sampling it at efermi + bias",
       note="patches/0011-spin-polarized-stm.patch"),
    _b("elkpy_transport_exit", shape="vector", type=("int", "real"), n=2,
       var="elkpy_trans_axis/elkpy_trans_height", module="modmain",
       cat="response", src=0, source="elkpy",
       desc="the exit (substrate) plane of the vertical transport task "
            "(task 9005): 'axis height' -- the lattice vector the plane is "
            "normal to, and its fractional coordinate along that vector",
       note="patches/0012-vertical-transport.patch"),
    _b("elkpy_transport_window", "real", shape="vector", n=2,
       var="elkpy_trans_window", module="modmain", cat="response", src=0,
       source="elkpy",
       desc="energy window in Hartree, relative to the Fermi energy, within "
            "which states are exported by the vertical transport task",
       note="patches/0012-vertical-transport.patch"),
    _b("elkpy_transport_kgrid", "int", shape="vector", n=3,
       var="elkpy_trans_ngrid", module="modmain", cat="response", src=0,
       source="elkpy",
       desc="the transport k-grid, diagonalised fresh by the task itself and "
            "so independent of ngridk and reducek",
       note="patches/0012-vertical-transport.patch"),
    _b("elkpy_transport_koffset", "real", shape="vector", n=3,
       var="elkpy_trans_koff", module="modmain", cat="response", src=0,
       source="elkpy",
       desc="offset of the transport k-grid, in units of one grid spacing",
       note="patches/0012-vertical-transport.patch"),
    _b("elkpy_transport_sdir", "real", shape="vector", n=3,
       default=(0.0, 0.0, 1.0), var="elkpy_trans_sdir", module="modmain",
       cat="response", src=0, source="elkpy",
       desc="substrate magnetisation direction, in Cartesian coordinates "
            "(normalised internally)",
       note="patches/0012-vertical-transport.patch"),
    _b("elkpy_transport_spol", "real", default=0.0, lo=-1.0, hi=1.0,
       var="elkpy_trans_spol", module="modmain", cat="response", src=0,
       source="elkpy",
       desc="the substrate's effective spin polarisation, in [-1, 1]",
       note="patches/0012-vertical-transport.patch"),
]


def _index(blocks):
    out = {}
    for b in blocks:
        for nm in b.names:
            if nm in out:
                raise AssertionError("duplicate block name %r" % nm)
            out[nm] = b
    return out


#: canonical name -> ParamBlock, upstream Elk only
BLOCKS = dict((b.name, b) for b in _UPSTREAM)
#: canonical name -> ParamBlock, this project's Fortran extensions only
EXTENSION_BLOCKS = dict((b.name, b) for b in _EXTENSIONS)

#: every accepted spelling (canonical names and aliases) -> ParamBlock
_BY_NAME = _index(_UPSTREAM + _EXTENSIONS)

#: category -> sorted list of canonical block names
CATEGORIES = {}
for _b_ in _UPSTREAM + _EXTENSIONS:
    CATEGORIES.setdefault(_b_.cat, []).append(_b_.name)
for _k in CATEGORIES:
    CATEGORIES[_k].sort()
del _b_, _k

#: Blocks `Calculation` writes itself, from its own constructor arguments or
#: from the `Structure`.  Setting these through the generic parameter surface
#: is legal (Elk's sequential reader takes the LAST occurrence, and
#: `_add_base_blocks` emits extra blocks after its own) but silently
#: overrides the named argument, so `set_parameters` warns.  `tasks` is the
#: exception: `Calculation` writes it FIRST, so a user copy would hijack
#: every get_* call -- that one is refused outright.
CALCULATION_OWNED = frozenset([
    "sppath", "scale", "avec", "atoms", "xctype", "spinpol", "spinorb",
    "elkpy_socscale", "rgkmax", "ngridk", "vkloff",
])
RESERVED = frozenset(["tasks"])


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def known(include_aliases=False, include_extensions=True):
    """Every block name this module knows, sorted."""
    src = list(_UPSTREAM) + (list(_EXTENSIONS) if include_extensions else [])
    if include_aliases:
        return sorted(nm for b in src for nm in b.names)
    return sorted(b.name for b in src)


def is_known(name):
    return name in _BY_NAME


def describe(name):
    """Return the :class:`ParamBlock` for `name` (canonical or alias).

    Raises :class:`ParameterError`, with a case-insensitive "did you mean",
    for an unknown name -- Fortran's `select case` is case-sensitive, so
    `NGRIDK` really is a different (invalid) block from `ngridk`.
    """
    b = _BY_NAME.get(name)
    if b is not None:
        return b
    raise ParameterError(_unknown_message(name))


def _unknown_message(name):
    allnames = list(_BY_NAME)
    close = difflib.get_close_matches(name, allnames, n=3, cutoff=0.7)
    lower = dict((nm.lower(), nm) for nm in allnames)
    if name.lower() in lower and lower[name.lower()] != name:
        close = [lower[name.lower()]] + [c for c in close if c != lower[name.lower()]]
    msg = "unknown elk.in block %r" % name
    if close:
        msg += " -- did you mean %s?" % " or ".join(repr(c) for c in close)
    msg += (" (block names are case-sensitive; elkpy.params.search() lists "
            "what is available)")
    return msg


def search(substring, include_description=True):
    """Blocks whose name, aliases or description contain `substring`."""
    q = substring.lower()
    out = []
    for b in _UPSTREAM + _EXTENSIONS:
        hay = " ".join(b.names)
        if include_description:
            hay += " " + (b.desc or "")
        if q in hay.lower():
            out.append(b)
    out.sort(key=lambda b: b.name)
    return out


def categories():
    """Sorted list of category keys."""
    return sorted(CATEGORIES)


def in_category(cat):
    """The :class:`ParamBlock`s in one category."""
    if cat not in CATEGORIES:
        raise ParameterError(
            "unknown category %r -- known categories: %s"
            % (cat, ", ".join(categories())))
    return [_BY_NAME[nm] for nm in CATEGORIES[cat]]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _check_scalar(b, v, where, type=None, rng=True):
    t = type or b.type
    if t == "bool":
        if not isinstance(v, bool):
            raise ParameterError(
                "%s: expected a Python bool (rendered .true./.false.), got %r"
                " -- Elk's logical read does not accept 0/1" % (where, v))
        return v
    if t == "int":
        if isinstance(v, bool) or not _is_int(v):
            if isinstance(v, float) and float(v).is_integer():
                raise ParameterError(
                    "%s: expected an integer, got the float %r" % (where, v))
            raise ParameterError("%s: expected an integer, got %r" % (where, v))
        if rng:
            _check_range(b, v, where)
        return v
    if t == "real":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ParameterError("%s: expected a real number, got %r" % (where, v))
        if rng:
            _check_range(b, float(v), where)
        return FortranReal(v)
    if t == "str":
        if not isinstance(v, str):
            raise ParameterError("%s: expected a string, got %r" % (where, v))
        return v
    raise ParameterError("%s: block has no value type" % where)


def _check_range(b, v, where):
    if b.lo is not None:
        if (v <= b.lo) if b.lo_ex else (v < b.lo):
            raise ParameterError(
                "%s: value %r is outside the range Elk accepts (needs %s %s)"
                % (where, v, ">" if b.lo_ex else ">=", b.lo))
    if b.hi is not None:
        if (v >= b.hi) if b.hi_ex else (v > b.hi):
            raise ParameterError(
                "%s: value %r is outside the range Elk accepts (needs %s %s)"
                % (where, v, "<" if b.hi_ex else "<=", b.hi))


def _seq(v):
    return isinstance(v, (list, tuple))


def _check_line(b, v, spec, where, nmin=None, rng=True, nrange=None):
    """One elk.in line: `spec` is (type, n) with type a str or a per-column
    tuple.  Returns the rendered tuple (or a bare value for n == 1).

    `nrange` bounds how many leading columns the block's lo/hi checks apply
    to.  It matters for the optional-tail blocks (`rgkmax 7.0 0.1`,
    `ngridk 4 4 4 -1 -1 -1`): readinput.f90 range-checks `rgkmax`/`ngridk(:)`
    and then reads the trailing derivative into `drgkmax`/`dngridk(:)`
    without checking it at all, so a negative derivative is perfectly legal
    and must not be rejected here."""
    t, n = spec[0], spec[1]
    lo = nmin if nmin is not None else n
    if not _seq(v):
        v = (v,)
    if not (lo <= len(v) <= n):
        want = str(n) if lo == n else "%d to %d" % (lo, n)
        raise ParameterError(
            "%s: expected %s value%s on this line, got %d (%r)"
            % (where, want, "" if n == 1 else "s", len(v), v))
    out = []
    for i, item in enumerate(v):
        ti = t[i] if isinstance(t, tuple) else t
        this_rng = rng and (nrange is None or i < nrange)
        out.append(_check_scalar(b, item, "%s value %d" % (where, i + 1),
                                 type=ti, rng=this_rng))
    if len(out) == 1 and n == 1:
        return out[0]
    return tuple(out)


def _validate_one(b, value):
    where = "block %r" % b.name
    if b.status == "removed":
        raise ParameterError(
            "%s was removed from this version of Elk and makes it stop with an "
            "error -- %s" % (where, b.desc))
    if b.shape == "deprecated":
        # Elk still consumes exactly one line and prints "no longer used"
        warnings.warn(
            "elk.in block %r is no longer used by this version of Elk; "
            "readinput.f90:%d reads and discards its value" % (b.name, b.src),
            DeprecationWarning, stacklevel=4)
        return [_flat(value)]
    if b.shape == "scalar":
        if _seq(value):
            if not (1 <= len(value) <= 1 + b.opt):
                raise ParameterError(
                    "%s: expected 1%s value, got %d (%r)"
                    % (where, " to %d" % (1 + b.opt) if b.opt else "",
                       len(value), value))
            line = _check_line(b, value, (b.type, 1 + b.opt), where, nmin=1,
                               nrange=1)
            return [line]
        return [_check_scalar(b, value, where)]
    if b.shape == "vector":
        if not _seq(value):
            raise ParameterError(
                "%s: expected a sequence of %d values, got %r" % (where, b.n, value))
        return [_check_line(b, value, (b.type, b.n + b.opt), where,
                            nmin=b.nmin, nrange=b.n)]
    if b.shape == "block":
        if not _seq(value) or len(value) != len(b.lines):
            raise ParameterError(
                "%s: expected %d lines (%s), got %r"
                % (where, len(b.lines), b.signature(), value))
        return [_check_line(b, row, (ln[0], ln[1] + ln[2]),
                            "%s line %d" % (where, i + 1), nmin=ln[1], rng=False)
                for i, (row, ln) in enumerate(zip(value, b.lines))]
    if b.shape == "list":
        rows = _rows(b, value, where)
        if b.maxrows is not None and len(rows) > b.maxrows:
            raise ParameterError(
                "%s: Elk accepts at most %d lines here, got %d"
                % (where, b.maxrows, len(rows)))
        return [_check_line(b, row, b.line, "%s line %d" % (where, i + 1),
                            nmin=b.linemin, rng=False)
                for i, row in enumerate(rows)]
    if b.shape == "counted":
        return _validate_counted(b, value, where)
    if b.shape == "raw":
        return _validate_raw(b, value, where)
    if b.shape in ("special", "opaque"):
        return _validate_passthrough(b, value, where)
    raise ParameterError("%s: unhandled shape %r" % (where, b.shape))


def _rows(b, value, where):
    if not _seq(value):
        raise ParameterError(
            "%s: expected a list of lines (%s), got %r"
            % (where, b.signature(), value))
    return list(value)


def _validate_counted(b, value, where):
    rows = _rows(b, value, where)
    ht, hn = b.header[0], b.header[1]
    header = None
    if rows and _seq(rows[0]) and len(rows[0]) == hn and hn != b.line[1]:
        header, rows = rows[0], rows[1:]
    elif rows and hn == 1 and _is_int(rows[0]):
        header, rows = (rows[0],), rows[1:]
    if header is None:
        if hn != 1:
            raise ParameterError(
                "%s: expected a header line of %d %s first (%s)"
                % (where, hn, ht, b.signature()))
        header = (len(rows),)          # count-only header: fill it in
    count = header[b.count_at]
    if count != len(rows):
        raise ParameterError(
            "%s: header says %d rows but %d were given" % (where, count, len(rows)))
    out = [_check_line(b, header, (ht, hn), "%s header" % where)]
    for i, row in enumerate(rows):
        out.append(_check_line(b, row, b.line, "%s row %d" % (where, i + 1),
                               nmin=b.linemin, rng=False))
    return out


def _validate_raw(b, value, where):
    if isinstance(value, str):
        value = [value]
    if not _seq(value) or not all(isinstance(x, str) for x in value):
        raise ParameterError(
            "%s: expected a string or a list of strings (verbatim elk.in "
            "lines), got %r" % (where, value))
    if b.nlines is not None and len(value) != b.nlines:
        raise ParameterError(
            "%s: expected exactly %d line(s), got %d" % (where, b.nlines, len(value)))
    for x in value:
        if "\n" in x:
            raise ParameterError("%s: a verbatim line may not contain a newline" % where)
        if x.strip() == "":
            raise ParameterError(
                "%s: a blank line terminates the block in Elk's reader and "
                "cannot be part of it" % where)
    return [Verbatim(x) for x in value]


def _validate_passthrough(b, value, where):
    """`atoms`, `DFT+U`, `tm3fix`, `species`: irregular layouts that are
    checked only for gross shape and rendered as given."""
    lines = _rows(b, value, where)
    out = []
    for i, row in enumerate(lines):
        if isinstance(row, str):
            # a near-literal Fortran line ("1 2 0.29 0.0"): a bare str would
            # come out of the writer single-quoted, which Elk's
            # list-directed read of that line cannot parse
            out.append(Verbatim(row))
        elif _seq(row):
            out.append(tuple(
                FortranReal(x) if isinstance(x, float) else x for x in row))
        elif isinstance(row, (bool, int, float)):
            out.append(FortranReal(row) if isinstance(row, float) else row)
        else:
            raise ParameterError(
                "%s line %d: expected a scalar, a sequence of scalars or a "
                "string, got %r" % (where, i + 1, row))
    return out


def _flat(value):
    if _seq(value):
        return tuple(FortranReal(x) if isinstance(x, float) else x for x in value)
    return FortranReal(value) if isinstance(value, float) else value


def validate_blocks(blocks, allow_deprecated=True, allow_reserved=False):
    """Check a {block name: value} mapping and return it rendered.

    Returns a new dict of {name: [lines]} ready to hand to
    :class:`elkpy.inputfile.InputFile.add_block` (which is exactly the shape
    `Calculation.extra_blocks` already uses).  Raises
    :class:`ParameterError` -- a plain `ValueError` subclass -- naming the
    block and what was expected, instead of letting the mistake surface as a
    Fortran error inside the subprocess.
    """
    if not isinstance(blocks, dict):
        raise ParameterError("expected a dict of {block name: value}, got %r" % (blocks,))
    out = {}
    for name, value in blocks.items():
        if not isinstance(name, str):
            raise ParameterError("block names must be strings, got %r" % (name,))
        if name in RESERVED and not allow_reserved:
            raise ParameterError(
                "block %r is written by Calculation itself, first in every "
                "elk.in it generates; setting it here would be read LAST by "
                "Elk and would silently replace the task list of every get_* "
                "call. Use Calculation.run_tasks(...) instead." % name)
        b = describe(name)
        if b.status != "active" and not allow_deprecated:
            raise ParameterError(
                "block %r is %s in this version of Elk" % (name, b.status))
        out[name] = _validate_one(b, value)
    return out


def render_blocks(blocks, **kw):
    """Alias of :func:`validate_blocks` -- validation and rendering are one
    pass, since the rendered form is what the check produces."""
    return validate_blocks(blocks, **kw)
