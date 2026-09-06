"""Calculation-facing access to the full elk.in input-parameter surface.

`Calculation` types about a dozen of Elk's ~315 input blocks as named
constructor arguments and forwards everything else through `extra_blocks`,
an untyped dict written into every elk.in it generates.  That escape hatch
is what makes the other ~300 blocks reachable at all, but it is also where
a misspelt name (`ngrdk`), a wrong arity (`ngridk: [4]`) or a Python `0`
where Elk wants `.false.` becomes a Fortran error inside a subprocess --
usually the unhelpfully generic

    Error(readinput): error reading from elk.in
    Problem occurred in 'ngridk' block

or, for a bad name, a hard `stop` from `readinput.f90`'s `case default`.

This mixin puts the check on the Python side of that boundary, using
`elkpy.params`' table of every `case(...)` branch in
`vendor/elk/src/readinput.f90`: `set_parameters()` validates, renders and
merges into `self.extra_blocks`, so a mistake raises immediately, naming
the block and what was expected, before any process is started.  The
introspection helpers (`describe_parameter`, `search_parameters`,
`parameters_in_category`) make the same table browsable from a session.

Nothing here runs Elk.  `extra_blocks` is part of
`Calculation._basis_signature()`, so changing a parameter after
construction correctly invalidates the cached ground state, exactly as
passing `extra_blocks=` to the constructor would.
"""

import warnings

from .. import params
from ..inputfile import InputFile
from ..params import ParameterError


class ParametersMixin(object):
    """Typed access to every elk.in input block (see `elkpy.params`)."""

    # -- setting -----------------------------------------------------------

    def set_parameters(self, blocks=None, **kwargs):
        """Validate elk.in blocks and add them to this Calculation.

        Accepts a dict (needed for the names Python cannot spell as
        keywords, such as ``DFT+U``) and/or keyword arguments::

            calc.set_parameters(maxscl=40, epsengy=1e-8, stype=0)
            calc.set_parameters({"DFT+U": [(1, 0), (1, 2, 0.29, 0.0)]})

        Every value is checked against `elkpy.params.BLOCKS` -- name, type,
        arity, and the range checks `readinput.f90` performs itself -- and
        then rendered into the exact elk.in line form.  The result is merged
        into `self.extra_blocks`, which `_add_base_blocks` writes into every
        elk.in this Calculation generates (ground state and every ``get_*``).

        Returns the rendered {name: [lines]} for the blocks just set.

        Two guards, both about the order `_add_base_blocks` writes blocks in:

        * ``tasks`` is refused.  `Calculation` writes it FIRST and extra
          blocks LAST, and Elk's reader takes the last occurrence, so a
          ``tasks`` here would silently replace the task list of every
          ``get_*`` call.  Use :meth:`run_tasks` instead.
        * the blocks `Calculation` derives from its own arguments or from
          the `Structure` (``avec``, ``atoms``, ``ngridk``, ``rgkmax``,
          ``xctype``, ``spinpol``, ``spinorb``, ``vkloff``, ``sppath``,
          ``scale``) are accepted but warned about, since setting one here
          overrides the named argument rather than being merged with it.
        """
        merged = dict(blocks or {})
        overlap = set(merged) & set(kwargs)
        if overlap:
            raise ParameterError(
                "block(s) %s given both in the dict and as keywords"
                % ", ".join(sorted(overlap)))
        merged.update(kwargs)
        rendered = params.validate_blocks(merged)
        owned = sorted(set(merged) & params.CALCULATION_OWNED)
        if owned:
            warnings.warn(
                "block(s) %s are also written by Calculation from its own "
                "arguments; the value set here is written later in elk.in and "
                "therefore wins, silently overriding the constructor argument"
                % ", ".join(owned),
                stacklevel=2,
            )
        self.extra_blocks.update(rendered)
        return rendered

    def validate_blocks(self, blocks):
        """Check blocks without applying them; returns the rendered form.

        The same validation :meth:`set_parameters` performs, exposed on its
        own so a caller can check a candidate dict (e.g. one assembled by a
        sweep) before committing to a run.
        """
        return params.validate_blocks(blocks)

    def unset_parameters(self, *names):
        """Remove previously-set blocks from `extra_blocks`.

        Unknown-but-unset names are ignored; names this module does not know
        at all raise, since that is almost always a typo.
        """
        for name in names:
            params.describe(name)
            self.extra_blocks.pop(name, None)

    def parameters(self):
        """The blocks currently set on this Calculation, rendered."""
        return dict(self.extra_blocks)

    def effective_parameters(self, include_defaults=False):
        """What this Calculation actually sends Elk, as {name: value}.

        Includes the blocks `Calculation` builds from its own arguments
        (``rgkmax``, ``ngridk``, ``xctype``, ...) alongside anything set
        through :meth:`set_parameters`, so one call shows the whole input.
        With ``include_defaults=True`` every other known block is listed at
        its `readinput.f90` default as well -- useful for "what is Elk
        assuming here?", not for feeding back in.
        """
        # built by the same code that writes the real elk.in, so this cannot
        # drift from what Elk is actually given (it already includes
        # extra_blocks, hence everything set through set_parameters)
        f = InputFile()
        self._add_base_blocks(f)
        out = dict(f._blocks)
        if include_defaults:
            for name, block in params.BLOCKS.items():
                if name in out or block.status != "active":
                    continue
                if block.default is not None:
                    out[name] = block.default
        return out

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def describe_parameter(name):
        """The :class:`elkpy.params.ParamBlock` for one elk.in block."""
        return params.describe(name)

    @staticmethod
    def explain_parameter(name):
        """A short human-readable summary of one elk.in block."""
        b = params.describe(name)
        lines = ["%s -- %s" % (b.name, b.desc)]
        if b.aliases:
            lines.append("  also accepted as: %s" % ", ".join(b.aliases))
        lines.append("  value: %s" % b.signature())
        if b.default is not None:
            lines.append("  default: %r" % (b.default,))
        if b.lo is not None or b.hi is not None:
            rng = []
            if b.lo is not None:
                rng.append("%s %s" % (">" if b.lo_ex else ">=", b.lo))
            if b.hi is not None:
                rng.append("%s %s" % ("<" if b.hi_ex else "<=", b.hi))
            lines.append("  Elk requires: %s" % " and ".join(rng))
        where = b.var if b.var and b.var != "-" else "(no variable)"
        if b.module and b.module != "-":
            where += " in %s" % b.module
        lines.append("  category: %s   Fortran: %s" % (b.cat, where))
        if b.status != "active":
            lines.append("  STATUS: %s in this version of Elk" % b.status)
        if b.note:
            lines.append("  note: %s" % b.note)
        origin = ("readinput.f90:%d" % b.src if b.source == "elk"
                  else "an elkpy Fortran patch (not upstream Elk)")
        lines.append("  source: %s, description from the %s"
                     % (origin, "manual" if b.dsrc == "manual"
                        else "Fortran source"))
        return "\n".join(lines)

    @staticmethod
    def search_parameters(substring):
        """Blocks whose name, alias or description contains `substring`."""
        return params.search(substring)

    @staticmethod
    def parameter_categories():
        """Sorted list of the categories blocks are grouped into."""
        return params.categories()

    @staticmethod
    def parameters_in_category(category):
        """The blocks in one category (see :meth:`parameter_categories`)."""
        return params.in_category(category)
