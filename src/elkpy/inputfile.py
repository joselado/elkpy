"""Generic writer for Elk's elk.in block format.

Format (manual sec. 4.4): a block name on its own line, followed by its
values on subsequent lines, terminated by a blank line. Comments start with
`!`. Blocks may appear in any order. This writer is deliberately generic
(block name -> list of value lines) rather than special-cased per block, so
new upstream blocks don't require code changes here.
"""


def _format_float(v):
    """A fixed-point rendering that never silently turns a value into zero.

    ``f"{v:.10f}"`` is what almost every block wants -- lattice vectors and
    positions read far better as ``5.1300000000`` than as ``5.13`` -- but it
    is a floor as well as a format: anything below 5e-11 renders as the
    literal ``0.0000000000``, and Elk accepts that without complaint for the
    tolerances where it matters most. ``epspot``/``epsengy`` have no range
    check in ``readinput.f90`` (unlike ``epsband``), so a convergence
    tolerance of 1e-12 became a tolerance of zero, the loop ran to ``maxscl``,
    and the only symptom was non-convergence.

    So the fixed format is kept wherever it survives a round trip through
    zero, and anything it would annihilate is written in exponential form,
    which Elk's list-directed reads accept everywhere a real is expected.
    Values that merely lose digits (1/3) are left alone: changing those would
    alter every currently-correct file for no gain.
    """
    text = f"{v:.10f}"
    if v != 0.0 and float(text) == 0.0:
        return repr(float(v))
    return text


def _format_value(v):
    if isinstance(v, bool):
        return ".true." if v else ".false."
    if isinstance(v, float):
        return _format_float(v)
    if isinstance(v, str):
        return f"'{v}'"
    return str(v)


def _format_line(line):
    if isinstance(line, (list, tuple)):
        return "  ".join(_format_value(v) for v in line)
    return _format_value(line)


class InputFile:
    """An ordered collection of elk.in blocks.

    Each block is a name plus a list of "lines", where each line is either a
    scalar (int/float/bool/str) or a sequence of scalars rendered
    space-separated on one line. Example::

        f = InputFile()
        f.add_block("tasks", [0, 1])
        f.add_block("avec", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        f.add_block("sppath", ["/path/to/species/"])
        f.write(workdir / "elk.in")
    """

    def __init__(self):
        self._blocks = []

    def add_block(self, name, lines):
        self._blocks.append((name, list(lines)))

    def render(self):
        parts = []
        for name, lines in self._blocks:
            parts.append(name)
            for line in lines:
                parts.append(_format_line(line))
            parts.append("")
        return "\n".join(parts) + "\n"

    def write(self, path):
        path = str(path)
        with open(path, "w") as fh:
            fh.write(self.render())


def read_blocks(path):
    """Read a file in Elk's block format (elk.in itself, or an Elk output
    file written in the same syntax, e.g. GEOMETRY_OPT.OUT -- see
    src/writegeom.f90) into (name, [token_lists]) pairs.

    Trailing " : comment" and full-line "!..." comments are stripped, each
    remaining non-blank line becomes a list of whitespace-split tokens, and
    a block ends at the next blank line. If a block name repeats (as
    GEOMETRY_OPT.OUT does, once per optimisation step), every occurrence is
    returned in file order -- callers that want "the last one" take
    blocks[-1] themselves.
    """
    blocks = []
    current_name = None
    current_lines = []
    with open(path) as fh:
        for raw in fh:
            line = raw.split("!", 1)[0]
            line = line.split(":", 1)[0]
            line = line.strip()
            if not line:
                if current_name is not None:
                    blocks.append((current_name, current_lines))
                    current_name = None
                    current_lines = []
                continue
            if current_name is None:
                current_name = line
            else:
                current_lines.append(line.split())
    if current_name is not None:
        blocks.append((current_name, current_lines))
    return blocks
