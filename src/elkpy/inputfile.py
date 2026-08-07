"""Generic writer for Elk's elk.in block format.

Format (manual sec. 4.4): a block name on its own line, followed by its
values on subsequent lines, terminated by a blank line. Comments start with
`!`. Blocks may appear in any order. This writer is deliberately generic
(block name -> list of value lines) rather than special-cased per block, so
new upstream blocks don't require code changes here.
"""


def _format_value(v):
    if isinstance(v, bool):
        return ".true." if v else ".false."
    if isinstance(v, float):
        return f"{v:.10f}"
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
