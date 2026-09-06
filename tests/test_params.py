"""Unit tests for elkpy.params -- the elk.in input-parameter table.

The point of this file is coverage and fidelity to the Fortran, not physics.
It re-parses `vendor/elk/src/readinput.f90` (and `patches/*.patch`) at test
time and compares against the checked-in table in both directions, so the
next Elk version bump fails here, loudly, naming exactly which blocks
appeared or vanished -- the same "one file to edit on a version bump"
contract `spec.py`'s docstring asks for.

No Elk binary is involved anywhere in this file.
"""

import re
import warnings
from pathlib import Path

import pytest

from elkpy import params
from elkpy.inputfile import InputFile
from elkpy.params import ParameterError

REPO = Path(__file__).resolve().parent.parent
READINPUT = REPO / "vendor" / "elk" / "src" / "readinput.f90"
PATCHES = REPO / "patches"


# ---------------------------------------------------------------------------
# parse readinput.f90 independently of the table
# ---------------------------------------------------------------------------

def _case_branches(text):
    """Every `case('a','b')` branch of readinput.f90's block dispatch.

    Returns [(names, line_number)] in file order, excluding `case('')`
    (a blank block name, which Elk simply skips) and `case default`.
    """
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("select case(trim(block))"))
    end = next(i for i, l in enumerate(lines)
               if i > start and l.startswith("case default"))
    out = []
    for i in range(start + 1, end):
        if not lines[i].startswith("case("):
            continue
        names = re.findall(r"'([^']*)'", lines[i])
        names = [n for n in names if n]
        if names:
            out.append((names, i + 1))
    return out


@pytest.fixture(scope="module")
def branches():
    assert READINPUT.is_file(), "vendored Elk source missing: %s" % READINPUT
    return _case_branches(READINPUT.read_text())


# ---------------------------------------------------------------------------
# (a) completeness -- both directions
# ---------------------------------------------------------------------------

def test_every_case_branch_has_a_table_entry(branches):
    fortran = set()
    for names, _ in branches:
        fortran.update(names)
    table = set(params.known(include_aliases=True, include_extensions=False))
    missing = sorted(fortran - table)
    assert not missing, (
        "readinput.f90 has block(s) elkpy.params does not know: %s "
        "(regenerate the table for this Elk version)" % missing)


def test_table_has_no_entry_readinput_does_not_accept(branches):
    fortran = set()
    for names, _ in branches:
        fortran.update(names)
    table = set(params.known(include_aliases=True, include_extensions=False))
    extra = sorted(table - fortran)
    assert not extra, (
        "elkpy.params lists block(s) this Elk's readinput.f90 would reject: %s"
        % extra)


#: The one `case(...)` branch whose behaviour depends on WHICH of its names
#: was used -- `if (trim(block) == 'amixpm')` at readinput.f90:692 -- so its
#: three names cannot share one table entry: amixpm takes two values, beta0
#: and betamax one each.  Every other multi-name branch is a true alias
#: group, which is what the test below enforces.
SHAPE_SPLIT_BRANCHES = {("amixpm", "beta0", "betamax")}


def test_alias_grouping_matches_the_case_branches(branches):
    """Names sharing one `case(...)` must share one table entry, since they
    set the same variable and take the same value shape."""
    for names, line in branches:
        if tuple(names) in SHAPE_SPLIT_BRANCHES:
            for nm in names:
                assert params.describe(nm).name == nm
                assert params.describe(nm).src == line
            continue
        entries = set()
        for nm in names:
            entries.add(params.describe(nm).name)
        assert len(entries) == 1, (
            "readinput.f90:%d groups %s in one branch but elkpy.params splits "
            "them across %s" % (line, names, sorted(entries)))
        canonical = params.describe(names[0])
        assert canonical.names == tuple(names), (
            "readinput.f90:%d spells the branch %s, table has %s"
            % (line, names, list(canonical.names)))


def test_the_shape_split_branch_really_does_split(branches):
    """readinput.f90:692 dispatches on the block name inside the branch --
    if that ever stops being true this table is over-specified."""
    text = READINPUT.read_text()
    assert "if (trim(block) == 'amixpm') then" in text
    assert params.describe("amixpm").shape == "vector"
    assert params.describe("beta0").shape == "scalar"
    assert params.describe("betamax").shape == "scalar"
    assert params.render_blocks({"beta0": 0.1}) == {"beta0": [0.1]}
    with pytest.raises(ParameterError):
        params.render_blocks({"betamax": 1.5})       # amixpm(2) must be <= 1
    params.render_blocks({"amixpm": (2.0, 1.0)})     # amixpm(1) has no upper bound


def test_source_line_points_at_the_right_branch(branches):
    """`src` must be the readinput.f90 line of the block's own `case(...)`."""
    by_line = dict((line, names) for names, line in branches)
    for block in params.BLOCKS.values():
        assert block.src in by_line, (
            "%s claims readinput.f90:%d, which is not a case branch"
            % (block.name, block.src))
        assert block.name in by_line[block.src], (
            "%s claims readinput.f90:%d, whose branch is %s"
            % (block.name, block.src, by_line[block.src]))


def test_counts_of_the_awkward_kinds(branches):
    """Aliases, deprecated blocks and variable-length lists, counted rather
    than dropped (this is the "record how many of each" part)."""
    assert len(branches) == 315
    # 315 case branches, but readinput.f90:691 dispatches on the name inside
    # the branch (see SHAPE_SPLIT_BRANCHES), so amixpm/beta0/betamax are
    # three entries rather than one
    assert len(params.BLOCKS) == 317

    alias_names = [nm for b in params.BLOCKS.values() for nm in b.aliases]
    assert len(alias_names) == 47
    assert len(set(alias_names)) == 47
    assert len(params.known(include_aliases=True, include_extensions=False)) == 364

    deprecated = [b.name for b in params.BLOCKS.values()
                  if b.status == "deprecated"]
    removed = [b.name for b in params.BLOCKS.values() if b.status == "removed"]
    assert len(deprecated) == 27
    # tmomfix is the one block that makes Elk stop rather than warn
    assert removed == ["tmomfix"]

    lists = [b.name for b in params.BLOCKS.values() if b.shape == "list"]
    counted = [b.name for b in params.BLOCKS.values() if b.shape == "counted"]
    assert len(lists) == 9        # blank-line terminated
    assert len(counted) == 5      # count-prefixed
    assert sorted(counted) == ["phwrite", "plot1d", "pulse", "ramp", "step"]

    raw = [b.name for b in params.BLOCKS.values() if b.shape == "raw"]
    assert sorted(raw) == ["idxw90", "notes", "xlwin"]
    assert [b.name for b in params.BLOCKS.values() if b.shape == "opaque"] == \
        ["species"]
    assert sorted(b.name for b in params.BLOCKS.values()
                  if b.shape == "special") == ["DFT+U", "atoms", "tm3fix"]


def test_deprecated_status_matches_the_fortran(branches):
    text = READINPUT.read_text().split("\n")
    starts = [line for _, line in branches] + [len(text) + 1]
    for i, (names, line) in enumerate(branches):
        body = "\n".join(text[line:starts[i + 1] - 1])
        says_gone = "no longer used" in body
        block = params.describe(names[0])
        assert (block.status != "active") == says_gone, (
            "%s: table says status=%r, readinput.f90:%d says %s"
            % (names[0], block.status, line,
               "no longer used" if says_gone else "in use"))


def test_every_block_has_a_description_and_a_category():
    for block in params.BLOCKS.values():
        assert block.desc and len(block.desc) > 4, block.name
        assert block.dsrc in ("manual", "fortran"), block.name
        assert block.cat in params.CATEGORIES, block.name
    # the manual documents well under half of the blocks; the rest are
    # described from the Fortran, and that split is worth pinning so a
    # regenerated table cannot quietly lose the manual text
    from_manual = [b.name for b in params.BLOCKS.values() if b.dsrc == "manual"]
    assert 120 <= len(from_manual) <= 200


def test_extension_blocks_match_the_patch_series():
    """The elkpy_* blocks live only in the BUILD copy of readinput.f90, so
    they are checked against patches/*.patch instead of vendor/."""
    patched = set()
    for patch in sorted(PATCHES.glob("*.patch")):
        for line in patch.read_text().split("\n"):
            if line.startswith("+case('elkpy"):
                patched.update(re.findall(r"'([^']*)'", line))
    assert patched, "no elkpy_* blocks found in patches/"
    assert patched == set(params.EXTENSION_BLOCKS)
    for block in params.EXTENSION_BLOCKS.values():
        assert block.source == "elkpy"
        assert "patches/" in block.note
    # and they must not be counted as upstream Elk
    assert not (set(params.EXTENSION_BLOCKS) & set(params.BLOCKS))


# ---------------------------------------------------------------------------
# (b) validate-then-render round trip, one per distinct shape
# ---------------------------------------------------------------------------

def _render(blocks):
    """Validate, then push through the real elk.in writer."""
    rendered = params.render_blocks(blocks)
    f = InputFile()
    for name, lines in rendered.items():
        f.add_block(name, lines)
    return f.render()


ROUND_TRIP = [
    # (block, value, substring that must appear in the rendered elk.in)
    ("maxscl", 40, "maxscl\n40\n"),                         # scalar int
    ("swidth", 0.02, "swidth\n0.02\n"),                     # scalar real
    ("spinorb", True, "spinorb\n.true.\n"),                 # scalar bool
    ("sppath", "/opt/species/", "sppath\n'/opt/species/'\n"),  # scalar str
    ("ngridk", (4, 4, 4), "ngridk\n4  4  4\n"),             # vector int
    ("vkloff", (0.5, 0.5, 0.5), "vkloff\n0.5  0.5  0.5\n"),  # vector real
    ("xctype", (20,), "xctype\n20\n"),                      # padded vector
    ("rgkmax", (8.0, 0.5), "rgkmax\n8.0  0.5\n"),           # optional tail
    ("avec", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
     "avec\n1.0  0.0  0.0\n0.0  1.0  0.0\n0.0  0.0  1.0\n"),  # block
    ("wplot", ((300, 100, 1), (-0.4, 0.4)),
     "wplot\n300  100  1\n-0.4  0.4\n"),                    # heterogeneous block
    ("optcomp", [(1, 1, 1), (2, 2, 2)],
     "optcomp\n1  1  1\n2  2  2\n"),                        # blank-terminated list
    ("mommtfix", [(1, 1, 0.0, 0.0, 2.5)],
     "mommtfix\n1  1  0.0  0.0  2.5\n"),                    # mixed-type list
    ("phwrite", [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
     "phwrite\n2\n0.0  0.0  0.0\n0.5  0.0  0.0\n"),         # counted list
    ("plot1d", ((2, 100), (0.0, 0.0, 0.0), (0.5, 0.5, 0.0)),
     "plot1d\n2  100\n0.0  0.0  0.0\n0.5  0.5  0.0\n"),     # counted, 2-value header
    ("notes", ["run A", "second line"],
     "notes\nrun A\nsecond line\n"),                        # raw text
    ("idxw90", "1-4,7", "idxw90\n1-4,7\n"),                 # raw, single line
    ("lmaxmat", 8, "lmaxmat\n8\n"),                         # deprecated
    ("elkpy_stmpol", 0.7, "elkpy_stmpol\n0.7\n"),           # elkpy extension
]


@pytest.mark.parametrize("name,value,expected", ROUND_TRIP)
def test_round_trip_renders_the_expected_elk_in_text(name, value, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = _render({name: value})
    assert expected in text


def test_every_shape_is_covered_by_the_round_trip():
    covered = set(params.describe(n).shape for n, _, _ in ROUND_TRIP)
    present = set(b.shape for b in params.BLOCKS.values())
    # special/opaque are irregular by construction and only pass through
    assert present - covered <= {"special", "opaque"}


def test_special_and_opaque_blocks_pass_through():
    text = _render({"DFT+U": [(1, 0), (1, 2, 0.29, 0.0)]})
    assert "DFT+U\n1  0\n1  2  0.29  0.0\n" in text


def test_special_blocks_accept_literal_lines_unquoted():
    """These are the blocks a user is most likely to copy verbatim out of an
    existing elk.in; a bare str would come back single-quoted, which Elk's
    list-directed read of that line cannot parse."""
    text = _render({"DFT+U": ["1 0", "1 2 0.29 0.0"]})
    assert "DFT+U\n1 0\n1 2 0.29 0.0\n" in text
    assert "'1 0'" not in text


def test_optional_derivative_tail_is_not_range_checked():
    """readinput.f90 checks rgkmax and ngridk(:) and then reads the trailing
    derivative into drgkmax/dngridk(:) with no check at all, so a negative
    derivative -- the usual case, since it is a step for a parameter sweep --
    must be accepted while a negative rgkmax is still refused."""
    assert "rgkmax\n7.0  -0.1\n" in _render({"rgkmax": (7.0, -0.1)})
    assert "ngridk\n4  4  4  -1  -1  -1\n" in _render(
        {"ngridk": (4, 4, 4, -1, -1, -1)})
    with pytest.raises(ParameterError):
        params.render_blocks({"rgkmax": (-7.0, 0.1)})
    with pytest.raises(ParameterError):
        params.render_blocks({"ngridk": (0, 4, 4, -1, -1, -1)})


def test_deprecated_block_warns_but_still_renders():
    with pytest.warns(DeprecationWarning, match="no longer used"):
        out = params.render_blocks({"lmaxmat": 8})
    assert out == {"lmaxmat": [8]}


def test_scalar_accepts_a_one_element_sequence():
    assert params.render_blocks({"maxscl": [40]}) == {"maxscl": [40]}


def test_int_is_accepted_where_elk_wants_a_real():
    out = params.render_blocks({"swidth": 1})
    assert str(out["swidth"][0]) == "1.0"


def test_small_reals_survive_the_writer():
    """inputfile._format_value renders a bare float as '%.10f', which would
    turn epsband's own default into 0.0000000000; params.FortranReal is what
    stops that."""
    assert "epsband\n1.0E-12\n" in _render({"epsband": 1e-12})
    assert "%.10f" % 1e-12 == "0.0000000000"      # the bug being avoided


def test_verbatim_lines_are_not_quoted():
    """A plain Python str is rendered 'quoted' by the writer, which would put
    literal quotes inside the note / the Wannier90 .win file."""
    text = _render({"notes": "no quotes here"})
    assert "notes\nno quotes here\n" in text
    assert "'no quotes here'" not in text


def test_rendered_values_are_json_serialisable():
    """Calculation._basis_signature json-dumps extra_blocks into the ground
    state cache manifest, so every rendered value must survive that, and two
    different values must not collide."""
    import json

    a = json.dumps(params.render_blocks({"epsband": 1e-12, "notes": "a"}),
                   sort_keys=True)
    b = json.dumps(params.render_blocks({"epsband": 1e-11, "notes": "b"}),
                   sort_keys=True)
    assert a != b
    assert "1e-12" in a


# ---------------------------------------------------------------------------
# (c) rejection
# ---------------------------------------------------------------------------

def test_unknown_block_name_raises_with_a_suggestion():
    with pytest.raises(ParameterError) as exc:
        params.render_blocks({"ngrdk": (4, 4, 4)})
    assert "ngrdk" in str(exc.value)
    assert "ngridk" in str(exc.value)


def test_case_matters_and_is_explained():
    with pytest.raises(ParameterError) as exc:
        params.render_blocks({"NGRIDK": (4, 4, 4)})
    assert "case-sensitive" in str(exc.value)
    assert "ngridk" in str(exc.value)


def test_wrong_arity_raises():
    with pytest.raises(ParameterError) as exc:
        params.render_blocks({"ngridk": (4, 4)})
    assert "ngridk" in str(exc.value)
    assert "3" in str(exc.value)


def test_too_many_values_raises():
    with pytest.raises(ParameterError):
        params.render_blocks({"vkloff": (0.0, 0.0, 0.0, 0.0)})


def test_wrong_type_raises():
    with pytest.raises(ParameterError):
        params.render_blocks({"maxscl": 40.5})
    with pytest.raises(ParameterError):
        params.render_blocks({"spinpol": 1})       # Elk wants .true./.false.
    with pytest.raises(ParameterError):
        params.render_blocks({"sppath": 3})


def test_range_checks_from_readinput_are_applied():
    # readinput.f90:496 stops on epslat <= 0
    with pytest.raises(ParameterError) as exc:
        params.render_blocks({"epslat": 0.0})
    assert "> 0" in str(exc.value)
    # readinput.f90:1114 stops on symtype outside 0..2
    with pytest.raises(ParameterError):
        params.render_blocks({"symtype": 3})
    # vkloff components must be in [0,1)
    with pytest.raises(ParameterError):
        params.render_blocks({"vkloff": (0.0, 0.0, 1.0)})
    params.render_blocks({"vkloff": (0.0, 0.0, 0.99)})


def test_removed_block_raises():
    with pytest.raises(ParameterError) as exc:
        params.render_blocks({"tmomfix": True})
    assert "removed" in str(exc.value)


def test_tasks_is_refused():
    with pytest.raises(ParameterError) as exc:
        params.render_blocks({"tasks": [0, 10]})
    assert "run_tasks" in str(exc.value)


def test_counted_list_count_mismatch_raises():
    with pytest.raises(ParameterError) as exc:
        params.render_blocks({"plot1d": ((3, 100), (0.0, 0.0, 0.0),
                                         (0.5, 0.5, 0.0))})
    assert "3 rows but 2" in str(exc.value)


def test_list_length_limit_is_enforced():
    with pytest.raises(ParameterError) as exc:
        params.render_blocks({"optcomp": [(1, 1, 1)] * 28})
    assert "at most 27" in str(exc.value)


def test_blank_verbatim_line_raises():
    with pytest.raises(ParameterError):
        params.render_blocks({"notes": ["fine", "  "]})


# ---------------------------------------------------------------------------
# discovery surface
# ---------------------------------------------------------------------------

def test_describe_returns_the_documented_fields():
    b = params.describe("rgkmax")
    assert b.default == 7.0
    assert b.type == "real"
    assert b.var == "rgkmax"
    assert b.module == "modmain"
    assert b.cat == "basis"
    assert b.opt == 1


def test_alias_lookup_returns_the_canonical_block():
    assert params.describe("lmaxvr").name == "lmaxo"
    assert params.describe("OMP_NUM_THREADS").name == "maxthd"
    assert params.describe("dft+u").name == "DFT+U"


def test_search_finds_blocks_by_name_and_by_description():
    # by description: neither of these has "smear" in its own name
    names = [b.name for b in params.search("smear")]
    assert "stype" in names and "autoswidth" in names
    # by name
    assert "ngridk" in [b.name for b in params.search("ngridk")]
    # by alias
    assert "lmaxo" in [b.name for b in params.search("lmaxvr")]


def test_categories_partition_the_table():
    total = sum(len(v) for v in params.CATEGORIES.values())
    assert total == len(params.BLOCKS) + len(params.EXTENSION_BLOCKS)
    for cat in params.categories():
        assert params.in_category(cat)


def test_signature_is_a_sentence_for_every_block():
    for b in list(params.BLOCKS.values()) + list(params.EXTENSION_BLOCKS.values()):
        assert isinstance(b.signature(), str) and b.signature()


def test_defaults_are_typed_consistently():
    for b in params.BLOCKS.values():
        if b.default is None or b.status != "active":
            continue
        if b.shape == "scalar":
            if b.type == "bool":
                assert isinstance(b.default, bool), b.name
            elif b.type == "int":
                assert isinstance(b.default, int) and not isinstance(b.default, bool), b.name
            elif b.type == "real":
                assert isinstance(b.default, (int, float)), b.name
            elif b.type == "str":
                assert isinstance(b.default, str), b.name
        elif b.shape == "vector":
            assert isinstance(b.default, tuple) and len(b.default) == b.n, b.name


#: Blocks whose readinput.f90 default is a SENTINEL that the same branch
#: would refuse if it were written in elk.in.  Only one exists in this
#: release; it is listed rather than skipped silently so a future one shows
#: up as a test failure.
SENTINEL_DEFAULTS = {"ngridq"}


def test_every_active_default_validates_against_its_own_block():
    """Feeding a block its own documented default must be accepted -- a
    cheap consistency check between the default column and the shape."""
    rejected = set()
    for b in params.BLOCKS.values():
        if b.status != "active" or b.default is None:
            continue
        if b.shape in ("special", "opaque", "raw", "counted"):
            continue
        try:
            params.render_blocks({b.name: b.default})
        except ParameterError:
            rejected.add(b.name)
    assert rejected == SENTINEL_DEFAULTS
    # and the one that is rejected says why in its note
    assert "sentinel" in params.describe("ngridq").note
