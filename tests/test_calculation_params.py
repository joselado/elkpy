"""Tests for the Calculation-facing parameter surface (elkpy.tasks.params).

Two halves:

* the mixin's own behaviour -- merging into `extra_blocks`, refusing
  `tasks`, warning about Calculation-owned blocks, cache invalidation.
  These construct a `Calculation` but never run it, so they need no binary.
* one real Elk run that feeds a rendered set of blocks -- covering every
  rendering shape the writer has to get right -- to the actual `readinput`,
  and checks it neither rejected a block name nor failed to parse a line.
  That is the only claim about this bucket a binary can actually settle:
  the table is validated against `readinput.f90` in test_params.py, but only
  Elk can confirm that what the renderer emits is what Elk reads back.

The mixin is applied here by a local subclass rather than by editing
`Calculation`; the integrator makes `Calculation` inherit it for real.
"""

import warnings

import pytest

from elkpy import config, params
from elkpy.calculation import Calculation
from elkpy.params import ParameterError
from elkpy.structure import Structure
from elkpy.tasks.params import ParametersMixin

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
SI_SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}


# Calculation now inherits ParametersMixin (see elkpy/tasks/__init__.py), so
# this alias IS the wired class. It must not be a subclass declared as
# ``(ParametersMixin, Calculation)``: with Calculation already inheriting the
# mixin, C3 linearization has no consistent order and the import fails.
ParamCalculation = Calculation


@pytest.fixture
def calc(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    return ParamCalculation(s, tmp_path / "si", ngridk=(2, 2, 2), rgkmax=6.0)


# ---------------------------------------------------------------------------
# mixin behaviour -- no Elk run
# ---------------------------------------------------------------------------

def test_set_parameters_merges_into_extra_blocks(calc):
    calc.set_parameters(maxscl=40, epsengy=1e-8)
    assert calc.extra_blocks["maxscl"] == [40]
    assert str(calc.extra_blocks["epsengy"][0]) == "1.0E-08"


def test_set_parameters_accepts_a_dict_for_unspellable_names(calc):
    calc.set_parameters({"DFT+U": [(1, 0), (1, 2, 0.29, 0.0)]})
    assert calc.extra_blocks["DFT+U"][0] == (1, 0)


def test_set_parameters_rejects_a_bad_name_before_any_run(calc):
    with pytest.raises(ParameterError) as exc:
        calc.set_parameters(maxscf=40)
    assert "maxscl" in str(exc.value)
    assert calc.extra_blocks == {}


def test_set_parameters_rejects_tasks(calc):
    with pytest.raises(ParameterError) as exc:
        calc.set_parameters({"tasks": [0, 10]})
    assert "run_tasks" in str(exc.value)


def test_set_parameters_warns_on_calculation_owned_blocks(calc):
    with pytest.warns(UserWarning, match="also written by Calculation"):
        calc.set_parameters(rgkmax=9.0)
    assert str(calc.extra_blocks["rgkmax"][0]) == "9.0"


def test_set_parameters_changes_the_ground_state_cache_signature(tmp_path):
    """extra_blocks is part of the manifest, so a parameter set after
    construction must invalidate a previously cached ground state rather
    than silently reuse it.

    This is the one test in this half that cannot use the plain `calc`
    fixture: `_basis_signature` includes the binary's own path and mtime --
    that is what stops `ensure_ground_state` reusing a run made by a
    different build -- so it needs a file to stat.  An empty one does: the
    claim here is about `extra_blocks`, and the binary is incidental to it.
    """
    import json

    from elkpy.launcher import LocalLauncher

    binary = tmp_path / "fake-elk"
    binary.touch()
    s = Structure(SI_AVEC, SI_SPECIES)
    calc = ParamCalculation(s, tmp_path / "si", ngridk=(2, 2, 2), rgkmax=6.0,
                            launcher=LocalLauncher(elk_binary=binary))

    # snapshot, not a reference: _basis_signature hands back the live dict
    before = json.dumps(calc._basis_signature(), sort_keys=True)
    calc.set_parameters(maxscl=40)
    after = json.dumps(calc._basis_signature(), sort_keys=True)
    assert after != before
    assert '"maxscl": [40]' in after


def test_unset_parameters_removes_and_still_checks_the_name(calc):
    calc.set_parameters(maxscl=40)
    calc.unset_parameters("maxscl")
    assert "maxscl" not in calc.extra_blocks
    with pytest.raises(ParameterError):
        calc.unset_parameters("no_such_block")


def test_validate_blocks_does_not_apply(calc):
    calc.validate_blocks({"maxscl": 40})
    assert calc.extra_blocks == {}


def test_effective_parameters_shows_the_whole_input(calc):
    calc.set_parameters(maxscl=40)
    eff = calc.effective_parameters()
    # built through _add_base_blocks, so it is the real elk.in content --
    # including the blocks derived from the Structure, not just the typed ones
    assert eff["ngridk"] == [(2, 2, 2)]
    assert str(eff["rgkmax"][0]) == "6.0"
    assert eff["maxscl"] == [40]
    assert "avec" in eff and "atoms" in eff
    assert "swidth" not in eff
    assert "swidth" in calc.effective_parameters(include_defaults=True)


def test_discovery_helpers_are_reachable_from_the_calculation(calc):
    assert calc.describe_parameter("lmaxvr").name == "lmaxo"
    assert "basis" in calc.parameter_categories()
    assert calc.parameters_in_category("phonons")
    assert "ngridk" in [b.name for b in calc.search_parameters("k-point mesh")]
    text = calc.explain_parameter("rgkmax")
    assert "readinput.f90:" in text and "default: 7.0" in text


def test_set_parameters_rejects_a_name_given_twice(calc):
    with pytest.raises(ParameterError):
        calc.set_parameters({"maxscl": 40}, maxscl=50)


# ---------------------------------------------------------------------------
# one real Elk run: does readinput actually accept what the renderer writes?
# ---------------------------------------------------------------------------

# every rendering shape in one input: scalar int/real/bool, vector int/real,
# a padded vector, an optional-tail scalar, a fixed multi-line block, a
# blank-line-terminated list, a count-prefixed list, verbatim text, and a
# real small enough that inputfile.py's own "%.10f" would flatten it to zero.
RENDER_COVER = {
    # maxscl is deliberately small: this test asks whether readinput accepts
    # what the renderer wrote, not whether Si converges, and the calculation
    # below is built with raise_on_nonconvergence=False for the same reason
    "maxscl": 12,                                   # scalar int
    "swidth": 0.02,                                 # scalar real
    "tshift": False,                                # scalar bool
    "tforce": True,
    "ngridq": (1, 1, 1),                            # vector int
    "sqaxis": (0.0, 0.0, 1.0),                      # vector real
    "xctype": (3,),                                 # padded vector
    "gmaxvr": (12.0, 0.0),                          # optional tail
    "wplot": ((200, 100, 1), (-0.5, 0.5)),          # heterogeneous block
    "plot2d": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
               (0.0, 1.0, 0.0), (10, 10)),          # fixed multi-line block
    "optcomp": [(1, 1, 1), (2, 2, 2)],              # blank-terminated list
    "phwrite": [(0.0, 0.0, 0.0)],                   # count-prefixed list
    "epsband": 1e-12,                               # sub-1e-10 real
    "notes": ["written by elkpy.params"],           # verbatim text
    "lmaxdb": 3,
}


@pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)
def test_rendered_blocks_are_accepted_by_the_real_readinput(tmp_path):
    s = Structure(SI_AVEC, SI_SPECIES)
    calc = ParamCalculation(s, tmp_path / "si", ngridk=(2, 2, 2), rgkmax=5.0,
                            raise_on_nonconvergence=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # xctype is Calculation-owned
        calc.set_parameters(RENDER_COVER)
    energy = calc.get_energy()

    log = (calc.workdir / "elk.out").read_text()
    # readinput.f90 stops with one of these on a bad block name or a line it
    # cannot parse; both are fatal, so getting an energy at all is already
    # strong evidence, but naming them makes a failure diagnosable
    assert "Error(readinput)" not in log
    assert "invalid block name" not in log
    # every block was actually read: Elk echoes the notes block into INFO.OUT
    info = (calc.workdir / "INFO.OUT").read_text()
    assert "written by elkpy.params" in info
    assert -600 < energy < -550


@pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)
def test_render_cover_touches_every_ordinary_shape():
    """Guards the coverage claim above rather than the run itself."""
    shapes = set(params.describe(n).shape for n in RENDER_COVER)
    assert {"scalar", "vector", "block", "list", "counted", "raw"} <= shapes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params.render_blocks(RENDER_COVER)
