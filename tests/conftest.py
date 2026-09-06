"""Shared fixtures.

The only thing here is the LAPW export used by every Phase 1 test of the JAX
port.  It lives at SESSION scope because building it converges three real
ground states, and four separate test modules read it: without this each of
them would reconverge all three into its own `tmp_path`.
"""

import importlib.util

import numpy as np
import pytest

from elkpy import config
from elkpy.structure import Structure

SI_AVEC = [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)]
LAPW_KPOINT = (0.1, 0.2, 0.05)          # generic: no symmetry, no degeneracy
LAPW_CASES = ["si_apword1", "si_apword2", "hbn"]


def apword2_species_dir(tmp_path):
    """Elk's own Si.in with the APW order raised from 1 to 2 -- derivative
    orders 0 and 1, the textbook LAPW basis u_l and du_l/dE.  Every species
    file Elk ships sets apword = 1, so without this the order axis of haa and
    hloa is length 1 and never tested."""
    source = config.resolve_species_path() / "Si.in"
    out = []
    for line in source.read_text().splitlines(keepends=True):
        if ": apword" in line:
            out.append("   2                                        : apword\n")
            out.append("    0.1500   0  F                           : apwe0, apwdm, apwve\n")
            out.append("    0.1500   1  F\n")
            continue
        if ": apwe0, apwdm, apwve" in line and out and "apword" in out[-3]:
            continue
        out.append(line)
    directory = tmp_path / "species_apword2"
    directory.mkdir(exist_ok=True)
    (directory / "Si.in").write_text("".join(out))
    return directory


@pytest.fixture(scope="session")
def _lapw_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("lapw")


def _export(structure, workdir, sppath=None, **kwargs):
    """The LAPW export at `LAPW_KPOINT`, plus Elk's own momentum matrix there.

    `pmat` rides along under a key of its own because it is the independent
    Fortran reference for the k-derivative in the assembly test -- genpmatk,
    which shares no code with hmlfv/olpfv.
    """
    from elkpy.calculation import Calculation
    calc = Calculation(structure=structure, workdir=workdir, **kwargs)
    if sppath is not None:
        calc.sppath = sppath
    calc.ensure_ground_state()
    with calc.eigenstate_session() as session:
        export = session.lapw_problem(LAPW_KPOINT)
    export["_pmat"] = np.asarray(calc.get_momentum_matrix(LAPW_KPOINT).pmat)
    return export


@pytest.fixture(scope="session")
def exports(_lapw_tmp):
    """All three cases, each ground state converged once per session.

    A single dict rather than three parametrized fixtures because the tests
    select among them by name: pytest's `getfixturevalue` cannot reach into a
    parametrized fixture.
    """
    if not config.default_elk_binary().is_file():
        pytest.skip("elk binary not built; see docs/design.md #8")
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax not installed; pip install -e .[jax]")
    out = {}
    silicon = Structure(
        avec=SI_AVEC,
        species={"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]})
    out["si_apword1"] = _export(silicon, _lapw_tmp / "si1",
                                ngridk=(2, 2, 2), rgkmax=7.0)
    out["si_apword2"] = _export(
        silicon, _lapw_tmp / "si2", sppath=apword2_species_dir(_lapw_tmp),
        ngridk=(2, 2, 2), rgkmax=7.0)
    a, c = 4.746, 20.0
    hbn = Structure(
        avec=[(a, 0.0, 0.0), (-a / 2, a * 3 ** 0.5 / 2, 0.0), (0.0, 0.0, c)],
        species={"B": [(0.0, 0.0, 0.0)], "N": [(1 / 3, 2 / 3, 0.0)]})
    out["hbn"] = _export(hbn, _lapw_tmp / "hbn", ngridk=(2, 2, 1), rgkmax=6.0)
    return out
