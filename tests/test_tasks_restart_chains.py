"""Unit tests for the three restart-style task wrappers (no Elk run).

These are the three high-severity entries of `docs/review_findings.md`, and
they were one bug: a method offered a reuse or restart flag whose Elk task
exists to READ a file, then dispatched through
``Calculation._run_resumed``, whose unconditional ``shutil.rmtree``
(docs/design.md #4) had just deleted it. Every one was therefore dead under
ANY label -- a fresh label gives an empty directory and the same Fortran
abort -- and destructive under the label it was given.

Nothing here runs Elk. A fake launcher stands in for the subprocess and
records the ``elk.in`` each call writes, because what these tests check is
exactly what the bug got wrong: which directory the task runs in, which
task numbers reach the ``tasks`` block, and which input blocks a reused
file pins.
"""

import json

import pytest

from elkpy.calculation import Calculation
from elkpy.structure import Structure
from elkpy.tasks import magnetism_manybody as mmb

AVEC = [(5.13, 5.13, 0.0), (5.13, 0.0, 5.13), (0.0, 5.13, 5.13)]
SPECIES = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}


class FakeLauncher:
    """Records each run's directory and ``elk.in``, and creates the output
    files the calling method checks for afterwards."""

    def __init__(self, produces=()):
        self.runs = []
        self.produces = tuple(produces)

    def run(self, workdir, log_name="elk.out"):
        self.runs.append((workdir, (workdir / "elk.in").read_text()))
        for name in self.produces:
            (workdir / name).touch()
        return workdir / log_name


def make(tmp_path, launcher=None, **kwargs):
    return Calculation(
        Structure(AVEC, SPECIES), tmp_path / "si", xc="PW", ngridk=(2, 2, 2),
        rgkmax=6.0, launcher=launcher, **kwargs
    )


def tasks_of(text):
    """The integers in the ``tasks`` block of an ``elk.in``."""
    lines = text.splitlines()
    start = lines.index("tasks") + 1
    out = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            break
        out.append(int(line.split()[0]))
    return out


def stage(subdir, blocks, ngridk=(2, 2, 2)):
    """Write the producing run's sidecar, as the real method does."""
    subdir.mkdir(parents=True, exist_ok=True)
    payload = {"blocks": blocks, "ngridk": list(ngridk), "vkloff": None}
    with open(subdir / mmb._STAGE_MANIFEST, "w") as fh:
        json.dump(payload, fh, default=str)


# ---------------------------------------------------------------------------
# Finding 3 -- get_anomalous_entropy's prerequisite chain
# ---------------------------------------------------------------------------


def test_anomalous_entropy_chain_writes_ephmat_and_raises_lmaxi(tmp_path, monkeypatch):
    """Task 270 reads EPHMAT.OUT, which ONLY task 241 writes
    (ephcouple.f90:131 is the single putephmat call site in the tree), and
    task 205 stops outright at readinput.f90:84's default lmaxi=1
    (phonon.f90:34)."""
    calc = make(tmp_path)
    seen = {}

    def record(label, tasks, extra_blocks=None, ngridk=None, vkloff=None):
        seen["tasks"] = list(tasks)
        seen["blocks"] = dict(extra_blocks or {})
        return tmp_path / "nowhere"

    monkeypatch.setattr(calc, "_run_resumed", record)
    with pytest.raises(FileNotFoundError):  # the parse of a directory that is not there
        calc.get_anomalous_entropy((2, 2, 2))

    assert seen["tasks"] == [205, 241, 270, 285]
    assert 240 not in seen["tasks"]
    assert seen["blocks"]["lmaxi"] == [2]
    assert seen["blocks"]["ngridq"] == [(2, 2, 2)]


def test_anomalous_entropy_refuses_an_off_mesh_qgrid(tmp_path):
    """ephcouple.f90 looks k+q up with findkpt, which stops off-mesh."""
    calc = make(tmp_path)
    with pytest.raises(ValueError, match="commensurate"):
        calc.get_anomalous_entropy((3, 3, 3))


def test_anomalous_entropy_lmaxi_floor_is_the_phonon_familys(tmp_path):
    """The floor is enforced in one place -- phonons._phonon_blocks -- not
    transcribed a second time here."""
    calc = make(tmp_path)
    with pytest.raises(ValueError, match="lmaxi must be >= 2"):
        calc.get_anomalous_entropy((2, 2, 2), lmaxi=1)


# ---------------------------------------------------------------------------
# Finding 1 -- get_gw_self_energy(reuse_epsinv=True), task 601
# ---------------------------------------------------------------------------


def test_gw_reuse_epsinv_refuses_a_directory_without_the_file(tmp_path):
    calc = make(tmp_path)
    with pytest.raises(ValueError, match="EPSINV.OUT"):
        calc.get_gw_self_energy(reuse_epsinv=True)


def test_gw_reuse_epsinv_runs_in_place_without_task_1(tmp_path):
    """Task 601 calls readstate itself (gwsefm.f90:19-24), so the run needs
    no prepended task 1 -- and must not get a wiped directory, since
    EPSINV.OUT is the whole point."""
    launcher = FakeLauncher(produces=[mmb.FILE_GW_SELF_ENERGY])
    calc = make(tmp_path, launcher=launcher)
    subdir = calc.workdir / "gw"
    blocks = {
        "wmaxgw": [5.0], "tempk": [1500.0], "nempty": [20],
        "gmaxrf": [3.0], "actype": [10],
    }
    stage(subdir, blocks)
    (subdir / mmb.FILE_GW_EPSINV).write_bytes(b"opaque")
    (subdir / "STATE.OUT").write_bytes(b"opaque")

    out = calc.get_gw_self_energy(reuse_epsinv=True)

    assert out == subdir
    assert (subdir / mmb.FILE_GW_EPSINV).read_bytes() == b"opaque"  # not wiped
    assert len(launcher.runs) == 1
    where, text = launcher.runs[0]
    assert where == subdir
    assert tasks_of(text) == [601]


def test_gw_reuse_epsinv_refuses_a_changed_tempk(tmp_path):
    """genwgw.f90 builds the Matsubara count from wmaxgw and tempk, so
    either one changes nwrf and getcfgq.f90 stops with 'differing m'. The
    docstring used to advertise exactly this as the reason to use 601."""
    calc = make(tmp_path)
    subdir = calc.workdir / "gw"
    stage(subdir, {"wmaxgw": [5.0], "tempk": [1500.0], "nempty": [20], "gmaxrf": [3.0]})
    (subdir / mmb.FILE_GW_EPSINV).write_bytes(b"opaque")
    with pytest.raises(ValueError, match="tempk"):
        calc.get_gw_self_energy(reuse_epsinv=True, tempk=1000.0)


def test_gw_reuse_epsinv_refuses_a_changed_nempty(tmp_path):
    """The one Elk cannot catch: nempty changes the states epsinv is built
    from without changing any record dimension, so the stale file reads
    cleanly and the self-energy is wrong."""
    calc = make(tmp_path)
    subdir = calc.workdir / "gw"
    stage(subdir, {"wmaxgw": [5.0], "tempk": [1500.0], "nempty": [20], "gmaxrf": [3.0]})
    (subdir / mmb.FILE_GW_EPSINV).write_bytes(b"opaque")
    with pytest.raises(ValueError, match="nempty"):
        calc.get_gw_self_energy(reuse_epsinv=True, nempty=40)


def test_gw_reuse_epsinv_allows_a_changed_tsediag(tmp_path):
    """tsediag reaches only gwsefmk.f90:209 and dysonr.f90:33 -- the
    self-energy's own matrix structure, not the screening."""
    launcher = FakeLauncher(produces=[mmb.FILE_GW_SELF_ENERGY])
    calc = make(tmp_path, launcher=launcher)
    subdir = calc.workdir / "gw"
    stage(subdir, {"wmaxgw": [5.0], "tempk": [1500.0], "nempty": [20], "gmaxrf": [3.0]})
    (subdir / mmb.FILE_GW_EPSINV).write_bytes(b"opaque")
    calc.get_gw_self_energy(reuse_epsinv=True, tsediag=True)
    assert "tsediag" in launcher.runs[0][1]


def test_gw_reuse_epsinv_records_the_merged_blocks(tmp_path):
    """The sidecar the dependent tasks (610/630/640) replay must describe
    what actually ran: the producing run's blocks with this call's on top."""
    launcher = FakeLauncher(produces=[mmb.FILE_GW_SELF_ENERGY])
    calc = make(tmp_path, launcher=launcher)
    subdir = calc.workdir / "gw"
    stage(subdir, {
        "wmaxgw": [5.0], "tempk": [1500.0], "nempty": [20], "gmaxrf": [3.0],
        "ngridq": [[1, 1, 1]],
    })
    (subdir / mmb.FILE_GW_EPSINV).write_bytes(b"opaque")
    calc.get_gw_self_energy(
        reuse_epsinv=True, wmaxgw=5.0, tempk=1500.0, nempty=20, gmaxrf=3.0,
        ngridq=(1, 1, 1), nspade=4,
    )
    recorded = json.loads((subdir / mmb._STAGE_MANIFEST).read_text())
    assert recorded["blocks"]["nspade"] == [4]        # this call's
    assert recorded["blocks"]["ngridq"] == [[1, 1, 1]]  # the producing run's
    assert recorded["ngridk"] == [2, 2, 2]


# ---------------------------------------------------------------------------
# Finding 2 -- get_ulr_ground_state(from_state=True), task 701
# ---------------------------------------------------------------------------

AVECU = [(2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]


def test_ulr_from_state_refuses_a_directory_without_the_state(tmp_path):
    calc = make(tmp_path)
    with pytest.raises(ValueError, match="STATE_ULR.OUT"):
        calc.get_ulr_ground_state(AVECU, (2, 2, 2), from_state=True)


def test_ulr_from_state_runs_in_place_without_task_1(tmp_path):
    """gndstulr.f90:57-63 takes the readstulr branch for task 701; the file
    it reads is in the run directory _run_resumed would have wiped."""
    launcher = FakeLauncher()
    calc = make(tmp_path, launcher=launcher)
    subdir = calc.workdir / "ulr"
    stage(subdir, {"avecu": AVECU, "ngridq": [[2, 2, 2]]})
    (subdir / mmb.FILE_ULR_STATE).write_bytes(b"old")
    (subdir / "STATE.OUT").write_bytes(b"opaque")

    # the fake launcher writes nothing, so the state file's timestamp cannot
    # advance -- which is the failure this restart path must report
    with pytest.raises(RuntimeError, match="untouched"):
        calc.get_ulr_ground_state(AVECU, (2, 2, 2), from_state=True)

    assert (subdir / mmb.FILE_ULR_STATE).read_bytes() == b"old"  # not wiped
    where, text = launcher.runs[0]
    assert where == subdir
    assert tasks_of(text) == [701]


def test_ulr_from_state_refuses_a_changed_ultracell(tmp_path):
    """readstulr.f90 checks natmtot/npcmtmax/ngtc/ngtot/ndmag/fsmtype -- all
    unit-cell quantities. A different avecu is read without complaint and
    means nothing, so the refusal has to be here."""
    calc = make(tmp_path)
    subdir = calc.workdir / "ulr"
    stage(subdir, {"avecu": AVECU, "ngridq": [[2, 2, 2]]})
    (subdir / mmb.FILE_ULR_STATE).write_bytes(b"old")
    bigger = [(3.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    with pytest.raises(ValueError, match="avecu"):
        calc.get_ulr_ground_state(bigger, (2, 2, 2), from_state=True)


def test_ulr_from_state_allows_a_finer_qgrid(tmp_path):
    """readstulr.f90:114-127 maps the file's own Q-vectors onto the new grid
    and zeroes the rest, so this restart is supported rather than a
    mismatch -- ngridq is deliberately not pinned."""
    launcher = FakeLauncher()
    calc = make(tmp_path, launcher=launcher)
    subdir = calc.workdir / "ulr"
    stage(subdir, {"avecu": AVECU, "ngridq": [[2, 2, 2]]})
    (subdir / mmb.FILE_ULR_STATE).write_bytes(b"old")
    with pytest.raises(RuntimeError, match="untouched"):  # got past every guard
        calc.get_ulr_ground_state(AVECU, (4, 4, 4), from_state=True)
    assert "ngridq 4 4 4" in " ".join(launcher.runs[0][1].split())


def test_ulr_from_state_refuses_a_changed_kmesh(tmp_path):
    """_run_dependent replays the producing run's mesh, so a different one
    here would be silently ignored rather than applied."""
    calc = make(tmp_path)
    subdir = calc.workdir / "ulr"
    stage(subdir, {"avecu": AVECU}, ngridk=(2, 2, 2))
    (subdir / mmb.FILE_ULR_STATE).write_bytes(b"old")
    with pytest.raises(ValueError, match="ngridk"):
        calc.get_ulr_ground_state(AVECU, (2, 2, 2), from_state=True, ngridk=(4, 4, 4))
