"""Unit tests for the medium-severity findings of docs/review_findings.md
(no Elk run).

Each of these is a guard, a normalisation or a refusal that can be decided
in Python. The two that need a real run -- the empty-WFCORE skip on a
hydride and the stress pressure's scale factor -- are in
``tests/test_calculation_medium_findings.py`` instead.
"""

import numpy as np
import pytest

from elkpy.calculation import Calculation
from elkpy.structure import Structure
from elkpy.tasks.phonons import PhononTasks

AVEC = [(5.13, 5.13, 0.0), (5.13, 0.0, 5.13), (0.0, 5.13, 5.13)]
SI = {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]}


def make(tmp_path, species=None, **kwargs):
    return Calculation(
        Structure(AVEC, species or SI), tmp_path / "si", xc="PW",
        ngridk=(2, 2, 2), rgkmax=6.0, **kwargs
    )


# ---------------------------------------------------------------------------
# Finding 9 -- _check_bse_states read the nempty block as the state count
# ---------------------------------------------------------------------------


def test_bse_accepts_elks_own_defaults(tmp_path):
    """init1.f90:316 scales nempty0 PER ATOM, so 2-atom Si at nempty0=4 has
    8 empty states against a default ncbse of 3. The old guard refused this
    outright, before any Elk process started."""
    make(tmp_path)._check_bse_states(3)


def test_bse_counts_empty_states_per_atom(tmp_path):
    """extra_blocks={'nempty': [2]} on a 2-atom cell is nempty0=2, i.e. FOUR
    empty states -- the old guard compared ncbse against the raw 2."""
    calc = make(tmp_path, extra_blocks={"nempty": [2]})
    calc._check_bse_states(4)          # 4 <= nempty + 1 = 5
    with pytest.raises(ValueError, match="ncbse <= nempty \\+ 1 = 5"):
        calc._check_bse_states(6)


def test_bse_refuses_what_genidxbse_would_stop_on(tmp_path):
    """genidxbse.f90:82 stops on ntop + ncbse0 > nstsv, i.e. ncbse > nempty+1."""
    with pytest.raises(ValueError, match="not enough conduction states"):
        make(tmp_path)._check_bse_states(12)   # nempty = 8, bound 9


def test_bse_bound_scales_with_the_atom_count(tmp_path):
    """The same nempty0 buys more states in a bigger cell, so the refusal
    must move with natmtot rather than with the block value."""
    big = {"Si": [(0.1 * i, 0.0, 0.0) for i in range(4)]}
    make(tmp_path, species=big)._check_bse_states(16)  # nempty = 16, bound 17
    with pytest.raises(ValueError):
        make(tmp_path)._check_bse_states(16)           # nempty = 8, bound 9


# ---------------------------------------------------------------------------
# Finding 5 -- LAMBDAQ.OUT holds half the Allen mode coupling
# ---------------------------------------------------------------------------


def test_lambdaq_is_half_the_allen_coupling():
    """occupy.f90:94 makes fermidos the TOTAL both-spin DOS and
    ephcouple.f90:136 puts the spin sum into GAMMAQ, so writelambda.f90:25's
    division by pi*fermidos gives half of Allen's gamma/(pi N_spin w^2) --
    which is what alpha2f.f90:99's twopi*(fermidos/2) uses, and therefore
    what get_eliashberg_function() reports as `lambda`."""
    assert PhononTasks.EPH_LAMBDAQ_TO_ALLEN == 2.0


def test_eph_tables_apply_the_factor_and_keep_the_raw_column(tmp_path):
    """Both keys come back, and they differ by exactly that factor -- so a
    caller comparing against LAMBDAQ.OUT itself still can."""
    def table(path, values):
        with open(path, "w") as fh:
            fh.write("\n   2 : total number of atoms\n     1 : number of q-points\n\n")
            fh.write("     1 : q-point\n")
            fh.write("  0.0  0.0  0.0 : q-vector (lattice coordinates)\n")
            fh.write("  0.0  0.0  0.0 : q-vector (Cartesian coordinates)\n")
            for i, v in enumerate(values, start=1):
                fh.write(f"{i:4d}{v:18.10G}\n")
            fh.write("\n")

    from elkpy.tasks.phonons import PHONON_OUTPUT_FILES
    table(tmp_path / PHONON_OUTPUT_FILES["gammaq"], [1e-4, 2e-4, 3e-4])
    table(tmp_path / PHONON_OUTPUT_FILES["lambdaq"], [0.25, 0.5, 0.75])

    result = make(tmp_path)._parse_eph_tables(tmp_path)
    np.testing.assert_allclose(result["couplings_as_written"], [[0.25, 0.5, 0.75]])
    np.testing.assert_allclose(result["couplings"], [[0.5, 1.0, 1.5]])


# ---------------------------------------------------------------------------
# Finding 8 -- get_expiqr's q = 0 default returned a bare identity
# ---------------------------------------------------------------------------


def test_expiqr_refuses_zero_q(tmp_path):
    """genexpmat.f90:30-38 returns the identity before it looks at a
    wavefunction, so q=0 was a full Elk run to obtain delta_ij -- and it was
    the DEFAULT."""
    with pytest.raises(ValueError, match="delta_ij"):
        make(tmp_path).get_expiqr(vecql=(0.0, 0.0, 0.0))


def test_expiqr_requires_q(tmp_path):
    with pytest.raises(TypeError):
        make(tmp_path).get_expiqr()


# ---------------------------------------------------------------------------
# Finding 7 -- a bare Fortran `stop` exits 0
# ---------------------------------------------------------------------------


def test_md_restart_needs_the_file_it_reads(tmp_path):
    """The message named ATDVC.OUT while the check tested TIMESTEP.OUT, and
    ATDVC.OUT is the one moldyn.f90:36-43 actually reads. A run too short to
    reach a force step leaves the second and not the first."""
    calc = make(tmp_path)
    subdir = calc.workdir / "md"
    subdir.mkdir(parents=True)
    (subdir / "TIMESTEP.OUT").write_text("0.1\n")
    with pytest.raises(FileNotFoundError, match="ATDVC.OUT"):
        calc.get_molecular_dynamics(restart=True, label="md")


def test_elk_error_line_is_raised_despite_exit_zero(tmp_path):
    """readatdvc and readtimes both print and then `stop`, which exits 0, so
    the launcher's return-code test sees success. On the one non-wiping run
    mode that means the PREVIOUS trajectory is parsed and returned as new."""
    log = tmp_path / "elk.out"
    log.write_text(
        "Elk code version 11.0.2 started\n"
        " Error(readatdvc): error opening ATDVC.OUT\n"
    )
    with pytest.raises(RuntimeError, match="readatdvc"):
        Calculation._raise_on_elk_error(log, tmp_path)


def test_a_clean_log_raises_nothing(tmp_path):
    log = tmp_path / "elk.out"
    log.write_text("Elk code version 11.0.2 started\n Elk code stopped\n")
    Calculation._raise_on_elk_error(log, tmp_path)


def test_a_missing_log_raises_nothing(tmp_path):
    """The check must not turn an unreadable log into a spurious failure."""
    Calculation._raise_on_elk_error(tmp_path / "nope.out", tmp_path)
