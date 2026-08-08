"""Tests for per-species spin-orbit coupling scaling (Calculation(spinorb=,
soc_scale=)), backed by the elkpy Fortran extension in
patches/0001-per-species-soc-scale.patch (vendor/elk carries no per-species
SOC control upstream -- only the single global `socscf` scalar).

The constructor-validation tests don't need the elk binary; the energy
comparisons do and are skipped if it hasn't been built, same as
test_calculation_si.py.
"""

import pytest

from elkpy import config
from elkpy.structure import Structure

BI_AVEC = [(0.0, 5.0, 5.0), (5.0, 0.0, 5.0), (5.0, 5.0, 0.0)]
BISI_AVEC = [(10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)]

needs_elk = pytest.mark.skipif(
    not config.default_elk_binary().is_file(),
    reason="elk binary not built; see docs/design.md #8",
)


def test_soc_scale_requires_spinorb(tmp_path):
    s = Structure(BI_AVEC, {"Bi": [(0.0, 0.0, 0.0)]})
    with pytest.raises(ValueError, match="spinorb"):
        s.get_calculation(tmp_path / "calc", soc_scale={"Bi": 1.5})


def test_soc_scale_rejects_unknown_species(tmp_path):
    s = Structure(BI_AVEC, {"Bi": [(0.0, 0.0, 0.0)]})
    with pytest.raises(ValueError, match="Si"):
        s.get_calculation(tmp_path / "calc", spinorb=True, soc_scale={"Si": 1.0})


def test_soc_scale_rejects_negative(tmp_path):
    s = Structure(BI_AVEC, {"Bi": [(0.0, 0.0, 0.0)]})
    with pytest.raises(ValueError, match="soc_scale"):
        s.get_calculation(tmp_path / "calc", spinorb=True, soc_scale={"Bi": -1.0})


@needs_elk
def test_soc_scale_matches_global_socscf_for_single_species(tmp_path):
    """A per-species override on the sole species in the structure should
    exactly reproduce the effect of the global `socscf` scalar -- both paths
    go through the same gensocfr.f90 computation, just with a different
    scale value selected."""
    s = Structure(BI_AVEC, {"Bi": [(0.0, 0.0, 0.0)]})
    e_global = s.get_calculation(
        tmp_path / "global",
        xc="PW",
        spinorb=True,
        ngridk=(1, 1, 1),
        extra_blocks={"socscf": [3.0]},
    ).get_energy()
    e_per_species = s.get_calculation(
        tmp_path / "per_species",
        xc="PW",
        spinorb=True,
        ngridk=(1, 1, 1),
        soc_scale={"Bi": 3.0},
    ).get_energy()
    assert e_global == pytest.approx(e_per_species, abs=1e-8)


@needs_elk
def test_soc_scale_is_independent_per_species(tmp_path):
    """Bi (Z=83, strong SOC) and Si (Z=14, weak SOC) share a cell far enough
    apart to converge independently. Zeroing each species' SOC scale in turn
    should move the energy away from the spinorb=True default, and the Bi
    effect should dwarf the Si effect -- confirms the scale is actually
    applied per-atom-via-species rather than globally."""
    s = Structure(BISI_AVEC, {"Bi": [(0.0, 0.0, 0.0)], "Si": [(0.5, 0.5, 0.5)]})

    def energy(label, soc_scale):
        calc = s.get_calculation(
            tmp_path / label, xc="PW", spinorb=True, ngridk=(1, 1, 1), soc_scale=soc_scale
        )
        return calc.get_energy()

    e_default = energy("default", None)
    e_bi_off = energy("bi_off", {"Bi": 0.0})
    e_si_off = energy("si_off", {"Si": 0.0})

    delta_bi = abs(e_default - e_bi_off)
    delta_si = abs(e_default - e_si_off)
    assert delta_bi > 1e-4
    assert delta_si > 1e-6
    assert delta_bi > 10 * delta_si
