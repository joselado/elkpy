"""Unit tests for the small text-report parsers of the atomic /
expectation-value tasks (no Elk run).

Fixtures transcribe the ``write`` statements of ``src/writelsj.f90``
(tasks 15/16), ``src/writeevsp.f90`` (150), ``src/wfcrplot.f90`` (65) and
``src/writesf.f90`` (14) -- FORMAT-DERIVED verification, not
binary-verified.

These formats are prose, not columns, and two details are easy to get wrong
in a way no shape check would catch: the ``L``/``S``/``J`` label lines are
indented by TWO spaces in LSJ.OUT and by ONE in LSJ_KST.OUT (so a parser
keyed on the exact prefix breaks on one of them), and the ``Species`` line
of LSJ.OUT begins with an ``S`` just like the ``S :`` value line, so a
naive first-character dispatch misreads the species header as a spin vector.
"""

import math

import numpy as np
import pytest

from elkpy.parsers import atomicstates


def g(v, w=18, d=10):
    """One Fortran ``Gw.d`` field (see tests/test_parsers_plots.py)."""
    if v == 0.0:
        n = 0
    else:
        n = math.floor(math.log10(abs(v))) + 1
    if 0 <= n <= d:
        return f"{v:{w - 4}.{d - n}f}" + "    "
    return f"{v:{w}.{d - 1}E}"


def write_lsj(path, species):
    """src/writelsj.f90, task 15::

        write(50,*)
        write(50,'("Expectation values are computed only over the muffin-tin")')
        do is=1,nspecies
          write(50,*)
          write(50,'("Species : ",I0," (",A,")")') is,trim(spsymb(is))
          do ia=1,natoms(is)
            write(50,'(" atom : ",I0)') ia
            write(50,'("  L : ",3G18.10)') xl(:)
            write(50,'("  S : ",3G18.10)') xs(:)
            write(50,'("  J : ",3G18.10)') xj(:,ias)

    ``species`` is a list of (symbol, [ (L, S) per atom ]).
    """
    with open(path, "w") as fh:
        fh.write("\n")
        fh.write("Expectation values are computed only over the muffin-tin\n")
        for i, (symbol, atoms) in enumerate(species, start=1):
            fh.write("\n")
            fh.write(f"Species : {i} ({symbol})\n")
            for ia, (xl, xs) in enumerate(atoms, start=1):
                xj = [a + b for a, b in zip(xl, xs)]
                fh.write(f" atom : {ia}\n")
                fh.write("  L : " + "".join(g(v) for v in xl) + "\n")
                fh.write("  S : " + "".join(g(v) for v in xs) + "\n")
                fh.write("  J : " + "".join(g(v) for v in xj) + "\n")


def write_lsj_kst(path, groups):
    """src/writelsj.f90, task 16 -- note the ONE-space indent on the value
    lines, unlike task 15's two::

        write(50,*)
        write(50,'("k-point : ",I0,3G18.10)') ik,vkl(:,ik)
        write(50,'("state : ",I0)') ist
        write(50,'("species : ",I0," (",A,"), atom : ",I0)') is,spsymb(is),ia
        write(50,'(" L : ",3G18.10)') xl(:)
        write(50,'(" S : ",3G18.10)') xs(:)
        write(50,'(" J : ",3G18.10)') xj(:,ias)
    """
    with open(path, "w") as fh:
        fh.write("\n")
        fh.write("Expectation values are computed only over the muffin-tin\n")
        for ik, kvec, ist, isp, symbol, ia, xl, xs in groups:
            xj = [a + b for a, b in zip(xl, xs)]
            fh.write("\n")
            fh.write(f"k-point : {ik}" + "".join(g(v) for v in kvec) + "\n")
            fh.write(f"state : {ist}\n")
            fh.write(f"species : {isp} ({symbol}), atom : {ia}\n")
            fh.write(" L : " + "".join(g(v) for v in xl) + "\n")
            fh.write(" S : " + "".join(g(v) for v in xs) + "\n")
            fh.write(" J : " + "".join(g(v) for v in xj) + "\n")


def write_evalsp(path, species):
    """src/writeevsp.f90, task 150::

        write(50,'("Exchange-correlation functional : ",3I6)') xctsp(:)
        write(50,'("Species : ",I4," (",A,")",I4)') is,trim(spsymb(is))
        write(50,'(" n = ",I2,", l = ",I2,", k = ",I2," : ",G18.10)') &
         nsp(ist,is),lsp(ist,is),ksp(ist,is),evalsp(ist,is)

    (the trailing ``I4`` on the Species line has no matching output item, so
    Fortran stops the format there and prints nothing for it.)
    """
    with open(path, "w") as fh:
        fh.write("\n")
        fh.write("Kohn-Sham-Dirac eigenvalues for all atomic species\n")
        fh.write("\n")
        fh.write("Exchange-correlation functional : " + f"{3:6d}{0:6d}{0:6d}" + "\n")
        for i, (symbol, states) in enumerate(species, start=1):
            fh.write("\n")
            fh.write(f"Species : {i:4d} ({symbol})\n")
            for n, l, k, energy in states:
                fh.write(
                    f" n = {n:2d}, l = {l:2d}, k = {k:2d} : " + g(energy) + "\n"
                )


def write_wfcore(path, r, functions):
    """src/wfcrplot.f90, task 65::

        do ist=1,nstsp(is)
          if (spcore(ist,is)) then
            do ir=1,nrsp(is)
              write(50,'(2G18.10)') rsp(ir,is),rwfcr(ir,1,ist,ias)
            end do
            write(50,*)
    """
    with open(path, "w") as fh:
        for u in functions:
            for ri, ui in zip(r, u):
                fh.write(g(ri) + g(ui) + "\n")
            fh.write("\n")


def test_parse_lsj(tmp_path):
    path = tmp_path / "LSJ.OUT"
    write_lsj(
        path,
        [
            ("Fe", [([0.0, 0.0, 0.12], [0.0, 0.0, 1.1]),
                    ([0.0, 0.0, 0.13], [0.0, 0.0, 1.2])]),
            ("O", [([0.0, 0.0, 0.001], [0.0, 0.0, 0.02])]),
        ],
    )
    entries = atomicstates.parse_lsj(path)
    assert len(entries) == 3
    assert [(e["species"], e["symbol"], e["atom"]) for e in entries] == [
        (1, "Fe", 1), (1, "Fe", 2), (2, "O", 1)
    ]
    np.testing.assert_allclose(entries[0]["L"], [0.0, 0.0, 0.12])
    np.testing.assert_allclose(entries[0]["S"], [0.0, 0.0, 1.1])
    np.testing.assert_allclose(entries[0]["J"], [0.0, 0.0, 1.22])
    np.testing.assert_allclose(entries[2]["S"], [0.0, 0.0, 0.02])


def test_parse_lsj_does_not_mistake_the_species_header_for_a_spin_vector(tmp_path):
    """"Species : 1 (Fe)" and "  S : ..." both start with S once stripped;
    only the longer prefix distinguishes them."""
    path = tmp_path / "LSJ.OUT"
    write_lsj(path, [("Si", [([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])])])
    entries = atomicstates.parse_lsj(path)
    assert len(entries) == 1
    np.testing.assert_allclose(entries[0]["S"], [0.4, 0.5, 0.6])
    assert entries[0]["symbol"] == "Si"


def test_parse_lsj_kst(tmp_path):
    path = tmp_path / "LSJ_KST.OUT"
    write_lsj_kst(
        path,
        [
            (3, [0.25, 0.25, 0.0], 7, 1, "W", 1, [0.0, 0.0, -1.9], [0.0, 0.0, -0.48]),
            (3, [0.25, 0.25, 0.0], 7, 2, "Se", 2, [0.0, 0.0, 0.01], [0.0, 0.0, 0.02]),
        ],
    )
    entries = atomicstates.parse_lsj_kst(path)
    assert len(entries) == 2
    assert entries[0]["ik"] == 3
    assert entries[0]["ist"] == 7
    assert entries[0]["symbol"] == "W"
    assert entries[1]["atom"] == 2
    np.testing.assert_allclose(entries[0]["k"], [0.25, 0.25, 0.0])
    np.testing.assert_allclose(entries[0]["L"], [0.0, 0.0, -1.9])
    np.testing.assert_allclose(entries[0]["J"], [0.0, 0.0, -2.38])


def test_parse_evalsp(tmp_path):
    path = tmp_path / "EVALSP.OUT"
    write_evalsp(
        path,
        [
            ("Si", [(1, 0, 1, -65.184), (2, 0, 1, -5.075),
                    (2, 1, 1, -3.5162), (2, 1, 2, -3.5010)]),
            ("O", [(1, 0, 1, -18.758)]),
        ],
    )
    species = atomicstates.parse_evalsp(path)
    assert len(species) == 2
    assert species[0]["symbol"] == "Si"
    assert len(species[0]["states"]) == 4
    assert species[0]["states"][0] == {"n": 1, "l": 0, "k": 1, "energy": -65.184}
    # the l=1 shell appears twice: Elk's ksp is |kappa| = j + 1/2, so k = l
    # is j = 1/2 and k = l + 1 is j = 3/2, and the gap between them is the
    # free-atom spin-orbit splitting
    p12, p32 = species[0]["states"][2], species[0]["states"][3]
    assert (p12["l"], p12["k"]) == (1, 1)
    assert (p32["l"], p32["k"]) == (1, 2)
    assert p32["energy"] - p12["energy"] == pytest.approx(0.0152, abs=1e-6)
    assert species[1]["states"][0]["energy"] == pytest.approx(-18.758)


def test_parse_wfcore(tmp_path):
    path = tmp_path / "WFCORE_S01_A0001.OUT"
    r = np.linspace(1e-5, 2.0, 40)
    functions = [np.exp(-r), r * np.exp(-2 * r)]
    write_wfcore(path, r, functions)
    parsed_r, u = atomicstates.parse_wfcore(path)
    assert parsed_r.shape == (40,)
    assert u.shape == (2, 40)
    np.testing.assert_allclose(parsed_r, r, rtol=1e-8)
    np.testing.assert_allclose(u[1], functions[1], rtol=1e-8)


def test_parse_two_column(tmp_path):
    """SDELTA.OUT / STHETA.OUT (task 14) and ELNES.OUT (task 140)."""
    path = tmp_path / "SDELTA.OUT"
    with open(path, "w") as fh:
        for i in range(11):
            w = -0.01 + 0.002 * i
            fh.write(g(w) + g(math.exp(-((w / 0.005) ** 2))) + "\n")
    x, y = atomicstates.parse_two_column(path)
    assert x.shape == (11,)
    assert y.shape == (11,)
    assert y.argmax() == 5
