r"""Phase 1 item 1l: the position derivative, and the sum rule that pins it.

`docs/jax_port.md` states Phase 1's gradient criterion as
$d\varepsilon_j/d\mathbf R$ on displaced h-BN.  Half of what that needs now
exists: §1k built the radial integrals from the muffin-tin potential, so they
are no longer imported constants.  The other half does not: moving an atom
moves the potential *inside* its sphere, and where that potential comes from is
Phase 2.

What is available, and what this module computes, is the derivative at
**frozen potential** -- the rigid-muffin-tin picture, in which the muffin-tin
potential rides with its sphere unchanged and the interstitial contribution is
held fixed.  Atomic positions then enter the eigenproblem in exactly one place,
the structure factor of the matching coefficients:

.. math::
   A^{\alpha}_{\mathbf G+\mathbf k,\,\ell m,\,i_o}
     \;\propto\; e^{\,i(\mathbf G+\mathbf k)\cdot\mathbf r_\alpha}\,
       \bigl(D^{\alpha\ell}\bigr)^{-1}\!\cdots

**This is not a force**, and no amount of finite-difference agreement would
make it one: `forcek.f90`'s incomplete-basis-set term is the counterpart of the
piece computed here, but Elk's total force also carries the Hellmann-Feynman
and core terms, and the interstitial characteristic function moves with the
sphere too.  Stating it as a force is exactly the overclaim
`docs/jax_port.md` §Phase 4 warns about.

**The oracle is a sum rule, not a finite difference.**  Translating *every*
atom by the same $\boldsymbol\delta$ cannot change the spectrum.  Under such a
translation every basis function picks up a phase and the whole matrix
transforms by a diagonal unitary,

.. math::
   M(\boldsymbol\delta) = U^\dagger M(0)\,U,\qquad
   U = \mathrm{diag}\bigl(e^{\,i(\mathbf G_i+\mathbf k)\cdot\boldsymbol\delta}
       \bigr)\ \text{on the APW rows},\ 1\ \text{on the local orbitals},

so the eigenvalues are invariant and the derivative is exactly zero.  The
muffin-tin blocks achieve that on their own through the structure factor.  The
interstitial blocks do not, because they are imported: their true response is
the same phase (the characteristic function obeys
$\tilde\Theta(\mathbf G)\to\tilde\Theta(\mathbf G)e^{-i\mathbf G\cdot
\boldsymbol\delta}$ under $\Theta(\mathbf r)\to\Theta(\mathbf r-
\boldsymbol\delta)$), which is closed-form and is applied here.  That closes
the loop: a wrong sign or a wrong factor in `match`'s position dependence
breaks the invariance, and nothing else in it can.

The contrast that stops the null being vacuous is a SINGLE-atom displacement on
the same fixture, where the derivative is O(1) -- and for that one there is no
oracle beyond central differences of the same function, which is what it is
reported as.

Run as ``python3 -m elkjax.phase1_position``.
"""

import sys

import numpy as np

import jax
import jax.numpy as jnp

from . import hamiltonian as ham
from .memory import limit_address_space

limit_address_space()


def _reciprocal(export):
    ngp = int(export["ngp"])
    return (np.asarray(export["vgpc"])[:, :ngp]
            - np.asarray(export["vkc"])[:, None]).T          # (ngp, 3)


def assemble_at_positions(export, atposc, translation=None,
                          interstitial="built"):
    """(H, O) with the atoms at `atposc`, at the exported k-point.

    ``interstitial="built"`` takes the characteristic function from
    `hamiltonian.characteristic_function_matrix` -- closed-form geometry, so
    it follows the atoms -- leaving only the interstitial Kohn-Sham potential
    imported.  ``"frozen"`` takes both blocks from the export, which is what
    this module did before `gencfun` was transcribed and is kept as the
    contrast.

    `translation`, when given, applies the exact response of whatever is still
    imported to a rigid shift: the same diagonal phase, since
    :math:`f(\\mathbf r)\\to f(\\mathbf r-\\boldsymbol\\delta)` gives
    :math:`\\tilde f(\\mathbf G)\\to\\tilde f(\\mathbf G)
    e^{-i\\mathbf G\\cdot\\boldsymbol\\delta}` for the potential exactly as for
    :math:`\\Theta`.  With ``interstitial="built"`` that is one imported
    quantity rather than two, and :math:`\\Theta`'s own response is then
    DERIVED rather than imposed.
    """
    ngp = int(export["ngp"])
    nmatp = int(export["nmatp"])
    vgkc = jnp.asarray(np.asarray(export["vgpc"])[:, :ngp].T)
    local = dict(export)
    local["apwalm"] = ham.matching_coefficients(export, vgkc, atposc=atposc)
    kinetic = 0.5 * (vgkc @ vgkc.T)
    vsig = jnp.asarray(ham.interstitial_potential_matrix(export))
    if translation is None:
        conj = 1.0
    else:
        gvec = jnp.asarray(_reciprocal(export))
        phase = jnp.exp(-1j * (gvec @ jnp.asarray(translation)))
        conj = phase[:, None] * jnp.conj(phase)[None, :]
    if interstitial == "built":
        o_istl = ham.characteristic_function_matrix(export, atposc=atposc)
    elif interstitial == "frozen":
        o_istl = jnp.asarray(export["omat_istl"]) * conj
    else:
        raise ValueError(f"unknown interstitial mode {interstitial!r}")
    h_istl = vsig * conj + kinetic * o_istl

    def _pad(block):
        return jnp.zeros((nmatp, nmatp), dtype=complex).at[:ngp, :ngp].add(
            block)

    return (ham.muffin_tin_hamiltonian(local) + _pad(h_istl),
            ham.muffin_tin_overlap(local) + _pad(o_istl))


def spectrum_at_positions(export, atposc, translation=None, nocc=None,
                          interstitial="built"):
    h, o = assemble_at_positions(export, atposc, translation=translation,
                                 interstitial=interstitial)
    reduced, _ = ham.cholesky_reduce(h, o)
    evals = jnp.linalg.eigvalsh(reduced)
    return evals if nocc is None else evals[:nocc]


def band_energy_at_positions(export, atposc, nocc, translation=None,
                             interstitial="built"):
    """The occupied-window trace -- a sum, because an individual branch of a
    degenerate group is not differentiable and its trace is."""
    return jnp.sum(spectrum_at_positions(
        export, atposc, translation=translation, nocc=nocc,
        interstitial=interstitial))


def translation_null(export, nocc, delta=(0.031, -0.017, 0.023)):
    """The sum rule: rigid translation cannot move the spectrum.

    Four rows, and the point is the contrast between them.  With the
    characteristic function BUILT, the only imported quantity left is the
    interstitial Kohn-Sham potential, so ``built + phase`` is the null with
    just one exact response supplied by hand; ``built`` alone measures what
    that potential contributes; and the two ``frozen`` rows are what the same
    test said before `gencfun` was transcribed.
    """
    atposc = np.asarray(export["atposc"])
    shift = np.asarray(delta, dtype=float)
    moved = atposc + shift[:, None]
    base = np.asarray(spectrum_at_positions(export, atposc, nocc=nocc))
    out = {"delta": shift.tolist(), "reference": base}
    modes = (("built+phase", "built", True), ("built", "built", False),
             ("frozen+phase", "frozen", True), ("frozen", "frozen", False))
    for tag, mode, corrected in modes:
        translation = shift if corrected else None
        shifted = np.asarray(spectrum_at_positions(
            export, moved, translation=translation, nocc=nocc,
            interstitial=mode))
        out[f"forward_{tag}"] = float(np.abs(shifted - base).max())
        fn = lambda d, mode=mode, corrected=corrected: (
            band_energy_at_positions(
                export, atposc + d[:, None], nocc,
                translation=d if corrected else None, interstitial=mode))
        out[f"grad_{tag}"] = np.asarray(jax.grad(fn)(jnp.zeros(3)))
    return out


def characteristic_function_covariance(export, delta=(0.031, -0.017, 0.023)):
    r"""The same sum rule on :math:`\tilde\Theta` alone, with no eigensolve.

    Translating every atom by :math:`\boldsymbol\delta` must multiply
    :math:`\tilde\Theta(\mathbf G_i-\mathbf G_j)` by
    :math:`e^{-i(\mathbf G_i-\mathbf G_j)\cdot\boldsymbol\delta}` -- an
    identity that pins the sign of the exponent against the $(i,j)$ ordering
    of the difference vectors, which is the one convention in
    `characteristic_function_matrix` that an agreement with Elk's own
    `omat_istl` at the ORIGINAL positions cannot check.
    """
    atposc = np.asarray(export["atposc"])
    shift = np.asarray(delta, dtype=float)
    gvec = np.asarray(_reciprocal(export))
    phase = np.exp(-1j * (gvec @ shift))
    base = np.asarray(ham.characteristic_function_matrix(export))
    moved = np.asarray(
        ham.characteristic_function_matrix(export, atposc + shift[:, None]))
    predicted = base * (phase[:, None] * np.conj(phase)[None, :])
    return float(np.abs(moved - predicted).max()), float(np.abs(base).max())


def single_atom(export, nocc, ias=1, seed=5, steps=(1e-3, 1e-4, 1e-5)):
    """One atom displaced along a general direction, AD against central FD of
    the same function.  No external oracle -- see the module docstring."""
    atposc = np.asarray(export["atposc"])
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)

    def fn(t):
        moved = jnp.asarray(atposc).at[:, ias].add(t * jnp.asarray(direction))
        return band_energy_at_positions(export, moved, nocc)

    ad = float(jax.jvp(fn, (0.0,), (1.0,))[1])
    fd = [float((fn(h) - fn(-h)) / (2.0 * h)) for h in steps]
    return {"atom": ias, "direction": direction, "ad": ad, "fd": fd,
            "steps": list(steps)}


def report(null, single):
    lines = [f"rigid translation delta = {null['delta']}"]
    for tag in ("built+phase", "built", "frozen+phase", "frozen"):
        lines.append(f"  {tag:13s} forward {null[f'forward_{tag}']:.3e} Ha"
                     f"   gradient "
                     f"{np.array2string(null[f'grad_{tag}'], precision=3)}")
    lines += ["",
              f"single atom {single['atom']} along "
              f"{np.array2string(single['direction'], precision=4)}",
              f"  AD {single['ad']: .12e}"]
    for h, fd in zip(single["steps"], single["fd"]):
        lines.append(f"  FD h={h:.0e}  {fd: .12e}   rel "
                     f"{abs(fd - single['ad']) / abs(single['ad']):.3e}")
    return "\n".join(lines)


SI = (
    [(5.13, 5.13, 0.00), (5.13, 0.00, 5.13), (0.00, 5.13, 5.13)],
    {"Si": [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]},
    (4, 4, 4),
)
_A = 4.7443                      # monolayer h-BN, a = 2.511 Angstrom
HBN = (
    [(_A, 0.0, 0.0), (-_A / 2, _A * 3 ** 0.5 / 2, 0.0), (0.0, 0.0, 20.0)],
    {"B": [(0.0, 0.0, 0.0)], "N": [(1 / 3, 2 / 3, 0.0)]},
    (6, 6, 1),
)
CASES = [("Si generic", SI, (0.1, 0.2, 0.05)),
         ("hBN generic", HBN, (0.1, 0.2, 0.0))]


def main(workdir="phase1_position"):
    from elkpy.structure import Structure

    for label, (avec, species, ngridk), k in CASES:
        calculation = Structure(avec, species).get_calculation(
            f"{workdir}/{label.split()[0]}", xc="PW", ngridk=ngridk,
            rgkmax=7.0)
        calculation.ensure_ground_state()
        efermi = float(
            (calculation.workdir / "EFERMI.OUT").read_text().split()[0])
        with calculation.eigenstate_session() as session:
            export = session.lapw_problem(k)
        nocc = ham.occupied_band_count(export["evalfv"], efermi)
        print(f"\n{'=' * 78}\n{label}   k={k}   nocc={nocc}\n{'=' * 78}",
              flush=True)
        print(report(translation_null(export, nocc),
                     single_atom(export, nocc)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "phase1_position")
