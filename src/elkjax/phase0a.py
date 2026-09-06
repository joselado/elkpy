r"""Phase 0a: does reverse-mode implicit differentiation survive the eigensolve?

Study §6 item 0a.  Run with::

    PYTHONPATH=src taskset -c 0-3 python3 -m elkjax.phase0a

The kill criterion the study states is "agreement with central FD on the toy".  That is
weaker than it needs to be — §8(b)'s own measurements show FD failing at exactly the
degeneracies this item engineers — so every comparison here is against the **dense
implicit-function-theorem solve** of :meth:`elkjax.scftoy.ScfToy.reference_gradient`,
which uses the closed-form projector derivative, no autodiff, a different linear solver
(LU, not GMRES) and a different matrix dimension (the undoubled block).  Central FD is
reported alongside as a third opinion, not as the reference.

The load-bearing structural fact is that the adjoint matvec
:math:`u\mapsto u-(\partial F/\partial v)^{\mathsf T}u` is one full Kohn-Sham JVP,
transposed — so ``eigh``'s derivative is evaluated at every multiplet on every GMRES
iteration.  Experiment D is the control that shows this matters: the identical machinery
with the naive projector instead of the safe-:math:`K` rule.
"""

import numpy as np

import jax
import jax.numpy as jnp

from . import fixedpoint, memory, scftoy

COUPLING = 2.0     # gives rho(K chi0) ~ 0.84: the self-consistency is doing real work
SIZE, NOCC = 8, 3


def _solver(toy, mixing=0.4, tol=1e-13, history=0, gmres_kwargs=None):
    return fixedpoint.implicit_fixed_point(
        toy.step,
        solver_kwargs=dict(mixing=mixing, tol=tol, maxiter=4000, history=history),
        gmres_kwargs=gmres_kwargs)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def gradient(toy, theta=0.0, quantity="loss", **kwargs):
    """Reverse-mode :math:`dL/d\\theta` through the fixed point, plus diagnostics."""
    solve = _solver(toy, **kwargs)
    v0 = jnp.zeros(toy.size)
    observable = toy.loss if quantity == "loss" else toy.band_energy
    value = lambda th: observable(solve(th, th * 0.0 + v0), th)
    grad = float(jax.grad(value)(theta))
    v = solve(theta, v0)
    residual = float(jnp.linalg.norm(toy.step(v, theta) - v))
    cotangent = jax.grad(lambda vv: observable(vv, theta))(v)
    _, adjoint = fixedpoint.adjoint_residual(toy.step, theta, v, cotangent)
    frozen = float(jax.grad(lambda th: observable(jax.lax.stop_gradient(v), th))(theta))
    return dict(grad=grad, v=np.asarray(v), scf_residual=residual,
                adjoint_residual=adjoint, frozen=frozen,
                reference=toy.reference_gradient(np.asarray(v), theta, quantity),
                radius=toy.spectral_radius(np.asarray(v), theta))


def unrolled_gradient(toy, theta=0.0, quantity="loss", mixing=0.4, iterations=200,
                      history=0, ridge=1e-10):
    r"""Differentiate through a FIXED number of mixer steps instead of the fixed point.

    Study §8(a) corrects the received view here: unrolling is not particularly
    *inaccurate*.  The real objections are tape size (one iteration's residuals include
    every k-point's eigenvectors — 1.44 GB at production shapes, times the iteration
    count), mixer dependence, and that ``lax.while_loop`` has no reverse rule at all, so
    an unrolled reverse-mode SCF needs a fixed trip count in the first place — which is
    exactly what this function has to hard-code.
    """
    observable = toy.loss if quantity == "loss" else toy.band_energy

    def value(th):
        v = jnp.zeros(toy.size)
        iterates, residuals = [], []
        for _ in range(iterations):
            residual = toy.step(v, th) - v
            iterates.append(v)
            residuals.append(residual)
            keep = max(history, 1)
            if len(iterates) > keep:
                iterates, residuals = iterates[-keep:], residuals[-keep:]
            if history and len(iterates) > 1:
                v = fixedpoint._anderson(iterates, residuals, mixing, ridge)
            else:
                v = v + mixing * residual
        return observable(v, th)

    return float(jax.grad(value)(theta)), float(value(theta))


def central_difference(toy, theta=0.0, step=1e-4, quantity="loss", **kwargs):
    """FD of the *converged* fixed point — a third opinion, not the reference."""
    solve = _solver(toy, **kwargs)
    v0 = jnp.zeros(toy.size)
    observable = toy.loss if quantity == "loss" else toy.band_energy
    plus = float(observable(solve(theta + step, v0), theta + step))
    minus = float(observable(solve(theta - step, v0), theta - step))
    return (plus - minus) / (2 * step)


# --------------------------------------------------------------------------- A-D


def experiment_degeneracies(rule="safe", quantity="loss"):
    """The three spectra: none, roundoff-split by doubling, and bitwise degenerate."""
    cases = [
        ("no degeneracy", dict(degeneracy=1)),
        ("doubled, rotated", dict(degeneracy=2, rotate=True)),
        ("doubled, bitwise", dict(degeneracy=2, rotate=False)),
        ("doubled, symmetry-broken", dict(degeneracy=2, rotate=True,
                                          break_symmetry=True)),
    ]
    rows = []
    for label, options in cases:
        toy = scftoy.build(size=SIZE, nocc=NOCC, seed=0, coupling=COUPLING,
                           rule=rule, **options)
        result = gradient(toy, quantity=quantity)
        result["label"] = label
        result["splitting"] = _min_splitting(toy, result["v"])
        result["fd"] = central_difference(toy, quantity=quantity)
        rows.append(result)
    return rows


def _min_splitting(toy, v, theta=0.0):
    evals = np.asarray(jnp.linalg.eigvalsh(toy.hamiltonian(jnp.asarray(v), theta)))
    return float(np.min(np.diff(evals)))


# --------------------------------------------------------------------------- E


def experiment_mixer_independence(quantity="loss"):
    r"""Two mixers reach different points inside the same tolerance ball.

    Study §8(a): :math:`F` is defined without the mixer, the converged answer does not
    depend on it, and neither may the gradient.  This needs no reference value at all,
    which makes it the sharpest test here.
    """
    toy = scftoy.build(size=SIZE, nocc=NOCC, seed=0, coupling=COUPLING,
                       degeneracy=2, rotate=True, break_symmetry=True)
    linear = gradient(toy, quantity=quantity, mixing=0.4, history=0)
    anderson = gradient(toy, quantity=quantity, mixing=0.4, history=5)
    return dict(linear=linear, anderson=anderson,
                v_difference=float(np.linalg.norm(linear["v"] - anderson["v"])),
                grad_difference=_rel(linear["grad"], anderson["grad"]))


# --------------------------------------------------------------------------- F


def experiment_tolerance(quantity="loss", tolerances=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12)):
    """Implicit gradients should plateau in the SCF tolerance well before the value does."""
    toy = scftoy.build(size=SIZE, nocc=NOCC, seed=0, coupling=COUPLING,
                       degeneracy=2, rotate=True, break_symmetry=True)
    converged = gradient(toy, quantity=quantity, tol=1e-13)["reference"]
    rows = []
    for tol in tolerances:
        result = gradient(toy, quantity=quantity, tol=tol)
        rows.append(dict(tol=tol, grad=result["grad"], reference=result["reference"],
                         converged=converged, scf_residual=result["scf_residual"]))
    return rows


def experiment_unrolled(quantity="loss", counts=(20, 50, 100, 200, 400)):
    """Implicit versus unrolled at matched forward accuracy, and the tape argument.

    The unrolled gradient here differentiates a hard-coded trip count of an 8x8 problem;
    at production shapes the same tape carries every k-point's eigenvectors at every
    iteration (study §8a: ~1.44 GB per iteration at (100, 3000, 300)).
    """
    toy = scftoy.build(size=SIZE, nocc=NOCC, seed=0, coupling=COUPLING,
                       degeneracy=2, rotate=True, break_symmetry=True)
    exact = gradient(toy, quantity=quantity, tol=1e-13)
    rows = []
    for count in counts:
        try:
            grad, value = unrolled_gradient(toy, quantity=quantity, iterations=count)
        except Exception as exc:            # noqa: BLE001
            grad, value = float("nan"), f"{type(exc).__name__}"
        rows.append(dict(iterations=count, unrolled=grad, value=value))
    return dict(implicit=exact["grad"], reference=exact["reference"], rows=rows)


# --------------------------------------------------------------------------- G


def experiment_smearing(mu=0.0, width=0.3, quantity="loss"):
    """The metallic case, at fixed chemical potential."""
    toy = scftoy.build(size=SIZE, nocc=NOCC, seed=0, coupling=COUPLING,
                       degeneracy=2, rotate=True, smearing=(mu, width),
                       break_symmetry=True)
    return gradient(toy, quantity=quantity)


# --------------------------------------------------------------------------- H


def experiment_second_order(quantity="loss"):
    r"""Phase 0a′: :math:`d^2L/d\theta^2` through the same fixed point.

    Study §6: this is the item that decides the full port over §9.2's hybrid.  Three
    routes are tried on each spectrum because they fail differently — ``jax.hessian`` is
    ``jacfwd(jacrev)`` and a ``custom_vjp`` cannot be forward-differentiated at all, so
    a ``TypeError`` there is a JAX limitation and says nothing about the physics, while
    reverse-over-reverse exercises the rule's own second derivative.
    """
    cases = [("no degeneracy", dict(degeneracy=1)),
             ("doubled, rotated", dict(degeneracy=2, rotate=True)),
             ("doubled, symmetry-broken", dict(degeneracy=2, rotate=True,
                                               break_symmetry=True)),
             ("symmetry-broken, sign rule", dict(degeneracy=2, rotate=True,
                                                 break_symmetry=True, rule="sign"))]
    routes = (("grad(grad)", lambda f: jax.grad(jax.grad(f))),
              ("jacrev(jacrev)", lambda f: jax.jacrev(jax.jacrev(f))),
              ("jax.hessian", jax.hessian),
              ("jacfwd(grad)", lambda f: jax.jacfwd(jax.grad(f))))
    rows = []
    for label, options in cases:
        toy = scftoy.build(size=SIZE, nocc=NOCC, seed=0, coupling=COUPLING, **options)
        solve = _solver(toy)
        v0 = jnp.zeros(toy.size)
        observable = toy.loss if quantity == "loss" else toy.band_energy
        value = lambda th: observable(solve(th, v0), th)
        out = {"label": label}
        for name, transform in routes:
            try:
                out[name] = float(transform(value)(0.0))
            except Exception as exc:        # noqa: BLE001 - the failure IS the result
                out[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}"
        first = lambda th: float(jax.grad(value)(th))
        for step in (1e-3, 1e-5):
            out[f"central FD h={step:.0e}"] = (first(step) - first(-step)) / (2 * step)
        rows.append(out)
    return rows


def experiment_unrolled_mixers(quantity="loss",
                               counts=(20, 40, 60, 80, 120, 160, 240, 320)):
    r"""Mixer dependence with teeth: **unrolled** linear versus **unrolled** Anderson.

    The implicit gradient cannot depend on the mixer for a structural reason — the
    ``custom_vjp`` backward pass sees only :math:`(	heta, v^*)` — so agreeing there
    pins :math:`v^*`, not the machinery.  The test that carries information is study
    §8(a)'s own: two *unrolled* mixers at matched forward accuracy reach different
    gradients (measured there, :math:`1.9215	imes10^3` vs :math:`1.9252	imes10^3`
    for a quantity whose true value is :math:`1.9281	imes10^3`), while the implicit
    route returns one number for both.
    """
    toy = scftoy.build(size=SIZE, nocc=NOCC, seed=0, coupling=COUPLING,
                       degeneracy=2, rotate=True, break_symmetry=True)
    exact = gradient(toy, quantity=quantity, tol=1e-13)
    converged_value = float((toy.loss if quantity == "loss" else toy.band_energy)(
        jnp.asarray(exact["v"]), 0.0))
    rows = []
    for history in (0, 5):
        for count in counts:
            grad, value = unrolled_gradient(toy, quantity=quantity, iterations=count,
                                            history=history)
            rows.append(dict(history=history, iterations=count, grad=grad,
                             value_error=_rel(value, converged_value),
                             grad_error=_rel(grad, exact["reference"])))
    return dict(implicit=exact["grad"], reference=exact["reference"], rows=rows)


# --------------------------------------------------------------------------- main


def main():
    memory.limit_address_space(16.0)
    for quantity in ("loss", "band_energy"):
        print(f"== A-C. implicit gradient of `{quantity}` vs the dense reference ==")
        print(f"  {'spectrum':>25} {'min split':>11} {'AD':>15} {'reference':>15} "
              f"{'rel':>9} {'FD':>15} {'frozen-v':>15} {'adj resid':>10}")
        for r in experiment_degeneracies(quantity=quantity):
            print(f"  {r['label']:>25} {r['splitting']:>11.2e} {r['grad']:>15.10f} "
                  f"{r['reference']:>15.10f} {_rel(r['grad'], r['reference']):>9.1e} "
                  f"{r['fd']:>15.10f} {r['frozen']:>15.10f} {r['adjoint_residual']:>10.1e}")
        print()

    print("== D. the same machinery with the NAIVE projector (control) ==")
    print(f"  {'spectrum':>25} {'AD':>15} {'reference':>15} {'rel':>9} {'adj resid':>10}")
    for r in experiment_degeneracies(rule="naive"):
        print(f"  {r['label']:>25} {r['grad']:>15.10f} {r['reference']:>15.10f} "
              f"{_rel(r['grad'], r['reference']):>9.1e} {r['adjoint_residual']:>10.1e}")

    print("\n== E. mixer independence (no reference value needed) ==")
    e = experiment_mixer_independence()
    print(f"  linear   v* {np.linalg.norm(e['linear']['v']):.12f}  grad {e['linear']['grad']:.12f}")
    print(f"  Anderson v* {np.linalg.norm(e['anderson']['v']):.12f}  grad {e['anderson']['grad']:.12f}")
    print(f"  |v_lin - v_and| = {e['v_difference']:.2e}   relative gradient difference "
          f"{e['grad_difference']:.2e}")

    print("\n== F. gradient vs the SCF tolerance ==")
    for r in experiment_tolerance():
        print(f"  tol {r['tol']:.0e}  residual {r['scf_residual']:.2e}  grad {r['grad']:.12f}  "
              f"rel to the IFT solve at the SAME v* {_rel(r['grad'], r['reference']):.1e}  "
              f"rel to the CONVERGED gradient {_rel(r['grad'], r['converged']):.1e}")

    print("\n== F2. implicit versus unrolled ==")
    u = experiment_unrolled()
    print(f"  implicit {u['implicit']:.12f} (dense IFT reference {u['reference']:.12f})")
    for r in u["rows"]:
        shown = f"{r['value']:.12f}" if isinstance(r["value"], float) else r["value"]
        print(f"    {r['iterations']:>4} unrolled steps: grad {r['unrolled']:.12f}  "
              f"rel {_rel(r['unrolled'], u['reference']):.2e}   L = {shown}")

    print("\n== G. Fermi-Dirac smearing at fixed mu ==")
    g = experiment_smearing()
    print(f"  radius {g['radius']:.4f}  SCF residual {g['scf_residual']:.1e}  "
          f"AD {g['grad']:.12f}  reference {g['reference']:.12f}  "
          f"rel {_rel(g['grad'], g['reference']):.2e}  adjoint residual {g['adjoint_residual']:.1e}")

    print("\n== E2. mixer dependence of the UNROLLED gradient (study §8a's own test) ==")
    u = experiment_unrolled_mixers()
    print(f"  implicit {u['implicit']:.12f} for either mixer; dense reference {u['reference']:.12f}")
    print(f"  {'mixer':>10} {'steps':>6} {'|value error|':>14} {'|grad error|':>13}")
    for r in u["rows"]:
        name = "Anderson" if r["history"] else "linear"
        print(f"  {name:>10} {r['iterations']:>6} {r['value_error']:>14.2e} "
              f"{r['grad_error']:>13.2e}")

    print("\n== H. second order through the fixed point (Phase 0a-prime) ==")
    for row in experiment_second_order():
        print(f"  {row['label']}:")
        for name, value in row.items():
            if name == "label":
                continue
            shown = f"{value:.12f}" if isinstance(value, float) else value
            print(f"    {name:>16}: {shown}")

    print(f"\npeak RSS {memory.peak_rss_bytes() / memory.GB:.2f} GB")


if __name__ == "__main__":
    main()
