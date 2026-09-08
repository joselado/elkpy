r"""Phase 2 item 2b: the interstitial valence density from the eigenvectors.

This is the step that closes the SCF loop's circle.  §§2a-2g go from a density
to a total energy; this goes the other way, from the first-variational
eigenvectors back to a density.  What is here is the **interstitial** half, and
the muffin-tin half is not (see "What this does not do" below).

**The formula.**  For a non-magnetic ground state ``rhomagk``'s interstitial
accumulation is

.. math::

    \rho^{\rm I}({\bf r})=\frac1\Omega\sum_{\bf k}w_{\bf k}\sum_n f_{n\bf k}
    \,\bigl|\psi_{n\bf k}({\bf r})\bigr|^2 ,

with :math:`\psi` built by scattering ``evecfv(igp, n)`` into the **coarse** FFT
grid at ``igfc(igkig(igp))`` and transforming.  In the interstitial an LAPW
state is an exact plane-wave sum, so this is not an approximation: the only
truncation is the basis's own.

**Three things about it are easy to get wrong and are pinned by tests.**

* The density is accumulated on the **coarse** grid (``ngdgc``, ``ngtc``), not
  the fine one the ``GROUNDSTATE`` query's ``rhoir`` lives on.  ``rfirctof``
  interpolates afterwards by zero-padding, so the coarse function is recoverable
  from the fine one *exactly* -- which is what :func:`coarsen` does, and why the
  comparison needs no interpolation of its own.
* ``rhonorm`` then adds a **uniform constant** :math:`(N-N_{\rm calc})/\Omega`,
  not a rescaling.  So the difference from Elk is a constant, and asserting
  *that* is sharper than any tolerance -- a constant is a one-parameter family
  and a pointwise error breaks it.
* The zone sum needs the k-set Elk actually used, weights included.  The
  ``DENSITYK`` query supplies it; rebuilding it from ``ngridk`` would differ
  whenever ``reducek`` is nonzero, which is the default.

**What this does not do.**  The muffin-tin half needs ``wfmtsv`` on the coarse
*radial* mesh, then ``rhomagsh``, ``rfmtctof`` and ``rhocore`` -- which brings in
the core states and so ``gencore``/``rdirac``.  None of that is here.  Nor is the
magnetic case: ``rmk1``/``rmk2`` are not transcribed, only ``rmk3``.
"""

import numpy as np

import jax
import jax.numpy as jnp

__all__ = ["skip_below_epsocc", "interstitial_wavefunction",
           "interstitial_density", "coarsen", "normalisation_offset",
           "coarse_radial_indices", "muffin_tin_wavefunctions",
           "muffin_tin_density", "pack_coarse", "pack_fine", "to_harmonics",
           "coarse_to_fine", "state_factors", "zone_matrices", "solve_zone",
           "density_from_potential", "add_core", "normalise", "refine",
           "converged_density", "symmetrise_interstitial"]

EPSOCC = 1.0e-8


def skip_below_epsocc(occupations):
    r"""``rhomagk``'s ``epsocc`` skip, written as a weight rather than a branch.

    Elk leaves a state out of the density sum when :math:`|f|<` ``epsocc``.
    Multiplying that state by an exact zero is the same arithmetic -- it
    contributes nothing either way -- but it is a *value* rather than Python
    control flow, so the accumulation stays a function of the occupations and
    can be traced.  That is what every Phase 3 gradient needs; as a ``continue``
    the density could only ever be evaluated on concrete arrays.

    The mask is treated as a constant of the derivative, which is what the
    skip itself means: a state Elk drops contributes nothing to
    :math:`d\rho` either.
    """
    occupations = jnp.asarray(occupations)
    return jnp.where(jnp.abs(occupations) < EPSOCC, 0.0, occupations)


def _inverse_fft(values, grid):
    """Elk's ``zfftifc(3, n, +1, z)``: sign +1 and no normalisation."""
    return jnp.fft.ifftn(jnp.asarray(values).reshape(grid, order="F")
                         ).reshape(-1, order="F") * np.prod(grid)


def _forward_fft(values, grid):
    """Elk's ``zfftifc(3, n, -1, z)``: sign -1 with a 1/N."""
    return jnp.fft.fftn(jnp.asarray(values).reshape(grid, order="F")
                        ).reshape(-1, order="F") / np.prod(grid)


def interstitial_wavefunction(coefficients, igkig, densityk):
    r""":math:`\psi_n({\bf r})` on the coarse interstitial grid.

    ``coefficients`` is ``evecfv(1:ngk, n)`` for one state, or a ``(nst, nmat)``
    stack of them -- the leading axis is kept, so a whole band set costs one
    call.  ``igkig`` maps each :math:`{\bf G+k}` to the global :math:`{\bf G}`
    index, and ``igfc`` maps that to a slot of the coarse FFT array.  The
    two-step indirection is the part a transcription gets wrong, and it is
    Elk's own: ``igkig`` is a property of the k-point, ``igfc`` of the grid.
    """
    grid = tuple(int(n) for n in densityk["ngdgc"])
    igfc = np.asarray(densityk["igfc"])
    igkig = np.asarray(igkig)
    slots = jnp.asarray(igfc[igkig - 1] - 1)
    coefficients = jnp.asarray(coefficients)
    flat = coefficients.ndim == 1
    coefficients = coefficients[None, :] if flat else coefficients
    array = jnp.zeros((coefficients.shape[0], int(densityk["ngtc"])),
                      dtype=complex)
    # only the first `ngk` coefficients: the rest are local orbitals, which
    # live entirely inside the muffin tins and contribute nothing here.  The
    # muffin-tin half uses all `nmat` of them.
    array = array.at[:, slots].set(coefficients[:, :igkig.size])
    out = jax.vmap(lambda row: _inverse_fft(row, grid))(array)
    return out[0] if flat else out


def state_factors(densityk, ispn=0):
    r"""Elk's own occupations as the two factors the density is bilinear in.

    Returns ``{(ik, ispn): (bra, ket)}`` with ``bra`` the eigenvectors and
    ``ket = f bra``, so that
    :math:`\rho=\sum_a\mathrm{Re}[\overline{\psi(\bar c_a)}\,\psi(x_a)]`
    is :math:`\sum_af_a|\psi_a|^2`.  Splitting the occupation off like this is
    what lets `elkjax.response.density_factors` hand the same accumulation a
    pair whose *derivative* carries the whole degenerate-multiplet
    cancellation; with Elk's own occupations it is just a rewriting.
    """
    out = {}
    for ik in range(int(densityk["nkpt"])):
        vectors = jnp.asarray(densityk["evecfv"][(ik, ispn)])
        occupied = skip_below_epsocc(
            jnp.asarray(densityk["occsv"][ik])[:vectors.shape[0]])
        out[(ik, ispn)] = (vectors, vectors * occupied[:, None])
    return out


def interstitial_density(densityk, omega, ispn=0, factors=None):
    r"""``rhomagk``'s ``rmk3`` branch, summed over the zone.

    Returns the coarse-grid interstitial valence density, *before* ``rhonorm``.
    ``factors`` defaults to :func:`state_factors`, i.e. to Elk's own
    occupations; `elkjax.response.density_factors` is what replaces them when
    the density is being differentiated.
    """
    factors = state_factors(densityk, ispn) if factors is None else factors
    total = jnp.zeros(int(densityk["ngtc"]))
    for ik in range(int(densityk["nkpt"])):
        weight = float(densityk["wkpt"][ik])
        igkig = densityk["igkig"][(ik, ispn)]
        bra, ket = factors[(ik, ispn)]
        left = interstitial_wavefunction(bra, igkig, densityk)
        right = interstitial_wavefunction(ket, igkig, densityk)
        total = total + (weight / omega) * jnp.sum(
            jnp.real(jnp.conj(left) * right), axis=0)
    return total


def coarsen(fine, groundstate, densityk):
    """The exact inverse of ``rfirctof``: the fine grid back to the coarse one.

    ``rfirctof`` transforms the coarse function, copies its ``ngvc`` Fourier
    components into the fine array and zeroes the rest.  So the fine function
    carries no content the coarse one did not, and taking those same components
    back recovers it exactly -- no interpolation, no smoothing, and no
    tolerance introduced by the comparison itself.
    """
    fine_grid = tuple(int(n) for n in groundstate["ngridg"])
    coarse_grid = tuple(int(n) for n in densityk["ngdgc"])
    ngvc = int(densityk["ngvc"])
    igfft = np.asarray(groundstate["igfft"])[:ngvc] - 1
    igfc = np.asarray(densityk["igfc"])[:ngvc] - 1

    spectrum = _forward_fft(fine, fine_grid)
    array = jnp.zeros(int(densityk["ngtc"]), dtype=complex)
    array = array.at[jnp.asarray(igfc)].set(spectrum[jnp.asarray(igfft)])
    return jnp.real(_inverse_fft(array, coarse_grid))


def normalisation_offset(mine, elk_coarse):
    r"""``rhonorm``'s uniform shift, as ``(mean difference, deviation)``.

    ``rhonorm`` adds :math:`(N-N_{\rm calc})/\Omega` to every point rather than
    rescaling, so the difference between a correct transcription and Elk's
    stored density is a *constant*.  The second return value is how far from
    constant it actually is, which is the quantity worth asserting.
    """
    difference = np.asarray(elk_coarse) - np.asarray(mine)
    return float(difference.mean()), float(difference.std())


# ------------------------------------------------------- the muffin-tin half


def coarse_radial_indices(densityk, groundstate, ias):
    r"""Which fine radial points the coarse mesh is, for one atom.

    ``wfmtsv``'s inner ``zfzrf`` declares its radial argument ``rf(lrstp, n)``
    and uses ``rf(1, 1:n)``, i.e. every ``lradstp``-th element from wherever it
    was handed.  It is handed ``apwfr(1, ...)`` for the inner region and
    ``apwfr(iro, ...)`` with :math:`i_{ro}=n_{r}^{\rm i}+l_{\rm rstp}` for the
    outer one -- so the outer region does **not** continue the inner region's
    stride from where it stopped, it restarts one full step past the inner
    boundary.  Getting that wrong shifts the outer half of the density by one
    radial point and leaves it perfectly smooth.
    """
    isp = int(groundstate["idxis"][ias]) - 1
    step = int(densityk["lradstp"])
    nrcmti = int(densityk["nrcmti"][isp])
    nrcmt = int(densityk["nrcmt"][isp])
    nrmti = int(groundstate["nrmti"][isp])
    inner = step * np.arange(nrcmti)
    outer = (nrmti + step - 1) + step * np.arange(nrcmt - nrcmti)
    return inner, outer


def muffin_tin_wavefunctions(coefficients, apwalm, ngk, ias, densityk,
                             groundstate, lapw):
    r"""``wfmtsv`` with ``tsh=.false.``: states on the coarse angular grid.

    ``coefficients`` is one state's ``evecfv(:, n)`` or a ``(nst, nmat)`` stack
    of them; the return is correspondingly ``(nrcmt, lmmaxo)`` or
    ``(nst, nrcmt, lmmaxo)`` of complex VALUES on the angular grid (not
    harmonic coefficients), which is what ``rhomagk`` squares.

    The augmented part and the local-orbital part are summed into the same
    array, in that order, exactly as ``wfmtsv`` does; the sum over APW orders
    ``io`` is inside the sum over :math:`lm`, and the coefficient is a plain
    ``zdotu`` -- **not** conjugated, since these are expansion coefficients of
    the state rather than an inner product with it.

    The :math:`lm` loop is a matrix product rather than a Python loop: at 49
    harmonics and two APW orders it is the difference between ~100 traced
    operations per state and two, which is what makes a traced Kohn-Sham step
    compile in seconds instead of minutes.  Orders above ``apword(l)`` are
    masked out of BOTH factors rather than multiplied by a zero radial
    function, since an unset ``apwalm`` slot may hold anything and
    :math:`0\times\mathrm{NaN}` is not zero.
    """
    isp = int(groundstate["idxis"][ias]) - 1
    lmaxo = int(groundstate["lmaxo"])
    lmmaxi, lmmaxo = int(groundstate["lmmaxi"]), int(groundstate["lmmaxo"])
    nrcmti = int(densityk["nrcmti"][isp])
    nrcmt = int(densityk["nrcmt"][isp])
    inner, outer = coarse_radial_indices(densityk, groundstate, ias)
    rows = np.concatenate([inner, outer])

    apword = np.asarray(lapw["apword"])
    # the radial functions are a FUNCTION of the potential (patch 0015), so
    # they arrive traced whenever the density is differentiated in v
    apwfr = jnp.asarray(lapw["apwfr_full"])[:, 0]         # the function itself
    lofr = jnp.asarray(lapw["lofr"])[:, 0]
    idxlo = np.asarray(lapw["idxlo"])
    nlorb = int(np.asarray(lapw["nlorb"])[isp])
    # `lorbl` is a LIST over species of per-species arrays, not a rectangular
    # array: two species with different local-orbital counts make it ragged,
    # and `np.asarray` on the whole thing then raises.  On a one-species cell
    # it does not, which is why silicon never saw this.
    lorbl = np.asarray(lapw["lorbl"][isp])

    coefficients = jnp.asarray(coefficients)
    flat = coefficients.ndim == 1
    coefficients = coefficients[None, :] if flat else coefficients
    lvec = np.concatenate([np.full(2 * l + 1, l) for l in range(lmaxo + 1)])
    out = jnp.zeros((coefficients.shape[0], nrcmt, lmmaxo), dtype=complex)

    for io in range(int(apword[:lmaxo + 1, isp].max())):
        live = jnp.asarray(io < apword[lvec, isp])
        matched = jnp.where(live[None, :],
                            jnp.asarray(apwalm)[:ngk, io, :lmmaxo], 0.0)
        radial = jnp.where(live[None, :], apwfr[rows[:, None], io,
                                               lvec[None, :], ias], 0.0)
        out = out + (coefficients[:, :ngk] @ matched)[:, None, :] \
            * radial[None, :, :]

    slots = np.array([lm for ilo in range(nlorb)
                      for lm in range(int(lorbl[ilo]) ** 2,
                                      (int(lorbl[ilo]) + 1) ** 2)], dtype=int)
    if slots.size:
        which = np.array([ilo for ilo in range(nlorb)
                          for _ in range(2 * int(lorbl[ilo]) + 1)], dtype=int)
        index = np.array([int(idxlo[lm, ilo, ias]) - 1
                          for ilo, lm in zip(which, slots)], dtype=int)
        out = out.at[:, :, jnp.asarray(slots)].add(
            coefficients[:, ngk + jnp.asarray(index)][:, None, :]
            * lofr[rows[:, None], which[None, :], ias][None, :, :])

    # harmonic coefficients -> values on the angular grid, region by region.
    # The COMPLEX transform, not patch 0016's real `rbsht`: a wavefunction is
    # complex and Elk keeps a separate matrix for it.
    zbshti = jnp.asarray(densityk["zbshti"])
    zbshto = jnp.asarray(densityk["zbshto"])
    top = out[:, :nrcmti, :lmmaxi] @ zbshti.T
    bottom = out[:, nrcmti:] @ zbshto.T
    values = jnp.concatenate(
        [jnp.zeros((out.shape[0], nrcmti, lmmaxo),
                   dtype=complex).at[:, :, :lmmaxi].set(top), bottom], axis=1)
    return values[0] if flat else values


def muffin_tin_density(densityk, groundstate, lapw, ispn=0, factors=None):
    r"""``rhomagk``'s muffin-tin accumulation, summed over the zone.

    Returned dense per atom as ``(natmtot, nrcmt, lmmaxo)`` VALUES on the
    angular grid -- the representation Elk holds before ``rhomagsh``.  Note
    there is **no** :math:`1/\Omega` here: the muffin-tin weight is
    :math:`f_{n\mathbf k}w_{\mathbf k}` and only the interstitial carries the
    cell volume.

    ``factors`` is the ``(bra, ket)`` pair per k-point; see
    :func:`state_factors`.
    """
    from .hamiltonian import matching_coefficients

    factors = state_factors(densityk, ispn) if factors is None else factors
    natmtot = int(groundstate["natmtot"])
    bvec = np.asarray(groundstate["bvec"])
    vgc = np.asarray(groundstate["vgc"])

    out = [None] * natmtot
    for ik in range(int(densityk["nkpt"])):
        weight = float(densityk["wkpt"][ik])
        igkig = np.asarray(densityk["igkig"][(ik, ispn)])
        ngk = int(densityk["ngk"][ik, ispn])
        vgkc = (vgc[:, igkig - 1].T
                + (bvec @ np.asarray(densityk["vkl"])[:, ik])[None, :])
        apwalm = matching_coefficients(lapw, jnp.asarray(vgkc))
        bra, ket = factors[(ik, ispn)]
        for ias in range(natmtot):
            left = muffin_tin_wavefunctions(bra, apwalm[..., ias], ngk, ias,
                                            densityk, groundstate, lapw)
            right = muffin_tin_wavefunctions(ket, apwalm[..., ias], ngk, ias,
                                             densityk, groundstate, lapw)
            term = weight * jnp.sum(jnp.real(jnp.conj(left) * right), axis=0)
            out[ias] = term if out[ias] is None else out[ias] + term
    return jnp.stack(out)


def pack_coarse(values, densityk, groundstate, ias):
    """A dense coarse ``(nrcmt, lmmaxo)`` array into Elk's packed layout."""
    isp = int(groundstate["idxis"][ias]) - 1
    nrcmti = int(densityk["nrcmti"][isp])
    lmmaxi = int(groundstate["lmmaxi"])
    return jnp.concatenate([values[:nrcmti, :lmmaxi].reshape(-1),
                            values[nrcmti:].reshape(-1)])


# ------------------------------------------------ from rhomagk to the density


def to_harmonics(values, densityk, groundstate, ias):
    r"""``rhomagsh``: spherical coordinates back to harmonics, coarse mesh.

    ``rhomagk`` accumulates :math:`|\psi|^2` on the angular grid, because a
    modulus is pointwise and a harmonic expansion is not.  ``rfshtip`` maps it
    back.  The transform matrices are the REAL ones (patch 0016's ``rfshti``,
    ``rfshto``) even though `muffin_tin_wavefunctions` used the complex pair --
    a wavefunction is complex and a density is not.

    Takes and returns dense ``(nrcmt, lmmaxo)``.
    """
    isp = int(groundstate["idxis"][ias]) - 1
    nrcmti = int(densityk["nrcmti"][isp])
    lmmaxi = int(groundstate["lmmaxi"])
    lmmaxo = int(groundstate["lmmaxo"])
    rfshti = jnp.asarray(groundstate["rfshti"])
    rfshto = jnp.asarray(groundstate["rfshto"])
    values = jnp.asarray(values)
    top = values[:nrcmti, :lmmaxi] @ rfshti.T
    bottom = values[nrcmti:] @ rfshto.T
    return jnp.concatenate(
        [jnp.zeros((nrcmti, lmmaxo)).at[:, :lmmaxi].set(top), bottom], axis=0)


def coarse_to_fine(values, densityk, groundstate, ias):
    r"""``rfmtctof``: the coarse radial mesh to the fine one, per harmonic.

    Elk interpolates with a cubic spline whose weights come from ``wspline``
    and depend only on the mesh, so this is a **fixed linear map** and patch
    0020 exports it as a matrix rather than leaving ``splinew``'s weighted
    construction to be re-derived -- the same call as patch 0018's `symrfmt`.

    There are two of them because ``rfmtctof`` treats the two harmonic ranges
    differently: for :math:`l\le l_{\max}^{\rm i}` the function exists over
    the whole radial range and is interpolated over all of it, while for
    :math:`l>l_{\max}^{\rm i}` only the outer region carries it and only the
    outer region is interpolated.  Using the full map on an outer-only harmonic
    would read the inner region's zeros as data and pull the result toward zero
    near :math:`R_{\rm MT}`... smoothly.

    Takes dense ``(nrcmt, lmmaxo)`` and returns dense ``(nrmt, lmmaxo)``.
    """
    isp = int(groundstate["idxis"][ias]) - 1
    nrcmti = int(densityk["nrcmti"][isp])
    nrmt = int(densityk["nrmt"][isp])
    nrmti = int(densityk["nrmti"][isp])
    lmmaxi = int(groundstate["lmmaxi"])
    lmmaxo = int(groundstate["lmmaxo"])
    full = jnp.asarray(densityk["ctof_full"][isp])
    outer = jnp.asarray(densityk["ctof_outer"][isp])
    values = jnp.asarray(values)

    inner_block = full @ values[:, :lmmaxi]
    outer_block = outer @ values[nrcmti:, lmmaxi:]
    out = jnp.zeros((nrmt, lmmaxo))
    out = out.at[:, :lmmaxi].set(inner_block)
    return out.at[nrmti:, lmmaxi:].set(outer_block)


def pack_fine(values, groundstate, ias):
    """A dense fine ``(nrmt, lmmaxo)`` array into Elk's packed layout."""
    isp = int(groundstate["idxis"][ias]) - 1
    nrmti = int(groundstate["nrmti"][isp])
    lmmaxi = int(groundstate["lmmaxi"])
    return jnp.concatenate([values[:nrmti, :lmmaxi].reshape(-1),
                            values[nrmti:].reshape(-1)])


# --------------------------------------------- the loop closed, one iteration


def zone_matrices(lapw, groundstate, densityk, ispn=0):
    r"""``(H, O)`` at every k-point of Elk's own set, as a list.

    Split out of :func:`solve_zone` because the differentiable path needs the
    matrices themselves: `elkjax.response.density_factors` is a function of
    :math:`(H,O,\mu)` and rebuilds the eigenvectors inside its own rule.
    """
    from .hamiltonian import eigenproblem_on_gset

    bvec = np.asarray(groundstate["bvec"])
    vgc = np.asarray(groundstate["vgc"])
    out = []
    for ik in range(int(densityk["nkpt"])):
        igkig = np.asarray(densityk["igkig"][(ik, ispn)])
        ngp = int(densityk["ngk"][ik, ispn])
        vgkc = (vgc[:, igkig - 1].T
                + (bvec @ np.asarray(densityk["vkl"])[:, ik])[None, :])
        out.append(eigenproblem_on_gset(lapw, groundstate, igkig, vgkc, ngp))
    return out


def solve_zone(lapw, groundstate, densityk, ispn=0, eigenvalues=False):
    r"""Diagonalise at every k-point of Elk's own set, from the potential.

    Returns a dict keyed ``(ik, ispn)`` of ``evecfv`` in Elk's normalisation
    (:math:`c^\dagger Oc=\mathbb 1`), shaped ``(nstfv, nmat)`` so it can be
    dropped straight into :func:`muffin_tin_density` and
    :func:`interstitial_density` in place of the exported one.

    The muffin-tin blocks come from the radial integrals -- which
    `elkjax.radial` builds from the potential (§1k) -- and the interstitial
    ones from ``vsig``/``cfunig``, so **nothing here reads an eigenvector**.
    That is the whole point: it is the other half of the SCF step.

    ``eigenvalues=True`` also returns the ``(nkpt, nstfv)`` spectrum, which is
    what `elkjax.occupations` needs and what makes the step a closed loop
    rather than a half-step at Elk's own occupations.  For a scalar
    (non-spin-polarised, no spin-orbit) calculation ``eveqnsv`` is the
    identity, so these first-variational eigenvalues **are** Elk's ``evalsv``.
    """
    from .hamiltonian import cholesky_reduce

    nstfv = int(densityk["nstfv"])
    out, spectrum = {}, []
    for ik, (h, o) in enumerate(zone_matrices(lapw, groundstate, densityk,
                                              ispn=ispn)):
        reduced, chol = cholesky_reduce(h, o)
        values, y = jnp.linalg.eigh(reduced)
        # c = L^{-dagger} y, which restores Elk's own normalisation
        vectors = jnp.linalg.solve(chol.conj().T, y[:, :nstfv])
        out[(ik, ispn)] = vectors.T
        spectrum.append(values[:nstfv])
    return (out, jnp.stack(spectrum)) if eigenvalues else out


def density_from_potential(lapw, groundstate, densityk, ispn=0):
    """One SCF half-step: potential -> eigenvectors -> valence density.

    Returns ``(muffin tin values on the coarse angular grid, interstitial on
    the coarse FFT grid)``, in the same representation ``rhomagk`` produces,
    so it is directly comparable with patch 0019's exported reference.

    Elk's own occupations are used.  They are a functional of the eigenvalues
    through the Fermi level, and a zone-summed Fermi level is not built here
    (§1i has it at a single k) -- so this is the density given the occupations,
    which is what makes it a test of the assembly and the density rather than
    of the smearing.
    """
    vectors = solve_zone(lapw, groundstate, densityk, ispn=ispn)
    local = dict(densityk)
    local["evecfv"] = dict(densityk["evecfv"])
    for key, value in vectors.items():
        local["evecfv"][key] = value
    return (muffin_tin_density(local, groundstate, lapw, ispn=ispn),
            interstitial_density(local, float(groundstate["omega"]),
                                 ispn=ispn))


# ------------------------------------------------------- core and normalisation


def add_core(values, densityk, groundstate, ias):
    r"""``rhocore``: the core density into the :math:`l=0` slot.

    ``rhocr`` is stored the way ``vcln`` is -- as the :math:`(0,0)`
    *coefficient*, with the :math:`1/y_{00}` already folded in -- so it is added
    to column 0 of the dense array and nowhere else.  The core is spherical by
    construction, which is why there is nothing to add anywhere else.

    It is a functional of the potential (``gencore`` solves the core states in
    it), so patch 0021 exports it for the same reason `vsmt` is exported: an
    input at fixed potential.  Summed over ``nspncr``, which is 1 unless
    ``spincore``.
    """
    isp = int(groundstate["idxis"][ias]) - 1
    nr = int(densityk["nrmt"][isp])
    core = jnp.asarray(densityk["rhocr"])[ias, :, :nr].sum(axis=0)
    return jnp.asarray(values).at[:, 0].add(core)


def normalise(muffin, interstitial, densityk, groundstate):
    r"""``charge`` + ``rhonorm``: the uniform shift that fixes the electron count.

    .. math::

        \rho\to\rho+\frac{N-N_{\rm calc}}{\Omega},\qquad
        N_{\rm calc}=\sum_\alpha\!\int_{\rm MT}\!\rho
        +\frac{\Omega}{N_{\rm FFT}}\sum_{\bf r}\rho\,\Theta ,

    an ADDITIVE shift and not a rescaling -- so it moves only the
    :math:`(0,0)` coefficient in the muffin tin, where the same
    :math:`1/y_{00}` applies as for the core and the nucleus.

    ``muffin`` is a stack of dense fine-mesh arrays, ``interstitial`` is on the
    fine FFT grid.  Returns the pair, shifted.
    """
    from . import integrate

    natmtot = int(groundstate["natmtot"])
    packed = jnp.stack([pack_fine(muffin[ias], groundstate, ias)
                        for ias in range(natmtot)])
    count = integrate.cell_integral(packed, interstitial, groundstate)
    omega = float(groundstate["omega"])
    shift = (float(densityk["chgtot"]) - count) / omega
    y00i = 3.54490770181103205460
    return (jnp.stack([muffin[ias].at[:, 0].add(shift * y00i)
                       for ias in range(natmtot)]),
            jnp.asarray(interstitial) + shift)


def refine(coarse, groundstate, densityk):
    """``rfirctof``: the coarse interstitial grid to the fine one.

    The exact inverse of :func:`coarsen` -- zero-padding in :math:`G`-space --
    written here rather than through Elk's own `rzfftifc` real-to-complex
    packing, which carries its own `nfgrz`/`igrzf` indexing and would be one
    more set of conventions for no gain.
    """
    coarse_grid = tuple(int(n) for n in densityk["ngdgc"])
    fine_grid = tuple(int(n) for n in groundstate["ngridg"])
    ngvc = int(densityk["ngvc"])
    igfft = np.asarray(groundstate["igfft"])[:ngvc] - 1
    igfc = np.asarray(densityk["igfc"])[:ngvc] - 1
    spectrum = _forward_fft(coarse, coarse_grid)
    array = jnp.zeros(int(groundstate["ngtot"]), dtype=complex)
    array = array.at[jnp.asarray(igfft)].set(spectrum[jnp.asarray(igfc)])
    return jnp.real(_inverse_fft(array, fine_grid))


def converged_density(densityk, groundstate, lapw, vectors=None, ispn=0,
                      symmetrise=None, factors=None):
    r"""The whole of ``rhomag``: eigenvectors to Elk's converged ``rhomt``/``rhoir``.

    ``rhomagk`` → ``rhomagsh`` → ``symrf`` → ``rfmtctof`` / ``rfirctof`` →
    ``rhocore`` → ``rhonorm``, i.e. every step between the first-variational
    eigenvectors and the arrays the next iteration's potential is built from.

    ``symrf`` runs when the cell has more than one crystal symmetry, which is
    detected rather than asked for; ``symmetrise=False`` forces it off.  It is
    applied on the COARSE mesh and before the radial interpolation, where Elk
    applies it -- and with the coarse region boundary, which is the detail that
    matters: passing the fine ``nrmti`` treats every coarse point as interior
    and leaves a smooth, positive, wrong density (measured 8e-6 against 2e-13).

    ``vectors`` overrides the exported ``evecfv`` -- pass
    :func:`solve_zone`'s output to drive the whole thing from the potential
    instead.  ``factors`` overrides the ``(bra, ket)`` pair outright and is
    what `elkjax.scf` uses: the occupations then reach the accumulation inside
    a pair whose derivative is `elkjax.response`'s, rather than as a separate
    array.

    Returns ``(muffin tin dense on the fine mesh, interstitial on the fine
    grid)``.
    """
    natmtot = int(groundstate["natmtot"])
    local = dict(densityk)
    if vectors is not None:
        local["evecfv"] = dict(densityk["evecfv"])
        local["evecfv"].update(vectors)

    from . import symmetry

    if symmetrise is None:
        symmetrise = int(densityk.get("nsymcrys", 1)) > 1

    values = muffin_tin_density(local, groundstate, lapw, ispn=ispn,
                                factors=factors)
    harmonics = jnp.stack([to_harmonics(values[ias], densityk, groundstate, ias)
                           for ias in range(natmtot)])
    coarse = interstitial_density(local, float(groundstate["omega"]),
                                  ispn=ispn, factors=factors)
    if symmetrise:
        harmonics = symmetry.symmetrise(harmonics, groundstate,
                                        inner_points=densityk["nrcmti"])
        coarse = symmetrise_interstitial(coarse, groundstate, densityk)

    muffin = [add_core(coarse_to_fine(harmonics[ias], densityk, groundstate,
                                      ias), densityk, groundstate, ias)
              for ias in range(natmtot)]
    return normalise(jnp.stack(muffin), refine(coarse, groundstate, densityk),
                     densityk, groundstate)


def symmetrise_interstitial(coarse, groundstate, densityk):
    r"""``symrfir``: the interstitial symmetrisation, on the coarse grid.

    .. math::

        \hat S\rho(\mathbf G)=\frac1{n_{\rm sym}}\sum_{\rm isym}
        \rho(S_{\rm isym}\mathbf G)\,
        e^{-i(S_{\rm isym}\mathbf G)\cdot\mathbf t_{\rm isym}} ,

    a permutation of the :math:`\mathbf G` vectors and a phase, which is why
    patch 0022 exports the operator as two small arrays rather than the
    ``ngtc``-square matrix a real-space form would need.  The counterpart of
    §2g's `symrfmt` for the interstitial, and what lifts §2h's and §2j's
    restriction to ``symtype=0``.

    Components beyond ``ngvc`` are zeroed, as ``symrfir`` leaves them: it skips
    ``ig > ngvec`` and its accumulator starts at zero.
    """
    grid = tuple(int(n) for n in densityk["ngdgc"])
    igfc = np.asarray(densityk["igfc"])
    ngvc = int(densityk["ngvc"])
    symmap = np.asarray(densityk["symmap"])
    symphase = jnp.asarray(densityk["symphase"])

    spectrum = _forward_fft(coarse, grid)
    slots = igfc[:ngvc] - 1
    source = igfc[symmap - 1] - 1                      # (nsymcrys, ngvc)
    averaged = jnp.mean(spectrum[jnp.asarray(source)] * symphase, axis=0)

    out = jnp.zeros(int(densityk["ngtc"]), dtype=complex)
    out = out.at[jnp.asarray(slots)].set(averaged)
    return jnp.real(_inverse_fft(out, grid))
