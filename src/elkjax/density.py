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

import jax.numpy as jnp

__all__ = ["interstitial_wavefunction", "interstitial_density", "coarsen",
           "normalisation_offset", "coarse_radial_indices",
           "muffin_tin_wavefunctions", "muffin_tin_density",
           "pack_coarse", "pack_fine", "to_harmonics",
           "coarse_to_fine", "solve_zone", "density_from_potential",
           "add_core", "normalise", "refine", "converged_density"]

EPSOCC = 1.0e-8


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

    ``coefficients`` is one state's ``evecfv(1:ngk, n)``; ``igkig`` maps each
    :math:`{\bf G+k}` to the global :math:`{\bf G}` index, and ``igfc`` maps
    that to a slot of the coarse FFT array.  The two-step indirection is the
    part a transcription gets wrong, and it is Elk's own: ``igkig`` is a
    property of the k-point, ``igfc`` of the grid.
    """
    grid = tuple(int(n) for n in densityk["ngdgc"])
    igfc = np.asarray(densityk["igfc"])
    igkig = np.asarray(igkig)
    slots = igfc[igkig - 1] - 1
    array = jnp.zeros(int(densityk["ngtc"]), dtype=complex)
    # only the first `ngk` coefficients: the rest are local orbitals, which
    # live entirely inside the muffin tins and contribute nothing here.  The
    # muffin-tin half uses all `nmat` of them.
    array = array.at[jnp.asarray(slots)].set(
        jnp.asarray(coefficients)[:igkig.size])
    return _inverse_fft(array, grid)


def interstitial_density(densityk, omega, ispn=0):
    r"""``rhomagk``'s ``rmk3`` branch, summed over the zone.

    Returns the coarse-grid interstitial valence density, *before* ``rhonorm``.
    States with :math:`|f|<` ``epsocc`` are skipped exactly as Elk skips them;
    including them would change the result by less than :math:`10^{-8}` times a
    wavefunction modulus, which is above the tolerance this is compared at.
    """
    total = jnp.zeros(int(densityk["ngtc"]))
    for ik in range(int(densityk["nkpt"])):
        weight = float(densityk["wkpt"][ik])
        occupations = np.asarray(densityk["occsv"][ik])
        vectors = densityk["evecfv"][(ik, ispn)]
        igkig = densityk["igkig"][(ik, ispn)]
        for ist, occupation in enumerate(occupations[:vectors.shape[0]]):
            if abs(occupation) < EPSOCC:
                continue
            psi = interstitial_wavefunction(vectors[ist], igkig, densityk)
            total = total + (occupation * weight / omega) * jnp.abs(psi) ** 2
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
    r"""``wfmtsv`` with ``tsh=.false.``: one state on the coarse angular grid.

    Returns a dense ``(nrcmt, lmmaxo)`` complex array of VALUES on the angular
    grid (not harmonic coefficients), which is what ``rhomagk`` squares.

    The augmented part and the local-orbital part are summed into the same
    array, in that order, exactly as ``wfmtsv`` does; the sum over APW orders
    ``io`` is inside the sum over :math:`lm`, and the coefficient is a plain
    ``zdotu`` -- **not** conjugated, since these are expansion coefficients of
    the state rather than an inner product with it.
    """
    isp = int(groundstate["idxis"][ias]) - 1
    lmaxi, lmaxo = int(groundstate["lmaxi"]), int(groundstate["lmaxo"])
    lmmaxi, lmmaxo = int(groundstate["lmmaxi"]), int(groundstate["lmmaxo"])
    nrcmti = int(densityk["nrcmti"][isp])
    nrcmt = int(densityk["nrcmt"][isp])
    inner, outer = coarse_radial_indices(densityk, groundstate, ias)
    rows = np.concatenate([inner, outer])

    apword = np.asarray(lapw["apword"])
    apwfr = np.asarray(lapw["apwfr_full"])[:, 0]          # the function itself
    lofr = np.asarray(lapw["lofr"])[:, 0]
    idxlo = np.asarray(lapw["idxlo"])
    nlorb = int(np.asarray(lapw["nlorb"])[isp])
    lorbl = np.asarray(lapw["lorbl"])[isp]

    coefficients = jnp.asarray(coefficients)
    out = jnp.zeros((nrcmt, lmmaxo), dtype=complex)

    for l in range(lmaxo + 1):
        for io in range(int(apword[l, isp])):
            radial = jnp.asarray(apwfr[rows, io, l, ias])
            for lm in range(l * l, (l + 1) ** 2):
                y = jnp.dot(coefficients[:ngk], jnp.asarray(apwalm[:ngk, io, lm]))
                out = out.at[:, lm].add(y * radial)

    for ilo in range(nlorb):
        l = int(lorbl[ilo])
        radial = jnp.asarray(lofr[rows, ilo, ias])
        for lm in range(l * l, (l + 1) ** 2):
            index = int(idxlo[lm, ilo, ias]) - 1
            y = coefficients[ngk + index]
            out = out.at[:, lm].add(y * radial)

    # harmonic coefficients -> values on the angular grid, region by region.
    # The COMPLEX transform, not patch 0016's real `rbsht`: a wavefunction is
    # complex and Elk keeps a separate matrix for it.
    zbshti = jnp.asarray(densityk["zbshti"])
    zbshto = jnp.asarray(densityk["zbshto"])
    top = out[:nrcmti, :lmmaxi] @ zbshti.T
    bottom = out[nrcmti:] @ zbshto.T
    return jnp.concatenate(
        [jnp.zeros((nrcmti, lmmaxo), dtype=complex).at[:, :lmmaxi].set(top),
         bottom], axis=0)


def muffin_tin_density(densityk, groundstate, lapw, ispn=0):
    r"""``rhomagk``'s muffin-tin accumulation, summed over the zone.

    Returned dense per atom as ``(natmtot, nrcmt, lmmaxo)`` VALUES on the
    angular grid -- the representation Elk holds before ``rhomagsh``.  Note
    there is **no** :math:`1/\Omega` here: the muffin-tin weight is
    :math:`f_{n\mathbf k}w_{\mathbf k}` and only the interstitial carries the
    cell volume.
    """
    from .hamiltonian import matching_coefficients

    natmtot = int(groundstate["natmtot"])
    lmmaxo = int(groundstate["lmmaxo"])
    bvec = np.asarray(groundstate["bvec"])
    vgc = np.asarray(groundstate["vgc"])

    out = [None] * natmtot
    for ik in range(int(densityk["nkpt"])):
        weight = float(densityk["wkpt"][ik])
        vectors = densityk["evecfv"][(ik, ispn)]
        igkig = np.asarray(densityk["igkig"][(ik, ispn)])
        ngk = int(densityk["ngk"][ik, ispn])
        vgkc = (vgc[:, igkig - 1].T
                + (bvec @ np.asarray(densityk["vkl"])[:, ik])[None, :])
        apwalm = matching_coefficients(lapw, jnp.asarray(vgkc))
        occupations = np.asarray(densityk["occsv"][ik])
        for ias in range(natmtot):
            for ist, occupation in enumerate(occupations[:vectors.shape[0]]):
                if abs(occupation) < EPSOCC:
                    continue
                psi = muffin_tin_wavefunctions(
                    vectors[ist], apwalm[..., ias], ngk, ias, densityk,
                    groundstate, lapw)
                term = (occupation * weight) * jnp.abs(psi) ** 2
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


def solve_zone(lapw, groundstate, densityk, ispn=0):
    r"""Diagonalise at every k-point of Elk's own set, from the potential.

    Returns a dict keyed ``(ik, ispn)`` of ``evecfv`` in Elk's normalisation
    (:math:`c^\dagger Oc=\mathbb 1`), shaped ``(nstfv, nmat)`` so it can be
    dropped straight into :func:`muffin_tin_density` and
    :func:`interstitial_density` in place of the exported one.

    The muffin-tin blocks come from the radial integrals -- which
    `elkjax.radial` builds from the potential (§1k) -- and the interstitial
    ones from ``vsig``/``cfunig``, so **nothing here reads an eigenvector**.
    That is the whole point: it is the other half of the SCF step.
    """
    from .hamiltonian import cholesky_reduce, eigenproblem_on_gset

    bvec = np.asarray(groundstate["bvec"])
    vgc = np.asarray(groundstate["vgc"])
    nstfv = int(densityk["nstfv"])
    out = {}
    for ik in range(int(densityk["nkpt"])):
        igkig = np.asarray(densityk["igkig"][(ik, ispn)])
        ngp = int(densityk["ngk"][ik, ispn])
        vgkc = (vgc[:, igkig - 1].T
                + (bvec @ np.asarray(densityk["vkl"])[:, ik])[None, :])
        h, o = eigenproblem_on_gset(lapw, groundstate, igkig, vgkc, ngp)
        reduced, chol = cholesky_reduce(h, o)
        _, y = jnp.linalg.eigh(reduced)
        # c = L^{-dagger} y, which restores Elk's own normalisation
        vectors = jnp.linalg.solve(chol.conj().T, y[:, :nstfv])
        out[(ik, ispn)] = vectors.T
    return out


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
    count = float(integrate.cell_integral(packed, interstitial, groundstate))
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


def converged_density(densityk, groundstate, lapw, vectors=None, ispn=0):
    r"""The whole of ``rhomag``: eigenvectors to Elk's converged ``rhomt``/``rhoir``.

    ``rhomagk`` → ``rhomagsh`` → ``rfmtctof`` / ``rfirctof`` → ``rhocore`` →
    ``rhonorm``, i.e. every step between the first-variational eigenvectors and
    the arrays the next iteration's potential is built from.  ``symrf`` is
    absent because these fixtures run ``symtype=0``, where it is the identity.

    ``vectors`` overrides the exported ``evecfv`` -- pass
    :func:`solve_zone`'s output to drive the whole thing from the potential
    instead.

    Returns ``(muffin tin dense on the fine mesh, interstitial on the fine
    grid)``.
    """
    natmtot = int(groundstate["natmtot"])
    local = dict(densityk)
    if vectors is not None:
        local["evecfv"] = dict(densityk["evecfv"])
        local["evecfv"].update(vectors)

    values = muffin_tin_density(local, groundstate, lapw, ispn=ispn)
    muffin = []
    for ias in range(natmtot):
        harmonics = to_harmonics(values[ias], densityk, groundstate, ias)
        fine = coarse_to_fine(harmonics, densityk, groundstate, ias)
        muffin.append(add_core(fine, densityk, groundstate, ias))
    interstitial = refine(
        interstitial_density(local, float(groundstate["omega"]), ispn=ispn),
        groundstate, densityk)
    return normalise(jnp.stack(muffin), interstitial, densityk, groundstate)
