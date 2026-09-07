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
           "muffin_tin_wavefunctions", "muffin_tin_density"]

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
