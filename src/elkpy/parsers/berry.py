"""Parse ELKPY_BERRY.OUT (elkpy task 9000, src/elkpy_berry.f90) and compute the
Wilson-loop / Fukui-Hatsugai-Suzuki (FHS) discretized Berry curvature from the
mesh-neighbour wavefunction overlap matrices it contains.

FHS: T. Fukui, Y. Hatsugai, H. Suzuki, "Chern Numbers in Discretized
Brillouin Zone: Efficient Method of Computing (Spin) Hall Conductances",
J. Phys. Soc. Jpn. 74, 1674 (2005) (arXiv:cond-mat/0503172). See
docs/design.md #13 / docs/physics.tex Part II for the physics writeup this
implements (link variables, plaquette field strength, admissibility).

The Fortran side (src/elkpy_berry.f90) only exports raw overlap matrices
(and boundary eigenvalues for a gap check) -- all Wilson-loop arithmetic
(link variables, plaquette flux, Chern number) happens here, in pure numpy,
so it's independently unit-testable against synthetic data (see
tests/test_berry_gauge_invariance.py) without needing an Elk run.
"""

import numpy as np


def parse_berry_overlaps(path):
    """Parse ELKPY_BERRY.OUT into its raw ingredients.

    Returns a dict:
      "ngridk": (n1, n2, n3)
      "directions": (dir1, dir2) -- the two ngridk grid axes (1, 2 or 3)
          spanning the Wilson-loop plane
      "band_window": (ist0, ist1) -- the requested contiguous band window
      "boundary_window": (nlo, nhi) -- band_window widened by one state on
          each side (clipped to [1, nstsv]), matching the eigenvalues below
      "bvec": (3, 3) array, Cartesian reciprocal lattice vectors (rows)
      "eigenvalues": {(i1, i2, i3): array of shape (nhi-nlo+1,)}, Hartree
      "overlaps": {(i1, i2, i3): (M1, M2)} -- M1/M2 are (nst, nst) complex
          arrays with M_d(a, b) = <psi_{ist0+a}(k) | psi_{ist0+b}(k+e_d)>,
          d=1 -> directions[0], d=2 -> directions[1], e_d the mesh step in
          direction d (1/ngridk(d) of a reciprocal lattice vector)
    """
    text = open(path).read()
    _, rest = text.split("\n", 1)
    toks = rest.split()
    pos = 0

    def take(n, cast):
        nonlocal pos
        vals = [cast(t) for t in toks[pos : pos + n]]
        pos += n
        return vals

    ngridk = tuple(take(3, int))
    directions = tuple(take(2, int))
    ist0, ist1 = take(2, int)
    nlo, nhi = take(2, int)
    bvec = np.array([take(3, float) for _ in range(3)])
    nkptnr = take(1, int)[0]
    nst = ist1 - ist0 + 1
    nbnd = nhi - nlo + 1

    eigenvalues = {}
    overlaps = {}
    for _ in range(nkptnr):
        i1, i2, i3 = take(3, int)
        eigenvalues[(i1, i2, i3)] = np.array(take(nbnd, float))
        mats = []
        for _d in range(2):
            flat = np.array(take(2 * nst * nst, float))
            reim = flat.reshape(nst * nst, 2)
            values = reim[:, 0] + 1j * reim[:, 1]
            # Fortran wrote the fastest-varying index first ("do b; do a"
            # with a innermost) -- i.e. Fortran/column-major order, matching
            # M(a, b)'s own natural storage order.
            mat = values.reshape(nst, nst, order="F")
            mats.append(mat)
        overlaps[(i1, i2, i3)] = tuple(mats)

    return {
        "ngridk": ngridk,
        "directions": directions,
        "band_window": (ist0, ist1),
        "boundary_window": (nlo, nhi),
        "bvec": bvec,
        "eigenvalues": eigenvalues,
        "overlaps": overlaps,
    }


def _berry_phase(w):
    """The Berry phase of a closed loop whose link-variable product is `w`.

    This is where elkpy's Berry-curvature sign convention is set, in ONE
    place, so every consumer (mesh curvature, path curvature, Chern number,
    parsers.quantum_geometry) inherits it identically.

    The negation is not cosmetic. With link variables
    U(k -> k') = <u(k)|u(k')>/|.|, expanding
    <u|u + delta.grad u> = 1 + delta.<u|grad u> = exp(-i A.delta)
    (using that <u|grad u> is purely imaginary, with
    A = i<u|grad_k u> real) makes the product around a closed
    counter-clockwise loop equal to exp(-i * closed integral of A). Its
    ARGUMENT is therefore minus the Berry phase, which is why the standard
    discrete Berry phase carries an explicit negation,

        gamma = -Im ln prod_j <u_j|u_{j+1}>

    (the King-Smith--Vanderbilt/Resta convention, PRB 47, 1651(R) (1993)).
    Omitting it yields -Omega in the standard convention A = i<u|grad_k u>,
    Omega = curl A (Xiao, Chang & Niu, RMP 82, 1959 (2010)) -- and hence a
    Chern number of the wrong sign relative to the literature.

    That omission was elkpy's behaviour until this convention was
    unified: it went unnoticed because every check on the Berry/Z2
    machinery is sign-blind (bulk Si's Chern number is 0 = -0; the h-BN
    benchmark is a RELATIVE K/K' antisymmetry; Z2 is a parity of
    crossings; det g >= (F/2)^2 is even in F). It was caught only when
    parsers.optical added an independent Kubo-form route whose sign is
    pinned analytically against direct differentiation
    (tests/test_parsers_optical.py). See docs/design.md #22.
    """
    return -float(np.angle(w))


def _link_variable(mat):
    d = np.linalg.det(mat)
    if d == 0:
        raise ValueError(
            "singular overlap matrix (det=0) -- the requested band window is not "
            "resolved from the rest of the spectrum at this k-point, or the mesh/basis "
            "is too coarse; check_gap() flags a closing gap explicitly"
        )
    return d / abs(d)


def check_gap(parsed, tol=1e-4):
    """Raise ValueError if the requested band window isn't separated from the
    bands immediately below/above it (Hartree) at every k-point on the mesh --
    the FHS non-Abelian construction (eq. 16) requires the window to stay
    gapped from the rest of the spectrum everywhere on the Brillouin zone."""
    ist0, ist1 = parsed["band_window"]
    nlo, nhi = parsed["boundary_window"]
    for k, evals in parsed["eigenvalues"].items():
        window = evals[ist0 - nlo : ist1 - nlo + 1]
        if nlo < ist0:
            gap = window[0] - evals[0]
            if gap < tol:
                raise ValueError(
                    f"band window [{ist0},{ist1}] not gapped from band {ist0 - 1} at "
                    f"k-grid-index {k}: gap = {gap:.2e} Ha < tol = {tol:.2e}"
                )
        if nhi > ist1:
            gap = evals[-1] - window[-1]
            if gap < tol:
                raise ValueError(
                    f"band window [{ist0},{ist1}] not gapped from band {ist1 + 1} at "
                    f"k-grid-index {k}: gap = {gap:.2e} Ha < tol = {tol:.2e}"
                )


def compute_berry_curvature(parsed):
    """FHS discretized Berry curvature and Chern number from parsed
    ELKPY_BERRY.OUT data (parse_berry_overlaps()).

    Sign convention (see _berry_phase() below): A = i<u|grad_k u>,
    Omega = curl A -- the standard Xiao, Chang & Niu (RMP 82, 1959 (2010))
    convention, which is what the literature's Chern numbers and curvature
    plots use, and what parsers.optical's independent Kubo-form route
    computes.

    Returns a dict:
      "flux": (n1, n2, n3) array -- the dimensionless plaquette Berry
          phase theta(k) in (-pi, pi], i.e. the negated argument of FHS
          eq. 8's link product; NOT yet divided by plaquette area.
      "chern_number": (n_free,) array -- one Chern number per slice along
          the direction not spanned by the loop (eq. 9: sum of flux/(2*pi)
          over each (dir1, dir2) plane).
      "max_flux": float -- max |theta(k)| over all plaquettes, the FHS
          admissibility diagnostic (eq. 14): should be well under pi for a
          mesh-converged result; values approaching pi mean the mesh is too
          coarse to trust the resulting Chern number.
    """
    n1, n2, n3 = parsed["ngridk"]
    dir1, dir2 = parsed["directions"]
    overlaps = parsed["overlaps"]
    ns = (n1, n2, n3)

    def neighbor(k, direction):
        idx = list(k)
        idx[direction - 1] = (idx[direction - 1] + 1) % ns[direction - 1]
        return tuple(idx)

    def link(k, direction):
        mat = overlaps[k][0] if direction == dir1 else overlaps[k][1]
        return _link_variable(mat)

    flux = np.zeros((n1, n2, n3))
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                k = (i1, i2, i3)
                k_plus_1 = neighbor(k, dir1)
                k_plus_2 = neighbor(k, dir2)
                w = link(k, dir1) * link(k_plus_1, dir2) / (link(k_plus_2, dir1) * link(k, dir2))
                flux[i1, i2, i3] = _berry_phase(w)

    dir1_axis, dir2_axis = dir1 - 1, dir2 - 1
    chern_number = flux.sum(axis=(dir1_axis, dir2_axis)) / (2 * np.pi)

    return {"flux": flux, "chern_number": chern_number, "max_flux": float(np.max(np.abs(flux)))}


def parse_berry_curvature(path, check_gapped=True, gap_tol=1e-4):
    """Parse ELKPY_BERRY.OUT and compute the Wilson-loop Berry curvature in
    one call -- see parse_berry_overlaps()/compute_berry_curvature() for the
    two steps this composes."""
    parsed = parse_berry_overlaps(path)
    if check_gapped:
        check_gap(parsed, tol=gap_tol)
    return compute_berry_curvature(parsed)


def parse_berry_path_overlaps(path):
    """Parse ELKPY_BERRY_PATH.OUT (elkpy task 9001) into its raw ingredients.

    Unlike parse_berry_overlaps() (task 9000, a periodic mesh covering the
    whole Brillouin zone), this is a small Wilson loop -- four corners
    k0 +- dir1*dk +- dir2*dk -- independently evaluated at each of an
    arbitrary, explicitly requested list of k-points (pyqula's
    berry_curvature() convention), via fresh on-the-fly diagonalisation in
    Fortran rather than a mesh lookup. No global topological invariant
    (Chern number) is defined for this mode -- see compute_berry_curvature()
    for that.

    Returns a dict:
      "directions": (dir1, dir2)
      "dk": float -- fractional-coordinate half-width of the loop
      "band_window": (ist0, ist1)
      "bvec": (3, 3) array, Cartesian reciprocal lattice vectors (rows)
      "points": [(k0, (M12, M23, M34, M41)), ...] -- k0 is the requested
          (kx, ky, kz) in fractional coordinates; the four (nst, nst)
          complex matrices are M_edge(a, b) = <psi_a(corner_i)|psi_b(corner_j)>
          walked cyclically around the loop corners 1->2->3->4->1, where
          corner 1 = k0-dir1*dk-dir2*dk, 2 = k0+dir1*dk-dir2*dk,
          3 = k0+dir1*dk+dir2*dk, 4 = k0-dir1*dk+dir2*dk (matching pyqula's
          wf1..wf4 convention).
    """
    text = open(path).read()
    _, rest = text.split("\n", 1)
    toks = rest.split()
    pos = 0

    def take(n, cast):
        nonlocal pos
        vals = [cast(t) for t in toks[pos : pos + n]]
        pos += n
        return vals

    dir1, dir2 = take(2, int)
    dk = take(1, float)[0]
    ist0, ist1 = take(2, int)
    bvec = np.array([take(3, float) for _ in range(3)])
    npoints = take(1, int)[0]
    nst = ist1 - ist0 + 1

    points = []
    for _ in range(npoints):
        k0 = tuple(take(3, float))
        edges = []
        for _e in range(4):
            flat = np.array(take(2 * nst * nst, float))
            reim = flat.reshape(nst * nst, 2)
            values = reim[:, 0] + 1j * reim[:, 1]
            mat = values.reshape(nst, nst, order="F")
            edges.append(mat)
        points.append((k0, tuple(edges)))

    return {
        "directions": (dir1, dir2),
        "dk": dk,
        "band_window": (ist0, ist1),
        "bvec": bvec,
        "points": points,
    }


def compute_berry_curvature_path(parsed):
    """Berry curvature at each point of a parsed ELKPY_BERRY_PATH.OUT
    (parse_berry_path_overlaps()), one small Wilson loop per point -- the
    single-plaquette special case of compute_berry_curvature()'s FHS
    construction (eq. 8), normalized by the loop's actual Cartesian area
    (accounting for a non-orthogonal reciprocal lattice, unlike a fixed
    `dk*dk`-style normalization that implicitly assumes an orthogonal one).

    Returns a list of dicts, one per requested k-point, in the order given:
    {"k": (kx, ky, kz) fractional, "flux": dimensionless plaquette phase in
    (-pi, pi], "curvature": flux / loop area (Bohr^2 -- flux is dimensionless and the
    area is Bohr^-2, since bvec is
    Cartesian reciprocal Bohr^-1)}.
    """
    dir1, dir2 = parsed["directions"]
    dk = parsed["dk"]
    bvec = parsed["bvec"]
    edge1_cart = 2 * dk * bvec[dir1 - 1]
    edge2_cart = 2 * dk * bvec[dir2 - 1]
    area = float(np.linalg.norm(np.cross(edge1_cart, edge2_cart)))

    results = []
    for k0, (m12, m23, m34, m41) in parsed["points"]:
        w = (
            _link_variable(m12)
            * _link_variable(m23)
            * _link_variable(m34)
            * _link_variable(m41)
        )
        flux = _berry_phase(w)
        results.append({"k": k0, "flux": flux, "curvature": flux / area})
    return results


def parse_berry_curvature_path(path):
    """Parse ELKPY_BERRY_PATH.OUT and compute the per-point Berry curvature
    in one call -- see parse_berry_path_overlaps()/compute_berry_curvature_path()."""
    return compute_berry_curvature_path(parse_berry_path_overlaps(path))
