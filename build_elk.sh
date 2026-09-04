#!/usr/bin/env bash
# Build Elk out-of-tree, per docs/design.md #8: vendor/elk/ is never built in
# place. Copies vendor/elk/ to build/elk/, applies patches/ (if any), drops
# in build-config/make.inc, resolves the toolchain flags for THIS machine, and
# builds serially.
#
# Serial `make` is deliberate, not a mistake: Elk's src/Makefile has an
# implicit ordering dependency (mpi_stub.f90 must compile before modmpi.f90,
# libxcifc_stub.f90 before modxcifc.f90/moddftu.f90, etc.) that isn't
# expressed as explicit make prerequisites, so `make -j` races and fails
# nondeterministically. This is an upstream Elk issue -- not something to fix
# by editing vendor/elk/.
#
# Toolchain resolution (added because build-config/make.inc's workstation
# defaults do not survive an HPC cluster -- Aalto's Triton being the concrete
# case). Three environment overrides, each taking precedence over detection:
#
#   ELKPY_F90_LIB   link line, verbatim (e.g. "-lmkl_rt -lfftw3 -lfftw3f").
#                   Link-tested and a failure is fatal: an explicit request is
#                   never silently replaced by a guess.
#   ELKPY_MARCH     -march/-mtune flags, verbatim; a bare name such as
#                   "znver3" is taken as "-march=znver3".
#   ELKPY_MODULES   environment modules to load if the default link line does
#                   not work (default: "openblas fftw").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build/elk"
MAKE_INC="$ROOT/build-config/make.inc"

# ---------------------------------------------------------------------------
# Toolchain resolution
# ---------------------------------------------------------------------------

# Read a variable's value out of build-config/make.inc (last assignment wins,
# matching make's own semantics).
makeinc_value() {
    sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$MAKE_INC" | tail -1
}

F90="$(makeinc_value F90)"
F90="${F90:-gfortran}"

# A link test needs real symbol references or the linker never has to resolve
# the libraries at all and every candidate "works". These are the actual
# externals Elk pulls in: LAPACK/BLAS eigensolvers plus both precisions of
# FFTW (vendor/elk/src/{z,c}fftifc_fftw.f90 -- which use no fftw threading, so
# -lfftw3_omp is deliberately not required).
PROBE_DIR="$(mktemp -d)"
trap 'rm -rf "$PROBE_DIR"' EXIT
cat > "$PROBE_DIR/probe.f90" <<'PROBE'
program elkpy_link_probe
external zgemm,zheevx,zhegvx,dsyevd
external dfftw_plan_dft,dfftw_execute,dfftw_destroy_plan
external sfftw_plan_dft,sfftw_execute,sfftw_destroy_plan
call zgemm
call zheevx
call zhegvx
call dsyevd
call dfftw_plan_dft
call dfftw_execute
call dfftw_destroy_plan
call sfftw_plan_dft
call sfftw_execute
call sfftw_destroy_plan
end program
PROBE

PROBE_ERR=""
# probe_link "<link flags>" -- true if a program referencing the symbols above
# links with those flags in the CURRENT environment.
probe_link() {
    local libs="$1"
    if PROBE_ERR="$("$F90" -o "$PROBE_DIR/probe" "$PROBE_DIR/probe.f90" $libs 2>&1)"; then
        return 0
    fi
    return 1
}

have_modules() {
    [ -n "${LMOD_CMD:-}" ] || [ -n "${MODULESHOME:-}" ] || type module >/dev/null 2>&1
}

# Load environment modules in THIS shell. Never pipe `module` -- it is a shell
# function, and a pipe runs it in a subshell where the environment changes it
# makes are discarded. Lmod's eval'd output also trips `set -u`.
load_modules() {
    local mods="$1" rc=0
    set +u
    module load $mods > "$PROBE_DIR/module.log" 2>&1 || rc=$?
    set -u
    return $rc
}

# What ended up loaded, on one line, for the build record. `module -t list`
# writes to stderr and must not be piped in a way that loses the environment --
# reading its output in a subshell is fine, only `module load` must not be.
loaded_modules() {
    if have_modules; then
        # `|| true`: grep exits 1 if it filters everything out, which under
        # `set -e` would abort the build at the very last step.
        (set +u; module -t list 2>&1) | grep -v ':$' | tr '\n' ' ' || true
    fi
}

F90_LIB_DEFAULT="$(makeinc_value F90_LIB)"
F90_LIB=""
LIB_SOURCE=""

if [ -n "${ELKPY_F90_LIB:-}" ]; then
    if probe_link "$ELKPY_F90_LIB"; then
        F90_LIB="$ELKPY_F90_LIB"
        LIB_SOURCE="ELKPY_F90_LIB"
    else
        echo "ERROR: ELKPY_F90_LIB does not link:" >&2
        echo "    $ELKPY_F90_LIB" >&2
        echo "$PROBE_ERR" | sed 's/^/    /' >&2
        exit 1
    fi
elif probe_link "$F90_LIB_DEFAULT"; then
    F90_LIB="$F90_LIB_DEFAULT"
    LIB_SOURCE="build-config/make.inc"
else
    echo "build-config/make.inc's link line does not work here:"
    echo "    $F90_LIB_DEFAULT"
    echo "$PROBE_ERR" | sed 's/^/    /'
    echo "Falling back to detection."

    # On a cluster the libraries usually exist but are not on the default
    # search path until the relevant environment modules are loaded.
    if have_modules; then
        mods="${ELKPY_MODULES:-openblas fftw}"
        echo "Environment modules detected; loading: $mods"
        if load_modules "$mods"; then
            echo "  loaded ($(module -t list 2>&1 | tr '\n' ' '))"
        else
            echo "  WARNING: 'module load $mods' failed; continuing without it"
            sed 's/^/    /' "$PROBE_DIR/module.log" 2>/dev/null
        fi
    fi

    # Candidates, most-likely first. The default is retried because a module
    # load may be all that was missing.
    candidates=(
        "$F90_LIB_DEFAULT"
        "-lopenblas -lfftw3 -lfftw3f"
        "-lopenblas -llapack -lfftw3 -lfftw3f"
        "-lflexiblas -lfftw3 -lfftw3f"
        "-llapack -lblas -lfftw3 -lfftw3f"
        "-lmkl_rt -lfftw3 -lfftw3f"
    )
    if command -v pkg-config >/dev/null 2>&1; then
        for blas in openblas flexiblas lapack blas; do
            if pkg-config --exists "$blas" 2>/dev/null; then
                candidates+=("$(pkg-config --libs "$blas" fftw3 fftw3f 2>/dev/null)")
            fi
        done
    fi

    for cand in "${candidates[@]}"; do
        [ -n "$cand" ] || continue
        if probe_link "$cand"; then
            F90_LIB="$cand"
            LIB_SOURCE="auto-detected"
            break
        fi
    done

    if [ -z "$F90_LIB" ]; then
        cat >&2 <<'MSG'
ERROR: could not find a working BLAS/LAPACK + FFTW3 link line.

Elk needs LAPACK/BLAS (OpenBLAS bundles both) and FFTW3 in double AND single
precision. Fixes, in order of likelihood:

  * On a cluster, load the modules first, e.g.
        module load openblas fftw
    (set ELKPY_MODULES if yours are named differently, and note that
    build_elk.sh already tried "openblas fftw" itself.)

  * On Debian/Ubuntu:
        sudo apt-get install gfortran libopenblas-dev libfftw3-dev

  * Otherwise state the link line explicitly, e.g.
        export ELKPY_F90_LIB="-L/path/to/lib -lopenblas -lfftw3 -lfftw3f"
MSG
        exit 1
    fi
fi

# Runtime search path. Without this the binary links but dies with
# "libopenblas.so.0: cannot open shared object file" the moment it runs
# somewhere the build-time modules are not loaded -- a batch job, or
# launcher.py spawning `elk` from a plain subprocess. LIBRARY_PATH is what
# environment modules manipulate, and it carries more than the numerics: on a
# module-provided gcc it also points at that compiler's own libgfortran/libgomp,
# which the binary needs and the system compiler's copy may not satisfy.
rpaths=""
seen=":"
for dir in $(printf '%s' "${LIBRARY_PATH:-}" | tr ':' '\n'); do
    [ -n "$dir" ] && [ -d "$dir" ] || continue
    case "$seen" in *":$dir:"*) continue ;; esac
    seen="$seen$dir:"
    rpaths="$rpaths -Wl,-rpath,$dir"
done
# Any -L in the chosen link line deserves the same treatment.
set -- $F90_LIB
for tok in "$@"; do
    case "$tok" in
        -L*) dir="${tok#-L}" ;;
        *) continue ;;
    esac
    [ -n "$dir" ] && [ -d "$dir" ] || continue
    case "$seen" in *":$dir:"*) continue ;; esac
    seen="$seen$dir:"
    rpaths="$rpaths -Wl,-rpath,$dir"
done
F90_LIB="$F90_LIB$rpaths"

# Instruction set. -march=native is right on a workstation and wrong on a
# cluster, where the login node is routinely newer than the compute nodes: on
# Triton it resolves to skylake-avx512 on login4 while the default batch
# partition is Broadwell and others are Haswell or Zen3, so a "successful"
# build dies with SIGILL in the first job. Haswell is the baseline Aalto's own
# module tree targets and every current partition supports it; Elk's hot loops
# are inside BLAS anyway, and OpenBLAS dispatches on the actual CPU at runtime.
if [ -n "${ELKPY_MARCH:-}" ]; then
    case "$ELKPY_MARCH" in
        -*) march_flags="$ELKPY_MARCH" ;;
        *) march_flags="-march=$ELKPY_MARCH" ;;
    esac
    march_source="ELKPY_MARCH"
elif have_modules; then
    march_flags="-march=haswell -mtune=generic"
    march_source="portable default (environment modules detected -> assuming a cluster)"
else
    march_flags="-march=native -mtune=native"
    march_source="build-config/make.inc default"
fi

# ... but only if the compiler actually accepts them. This is one check for two
# failure modes: a non-x86 machine (where -march=haswell is meaningless) and a
# gfortran too old to know the name.
if ! "$F90" $march_flags -c "$PROBE_DIR/probe.f90" -o "$PROBE_DIR/probe.o" >/dev/null 2>&1; then
    echo "NOTE: $F90 rejects '$march_flags'; building without -march/-mtune."
    march_flags=""
    march_source="none ($F90 rejected the requested -march)"
fi

F90_OPTS_DEFAULT="$(makeinc_value F90_OPTS)"
# Substitute rather than replace, so hand edits to the other flags survive.
F90_OPTS="$(printf '%s' "$F90_OPTS_DEFAULT" \
    | sed -E 's/(^| )-m(arch|tune)=[^ ]*//g' \
    | sed -E 's/  +/ /g; s/^ //; s/ $//')"
# Not `[ -n ... ] && F90_OPTS=...`: as the final command of a statement that
# would return 1 under `set -e`, which exits the script.
if [ -n "$march_flags" ]; then
    F90_OPTS="$F90_OPTS $march_flags"
fi

echo "Toolchain:"
echo "  F90      = $F90"
echo "  F90_OPTS = $F90_OPTS      [$march_source]"
echo "  F90_LIB  = $F90_LIB      [$LIB_SOURCE]"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

rm -rf "$BUILD_DIR"
mkdir -p "$ROOT/build"
cp -r "$ROOT/vendor/elk" "$BUILD_DIR"

shopt -s nullglob
patches=("$ROOT"/patches/*.patch)
if [ ${#patches[@]} -gt 0 ]; then
    for patch in "${patches[@]}"; do
        echo "Applying $patch"
        patch -d "$BUILD_DIR" -p1 < "$patch"
    done
fi

cp "$MAKE_INC" "$BUILD_DIR/make.inc"

# Append rather than rewrite: make takes the last assignment, so the copied
# make.inc stays readable as "the defaults, then what this machine needed".
loaded="$(loaded_modules)"
cat >> "$BUILD_DIR/make.inc" <<EOF

# --- appended by build_elk.sh on $(date -u '+%Y-%m-%dT%H:%M:%SZ') --------------
# Resolved for this machine ($(uname -n)). Later assignments win in make, so
# these override the defaults above. See build_elk.sh for the detection order
# and the ELKPY_F90_LIB / ELKPY_MARCH / ELKPY_MODULES overrides.
# F90_OPTS: $march_source
# F90_LIB:  $LIB_SOURCE
# Modules loaded at build time: ${loaded:-<none>}
F90_OPTS = $F90_OPTS
F90_LIB = $F90_LIB
EOF

echo "Building (serially -- see comment above) ..."
make -C "$BUILD_DIR" all

echo "Built: $BUILD_DIR/src/elk"
