# elkpy-managed make.inc for building the vendored Elk source.
#
# This file is NOT part of vendor/elk/ — it lives here so vendor/elk/ stays
# byte-for-byte upstream (see docs/design.md #8). The build step copies
# vendor/elk/ to build/elk/, applies any patches/ series, drops this file in
# as build/elk/make.inc, then runs `make` there.
#
# Configuration: GNU Fortran, OpenBLAS (which bundles LAPACK), FFTW3 (double +
# single precision), no MPI (serial build). Revisit if MPI parallelism is
# needed.
#
# These are the DEFAULTS, tuned for a workstation with system-packaged
# libraries. build_elk.sh link-tests F90_LIB before using it and, if it fails
# (e.g. on an HPC cluster where the libraries live behind environment
# modules), falls back to auto-detection and APPENDS the values it settled on
# to the copied build/elk/make.inc — so that file is always a self-contained
# record of what was actually built. See docs/design.md #8 for the override
# environment variables (ELKPY_F90_LIB, ELKPY_MARCH, ELKPY_MODULES).

MAKE = make
AR = ar

SRC_MKL = mkl_stub.f90
SRC_MPI = mpi_stub.f90
SRC_FFT = zfftifc_fftw.f90 cfftifc_fftw.f90
SRC_LIBXC = libxcifc_stub.f90
SRC_W90S = w90_stub.f90

F90 = gfortran
F90_OPTS = -O3 -march=native -mtune=native -fopenmp -ffpe-summary=none

# No -llapack: OpenBLAS bundles a full LAPACK, so a separate reference LAPACK
# is redundant on a Debian/Ubuntu/RHEL workstation and actively absent in some
# Spack-built OpenBLAS installations (Aalto's Triton being the case that
# prompted this) — asking for it there is a hard "cannot find -llapack".
F90_LIB = -lopenblas -lfftw3 -lfftw3f
