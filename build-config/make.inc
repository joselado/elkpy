# elkpy-managed make.inc for building the vendored Elk source.
#
# This file is NOT part of vendor/elk/ — it lives here so vendor/elk/ stays
# byte-for-byte upstream (see docs/design.md #8). The build step copies
# vendor/elk/ to build/elk/, applies any patches/ series, drops this file in
# as build/elk/make.inc, then runs `make` there.
#
# Configuration: GNU Fortran, OpenBLAS + LAPACK, FFTW3 (double + single
# precision), no MPI (serial build). Revisit if MPI parallelism is needed.

MAKE = make
AR = ar

SRC_MKL = mkl_stub.f90
SRC_MPI = mpi_stub.f90
SRC_FFT = zfftifc_fftw.f90 cfftifc_fftw.f90
SRC_LIBXC = libxcifc_stub.f90
SRC_W90S = w90_stub.f90

F90 = gfortran
F90_OPTS = -O3 -march=native -mtune=native -fopenmp -ffpe-summary=none
F90_LIB = -lopenblas -llapack -lfftw3 -lfftw3f
