#!/usr/bin/env bash
# Build Elk out-of-tree, per docs/design.md #8: vendor/elk/ is never built in
# place. Copies vendor/elk/ to build/elk/, applies patches/ (if any), drops
# in build-config/make.inc, and builds serially.
#
# Serial `make` is deliberate, not a mistake: Elk's src/Makefile has an
# implicit ordering dependency (mpi_stub.f90 must compile before modmpi.f90,
# libxcifc_stub.f90 before modxcifc.f90/moddftu.f90, etc.) that isn't
# expressed as explicit make prerequisites, so `make -j` races and fails
# nondeterministically. This is an upstream Elk issue -- not something to fix
# by editing vendor/elk/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build/elk"

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

cp "$ROOT/build-config/make.inc" "$BUILD_DIR/make.inc"

echo "Building (serially -- see comment above) ..."
make -C "$BUILD_DIR" all

echo "Built: $BUILD_DIR/src/elk"
