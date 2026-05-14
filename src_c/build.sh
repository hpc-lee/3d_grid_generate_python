#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building libgrid3d.so ..."

gcc -Wall -Wno-unused-parameter -O3 -std=c99 -fPIC -shared \
    -o libgrid3d.so \
    parabolic.c \
    hyperbolic.c \
    elliptic.c \
    post_process.c \
    quality.c \
    lib_math.c \
    -lm

echo "Build successful: libgrid3d.so"
ls -lh libgrid3d.so
