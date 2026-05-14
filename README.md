# 3D Curvilinear Grid Generation (Python + C)

Python implementation of 3D structured curvilinear grid generation with C backend for compute-intensive kernels.

## Overview

This project provides three grid generation methods and a post-processing toolkit for 3D structured curvilinear grids:

| Method | Algorithm | MPI | Use Case |
|--------|-----------|-----|----------|
| **Parabolic** | Predict-correct + Thomas solver | 1D (y-direction) | Simple marching from one boundary surface to another |
| **Hyperbolic** | Block Thomas (3x3 blocks) | None | Orthogonality + volume conservation |
| **Elliptic** | SOR iteration (Dirichlet/Higenstock) | 3D Cartesian | Smooth interior with boundary control |
| **Post-process** | Tri-linear sampling, arc stretching | None | Grid merge, resample, quality check |

## Architecture

```
Python (control + I/O + MPI) + C (compute kernels via ctypes)
```

- **Python**: Configuration parsing (JSON), memory allocation (NumPy), MPI communication (mpi4py), NetCDF I/O, coordinate permutation
- **C backend** (`libgrid3d.so`): Predict/correct, Thomas solver, SOR update, TFI interpolation, quality metrics, sampling

## Directory Structure

```
3d_grid_generate_python/
├── common/                  # Shared modules
│   ├── bdry_operations.py   #   Boundary extend/stretch
│   ├── grid_quality.py      #   Quality metrics (ctypes)
│   └── io_operations.py     #   NetCDF I/O
├── parabolic/               # Parabolic method
│   ├── generate_grid/       #   Main driver + MPI
│   └── prep_grid_bdry/      #   Boundary data scripts (x/y/z)
├── hyperbolic/              # Hyperbolic method
│   ├── generate_grid/
│   └── prep_grid_bdry/      #   Boundary data scripts (x/y/z)
├── elliptic/                # Elliptic method
│   ├── generate_grid/       #   Main driver + MPI + Dirichlet/Higenstock
│   └── prep_grid_bdry/      #   Boundary + export scripts
├── grid_post_process/       # Post-processing (merge, sample, stretch)
├── plotting/                # Visualization
│   ├── draw_grid_func.py    #   Core read/assemble functions
│   ├── draw_grid.py         #   Grid slice & 3D surface plots
│   └── draw_quality.py      #   Quality metric plots
├── src_c/                   # C backend
│   ├── parabolic.c/.h
│   ├── hyperbolic.c/.h
│   ├── elliptic.c/.h
│   ├── post_process.c/.h
│   ├── quality.c/.h         #   Includes cal_ratio_c (aspect ratio)
│   ├── lib_math.c/.h
│   └── build.sh
└── docs/                    # Documentation + verification
    ├── user_manual.pdf/tex  #   Full user manual
    ├── compare_verify.py    #   C vs Python+C comparison
    ├── verify_merge.py      #   Merge verification
    ├── quick_verify.py      #   Quick precision check
    └── run_verify_all.sh    #   One-click verification
```

## Installation

### Prerequisites

- Python 3.8+
- GCC (C99 support)
- MPI (OpenMPI or MPICH)
- Python packages: `numpy`, `netCDF4`, `mpi4py`, `numba`

### Build C library

```bash
cd src_c && bash build.sh
```

### Install Python dependencies

```bash
pip install numpy netCDF4 mpi4py numba
```

## Usage

### 1. Prepare boundary data

```bash
cd parabolic/prep_grid_bdry
python creat_bdry_z.py    # Generate boundary surfaces
python creat_step_norm.py # Generate step sizes
```

### 2. Generate grid

```bash
# Parabolic (MPI)
cd parabolic/generate_grid
bash start.sh

# Hyperbolic (serial)
cd hyperbolic/generate_grid
bash start.sh

# Elliptic (MPI)
cd elliptic/generate_grid
bash start.sh
```

### 3. Post-processing

```bash
cd grid_post_process
bash start.sh
```

## Configuration

All methods use JSON configuration files. Key parameters:

### Common
- `number_of_grid_points_x/y/z`: Grid dimensions
- `direction`: Marching direction (`"x"`, `"y"`, `"z"`)
- `geometry_input_file`: Boundary geometry file
- `grid_export_dir`: Output directory
- `grid_check`: Enable quality checks (0/1)

### Parabolic
- `number_of_mpiprocs`: MPI processes
- `coef`: Clustering coefficient
- `t2b`: March from top to bottom (0/1)

### Hyperbolic
- `coef`: Smoothing coefficient
- `flag_stretch`: Arc-length stretching (0/1)
- `bdry_x_type`, `bdry_y_type`: Boundary condition type

### Elliptic
- `number_of_mpiprocs_x/y/z`: 3D MPI decomposition
- `grid_method`: `"linear_tfi"`, `"elli_diri"`, or `"elli_higen"`
- `coef[6]`, `iter_err`, `max_iter`: Solver parameters

## Output Format

Grid coordinates are exported as NetCDF4 files with variables `x`, `y`, `z` (float32, shape `[k, j, i]`) and global attributes:
- `global_index_of_first_physical_points`: `[gni1, gnj1, gnk1]`
- `count_of_physical_points`: `[ni, nj, nk]`

## Grid Quality Metrics

11 quality metrics available:
- 3 orthogonality angles (xi-et, xi-zt, et-zt)
- Jacobian
- 3 step sizes (xi, et, zt)
- 3 smoothness ratios (xi, et, zt)
- 1 aspect ratio

## Plotting

Visualization scripts are in `plotting/`. Parameters are set directly in each script (not via command-line arguments):

```bash
cd plotting

# Draw grid slice (edit cfs_file, subs/subc/subt in script first)
python draw_grid.py

# Draw quality metric (edit varnm, cfs_file, subs/subc/subt in script first)
python draw_quality.py
```

Key parameters in each script:
- `cfs_file`: Path to config JSON of the grid to visualize
- `subs/subc/subt`: Start index / count / stride for x, y, z (`subc[n]=1` selects a plane)
- `flag_km`: Convert coordinates to km (0/1)
- `varnm` (quality only): Quality variable name (e.g. `orth_xiet`, `step_zt`, `ratio`)

## Verification

```bash
cd docs

# Run all verification tests (parabolic, hyperbolic, elliptic)
bash run_verify_all.sh

# Run merge verification only
python verify_merge.py

# Quick precision check
python quick_verify.py
```

Verification compares Python+C output against the original C code output, reporting coordinate precision and quality metric agreement.

## License

See the original 3D C code repository for license information.
