"""Constants used across the 3D grid generation project."""

import numpy as np

PI = np.float32(np.pi)

# Direction type enumeration
X_DIRE = 0
Y_DIRE = 1
Z_DIRE = 2

DIRE_MAP = {"x": X_DIRE, "y": Y_DIRE, "z": Z_DIRE}

# Method type enumeration (elliptic)
LINEAR_TFI = 0
ELLI_DIRI = 1
ELLI_HIGEN = 2

# Boundary condition type (hyperbolic)
BDRY_FLOATING = 1
BDRY_CARTESIAN = 2

# Face IDs for 6 boundaries
FACE_X1 = 0
FACE_X2 = 1
FACE_Y1 = 2
FACE_Y2 = 3
FACE_Z1 = 4
FACE_Z2 = 5
