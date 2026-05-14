"""Draw 3D grid visualization.

Reads MPI-partitioned NetCDF coordinate files and generates
2D slice plots of the grid. Parameters are set directly below.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from draw_grid_func import locate_coord, gather_coord

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -------------------------- parameters input -------------------------- %
# file and path name
cfs_file = '../elliptic/output/config.json'
#cfs_file = '../hyperbolic/output/config.json'
#cfs_file = '../parabolic/output/config.json'
#cfs_file = '../grid-post-process/output/config.json'

# which grid profile to plot
# subc[n]=1 selects a plane: subc[0]=1 -> YOZ, subc[1]=1 -> XOZ, subc[2]=1 -> XOY
subs = [0, 0, 0]
subc = [-1, 1, -1]   # '-1' to plot all points in this dimension
subt = [1, 1, 1]

# figure control parameters
flag_km = 1
flag_print = 1

# Check parameter file exists
if not os.path.exists(cfs_file):
    raise FileNotFoundError(f"config file {cfs_file} does not exist")

# Read parameters file
with open(cfs_file, 'r') as f:
    cfgs = json.load(f)

# -----------------------------------------------------------
# -- load coord
# -----------------------------------------------------------
coordinfo = locate_coord(cfgs, subs, subc, subt)
if not coordinfo:
    raise RuntimeError(f"No coordinate NetCDF files found in {cfgs['grid_export_dir']}")

x, y, z = gather_coord(coordinfo, cfgs['grid_export_dir'])

# - set coord unit
if flag_km:
    x = x / 1e3
    y = y / 1e3
    z = z / 1e3
    str_unit = 'km'
else:
    str_unit = 'm'

# -----------------------------------------------------------
# -- set figure
# -----------------------------------------------------------
fig, ax = plt.subplots()
fig.set_facecolor('white')

if subc[0] == 1:
    # YOZ plane
    ax.plot(y, z, 'k-', linewidth=0.1)
    ax.plot(y.T, z.T, 'k-', linewidth=0.1)
    ax.set_xlabel(f'Y axis ({str_unit})', fontsize=10)
    ax.set_ylabel(f'Z axis ({str_unit})', fontsize=10)
    gridtitle = 'YOZ-Grid'
elif subc[1] == 1:
    # XOZ plane
    ax.plot(x, z, 'k-', linewidth=0.1)
    ax.plot(x.T, z.T, 'k-', linewidth=0.1)
    ax.set_xlabel(f'X axis ({str_unit})', fontsize=10)
    ax.set_ylabel(f'Z axis ({str_unit})', fontsize=10)
    gridtitle = 'XOZ-Grid'
elif subc[2] == 1:
    # XOY plane
    ax.plot(x, y, 'k-', linewidth=0.1)
    ax.plot(x.T, y.T, 'k-', linewidth=0.1)
    ax.set_xlabel(f'X axis ({str_unit})', fontsize=10)
    ax.set_ylabel(f'Y axis ({str_unit})', fontsize=10)
    gridtitle = 'XOY-Grid'

ax.set_axis_on()
ax.tick_params(labelsize=10)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('normal')

ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

if flag_print:
    plt.savefig('grid.png', dpi=300, bbox_inches='tight', facecolor='white')
