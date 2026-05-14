"""Draw 3D grid quality metric visualization.

Reads MPI-partitioned NetCDF coordinate and quality files,
generates 2D slice plots of quality metrics. Parameters are set directly below.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from draw_grid_func import locate_coord, gather_coord, gather_quality

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

# variable to plot
# 'orth_xiet', 'orth_xizt', 'orth_etzt', 'jacobi',
# 'smooth_xi', 'smooth_et', 'smooth_zt',
# 'step_xi', 'step_et', 'step_zt', 'ratio'
varnm = 'step_zt'

# which grid profile to plot
# subc[n]=1 selects a plane: subc[0]=1 -> YOZ, subc[1]=1 -> XOZ, subc[2]=1 -> XOY
subs = [0, 0, 0]
subc = [-1, 1, -1]   # '-1' to plot all points in this dimension
subt = [1, 1, 1]

# figure control parameters
flag_km = 1
flag_print = 1
flag_title = 0

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
var_data = gather_quality(coordinfo, cfgs['grid_export_dir'], varnm)

print(f"{varnm} max value is {np.max(var_data)}")
print(f"{varnm} min value is {np.min(var_data)}")

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
plt.set_cmap('jet')

if subc[0] == 1:
    # YOZ plane
    im = ax.pcolormesh(y, z, var_data, shading='gouraud')
    ax.set_xlabel(f'Y axis ({str_unit})', fontsize=10)
    ax.set_ylabel(f'Z axis ({str_unit})', fontsize=10)
    gridtitle = 'YOZ-Grid'
elif subc[1] == 1:
    # XOZ plane
    im = ax.pcolormesh(x, z, var_data, shading='gouraud')
    ax.set_xlabel(f'X axis ({str_unit})', fontsize=10)
    ax.set_ylabel(f'Z axis ({str_unit})', fontsize=10)
    gridtitle = 'XOZ-Grid'
elif subc[2] == 1:
    # XOY plane
    im = ax.pcolormesh(x, y, var_data, shading='gouraud')
    ax.set_xlabel(f'X axis ({str_unit})', fontsize=10)
    ax.set_ylabel(f'Y axis ({str_unit})', fontsize=10)
    gridtitle = 'XOY-Grid'

ax.set_axis_on()
ax.tick_params(labelsize=10)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('normal')

ax.set_aspect('equal', adjustable='box')
cbar = plt.colorbar(im, ax=ax)

if flag_title:
    ax.set_title(gridtitle)

plt.tight_layout()
plt.show()

if flag_print:
    plt.savefig('quality.png', dpi=300, bbox_inches='tight', facecolor='white')
