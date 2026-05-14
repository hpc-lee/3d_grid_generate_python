"""Generate normalized step sizes for 3D parabolic grid generation."""

import numpy as np

nz = 21
num_of_step = nz - 1
flag_gradient = 0
flag_flip = 1
file_name = './step_file_3d.txt'

step = np.zeros(num_of_step)
if flag_gradient:
    incre_layer = 12
    max_ratio = 3
    incre_ratio = np.exp((np.log(max_ratio) / incre_layer))

    step[:10] = 1

    start_value = step[9]
    geom_seq = start_value * (incre_ratio ** np.arange(1, incre_layer + 1))
    step[10:10 + incre_layer] = geom_seq

    last_value = step[10 + incre_layer - 1]
    step[10 + incre_layer:] = last_value

    if flag_flip:
        step = np.flip(step)
else:
    step[:] = 1

sum_step = np.sum(step)
step_nor = step / sum_step

if abs(np.sum(step_nor) - 1.0) > 1e-8:
    raise ValueError("Step normalization error")

with open(file_name, 'w') as fid:
    fid.write(f"# number of step, num_of_step is {num_of_step}\n")
    fid.write(f"{num_of_step}\n")
    fid.write(f"# step\n")
    for i in range(num_of_step):
        fid.write(f"{step_nor[i]:.9e}\n")

print(f"nz = {nz}, num_of_step = {num_of_step}")
print("Step file created:", file_name)
