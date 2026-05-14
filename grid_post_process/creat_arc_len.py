"""Create arc-length file for 3D grid stretching.

Generates step length distribution along a specified direction
and exports it as a text file for use by the arc-length stretching
functions in common.algebra.
"""

import os
import numpy as np


def create_arc_len(nz, step=None, flag_flip=True, flag_gradient=False,
                   incre_layer=12, max_ratio=3, dense_start=10,
                   file_name='./arc_len_file1.txt', flag_printf=True):
    """Create arc-length file for grid stretching.

    Args:
        nz: number of grid points in the stretching direction.
        step: pre-computed step array. If None, uniform or gradient steps are generated.
        flag_flip: flip step array (for top-dense grids).
        flag_gradient: use gradient stretching instead of uniform.
        incre_layer: number of layers for gradient transition.
        max_ratio: maximum step ratio for gradient.
        dense_start: starting index for dense layer.
        file_name: output file path.
        flag_printf: whether to generate a plot.

    Returns:
        arc_len: arc-length array of shape (nz,).
    """
    num_of_step = nz - 1

    if step is None:
        step = np.ones(num_of_step)

        if flag_gradient:
            incre_ratio = np.exp(np.log(max_ratio) / incre_layer)
            step[:dense_start] = 1.0

            start_value = step[dense_start - 1]
            geom_seq = start_value * (incre_ratio ** np.arange(1, incre_layer + 1))
            step[dense_start:dense_start + incre_layer] = geom_seq

            last_value = step[dense_start + incre_layer - 1]
            step[dense_start + incre_layer:] = last_value

    if flag_flip:
        step = np.flip(step)

    sum_step = np.sum(step)
    step_nor = step / sum_step

    arc_len = np.zeros(nz)
    for i in range(1, nz):
        arc_len[i] = arc_len[i - 1] + step_nor[i - 1]

    if (arc_len[nz - 1] - 1) > 1e-8:
        raise ValueError("step set is error, please check and reset")

    with open(file_name, 'w') as fid:
        fid.write(f"# number of points is {nz}\n")
        for i in range(nz):
            fid.write(f'{arc_len[i]:.9e}\n')

    if flag_printf:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plot_dir = os.path.join(os.path.dirname(file_name), '..', 'plot')
        os.makedirs(plot_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, nz + 1), arc_len, 'b-', linewidth=1.5)
        ax.set_xlabel('Point index', fontweight='bold')
        ax.set_ylabel('Arc length', fontweight='bold')
        ax.set_title('Arc length distribution', fontweight='bold')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'arc_len.png'),
                    dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'Saved: {os.path.join(plot_dir, "arc_len.png")}')

    print(f"nz is {nz}")
    print(f"Arc length file saved to: {file_name}")

    return arc_len


if __name__ == '__main__':
    # Default: uniform stretching for z direction
    nz = 201
    create_arc_len(nz, flag_flip=True, flag_gradient=False)
