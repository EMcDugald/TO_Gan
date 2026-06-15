import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_voxel_grid_3d(voxel_grid, title='', save_path=None):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid > 0, edgecolor='k', linewidth=0.15)
    ax.set_title(title)
    ax.set_axis_off()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot N random voxel structures from a labeled .npy dataset.')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the labeled voxel .npy file')
    parser.add_argument('--n', type=int, required=True,
                        help='Number of random structures to plot')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Directory where plots will be saved')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--replace', action='store_true',
                        help='Sample with replacement if set')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data = np.load(args.data_path, allow_pickle=True)
    total = len(data)
    if total == 0:
        raise ValueError('Dataset is empty.')

    rng = np.random.default_rng(args.seed)
    n = args.n
    if (not args.replace) and n > total:
        raise ValueError(f'Requested n={n}, but dataset only has {total} samples. Use --replace to allow repeats.')

    indices = rng.choice(total, size=n, replace=args.replace)

    for plot_idx, data_idx in enumerate(indices):
        voxel_arr, cond_vec, label_val, cond_str = data[data_idx]
        voxel_arr = np.asarray(voxel_arr)
        title = f'idx={int(data_idx)} | label={int(label_val)} | {cond_str}'
        fname = f'voxel_{plot_idx:04d}_dataidx_{int(data_idx):05d}_label_{int(label_val)}.png'
        save_path = os.path.join(args.outdir, fname)
        plot_voxel_grid_3d(voxel_arr, title=title, save_path=save_path)

    index_log = os.path.join(args.outdir, 'plotted_indices.txt')
    with open(index_log, 'w') as f:
        f.write('# plot_idx data_idx label condition_string\n')
        for plot_idx, data_idx in enumerate(indices):
            _, _, label_val, cond_str = data[data_idx]
            f.write(f'{plot_idx:04d} {int(data_idx):05d} {int(label_val)} {cond_str}\n')

    print(f'Saved {n} plots to {args.outdir}')
    print(f'Index log: {index_log}')


if __name__ == '__main__':
    main()