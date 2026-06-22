import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def sanitize_for_filename(s, max_len=180):
    s = re.sub(r'[^A-Za-z0-9._=-]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s[:max_len]


def parse_load_arr(load_arr):
    ld = np.asarray(load_arr, dtype=np.float64)
    if ld.ndim == 3:
        ld = ld[0]
    if ld.ndim != 2 or ld.shape[0] < 1 or ld.shape[1] < 6:
        raise ValueError(f"Unexpected load array shape: {ld.shape}")
    return ld[0, :3], ld[0, 3:6]


def parse_bc_arr(bc_arr):
    bc = np.asarray(bc_arr, dtype=np.float64)
    if bc.ndim != 2 or bc.shape[1] < 3:
        raise ValueError(f"Unexpected BC array shape: {bc.shape}")
    return bc[:, :3]


def compute_quantile_edges(values, n_bins):
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-8
    return edges.astype(np.float64)


def bin_index(x, edges):
    idx = int(np.searchsorted(edges, x, side='right') - 1)
    n_bins = len(edges) - 1
    return max(0, min(idx, n_bins - 1))


def build_spatial_edges(bcs, loads, indices, n_spatial_bins):
    all_bc = np.concatenate([parse_bc_arr(bcs[i]) for i in indices], axis=0)
    all_load = np.stack([parse_load_arr(loads[i])[0] for i in indices], axis=0)
    xyz_all = np.concatenate([all_bc, all_load], axis=0)
    return tuple(compute_quantile_edges(xyz_all[:, ax], n_spatial_bins) for ax in range(3))


def make_condition_key(idx, bc_arr, load_arr, conditioning_mode, spatial_edges=None):
    bc_pts = parse_bc_arr(bc_arr)
    load_pt, load_vec = parse_load_arr(load_arr)

    if conditioning_mode == 'fine':
        bc_str = '_'.join([f'bc{i}({p[0]:.4f},{p[1]:.4f},{p[2]:.4f})' for i, p in enumerate(bc_pts)])
        load_pt_str = f'loadPt({load_pt[0]:.4f},{load_pt[1]:.4f},{load_pt[2]:.4f})'
        load_vec_str = f'loadVec({load_vec[0]:.4f},{load_vec[1]:.4f},{load_vec[2]:.4f})'
        key = f'{conditioning_mode}_idx{idx}_bcCount{len(bc_pts)}_{bc_str}_{load_pt_str}_{load_vec_str}'
    elif conditioning_mode == 'coarse':
        if spatial_edges is None:
            raise ValueError('spatial_edges required for coarse condition key')
        bx = [bin_index(p[0], spatial_edges[0]) for p in bc_pts]
        by = [bin_index(p[1], spatial_edges[1]) for p in bc_pts]
        bz = [bin_index(p[2], spatial_edges[2]) for p in bc_pts]
        lx = bin_index(load_pt[0], spatial_edges[0])
        ly = bin_index(load_pt[1], spatial_edges[1])
        lz = bin_index(load_pt[2], spatial_edges[2])
        bc_rng = (
            f'bcx[{bc_pts[:,0].min():.4f},{bc_pts[:,0].max():.4f}]_'
            f'bcy[{bc_pts[:,1].min():.4f},{bc_pts[:,1].max():.4f}]_'
            f'bcz[{bc_pts[:,2].min():.4f},{bc_pts[:,2].max():.4f}]'
        )
        bc_bins = '_'.join([f'bc{i}b({bx[i]},{by[i]},{bz[i]})' for i in range(len(bc_pts))])
        load_bins = f'loadBin({lx},{ly},{lz})'
        load_vec_str = f'loadVec({load_vec[0]:.4f},{load_vec[1]:.4f},{load_vec[2]:.4f})'
        key = f'{conditioning_mode}_idx{idx}_bcCount{len(bc_pts)}_{bc_rng}_{bc_bins}_{load_bins}_{load_vec_str}'
    else:
        raise ValueError("conditioning_mode must be 'fine' or 'coarse'")

    return sanitize_for_filename(key)


def draw_coarse_boxes(ax, bc_pts, spatial_edges, shape):
    nx, ny, nz = shape
    x_edges, y_edges, z_edges = spatial_edges
    colors = ['red', 'darkred', 'firebrick', 'indianred', 'maroon', 'salmon']
    for i, p in enumerate(bc_pts):
        xb = bin_index(p[0], x_edges)
        yb = bin_index(p[1], y_edges)
        zb = bin_index(p[2], z_edges)
        x0, x1 = x_edges[xb] * (nx - 1), x_edges[xb + 1] * (nx - 1)
        y0, y1 = y_edges[yb] * (ny - 1), y_edges[yb + 1] * (ny - 1)
        z0, z1 = z_edges[zb] * (nz - 1), z_edges[zb + 1] * (nz - 1)
        c = colors[i % len(colors)]
        lines = [
            ([x0, x1], [y0, y0], [z0, z0]), ([x0, x1], [y1, y1], [z0, z0]),
            ([x0, x1], [y0, y0], [z1, z1]), ([x0, x1], [y1, y1], [z1, z1]),
            ([x0, x0], [y0, y1], [z0, z0]), ([x1, x1], [y0, y1], [z0, z0]),
            ([x0, x0], [y0, y1], [z1, z1]), ([x1, x1], [y0, y1], [z1, z1]),
            ([x0, x0], [y0, y0], [z0, z1]), ([x1, x1], [y0, y0], [z0, z1]),
            ([x0, x0], [y1, y1], [z0, z1]), ([x1, x1], [y1, y1], [z0, z1]),
        ]
        for xs, ys, zs in lines:
            ax.plot(xs, ys, zs, color=c, linestyle='--', linewidth=1.2, alpha=0.8)


def plot_voxel_with_conditions(binary_arr, save_path, title='', bc_arr=None, load_arr=None, conditioning_mode='fine', spatial_edges=None):
    binary_arr = np.asarray(binary_arr)
    nx, ny, nz = binary_arr.shape

    bc_pts = parse_bc_arr(bc_arr) if bc_arr is not None else None
    load_pt, load_vec = parse_load_arr(load_arr) if load_arr is not None else (None, None)

    bc_plot = None
    load_plot = None
    if bc_pts is not None:
        bc_plot = np.column_stack([
            bc_pts[:, 0] * (nx - 1),
            bc_pts[:, 1] * (ny - 1),
            bc_pts[:, 2] * (nz - 1),
        ])
    if load_pt is not None:
        load_plot = np.array([
            load_pt[0] * (nx - 1),
            load_pt[1] * (ny - 1),
            load_pt[2] * (nz - 1),
        ], dtype=np.float64)

    fig = plt.figure(figsize=(18, 6))
    views = [(25, 35, 'View 1'), (25, 125, 'View 2'), (65, 35, 'View 3')]

    for k, (elev, azim, subtitle) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, k, projection='3d')
        ax.voxels(binary_arr > 0, edgecolor='k', linewidth=0.15, alpha=0.72)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)

        if bc_plot is not None:
            ax.scatter(
                bc_plot[:, 0], bc_plot[:, 1], bc_plot[:, 2],
                c='red', s=36, marker='o', edgecolors='white', linewidths=0.6,
                depthshade=False, label='BC'
            )
            for i, p in enumerate(bc_plot):
                ax.text(p[0], p[1], p[2], f'BC{i}', color='red', fontsize=8)

        if conditioning_mode == 'coarse' and bc_pts is not None and spatial_edges is not None:
            draw_coarse_boxes(ax, bc_pts, spatial_edges, (nx, ny, nz))

        if load_plot is not None:
            ax.scatter(
                [load_plot[0]], [load_plot[1]], [load_plot[2]],
                c='dodgerblue', s=64, marker='^', edgecolors='white', linewidths=0.7,
                depthshade=False, label='Load point'
            )
            if load_vec is not None:
                scale = max(nx, ny, nz) * 0.18
                ax.quiver(
                    load_plot[0], load_plot[1], load_plot[2],
                    load_vec[0], load_vec[1], load_vec[2],
                    color='dodgerblue', linewidth=2.0, length=scale, normalize=True
                )
            ax.text(load_plot[0], load_plot[1], load_plot[2], 'Load', color='dodgerblue', fontsize=8)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(subtitle)
        ax.set_box_aspect((nx, ny, nz))
        ax.set_axis_off()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize voxel samples with BC/load overlays for the new padded BC/load conditioning datasets.')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--shape', type=int, nargs=3, default=[32, 32, 32])
    parser.add_argument('--conditioning-mode', type=str, choices=['fine', 'coarse'], default='fine')
    parser.add_argument('--n-spatial-bins', type=int, default=10)
    parser.add_argument('--max-plots', type=int, default=200)
    parser.add_argument('--rng-seed', type=int, default=0)
    args = parser.parse_args()

    topo = np.load(os.path.join(args.data_root, 'topologies.npy'), allow_pickle=True)
    shapes = np.load(os.path.join(args.data_root, 'shapes.npy'), allow_pickle=True)
    bcs = np.load(os.path.join(args.data_root, 'boundary_conditions.npy'), allow_pickle=True)
    loads = np.load(os.path.join(args.data_root, 'loads.npy'), allow_pickle=True)

    target_shape = tuple(args.shape)
    indices = [i for i, shp in enumerate(shapes) if tuple(shp) == target_shape]
    if not indices:
        raise ValueError(f'No entries with shape {target_shape}')

    os.makedirs(args.outdir, exist_ok=True)

    spatial_edges = None
    if args.conditioning_mode == 'coarse':
        spatial_edges = build_spatial_edges(bcs, loads, indices, args.n_spatial_bins)

    if len(indices) > args.max_plots:
        rng = np.random.default_rng(args.rng_seed)
        plot_indices = list(rng.choice(indices, size=args.max_plots, replace=False))
    else:
        plot_indices = indices

    catalog_path = os.path.join(args.outdir, 'condition_key_catalog.txt')
    with open(catalog_path, 'w') as f:
        f.write(f'target_shape = {target_shape}\n')
        f.write(f'conditioning_mode = {args.conditioning_mode}\n')
        f.write(f'n_spatial_bins = {args.n_spatial_bins}\n')
        if spatial_edges is not None:
            f.write(f'spatial_edges_x = {spatial_edges[0].tolist()}\n')
            f.write(f'spatial_edges_y = {spatial_edges[1].tolist()}\n')
            f.write(f'spatial_edges_z = {spatial_edges[2].tolist()}\n')
        f.write('\nindex\tfilename\tcondition_key\n')

        for j, idx in enumerate(plot_indices, start=1):
            voxel_arr = np.asarray(topo[idx]).astype(np.int8).reshape(target_shape)
            key = make_condition_key(idx, bcs[idx], loads[idx], args.conditioning_mode, spatial_edges)
            fname = f'condition_key_{j:04d}_{key}.png'
            out_png = os.path.join(args.outdir, fname)
            title = f'idx={idx} | {key}'
            plot_voxel_with_conditions(
                voxel_arr,
                out_png,
                title=title,
                bc_arr=bcs[idx],
                load_arr=loads[idx],
                conditioning_mode=args.conditioning_mode,
                spatial_edges=spatial_edges,
            )
            f.write(f'{idx}\t{fname}\t{key}\n')

    print(f'Saved {len(plot_indices)} plots to {args.outdir}')
    print(f'Saved catalog to {catalog_path}')


if __name__ == '__main__':
    main()