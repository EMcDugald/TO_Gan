import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class DataNotFoundError(Exception):
    pass


class CondVoxelDataset(torch.utils.data.Dataset):
    """
    Matches the trainer: loads (voxel_arr, cond_vec, label_val, cond_str)
    object-array .npy and rescales X to [-1, 1].
    """
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise DataNotFoundError(f"File not found: {filepath}")
        data = np.load(filepath, allow_pickle=True)
        voxels = [x[0] for x in data]
        conds = [x[1] for x in data]
        labels = [x[2] for x in data]
        cond_strs = [x[3] for x in data]

        X = torch.tensor(np.stack(voxels), dtype=torch.float32)
        C = torch.tensor(np.stack(conds), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        self.cond_strs = cond_strs

        if X.ndim == 4:
            X = X.unsqueeze(1)
        # match trainer: scale to [-1, 1]
        X = X * 2 - 1

        self.X = X
        self.C = C
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.C[idx], self.y[idx]


class CondGenerator3d(nn.Module):
    """
    Same architecture as trainer’s CondGenerator3d.
    """
    def __init__(self, nz, ngf, output_shape, cond_dim, cond_embed_dim=32):
        super().__init__()
        D, H, W = output_shape
        self.cond_fc = nn.Linear(cond_dim, cond_embed_dim)
        self.init_shape = (ngf * 8, D // 16, H // 16, W // 16)
        self.fc = nn.Linear(nz + cond_embed_dim, int(np.prod(self.init_shape)))
        self.main = nn.Sequential(
            nn.BatchNorm3d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf), nn.ReLU(True),
            nn.ConvTranspose3d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, c):
        c_emb = torch.relu(self.cond_fc(c))
        x = self.fc(torch.cat([z, c_emb], dim=1))
        return self.main(x.view(x.size(0), *self.init_shape))


def parse_hparams_txt(checkpoint_dir):
    hp_path = os.path.join(checkpoint_dir, 'hparams.txt')
    hparams = {}
    with open(hp_path, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=', 1)
                hparams[k.strip()] = v.strip()
    return hparams


def resolve_meta_path(data_path, meta_path=None):
    if meta_path is not None:
        return meta_path
    if data_path.endswith('.npy'):
        return data_path[:-4] + '_meta.npz'
    raise ValueError('Could not infer meta path; please provide --meta-path')


def get_bin_centers(edges):
    edges = np.asarray(edges, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def decode_condition_overlay(cond_vec, meta):
    """
    Use the same decoding as the trainer: fine mode only for now.
    """
    cond_vec = np.asarray(cond_vec, dtype=np.float32)
    conditioning_mode = str(meta['conditioning_mode'])
    max_bc_points = int(meta['bc_count_max'])

    bc_points = []
    load_point = None
    load_dir = None

    if conditioning_mode == 'fine':
        bc_flat_len = max_bc_points * 3
        mask_len = max_bc_points

        bc_flat = cond_vec[:bc_flat_len]
        bc_mask = cond_vec[bc_flat_len:bc_flat_len + mask_len]
        bc_count = int(round(float(cond_vec[bc_flat_len + mask_len])))

        lp_start = bc_flat_len + mask_len + 1
        load_point = cond_vec[lp_start:lp_start + 3]
        load_dir = cond_vec[lp_start + 3:lp_start + 6]

        bc_arr = bc_flat.reshape(max_bc_points, 3)
        valid = bc_mask > 0.5
        bc_points = bc_arr[valid][:bc_count]

    elif conditioning_mode == 'coarse':
        # Coarse decoding is included for completeness, but the script
        # is intended for fine mode; you can assert on mode if desired.
        x_edges = np.asarray(meta['spatial_bin_edges_x'], dtype=np.float64)
        y_edges = np.asarray(meta['spatial_bin_edges_y'], dtype=np.float64)
        z_edges = np.asarray(meta['spatial_bin_edges_z'], dtype=np.float64)
        x_centers = get_bin_centers(x_edges)
        y_centers = get_bin_centers(y_edges)
        z_centers = get_bin_centers(z_edges)

        pos = 0
        pos += 2  # bc_x_rng
        pos += 2  # bc_y_rng
        pos += 2  # bc_z_rng

        bc_bins_flat = cond_vec[pos:pos + max_bc_points * 3]
        pos += max_bc_points * 3

        bc_mask = cond_vec[pos:pos + max_bc_points]
        pos += max_bc_points

        bc_count = int(round(float(cond_vec[pos])))
        pos += 1

        load_bins = cond_vec[pos:pos + 3]
        pos += 3

        load_dir = cond_vec[pos:pos + 3]

        bc_bins = bc_bins_flat.reshape(max_bc_points, 3)
        bc_bins = bc_bins[bc_mask > 0.5][:bc_count]

        bc_points = []
        for bx, by, bz in bc_bins:
            bx = int(np.clip(round(float(bx)), 0, len(x_centers) - 1))
            by = int(np.clip(round(float(by)), 0, len(y_centers) - 1))
            bz = int(np.clip(round(float(bz)), 0, len(z_centers) - 1))
            bc_points.append([x_centers[bx], y_centers[by], z_centers[bz]])
        bc_points = np.asarray(bc_points, dtype=np.float32)

        lbx = int(np.clip(round(float(load_bins[0])), 0, len(x_centers) - 1))
        lby = int(np.clip(round(float(load_bins[1])), 0, len(y_centers) - 1))
        lbz = int(np.clip(round(float(load_bins[2])), 0, len(z_centers) - 1))
        load_point = np.array([x_centers[lbx], y_centers[lby], z_centers[lbz]], dtype=np.float32)

    else:
        raise ValueError(f"Unknown conditioning_mode: {conditioning_mode}")

    return {
        "bc_points": np.asarray(bc_points, dtype=np.float32),
        "load_point": None if load_point is None else np.asarray(load_point, dtype=np.float32),
        "load_dir": None if load_dir is None else np.asarray(load_dir, dtype=np.float32),
        "conditioning_mode": conditioning_mode,
    }


def plot_voxel_grid_3d(voxel_grid, title='', save_path=None,
                       bc_points=None, load_point=None, load_vec=None):
    """
    Slightly simplified plotter: 2 views, reference vs ensemble panels
    can call this for each voxel.
    """
    voxel_grid = np.asarray(voxel_grid)
    nx, ny, nz = voxel_grid.shape

    bc_plot = None
    if bc_points is not None and len(bc_points) > 0:
        bc_points = np.asarray(bc_points, dtype=np.float64)
        bc_plot = np.column_stack([
            bc_points[:, 0] * (nx - 1),
            bc_points[:, 1] * (ny - 1),
            bc_points[:, 2] * (nz - 1),
        ])

    load_plot = None
    if load_point is not None:
        load_point = np.asarray(load_point, dtype=np.float64)
        load_plot = np.array([
            load_point[0] * (nx - 1),
            load_point[1] * (ny - 1),
            load_point[2] * (nz - 1),
        ], dtype=np.float64)

    fig = plt.figure(figsize=(10, 4))
    views = [(25, 35, 'View 1'), (25, 125, 'View 2')]

    for k, (elev, azim, subtitle) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, k, projection='3d')
        ax.voxels(voxel_grid > 0.1, edgecolor='k', linewidth=0.2, alpha=0.72)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)

        if bc_plot is not None:
            ax.scatter(
                bc_plot[:, 0], bc_plot[:, 1], bc_plot[:, 2],
                c='red', s=36, marker='o', edgecolors='white', linewidths=0.6,
                depthshade=False, label='BC'
            )

        if load_plot is not None:
            ax.scatter(
                [load_plot[0]], [load_plot[1]], [load_plot[2]],
                c='dodgerblue', s=64, marker='^', edgecolors='white', linewidths=0.7,
                depthshade=False, label='Load point'
            )
            if load_vec is not None and np.linalg.norm(load_vec) > 0:
                load_vec = np.asarray(load_vec, dtype=np.float64)
                scale = max(nx, ny, nz) * 0.18
                ax.quiver(
                    load_plot[0], load_plot[1], load_plot[2],
                    load_vec[0], load_vec[1], load_vec[2],
                    color='dodgerblue', linewidth=2.0, length=scale, normalize=True
                )

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(subtitle)
        ax.set_box_aspect((nx, ny, nz))
        ax.set_axis_off()

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=220)
    plt.close(fig)


def resolve_ckpt_path(checkpoint_dir, ckpt_id):
    if ckpt_id == 'best':
        path = os.path.join(checkpoint_dir, 'ckpt_best.pt')
    elif ckpt_id == 'final':
        path = os.path.join(checkpoint_dir, 'ckpt_final.pt')
    else:
        path = os.path.join(checkpoint_dir, f'ckpt_step_{ckpt_id}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def main():
    parser = argparse.ArgumentParser(
        description='Sample non-UNet 3D conditional GAN (fine mode) with ensembles.'
    )
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Trainer checkpoint directory (contains ckpt_*.pt and hparams.txt).')
    parser.add_argument('--ckpt-id', type=str, default='best',
                        help="'best', 'final', or a specific step number, e.g. 1000.")
    parser.add_argument('--data-path', type=str, default=None,
                        help='Optional override of data_path; default from hparams.txt.')
    parser.add_argument('--meta-path', type=str, default=None,
                        help='Optional override of meta path; default: data_path -> *_meta.npz.')
    parser.add_argument('--M', type=int, default=4,
                        help='Number of conditions to sample from training data.')
    parser.add_argument('--N', type=int, default=4,
                        help='Number of ensemble samples per condition.')
    parser.add_argument('--use-positives-only', action='store_true',
                        help='If set, sample conditions only from label=1 (positive) class.')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    hparams = parse_hparams_txt(args.checkpoint_dir)

    data_path = args.data_path if args.data_path is not None else hparams['data_path']
    meta_path = resolve_meta_path(data_path, args.meta_path)
    meta = np.load(meta_path, allow_pickle=True)

    conditioning_mode = str(meta['conditioning_mode'])
    if conditioning_mode != 'fine':
        print(f'Warning: conditioning_mode is {conditioning_mode}, '
              f'but this sampler is designed for fine mode.')

    ## v3 parserf 
    nz_line = hparams.get('nz', '')
    nz, ngf = None, None

    parts = [p.strip() for p in nz_line.split(',')]

    # First part is just the value for nz, because the key was already stripped off
    # when parse_hparams_txt did line.split('=', 1).
    if len(parts) >= 1 and parts[0] != '':
        nz = int(parts[0])

    # Remaining parts still look like "ngf = 256", "ndf = 128"
    for part in parts[1:]:
        if '=' in part:
            k, v = [x.strip() for x in part.split('=', 1)]
            if k == 'ngf':
                ngf = int(v)

    if nz is None:
        raise ValueError(f"Could not parse nz from hparams['nz']={nz_line!r}")
    if ngf is None:
        raise ValueError(f"Could not parse ngf from hparams['nz']={nz_line!r}")

    print("Sampler nz:", nz, "ngf:", ngf)

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    dataset = CondVoxelDataset(data_path)
    X = dataset.X    # [-1,1]
    C = dataset.C
    y = dataset.y
    cond_strs = dataset.cond_strs

    if args.use_positives_only:
        mask = (y == 1)
    else:
        mask = torch.ones_like(y, dtype=torch.bool)

    # indices of eligible conditions
    idx_all = torch.nonzero(mask, as_tuple=False).view(-1)
    if idx_all.numel() == 0:
        raise RuntimeError('No eligible samples found under the chosen mask.')

    M = min(args.M, idx_all.numel())
    # random subset of conditions
    perm = torch.randperm(idx_all.numel())[:M]
    idx_sel = idx_all[perm]

    shape3d = X.shape[2:]
    cond_dim = C.shape[1]

    print("nz:", nz, "ngf:", ngf, "cond_dim from data:", cond_dim)
    netG = CondGenerator3d(nz, ngf, shape3d, cond_dim).to(device)
    ckpt_path = resolve_ckpt_path(args.checkpoint_dir, args.ckpt_id)
    ckpt = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(ckpt['netG_state'])
    netG.eval()

    # output directory
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    outdir = os.path.join(args.checkpoint_dir, f'samples_fine_M{M}_N{args.N}_{args.ckpt_id}_{ts}')
    os.makedirs(outdir, exist_ok=True)

    # allocate arrays for saving ensembles:
    # raw: [-1,1], binary: {0,1}
    D, H, W = shape3d
    samples_raw = np.zeros((M, args.N, D, H, W), dtype=np.float32)
    samples_bin = np.zeros((M, args.N, D, H, W), dtype=np.float32)
    ref_voxels_raw = np.zeros((M, D, H, W), dtype=np.float32)   # reference training voxel [-1,1]
    ref_voxels_bin = np.zeros((M, D, H, W), dtype=np.float32)   # thresholded reference

    cond_vectors = np.zeros((M, cond_dim), dtype=np.float32)
    cond_labels = np.zeros((M,), dtype=np.int64)
    cond_indices = np.zeros((M,), dtype=np.int64,)

    manifest_path = os.path.join(outdir, 'manifest.txt')
    with open(manifest_path, 'w') as fman:
        fman.write(
            '# ci  train_idx  label  ref_raw_npy  ref_bin_npy  '
            'ensemble_raw_npy  ensemble_bin_npy  figure_png  cond_str\n'
        )

        for ci, idx in enumerate(idx_sel.tolist()):
            idx_int = int(idx)
            cond_indices[ci] = idx_int

            x_ref = X[idx_int]  # shape (1,D,H,W)
            c_ref = C[idx_int]
            y_ref = int(y[idx_int].item())
            cond_str = cond_strs[idx_int]

            # store cond vector and label
            cond_vectors[ci] = c_ref.cpu().numpy()
            cond_labels[ci] = y_ref

            # reference voxel as np arrays
            # x_ref is in [-1,1], x_ref[0] is (D,H,W)
            ref_raw = x_ref[0].cpu().numpy()
            ref_voxels_raw[ci] = ref_raw
            ref_bin = (ref_raw > 0.0).astype(np.float32)
            ref_voxels_bin[ci] = ref_bin

            # decode BC/load overlays
            overlay = decode_condition_overlay(cond_vectors[ci], meta)
            bc_points = overlay["bc_points"]
            load_point = overlay["load_point"]
            load_dir = overlay["load_dir"]

            # generate ensemble
            c_tensor = c_ref.to(device).unsqueeze(0)  # (1,cond_dim)

            for si in range(args.N):
                z = torch.randn(1, nz, device=device)
                with torch.no_grad():
                    fake = netG(z, c_tensor).cpu().numpy()[0, 0]  # (D,H,W)

                samples_raw[ci, si] = fake
                samples_bin[ci, si] = (fake > 0.0).astype(np.float32)

            # # save per-condition comparison figure
            # fig_path = os.path.join(outdir, f'condition_{ci:03d}_ensemble.png')

            # # simple figure: reference (raw+binary) and a subset of ensemble members (binary)
            # fig, axes = plt.subplots(2, max(2, min(args.N, 4)), figsize=(12, 6))
            # axes = np.atleast_2d(axes)

            # # reference raw: central slice
            # mid_z = D // 2
            # im0 = axes[0, 0].imshow(ref_raw[:, :, mid_z], cmap='viridis', vmin=-1.0, vmax=1.0)
            # axes[0, 0].set_title('Ref raw (mid-z)')
            # axes[0, 0].axis('off')
            # fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

            # # reference binary
            # axes[1, 0].imshow(ref_bin[:, :, mid_z], cmap='gray', vmin=0.0, vmax=1.0)
            # axes[1, 0].set_title('Ref bin (mid-z)')
            # axes[1, 0].axis('off')

            # # some ensemble members (binary, mid-z)
            # for j in range(1, max(2, min(args.N, 4))):
            #     if j - 1 >= args.N:
            #         axes[0, j].axis('off')
            #         axes[1, j].axis('off')
            #         continue
            #     s_bin = samples_bin[ci, j - 1]
            #     axes[0, j].imshow(s_bin[:, :, mid_z], cmap='gray', vmin=0.0, vmax=1.0)
            #     axes[0, j].set_title(f'Sample {j-1} bin')
            #     axes[0, j].axis('off')

            #     # optional: show raw as well (same slice)
            #     s_raw = samples_raw[ci, j - 1]
            #     imj = axes[1, j].imshow(s_raw[:, :, mid_z], cmap='viridis', vmin=-1.0, vmax=1.0)
            #     axes[1, j].set_title(f'Sample {j-1} raw')
            #     axes[1, j].axis('off')
            #     fig.colorbar(imj, ax=axes[1, j], fraction=0.046, pad=0.04)

            # fig.suptitle(f'Condition {ci} | idx={idx_int} | label={y_ref}\n{cond_str}')
            # fig.tight_layout()
            # fig.savefig(fig_path, dpi=200, bbox_inches='tight')
            # plt.close(fig)

            # # Optionally, also save a full 3D plot of the reference using the overlay
            # ref_3d_path = os.path.join(outdir, f'condition_{ci:03d}_ref3d.png')
            # plot_voxel_grid_3d(
            #     ref_bin,
            #     title=f'Ref condition {ci} | idx={idx_int}',
            #     save_path=ref_3d_path,
            #     bc_points=bc_points,
            #     load_point=load_point,
            #     load_vec=load_dir,
            # )

            # fman.write(
            #     f'{ci:03d}  {idx_int:06d}  {y_ref}  '
            #     f'{os.path.basename(ref_3d_path)}  '
            #     f'samples_raw_Mci.npy  samples_bin_Mci.npy  '
            #     f'{os.path.basename(fig_path)}  '
            #     f'{cond_str}\n'
            # )

            # 3D multiview plot of reference (binary field)
            ref_3d_path = os.path.join(outdir, f'condition_{ci:03d}_ref3d.png')
            plot_voxel_grid_3d(
                ref_bin,
                title=f'Ref condition {ci} | idx={idx_int} | label={y_ref}',
                save_path=ref_3d_path,
                bc_points=bc_points,
                load_point=load_point,
                load_vec=load_dir,
            )

            # 3D multiview plots for each ensemble member (binary and optional raw)
            sample_bin_paths = []
            sample_raw_paths = []

            for si in range(args.N):
                samp_bin = samples_bin[ci, si]
                samp_raw = samples_raw[ci, si]

                samp3d_bin_path = os.path.join(
                    outdir, f'condition_{ci:03d}_sample_{si:03d}_bin3d.png'
                )
                plot_voxel_grid_3d(
                    samp_bin,
                    title=f'Cond {ci} sample {si} | binary',
                    save_path=samp3d_bin_path,
                    bc_points=bc_points,
                    load_point=load_point,
                    load_vec=load_dir,
                )
                sample_bin_paths.append(os.path.basename(samp3d_bin_path))

                # Optional: 3D view of raw field thresholded internally by plot_voxel_grid_3d
                samp3d_raw_path = os.path.join(
                    outdir, f'condition_{ci:03d}_sample_{si:03d}_raw3d.png'
                )
                plot_voxel_grid_3d(
                    samp_raw,
                    title=f'Cond {ci} sample {si} | raw>0.1 view',
                    save_path=samp3d_raw_path,
                    bc_points=bc_points,
                    load_point=load_point,
                    load_vec=load_dir,
                )
                sample_raw_paths.append(os.path.basename(samp3d_raw_path))

            sample_bin_list = ";".join(sample_bin_paths)

            fman.write(
                f'{ci:03d}  {idx_int:06d}  {y_ref}  '
                f'{os.path.basename(ref_3d_path)}  '
                f'samples_raw.npy  samples_bin.npy  '
                f'sample_bin_pngs={sample_bin_list}  '
                f'{cond_str}\n'
            )

    # Save arrays and run-level metadata as npz/npy
    np.save(os.path.join(outdir, 'samples_raw.npy'), samples_raw)
    np.save(os.path.join(outdir, 'samples_bin.npy'), samples_bin)
    np.save(os.path.join(outdir, 'ref_voxels_raw.npy'), ref_voxels_raw)
    np.save(os.path.join(outdir, 'ref_voxels_bin.npy'), ref_voxels_bin)
    np.save(os.path.join(outdir, 'cond_vectors.npy'), cond_vectors)
    np.save(os.path.join(outdir, 'cond_labels.npy'), cond_labels)
    np.save(os.path.join(outdir, 'cond_indices.npy'), cond_indices)

    meta_run = {
        'checkpoint_dir': args.checkpoint_dir,
        'ckpt_id': args.ckpt_id,
        'data_path': data_path,
        'meta_path': meta_path,
        'M': int(M),
        'N': int(args.N),
        'nz': int(nz),
        'ngf': int(ngf),
        'device': str(device),
        'conditioning_mode': conditioning_mode,
        'use_positives_only': bool(args.use_positives_only),
    }
    np.savez(os.path.join(outdir, 'run_meta.npz'), **meta_run)

    print(f'Wrote ensembles and figures to {outdir}')


if __name__ == '__main__':
    main()