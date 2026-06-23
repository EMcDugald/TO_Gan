#!/usr/bin/env python
import argparse
import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class DataNotFoundError(Exception):
    pass


class CondVoxelDataset(torch.utils.data.Dataset):
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
        X = X * 2 - 1

        self.X = X
        self.C = C
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.C[idx], self.y[idx]


class CondGenerator3d(nn.Module):
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


def parse_cond_vector(s):
    if s.strip().startswith('['):
        return np.array(ast.literal_eval(s), dtype=np.float32)
    return np.array([float(x) for x in s.split(',')], dtype=np.float32)


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


def load_meta_for_data(data_path, hparams):
    meta_path = hparams.get('meta_path', None)
    if meta_path is None:
        meta_path = data_path[:-4] + '_meta.npz'
    return np.load(meta_path, allow_pickle=True)


def get_bin_centers(edges):
    edges = np.asarray(edges, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def decode_condition_overlay(cond_vec, meta):
    cond_vec = np.asarray(cond_vec, dtype=np.float32)
    conditioning_mode = str(meta['conditioning_mode'])
    max_bc_points = int(meta['bc_count_max'])

    if conditioning_mode == 'fine':
        bc_flat_len = max_bc_points * 3
        mask_len = max_bc_points
        bc_flat = cond_vec[:bc_flat_len]
        bc_mask = cond_vec[bc_flat_len:bc_flat_len + mask_len]
        bc_count = int(round(float(cond_vec[bc_flat_len + mask_len])))
        load_point = cond_vec[bc_flat_len + mask_len + 1:bc_flat_len + mask_len + 4]
        load_dir = cond_vec[bc_flat_len + mask_len + 4:bc_flat_len + mask_len + 7]
        bc_arr = bc_flat.reshape(max_bc_points, 3)
        bc_points = bc_arr[bc_mask > 0.5][:bc_count]
    else:
        x_edges = np.asarray(meta['spatial_bin_edges_x'], dtype=np.float64)
        y_edges = np.asarray(meta['spatial_bin_edges_y'], dtype=np.float64)
        z_edges = np.asarray(meta['spatial_bin_edges_z'], dtype=np.float64)
        x_centers = get_bin_centers(x_edges)
        y_centers = get_bin_centers(y_edges)
        z_centers = get_bin_centers(z_edges)

        pos = 0
        pos += 2
        pos += 2
        pos += 2
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

    return {
        'bc_points': np.asarray(bc_points, dtype=np.float32),
        'load_point': np.asarray(load_point, dtype=np.float32),
        'load_dir': np.asarray(load_dir, dtype=np.float32),
        'conditioning_mode': conditioning_mode,
    }


def plot_voxel_grid_3d(voxel_grid, title='', save_path=None, bc_points=None, load_point=None, load_dir=None):
    voxel_grid = np.asarray(voxel_grid)
    nx, ny, nz = voxel_grid.shape

    fig = plt.figure(figsize=(18, 6))
    views = [(25, 35, 'View 1'), (25, 125, 'View 2'), (65, 35, 'View 3')]

    bc_plot = None
    if bc_points is not None and len(bc_points) > 0:
        bc_points = np.asarray(bc_points, dtype=np.float32)
        bc_plot = np.column_stack([
            bc_points[:, 0] * (nx - 1),
            bc_points[:, 1] * (ny - 1),
            bc_points[:, 2] * (nz - 1),
        ])

    load_plot = None
    if load_point is not None:
        load_plot = np.array([
            load_point[0] * (nx - 1),
            load_point[1] * (ny - 1),
            load_point[2] * (nz - 1),
        ], dtype=np.float32)

    for k, (elev, azim, subtitle) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, k, projection='3d')
        ax.voxels(voxel_grid > 0.1, edgecolor='k', linewidth=0.15, alpha=0.72)
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

        if load_plot is not None:
            ax.scatter(
                [load_plot[0]], [load_plot[1]], [load_plot[2]],
                c='dodgerblue', s=64, marker='^', edgecolors='white', linewidths=0.7,
                depthshade=False, label='Load point'
            )
            if load_dir is not None and np.linalg.norm(load_dir) > 0:
                scale = max(nx, ny, nz) * 0.18
                ax.quiver(
                    load_plot[0], load_plot[1], load_plot[2],
                    load_dir[0], load_dir[1], load_dir[2],
                    color='dodgerblue', linewidth=2.0, length=scale, normalize=True
                )
            ax.text(load_plot[0], load_plot[1], load_plot[2], 'Load', color='dodgerblue', fontsize=8)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(subtitle)
        ax.set_box_aspect((nx, ny, nz))
        ax.set_axis_off()

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', type=str, required=True)
    ap.add_argument('--ckpt-id', type=str, default='best')
    ap.add_argument('--num-cond-vectors', type=int, default=4)
    ap.add_argument('--samples-per-cond', type=int, default=4)
    ap.add_argument('--condition-vector', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    hparams = parse_hparams_txt(args.checkpoint_dir)
    data_path = hparams['data_path']
    meta = load_meta_for_data(data_path, hparams)

    nz_line = hparams.get('nz', '')
    nz, ngf = None, None
    for part in nz_line.split(','):
        if '=' in part:
            k, v = [x.strip() for x in part.split('=')]
            if k == 'nz':
                nz = int(v)
            elif k == 'ngf':
                ngf = int(v)
    nz = 300 if nz is None else nz
    ngf = 256 if ngf is None else ngf

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    dataset = CondVoxelDataset(data_path)
    pos_mask = (dataset.y == 1)
    C_P = dataset.C[pos_mask]
    shape3d = dataset.X.shape[2:]
    cond_dim = C_P.shape[1]

    netG = CondGenerator3d(nz, ngf, shape3d, cond_dim).to(device)
    ckpt = torch.load(resolve_ckpt_path(args.checkpoint_dir, args.ckpt_id), map_location=device)
    netG.load_state_dict(ckpt['netG_state'])
    netG.eval()

    outdir = os.path.join(args.checkpoint_dir, f'samples_{args.ckpt_id}')
    os.makedirs(outdir, exist_ok=True)

    if args.condition_vector is not None:
        cond_vectors = [parse_cond_vector(args.condition_vector)]
        cond_descs = ['user_supplied']
    else:
        idx = torch.randint(low=0, high=C_P.shape[0], size=(min(args.num_cond_vectors, C_P.shape[0]),))
        cond_vectors = C_P[idx].cpu().numpy()
        cond_descs = [dataset.cond_strs[int(i)] for i in idx.cpu().numpy()]

    manifest_path = os.path.join(outdir, 'sample_conditions.txt')
    with open(manifest_path, 'w') as f:
        f.write('# c_idx  s_idx  filename  condition_vector  condition_string  bc_points  load_point  load_dir\n')
        for ci, (c_vec, c_desc) in enumerate(zip(cond_vectors, cond_descs)):
            if len(c_vec) != cond_dim:
                raise ValueError(f'Condition dim mismatch: got {len(c_vec)}, expected {cond_dim}')
            overlay = decode_condition_overlay(c_vec, meta)
            c_tensor = torch.tensor(c_vec, dtype=torch.float32, device=device).unsqueeze(0)

            for si in range(args.samples_per_cond):
                z = torch.randn(1, nz, device=device)
                with torch.no_grad():
                    fake = netG(z, c_tensor).cpu().numpy()[0, 0]

                filename = f'sample_c{ci:03d}_s{si:03d}.png'
                save_path = os.path.join(outdir, filename)
                plot_voxel_grid_3d(
                    fake,
                    title=f'non-UNet sample c{ci} s{si}',
                    save_path=save_path,
                    bc_points=overlay['bc_points'],
                    load_point=overlay['load_point'],
                    load_dir=overlay['load_dir'],
                )

                cond_str_num = ' '.join(f'{v:.6f}' for v in c_vec)
                bc_str = ';'.join([f'({p[0]:.4f},{p[1]:.4f},{p[2]:.4f})' for p in overlay['bc_points']])
                lp = overlay['load_point']
                ld = overlay['load_dir']
                lp_str = f'({lp[0]:.4f},{lp[1]:.4f},{lp[2]:.4f})'
                ld_str = f'({ld[0]:.4f},{ld[1]:.4f},{ld[2]:.4f})'
                f.write(f'{ci:03d}  {si:03d}  {filename}  {cond_str_num}  {c_desc}  {bc_str}  {lp_str}  {ld_str}\n')


if __name__ == '__main__':
    main()