import os
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


class DataNotFoundError(Exception):
    pass


class CondVoxelDataset(Dataset):
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

        if X.ndim == 4:
            X = X.unsqueeze(1)
        X = X * 2.0 - 1.0

        self.X = X
        self.C = C
        self.y = y
        self.cond_strs = cond_strs

        print("Total samples:", X.shape[0])
        print("Condition dim:", C.shape[1])
        print("Positives:", int((y == 1).sum().item()), "Negatives:", int((y == 0).sum().item()))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.C[idx], self.y[idx]


def mass_fraction_batch(batch_np):
    v = (batch_np > 0).astype(np.float64)
    return v.mean(axis=(1, 2, 3))


def compactness_batch(batch_np):
    B = batch_np.shape[0]
    v = (batch_np > 0)
    comp = np.zeros(B, dtype=np.float64)
    for i in range(B):
        vi = v[i]
        total = vi.sum()
        if total == 0:
            comp[i] = 0.0
            continue
        xs = np.where(vi.any(axis=(1, 2)))[0]
        ys = np.where(vi.any(axis=(0, 2)))[0]
        zs = np.where(vi.any(axis=(0, 1)))[0]
        bbox_vol = (xs[-1] - xs[0] + 1) * (ys[-1] - ys[0] + 1) * (zs[-1] - zs[0] + 1)
        comp[i] = float(total) / float(bbox_vol)
    return comp


def score_batch_mass_compactness(batch_np, m_min, m_max, c_min, c_max):
    eps = 1e-8
    mass_fracs = mass_fraction_batch(batch_np)
    comp_vals = compactness_batch(batch_np)
    m_norm = (mass_fracs - m_min) / (m_max - m_min + eps)
    m_norm = np.clip(m_norm, 0.0, 1.0)
    c_raw_norm = (comp_vals - c_min) / (c_max - c_min + eps)
    c_raw_norm = np.clip(c_raw_norm, 0.0, 1.0)
    c_norm = 1.0 - c_raw_norm
    score = 0.5 * m_norm + 0.5 * c_norm
    return score, mass_fracs, comp_vals


def diversity_loss(x):
    r = torch.sum(x ** 2, dim=1, keepdim=True)
    D = r - 2 * torch.matmul(x, x.T) + r.T
    S = torch.exp(-0.5 * D ** 2)
    try:
        eig_val = torch.linalg.eigvalsh(S)
    except Exception:
        eig_val = torch.ones(x.size(0), device=x.device)
    loss = -torch.mean(torch.log(torch.clamp(eig_val, min=1e-7)))
    return loss


def eval_dpp_div_from_voxels(batch_np, device):
    x = torch.tensor(batch_np.reshape(batch_np.shape[0], -1), dtype=torch.float32, device=device)
    return float(diversity_loss(x).item())


class ReusableDataLoader:
    def __init__(self, X, C, batch_size, shuffle=True):
        self.X = X
        self.C = C
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(X.shape[0]))
        self.previous_indices = []

    def _shuffle_indices(self):
        self.indices = torch.randperm(len(self.indices)).tolist()

    def get_batch(self):
        queued = self.previous_indices
        while len(queued) < self.batch_size:
            if self.shuffle:
                self._shuffle_indices()
            queued.extend(self.indices)
        self.previous_indices = queued[self.batch_size:]
        batch_indices = queued[:self.batch_size]
        x_batch = torch.stack([self.X[i] for i in batch_indices])
        c_batch = torch.stack([self.C[i] for i in batch_indices])
        return x_batch, c_batch


class CondBroadcast3D(nn.Module):
    def __init__(self, cond_dim, out_channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, out_channels)

    def forward(self, c, spatial_shape):
        B = c.shape[0]
        D, H, W = spatial_shape
        x = self.proj(c).view(B, -1, 1, 1, 1)
        return x.expand(B, x.shape[1], D, H, W)


class FiLM3D(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x, c):
        gamma_beta = self.to_gamma_beta(c)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1, 1)
        beta = beta.view(x.size(0), x.size(1), 1, 1, 1)
        return x * (1.0 + gamma) + beta


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, groups=8):
        super().__init__()
        g1 = min(groups, out_ch)
        while out_ch % g1 != 0 and g1 > 1:
            g1 -= 1
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g1, out_ch)
        self.film1 = FiLM3D(cond_dim, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g1, out_ch)
        self.film2 = FiLM3D(cond_dim, out_ch)
        self.act = nn.SiLU(inplace=True)
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x, c):
        s = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.film1(x, c)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film2(x, c)
        x = self.act(x + s)
        return x


class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.block = ConvBlock3D(in_ch, out_ch, cond_dim)
        self.down = nn.Conv3d(out_ch, out_ch, 4, 2, 1, bias=False)

    def forward(self, x, c):
        x = self.block(x, c)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.block = ConvBlock3D(out_ch + skip_ch, out_ch, cond_dim)

    def forward(self, x, skip, c):
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            ds, hs, ws = skip.shape[-3:]
            x = x[:, :, :ds, :hs, :ws]
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, c)
        return x


class CondUNetGenerator3D(nn.Module):
    def __init__(self, nz, base_ch, output_shape, cond_dim, cond_channels=8):
        super().__init__()
        D, H, W = output_shape
        assert D % 8 == 0 and H % 8 == 0 and W % 8 == 0, "UNet expects dims divisible by 8"
        self.output_shape = output_shape
        self.nz = nz
        self.cond_broadcast = CondBroadcast3D(cond_dim, cond_channels)
        self.noise_to_seed = nn.Linear(nz, D * H * W)

        in_ch = 1 + cond_channels
        self.enc1 = DownBlock3D(in_ch, base_ch, cond_dim)
        self.enc2 = DownBlock3D(base_ch, base_ch * 2, cond_dim)
        self.enc3 = DownBlock3D(base_ch * 2, base_ch * 4, cond_dim)
        self.bottleneck = ConvBlock3D(base_ch * 4, base_ch * 8, cond_dim)
        self.dec3 = UpBlock3D(base_ch * 8, base_ch * 4, base_ch * 4, cond_dim)
        self.dec2 = UpBlock3D(base_ch * 4, base_ch * 2, base_ch * 2, cond_dim)
        self.dec1 = UpBlock3D(base_ch * 2, base_ch, base_ch, cond_dim)
        self.out_conv = nn.Sequential(
            nn.Conv3d(base_ch, base_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(base_ch, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, c):
        B = z.size(0)
        D, H, W = self.output_shape
        seed = self.noise_to_seed(z).view(B, 1, D, H, W)
        cond_map = self.cond_broadcast(c, self.output_shape)
        x = torch.cat([seed, cond_map], dim=1)

        x, s1 = self.enc1(x, c)
        x, s2 = self.enc2(x, c)
        x, s3 = self.enc3(x, c)
        x = self.bottleneck(x, c)
        x = self.dec3(x, s3, c)
        x = self.dec2(x, s2, c)
        x = self.dec1(x, s1, c)
        x = self.out_conv(x)
        return x


class CondDiscriminator3DImproved(nn.Module):
    def __init__(self, ndf, input_shape, cond_dim, cond_channels=8, nc=3):
        super().__init__()
        D, H, W = input_shape
        assert D % 16 == 0 and H % 16 == 0 and W % 16 == 0, "Discriminator expects dims divisible by 16"
        self.cond_broadcast = CondBroadcast3D(cond_dim, cond_channels)
        in_ch = 1 + cond_channels
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        feat_size = (ndf * 8) * (D // 16) * (H // 16) * (W // 16)
        self.cond_proj = nn.Linear(cond_dim, feat_size)
        self.fc = nn.Linear(feat_size, nc)

    def forward(self, x, c):
        cond_map = self.cond_broadcast(c, x.shape[-3:])
        x = torch.cat([x, cond_map], dim=1)
        h = self.conv(x).view(x.size(0), -1)
        proj = self.cond_proj(c)
        h = h + proj
        return self.fc(h)


def unwrap_module(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def save_checkpoint(step, netG, netD, G_opt, D_opt, path):
    G_raw = unwrap_module(netG)
    D_raw = unwrap_module(netD)
    torch.save({
        'step': step,
        'netG_state': G_raw.state_dict(),
        'netD_state': D_raw.state_dict(),
        'G_opt_state': G_opt.state_dict(),
        'D_opt_state': D_opt.state_dict(),
    }, path)


def load_checkpoint(path, netG, netD, G_opt, D_opt, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f'No checkpoint found at {path}')
    ckpt = torch.load(path, map_location=device)
    netG.load_state_dict(ckpt['netG_state'])
    netD.load_state_dict(ckpt['netD_state'])
    G_opt.load_state_dict(ckpt['G_opt_state'])
    D_opt.load_state_dict(ckpt['D_opt_state'])
    for opt in (G_opt, D_opt):
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    return netG, netD, G_opt, D_opt, ckpt.get('step', 0)


def plot_voxel_grid_3d(voxel_grid, title='', save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid > 0.1, edgecolor='k', linewidth=0.2)
    ax.set_title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def GAN_step_MDD_3d_cond_3class(
    D, G, A, D_opt, G_opt, A_opt,
    P_batch, N_batch, c_batch, noise_batch,
    batch_size, device,
    diversity_weight=0.0,
    smooth_real=0.1,
    d_update=True, use_label_smoothing=True,
    use_diversity_loss=False,
):
    criterion = nn.CrossEntropyLoss()

    if d_update:
        D.zero_grad(set_to_none=True)
        y_pos = torch.full((batch_size,), 1, dtype=torch.long, device=device)
        y_neg = torch.full((batch_size,), 2, dtype=torch.long, device=device)
        y_fake = torch.full((batch_size,), 0, dtype=torch.long, device=device)

        out_real_pos = D(P_batch, c_batch)
        if use_label_smoothing and smooth_real > 0.0:
            # Label smoothing for real positives:
            # - mostly class 1 (positive)
            # - small mass shared between class 0 (fake) and class 2 (negative)
            log_probs = torch.log_softmax(out_real_pos, dim=1)
            p_target = torch.zeros_like(out_real_pos)
            p_target[:, 1] = 1.0 - smooth_real
            p_target[:, 0] = smooth_real / 2.0
            p_target[:, 2] = smooth_real / 2.0
            L_D_real = -(p_target * log_probs).sum(dim=1).mean()
        else:
            L_D_real = criterion(out_real_pos, y_pos)

        out_real_neg = D(N_batch, c_batch)
        L_D_neg = criterion(out_real_neg, y_neg)

        fake_data_for_D = G(noise_batch, c_batch)
        out_fake = D(fake_data_for_D.detach(), c_batch)
        L_D_fake = criterion(out_fake, y_fake)

        L_D_tot = L_D_real + L_D_neg + L_D_fake
        L_D_tot.backward()

        D_grad_norm = 0.0
        for p in D.parameters():
            if p.grad is not None:
                D_grad_norm += p.grad.detach().pow(2).sum().item()
        D_grad_norm = D_grad_norm ** 0.5
        D_opt.step()
    else:
        L_D_real = torch.tensor(0.0, device=device)
        L_D_neg = torch.tensor(0.0, device=device)
        L_D_fake = torch.tensor(0.0, device=device)
        D_grad_norm = 0.0

    G.zero_grad(set_to_none=True)
    fake_data = G(noise_batch, c_batch)
    out_fake_for_G = D(fake_data, c_batch)

    y_pos = torch.full((batch_size,), 1, dtype=torch.long, device=device)
    L_G = criterion(out_fake_for_G, y_pos)

    if use_diversity_loss and diversity_weight > 0:
        feat = fake_data.view(fake_data.size(0), -1)
        L_div = diversity_loss(feat)
        L_G_tot = L_G + diversity_weight * L_div
    else:
        L_div = None
        L_G_tot = L_G

    L_G_tot.backward()
    G_grad_norm = 0.0
    for p in G.parameters():
        if p.grad is not None:
            G_grad_norm += p.grad.detach().pow(2).sum().item()
    G_grad_norm = G_grad_norm ** 0.5
    G_opt.step()

    report = {
        'L_D_real': float(L_D_real.item()),
        'L_D_neg': float(L_D_neg.item()),
        'L_D_fake': float(L_D_fake.item()),
        'L_G': float(L_G.item()),
        'D_grad_norm': float(D_grad_norm),
        'G_grad_norm': float(G_grad_norm),
    }
    if L_div is not None:
        report['L_div'] = float(L_div.item())
    return report


def train_3d_cond(D, G, A, D_opt, G_opt, A_opt,
                  P_loader, N_loader,
                  num_steps, batch_size, noise_dim,
                  train_step_fn, device,
                  diversity_weight=0.0,
                  checkpoint_dir=None, ckpt_interval=None,
                  smooth_real=0.1, smooth_fake=0.0,
                  C_P_full=None, cond_strs=None,
                  eval_every_steps=None, nz=None,
                  start_step=0, d_every=3,
                  use_label_smoothing=True,
                  use_diversity_loss=False,
                  n_vis_samples=10):
    best_G_loss = float('inf')
    steps_per_epoch = max(1, len(P_loader.X) // batch_size)
    metrics_file = None
    metrics_writer = None
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        metrics_path = os.path.join(checkpoint_dir, 'metrics.csv')
        mode = 'a' if start_step > 0 and os.path.exists(metrics_path) else 'w'
        metrics_file = open(metrics_path, mode, newline='')
        metrics_writer = csv.writer(metrics_file)
        if mode == 'w':
            metrics_writer.writerow(['step', 'epoch', 'L_D_real', 'L_D_neg', 'L_D_fake', 'L_G', 'D_grad_norm', 'G_grad_norm', 'L_div'])

    steps_range = trange(start_step, num_steps, position=0, leave=True)
    for step in steps_range:
        P_batch, cP = P_loader.get_batch()
        N_batch, cN = N_loader.get_batch()
        c_batch = cP.to(device)
        P_batch = P_batch.to(device)
        N_batch = N_batch.to(device)
        noise_batch = torch.randn(batch_size, noise_dim, device=device)
        d_update = ((step + 1) % d_every == 0)

        report = train_step_fn(
            D, G, A, D_opt, G_opt, A_opt,
            P_batch, N_batch, c_batch, noise_batch,
            batch_size, device,
            diversity_weight=diversity_weight,
            smooth_real=smooth_real,
            d_update=d_update,
            use_label_smoothing=use_label_smoothing,
            use_diversity_loss=use_diversity_loss,
        )

        steps_range.set_postfix({k: f"{v:.4f}" for k, v in report.items() if isinstance(v, float)})

        if metrics_writer is not None:
            epoch = step // steps_per_epoch
            metrics_writer.writerow([
                step + 1, epoch,
                report.get('L_D_real', float('nan')),
                report.get('L_D_neg', float('nan')),
                report.get('L_D_fake', float('nan')),
                report.get('L_G', float('nan')),
                report.get('D_grad_norm', float('nan')),
                report.get('G_grad_norm', float('nan')),
                report.get('L_div', float('nan')),
            ])

        current_G_loss = report.get('L_G', None)
        if checkpoint_dir is not None and current_G_loss is not None and current_G_loss < best_G_loss:
            best_G_loss = current_G_loss
            save_checkpoint(step + 1, G, D, G_opt, D_opt, os.path.join(checkpoint_dir, 'ckpt_best.pt'))

        if checkpoint_dir is not None and ckpt_interval is not None and (step + 1) % ckpt_interval == 0:
            save_checkpoint(step + 1, G, D, G_opt, D_opt, os.path.join(checkpoint_dir, f'ckpt_step_{step+1}.pt'))

        if (checkpoint_dir is not None and eval_every_steps is not None and C_P_full is not None and cond_strs is not None and nz is not None and (step + 1) % eval_every_steps == 0):
            current_epoch = (step + 1) // steps_per_epoch
            samples_subdir = os.path.join(checkpoint_dir, f'epoch_{current_epoch:04d}')
            os.makedirs(samples_subdir, exist_ok=True)
            G_raw = unwrap_module(G)
            G_raw.eval()
            with torch.no_grad():
                z = torch.randn(n_vis_samples, nz, device=device)
                idx_vis = torch.randint(low=0, high=C_P_full.shape[0], size=(n_vis_samples,))
                c_vis = C_P_full[idx_vis].to(device)
                fake = G_raw(z, c_vis).cpu().numpy()
                cond_np = c_vis.cpu().numpy()
                cond_strs_vis = [cond_strs[int(i)] for i in idx_vis.cpu().numpy()]
            np.save(os.path.join(samples_subdir, 'fake_conditions.npy'), cond_np)
            np.save(os.path.join(samples_subdir, 'fake_indices.npy'), idx_vis.cpu().numpy())
            txt_path = os.path.join(samples_subdir, 'fake_voxel_conditions.txt')
            with open(txt_path, 'w') as f_txt:
                f_txt.write('# idx  filename  condition_vector  condition_string\n')
                for i in range(n_vis_samples):
                    filename = f'fake_voxel_{i}.png'
                    fig_path = os.path.join(samples_subdir, filename)
                    plot_voxel_grid_3d(fake[i, 0], title=f'Fake sample {i} (epoch {current_epoch})', save_path=fig_path)
                    cond_str_num = ' '.join(f'{v:.6f}' for v in cond_np[i])
                    f_txt.write(f'{i:03d}  {filename}  {cond_str_num}  {cond_strs_vis[i]}\n')
            G_raw.train()

    if checkpoint_dir is not None:
        save_checkpoint(num_steps, G, D, G_opt, D_opt, os.path.join(checkpoint_dir, 'ckpt_final.pt'))
    if metrics_file is not None:
        metrics_file.close()
    return D, G, A


if __name__ == '__main__':
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU device count (initial):', torch.cuda.device_count())
        print('GPU device name[0]:', torch.cuda.get_device_name(0))
    else:
        print('CUDA not available')

    data_root = '/xdisk/hdb/emcdugald/to_cond_gan/train_data/323232/octant'
    data_path = os.path.join(data_root, '10000_labeled_voxels_32x32x32_octmass_score0.7.npy')
    meta_path = os.path.join(data_root, '10000_labeled_voxels_32x32x32_octmass_score0.7_meta.npz')

    meta = np.load(meta_path)
    cutoff_train = float(meta['cutoff_train'])
    score_cutoff = float(meta['score_cutoff'])
    m_min = float(meta['m_min'])
    m_max = float(meta['m_max'])
    c_min = float(meta['c_min'])
    c_max = float(meta['c_max'])

    batch_size = 32
    nz = 128
    ngf = 64
    ndf = 32
    num_epochs = 1000
    lr_D = 2e-4
    lr_G = 2e-4
    smooth_real = 0.1
    smooth_fake = 0.0
    d_every = 3
    use_label_smoothing = True
    use_diversity_loss = True
    diversity_weight = 0.01
    n_vis_samples = 5

    dataset = CondVoxelDataset(data_path)
    pos_mask = (dataset.y == 1)
    neg_mask = (dataset.y == 0)
    P = dataset.X[pos_mask]
    C_P = dataset.C[pos_mask]
    N = dataset.X[neg_mask]
    C_N = dataset.C[neg_mask]
    n_samples = P.shape[0] + N.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print('CUDA device count (inside main):', torch.cuda.device_count())

    shape3d = P.shape[2:]
    cond_dim = C_P.shape[1]

    netG = CondUNetGenerator3D(nz, ngf, shape3d, cond_dim, cond_channels=8)
    netD = CondDiscriminator3DImproved(ndf, shape3d, cond_dim, cond_channels=8, nc=3)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Generator params:",
        count_parameters(unwrap_module(netG)))
    print("Discriminator params:",
        count_parameters(unwrap_module(netD)))

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f'Using DataParallel on {torch.cuda.device_count()} GPUs')
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    netG = netG.to(device)
    netD = netD.to(device)

    P_loader = ReusableDataLoader(P, C_P, batch_size)
    N_loader = ReusableDataLoader(N, C_N, batch_size)
    num_steps = num_epochs * len(P) // batch_size

    D_opt = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.5, 0.999))
    G_opt = optim.Adam(netG.parameters(), lr=lr_G, betas=(0.5, 0.999))

    base_ckpt_root = '/xdisk/hdb/emcdugald/to_cond_gan/checkpoints_323232_octant'
    hp_name = f'unet_octmass_epochs{num_epochs}_bs{batch_size}_nz{nz}_ngf{ngf}_ndf{ndf}_nsamp{n_samples}_lrD{lr_D}_lrG{lr_G}_smoothR{smooth_real}_dEvery{d_every}_div{int(use_diversity_loss)}'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    checkpoint_dir = os.path.join(base_ckpt_root, f'{hp_name}_{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    resume_path = None
    start_step = 0
    if resume_path is not None:
        print(f'Resuming from checkpoint: {resume_path}')
        netG, netD, G_opt, D_opt, start_step = load_checkpoint(resume_path, netG, netD, G_opt, D_opt, device)
    else:
        print('Starting from scratch')

    with open(os.path.join(checkpoint_dir, 'hparams.txt'), 'w') as f_hp:
        f_hp.write(f'lr_D = {lr_D}\n')
        f_hp.write(f'lr_G = {lr_G}\n')
        f_hp.write(f'smooth_real = {smooth_real}\n')
        f_hp.write(f'smooth_fake = {smooth_fake}\n')
        f_hp.write(f'd_every = {d_every}\n')
        f_hp.write(f'use_label_smoothing = {use_label_smoothing}\n')
        f_hp.write(f'use_diversity_loss = {use_diversity_loss}\n')
        f_hp.write(f'diversity_weight = {diversity_weight}\n')
        f_hp.write(f'batch_size = {batch_size}\n')
        f_hp.write(f'num_epochs = {num_epochs}\n')
        f_hp.write(f'nz = {nz}, ngf = {ngf}, ndf = {ndf}\n')
        f_hp.write(f'n_samples = {n_samples}\n')
        f_hp.write(f'score_cutoff_quantile_train = {score_cutoff}\n')
        f_hp.write(f'score_cutoff_value_train = {cutoff_train}\n')
        f_hp.write(f'cond_dim = {cond_dim}\n')
        f_hp.write(f'data_path = {data_path}\n')
        f_hp.write('conditioning = broadcast + FiLM in G, broadcast + projection in D\n')

    steps_per_epoch = max(1, len(P) // batch_size)
    ckpt_epochs = 5
    ckpt_interval = ckpt_epochs * steps_per_epoch
    eval_every_epochs = 5
    eval_every_steps = eval_every_epochs * steps_per_epoch

    netD, netG, _ = train_3d_cond(
        netD, netG, None,
        D_opt, G_opt, None,
        P_loader, N_loader,
        num_steps, batch_size, nz,
        GAN_step_MDD_3d_cond_3class,
        device,
        diversity_weight=diversity_weight,
        checkpoint_dir=checkpoint_dir,
        ckpt_interval=ckpt_interval,
        smooth_real=smooth_real,
        smooth_fake=smooth_fake,
        C_P_full=C_P,
        cond_strs=dataset.cond_strs,
        eval_every_steps=eval_every_steps,
        nz=nz,
        start_step=start_step,
        d_every=d_every,
        use_label_smoothing=use_label_smoothing,
        use_diversity_loss=use_diversity_loss,
        n_vis_samples=n_vis_samples)

    netG.eval()
    with torch.no_grad():
        z = torch.randn(batch_size, nz, device=device)
        idx_vis = torch.randint(low=0, high=C_P.shape[0], size=(batch_size,))
        c_vis = C_P[idx_vis].to(device)
        fake = netG(z, c_vis).cpu().numpy()
        cond_strs_vis = [dataset.cond_strs[int(i)] for i in idx_vis.cpu().numpy()]
        cond_np = c_vis.cpu().numpy()
        np.save(os.path.join(checkpoint_dir, 'fake_conditions_vis.npy'), cond_np)
        np.save(os.path.join(checkpoint_dir, 'fake_indices_vis.npy'), idx_vis.cpu().numpy())
        txt_path = os.path.join(checkpoint_dir, 'fake_voxel_conditions_final.txt')
        with open(txt_path, 'w') as f_txt:
            f_txt.write('# idx  filename  condition_vector  condition_string\n')
            for idx in range(batch_size):
                filename = f'fake_voxel_{idx}.png'
                fig_path = os.path.join(checkpoint_dir, filename)
                plot_voxel_grid_3d(fake[idx, 0], title=f'Fake sample {idx}', save_path=fig_path)
                cond_str_num = ' '.join(f'{v:.6f}' for v in cond_np[idx])
                f_txt.write(f'{idx:03d}  {filename}  {cond_str_num}  {cond_strs_vis[idx]}\n')

    print(f'Saved voxel plots and fake_voxel_conditions_final.txt for {batch_size} fake samples.')

    batches_eval = 10
    all_scores = []
    all_mass = []
    all_comp = []
    all_div = []
    netG.eval()
    with torch.no_grad():
        for _ in trange(batches_eval):
            z = torch.randn(batch_size, nz, device=device)
            idx_eval = torch.randint(low=0, high=C_P.shape[0], size=(batch_size,))
            c_eval = C_P[idx_eval].to(device)
            fake = netG(z, c_eval).cpu().numpy()
            fake_bin = (fake > 0.0).astype(np.float64)
            batch_np = fake_bin[:, 0, :, :, :]
            score, mass_fracs, comp_vals = score_batch_mass_compactness(batch_np, m_min, m_max, c_min, c_max)
            all_scores.append(score)
            all_mass.append(mass_fracs)
            all_comp.append(comp_vals)
            div_val = eval_dpp_div_from_voxels(batch_np, device=device)
            all_div.append(div_val)

    all_scores = np.concatenate(all_scores)
    all_mass = np.concatenate(all_mass)
    all_comp = np.concatenate(all_comp)
    positive_rate = float((all_scores < cutoff_train).mean() * 100.0)
    mean_mass = float(all_mass.mean())
    mean_comp = float(all_comp.mean())
    mean_diversity = float(np.mean(all_div))
    print('Training quantile (score_cutoff):', score_cutoff)
    print('Training raw cutoff value (cutoff_train):', cutoff_train)
    print('Mass+compactness positive rate wrt TRAIN cutoff (%):', positive_rate)
    print('Mean mass fraction:', mean_mass)
    print('Mean compactness:', mean_comp)
    print('Mean DPP diversity:', mean_diversity)

    results_txt = os.path.join(checkpoint_dir, 'evaluation_mass_compactness.txt')
    with open(results_txt, 'w') as f:
        f.write(f'score_cutoff_quantile_train: {score_cutoff:.4f}\n')
        f.write(f'score_cutoff_value_train: {cutoff_train:.6f}\n')
        f.write(f'Mass+compactness positive rate wrt train cutoff (%): {positive_rate:.4f}\n')
        f.write(f'Mean mass fraction: {mean_mass:.6f}\n')
        f.write(f'Mean compactness: {mean_comp:.6f}\n')
        f.write(f'Mean DPP diversity: {mean_diversity:.6f}\n')
