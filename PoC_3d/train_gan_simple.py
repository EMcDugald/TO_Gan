import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device count (initial):", torch.cuda.device_count())
    print("GPU device name[0]:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available")

import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
import csv
from datetime import datetime


class DataNotFoundError(Exception):
    pass


class VoxelDataset(Dataset):
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise DataNotFoundError(f"File not found: {filepath}")
        X = np.load(filepath).astype(np.float32)
        if X.ndim != 4:
            raise ValueError(f"Expected [N, D, H, W], got {X.shape}")
        X = torch.tensor(X, dtype=torch.float32)
        if X.ndim == 4:
            X = X.unsqueeze(1)
        X = X * 2 - 1
        self.X = X
        print("Total samples:", X.shape[0])
        print("Voxel shape:", X.shape[2:])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


class Generator3d(nn.Module):
    def __init__(self, nz, ngf, output_shape):
        super().__init__()
        D, H, W = output_shape
        self.init_shape = (ngf * 8, D // 16, H // 16, W // 16)
        self.fc = nn.Linear(nz, int(np.prod(self.init_shape)))
        self.main = nn.Sequential(
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), *self.init_shape)
        return self.main(x)


class Discriminator3d(nn.Module):
    def __init__(self, ndf, input_shape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, ndf, 4, 2, 1, bias=False),
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
        D, H, W = input_shape
        feat_size = (ndf * 8) * (D // 16) * (H // 16) * (W // 16)
        self.fc = nn.Linear(feat_size, 1)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)


def unwrap_module(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def save_checkpoint(step, netG, netD, G_opt, D_opt, path):
    G_raw = unwrap_module(netG)
    D_raw = unwrap_module(netD)
    torch.save(
        {
            "step": step,
            "netG_state": G_raw.state_dict(),
            "netD_state": D_raw.state_dict(),
            "G_opt_state": G_opt.state_dict(),
            "D_opt_state": D_opt.state_dict(),
        },
        path,
    )


def load_checkpoint(path, netG, netD, G_opt, D_opt, device):
    ckpt = torch.load(path, map_location=device)
    netG.load_state_dict(ckpt["netG_state"])
    netD.load_state_dict(ckpt["netD_state"])
    G_opt.load_state_dict(ckpt["G_opt_state"])
    D_opt.load_state_dict(ckpt["D_opt_state"])
    for opt in (G_opt, D_opt):
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    return netG, netD, G_opt, D_opt, ckpt.get("step", 0)


def plot_voxel_grid_3d(voxel_grid, title="", save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(voxel_grid > 0.1, edgecolor="k", linewidth=0.2)
    ax.set_title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def train_dcgan_3d(
    D, G,
    D_opt, G_opt,
    loader,
    num_steps, batch_size, noise_dim,
    device,
    checkpoint_dir=None, ckpt_interval=None,
    eval_every_steps=None,
    smooth_real=0.1,
    d_every=1,
    start_step=0,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    metrics_file = None
    metrics_writer = None
    if checkpoint_dir is not None:
        metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
        mode = "a" if start_step > 0 and os.path.exists(metrics_path) else "w"
        metrics_file = open(metrics_path, mode, newline="")
        metrics_writer = csv.writer(metrics_file)
        if mode == "w":
            metrics_writer.writerow(
                ["step", "epoch", "L_D_real", "L_D_fake", "L_G", "D_grad_norm", "G_grad_norm"]
            )

    criterion = nn.BCEWithLogitsLoss()
    best_G_loss = float("inf")
    steps_per_epoch = len(loader.dataset) // batch_size
    data_iter = iter(loader)

    steps_range = trange(start_step, num_steps, position=0, leave=True)
    for step in steps_range:
        try:
            real_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            real_batch = next(data_iter)

        real_batch = real_batch.to(device)
        B = real_batch.size(0)
        d_update = ((step + 1) % d_every == 0)

        if d_update:
            D.zero_grad()
            real_logits = D(real_batch)
            real_targets = torch.full((B, 1), 1.0 - smooth_real, device=device)

            z = torch.randn(B, noise_dim, device=device)
            fake_batch = G(z).detach()
            fake_logits = D(fake_batch)
            fake_targets = torch.zeros(B, 1, device=device)

            L_D_real = criterion(real_logits, real_targets)
            L_D_fake = criterion(fake_logits, fake_targets)
            L_D = L_D_real + L_D_fake
            L_D.backward()

            D_grad_norm = 0.0
            for p in D.parameters():
                if p.grad is not None:
                    D_grad_norm += p.grad.detach().pow(2).sum().item()
            D_grad_norm = D_grad_norm ** 0.5

            D_opt.step()
        else:
            L_D_real = torch.tensor(0.0, device=device)
            L_D_fake = torch.tensor(0.0, device=device)
            D_grad_norm = 0.0

        G.zero_grad()
        z2 = torch.randn(B, noise_dim, device=device)
        gen_batch = G(z2)
        gen_logits = D(gen_batch)
        gen_targets = torch.full((B, 1), 1.0 - smooth_real, device=device)
        L_G = criterion(gen_logits, gen_targets)
        L_G.backward()

        G_grad_norm = 0.0
        for p in G.parameters():
            if p.grad is not None:
                G_grad_norm += p.grad.detach().pow(2).sum().item()
        G_grad_norm = G_grad_norm ** 0.5

        G_opt.step()

        steps_range.set_postfix({
            "L_D_real": f"{L_D_real.item():.4f}",
            "L_D_fake": f"{L_D_fake.item():.4f}",
            "L_G": f"{L_G.item():.4f}",
        })

        if metrics_writer is not None:
            epoch = step // steps_per_epoch
            metrics_writer.writerow([
                step + 1,
                epoch,
                float(L_D_real.item()),
                float(L_D_fake.item()),
                float(L_G.item()),
                float(D_grad_norm),
                float(G_grad_norm),
            ])

        if checkpoint_dir is not None:
            if L_G.item() < best_G_loss:
                best_G_loss = L_G.item()
                save_checkpoint(step + 1, G, D, G_opt, D_opt, os.path.join(checkpoint_dir, "ckpt_best.pt"))

            if ckpt_interval is not None and (step + 1) % ckpt_interval == 0:
                save_checkpoint(step + 1, G, D, G_opt, D_opt, os.path.join(checkpoint_dir, f"ckpt_step_{step+1}.pt"))

        if (checkpoint_dir is not None and
            eval_every_steps is not None and
            (step + 1) % eval_every_steps == 0):

            samples_subdir = os.path.join(checkpoint_dir, f"step_{step+1:07d}")
            os.makedirs(samples_subdir, exist_ok=True)

            G_raw = unwrap_module(G)
            G_raw.eval()
            with torch.no_grad():
                num_vis = 10
                z = torch.randn(num_vis, noise_dim, device=device)
                fake = G_raw(z).cpu().numpy()

            fake_bin = (fake > 0.0).astype(np.float32)
            for i in range(num_vis):
                fig_path = os.path.join(samples_subdir, f"fake_voxel_{i}.png")
                plot_voxel_grid_3d(
                    fake_bin[i, 0],
                    title=f"Fake sample {i} (step {step+1})",
                    save_path=fig_path,
                )

            G_raw.train()

    if checkpoint_dir is not None:
        save_checkpoint(num_steps, G, D, G_opt, D_opt, os.path.join(checkpoint_dir, "ckpt_final.pt"))

    if metrics_file is not None:
        metrics_file.close()

    return D, G


if __name__ == "__main__":
    data_path = "/xdisk/hdb/emcdugald/simple_gan/train_data/323232/5000_uncond_voxels_32x32x32.npy"

    batch_size = 64
    nz = 300
    ngf = 256
    ndf = 64
    num_epochs = 100

    lr_D = 2e-4
    lr_G = 2e-4
    smooth_real = 0.1
    d_every = 2

    dataset = VoxelDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    shape3d = dataset.X.shape[2:]
    netG = Generator3d(nz, ngf, shape3d).to(device)
    netD = Discriminator3d(ndf, shape3d).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    D_opt = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.5, 0.999))
    G_opt = optim.Adam(netG.parameters(), lr=lr_G, betas=(0.5, 0.999))

    n_samples = len(dataset)
    steps_per_epoch = n_samples // batch_size
    num_steps = num_epochs * steps_per_epoch

    base_ckpt_root = "/xdisk/hdb/emcdugald/simple_gan/checkpoints_323232"
    hp_name = (
        f"uncond_"
        f"epochs{num_epochs}_"
        f"bs{batch_size}_"
        f"nz{nz}_"
        f"ngf{ngf}_"
        f"ndf{ndf}_"
        f"nsamp{n_samples}_"
        f"lrD{lr_D}_"
        f"lrG{lr_G}_"
        f"smoothR{smooth_real}_"
        f"dEvery{d_every}"
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(base_ckpt_root, f"{hp_name}_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(checkpoint_dir, "hparams.txt"), "w") as f_hp:
        f_hp.write(f"lr_D = {lr_D}\n")
        f_hp.write(f"lr_G = {lr_G}\n")
        f_hp.write(f"smooth_real = {smooth_real}\n")
        f_hp.write(f"d_every = {d_every}\n")
        f_hp.write(f"batch_size = {batch_size}\n")
        f_hp.write(f"num_epochs = {num_epochs}\n")
        f_hp.write(f"nz = {nz}, ngf = {ngf}, ndf = {ndf}\n")
        f_hp.write(f"n_samples = {n_samples}\n")

    ckpt_epochs = 100
    ckpt_interval = ckpt_epochs * steps_per_epoch
    eval_every_epochs = 10
    eval_every_steps = eval_every_epochs * steps_per_epoch

    resume_path = None
    start_step = 0
    if resume_path is not None:
        netG, netD, G_opt, D_opt, start_step = load_checkpoint(
            resume_path, netG, netD, G_opt, D_opt, device
        )

    netD, netG = train_dcgan_3d(
        netD, netG,
        D_opt, G_opt,
        loader,
        num_steps, batch_size, nz,
        device,
        checkpoint_dir=checkpoint_dir,
        ckpt_interval=ckpt_interval,
        eval_every_steps=eval_every_steps,
        smooth_real=smooth_real,
        d_every=d_every,
        start_step=start_step,
    )

    netG.eval()
    with torch.no_grad():
        z = torch.randn(batch_size, nz, device=device)
        fake = netG(z).cpu().numpy()
        fake_bin = (fake > 0.0).astype(np.float32)
        for i in range(min(16, batch_size)):
            fname = os.path.join(checkpoint_dir, f"fake_voxel_{i}.png")
            plot_voxel_grid_3d(fake_bin[i, 0], title=f"Fake sample {i}", save_path=fname)

    print("Finished unconditional GAN training and sample dump.")