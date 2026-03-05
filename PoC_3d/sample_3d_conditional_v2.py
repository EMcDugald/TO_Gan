#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -------------------------
# Models (match training)
# -------------------------


class CondGenerator3d(nn.Module):
    def __init__(self, nz, ngf, output_shape, cond_dim, cond_embed_dim=32):
        super().__init__()
        D, H, W = output_shape
        self.cond_fc = nn.Linear(cond_dim, cond_embed_dim)
        self.nz = nz
        self.init_shape = (ngf * 8, D // 16, H // 16, W // 16)
        self.fc = nn.Linear(nz + cond_embed_dim, int(np.prod(self.init_shape)))
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

    def forward(self, z, c):
        c_emb = torch.relu(self.cond_fc(c))
        zc = torch.cat([z, c_emb], dim=1)
        x = self.fc(zc)
        x = x.view(x.size(0), *self.init_shape)
        return self.main(x)


class CondDiscriminator3d(nn.Module):
    def __init__(self, ndf, input_shape, cond_dim, cond_embed_dim=32, nc=2):
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
        self.cond_fc = nn.Linear(cond_dim, cond_embed_dim)
        self.fc = nn.Linear(feat_size + cond_embed_dim, nc)

    def forward(self, x, c):
        h = self.conv(x).view(x.size(0), -1)
        c_emb = torch.relu(self.cond_fc(c))
        hc = torch.cat([h, c_emb], dim=1)
        out = self.fc(hc)
        return out


# -------------------------
# Simple voxel plotting
# -------------------------


def plot_voxel_grid_3d(voxel_grid, title="", save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Use same visualization threshold as training scripts
    ax.voxels(voxel_grid > 0.1, edgecolor="k", linewidth=0.2)
    ax.set_title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Sampling main
# -------------------------


def main():
    # 1) Paths and hyperparameters (set to match a specific run)
    # Example: label-smoothing 10k run
    data_path = "/xdisk/hdb/emcdugald/to_cond_gan/train_data/323232/10000_labeled_voxels_32x32x32_score0.7.npy"
    ckpt_path = "/xdisk/hdb/emcdugald/to_cond_gan/checkpoints_323232_10k/ls_epochs1000_bs32_nz300_ngf256_ndf64_nsamp10000_YYYYMMDD-HHMMSS/ckpt_best.pt"  # update this

    out_dir = "/xdisk/hdb/emcdugald/to_cond_gan/samples_323232"
    os.makedirs(out_dir, exist_ok=True)

    # Match the trainer hyperparameters for the chosen checkpoint
    nz = 300   # or 200, depending on the run
    ngf = 256  # or 128
    ndf = 64   # or 128

    samples_per_cond = 4       # how many samples per condition key
    max_num_conditions = 20    # how many distinct conditions to sample

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 2) Load training data to recover shape, condition vectors, and strings
    data = np.load(data_path, allow_pickle=True)
    voxels = np.stack([x[0] for x in data])
    conds = np.stack([x[1] for x in data])
    labels = np.array([x[2] for x in data], dtype=np.int64)
    cond_strs = [x[3] for x in data]  # human-readable condition strings

    # Restrict to positives if desired
    pos_mask = (labels == 1)
    conds_pos = conds[pos_mask]
    cond_strs_pos = [s for s, m in zip(cond_strs, pos_mask) if m]

    # Input / condition dimensions
    shape3d = voxels.shape[1:]   # [D, H, W]
    cond_dim = conds.shape[1]

    print("shape3d:", shape3d)
    print("cond_dim:", cond_dim)
    print("num positive conditions:", conds_pos.shape[0])

    # Choose representative positive conditions
    n_available = conds_pos.shape[0]
    n_cond = min(max_num_conditions, n_available)
    idx_unique = np.random.choice(n_available, size=n_cond, replace=False)
    cond_keys = conds_pos[idx_unique]              # [n_cond, cond_dim]
    cond_keys_str = [cond_strs_pos[i] for i in idx_unique]

    # 3) Build models and load checkpoint
    netG = CondGenerator3d(nz, ngf, shape3d, cond_dim)
    netD = CondDiscriminator3d(ndf, shape3d, cond_dim, nc=2)

    checkpoint = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(checkpoint["netG_state"])
    netD.load_state_dict(checkpoint["netD_state"])

    netG.to(device)
    netD.to(device)
    netG.eval()
    netD.eval()

    # 4) Sampling loop
    catalog_path = os.path.join(out_dir, "sampled_voxels_catalog.txt")
    with open(catalog_path, "w") as f_txt:
        f_txt.write("# cond_idx  sample_idx  filename                  cond_vector                      cond_string\n")

        for ci, (cond_vec, cond_str_human) in enumerate(zip(cond_keys, cond_keys_str)):
            c = torch.tensor(cond_vec, dtype=torch.float32, device=device).unsqueeze(0)  # [1, cond_dim]
            for si in range(samples_per_cond):
                z = torch.randn(1, nz, device=device)
                with torch.no_grad():
                    fake = netG(z, c).cpu().numpy()  # [1, 1, D, H, W]

                # Use same threshold as visualization for clarity
                vox = (fake[0, 0] > 0.1).astype(np.float32)

                filename = f"cond{ci:03d}_s{si:03d}.png"
                save_path = os.path.join(out_dir, filename)
                plot_voxel_grid_3d(vox, title=f"cond {ci}, sample {si}", save_path=save_path)

                cond_str_num = " ".join(f"{v:.6f}" for v in cond_vec)
                f_txt.write(
                    f"{ci:03d}  {si:03d}  {filename:<24}  {cond_str_num:<40}  {cond_str_human}\n"
                )

    print("Sampling complete. Catalog:", catalog_path)


if __name__ == "__main__":
    main()
