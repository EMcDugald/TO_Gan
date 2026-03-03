import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device count:", torch.cuda.device_count())
    print("GPU device name[0]:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available")

import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.data import Dataset
from scipy.ndimage import label as scipy_label
import csv
from datetime import datetime

# -------------------------
# Data loading
# -------------------------

class DataNotFoundError(Exception):
    pass


class CondVoxelDataset(Dataset):
    """
    Expects data as array of (voxel_arr, cond_vec, label) tuples.
    We will split into positives / negatives downstream.
    """
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise DataNotFoundError(f"File not found: {filepath}")
        data = np.load(filepath, allow_pickle=True)

        voxels = [x[0] for x in data]
        conds  = [x[1] for x in data]
        labels = [x[2] for x in data]

        X = torch.tensor(np.stack(voxels), dtype=torch.float32)
        C = torch.tensor(np.stack(conds), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)

        if X.ndim == 4:
            X = X.unsqueeze(1)  # [N, 1, D, H, W]

        # normalize voxel values to [-1, 1] assuming {0,1}
        X = X * 2 - 1

        self.X = X
        self.C = C
        self.y = y

        print("Total samples:", X.shape[0])
        print("Condition dim:", C.shape[1])
        print("Positives:", int((y == 1).sum().item()),
              "Negatives:", int((y == 0).sum().item()))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.C[idx], self.y[idx]



# -------------------------
# Mass / compactness scoring for batches
# -------------------------

def mass_fraction_batch(batch_np):
    """
    batch_np: [B, D, H, W] with values in {0,1} or [-1,1]
    Returns mass fraction per sample.
    """
    v = (batch_np > 0).astype(np.float64)
    return v.mean(axis=(1, 2, 3))


def compactness_batch(batch_np):
    """
    batch_np: [B, D, H, W]
    Returns compactness per sample: solid voxels / bounding-box volume.
    """
    B, D, H, W = batch_np.shape
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
    """
    batch_np: [B, D, H, W] binary (0/1) or thresholded.
    m_min, m_max, c_min, c_max: dataset-wide stats matching the data-maker script.
    Returns:
      score: [B] composite scores in [0,1]
      mass_fracs: [B]
      comp_vals: [B]
    """
    eps = 1e-8
    mass_fracs = mass_fraction_batch(batch_np)
    comp_vals = compactness_batch(batch_np)

    # normalize as in make_labeled_voxels_and_condition.py
    m_norm = (mass_fracs - m_min) / (m_max - m_min + eps)
    m_norm = np.clip(m_norm, 0.0, 1.0)

    c_raw_norm = (comp_vals - c_min) / (c_max - c_min + eps)
    c_raw_norm = np.clip(c_raw_norm, 0.0, 1.0)

    # flip: high compactness → low c_norm, diffuse → high c_norm
    c_norm = 1.0 - c_raw_norm

    score = 0.5 * m_norm + 0.5 * c_norm
    return score, mass_fracs, comp_vals



# -------------------------
# Conditional models
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
            nn.Tanh()
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
# ReusableDataLoader (x, c)
# -------------------------

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


# -------------------------
# GAN Step (conditional, MDD)
# -------------------------

def GAN_step_MDD_3d_cond(D, G, A, D_opt, G_opt, A_opt,
                         P_batch, N_batch, c_batch, noise_batch,
                         batch_size, device,
                         validity_weight=None, diversity_weight=0):
    criterion = nn.CrossEntropyLoss()
    D.zero_grad()
    t = torch.full((batch_size,), 1, dtype=torch.long, device=device)
    o = torch.full((batch_size,), 0, dtype=torch.long, device=device)

    # Discriminator on real positives
    output = D(P_batch, c_batch)
    L_D_real = criterion(output, t)

    # Discriminator on real negatives
    output = D(N_batch, c_batch)
    L_D_neg = criterion(output, o)

    # Discriminator on fake
    fake_data = G(noise_batch, c_batch)
    output = D(fake_data.detach(), c_batch)
    L_D_fake = criterion(output, o)

    L_D_tot = L_D_real + L_D_neg + L_D_fake
    L_D_tot.backward()
    D_opt.step()

    # Generator
    G.zero_grad()
    fake_data = G(noise_batch, c_batch)
    output = D(fake_data, c_batch)
    L_G = criterion(output, t)
    L_G.backward()
    G_opt.step()

    report = {
        "L_D_real": L_D_real.item(),
        "L_D_neg": L_D_neg.item(),
        "L_D_fake": L_D_fake.item(),
        "L_G": L_G.item()
    }
    return report


# -------------------------
# Checkpoint saving
# -------------------------

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


# -------------------------
# Training loop with metrics
# -------------------------

def train_3d_cond(D, G, A, D_opt, G_opt, A_opt,
                  P_loader, N_loader,
                  num_steps, batch_size, noise_dim,
                  train_step_fn, device,
                  validity_weight, diversity_weight=0,
                  checkpoint_dir=None, ckpt_interval=None):
    best_G_loss = float("inf")

    metrics_file = None
    metrics_writer = None
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
        metrics_file = open(metrics_path, "w", newline="")
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(
            ["step", "L_D_real", "L_D_neg", "L_D_fake", "L_G"]
        )

    steps_range = trange(num_steps, position=0, leave=True)
    for step in steps_range:
        P_batch, cP = P_loader.get_batch()
        N_batch, cN = N_loader.get_batch()

        # use same conditions for P and N batches (assuming same sampling)
        # if shapes differ, fall back to cP
        c_batch = cP.to(device)

        P_batch = P_batch.to(device)
        N_batch = N_batch.to(device)

        noise_batch = torch.randn(batch_size, noise_dim, device=device)

        report = train_step_fn(
            D, G, A, D_opt, G_opt, A_opt,
            P_batch, N_batch, c_batch, noise_batch,
            batch_size, device,
            validity_weight=validity_weight,
            diversity_weight=diversity_weight,
        )

        postfix = {key: "{:.4f}".format(value) for key, value in report.items()}
        steps_range.set_postfix(postfix)

        if metrics_writer is not None:
            metrics_writer.writerow([
                step + 1,
                report.get("L_D_real", float("nan")),
                report.get("L_D_neg", float("nan")),
                report.get("L_D_fake", float("nan")),
                report.get("L_G", float("nan")),
            ])

        current_G_loss = report.get("L_G", None)
        if checkpoint_dir is not None and current_G_loss is not None:
            if current_G_loss < best_G_loss:
                best_G_loss = current_G_loss
                best_ckpt_path = os.path.join(checkpoint_dir, "ckpt_best.pt")
                save_checkpoint(step + 1, G, D, G_opt, D_opt, best_ckpt_path)

        if checkpoint_dir is not None and ckpt_interval is not None:
            if (step + 1) % ckpt_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step_{step+1}.pt")
                save_checkpoint(step + 1, G, D, G_opt, D_opt, ckpt_path)

    if checkpoint_dir is not None:
        ckpt_path = os.path.join(checkpoint_dir, "ckpt_final.pt")
        save_checkpoint(num_steps, G, D, G_opt, D_opt, ckpt_path)

    if metrics_file is not None:
        metrics_file.close()

    return D, G, A



# --------------------------
# 3D Plotting
# --------------------------

def plot_voxel_grid_3d(voxel_grid, title="", save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(voxel_grid > 0, edgecolor="k", linewidth=0.2)
    ax.set_title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main script
# -------------------------


if __name__ == "__main__":
    data_root = "/xdisk/hdb/emcdugald/to_cond_gan/train_data/323232"

    # Adjust names to match your generated files
    data_path = os.path.join(
        data_root,
        "10000_labeled_voxels_32x32x32_score0.7.npy"
    )
    meta_path = os.path.join(
        data_root,
        "10000_labeled_voxels_32x32x32_score0.7_meta.npz"
    )

    # Load cutoff and normalization from generator meta
    meta = np.load(meta_path)
    cutoff_train = float(meta["cutoff_train"])
    score_cutoff = float(meta["score_cutoff"])
    m_min = float(meta["m_min"])
    m_max = float(meta["m_max"])
    c_min = float(meta["c_min"])
    c_max = float(meta["c_max"])

    batch_size = 32
    nz = 200
    ngf = 128
    ndf = 128
    num_epochs = 1000

    dataset = CondVoxelDataset(data_path)

    # split into P / N tensors and conditions using labels
    pos_mask = (dataset.y == 1)
    neg_mask = (dataset.y == 0)

    P = dataset.X[pos_mask]
    C_P = dataset.C[pos_mask]

    N = dataset.X[neg_mask]
    C_N = dataset.C[neg_mask]

    n_samples = P.shape[0] + N.shape[0]

    print("P size:", P.shape)
    print("C_P size:", C_P.shape)
    print("N size:", N.shape)
    print("C_N size:", C_N.shape)

    print("Eval stats (from generator meta):")
    print("  mass_frac min/max:", m_min, m_max)
    print("  compactness min/max:", c_min, c_max)
    print("  score_cutoff quantile (generator):", score_cutoff)
    print("  score_cutoff VALUE (cutoff_train):", cutoff_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    shape3d = P.shape[2:]
    cond_dim = C_P.shape[1]

    netG = CondGenerator3d(nz, ngf, shape3d, cond_dim)
    netD = CondDiscriminator3d(ndf, shape3d, cond_dim, nc=2)

    netG = netG.to(device)
    netD = netD.to(device)

    P_loader = ReusableDataLoader(P, C_P, batch_size)
    N_loader = ReusableDataLoader(N, C_N, batch_size)
    num_steps = num_epochs * len(P) // batch_size

    D_opt = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_opt = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    base_ckpt_root = "/xdisk/hdb/emcdugald/to_cond_gan/checkpoints_323232_10k"

    hp_name = (
        f"epochs{num_epochs}_"
        f"bs{batch_size}_"
        f"nz{nz}_"
        f"ngf{ngf}_"
        f"ndf{ndf}_"
        f"nsamp{n_samples}"
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(base_ckpt_root, f"{hp_name}_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    steps_per_epoch = len(P) // batch_size
    ckpt_epochs = 100
    ckpt_interval = ckpt_epochs * steps_per_epoch

    netD, netG, _ = train_3d_cond(
        netD, netG, None,
        D_opt, G_opt, None,
        P_loader, N_loader,
        num_steps, batch_size, nz,
        GAN_step_MDD_3d_cond, device,
        validity_weight=.9, diversity_weight=.1,
        checkpoint_dir=checkpoint_dir,
        ckpt_interval=ckpt_interval,
    )

    # -------------------------
    # Generate and plot fake samples (saved into checkpoint_dir)
    # -------------------------

    netG.eval()
    with torch.no_grad():
        z = torch.randn(batch_size, nz, device=device)
        idx_vis = torch.randint(low=0, high=C_P.shape[0], size=(batch_size,))
        c_vis = C_P[idx_vis].to(device)
        fake = netG(z, c_vis).cpu().numpy()

        cond_np = c_vis.cpu().numpy()
        np.save(os.path.join(checkpoint_dir, "fake_conditions_vis.npy"), cond_np)
        np.save(
            os.path.join(checkpoint_dir, "fake_indices_vis.npy"),
            idx_vis.cpu().numpy(),
        )

        txt_path = os.path.join(checkpoint_dir, "fake_voxel_conditions.txt")
        with open(txt_path, "w") as f_txt:
            f_txt.write("# idx  filename           condition_vector\n")
            for idx in range(batch_size):
                filename = f"fake_voxel_{idx}.png"
                fig_path = os.path.join(checkpoint_dir, filename)
                plot_voxel_grid_3d(
                    fake[idx, 0],
                    title=f"Fake sample {idx}",
                    save_path=fig_path,
                )
                cond_str = " ".join(f"{v:.6f}" for v in cond_np[idx])
                f_txt.write(f"{idx:03d}  {filename}  {cond_str}\n")

    print(f"Saved voxel plots and fake_voxel_conditions.txt for {batch_size} fake samples.")

    # -------------------------
    # Evaluate generator by mass+compactness score
    # -------------------------

    batches_eval = 10
    all_scores = []
    all_mass = []
    all_comp = []

    netG.eval()
    with torch.no_grad():
        for _ in trange(batches_eval):
            z = torch.randn(batch_size, nz, device=device)
            idx_eval = torch.randint(low=0, high=C_P.shape[0], size=(batch_size,))
            c_eval = C_P[idx_eval].to(device)

            fake = netG(z, c_eval).cpu().numpy()  # [B, 1, D, H, W]
            fake_bin = (fake > 0).astype(np.float64)
            batch_np = fake_bin[:, 0, :, :, :]

            score, mass_fracs, comp_vals = score_batch_mass_compactness(
                batch_np, m_min, m_max, c_min, c_max
            )
            all_scores.append(score)
            all_mass.append(mass_fracs)
            all_comp.append(comp_vals)

    all_scores = np.concatenate(all_scores)
    all_mass = np.concatenate(all_mass)
    all_comp = np.concatenate(all_comp)

    # Use the same cutoff the generator used to label data
    positive_rate = float((all_scores < cutoff_train).mean() * 100.0)
    mean_mass = float(all_mass.mean())
    mean_comp = float(all_comp.mean())

    print("Training quantile (score_cutoff):", score_cutoff)
    print("Training raw cutoff value (cutoff_train):", cutoff_train)
    print("Mass+compactness positive rate wrt TRAIN cutoff (%):", positive_rate)
    print("Mean mass fraction:", mean_mass)
    print("Mean compactness:", mean_comp)

    results_txt = os.path.join(checkpoint_dir, "evaluation_mass_compactness.txt")
    with open(results_txt, "w") as f:
        f.write(f"score_cutoff_quantile_train: {score_cutoff:.4f}\n")
        f.write(f"score_cutoff_value_train: {cutoff_train:.6f}\n")
        f.write(f"Mass+compactness positive rate wrt train cutoff (%): {positive_rate:.4f}\n")
        f.write(f"Mean mass fraction: {mean_mass:.6f}\n")
        f.write(f"Mean compactness: {mean_comp:.6f}\n")


