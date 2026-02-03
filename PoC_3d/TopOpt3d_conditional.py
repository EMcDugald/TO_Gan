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
# Validity Checking
# -------------------------

def eval_batch_validity_3d(batch):
    # batch: numpy array [B, D, H, W], values in {0,1} or [-1,1]
    validity = []
    all_areas = []
    for i in range(len(batch)):
        voxel_arr = batch[i]
        empty_voxels = (voxel_arr <= 0)
        labeled_voids, num_features = scipy_label(empty_voxels)
        border_labels = set()
        border_labels.update(np.unique(labeled_voids[0, :, :]))
        border_labels.update(np.unique(labeled_voids[-1, :, :]))
        border_labels.update(np.unique(labeled_voids[:, 0, :]))
        border_labels.update(np.unique(labeled_voids[:, -1, :]))
        border_labels.update(np.unique(labeled_voids[:, :, 0]))
        border_labels.update(np.unique(labeled_voids[:, :, -1]))
        border_labels.discard(0)
        internal_void_volume = 0
        for void_label in range(1, num_features + 1):
            if void_label not in border_labels:
                internal_void_volume += np.sum(labeled_voids == void_label)
        total_volume = np.prod(voxel_arr.shape)
        relative_void_volume = internal_void_volume / total_volume
        valid = relative_void_volume <= 1e-8
        validity.append(valid)
        all_areas.append(internal_void_volume)
    return np.array(validity), np.array(all_areas)


# -------------------------
# Diversity metric (DPP)
# -------------------------

def eval_dpp_div_3d(batch):
    # batch: [B, D, H, W] or [B, 1, D, H, W]
    if batch.ndim == 5:
        batch = batch[:, 0]  # remove channel
    batch = (batch <= 0).astype(np.float64)
    x = batch.reshape(batch.shape[0], -1) / np.sqrt(batch[0].size)
    r = np.sum(np.square(x), axis=1, keepdims=True)
    D = r - 2 * np.dot(x, x.T) + r.T
    S = np.exp(-0.5 * np.square(D))
    try:
        eig_val, _ = np.linalg.eigh(S)
    except Exception:
        eig_val = np.ones(x.shape[0])
    loss = -np.mean(np.log(np.maximum(eig_val, 1e-10)))
    return loss


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


# -------------------------
# Evaluation (conditional)
# -------------------------

def evaluate_n_batches_3d_cond(netG, device, nz, cond_dim,
                               batches=100, batch_size=8):
    with torch.no_grad():
        all_valid = []
        all_areas = []
        all_diversity = []
        for _ in trange(batches):
            z = torch.randn(batch_size, nz, device=device)
            # random conditions from N(0,1) for now; you can sample real c's instead
            c = torch.randn(batch_size, cond_dim, device=device)
            fake = netG(z, c).detach().cpu()
            fake_bin = (fake > 0).float().numpy()
            batch_np = fake_bin[:, 0, :, :, :]
            valid, area = eval_batch_validity_3d(batch_np)
            all_valid.append(valid)
            all_areas.append(area)
            all_diversity.append(eval_dpp_div_3d(batch_np))
        all_valid = np.concatenate(all_valid)
        all_areas = np.concatenate(all_areas)
        all_diversity = np.array(all_diversity)
    return all_valid, all_areas, all_diversity


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

# path to conditional training data
data_path = "/xdisk/hdb/emcdugald/to_cond_gan/train_data/323232/10000_labeled_voxels_32x32x32_medianScore.npy"

batch_size = 8
nz = 200
ngf = 128
ndf = 128
num_epochs = 2000

dataset = CondVoxelDataset(data_path)

# split into P / N tensors and conditions
pos_mask = (dataset.y == 1)
neg_mask = (dataset.y == 0)

P = dataset.X[pos_mask]
C_P = dataset.C[pos_mask]

N = dataset.X[neg_mask]
C_N = dataset.C[neg_mask]

n_samples = min(P.shape[0], N.shape[0])
P = P[:n_samples]
C_P = C_P[:n_samples]
N = N[:n_samples]
C_N = C_N[:n_samples]

validity, _ = eval_batch_validity_3d(N.detach().numpy()[:, 0])
assert np.all(validity == False)

validity, _ = eval_batch_validity_3d(P.detach().numpy()[:, 0])
assert np.all(validity == True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

shape3d = P.shape[2:]
cond_dim = C_P.shape[1]

netG = CondGenerator3d(nz, ngf, shape3d, cond_dim)
netD = CondDiscriminator3d(ndf, shape3d, cond_dim, nc=2)

if device.type == "cuda" and torch.cuda.device_count() > 1:
    print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

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
    validity_weight=1, diversity_weight=0,
    checkpoint_dir=checkpoint_dir,
    ckpt_interval=ckpt_interval,
)

# -------------------------
# Generate and plot fake samples (saved into checkpoint_dir)
# -------------------------

netG.eval()
with torch.no_grad():
    z = torch.randn(batch_size, nz, device=device)
    # sample real conditions from positive set for visualization
    idx_vis = torch.randint(low=0, high=C_P.shape[0], size=(batch_size,))
    c_vis = C_P[idx_vis].to(device)
    fake = netG(z, c_vis).cpu().numpy()
    for idx in range(batch_size):
        fig_path = os.path.join(checkpoint_dir, f"fake_voxel_{idx}.png")
        plot_voxel_grid_3d(fake[idx, 0], title=f"Fake sample {idx}", save_path=fig_path)
print(f"Saved voxel plots for {batch_size} fake samples.")

# -------------------------
# Evaluate generator and save scores to file
# -------------------------

all_valid, all_areas, all_div = evaluate_n_batches_3d_cond(
    netG, device, nz, cond_dim, batches=10, batch_size=batch_size
)

shape3d = P.shape[2:]
mean_invalidity = (1 - np.mean(all_valid)) * 100
mean_violation = np.mean(all_areas) / np.prod(shape3d) * 100
mean_diversity = np.mean(all_div)

print("Mean invalidity rate (%):", mean_invalidity)
print("Mean violation magnitude (%):", mean_violation)
print("Mean diversity:", mean_diversity)

results_txt = os.path.join(checkpoint_dir, "evaluation.txt")
with open(results_txt, "w") as f:
    f.write(f"Mean invalidity rate (%): {mean_invalidity:.4f}\n")
    f.write(f"Mean violation magnitude (%): {mean_violation:.4f}\n")
    f.write(f"Mean diversity: {mean_diversity:.6f}\n")
