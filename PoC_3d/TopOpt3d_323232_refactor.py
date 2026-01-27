import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available")

import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset
from scipy.ndimage import label
import csv
from datetime import datetime

# -------------------------
# Data loading & Augmentation
# -------------------------

class DataNotFoundError(Exception):
    pass

def load_data_3d(filepath):
    # Load (voxel_arr, metric, label) tuples
    data = np.load(filepath, allow_pickle=True)
    P = [x[0] for x in data if x[2] == 1]
    N = [x[0] for x in data if x[2] == 0]
    P = torch.tensor(np.stack(P), dtype=torch.float32)
    N = torch.tensor(np.stack(N), dtype=torch.float32)
    # Add channel dimension
    if P.ndim == 4: P = P.unsqueeze(1)
    if N.ndim == 4: N = N.unsqueeze(1)
    # Normalize to [-1, 1]
    P = P * 2 - 1
    N = N * 2 - 1
    print("Positives:", P.shape[0], "Negatives:", N.shape[0])
    return P, N

def augment_all_3d(data):
    # Augment each 3D array with flips along axes (x, y, z)
    aug_list = []
    for img in data:
        img = img[0] if img.shape[0] == 1 else img  # Remove channel if present
        variants = [
            img, img[::-1, :, :], img[:, ::-1, :], img[:, :, ::-1],  # simple flips
            img[::-1, ::-1, :], img[::-1, :, ::-1], img[:, ::-1, ::-1],  # two axes
            img[::-1, ::-1, ::-1],  # all three axes
        ]
        aug_list.extend(variants)
    aug_tensor = torch.tensor(np.stack(aug_list), dtype=torch.float32) * 2 - 1
    aug_tensor = aug_tensor.unsqueeze(1) if aug_tensor.ndim == 4 else aug_tensor
    return aug_tensor

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
        labeled_voids, num_features = label(empty_voxels)
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
    except:
        eig_val = np.ones(x.shape[0])
    loss = -np.mean(np.log(np.maximum(eig_val, 1e-10)))
    return loss

# -------------------------
# Model definitions
# -------------------------

class Generator3d(nn.Module):
    def __init__(self, nz, ngf, output_shape):
        super().__init__()
        D, H, W = output_shape
        self.init_shape = (ngf * 8, D // 16, H // 16, W // 16)
        self.fc = nn.Linear(nz, np.prod(self.init_shape))
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
    def forward(self, input):
        x = self.fc(input)
        x = x.view(x.size(0), *self.init_shape)
        return self.main(x)

class Discriminator3d(nn.Module):
    def __init__(self, ndf, input_shape, nc=2):
        super().__init__()
        self.main = nn.Sequential(
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
            nn.Conv3d(ndf * 8, nc, 2, 1, 0, bias=False),
        )
    def forward(self, input):
        return self.main(input).view(input.size(0), -1)

# -------------------------
# ReusableDataLoader
# -------------------------

class ReusableDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))
        self.previous_indices = []
    def _shuffle_indices(self):
        self.indices = torch.randperm(len(self.dataset)).tolist()
    def get_batch(self):
        queued = self.previous_indices
        while len(queued) < self.batch_size:
            if self.shuffle:
                self._shuffle_indices()
            queued.extend(self.indices)
        self.previous_indices = queued[self.batch_size:]
        batch_indices = queued[:self.batch_size]
        return torch.stack([self.dataset[i][0] for i in batch_indices])

# -------------------------
# GAN Step (MDD for 3d)
# -------------------------

def GAN_step_MDD_3d(D, G, A, D_opt, G_opt, A_opt,
                    P_batch, N_batch, noise_batch,
                    batch_size, device,
                    validity_weight=None, diversity_weight=0):
    criterion = nn.CrossEntropyLoss()
    D.zero_grad()
    t = torch.full((batch_size,), 1, dtype=torch.long, device=device)
    o = torch.full((batch_size,), 0, dtype=torch.long, device=device)
    # Discriminator
    output = D(P_batch)
    L_D_real = criterion(output, t)
    output = D(N_batch)
    L_D_neg = criterion(output, o)
    fake_data = G(noise_batch)
    output = D(fake_data.detach())
    L_D_fake = criterion(output, o)
    L_D_tot = L_D_real + L_D_neg + L_D_fake
    L_D_tot.backward()
    D_opt.step()
    # Generator
    G.zero_grad()
    fake_data = G(noise_batch)
    output = D(fake_data)
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

def save_checkpoint(step, netG, netD, G_opt, D_opt, path):
    torch.save(
        {
            "step": step,
            "netG_state": netG.state_dict(),
            "netD_state": netD.state_dict(),
            "G_opt_state": G_opt.state_dict(),
            "D_opt_state": D_opt.state_dict(),
        },
        path,
    )

# -------------------------
# Training loop with metrics
# -------------------------

def train_3d(D, G, A, D_opt, G_opt, A_opt,
             P_loader, N_loader,
             num_steps, batch_size, noise_dim,
             train_step_fn, device,
             validity_weight, diversity_weight=0,
             checkpoint_dir=None, ckpt_interval=None):
    # Best-model tracking (by generator loss)
    best_G_loss = float("inf")
    best_ckpt_path = None

    # Metrics logging
    metrics_file = None
    metrics_writer = None
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
        metrics_file = open(metrics_path, "w", newline="")
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(["step", "L_D_real", "L_D_neg", "L_D_fake", "L_G"])

    steps_range = trange(num_steps, position=0, leave=True)
    for step in steps_range:
        P_batch = P_loader.get_batch().to(device)
        N_batch = N_loader.get_batch().to(device)
        noise_batch = torch.randn(batch_size, noise_dim, device=device)

        report = train_step_fn(
            D, G, A, D_opt, G_opt, A_opt,
            P_batch, N_batch, noise_batch,
            batch_size, device,
            validity_weight=validity_weight,
            diversity_weight=diversity_weight,
        )

        # tqdm display
        postfix = {key: "{:.4f}".format(value) for key, value in report.items()}
        steps_range.set_postfix(postfix)

        # Log metrics
        if metrics_writer is not None:
            metrics_writer.writerow([
                step + 1,
                report.get("L_D_real", float("nan")),
                report.get("L_D_neg", float("nan")),
                report.get("L_D_fake", float("nan")),
                report.get("L_G", float("nan")),
            ])

        # Best-checkpoint logic (based on generator loss)
        current_G_loss = report.get("L_G", None)
        if checkpoint_dir is not None and current_G_loss is not None:
            if current_G_loss < best_G_loss:
                best_G_loss = current_G_loss
                best_ckpt_path = os.path.join(checkpoint_dir, "ckpt_best.pt")
                save_checkpoint(step + 1, G, D, G_opt, D_opt, best_ckpt_path)

        # Periodic checkpoints every ckpt_interval steps
        if checkpoint_dir is not None and ckpt_interval is not None:
            if (step + 1) % ckpt_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step_{step+1}.pt")
                save_checkpoint(step + 1, G, D, G_opt, D_opt, ckpt_path)

    # Final checkpoint
    if checkpoint_dir is not None:
        ckpt_path = os.path.join(checkpoint_dir, "ckpt_final.pt")
        save_checkpoint(num_steps, G, D, G_opt, D_opt, ckpt_path)

    # Close metrics file if open
    if metrics_file is not None:
        metrics_file.close()

    return D, G, A

# -------------------------
# Evaluation
# -------------------------

def evaluate_n_batches_3d(netG, device, nz, batches=100, batch_size=8):
    with torch.no_grad():
        all_valid = []
        all_areas = []
        all_diversity = []
        for i in trange(batches):
            fake = netG(torch.randn(batch_size, nz, device=device)).detach().cpu()
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
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid > 0, edgecolor='k', linewidth=0.2)
    ax.set_title(title)
    plt.axis('off')
    if save_path:
        # save into given directory/file
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# -------------------------
# Main script
# -------------------------

#data_path = "/xdisk/hdb/emcdugald/to_gan/train_data/323232/10000_labeled_voxels_32x32x32.npy"
data_path = "/xdisk/hdb/emcdugald/to_gan/train_data/323232/10000_labeled_voxels_32x32x32_thr1.0e-06.npy"
void_threshold = 1e-6  # keep this in sync with the data file

batch_size = 8
nz = 200
ngf = 128
ndf = 128
num_epochs = 2000

P, N = load_data_3d(data_path)
n_samples = min(len(P), len(N))
P = P[:n_samples]
N = N[:n_samples]

validity, _ = eval_batch_validity_3d(N.detach().numpy()[:, 0])  # Remove channel for validity
assert np.all(validity == False)

validity, _ = eval_batch_validity_3d(P.detach().numpy()[:, 0])
assert np.all(validity == True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

shape3d = P.shape[2:]
netG = Generator3d(nz, ngf, shape3d).to(device)
netD = Discriminator3d(ndf, shape3d, nc=2).to(device)

P_loader = ReusableDataLoader(TensorDataset(P), batch_size)
N_loader = ReusableDataLoader(TensorDataset(N), batch_size)
num_steps = num_epochs * len(P) // batch_size

D_opt = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
G_opt = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# -------------------------
# Hyperparameter-based directory
# -------------------------

# base_ckpt_root = "/xdisk/hdb/emcdugald/to_gan/checkpoints_323232_10k"

# # include key hyperparams in directory name
# hp_name = (
#     f"epochs{num_epochs}_"
#     f"bs{batch_size}_"
#     f"nz{nz}_"
#     f"ngf{ngf}_"
#     f"ndf{ndf}_"
#     f"nsamp{n_samples}"
# )

base_ckpt_root = "/xdisk/hdb/emcdugald/to_gan/checkpoints_323232_10k"

hp_name = (
    f"epochs{num_epochs}_"
    f"bs{batch_size}_"
    f"nz{nz}_"
    f"ngf{ngf}_"
    f"ndf{ndf}_"
    f"nsamp{n_samples}_"
    f"thr{void_threshold:.1e}"
)


# optional timestamp to avoid collisions
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = os.path.join(base_ckpt_root, f"{hp_name}_{timestamp}")
os.makedirs(checkpoint_dir, exist_ok=True)

# steps & checkpoint interval
steps_per_epoch = len(P) // batch_size
ckpt_epochs = 100
ckpt_interval = ckpt_epochs * steps_per_epoch

# -------------------------
# Train
# -------------------------

netD, netG, _ = train_3d(
    netD, netG, None,
    D_opt, G_opt, None,
    P_loader, N_loader,
    num_steps, batch_size, nz,
    GAN_step_MDD_3d, device,
    validity_weight=1, diversity_weight=0,
    checkpoint_dir=checkpoint_dir,
    ckpt_interval=ckpt_interval,
)

# -------------------------
# Generate and plot fake samples (saved into checkpoint_dir)
# -------------------------

netG.eval()
with torch.no_grad():
    fake = netG(torch.randn(batch_size, nz, device=device)).cpu().numpy()
    for idx in range(batch_size):
        fig_path = os.path.join(checkpoint_dir, f"fake_voxel_{idx}.png")
        plot_voxel_grid_3d(fake[idx, 0], title=f"Fake sample {idx}", save_path=fig_path)
print(f"Saved voxel plots for {batch_size} fake samples.")

# -------------------------
# Evaluate generator and save scores to file
# -------------------------

all_valid, all_areas, all_div = evaluate_n_batches_3d(
    netG, device, nz, batches=10, batch_size=batch_size
)

mean_invalidity = (1 - np.mean(all_valid)) * 100
mean_violation = np.mean(all_areas) / np.prod(shape3d) * 100
mean_diversity = np.mean(all_div)

print("Mean invalidity rate (%):", mean_invalidity)
print("Mean violation magnitude (%):", mean_violation)
print("Mean diversity:", mean_diversity)

# Save these scores to a text file in the same hyperparameter directory
results_txt = os.path.join(checkpoint_dir, "evaluation.txt")
with open(results_txt, "w") as f:
    f.write(f"Mean invalidity rate (%): {mean_invalidity:.4f}\n")
    f.write(f"Mean violation magnitude (%): {mean_violation:.4f}\n")
    f.write(f"Mean diversity: {mean_diversity:.6f}\n")
