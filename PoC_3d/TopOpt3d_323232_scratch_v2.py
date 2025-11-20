import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.data import TensorDataset
from scipy.ndimage import label as connectedComponents3D

# -----------------------
# Data Loader & Checker
# -----------------------

class DataNotFoundError(Exception):
    pass

def load_data_3d(filepath):
    # Expect npy file of (voxel_array, metric, label)
    data = np.load(filepath, allow_pickle=True)
    positives = [x[0] for x in data if x[2] == 1]
    negatives = [x[0] for x in data if x[2] == 0]
    P = torch.tensor(np.stack(positives), dtype=torch.float32)
    N = torch.tensor(np.stack(negatives), dtype=torch.float32)
    if P.ndim == 4: P = P.unsqueeze(1)
    if N.ndim == 4: N = N.unsqueeze(1)
    P = P * 2 - 1
    N = N * 2 - 1
    print("Positives:", len(P), "Negatives:", len(N))
    return P, N

def eval_batch_validity_3d(batch):
    # batch: numpy array, shape [B, D, H, W]
    validity = []
    all_areas = []
    for i in range(len(batch)):
        structure = np.ones((3, 3, 3), dtype=int)
        # Consider background (0): invert for void
        labeled, num = connectedComponents3D((batch[i] <= 0).astype(np.uint8), structure=structure)
        borders = set()
        shape = batch[i].shape
        # Collect all labels touching the border
        borders.update(np.unique(labeled[0, :, :]))
        borders.update(np.unique(labeled[-1, :, :]))
        borders.update(np.unique(labeled[:, 0, :]))
        borders.update(np.unique(labeled[:, -1, :]))
        borders.update(np.unique(labeled[:, :, 0]))
        borders.update(np.unique(labeled[:, :, -1]))
        borders.discard(0)
        internal_vol = 0
        for lab in range(1, num + 1):
            if lab not in borders:
                internal_vol += np.sum(labeled == lab)
        valid = (num == 2)
        validity.append(valid)
        all_areas.append(internal_vol)
    return np.array(validity), np.array(all_areas)

def augment_all_3d(data):
    # Optional: Augmentation logic for 3D (rotations, flips)
    # For debugging, just return original
    return data

# -----------------------
# GAN Model
# -----------------------

class Generator3D(nn.Module):
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
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), *self.init_shape)
        return self.main(x)

class Discriminator3D(nn.Module):
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
    def forward(self, x):
        return self.main(x).view(x.size(0), -1)

# -----------------------
# Data Loader as in 2D
# -----------------------

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

# -----------------------
# GAN Training Step 3D
# -----------------------

def GAN_step_MDD_3d(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
    criterion = nn.CrossEntropyLoss()
    D.zero_grad()
    t = torch.full((batch_size,), 1, dtype=torch.long, device=device)
    o = torch.full((batch_size,), 0, dtype=torch.long, device=device)
    # Discriminator update
    output = D(P_batch)
    L_D_pos = criterion(output, t)
    output = D(N_batch)
    L_D_neg = criterion(output, o)
    fake_data = G(noise_batch)
    output = D(fake_data.detach())
    L_D_fake = criterion(output, o)
    L_D_tot = L_D_pos + L_D_neg + L_D_fake
    L_D_tot.backward()
    D_opt.step()
    # Generator update
    G.zero_grad()
    fake_data = G(noise_batch)
    output = D(fake_data)
    L_G = criterion(output, t)
    L_G.backward()
    G_opt.step()
    report = {"L_D_pos": L_D_pos.item(), "L_D_neg": L_D_neg.item(), "L_D_fake": L_D_fake.item(), "L_G": L_G.item()}
    return report

# -----------------------
# Training Loop as in 2D
# -----------------------

def train_3d(D, G, A, D_opt, G_opt, A_opt, P_loader, N_loader, num_steps, batch_size, noise_dim, train_step_fn, device, validity_weight, diversity_weight=0):
    steps_range = trange(num_steps, position=0, leave=True)
    for step in steps_range:
        P_batch = P_loader.get_batch().to(device)
        N_batch = N_loader.get_batch().to(device)
        noise_batch = torch.randn(batch_size, noise_dim, device=device)
        report = train_step_fn(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=validity_weight, diversity_weight=diversity_weight)
        postfix = {key: "{:.4f}".format(value) for key, value in report.items()}
        steps_range.set_postfix(postfix)
    return D, G, A

# -----------------------
# Plotting/Validation Example
# -----------------------

def plot_voxel_grid_3d(voxel_grid, title="", save_path=None):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid > 0, edgecolor='k', linewidth=0.2)
    ax.set_title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    data_path = "./TO_3D_data_scratch/data/labeled_voxels_32x32x32.npy"   # set path
    batch_size = 8
    nz = 100
    ngf = 32
    ndf = 32
    num_epochs = 1

    # Load data
    P, N = load_data_3d(data_path)
    n_samples = min(len(P), len(N))
    P, N = P[:n_samples], N[:n_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    shape3d = P.shape[2:]
    netG = Generator3D(nz, ngf, shape3d).to(device)
    netD = Discriminator3D(ndf, shape3d, nc=2).to(device)
    P_loader = ReusableDataLoader(TensorDataset(P), batch_size)
    N_loader = ReusableDataLoader(TensorDataset(N), batch_size)
    num_steps = num_epochs * len(P) // batch_size
    D_opt = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_opt = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Train
    netD, netG, _ = train_3d(netD, netG, None, D_opt, G_opt, None, P_loader, N_loader, num_steps, batch_size, nz, GAN_step_MDD_3d, device, 1, 0)

    # Plot samples
    netG.eval()
    with torch.no_grad():
        noise = torch.randn(batch_size, nz, device=device)
        fake = netG(noise).cpu().numpy()
        for idx in range(batch_size):
            plot_voxel_grid_3d(fake[idx, 0], title=f"Fake sample {idx}", save_path=f"fake_voxel_{idx}.png")
    print(f"Saved voxel plots for {batch_size} fake samples.")