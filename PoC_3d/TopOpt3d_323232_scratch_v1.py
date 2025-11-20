import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.data import TensorDataset

# -----------------------
# 3D Data Loader
# -----------------------
class DataNotFoundError(Exception):
    """Custom exception for missing data files."""
    pass

def load_voxel_data(filepath, positive_label=1, negative_label=0):
    # Load .npy of (voxel_arr, metric, label) tuples
    data = np.load(filepath, allow_pickle=True)
    print(f"Loaded {len(data)} samples from {filepath}")

    positives = [x[0] for x in data if x[2] == positive_label]
    negatives = [x[0] for x in data if x[2] == negative_label]
    print("Positives:", len(positives), "Negatives:", len(negatives))

    P = torch.tensor(np.stack(positives), dtype=torch.float32)
    N = torch.tensor(np.stack(negatives), dtype=torch.float32)
    # Add channel dimension if needed (for Conv3d: [B, 1, D, H, W])
    if P.ndim == 4:
        P = P.unsqueeze(1)
    if N.ndim == 4:
        N = N.unsqueeze(1)
    # Normalize to [-1, 1]
    P = P * 2 - 1
    N = N * 2 - 1
    return P, N

# -----------------------
# 3D Network Definitions
# -----------------------
class Generator3D(nn.Module):
    def __init__(self, nz, ngf, output_shape):
        super(Generator3D, self).__init__()
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
        x = self.main(x)
        return x

class Discriminator3D(nn.Module):
    def __init__(self, ndf, input_shape, num_classes=2):
        super(Discriminator3D, self).__init__()
        D, H, W = input_shape
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

            nn.Conv3d(ndf * 8, num_classes, 2, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input).view(input.size(0), -1)

# -----------------------
# Reusable DataLoader
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
# GAN Training Step
# -----------------------
def GAN_step_3D(D, G, D_opt, G_opt, P_batch, N_batch, noise_batch, batch_size, device):
    criterion = nn.CrossEntropyLoss()
    D.zero_grad()
    t = torch.full((batch_size,), 1, dtype=torch.long, device=device)  # label 1 for positives
    o = torch.full((batch_size,), 0, dtype=torch.long, device=device)  # label 0 for negatives

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

    G.zero_grad()
    fake_data = G(noise_batch)
    output = D(fake_data)
    L_G = criterion(output, t)
    L_G.backward()
    G_opt.step()

    return {"L_D_pos": L_D_pos.item(), "L_D_neg": L_D_neg.item(), "L_D_fake": L_D_fake.item(), "L_G": L_G.item()}

# -----------------------
# Training Loop
# -----------------------
def train_3D(D, G, D_opt, G_opt, P_loader, N_loader, num_steps, batch_size, noise_dim, train_step_fn, device):
    steps_range = trange(num_steps, position=0, leave=True)
    for step in steps_range:
        P_batch = P_loader.get_batch().to(device)
        N_batch = N_loader.get_batch().to(device)
        noise_batch = torch.randn(batch_size, noise_dim, device=device)
        report = train_step_fn(D, G, D_opt, G_opt, P_batch, N_batch, noise_batch, batch_size, device)
        postfix = {key: "{:.4f}".format(value) for key, value in report.items()}
        steps_range.set_postfix(postfix)
    return D, G

# -----------------------
# Evaluation/Plotting Helpers
# -----------------------
def plot_voxel_grid(voxel_grid, title="", save_path=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid > 0, edgecolor='k', linewidth=0.2)
    ax.set_title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# -----------------------
# Main Entry
# -----------------------
if __name__ == "__main__":
    # Paths and settings
    data_path = "./TO_3D_data_scratch/data/labeled_voxels_32x32x32.npy"   # adjust
    batch_size = 8
    nz = 100
    ngf = 32
    ndf = 32
    num_epochs = 1

    # Load and split data
    P, N = load_voxel_data(data_path, positive_label=1, negative_label=0)
    # Optional: shuffle, subsample for debugging
    n_samples = min(len(P), len(N))
    P, N = P[:n_samples], N[:n_samples]

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    input_shape = P.shape[2:]  # Should be (D, H, W)
    netG = Generator3D(nz, ngf, input_shape).to(device)
    netD = Discriminator3D(ndf, input_shape, num_classes=2).to(device)

    P_loader = ReusableDataLoader(TensorDataset(P), batch_size)
    N_loader = ReusableDataLoader(TensorDataset(N), batch_size)
    num_steps = num_epochs * len(P) // batch_size

    D_opt = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_opt = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training
    netD, netG = train_3D(
        netD, netG, D_opt, G_opt, P_loader, N_loader,
        num_steps, batch_size, nz, GAN_step_3D, device
    )

    # Generate and plot samples
    netG.eval()
    with torch.no_grad():
        noise = torch.randn(batch_size, nz, device=device)
        fake = netG(noise).cpu().numpy()
        for idx in range(batch_size):
            plot_voxel_grid(
                fake[idx, 0],  # Remove channel for plotting
                title=f"Fake sample {idx}",
                save_path=f"fake_voxel_{idx}.png"
            )
    print(f"Saved voxel plots for {batch_size} fake samples.")







