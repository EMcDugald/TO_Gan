from cv2 import connectedComponentsWithStats
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from scipy.ndimage import label

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
        # Label sample as valid if no significant internal voids (same as in your data-prep)
        valid = relative_void_volume <= 0.0
        validity.append(valid)
        all_areas.append(internal_void_volume)
    return np.array(validity), np.array(all_areas)


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
    aug_tensor = torch.tensor(np.stack(variants), dtype=torch.float32) * 2 - 1
    aug_tensor = aug_tensor.unsqueeze(1) if aug_tensor.ndim == 4 else aug_tensor
    return aug_tensor


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
    
def diversity_loss_3d(x):
    # x: shape [batch_size, 1, D, H, W] (or [batch_size, D, H, W])
    if x.ndim == 5:  # [B, 1, D, H, W]
        x = x[:, 0]
    x = x.view(x.size(0), -1)
    # The rest matches your 2D loss
    r = torch.sum(x ** 2, dim=1, keepdim=True)
    D = r - 2 * torch.matmul(x, x.T) + r.T
    S = torch.exp(-0.5 * D ** 2)
    try:
        eig_val = torch.linalg.eigvalsh(S)
    except:
        eig_val = torch.ones(x.size(0), device=x.device)
    loss = -torch.mean(torch.log(torch.clamp(eig_val, min=1e-7)))
    return loss
    

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