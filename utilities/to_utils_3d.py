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

def eval_dpp_div(batch):
    batch = batch<=128
    x = batch.reshape(batch.shape[0], -1).astype(np.float64)
    x = x/np.sqrt(x.shape[1])
    r = np.sum(np.square(x), axis=1, keepdims=True)
    D = r - 2 * np.dot(x, x.T) + r.T
    S = np.exp(-0.5 * np.square(D))
    try:
        eig_val, _ = np.linalg.eigh(S)
    except: 
        eig_val = np.ones(x.shape[0])
    loss = -np.mean(np.log(np.maximum(eig_val, 1e-10)))
    return loss


def compute_relative_internal_void_volume(voxel_arr):
    empty_voxels = np.logical_not(voxel_arr)
    labeled_voids, num_features = label(empty_voxels)
    border_labels = set()
    # Identify void components touching the border
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
    total_volume = np.prod(voxel_arr.shape)  # total voxels in the part
    relative_void_volume = internal_void_volume / total_volume
    return relative_void_volume


def eval_batch_validity_3d(batch, threshold=0.0):
    validity = []
    all_void_volumes = []
    for voxel_arr in batch:
        relative_void_vol = compute_relative_internal_void_volume(voxel_arr)
        is_valid = relative_void_vol <= threshold
        validity.append(is_valid)
        all_void_volumes.append(relative_void_vol)
    return np.array(validity), np.array(all_void_volumes)

def evaluate_n_batches(netG, device, nz, batches=1000, batch_size=128):
    with torch.no_grad():
        all_valid = []
        all_areas = []
        all_diversity = []
        for i in trange(batches):
            fake = netG(torch.randn(batch_size, nz, 1, 1, device=device)).detach().cpu()
            fake = (fake>0)*255
            valid, area = eval_batch_validity((fake[:,0,:,:]).numpy().astype(np.uint8))
            #flatten last two dims to one
            all_valid.append(valid)
            all_areas.append(area)
            all_diversity.append(eval_dpp_div(fake.numpy()))
        all_valid = np.concatenate(all_valid)
        all_areas = np.concatenate(all_areas)
        all_diversity = np.array(all_diversity)
    return all_valid, all_areas, all_diversity

class DataNotFoundError(Exception):
    pass

def load_data(directory):
    try:
        # Load list of tuples: (voxel_arr, metric_val, label_val)
        labeled_data = np.load(os.path.join(directory, "labeled_voxels.npy"), allow_pickle=True)

        voxel_arrays = []
        labels = []

        for item in labeled_data:
            voxel_arr, metric_val, label_val = item
            voxel_arrays.append(voxel_arr)
            labels.append(label_val)

        labels = np.array(labels)
        voxel_tensors = [torch.tensor(voxel, dtype=torch.float32) for voxel in voxel_arrays]

        # Split into positives (label==1) and negatives (label==0)
        P = [t for t, l in zip(voxel_tensors, labels) if l == 1]
        N = [t for t, l in zip(voxel_tensors, labels) if l == 0]

        # For compatibility, return N as N_procedural and empty N_real
        N_real = []
        N_procedural = N

        return P, N_real, N_procedural

    except FileNotFoundError as e:
        raise DataNotFoundError(
            f"Missing file: {e.filename}. Did you download the TO data? Please ensure all required data files are in the directory."
        ) from e


def pad_voxel_tensors(voxel_tensors):
    # Handle empty list case
    if len(voxel_tensors) == 0:
        return torch.tensor([])

    max_depth = max(t.shape[0] for t in voxel_tensors)
    max_height = max(t.shape[1] for t in voxel_tensors)
    max_width = max(t.shape[2] for t in voxel_tensors)

    padded_tensors = []
    for t in voxel_tensors:
        d, h, w = t.shape
        padded = torch.zeros((max_depth, max_height, max_width), dtype=t.dtype)
        padded[:d, :h, :w] = t
        padded_tensors.append(padded)

    return torch.stack(padded_tensors)

def augment_all_3d(data):
    def augment(volume):
        vols = []
        # Original volume
        vols.append(volume)
        # Flips along X, Y, Z axes
        vols.append(volume.flip(0))
        vols.append(volume.flip(1))
        vols.append(volume.flip(2))
        # Combined flips
        vols.append(volume.flip(0).flip(1))
        vols.append(volume.flip(0).flip(2))
        vols.append(volume.flip(1).flip(2))
        vols.append(volume.flip(0).flip(1).flip(2))
        # Transpose permutations (swapping dims)
        vols.append(volume.permute(1, 0, 2))
        vols.append(volume.permute(2, 1, 0))
        vols.append(volume.permute(0, 2, 1))
        vols.append(volume.permute(1, 2, 0))
        vols.append(volume.permute(2, 0, 1))
        vols.append(volume.permute(0, 1, 2))  # redundant with original but included for completeness

        # Convert list to tensor stack along a new dim
        return torch.stack(vols, dim=0)

    augmented = [augment(data[i]) for i in range(data.shape[0])]
    # Concatenate all augmented volumes into a batch (first dim is augmentation count)
    return torch.cat(augmented, dim=0)


class Generator3D(nn.Module):
    def __init__(self, nz, ngf):
        super(Generator3D, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(nz, ngf * 8, 4, 1, 0, bias=False),
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
        input = input.view(input.size(0), input.size(1), 1, 1, 1)  # extra dim for 3D
        return self.main(input)


class Discriminator3D(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator3D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
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

            nn.Conv3d(ndf * 8, nc, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input).view(input.size(0), -1)