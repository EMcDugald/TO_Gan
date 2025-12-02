from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np

# def diversity_loss_3d(x):
#     # x: shape [batch_size, 1, D, H, W] (or [batch_size, D, H, W])
#     if x.ndim == 5:  # [B, 1, D, H, W]
#         x = x[:, 0]
#     x = x.view(x.size(0), -1)
#     # The rest matches your 2D loss
#     r = torch.sum(x ** 2, dim=1, keepdim=True)
#     D = r - 2 * torch.matmul(x, x.T) + r.T
#     S = torch.exp(-0.5 * D ** 2)
#     try:
#         eig_val = torch.linalg.eigvalsh(S)
#     except:
#         eig_val = torch.ones(x.size(0), device=x.device)
#     loss = -torch.mean(torch.log(torch.clamp(eig_val, min=1e-7)))
#     return loss


def GAN_step_MDD_3d(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
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
    report = {"L_D_real": L_D_real.item(), "L_D_neg": L_D_neg.item(), "L_D_fake": L_D_fake.item(), "L_G": L_G.item()}
    return report


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