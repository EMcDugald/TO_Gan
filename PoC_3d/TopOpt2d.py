import torch
print("PyTorch version:", torch.__version__)
print("CUDA version PyTorch built with:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available")
from cv2 import connectedComponentsWithStats
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
print(os.environ.get('CONDA_DEFAULT_ENV'))
from tqdm import tqdm, trange
import sys
from torch.utils.data import TensorDataset



class DataNotFoundError(Exception):
    """Custom exception for missing data files."""
    pass


def load_data(directory):
    try:
        all_images = torch.load(os.path.join(directory, "all_images.pth"))
        labels = np.load(os.path.join(directory, 'labels.npy'))

        # P = all_images[labels == 1]
        # N_procedural = all_images[labels == 0] 
        # N_real = N_procedural[35000:]  # Any after the first 35k are real
        # N_procedural = N_procedural[:35000]  # The first 35k are synthetic

        P = all_images[labels == 1][:1000]
        N_procedural = all_images[labels == 0][:500] 
        N_real = N_procedural[250:]  # Any after the first 35k are real
        N_procedural = N_procedural[:250]  # The first 35k are synthetic


        N_rejected = torch.load(os.path.join(directory, "GAN_generated_negatives.pth"))[:, 0, :, :]  # The GAN generated negatives
        N_rejected = (N_rejected > 0) * 255

        return P, N_real, N_procedural, N_rejected

    except FileNotFoundError as e:
        raise DataNotFoundError(
            f"Missing file: {e.filename}. Did you download the TO data? Please ensure all required data files are in the directory."
        ) from e


def eval_batch_validity(batch):
    
    validity = []
    all_areas = []
    for i in range(len(batch)):
        num_labels, _,b,_ = connectedComponentsWithStats((batch[i]<=128).astype(np.uint8), connectivity=8)
        if num_labels <= 2:
            tot_area=0
        else:
            areas = []
            for i in range(1, num_labels):
                area = b[i,-1]
                areas.append(area)
            areas = np.array(areas)
            tot_area = sum(areas) - max(areas)
        valid = num_labels == 2
        validity.append(valid)
        all_areas.append(tot_area)
    return np.array(validity), np.array(all_areas)

def augment_all(data):
    def augment(image):
        return torch.stack([image, image.flip(0), image.flip(1), image.flip(0).flip(1), image.transpose(0, 1), image.transpose(0, 1).flip(0), image.transpose(0, 1).flip(1), image.transpose(0, 1).flip(0).flip(1)], 0)
    return torch.cat([augment(data[i]) for i in range(data.shape[0])], 0)



def GAN_step_MDD(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
    criterion = nn.CrossEntropyLoss()
    D.zero_grad()

    t = torch.full((batch_size,), 2, dtype=torch.long, device=device)
    o = torch.full((batch_size,), 1, dtype=torch.long, device=device)
    z = torch.full((batch_size,), 0, dtype=torch.long, device=device)

    output = D(P_batch)
    L_D_real = criterion(output, o)

    output = D(N_batch)
    L_D_neg = criterion(output, t)

    fake_data = G(noise_batch)
    output = D(fake_data.detach())
    L_D_fake = criterion(output, z)

    L_D_tot = L_D_real + L_D_fake + L_D_neg
    L_D_tot.backward()
    D_opt.step()

    G.zero_grad()
    fake_data = G(noise_batch)
    output = D(fake_data)
    L_G = criterion(output, o)

    if diversity_weight > 0:
        L_div = diversity_loss(fake_data)
        L_G_tot = L_G + diversity_weight * L_div
    else:
        L_G_tot = L_G
        L_div = None

    L_G_tot.backward()
    G_opt.step()

    report = {"L_D_real": L_D_real.item(), "L_D_neg": L_D_neg.item(), "L_D_fake": L_D_fake.item(), "L_G": L_G.item()}
    if L_div is not None:
        report["L_div"] = L_div.item()
    return report


class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, nc, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input).view(input.size(0), -1)
    

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
            queued.extend(self.indices)  # Add individual elements to queued list
        
        self.previous_indices = queued[self.batch_size:]  # Store remaining indices for the next batch
        batch_indices = queued[:self.batch_size]  # Get the batch of the correct size
        return torch.stack([self.dataset[i][0] for i in batch_indices])


def train(D, G, A, D_opt, G_opt, A_opt, P_loader, N_loader, num_steps, batch_size, noise_dim, train_step_fn, device, validity_weight, diversity_weight=0):
    # Loss function
    
    steps_range = trange(num_steps, position=0, leave=True)
    for step in steps_range:
        P_batch = P_loader.get_batch().to(device)
        N_batch = N_loader.get_batch().to(device)
        noise_batch = torch.randn(batch_size, noise_dim).to(device)

        report = train_step_fn(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=validity_weight, diversity_weight=diversity_weight)
        postfix = {key: "{:.4f}".format(value) for key, value in report.items()}
        steps_range.set_postfix(postfix)
    return D, G, A

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



P, N_real, N_procedural, N_rejected = load_data("/Users/edwardmcdugald/TO_Gan/TO_Datasets/")

#select synthetic or rejection sampled negative data to add to the few real negatives

N = torch.cat((N_real, N_procedural), 0)
# N = torch.cat((N_real, N_rejected), 0)

validity, _ = eval_batch_validity(N.detach().numpy())
assert np.all(validity == False)

validity, _ = eval_batch_validity(P.detach().numpy())
assert np.all(validity == True)


#use all four rotations and two flips to augment the data 

P = augment_all(P)
N = augment_all(N)

P = P.unsqueeze(1)
N = N.unsqueeze(1)
P = P.type(torch.float32)/255*2-1
N = N.type(torch.float32)/255*2-1

P = P[torch.randperm(P.shape[0])]
N = N[torch.randperm(N.shape[0])]


train_step_fn = GAN_step_MDD

num_epochs = 1
lr = 0.0002
beta1 = 0.5
image_size = 64
batch_size = 128
nz = 100  # Size of the latent vector
ngf = 64  # Size of feature maps in the generator
ndf = 64  # Size of feature maps in the discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)
    
netG = Generator(nz, ngf).to(device)
netD = Discriminator(ndf, 3).to(device)


num_steps = num_epochs*len(P)//batch_size
D_opt = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
G_opt = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

def check_refill_pool(pool, length, batch_size):
    if length<batch_size:
        raise Exception("dataset is smaller than batch size!")
    while len(pool)<batch_size:
        remset = set(range(length))-set(pool)
        pool = pool + list(remset)
    return pool

P_loader = ReusableDataLoader(TensorDataset(P), batch_size)
N_loader = ReusableDataLoader(TensorDataset(N), batch_size)

netD, netG, _ = train(netD, netG, None, D_opt, G_opt, None, P_loader, N_loader, num_steps, batch_size, nz, train_step_fn, device, 1, 0)

all_valid, all_areas, all_diversity = evaluate_n_batches(netG, device, nz, 1000, 128)
print("Mean invalidity rate (%):", (1-np.mean(all_valid))*100)
print("Mean violation magnitude (%):", np.mean(all_areas)/64/64*100)
print("Mean diversity:", np.mean(all_diversity))

fake = netG(torch.randn(batch_size, nz, 1, 1, device=device)).detach().cpu()
fake = (fake>0)*255
samples=fake.cpu().numpy()[:,0,:,:]
validity, _ = eval_batch_validity(samples)
fig, ax = plt.subplots(4, 8, figsize=(16, 8), dpi=400)
for i in range(4):
    for j in range(8):
        idx = i*8+j
        ax[i, j].imshow(samples[idx], cmap='gray')
        ax[i, j].axis('off')
        if validity[idx] == 1:  # Valid sample
            ax[i, j].text(2, 2, '✔', color='#2BA3ED', fontsize=40, 
                        ha='left', va='top', fontweight='bold')
        else:  # Invalid sample
            ax[i, j].text(2, 2, '✘', color='red', fontsize=40, 
                        ha='left', va='top', fontweight='bold')
fig.tight_layout()
plt.savefig("2d_train_tst.png")