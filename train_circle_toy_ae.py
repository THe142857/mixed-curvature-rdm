import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torch.nn as nn
from plotting import plot_circular_hist
from manifold import Circle
from sdes import CircularBrownianMotion, CircularGenerative
from helpers import RunningAverageMeter
from utils.utils import stratified_uniform
import numpy as np

from data import datasets
import matplotlib.pyplot as plt
from collections import Counter

from ae import train_ae, Encoder, Decoder


# dataset = datasets.Dragonfly(split='all')

# ################### autoencoder ###################
data_dir = 'dataset'

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

train_transform = transforms.Compose([
transforms.ToTensor(),
])

test_transform = transforms.Compose([
transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
batch_size=256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# noinspection PyShadowingNames
class Swish(nn.Module):
    @staticmethod
    def forward(x):
        return torch.sigmoid(x) * x


class AModule(nn.Module):
    def __init__(self, dimh: int = 64):
        super(AModule, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(3, dimh),
            Swish(),
            nn.Linear(dimh, dimh),
            Swish(),
            nn.Linear(dimh, 1)
        )

    # noinspection PyShadowingNames
    def forward(self, x, t):
        return self.net(torch.cat([Circle.invphi(x), t], 1))


# noinspection PyShadowingNames
def aloss(a, y, t):
    a_ = a(y, t)
    norm = (a_ ** 2).sum(1, keepdim=True)
    sa = (a_.unsqueeze(-1)).squeeze(-1)
    g = torch.autograd.grad(sa[:, 0].sum(), y, create_graph=True, retain_graph=True)[0]
    ll = -0.5*norm - g

    loss = -ll
    return loss


T = 2.0
n = len(dataset)
lr = 0.01
num_iterations = 3000
a = AModule()
sde = CircularBrownianMotion()
gsde = CircularGenerative(sde.g, a, T=T)
opt = torch.optim.Adam(a.parameters(), lr=lr)
loss_meter = RunningAverageMeter()
losses = list()


# for i in range(1, num_iterations+1):
for image_batch, _ in train_loader:
    image_batch = image_batch.to(device)
    x = encoder(image_batch).detach()
    # x = (torch.rand(n) + dataset.data).unsqueeze(-1) / 360 * 2 * np.pi
    t = stratified_uniform(n, T)
    y = sde.sample(x, t).requires_grad_(True)

    loss = aloss(a, y, t).sum() / n
    loss_meter.update(loss.item())
    if i % 500 == 0 or i == 1:
        print(f'Iteration: {i} loss: {loss_meter.avg}')
        losses.append([i, loss_meter.avg])
    opt.zero_grad()
    loss.backward()
    opt.step()


x0 = torch.rand(1024, 1) * (2 * np.pi)
x = gsde.sample(x0, torch.ones_like(x0) * T, 1000)
fig, ax = plot_circular_hist(x.detach().cpu().numpy())

counted = Counter((dataset.data / 360 * 2 * np.pi).tolist())
counted = np.array([(v, c) for v, c in counted.items()])
levels = int(counted[:, 1].max(axis=0))
ylim = ax.get_ylim()[1]
base = ylim * 1.0
max_range = ylim * 0.2

all_data = np.array([])
heights = np.array([])
colors = np.array([])
for i in range(1, levels+1):
    data = counted[counted[:, 1] == i, 0]
    for j in range(1, i+1):
        all_data = np.concatenate([all_data, data])
        heights = np.concatenate([heights, np.ones_like(data) * (base + max_range * j / float(levels))])

plt.scatter(all_data, heights, s=2, c='gray', alpha=1.0)
plt.savefig('assets/dragonfly.pdf')
