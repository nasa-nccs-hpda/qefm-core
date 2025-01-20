#!/usr/bin/env python
# coding: utf-8

# # Spherical Fourier Neural Operators
# 
# A simple notebook to showcase spherical Fourier Neural Operators
# 

# ## Preparation

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp
from torch.optim.lr_scheduler import OneCycleLR

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from math import ceil, sqrt

import time

cmap='twilight_shifted'


# In[2]:


enable_amp = False

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device.index)


# In[3]:


import os, sys
sys.path.insert(0, "/home/gtamkin/_foundation-models/FMSfno/modulus-makani")
sys.path.insert(1, "/home/gtamkin/_foundation-models/FMSfno/torch-harmonics-0.7.2")
#sys.path.insert(2, "/home/gtamkin/_foundation-models/FMSfno/_torch-harmonics/examples")
sys.path.append("../")
print(sys.path)


# ### Training data
# to train our geometric FNOs, we require training data. To this end let us prepare a Dataloader which computes results on the fly:

# In[4]:


import torch_harmonics
from torch_harmonics.examples.sfno.utils.pde_dataset import PdeDataset
import cartopy


# In[5]:


# dataset
from torch_harmonics.examples.sfno.utils.pde_dataset import PdeDataset

# 1 hour prediction steps
dt = 1*3600
dt_solver = 150
nsteps = dt//dt_solver
dataset = PdeDataset(dt=dt, nsteps=nsteps, dims=(256, 512), device=device, normalize=True)
# There is still an issue with parallel dataloading. Do NOT use it at the moment
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, persistent_workers=False)
solver = dataset.solver.to(device)

nlat = dataset.nlat
nlon = dataset.nlon


# In[6]:


torch.manual_seed(0)
inp, tar = dataset[0]

fig = plt.figure()
im = solver.plot_griddata(inp[2], fig, vmax=3, vmin=-3)
plt.title("input")
plt.colorbar(im)
plt.show()

fig = plt.figure()
im = solver.plot_griddata(tar[2], fig, vmax=3, vmin=-3)
plt.title("target")
plt.colorbar(im)
plt.show()


# ### Defining the geometric Fourier Neural Operator

# In[7]:


from torch_harmonics.examples.sfno import SphericalFourierNeuralOperatorNet as SFNO


# In[8]:


model = SFNO(spectral_transform='sht', operator_type='driscoll-healy', img_size=(nlat, nlon), grid="equiangular",
                 num_layers=4, scale_factor=3, embed_dim=16, big_skip=True, pos_embed="lat", use_mlp=False, normalization_layer="none").to(device)


# In[9]:


# pointwise model for sanity checking
# class MLP(nn.Module):
#     def __init__(self,
#                  input_dim = 3,
#                  output_dim = 3,
#                  num_layers = 2,
#                  hidden_dim = 32,
#                  activation_function = nn.ReLU,
#                  bias = False):
#         super().__init__()
    
#         current_dim = input_dim
#         layers = []
#         for l in range(num_layers-1):
#             fc = nn.Conv2d(current_dim, hidden_dim, 1, bias=True)
#             # initialize the weights correctly
#             scale = sqrt(2. / current_dim)
#             nn.init.normal_(fc.weight, mean=0., std=scale)
#             if fc.bias is not None:
#                 nn.init.constant_(fc.bias, 0.0)
#             layers.append(fc)
#             layers.append(activation_function())
#             current_dim = hidden_dim
#         fc = nn.Conv2d(current_dim, output_dim, 1, bias=False)
#         scale = sqrt(1. / current_dim)
#         nn.init.normal_(fc.weight, mean=0., std=scale)
#         if fc.bias is not None:
#             nn.init.constant_(fc.bias, 0.0)
#         layers.append(fc)
#         self.mlp = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.mlp(x)

# model = MLP(num_layers=10).to(device)


# ## Training the model

# In[10]:


def l2loss_sphere(solver, prd, tar, relative=False, squared=True):
    loss = solver.integrate_grid((prd - tar)**2, dimensionless=True).sum(dim=-1)
    if relative:
        loss = loss / solver.integrate_grid(tar**2, dimensionless=True).sum(dim=-1)
    
    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss

def spectral_l2loss_sphere(solver, prd, tar, relative=False, squared=True):
    # compute coefficients
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    loss = torch.sum(norm2, dim=(-1,-2))

    if relative:
        tar_coeffs = torch.view_as_real(solver.sht(tar))
        tar_coeffs = tar_coeffs[..., 0]**2 + tar_coeffs[..., 1]**2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_norm2 = torch.sum(tar_norm2, dim=(-1,-2))
        loss = loss / tar_norm2

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss


# In[11]:


# training function
def train_model(model, dataloader, optimizer, scheduler=None, nepochs=20, nfuture=0, num_examples=256, num_valid=8, loss_fn='l2'):

    train_start = time.time()

    for epoch in range(nepochs):

        # time each epoch
        epoch_start = time.time()

        dataloader.dataset.set_initial_condition('random')
        dataloader.dataset.set_num_examples(num_examples)

        optimizer.zero_grad(set_to_none=True)

        # do the training
        acc_loss = 0
        model.train()
        for inp, tar in dataloader:
            with amp.autocast(enabled=enable_amp):
                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)
                if loss_fn == 'l2':
                    loss = l2loss_sphere(solver, prd, tar)
                elif loss_fn == "spectral l2":
                    loss = spectral_l2loss_sphere(solver, prd, tar)

            acc_loss += loss.item() * inp.size(0)

            optimizer.zero_grad(set_to_none=True)
            # gscaler.scale(loss).backward()
            loss.backward()
            optimizer.step()
            # gscaler.update()

        if scheduler is not None:
            scheduler.step()

        acc_loss = acc_loss / len(dataloader.dataset)

        dataloader.dataset.set_initial_condition('random')
        dataloader.dataset.set_num_examples(num_valid)

        # perform validation
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for inp, tar in dataloader:
                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)
                loss = l2loss_sphere(solver, prd, tar, relative=True)

                valid_loss += loss.item() * inp.size(0)

        valid_loss = valid_loss / len(dataloader.dataset)

        epoch_time = time.time() - epoch_start

        print(f'--------------------------------------------------------------------------------')
        print(f'Epoch {epoch} summary:')
        print(f'time taken: {epoch_time}')
        print(f'accumulated training loss: {acc_loss}')
        print(f'relative validation loss: {valid_loss}')

    train_time = time.time() - train_start

    print(f'--------------------------------------------------------------------------------')
    print(f'done. Training took {train_time}.')
    return valid_loss


# In[12]:


# set seed
torch.manual_seed(333)
torch.cuda.manual_seed(333)

optimizer = torch.optim.Adam(model.parameters(), lr=3E-3, weight_decay=0.0)
gscaler = amp.GradScaler(enabled=enable_amp)
train_model(model, dataloader, optimizer, nepochs=10)

# multistep training
# learning_rate = 5e-4
# optimizer = torch.optim.Adam(fno_model.parameters(), lr=learning_rate)
# dataloader.dataset.nsteps = 2 * dt//dt_solver
# train_model(fno_model, dataloader, optimizer, nepochs=10, nfuture=1)
# dataloader.dataset.nsteps = 1 * dt//dt_solver


# In[13]:


dataloader.dataset.set_initial_condition('random')

torch.manual_seed(0)

with torch.inference_mode():
    inp, tar = next(iter(dataloader))
    out = model(inp).detach()

s = 0; ch = 2

fig = plt.figure()
im = solver.plot_griddata(inp[s, ch], fig, projection='3d', title='input')
plt.colorbar(im)
plt.show()

fig = plt.figure()
im = solver.plot_griddata(out[s, ch], fig, projection='3d', title='prediction')
plt.colorbar(im)
plt.show()

fig = plt.figure()
im = solver.plot_griddata(tar[s, ch], fig, projection='3d', title='target')
plt.colorbar(im)
plt.show()

fig = plt.figure()
im = solver.plot_griddata((tar-out)[s, ch], fig, projection='3d', title='error')
plt.colorbar(im)
plt.show()
