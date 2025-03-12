#!/usr/bin/env python
# coding: utf-8

# # PrithviWxC Rollout Inference
# If you haven't already, take a look at the exmaple for the PrithviWxC core
# model, as we will pass over the points covered there.
# 
# Here we will introduce the PrithviWxC model that was trained furhter for
# autoregressive rollout, a common strategy to increase accuracy and stability of
# models when applied to forecasting-type tasks. 


import random
from pathlib import Path

#import matplotlib.pyplot as plt
import numpy as np
import torch
#from huggingface_hub import hf_hub_download, snapshot_download

# def hf_hub_download( repo_id: str, filename: str, local_dir: str ):
#     print("hf_hub_download stub: ", filename)
# def snapshot_download( repo_id: str, allow_patterns: str, local_dir: str ):
#     print("snapshot_download stub: ", allow_patterns)

# Set backend etc.
torch.jit.enable_onednn_fusion(True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

# Set seeds
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Set variables
surface_vars = [
    "EFLUX",
    "GWETROOT",
    "HFLUX",
    "LAI",
    "LWGAB",
    "LWGEM",
    "LWTUP",
    "PS",
    "QV2M",
    "SLP",
    "SWGNT",
    "SWTNT",
    "T2M",
    "TQI",
    "TQL",
    "TQV",
    "TS",
    "U10M",
    "V10M",
    "Z0M",
]
static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
levels = [
    34.0,
    39.0,
    41.0,
    43.0,
    44.0,
    45.0,
    48.0,
    51.0,
    53.0,
    56.0,
    63.0,
    68.0,
    71.0,
    72.0,
]
padding = {"level": [0, 0], "lat": [0, -1], "lon": [0, 0]}


# ### Lead time
# When performing auto-regressive rollout, the intermediate steps require the
# static data at those times and---if using `residual=climate`---the intermediate
# climatology. We provide a dataloader that extends the MERRA2 loader of the
# core model, adding in these additional terms. Further, it return target data for
# the intermediate steps if those are required for loss terms. 
# 
# The `lead_time` flag still lets the target time for the model, however now it
# only a single value and must be a positive integer multiple of the `-input_time`. 


lead_time = 12  # This variable can be change to change the task
input_time = -3  # This variable can be change to change the task


# ### Data file
# MERRA-2 data is available from 1980 to the present day,
# at 3-hour temporal resolution. The dataloader we have provided
# expects the surface data and vertical data to be saved in
# separate files, and when provided with the directories, will
# search for the relevant data that falls within the provided time range.
# 

time_range = ("2024-12-01T00:00:00", "2024-12-01T23:59:59")

surf_dir = Path("/discover/nobackup/projects/QEFM/data/FMPrithvi-WxC/merra-2")
# surf_dir = Path("../../../../checkpoints/FMPrithvi-WxC/merra-2")
# snapshot_download(
#     repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
#     allow_patterns="merra-2/MERRA2_sfc_2020010[1].nc",
#     local_dir=".",
# )

vert_dir = Path("/discover/nobackup/projects/QEFM/data/FMPrithvi-WxC/merra-2")
# snapshot_download(
#     repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
#     allow_patterns="merra-2/MERRA_pres_2020010[1].nc",
#     local_dir=".",
# )


# ### Climatology
# The PrithviWxC model was trained to calculate the output by
# producing a perturbation to the climatology at the target time.
#  This mode of operation is set via the `residual=climate` option.
#  This was chosen as climatology is typically a strong prior for
#  long-range prediction. When using the `residual=climate` option,
#  we have to provide the dataloader with the path of the
#  climatology data.

surf_clim_dir = Path("/discover/nobackup/projects/QEFM/data/FMPrithvi-WxC/climatology")
# snapshot_download(
#     repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
#     allow_patterns="climatology/climate_surface_doy00[1]*.nc",
#     local_dir=".",
# )

vert_clim_dir = Path("/discover/nobackup/projects/QEFM/data/FMPrithvi-WxC/climatology")
# snapshot_download(
#     repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
#     allow_patterns="climatology/climate_vertical_doy00[1]*.nc",
#     local_dir=".",
# )


positional_encoding = "fourier"


# ### Dataloader init
# We are now ready to instantiate the dataloader.


import os, sys
sys.path.insert(0, "/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/src/FMPrithvi-WxC/Prithvi-WxC")
#sys.path.insert(0, "/discover/nobackup/projects/QEFM/dev/models/FMPrithvi-WxC")
#sys.path.insert(0, "/panfs/ccds02/nobackup/people/gtamkin/dev/foundation-models/FMPrithvi-WxC")
sys.path.append("../")
from PrithviWxC.dataloaders.merra2_rollout import Merra2RolloutDataset

dataset = Merra2RolloutDataset(
    time_range=time_range,
    lead_time=lead_time,
    input_time=input_time,
    data_path_surface=surf_dir,
    data_path_vertical=vert_dir,
    climatology_path_surface=surf_clim_dir,
    climatology_path_vertical=vert_clim_dir,
    surface_vars=surface_vars,
    static_surface_vars=static_surface_vars,
    vertical_vars=vertical_vars,
    levels=levels,
    positional_encoding=positional_encoding,
)
print(len(dataset))
assert len(dataset) > 0, "There doesn't seem to be any valid data."


# ## Model
# ### Scalers and other hyperparameters
# Again, this setup is similar as before.


from PrithviWxC.dataloaders.merra2 import (
    input_scalers,
    output_scalers,
    static_input_scalers,
)

surf_in_scal_path = Path("/discover/nobackup/projects/QEFM/data/FMPrithvi-WxC/climatology/musigma_surface.nc")
# hf_hub_download(
#     repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
#     filename=f"climatology/{surf_in_scal_path.name}",
#     local_dir=".",
# )

vert_in_scal_path = Path("/discover/nobackup/projects/QEFM/data/FMPrithvi-WxC/climatology/musigma_vertical.nc")
# hf_hub_download(
#     repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
#     filename=f"climatology/{vert_in_scal_path.name}",
#     local_dir=".",
# )

surf_out_scal_path = Path("/discover/nobackup/projects/QEFM/data/FMPrithvi-WxC/climatology/anomaly_variance_surface.nc")
# hf_hub_download(
#     repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
#     filename=f"climatology/{surf_out_scal_path.name}",
#     local_dir=".",
# )

vert_out_scal_path = Path("/discover/nobackup/projects/QEFM/data/FMPrithvi-WxC/climatology/anomaly_variance_vertical.nc")
# hf_hub_download(
#     repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
#     filename=f"climatology/{vert_out_scal_path.name}",
#     local_dir=".",
# )

# hf_hub_download(
#     repo_id="Prithvi-WxC/prithvi.wxc.rollout.2300m.v1",
#     filename="config.yaml",
#     local_dir=".",
# )

in_mu, in_sig = input_scalers(
    surface_vars,
    vertical_vars,
    levels,
    surf_in_scal_path,
    vert_in_scal_path,
)

output_sig = output_scalers(
    surface_vars,
    vertical_vars,
    levels,
    surf_out_scal_path,
    vert_out_scal_path,
)

static_mu, static_sig = static_input_scalers(
    surf_in_scal_path,
    static_surface_vars,
)

residual = "climate"
masking_mode = "local"
decoder_shifting = True
masking_ratio = 0.99


# ### Model init
# We can now build and load the pretrained weights, note that you should use the
# rollout version of the weights.


weights_path = Path("/discover/nobackup/projects/QEFM/qefm-core/qefm/models/checkpoints/FMPrithvi-WxC/weights/prithvi.wxc.rollout.2300m.v1.pt")
# hf_hub_download(
#     repo_id="Prithvi-WxC/prithvi.wxc.rollout.2300m.v1",
#     filename=weights_path.name,
#     local_dir="./weights",
# )


import yaml

from PrithviWxC.model import PrithviWxC

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = PrithviWxC(
    in_channels=config["params"]["in_channels"],
    input_size_time=config["params"]["input_size_time"],
    in_channels_static=config["params"]["in_channels_static"],
    input_scalers_mu=in_mu,
    input_scalers_sigma=in_sig,
    input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
    static_input_scalers_mu=static_mu,
    static_input_scalers_sigma=static_sig,
    static_input_scalers_epsilon=config["params"][
        "static_input_scalers_epsilon"
    ],
    output_scalers=output_sig**0.5,
    n_lats_px=config["params"]["n_lats_px"],
    n_lons_px=config["params"]["n_lons_px"],
    patch_size_px=config["params"]["patch_size_px"],
    mask_unit_size_px=config["params"]["mask_unit_size_px"],
    mask_ratio_inputs=masking_ratio,
    embed_dim=config["params"]["embed_dim"],
    n_blocks_encoder=config["params"]["n_blocks_encoder"],
    n_blocks_decoder=config["params"]["n_blocks_decoder"],
    mlp_multiplier=config["params"]["mlp_multiplier"],
    n_heads=config["params"]["n_heads"],
    dropout=config["params"]["dropout"],
    drop_path=config["params"]["drop_path"],
    parameter_dropout=config["params"]["parameter_dropout"],
    residual=residual,
    masking_mode=masking_mode,
    decoder_shifting=decoder_shifting,
    positional_encoding=positional_encoding,
    checkpoint_encoder=[],
    checkpoint_decoder=[],
)


state_dict = torch.load(weights_path, weights_only=False)
if "model_state" in state_dict:
    state_dict = state_dict["model_state"]
model.load_state_dict(state_dict, strict=True)

if (hasattr(model, "device") and model.device != device) or not hasattr(
    model, "device"
):
    model = model.to(device)


# ## Rollout
# We are now ready to perform the rollout. Agin the data has to be run through a
# preprocessor. However this time we use a preprocessor that can handle the
# additional intermediate data. Also, rather than calling the model directly, we
# have a conveient wrapper function that performs the interation. This also
# simplifies the model loading when using a sharded cahckpoint. If you attempt to
# perform training steps upton this function, we should use an aggressive number
# of activation checkpoints as the memory consumption becomes quite high.

# In[11]:


from PrithviWxC.dataloaders.merra2_rollout import preproc
from PrithviWxC.rollout import rollout_iter

data = next(iter(dataset))
print(data.keys())
print(data['sur_tars'][12,0])
batch = preproc([data], padding)

for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.to(device)

rng_state_1 = torch.get_rng_state()
with torch.no_grad():
    model.eval()
    out,olist = rollout_iter(dataset.nsteps, model, batch)


print(len(olist))
t2m = out[0, 12].cpu().numpy()

# lat = np.linspace(-90, 90, out.shape[-2])
# lon = np.linspace(-180, 180, out.shape[-1])
# X, Y = np.meshgrid(lon, lat)

# plt.contourf(X, Y, t2m, 100)
# plt.gca().set_aspect("equal")
# plt.show()

print("Finished rollout, and this is t2m: ", str(t2m))
